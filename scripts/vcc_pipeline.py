#!/usr/bin/env python3
"""Run and track the staged STATE -> VCC training and submission workflow.

Runtime state is written below ``runs/pipelines/<pipeline-id>`` and complete
stdout/stderr logs are written below ``logs/``.  Network submission is never
implicit: it requires both ``--submit`` and ``--confirm-submit``.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import tomllib
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/vcc/vcc_pipeline.toml"
PIPELINE_ROOT = ROOT / "runs/pipelines"
LOG_ROOT = ROOT / "logs"
TRAIN_RUNS = {
    "h1-loco-warmup": ROOT / "runs/vcc_h1_loco_warmup",
    "h1-loco-joint": ROOT / "runs/vcc_h1_loco_joint",
    "full": ROOT / "runs/vcc_full",
}


def stage_run_dir(stage: str, args: argparse.Namespace) -> Path:
    suffix = args.run_suffix or ""
    if suffix and any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in suffix):
        raise ValueError("run suffix may contain only letters, digits, '-' and '_'")
    base = TRAIN_RUNS[stage]
    return base.with_name(base.name + suffix)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    required = {"pipeline", "training", "inference", "submission"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Pipeline config is missing sections: {sorted(missing)}")
    stages = config["training"].get("stages", [])
    unknown = set(stages) - set(TRAIN_RUNS)
    if unknown:
        raise ValueError(f"Unknown training stages: {sorted(unknown)}")
    return config


class Tracker:
    def __init__(self, pipeline_id: str, config_path: Path):
        if not pipeline_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in pipeline_id):
            raise ValueError("pipeline ID may contain only letters, digits, '-' and '_'")
        self.pipeline_id = pipeline_id
        self.directory = PIPELINE_ROOT / pipeline_id
        self.state_path = self.directory / "state.json"
        self.events_path = self.directory / "events.jsonl"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {
                "pipeline_id": pipeline_id,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "config": str(config_path.resolve()),
                "git_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                    capture_output=True, check=True,
                ).stdout.strip(),
                "status": "created",
                "steps": {},
            }
            self.save()

    def save(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_json(self.state_path, self.state)

    def event(self, kind: str, **fields: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {"time": utc_now(), "event": kind, **fields}
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def set_step(self, name: str, **fields: Any) -> None:
        step = self.state["steps"].setdefault(name, {})
        step.update(fields)
        self.save()

    def completed(self, name: str, artifact: Path | None = None) -> bool:
        step = self.state["steps"].get(name, {})
        return step.get("status") == "succeeded" and (artifact is None or artifact.exists())

    def run(
        self,
        name: str,
        command: list[str],
        category: str,
        *,
        dry_run: bool = False,
        capture: bool = False,
    ) -> str:
        log_dir = LOG_ROOT / category
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{self.pipeline_id}__{name}.log"
        printable = shlex.join(command)
        self.set_step(
            name, status="planned" if dry_run else "running", command=command,
            log=str(log_path.relative_to(ROOT)), started_at=utc_now(),
        )
        self.event("step_planned" if dry_run else "step_started", step=name, command=command)
        print(f"[{name}] {printable}", flush=True)
        if dry_run:
            return ""

        started = time.monotonic()
        tail: deque[str] = deque(maxlen=500)
        with log_path.open("a", buffering=1) as log:
            log.write(f"\n[{utc_now()}] START {printable}\n")
            process = subprocess.Popen(
                command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                if capture:
                    tail.append(line)
            return_code = process.wait()
            elapsed = time.monotonic() - started
            log.write(f"[{utc_now()}] END exit={return_code} seconds={elapsed:.3f}\n")
        status = "succeeded" if return_code == 0 else "failed"
        self.set_step(
            name, status=status, ended_at=utc_now(), exit_code=return_code,
            elapsed_seconds=round(elapsed, 3),
        )
        self.event("step_finished", step=name, status=status, exit_code=return_code)
        if return_code:
            self.state["status"] = "failed"
            self.save()
            raise subprocess.CalledProcessError(return_code, command)
        return "".join(tail)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def record_artifact(
    tracker: Tracker, step: str, path: Path, *, dry_run: bool, checksum: bool = False
) -> None:
    fields: dict[str, Any] = {"artifact": str(path)}
    if not dry_run:
        if not path.is_file():
            raise FileNotFoundError(f"Step completed without expected artifact: {path}")
        fields["artifact_bytes"] = path.stat().st_size
        if checksum:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            fields["sha256"] = digest.hexdigest()
    tracker.set_step(step, **fields)


def training_command(stage: str, config: dict[str, Any], args: argparse.Namespace) -> list[str]:
    settings = config["pipeline"]
    command = [
        str(ROOT / "scripts/train_vcc.py"), stage,
        "--gpu", str(args.gpu or settings["gpu"]),
        "--num-workers", str(settings["num_workers"]),
        "--batch-size", str(settings["batch_size"]),
        "--gradient-accumulation", str(settings["gradient_accumulation"]),
    ]
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    run_dir = stage_run_dir(stage, args)
    if args.run_suffix:
        command.extend(["--name", run_dir.name])
    previous = {"h1-loco-joint": "h1-loco-warmup", "full": "h1-loco-joint"}.get(stage)
    if previous is not None:
        previous_dir = stage_run_dir(previous, args)
        candidates = [
            previous_dir / "checkpoints/best.ckpt",
            previous_dir / "checkpoints/final.ckpt",
        ]
        initialization = next((candidate for candidate in candidates if candidate.is_file()), None)
        if initialization is None and args.dry_run:
            initialization = candidates[0]
        if initialization is not None:
            command.extend(["--init-from", str(initialization)])
    if args.overwrite:
        command.append("--overwrite")
    elif args.resume and run_dir.exists():
        command.append("--resume")
    return command


def inference_command(config: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], Path]:
    settings = config["pipeline"]
    inference = config["inference"]
    python = resolve_repo_path(settings["python"])
    run_dir = stage_run_dir("full", args) if args.run_suffix else resolve_repo_path(inference["run_dir"])
    default_output = run_dir / "prediction.h5ad" if args.run_suffix else resolve_repo_path(inference["output"])
    output = resolve_repo_path(args.output) if args.output else default_output
    command = [
        str(python), str(ROOT / "scripts/run_vcc_inference.py"),
        "--run-dir", str(run_dir),
        "--checkpoint", str(inference["checkpoint"]),
        "--output", str(output),
        "--device", "cuda:0",
        "--cell-chunk-size", str(inference["cell_chunk_size"]),
        "--gene-chunk-size", str(inference["gene_chunk_size"]),
        "--scratch-dir", str(resolve_repo_path(inference["scratch_dir"])),
    ]
    if args.smoke_targets is not None:
        command.extend(["--limit-targets", str(args.smoke_targets)])
    if args.overwrite:
        command.append("--overwrite")
    environment_gpu = str(args.gpu or settings["gpu"])
    command = [
        "env", f"CUDA_VISIBLE_DEVICES={environment_gpu}",
        f"PYTHONPATH={ROOT / 'src'}", *command,
    ]
    return command, output


def packaging_command(
    config: dict[str, Any], args: argparse.Namespace, prediction: Path
) -> tuple[list[str], Path]:
    settings = config["pipeline"]
    inference = config["inference"]
    python = resolve_repo_path(settings["python"])
    run_dir = stage_run_dir("full", args) if args.run_suffix else resolve_repo_path(inference["run_dir"])
    default_output = run_dir / "prediction.vcc" if args.run_suffix else resolve_repo_path(inference["vcc_output"])
    output = resolve_repo_path(args.vcc_output) if args.vcc_output else default_output
    command = [
        str(python), str(ROOT / "scripts/package_vcc.py"), str(prediction),
        "--output", str(output),
        "--scratch-dir", str(resolve_repo_path(inference["scratch_dir"])),
    ]
    if args.smoke_targets is not None:
        command.append("--debug-subset")
    if args.overwrite:
        command.append("--overwrite")
    return command, output


def extract_submission_id(output: str) -> str | None:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("entry_id", "submission_id", "id"):
        value = payload.get(key)
        if value is not None:
            return str(value)
    for value in payload.values():
        if isinstance(value, dict):
            for key in ("entry_id", "submission_id", "id"):
                if value.get(key) is not None:
                    return str(value[key])
    return None


def submit(tracker: Tracker, vcc_file: Path, config: dict[str, Any], args: argparse.Namespace) -> None:
    if not args.confirm_submit:
        raise ValueError("Submission requires the explicit --confirm-submit flag")
    if not vcc_file.is_file():
        raise FileNotFoundError(vcc_file)
    executable = shutil.which("vcc")
    if executable is None:
        raise FileNotFoundError("The official vcc CLI is not installed")
    settings = config["submission"]
    tracker.run("submission_identity", [executable, "whoami", "--json"], "submission")
    command = [
        executable, "submit", str(vcc_file),
        "--model-name", str(args.model_name or settings["model_name"]),
        "--description", str(args.description or settings.get("description", "")),
        "--json",
    ]
    if args.wait_submission or settings.get("wait", False):
        command.extend(["--wait", "--poll-interval", str(settings.get("poll_interval", 30.0))])
    response = tracker.run("submission", command, "submission", capture=True)
    submission_id = extract_submission_id(response)
    tracker.state["submission"] = {
        "id": submission_id,
        "file": str(vcc_file),
        "submitted_at": utc_now(),
    }
    tracker.save()
    if submission_id:
        print(f"Submission ID: {submission_id}")
    else:
        print("Submission succeeded, but its ID could not be parsed; see the submission log.")


def run_pipeline(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    config = load_config(config_path)
    tracker = Tracker(args.pipeline_id, config_path)
    tracker.state.update(status="running", pid=os.getpid(), started_at=utc_now())
    tracker.save()
    try:
        stages = list(config["training"]["stages"])
        if args.skip_training:
            stages = []
        if args.from_stage:
            stages = stages[stages.index(args.from_stage):]
        if args.through_stage:
            stages = stages[: stages.index(args.through_stage) + 1]
        for stage in stages:
            artifact = stage_run_dir(stage, args) / "checkpoints/final.ckpt"
            if args.resume and tracker.completed(f"train_{stage}", artifact):
                print(f"[train_{stage}] already completed; skipping", flush=True)
                continue
            tracker.run(
                f"train_{stage}", training_command(stage, config, args), "training",
                dry_run=args.dry_run,
            )
            record_artifact(tracker, f"train_{stage}", artifact, dry_run=args.dry_run)

        vcc_output: Path | None = None
        if config["inference"].get("enabled", True) and not args.skip_inference:
            command, prediction = inference_command(config, args)
            if args.resume and tracker.completed("inference", prediction):
                print("[inference] already completed; skipping", flush=True)
            else:
                tracker.run(
                    "inference", command, "inference",
                    dry_run=args.dry_run,
                )
                record_artifact(tracker, "inference", prediction, dry_run=args.dry_run)
            if bool(config["inference"].get("pack", True)) and not args.no_pack:
                pack_command, vcc_output = packaging_command(config, args, prediction)
                if args.resume and tracker.completed("packaging", vcc_output):
                    print("[packaging] already completed; skipping", flush=True)
                else:
                    tracker.run("packaging", pack_command, "packaging", dry_run=args.dry_run)
                    record_artifact(
                        tracker, "packaging", vcc_output,
                        dry_run=args.dry_run, checksum=True,
                    )

        if args.submit:
            if args.smoke_targets is not None:
                raise ValueError("Debug subset .vcc files cannot be submitted")
            if args.dry_run:
                print("[submission] dry-run: submission suppressed")
            else:
                if vcc_output is None:
                    if args.vcc_output:
                        vcc_output = resolve_repo_path(args.vcc_output)
                    elif args.run_suffix:
                        vcc_output = stage_run_dir("full", args) / "prediction.vcc"
                    else:
                        vcc_output = resolve_repo_path(config["inference"]["vcc_output"])
                submit(tracker, vcc_output, config, args)
        tracker.state["status"] = "planned" if args.dry_run else "succeeded"
        tracker.state["ended_at"] = utc_now()
        tracker.save()
    except BaseException:
        tracker.state["status"] = "failed"
        tracker.state["ended_at"] = utc_now()
        tracker.save()
        raise


def status_command(args: argparse.Namespace) -> None:
    state_path = PIPELINE_ROOT / args.pipeline_id / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text())
    print(f"pipeline={state['pipeline_id']} status={state['status']} updated={state['updated_at']}")
    for name, step in state.get("steps", {}).items():
        elapsed = step.get("elapsed_seconds", "-")
        print(f"{name:32} {step.get('status', 'unknown'):10} seconds={elapsed} log={step.get('log', '-')}")
    submission = state.get("submission")
    if submission:
        print(f"submission_id={submission.get('id')} file={submission.get('file')}")


def track_command(args: argparse.Namespace) -> None:
    config_path = resolve_repo_path(args.config)
    tracker = Tracker(args.pipeline_id, config_path)
    submission_id = args.entry_id or tracker.state.get("submission", {}).get("id")
    if not submission_id:
        raise ValueError("No submission ID is recorded; pass --entry-id")
    executable = shutil.which("vcc")
    if executable is None:
        raise FileNotFoundError("The official vcc CLI is not installed")
    command = [executable, "status", str(submission_id), "--json"]
    if args.wait:
        command.extend(["--wait", "--poll-interval", str(args.poll_interval)])
    response = tracker.run("submission_status", command, "submission", capture=True)
    try:
        tracker.state["submission"]["last_status"] = json.loads(response)
    except json.JSONDecodeError:
        tracker.state["submission"]["last_status_raw"] = response[-4000:]
    tracker.state["submission"]["checked_at"] = utc_now()
    tracker.save()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run staged training, inference, packaging, and optional submission")
    run.add_argument("--pipeline-id", required=True)
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--gpu")
    run.add_argument("--run-suffix", default="", help="Append a safe suffix to all three training run names")
    run.add_argument("--max-steps", type=int, help="Override every selected stage (intended for smoke tests)")
    run.add_argument("--from-stage", choices=tuple(TRAIN_RUNS))
    run.add_argument("--through-stage", choices=tuple(TRAIN_RUNS))
    run.add_argument("--skip-training", action="store_true")
    run.add_argument("--skip-inference", action="store_true")
    run.add_argument("--smoke-targets", type=int)
    run.add_argument("--output")
    run.add_argument("--vcc-output")
    run.add_argument("--no-pack", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--submit", action="store_true")
    run.add_argument("--confirm-submit", action="store_true")
    run.add_argument("--model-name")
    run.add_argument("--description")
    run.add_argument("--wait-submission", action="store_true")
    run.set_defaults(func=run_pipeline)

    status = commands.add_parser("status", help="Show locally recorded pipeline state")
    status.add_argument("--pipeline-id", required=True)
    status.set_defaults(func=status_command)

    track = commands.add_parser("track", help="Query a VCC submission and record its current status")
    track.add_argument("--pipeline-id", required=True)
    track.add_argument("--config", default=str(DEFAULT_CONFIG))
    track.add_argument("--entry-id")
    track.add_argument("--wait", action="store_true")
    track.add_argument("--poll-interval", type=float, default=30.0)
    track.set_defaults(func=track_command)
    return root


def main() -> None:
    args = parser().parse_args()
    if getattr(args, "resume", False) and getattr(args, "overwrite", False):
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if getattr(args, "smoke_targets", None) is not None and args.smoke_targets <= 0:
        raise ValueError("--smoke-targets must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
