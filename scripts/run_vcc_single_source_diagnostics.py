#!/usr/bin/env python3
"""Queue comparable, in-sample PDS diagnostics for K562, HCT116 and HEK293T.

All three runs use the current masked Set=128 architecture and the same 32
training-present VCC targets. This is a capacity test, not held-out validation.
Run on one GPU; the existing mixed run can occupy a second GPU independently.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "external/state-env/bin/python"
CELL_EVAL = ROOT.parent / "cell-eval2/src"
TARGETS = ROOT / "assets/vcc_single_source_shared_32.txt"
INIT = ROOT / "external/ST-SE-Tahoe/zeroshot/state_generalization_zeroshot_X_state/checkpoints/best.ckpt"
SOURCES = {
    "k562": (
        ROOT / "configs/vcc/vcc_k562_single_diagnostic.toml",
        ROOT / "data/prepared/focused/replogle_k562_gwps.focused.xstate.h5ad",
    ),
    "hct116": (
        ROOT / "configs/vcc/vcc_hct116_single_diagnostic.toml",
        ROOT / "data/prepared/focused/xatlas_hct116.focused.xstate.h5ad",
    ),
    "hek293t": (
        ROOT / "configs/vcc/vcc_hek293t_single_diagnostic.toml",
        ROOT / "data/prepared/hek293t/xatlas_hek293t.focused.xstate.h5ad",
    ),
}


def run_stage(command: list[str], log: Path, env: dict[str, str], *, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    print(f"Finished: {log}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="1", help="One physical GPU index")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sources", nargs="+", choices=tuple(SOURCES), default=list(SOURCES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if "," in args.gpu or args.steps <= 0 or args.num_workers <= 0:
        parser.error("Use exactly one GPU, positive steps and positive workers")
    for path in (PYTHON, CELL_EVAL, TARGETS, INIT):
        if not path.exists():
            raise FileNotFoundError(path)
    for manifest, data in SOURCES.values():
        if not manifest.is_file() or not data.is_file():
            raise FileNotFoundError(f"Missing manifest or data: {manifest}, {data}")

    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    env["PYTHONPATH"] = str(ROOT / "src")
    inference_env = dict(env, CUDA_VISIBLE_DEVICES=args.gpu)
    score_env = dict(env, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(CELL_EVAL))
    summary = []
    for source in args.sources:
        manifest, data = SOURCES[source]
        worker_suffix = f"_w{args.num_workers}" if args.num_workers != 2 else ""
        name = f"vcc_single_{source}_shared32_masked128_{args.steps}{worker_suffix}"
        run_dir = ROOT / "runs" / name
        checkpoint = run_dir / "checkpoints/final.ckpt"
        prediction = run_dir / "shared32_train_prediction.h5ad"
        real = run_dir / "shared32_train_real.h5ad"
        metrics = run_dir / "shared32_train_metrics"
        score_file = metrics / "agg_results.csv"
        if not checkpoint.is_file():
            if run_dir.exists():
                raise FileExistsError(f"Incomplete run directory; inspect before retrying: {run_dir}")
            run_stage(
                [str(PYTHON), str(ROOT / "scripts/train_vcc.py"), "full", "--name", name,
                 "--toml", str(manifest), "--data-config", "vcc_single_source_masked_128",
                 "--model-config", "state_vcc_all_cross_batch_128", "--gpu", args.gpu,
                 "--init-from", str(INIT), "--max-steps", str(args.steps),
                 "--val-freq", "250", "--batch-size", "4",
                 "--gradient-accumulation", "1", "--num-workers", str(args.num_workers)],
                ROOT / "logs" / f"{name}.log", env, dry_run=args.dry_run,
            )
        if not prediction.is_file() or not real.is_file():
            if prediction.exists() or real.exists():
                raise FileExistsError(f"Partial inference output; inspect before retrying: {run_dir}")
            run_stage(
                [str(PYTHON), str(ROOT / "scripts/evaluate_h1_loco.py"),
                 "--run-dir", str(run_dir), "--checkpoint", "final.ckpt",
                 "--data", str(data), "--targets-file", str(TARGETS),
                 "--no-shared-nonzero-only", "--output", str(prediction),
                 "--real-output", str(real), "--device", "cuda:0",
                 "--max-cells-per-pert", "400"],
                ROOT / "logs" / f"{name}_inference.log", inference_env, dry_run=args.dry_run,
            )
        if not score_file.is_file():
            run_stage(
                [str(PYTHON), "-c", "from cell_eval2.cli import main; main()", "run",
                 "-ap", str(prediction), "-ar", str(real), "--preset", "vcc2026",
                 "--profile", "anndata", "--pert-col", "target_gene", "-o", str(metrics)],
                ROOT / "logs" / f"{name}_metrics.log", score_env, dry_run=args.dry_run,
            )
        if not args.dry_run:
            with score_file.open(newline="") as handle:
                mean = next(row for row in csv.DictReader(handle) if row["statistic"] == "mean")
            result = {"source": source, "steps": args.steps, "run": name,
                      "pds_cosine": mean["pds_cosine"], "expr_mae": mean["expr_mae"],
                      "delta_pearson": mean["delta_pearson"]}
            summary.append(result)
            print("RESULT " + " ".join(f"{key}={value}" for key, value in result.items()), flush=True)
    if summary:
        summary_file = ROOT / "logs" / (
            f"vcc_single_source_shared32_masked128_{args.steps}"
            f"{'_w' + str(args.num_workers) if args.num_workers != 2 else ''}_summary.csv"
        )
        with summary_file.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(f"Summary: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
