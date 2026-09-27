import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vcc_pipeline", ROOT / "scripts/vcc_pipeline.py")
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pipeline)


def arguments(**overrides):
    values = {
        "gpu": "2",
        "max_steps": None,
        "overwrite": False,
        "resume": False,
        "dry_run": False,
        "run_suffix": "",
        "output": None,
        "vcc_output": None,
        "no_pack": False,
        "smoke_targets": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_training_command_uses_previous_final_checkpoint_for_short_smoke(tmp_path, monkeypatch):
    run_dirs = {
        "h1-loco-warmup": tmp_path / "warmup",
        "h1-loco-joint": tmp_path / "joint",
        "full": tmp_path / "full",
    }
    monkeypatch.setattr(pipeline, "TRAIN_RUNS", run_dirs)
    final = tmp_path / "warmup_smoke/checkpoints/final.ckpt"
    final.parent.mkdir(parents=True)
    final.touch()
    config = pipeline.load_config(ROOT / "configs/vcc/vcc_pipeline.toml")

    command = pipeline.training_command(
        "h1-loco-joint", config, arguments(run_suffix="_smoke", max_steps=2)
    )
    assert command[0].endswith("scripts/train_vcc.py")
    assert command[command.index("--name") + 1] == "joint_smoke"
    assert Path(command[command.index("--init-from") + 1]) == final
    assert command[command.index("--max-steps") + 1] == "2"


def test_smoke_inference_and_packaging_are_isolated(tmp_path, monkeypatch):
    run_dirs = {
        "h1-loco-warmup": tmp_path / "warmup",
        "h1-loco-joint": tmp_path / "joint",
        "full": tmp_path / "full",
    }
    monkeypatch.setattr(pipeline, "TRAIN_RUNS", run_dirs)
    config = pipeline.load_config(ROOT / "configs/vcc/vcc_pipeline.toml")
    command, output = pipeline.inference_command(
        config, arguments(run_suffix="_smoke", smoke_targets=1)
    )
    pack_command, vcc_output = pipeline.packaging_command(
        config, arguments(run_suffix="_smoke", smoke_targets=1), output
    )

    assert output == tmp_path / "full_smoke/prediction.h5ad"
    assert vcc_output == tmp_path / "full_smoke/prediction.vcc"
    assert command[command.index("--limit-targets") + 1] == "1"
    assert f"PYTHONPATH={ROOT / 'src'}" in command
    assert "--debug-subset" in pack_command
    assert "--output" in pack_command


def test_submission_id_parser_accepts_current_cli_shapes():
    assert pipeline.extract_submission_id('{"entry_id":"abc"}') == "abc"
    assert pipeline.extract_submission_id('{"submission":{"id":123}}') == "123"
    assert pipeline.extract_submission_id('{"status":"uploading"}\n{"entry_id":"xyz"}') == "xyz"
    assert pipeline.extract_submission_id("not json") is None


def test_warmup_submit_config_infers_from_warmup_best_checkpoint(tmp_path, monkeypatch):
    run_dirs = {
        "h1-loco-warmup": tmp_path / "warmup",
        "h1-loco-joint": tmp_path / "joint",
        "full": tmp_path / "full",
    }
    monkeypatch.setattr(pipeline, "TRAIN_RUNS", run_dirs)
    config = pipeline.load_config(ROOT / "configs/vcc/vcc_warmup_submit.toml")
    command, output = pipeline.inference_command(
        config, arguments(run_suffix="_probe")
    )
    expected_run = tmp_path / "warmup_probe"
    assert Path(command[command.index("--run-dir") + 1]) == expected_run
    assert command[command.index("--checkpoint") + 1] == "best.ckpt"
    assert output == expected_run / "prediction.h5ad"
