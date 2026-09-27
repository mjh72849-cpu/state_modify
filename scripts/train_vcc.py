#!/usr/bin/env python3
"""Launch reproducible staged VCC fine-tuning configurations.

This wrapper only assembles and executes the repository's normal ``state tx
train`` command. Use ``--dry-run`` to inspect the exact Hydra overrides.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "external/state-env/bin/state"
TAHOE = (
    ROOT
    / "external/ST-SE-Tahoe/zeroshot/state_generalization_zeroshot_X_state/checkpoints/best.ckpt"
)
LOCO = ROOT / "configs/vcc/vcc_h1_loco.toml"
ALL_TRAIN = ROOT / "configs/vcc/vcc_all_train.toml"

DEFAULTS = {
    "h1-loco-warmup": {
        "toml": LOCO,
        "name": "vcc_h1_loco_warmup",
        "steps": 2_000,
        "val_freq": 500,
        "freeze": True,
        "init": TAHOE,
    },
    "h1-loco-joint": {
        "toml": LOCO,
        "name": "vcc_h1_loco_joint",
        "steps": 12_000,
        "val_freq": 1_000,
        "freeze": False,
        "init": ROOT / "runs/vcc_h1_loco_warmup/checkpoints/best.ckpt",
    },
    "full": {
        "toml": ALL_TRAIN,
        "name": "vcc_full",
        "steps": 12_000,
        "val_freq": 2_000,
        "freeze": False,
        "init": ROOT / "runs/vcc_h1_loco_joint/checkpoints/best.ckpt",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(DEFAULTS))
    parser.add_argument("--gpu", default="0", help="One CUDA device index, e.g. 0")
    parser.add_argument("--name")
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of cell Sets per optimizer microbatch")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if "," in args.gpu:
        raise ValueError(
            "This balanced Set sampler is configured for one GPU; multi-GPU would duplicate batches."
        )

    profile = DEFAULTS[args.mode]
    init_from = (args.init_from or profile["init"]).resolve()
    toml = profile["toml"].resolve()
    if not STATE.is_file():
        raise FileNotFoundError(STATE)
    if not toml.is_file():
        raise FileNotFoundError(toml)
    if not init_from.is_file():
        raise FileNotFoundError(
            f"Initialization checkpoint does not exist: {init_from}. "
            "Finish the preceding stage or pass --init-from explicitly."
        )
    if args.batch_size <= 0 or args.gradient_accumulation <= 0 or args.num_workers < 0:
        raise ValueError("batch size/accumulation must be positive and workers cannot be negative")

    name = args.name or profile["name"]
    max_steps = args.max_steps or profile["steps"]
    run_dir = ROOT / "runs" / name
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Choose --name or explicitly pass --overwrite."
        )
    command = [
        str(STATE), "tx", "train",
        "data=vcc_panel_free",
        "model=state_vcc",
        f"data.kwargs.toml_config_path={toml}",
        f"data.kwargs.num_workers={args.num_workers}",
        f"model.kwargs.init_from={init_from}",
        f"model.kwargs.freeze_pretrained_backbone={str(profile['freeze']).lower()}",
        f"training.batch_size={args.batch_size}",
        f"training.gradient_accumulation_steps={args.gradient_accumulation}",
        f"training.max_steps={max_steps}",
        f"training.val_freq={profile['val_freq']}",
        f"output_dir={ROOT / 'runs'}",
        f"name={name}",
        f"use_wandb={str(args.wandb).lower()}",
        f"overwrite={str(args.overwrite).lower()}",
    ]
    print("CUDA_VISIBLE_DEVICES=" + args.gpu, shlex.join(command), flush=True)
    if args.dry_run:
        return
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment.setdefault("PYTHONPATH", str(ROOT / "src"))
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
