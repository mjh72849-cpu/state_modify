#!/usr/bin/env python3
"""Make a reproducible, cell-disjoint H1 train/validation diagnostic.

The two small H5AD files are required by cell-load's file-based split API.
They retain the original raw-count X and precomputed X_state without
renormalization or SE recomputation. The JSON records source row indices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def split_indices(
    labels: np.ndarray,
    batches: np.ndarray,
    targets: set[str],
    *,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratify by perturbation and batch; singletons stay in train."""
    if not 0 < fraction < 0.5:
        raise ValueError("fraction must be between 0 and 0.5")
    labels = np.asarray(labels, dtype=str)
    batches = np.asarray(batches, dtype=str)
    if labels.shape != batches.shape:
        raise ValueError("labels and batches must have identical shape")
    selected = np.isin(labels, ["non-targeting", *sorted(targets)])
    rng = np.random.default_rng(seed)
    train, held_out = [], []
    for label in sorted(set(labels[selected])):
        for batch in sorted(set(batches[selected & (labels == label)])):
            group = np.flatnonzero(selected & (labels == label) & (batches == batch))
            group = rng.permutation(group)
            n_val = min(len(group) - 1, max(1, round(len(group) * fraction))) if len(group) > 1 else 0
            held_out.extend(group[:n_val].tolist())
            train.extend(group[n_val:].tolist())
    train = np.asarray(sorted(train), dtype=np.int64)
    held_out = np.asarray(sorted(held_out), dtype=np.int64)
    if np.intersect1d(train, held_out).size:
        raise AssertionError("train/validation cell overlap")
    for label in targets | {"non-targeting"}:
        if not np.any(labels[train] == label) or not np.any(labels[held_out] == label):
            raise ValueError(f"Target {label!r} is absent from train or validation")
    for batch in set(batches[held_out]):
        if not np.any((batches[train] == batch) & (labels[train] == "non-targeting")):
            raise ValueError(f"No training control for held-out batch {batch!r}")
    return train, held_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "data/prepared/focused/arc_h1_train.focused.xstate.h5ad",
    )
    parser.add_argument("--targets", type=Path, default=ROOT / "assets/vcc_h1_shared_21.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/prepared/h1_holdout_21")
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    train_path = args.output_dir / "h1_21_train.xstate.h5ad"
    val_path = args.output_dir / "h1_21_val.xstate.h5ad"
    manifest_path = args.output_dir / "split.json"
    if any(path.exists() for path in (train_path, val_path, manifest_path)):
        raise FileExistsError(f"Split already exists in {args.output_dir}; refusing to overwrite it")

    targets = {line.strip().upper() for line in args.targets.read_text().splitlines() if line.strip()}
    source = ad.read_h5ad(args.source, backed="r")
    try:
        labels = source.obs["target_gene"].astype(str).to_numpy()
        batches = source.obs["batch"].astype(str).to_numpy()
        train, val = split_indices(labels, batches, targets, fraction=args.fraction, seed=args.seed)
        assert "X_state" in source.obsm
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path, indices in ((train_path, train), (val_path, val)):
            subset = source[indices].to_memory()
            subset.write_h5ad(path, compression="lzf")
            print(f"{path}: {subset.shape}, X_state={subset.obsm['X_state'].shape}", flush=True)
        counts = {
            label: {"train": int(np.sum(labels[train] == label)), "val": int(np.sum(labels[val] == label))}
            for label in sorted(targets | {"non-targeting"})
        }
        manifest_path.write_text(json.dumps({
            "source": str(args.source.resolve()), "seed": args.seed,
            "fraction": args.fraction, "targets": sorted(targets),
            "train_indices": train.tolist(), "val_indices": val.tolist(),
            "counts": counts,
        }, indent=2) + "\n")
        print(f"train={len(train)} val={len(val)} targets={len(targets)}", flush=True)
    finally:
        source.file.close()


if __name__ == "__main__":
    main()
