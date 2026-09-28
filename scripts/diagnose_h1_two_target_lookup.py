#!/usr/bin/env python3
"""Tiny identity-to-expression-effect control for the H1 ST overfit test.

This deliberately bypasses ST and the gene-conditioned decoder. A trainable
two-row embedding predicts a bulk log(CP10K) effect on exactly the same fixed
1024-gene panel. It establishes whether the identities and expression targets
are themselves learnable before testing the full model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state.tx.vcc import load_gene_name_list
from state.tx.vcc.data_module import PanelFreePerturbationDataModule


def bulk_log_cp10k(data: ad.AnnData, indices: np.ndarray, panel: np.ndarray) -> torch.Tensor:
    counts = data.X[np.sort(indices)]
    full_total = float(np.asarray(counts.sum()).reshape(()))
    selected = np.asarray(counts[:, panel].sum(axis=0)).ravel().astype(np.float32)
    return torch.from_numpy(np.log1p(10_000.0 * selected / max(full_total, 1.0)))


def retrieval(prediction: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    similarity = torch.nn.functional.normalize(prediction, dim=-1) @ torch.nn.functional.normalize(
        reference, dim=-1
    ).T
    diagonal = similarity.diag()
    rank = (similarity > diagonal[:, None]).sum(dim=1).float()
    rank += 0.5 * (similarity == diagonal[:, None]).sum(dim=1).float() - 0.5
    return {
        "mean_rank_score": float((1.0 - rank / (len(prediction) - 1)).mean()),
        "top1_accuracy": float((similarity.argmax(dim=1) == torch.arange(len(prediction))).float().mean()),
        "same_target_cosine": float(diagonal.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/vcc_h1_two_target_st_300")
    parser.add_argument("--data", type=Path, default=ROOT / "data/prepared/focused/arc_h1_train.focused.xstate.h5ad")
    parser.add_argument("--targets", type=Path, default=ROOT / "assets/vcc_h1_overfit_2.txt")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/vcc_h1_two_target_lookup/metrics.json")
    args = parser.parse_args()
    if args.steps <= 0 or args.lr <= 0:
        raise ValueError("steps and learning rate must be positive")
    targets = load_gene_name_list(args.targets)
    if len(targets) != 2:
        raise ValueError("This diagnostic requires exactly two distinct targets")

    config = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    module = PanelFreePerturbationDataModule(
        **config["data"]["kwargs"],
        batch_size=config["training"]["batch_size"],
        cell_sentence_len=config["model"]["kwargs"]["cell_set_len"],
    )
    module.setup(stage="fit")
    panel = np.asarray(module._shared_panel_indices["arc_h1_train"], dtype=np.int64)

    data = ad.read_h5ad(args.data, backed="r")
    try:
        labels = data.obs["target_gene"].astype(str).to_numpy()
        rng = np.random.default_rng(args.seed)
        controls = rng.permutation(np.flatnonzero(labels == "non-targeting"))
        if len(controls) < 800:
            raise ValueError("At least 800 H1 controls are required")
        control_train = bulk_log_cp10k(data, controls[:400], panel)
        control_test = bulk_log_cp10k(data, controls[400:800], panel)
        train_effects, test_effects, counts = [], [], {}
        for target in targets:
            indices = rng.permutation(np.flatnonzero(labels == target))
            if len(indices) < 4:
                raise ValueError(f"Too few cells for {target}")
            half = len(indices) // 2
            counts[target] = len(indices)
            train_effects.append(bulk_log_cp10k(data, indices[:half], panel) - control_train)
            test_effects.append(bulk_log_cp10k(data, indices[half:], panel) - control_test)
    finally:
        data.file.close()
    train_target = torch.stack(train_effects)
    test_target = torch.stack(test_effects)

    torch.manual_seed(args.seed)
    lookup = torch.nn.Embedding(2, len(panel))
    torch.nn.init.zeros_(lookup.weight)
    optimizer = torch.optim.Adam(lookup.parameters(), lr=args.lr)
    ids = torch.arange(2)
    history = []
    for step in range(args.steps + 1):
        prediction = lookup(ids)
        if step in {0, 10, 50, 100, args.steps}:
            history.append(
                {
                    "step": step,
                    "train_mse": float(torch.nn.functional.mse_loss(prediction, train_target)),
                    "test_mse": float(torch.nn.functional.mse_loss(prediction, test_target)),
                    "test_retrieval": retrieval(prediction, test_target),
                }
            )
        if step == args.steps:
            break
        loss = torch.nn.functional.mse_loss(prediction, train_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    result = {
        "targets": targets,
        "cells_per_target": counts,
        "genes": len(panel),
        "train_test_split": "half of each target; disjoint 400-cell control pools",
        "learning_rate": args.lr,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
