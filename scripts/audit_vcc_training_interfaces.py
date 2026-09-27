#!/usr/bin/env python3
"""Fail-fast audit for focused VCC raw/X_state files and gene interfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from state.tx.vcc.data import canonical_gene_name, load_gene_name_list


ROOT = Path(__file__).resolve().parents[1]


def _nonzero_embedding_names(path: Path) -> set[str]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return {
        canonical_gene_name(name)
        for name, value in raw.items()
        if torch.as_tensor(value).abs().sum().item() > 0
    }


def audit_file(
    path: Path,
    semantic: set[str],
    fallback: set[str],
    trainable_perturbations: set[str],
) -> dict:
    data = ad.read_h5ad(path, backed="r")
    try:
        required_obs = {"target_gene", "batch", "cell_type", "dataset"}
        missing_obs = sorted(required_obs - set(data.obs.columns))
        if missing_obs:
            raise ValueError(f"{path}: missing obs columns {missing_obs}")
        if "gene_name" not in data.var:
            raise ValueError(f"{path}: missing var['gene_name']")
        genes = [canonical_gene_name(value) for value in data.var["gene_name"].astype(str)]
        if len(genes) != len(set(genes)):
            duplicates = pd.Series(genes)[pd.Series(genes).duplicated()].unique().tolist()
            raise ValueError(f"{path}: duplicate canonical gene names {duplicates[:10]}")
        targets = {
            canonical_gene_name(value)
            for value in data.obs["target_gene"].astype(str).unique()
            if canonical_gene_name(value) != "NON-TARGETING"
        }
        unsupported_targets = sorted(targets - semantic - fallback)
        if unsupported_targets:
            raise ValueError(
                f"{path}: {len(unsupported_targets)} perturbations lack semantic/fallback embeddings: "
                f"{unsupported_targets[:10]}"
            )
        missing_trainable_ids = sorted(targets - trainable_perturbations)
        if missing_trainable_ids:
            raise ValueError(
                f"{path}: {len(missing_trainable_ids)} perturbations lack trainable IDs: "
                f"{missing_trainable_ids[:10]}"
            )
        xstate_shape = None
        if "X_state" in data.obsm:
            xstate_shape = list(data.obsm["X_state"].shape)
            if xstate_shape != [data.n_obs, 2058]:
                raise ValueError(f"{path}: expected X_state [{data.n_obs},2058], got {xstate_shape}")
            for start in range(0, data.n_obs, 4096):
                if not np.isfinite(np.asarray(data.obsm["X_state"][start : start + 4096])).all():
                    raise ValueError(f"{path}: X_state contains NaN/Inf near row {start}")
        eligible = set(genes) & (semantic | fallback)
        return {
            "file": str(path.resolve()),
            "cells": int(data.n_obs),
            "genes": int(data.n_vars),
            "unique_gene_names": True,
            "perturbations": len(targets),
            "unsupported_perturbations": 0,
            "perturbations_without_trainable_id": 0,
            "decoder_eligible_genes": len(eligible),
            "decoder_intentionally_unavailable_oov_genes": len(set(genes) - semantic - fallback),
            "xstate_shape": xstate_shape,
        }
    finally:
        data.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--protein-embeddings", type=Path,
        default=ROOT / "external/SE-600M/protein_embeddings.pt",
    )
    parser.add_argument(
        "--fallback", type=Path,
        default=ROOT / "assets/vcc_2026_se_fallback_genes.txt",
    )
    parser.add_argument(
        "--vcc-genes", type=Path,
        default=ROOT / "data/vcc_2026_controls/gene_names.csv",
    )
    parser.add_argument(
        "--trainable-perturbations", type=Path,
        default=ROOT / "assets/vcc_trainable_perturbations.txt",
    )
    parser.add_argument(
        "--vcc-targets", type=Path,
        default=ROOT / "data/vcc_2026_controls/pert_counts.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/vcc_training_interface_audit.json")
    args = parser.parse_args()

    semantic = _nonzero_embedding_names(args.protein_embeddings)
    fallback = set(load_gene_name_list(args.fallback))
    trainable_perturbations = set(load_gene_name_list(args.trainable_perturbations))
    vcc_genes = {
        canonical_gene_name(value)
        for value in pd.read_csv(args.vcc_genes)["gene_name"].astype(str)
    }
    missing_vcc = sorted(vcc_genes - semantic - fallback)
    if missing_vcc:
        raise ValueError(f"{len(missing_vcc)} VCC output genes remain unqueryable: {missing_vcc[:10]}")
    if fallback - vcc_genes:
        raise ValueError("Fallback registry contains genes outside the official VCC output panel")
    vcc_targets = {
        canonical_gene_name(value)
        for value in pd.read_csv(args.vcc_targets)["target_gene"].astype(str)
    }
    missing_vcc_target_ids = sorted(vcc_targets - trainable_perturbations)
    if missing_vcc_target_ids:
        raise ValueError(
            f"{len(missing_vcc_target_ids)} VCC targets lack trainable perturbation IDs: "
            f"{missing_vcc_target_ids[:10]}"
        )

    report = {
        "vcc_output_genes": len(vcc_genes),
        "vcc_semantic_genes": len(vcc_genes & semantic),
        "vcc_fallback_genes": len(vcc_genes & fallback),
        "vcc_unqueryable_genes": 0,
        "trainable_perturbations": len(trainable_perturbations),
        "vcc_targets_with_trainable_id": len(vcc_targets),
        "datasets": [
            audit_file(path, semantic, fallback, trainable_perturbations)
            for path in args.paths
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
