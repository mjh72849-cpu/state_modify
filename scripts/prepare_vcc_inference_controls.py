#!/usr/bin/env python3
"""Prepare official VCC controls, deterministic 4-donor pools, and SE states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from state.tx.vcc import build_vcc_control_pool, control_log_cp10k_baseline


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-dir", type=Path, default=ROOT / "data/vcc_2026_controls")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/prepared/vcc_inference")
    parser.add_argument("--seed", type=int, default=20_260_910)
    parser.add_argument("--run-se", action="store_true")
    parser.add_argument("--se-batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    genes = pd.read_csv(args.controls_dir / "gene_names.csv", dtype=str)["gene_name"].to_numpy()
    if len(genes) != 18_533 or len(set(genes)) != len(genes):
        raise ValueError("Expected 18,533 unique official VCC genes")

    inputs: dict[str, Path] = {}
    report = {"seed": args.seed, "cells": 400, "pool_k": 4, "contexts": {}}
    for offset, context in enumerate("ABC"):
        source = args.controls_dir / f"context_{context}.h5ad"
        data = ad.read_h5ad(source)
        if not np.array_equal(data.var_names.astype(str), genes):
            raise ValueError(f"Context {context}: gene order differs from gene_names.csv")
        pool = build_vcc_control_pool(data.X, seed=args.seed + offset)
        baseline = control_log_cp10k_baseline(data.X)
        np.savez_compressed(
            args.output_dir / f"context_{context}.pool.npz",
            donor_indices=pool.donor_indices,
            library_sizes=pool.library_sizes,
            log_cp10k_baseline=baseline,
            selected_obs_names=np.asarray(data.obs_names)[pool.donor_indices],
            seed=np.asarray(args.seed + offset),
        )
        report["contexts"][context] = {
            "controls": int(data.n_obs),
            "genes": int(data.n_vars),
            "selected_donors": int(pool.donor_indices.size),
            "output_cells": int(len(pool.library_sizes)),
            "min_library_size": int(pool.library_sizes.min()),
            "median_library_size": float(np.median(pool.library_sizes)),
            "max_library_size": int(pool.library_sizes.max()),
        }
        inputs[context] = source

    if args.run_se:
        from state.emb.inference import Inference

        protein = torch.load(
            ROOT / "external/SE-600M/protein_embeddings.pt", map_location="cpu", weights_only=False
        )
        inferer = Inference(cfg=None, protein_embeds=protein)
        inferer.load_model(ROOT / "external/SE-600M/se600m_epoch16.ckpt")
        for context, source in inputs.items():
            output = args.output_dir / f"context_{context}.xstate.h5ad"
            if output.exists() and not args.overwrite:
                cached = ad.read_h5ad(output, backed="r")
                try:
                    if cached.shape != (18_400, 18_533) or "X_state" not in cached.obsm:
                        raise ValueError(f"Invalid cached file {output}; use --overwrite")
                finally:
                    cached.file.close()
                continue
            if output.exists():
                output.unlink()
            inferer.encode_adata(
                input_adata_path=str(source),
                output_adata_path=str(output),
                emb_key="X_state",
                dataset_name=f"vcc_context_{context}",
                batch_size=args.se_batch_size,
            )
            cached = ad.read_h5ad(output, backed="r")
            try:
                if cached.obsm["X_state"].shape != (18_400, 2058):
                    raise ValueError(f"Unexpected X_state shape in {output}")
            finally:
                cached.file.close()

    report["has_xstate"] = bool(args.run_se)
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
