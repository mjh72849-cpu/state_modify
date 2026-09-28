#!/usr/bin/env python3
"""Generate held-out H1 predictions for PDS-focused local evaluation.

The script mirrors training: controls are matched within batch, the ST input is
the frozen offline ``X_state`` embedding, decoder targets are log1p(CP10K), and
the output is converted back to integer count space.  It deliberately caps
each perturbation at 400 cells to bound disk and evaluation cost.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.run_vcc_inference import _canonical_nonzero_features, _load_model
from state.tx.vcc import (
    build_gene_query_features,
    canonical_gene_name,
    control_log_cp10k_read_depth,
    control_log_cp10k_rows,
    load_gene_name_list,
    predict_vcc_log_expression,
)


def training_targets_for_run(run_dir: Path, held_out_data: Path) -> set[str]:
    """Collect perturbations observed in training sources other than held-out H1."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    toml_path = Path(cfg["data"]["kwargs"]["toml_config_path"])
    if not toml_path.is_absolute():
        toml_path = ROOT / toml_path
    with toml_path.open("rb") as handle:
        split = tomllib.load(handle)
    held_out_resolved = held_out_data.resolve()
    observed: set[str] = set()
    for name, raw_path in split.get("datasets", {}).items():
        if split.get("training", {}).get(name) != "train":
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.resolve() == held_out_resolved:
            continue
        source = ad.read_h5ad(path, backed="r")
        try:
            observed.update(
                canonical_gene_name(value)
                for value in source.obs["target_gene"].astype(str).unique()
                if canonical_gene_name(value) != "NON-TARGETING"
            )
        finally:
            source.file.close()
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.ckpt")
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/prepared/focused/arc_h1_train.focused.xstate.h5ad",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--real-output",
        type=Path,
        help="Write the matching real H1 subset used by cell-eval2",
    )
    parser.add_argument(
        "--write-real", action=argparse.BooleanOptionalAction, default=True,
        help="Write the real H1 subset (disable to reuse an existing identical reference)",
    )
    parser.add_argument("--protein-embeddings", type=Path, default=ROOT / "external/SE-600M/protein_embeddings.pt")
    parser.add_argument("--fallback", type=Path, default=ROOT / "assets/vcc_2026_se_fallback_genes.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cells-per-pert", type=int, default=400)
    parser.add_argument("--cell-chunk-size", type=int, default=256)
    parser.add_argument("--gene-chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20_260_927)
    parser.add_argument(
        "--targets-file", type=Path,
        help="Optional target list for a focused source-dataset diagnostic",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--shared-nonzero-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate only H1 targets observed in non-H1 training data whose learned "
            "identity embedding is non-zero (default: true)"
        ),
    )
    args = parser.parse_args()

    if args.max_cells_per_pert <= 0:
        raise ValueError("--max-cells-per-pert must be positive")
    real_output = args.real_output or args.output.with_name(
        args.output.stem + "_real" + args.output.suffix
    )
    outputs = (args.output, real_output) if args.write_real else (args.output,)
    for output in outputs:
        if output.exists():
            if not args.overwrite:
                raise FileExistsError(output)
            output.unlink()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = args.run_dir / "checkpoints" / checkpoint
    device = torch.device(args.device)
    model = _load_model(args.run_dir, checkpoint, device)

    raw_proteins = torch.load(args.protein_embeddings, map_location="cpu", weights_only=False)
    proteins = _canonical_nonzero_features(raw_proteins)
    del raw_proteins
    fallback = load_gene_name_list(args.fallback)

    data = ad.read_h5ad(args.data, backed="r")
    try:
        genes = data.var_names.astype(str).tolist()
        supported = []
        unsupported = []
        fallback_set = set(fallback)
        for index, gene in enumerate(genes):
            canonical = canonical_gene_name(gene)
            vector = proteins.get(canonical)
            if (vector is not None and vector.abs().sum().item() > 0) or canonical in fallback_set:
                supported.append(index)
            else:
                unsupported.append(index)
        supported_names = [genes[index] for index in supported]
        query_embeddings, query_fallback_ids = build_gene_query_features(
            supported_names, proteins, fallback
        )
        query_embeddings = query_embeddings.to(device)
        query_fallback_ids = query_fallback_ids.to(device)
        print(
            f"H1 genes={len(genes)} decoder_supported={len(supported)} "
            f"control-baseline-only={len(unsupported)}",
            flush=True,
        )

        obs = data.obs.copy()
        labels = obs["target_gene"].astype(str).to_numpy()
        batches = obs["batch"].astype(str).to_numpy()
        control_mask = labels == "non-targeting"
        all_control_indices = np.flatnonzero(control_mask)
        if not len(all_control_indices):
            raise ValueError("H1 data has no non-targeting controls")
        registry = model.trainable_perturbation_to_id
        targets = sorted(set(labels) - {"non-targeting"})
        if args.targets_file is not None:
            requested = set(load_gene_name_list(args.targets_file))
            targets = [target for target in targets if canonical_gene_name(target) in requested]
            if len(targets) < 2:
                raise ValueError("Fewer than two requested targets occur in the evaluation dataset")
        missing_targets = [name for name in targets if canonical_gene_name(name) not in registry]
        if missing_targets:
            raise ValueError(f"Held-out H1 targets absent from perturbation registry: {missing_targets[:10]}")
        if args.shared_nonzero_only:
            observed_training = training_targets_for_run(args.run_dir, args.data)
            # The active VCC profile intentionally keeps the pre-existing
            # semantic protein encoder plus its trainable residual table.  A
            # paper-like identity table is also supported for older runs, so
            # select whichever trainable per-target table the checkpoint has.
            identity_table = getattr(model, "perturbation_embedding", None)
            if identity_table is None:
                identity_table = getattr(model, "perturbation_residual", None)
            if identity_table is None:
                raise ValueError(
                    "shared-nonzero filtering requires a trainable perturbation table"
                )
            identity = identity_table.weight.detach()
            targets = [
                target
                for target in targets
                if canonical_gene_name(target) in observed_training
                and identity[registry[canonical_gene_name(target)]].norm().item() > 1e-8
            ]
            if not targets:
                raise ValueError("No shared H1 targets have non-zero perturbation rows")
            print(
                f"H1 shared non-zero targets={len(targets)}: {','.join(targets)}",
                flush=True,
            )

        rng = np.random.default_rng(args.seed)
        control_by_batch = {
            batch: np.flatnonzero(control_mask & (batches == batch)) for batch in np.unique(batches)
        }
        blocks: list[sparse.csr_matrix] = []
        output_obs: list[pd.DataFrame] = []

        # Include measured controls because cell-eval2 requires equal perturbation sets;
        # vcc2026 still uses the real control pool as its reference.
        kept_controls = rng.choice(
            all_control_indices,
            size=min(args.max_cells_per_pert, len(all_control_indices)),
            replace=False,
        )
        blocks.append(sparse.csr_matrix(data.X[kept_controls], dtype=np.float32))
        output_obs.append(obs.iloc[kept_controls].copy())

        supported_array = np.asarray(supported, dtype=np.int64)
        unsupported_array = np.asarray(unsupported, dtype=np.int64)
        supported_device = torch.as_tensor(supported_array, dtype=torch.long, device=device)
        unsupported_device = torch.as_tensor(unsupported_array, dtype=torch.long, device=device)
        for target_number, target in enumerate(targets, start=1):
            candidates = np.flatnonzero(labels == target)
            selected = rng.choice(
                candidates, size=min(args.max_cells_per_pert, len(candidates)), replace=False
            )
            # Inference uses per-cell batch-matched controls. Training may
            # assemble a Set across batches without changing this mapping.
            for batch in sorted(set(batches[selected])):
                pert_indices = selected[batches[selected] == batch]
                controls = control_by_batch.get(batch)
                if controls is None or not len(controls):
                    controls = all_control_indices
                donors = rng.choice(controls, size=len(pert_indices), replace=True)
                control_state = torch.from_numpy(
                    np.asarray(data.obsm["X_state"][donors], dtype=np.float32)
                ).to(device)
                canonical_target = canonical_gene_name(target)
                semantic = proteins.get(canonical_target)
                if semantic is None:
                    semantic = torch.zeros(query_embeddings.shape[-1], dtype=torch.float32)

                donor_baseline = control_log_cp10k_rows(data.X[donors])
                donor_read_depth = control_log_cp10k_read_depth(data.X[donors])
                decoded = predict_vcc_log_expression(
                    model,
                    control_state,
                    semantic.to(device),
                    query_embeddings,
                    perturbation_ids=registry[canonical_target],
                    query_gene_fallback_ids=query_fallback_ids,
                    query_gene_baseline=torch.from_numpy(
                        donor_baseline[:, supported_array]
                    ).to(device),
                    query_read_depth=torch.from_numpy(donor_read_depth).to(device),
                    cell_chunk_size=args.cell_chunk_size,
                    gene_chunk_size=args.gene_chunk_size,
                )
                # Three legacy H1 symbols have neither SE nor VCC fallback vectors.
                # Preserve the full measured panel by assigning their matched-control
                # baseline, rather than silently dropping genes or cells.
                abundance = torch.zeros(
                    (len(pert_indices), len(genes)), dtype=torch.float32, device=device
                )
                abundance[:, supported_device] = torch.expm1(decoded.clamp_min(0))
                if len(unsupported_array):
                    missing_baseline = torch.from_numpy(
                        np.expm1(donor_baseline[:, unsupported_array])
                    ).to(device)
                    abundance[:, unsupported_device] = missing_baseline
                probabilities = abundance / abundance.sum(dim=1, keepdim=True).clamp_min(1e-12)
                donor_depths = np.asarray(data.X[donors].sum(axis=1)).ravel().astype(np.int64)
                # Deterministic integer expected counts are sufficient for local
                # pseudobulk/PDS evaluation and avoid injecting sampling noise.
                counts = torch.round(
                    probabilities * torch.from_numpy(donor_depths).to(device)[:, None]
                ).cpu().numpy().astype(np.float32)
                blocks.append(sparse.csr_matrix(counts))
                output_obs.append(obs.iloc[pert_indices].copy())
            print(f"[{target_number:03d}/{len(targets):03d}] {target}", flush=True)
    finally:
        data.file.close()

    prediction = ad.AnnData(
        X=sparse.vstack(blocks, format="csr"),
        obs=pd.concat(output_obs, axis=0),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    prediction.obs_names_make_unique()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_h5ad(args.output, compression="lzf")
    print(f"Wrote {args.output} shape={prediction.shape} nnz={prediction.X.nnz}", flush=True)

    # Score against exactly the same perturbation panel.  If the full H1
    # reference were used, PDS would rank the shared predictions against 123
    # identity embeddings that had no training signal, obscuring the question
    # this LOCO diagnostic is intended to answer.
    if args.write_real:
        real_mask = np.isin(labels, ["non-targeting", *targets])
        real_data = ad.read_h5ad(args.data, backed="r")
        try:
            real_subset = real_data[real_mask].to_memory()
        finally:
            real_data.file.close()
        real_subset.obs_names_make_unique()
        real_output.parent.mkdir(parents=True, exist_ok=True)
        real_subset.write_h5ad(real_output, compression="lzf")
        print(f"Wrote {real_output} shape={real_subset.shape} nnz={real_subset.X.nnz}", flush=True)


if __name__ == "__main__":
    main()
