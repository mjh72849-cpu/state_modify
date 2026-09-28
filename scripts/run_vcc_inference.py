#!/usr/bin/env python3
"""Run a trained panel-free STATE model and build an official VCC submission.

The prediction H5AD is written one ``[400, 18533]`` target block at a time.
Pass ``--pack`` to validate and package it with the installed official ``vcc``
CLI. Temporary packaging files are forced onto ``--scratch-dir`` rather than
the system /tmp filesystem.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from state.tx.models.decoders import PanelFreeGeneDecoder
from state.tx.utils import get_lightning_module
from state.tx.vcc import (
    VCCPredictionWriter,
    build_gene_query_features,
    canonical_gene_name,
    control_log_cp10k_read_depth,
    control_log_cp10k_rows,
    load_gene_name_list,
    predict_vcc_pooled_counts,
)


ROOT = Path(__file__).resolve().parents[1]


def _canonical_nonzero_features(raw: dict) -> dict[str, torch.Tensor]:
    """Canonicalize aliases while preferring an available non-zero vector."""
    result: dict[str, torch.Tensor] = {}
    for raw_name, raw_value in raw.items():
        name = canonical_gene_name(raw_name)
        value = torch.as_tensor(raw_value, dtype=torch.float32).reshape(-1)
        current = result.get(name)
        if current is None or (current.abs().sum() == 0 and value.abs().sum() > 0):
            result[name] = value
        elif current.abs().sum() > 0 and value.abs().sum() > 0 and not torch.equal(current, value):
            raise ValueError(f"Conflicting protein embeddings canonicalize to {name!r}")
    return result


def _load_model(run_dir: Path, checkpoint: Path, device: torch.device):
    config_path = run_dir / "config.yaml"
    var_dims_path = run_dir / "var_dims.pkl"
    for path in (config_path, var_dims_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = yaml.safe_load(config_path.read_text())
    if cfg["model"]["name"].lower() != "state":
        raise ValueError("VCC inference currently requires model.name=state")
    with var_dims_path.open("rb") as handle:
        var_dims = pickle.load(handle)
    model = get_lightning_module(
        cfg["model"]["name"],
        cfg["data"]["kwargs"],
        cfg["model"]["kwargs"],
        cfg["training"],
        var_dims,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict, strict=True)
    if not isinstance(model.gene_decoder, PanelFreeGeneDecoder):
        raise TypeError("Checkpoint does not contain the panel-free VCC decoder")
    if not getattr(model, "trainable_perturbation_to_id", None):
        raise ValueError("Checkpoint has no saved trainable perturbation-name registry")
    model.to(device).eval()
    return model


def _verify_run_registries(run_dir: Path, model, fallback_names: list[str]) -> None:
    state_path = run_dir / "data_module.torch"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    saved_perts = state.get("trainable_perturbation_names")
    if saved_perts and list(saved_perts) != list(model.trainable_perturbation_names):
        raise ValueError("Model and saved data module use different perturbation-ID orders")
    saved_fallback = state.get("decoder_fallback_gene_names")
    if saved_fallback and list(saved_fallback) != fallback_names:
        raise ValueError("Current fallback registry differs from the registry used for training")


def _pack_submission(
    prediction: Path,
    output: Path,
    controls_dir: Path,
    scratch_dir: Path,
    overwrite: bool,
    expected_targets: list[str] | None = None,
) -> None:
    executable = shutil.which("vcc")
    if executable is None:
        raise FileNotFoundError("Official `vcc` CLI is not installed")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix="state_vcc_", dir=scratch_dir) as temporary:
        # vcc-cli currently documents --genes as a headerless ordered file.
        genes = pd.read_csv(controls_dir / "gene_names.csv", dtype=str)["gene_name"]
        headerless = Path(temporary) / "gene_names_headerless.csv"
        genes.to_csv(headerless, index=False, header=False)
        perturbations_path = controls_dir / "pert_counts.csv"
        if expected_targets is not None:
            official = pd.read_csv(perturbations_path, dtype={"target_gene": str})
            indexed = official.set_index("target_gene", drop=False)
            missing = [target for target in expected_targets if target not in indexed.index]
            if missing:
                raise ValueError(f"Debug targets are absent from the official manifest: {missing}")
            perturbations_path = Path(temporary) / "pert_counts_debug_subset.csv"
            indexed.loc[expected_targets].reset_index(drop=True).to_csv(
                perturbations_path, index=False
            )
        base = [
            executable, "prep", str(prediction),
            "--genes", str(headerless),
            "--perts", str(perturbations_path),
            "--pert-col", "target_gene",
            "--context-col", "context",
            "--contexts", "A,B,C",
            "--require-counts",
            "--reject-controls",
        ]
        environment = dict(os.environ)
        environment["TMPDIR"] = str(scratch_dir.resolve())
        subprocess.run([*base, "--dry-run"], check=True, env=environment)
        command = [*base, "--output", str(output)]
        if overwrite:
            command.append("--force")
        subprocess.run(command, check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.ckpt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, default=ROOT / "data/vcc_2026_controls")
    parser.add_argument("--prepared-controls", type=Path, default=ROOT / "data/prepared/vcc_inference")
    parser.add_argument("--protein-embeddings", type=Path, default=ROOT / "external/SE-600M/protein_embeddings.pt")
    parser.add_argument("--fallback", type=Path, default=ROOT / "assets/vcc_2026_se_fallback_genes.txt")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cell-chunk-size", type=int, default=256)
    parser.add_argument("--gene-chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20_260_927)
    parser.add_argument("--concentration", type=float)
    parser.add_argument(
        "--limit-targets",
        type=int,
        help="Debug only; use --pack-debug-subset to test packaging (not submittable)",
    )
    parser.add_argument("--pack", action="store_true")
    parser.add_argument(
        "--pack-debug-subset",
        action="store_true",
        help="Package --limit-targets output against a filtered manifest; the .vcc is not submittable",
    )
    parser.add_argument("--vcc-output", type=Path)
    parser.add_argument("--scratch-dir", type=Path, default=ROOT / "runs/vcc_pack_scratch")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = run_dir / "checkpoints" / checkpoint
    if args.output.suffix != ".h5ad":
        raise ValueError("--output must end in .h5ad")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        args.output.unlink()
    if args.cell_chunk_size <= 0 or args.gene_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")

    genes = pd.read_csv(args.controls_dir / "gene_names.csv", dtype=str)["gene_name"].tolist()
    targets = pd.read_csv(args.controls_dir / "pert_counts.csv", dtype=str)["target_gene"].tolist()
    if len(genes) != 18_533 or len(set(genes)) != len(genes):
        raise ValueError("Expected 18,533 unique official genes")
    if len(targets) != 300 or len(set(targets)) != len(targets):
        raise ValueError("Expected 300 unique official perturbations")
    if args.limit_targets is not None:
        if args.limit_targets <= 0 or args.limit_targets > len(targets):
            raise ValueError("--limit-targets must be in [1,300]")
        targets = targets[: args.limit_targets]
    if args.pack_debug_subset and not args.pack:
        raise ValueError("--pack-debug-subset requires --pack")
    if args.pack_debug_subset and args.limit_targets is None:
        raise ValueError("--pack-debug-subset requires --limit-targets")
    if args.pack and len(targets) != 300 and not args.pack_debug_subset:
        raise ValueError(
            "A debug --limit-targets output is not an official submission; "
            "pass --pack-debug-subset only for an end-to-end smoke test"
        )

    device = torch.device(args.device)
    model = _load_model(run_dir, checkpoint, device)
    fallback_names = load_gene_name_list(args.fallback)
    _verify_run_registries(run_dir, model, fallback_names)

    raw_proteins = torch.load(args.protein_embeddings, map_location="cpu", weights_only=False)
    proteins = _canonical_nonzero_features(raw_proteins)
    del raw_proteins
    query_embeddings, query_fallback_ids = build_gene_query_features(
        genes, proteins, fallback_names
    )
    query_embeddings = query_embeddings.to(device)
    query_fallback_ids = query_fallback_ids.to(device)

    target_names = [canonical_gene_name(target) for target in targets]
    missing_ids = sorted(set(target_names) - set(model.trainable_perturbation_to_id))
    missing_semantic = sorted(
        name for name in target_names if name not in proteins or proteins[name].abs().sum() == 0
    )
    if missing_ids or missing_semantic:
        raise ValueError(
            f"VCC target interface incomplete: missing_ids={missing_ids[:10]}, "
            f"missing_semantic={missing_semantic[:10]}"
        )

    with VCCPredictionWriter(args.output, targets, genes) as writer:
        for context_index, context in enumerate("ABC"):
            pool = np.load(args.prepared_controls / f"context_{context}.pool.npz")
            donor_indices = np.asarray(pool["donor_indices"], dtype=np.int64)
            library_sizes = torch.from_numpy(np.asarray(pool["library_sizes"], dtype=np.int64)).to(device)
            if donor_indices.shape != (400, 4):
                raise ValueError(f"Context {context}: invalid prepared pooling artifacts")
            controls = ad.read_h5ad(
                args.prepared_controls / f"context_{context}.xstate.h5ad", backed="r"
            )
            try:
                selected = np.asarray(
                    controls.obsm["X_state"][donor_indices.reshape(-1)], dtype=np.float32
                )
                # Preserve each donor's own control expression.  A single
                # mean log-CP10K baseline creates a Jensen/pseudobulk offset
                # that can dominate the perturbation-specific effect.
                baseline = control_log_cp10k_rows(
                    controls.X[donor_indices.reshape(-1)],
                    target_sum=10_000.0,
                )
                read_depth = control_log_cp10k_read_depth(
                    controls.X[donor_indices.reshape(-1)],
                    target_sum=10_000.0,
                )
            finally:
                controls.file.close()
            control_embeddings = torch.from_numpy(selected).to(device)
            baseline = torch.from_numpy(baseline).to(device)
            read_depth = torch.from_numpy(read_depth).to(device)
            local_pool = torch.arange(1_600, device=device).reshape(400, 4)

            progress = tqdm(zip(targets, target_names), total=len(targets), desc=f"context {context}")
            for target_index, (raw_target, target) in enumerate(progress):
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + context_index * len(targets) + target_index
                )
                counts = predict_vcc_pooled_counts(
                    model,
                    control_embeddings,
                    proteins[target].to(device),
                    query_embeddings,
                    local_pool,
                    library_sizes,
                    perturbation_ids=model.trainable_perturbation_to_id[target],
                    query_gene_fallback_ids=query_fallback_ids,
                    query_gene_baseline=baseline,
                    query_read_depth=read_depth,
                    cell_chunk_size=args.cell_chunk_size,
                    gene_chunk_size=args.gene_chunk_size,
                    concentration=args.concentration,
                    generator=generator,
                )
                writer.append(counts.cpu().numpy())
                progress.set_postfix_str(raw_target)

    print(f"Wrote {args.output}")
    if args.pack:
        output = args.vcc_output or args.output.with_suffix(".vcc")
        if args.pack_debug_subset:
            print("WARNING: packaging a debug target subset; this .vcc cannot be submitted")
        try:
            cli_version = version("vcc-cli")
        except PackageNotFoundError:
            cli_version = "external executable"
        print(f"Packaging with vcc-cli {cli_version}")
        _pack_submission(
            args.output,
            output,
            args.controls_dir,
            args.scratch_dir,
            args.overwrite,
            expected_targets=targets if args.pack_debug_subset else None,
        )
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
