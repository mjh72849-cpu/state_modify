#!/usr/bin/env python3
"""Prepare disk-conscious perturbation datasets for VCC training.

Modes: ``smoke`` builds tiny fixtures; ``focused`` keeps locally observed VCC
targets, strong extra targets, and batch-matched controls; ``hek293t-stream``
reads only selected remote Parquet shards and never downloads the full atlas.
Prepared files retain raw counts because CP10K/log1p is a training transform.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Iterable

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from state.tx.vcc.data import canonical_gene_name
from state.tx.vcc.focus_data import FocusDatasetSpec, default_focus_dataset_specs, standardize_focus_metadata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VCC_TARGETS = ROOT / "data/vcc_2026_controls/pert_counts.csv"
DEFAULT_HEK_GUIDES = ROOT / "data/raw/resource/XAtlas/HEK293T_filtered_guide_calls_per_cell.csv.gz"
HF_REPO = "Xaira-Therapeutics/X-Atlas-Orion"
FOCUSED_SEED_OFFSETS = {
    "arc_h1_train": 0, "replogle_k562_gwps": 1,
    "xatlas_hct116": 2, "replogle_k562_essential": 3,
}


def _decode(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O"}:
        return np.asarray(
            [value.decode() if isinstance(value, (bytes, bytearray)) else str(value) for value in values],
            dtype=object,
        )
    return values


def _read_h5_column(handle: h5py.File, group: str, name: str) -> np.ndarray:
    node = handle[f"{group}/{name}"]
    if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        codes = node["codes"][:]
        categories = _decode(node["categories"][:])
        result = np.empty(len(codes), dtype=object)
        valid = codes >= 0
        result[valid] = categories[codes[valid]]
        result[~valid] = ""
        return result
    if isinstance(node, h5py.Dataset) and "categories" in node.attrs:
        codes = node[:]
        categories = _decode(handle[node.attrs["categories"]][:])
        result = np.empty(len(codes), dtype=object)
        valid = codes >= 0
        result[valid] = categories[codes[valid]]
        result[~valid] = ""
        return result
    return _decode(node[:])


def _load_vcc_targets(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"No VCC targets found in {path}")
    preferred = next((c for c in ("target_gene", "gene", "perturbation") if c in frame), frame.columns[0])
    targets = list(dict.fromkeys(canonical_gene_name(v) for v in frame[preferred].astype(str)))
    if len(targets) != 300:
        raise ValueError(f"Expected 300 VCC targets, found {len(targets)} in {path}")
    return targets


def _protein_supported_targets(protein_embeddings: dict) -> set[str]:
    return {
        canonical_gene_name(name)
        for name, value in protein_embeddings.items()
        if torch.as_tensor(value).abs().sum().item() > 0
    }


def _source_metadata(spec: FocusDatasetSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(spec.path, "r") as handle:
        raw_pert = _read_h5_column(handle, "obs", spec.perturbation_key).astype(str)
        batches = _read_h5_column(handle, "obs", spec.batch_key).astype(str)
        eligible = np.ones(len(raw_pert), dtype=bool)
        if "pass_guide_filter" in spec.qc_keys:
            eligible &= _read_h5_column(handle, "obs", "pass_guide_filter").astype(bool)
    return raw_pert, batches, eligible


def _canonicalize_perturbations(values: Iterable[str], control_labels: tuple[str, ...]) -> np.ndarray:
    controls = {canonical_gene_name(label) for label in control_labels}
    return np.asarray(
        ["NON-TARGETING" if canonical_gene_name(v) in controls else canonical_gene_name(v) for v in values],
        dtype=object,
    )


def _sample_indices(pool: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    return pool if len(pool) <= cap else np.sort(rng.choice(pool, size=cap, replace=False))


def _select_smoke_indices(
    spec: FocusDatasetSpec,
    *,
    perturbations: int,
    cells_per_group: int,
    seed: int,
    supported_targets: set[str],
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    raw_pert, batch, eligible = _source_metadata(spec)
    pert = _canonicalize_perturbations(raw_pert, spec.control_labels)
    control_mask = eligible & (pert == "NON-TARGETING")
    control_batches = set(batch[control_mask])
    candidates: list[tuple[int, str, str]] = []
    for (batch_name, target), count in Counter(zip(batch[eligible], pert[eligible])).items():
        if target == "NON-TARGETING" or batch_name not in control_batches or target not in supported_targets:
            continue
        candidates.append((count, target, str(batch_name)))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _count, target, batch_name in candidates:
        if target in seen:
            continue
        chosen.append((target, batch_name))
        seen.add(target)
        if len(chosen) == perturbations:
            break
    if len(chosen) < perturbations:
        raise ValueError(f"{spec.name}: only {len(chosen)} perturbations have matched controls")
    selected: list[int] = []
    used_batches: set[str] = set()
    for target, batch_name in chosen:
        pool = np.flatnonzero(eligible & (pert == target) & (batch == batch_name))
        selected.extend(_sample_indices(pool, cells_per_group, rng).tolist())
        used_batches.add(batch_name)
    for batch_name in sorted(used_batches):
        pool = np.flatnonzero(control_mask & (batch == batch_name))
        selected.extend(_sample_indices(pool, cells_per_group, rng).tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int64), [target for target, _ in chosen]


def _select_focused_indices(
    spec: FocusDatasetSpec,
    *,
    vcc_targets: list[str],
    supported_targets: set[str],
    vcc_max_cells: int,
    extra_max_cells: int,
    extra_min_cells: int,
    max_extra_targets: int,
    controls_per_batch: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Select cells from metadata only; never touch X in the planning pass."""
    rng = np.random.default_rng(seed)
    raw_pert, batches, eligible = _source_metadata(spec)
    pert = _canonicalize_perturbations(raw_pert, spec.control_labels)
    control_mask = eligible & (pert == "NON-TARGETING")
    eligible &= np.isin(batches, np.unique(batches[control_mask]))
    counts = Counter(pert[eligible])
    vcc_set = set(vcc_targets)
    present_vcc = [target for target in vcc_targets if counts[target] > 0]
    extras = [
        (counts[target], target)
        for target in counts
        if target not in vcc_set
        and target != "NON-TARGETING"
        and target in supported_targets
        and counts[target] >= extra_min_cells
    ]
    extras.sort(key=lambda item: (-item[0], item[1]))
    chosen_extras = [target for _count, target in extras[:max_extra_targets]]
    rows: list[dict] = []
    selected: list[int] = []
    used_batches: set[str] = set()
    for category, targets, cap in (
        ("vcc", present_vcc, vcc_max_cells),
        ("extra", chosen_extras, extra_max_cells),
    ):
        for target in targets:
            pool = np.flatnonzero(eligible & (pert == target))
            chosen = _sample_indices(pool, cap, rng)
            selected.extend(chosen.tolist())
            used_batches.update(batches[chosen].tolist())
            rows.append({
                "dataset": spec.name, "category": category, "target_gene": target,
                "source_cells": len(pool), "selected_cells": len(chosen), "status": "selected",
            })
    for target in vcc_targets:
        if target not in present_vcc:
            rows.append({
                "dataset": spec.name, "category": "vcc", "target_gene": target,
                "source_cells": 0, "selected_cells": 0, "status": "absent_in_source",
            })
    control_selected = 0
    for batch_name in sorted(used_batches):
        chosen = _sample_indices(np.flatnonzero(control_mask & (batches == batch_name)), controls_per_batch, rng)
        selected.extend(chosen.tolist())
        control_selected += len(chosen)
    rows.append({
        "dataset": spec.name, "category": "control", "target_gene": "NON-TARGETING",
        "source_cells": int(control_mask.sum()), "selected_cells": control_selected, "status": "selected",
    })
    unique_indices = np.asarray(sorted(set(selected)), dtype=np.int64)
    summary = {
        "dataset": spec.name,
        "source_cells": int(len(pert)),
        "qc_and_control_batch_eligible_cells": int(eligible.sum()),
        "selected_cells": int(len(unique_indices)),
        "vcc_targets_present": len(present_vcc),
        "vcc_targets_absent": len(vcc_targets) - len(present_vcc),
        "extra_targets": len(chosen_extras),
        "control_batches": len(used_batches),
        "selected_controls": control_selected,
    }
    return unique_indices, pd.DataFrame(rows), summary


def _validate_raw_counts(adata: ad.AnnData, name: str) -> None:
    matrix = adata.X
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).reshape(-1)
    if values.size and (not np.isfinite(values).all() or values.min() < 0):
        raise ValueError(f"{name}: X must contain finite non-negative counts")
    sample = values[: min(values.size, 100_000)]
    if sample.size and not np.allclose(sample, np.rint(sample), atol=1e-5):
        raise ValueError(f"{name}: X does not appear to contain raw integer counts")


def _collapse_duplicate_gene_symbols(adata: ad.AnnData) -> ad.AnnData:
    """Sum duplicate gene-symbol columns so one decoder query has one target.

    Some legacy count matrices retain multiple Ensembl versions for the same
    canonical symbol (K562 has duplicate TBCE and HSPA14 columns).  Keeping
    both would silently give one semantic gene embedding two different target
    columns.  Counts are therefore summed, matching the older VCC preprocessing.
    """
    names = pd.Index(adata.var["gene_name"].astype(str).map(canonical_gene_name))
    if names.is_unique:
        return adata
    unique_names = pd.Index(pd.unique(names))
    name_to_column = {name: index for index, name in enumerate(unique_names)}
    columns = np.asarray([name_to_column[name] for name in names], dtype=np.int64)
    aggregation = sp.csr_matrix(
        (np.ones(len(names), dtype=np.float32), (np.arange(len(names)), columns)),
        shape=(len(names), len(unique_names)),
    )
    merged_x = sp.csr_matrix(adata.X) @ aggregation
    first = np.asarray([np.flatnonzero(names == name)[0] for name in unique_names])
    result = adata[:, first].copy()
    result.X = merged_x.tocsr()
    result.var["gene_name"] = unique_names.astype(str)
    result.var["source_feature_ids"] = [
        ";".join(adata.var_names[names == name].astype(str)) for name in unique_names
    ]
    # Canonical gene symbols are now unique and are the safest cross-dataset key.
    result.var_names = unique_names.astype(str)
    return result


def _materialize_local_subset(
    spec: FocusDatasetSpec,
    indices: np.ndarray,
    output: Path,
    preparation: dict,
    *,
    overwrite: bool,
) -> Path:
    if output.exists() and not overwrite:
        with h5py.File(output, "r") as handle:
            shape = tuple(handle["X"].attrs["shape"])
            saved = handle["uns/vcc_preparation"]
            for key, expected in preparation.items():
                if isinstance(expected, (str, int)):
                    if key not in saved:
                        raise ValueError(f"{output}: missing preparation field {key}; use a new output directory")
                    actual = saved[key][()]
                    if isinstance(actual, bytes):
                        actual = actual.decode()
                    if actual != expected:
                        raise ValueError(f"{output}: {key} differs; use a new output directory or --overwrite")
            if shape[0] != len(indices):
                raise ValueError(f"{output}: selection row count differs; use a new output directory or --overwrite")
        print(f"Reusing existing {output}", flush=True)
        return output
    backed = ad.read_h5ad(spec.path, backed="r")
    try:
        subset = backed[indices].to_memory()
    finally:
        if backed.file.is_open:
            backed.file.close()
    subset = standardize_focus_metadata(subset, spec, copy=False)
    subset = _collapse_duplicate_gene_symbols(subset)
    subset.obs["target_gene"] = [
        "non-targeting" if canonical_gene_name(v) == "NON-TARGETING" else canonical_gene_name(v)
        for v in subset.obs["target_gene"].astype(str)
    ]
    subset.obs_names = pd.Index([f"{spec.name}::{name}" for name in subset.obs_names.astype(str)])
    for column in ("target_gene", "batch", "cell_type", "dataset"):
        subset.obs[column] = subset.obs[column].astype("category")
    subset.var["gene_name"] = subset.var["gene_name"].astype("category")
    subset.uns.pop("log1p", None)
    _validate_raw_counts(subset, spec.name)
    subset.X = sp.csr_matrix(subset.X)
    subset.uns["vcc_preparation"] = preparation
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.h5ad")
    subset.write_h5ad(temporary, compression="gzip")
    temporary.replace(output)
    return output


def _embed_with_se(
    inputs: dict[str, Path], protein_embeddings: dict, batch_size: int, *, overwrite: bool
) -> dict[str, Path]:
    """Load SE once and embed every prepared dataset."""
    from state.emb.inference import Inference

    pending = {
        name: path for name, path in inputs.items()
        if overwrite or not path.with_name(path.stem + ".xstate.h5ad").exists()
    }
    outputs = {
        name: path.with_name(path.stem + ".xstate.h5ad")
        for name, path in inputs.items() if name not in pending
    }
    if not pending:
        return outputs
    inferer = Inference(cfg=None, protein_embeds=protein_embeddings)
    inferer.load_model(ROOT / "external/SE-600M/se600m_epoch16.ckpt")
    for name, input_path in pending.items():
        output_path = input_path.with_name(input_path.stem + ".xstate.h5ad")
        if overwrite and output_path.exists():
            output_path.unlink()
        inferer.encode_adata(
            input_adata_path=input_path, output_adata_path=output_path,
            emb_key="X_state", batch_size=batch_size,
        )
        outputs[name] = output_path
    return outputs


def _write_toml(outputs: dict[str, Path], output_path: Path) -> None:
    lines = ["[datasets]"]
    lines.extend(f'{name} = "{path.resolve()}"' for name, path in outputs.items())
    lines.extend(["", "[training]"])
    lines.extend(f'{name} = "train"' for name in outputs)
    lines.extend(["", "[zeroshot]", "", "[fewshot]"])
    output_path.write_text("\n".join(lines) + "\n")


def _write_smoke_toml(outputs: dict[str, Path], targets: dict[str, str], output_path: Path) -> None:
    _write_toml(outputs, output_path)
    with output_path.open("a") as handle:
        for name, target in targets.items():
            context = "H1" if "h1" in name else "K562" if "k562" in name else "HCT116"
            handle.write(f'\n[fewshot."{name}.{context}"]\nval = ["{target}"]\n')


def _selected_local_specs(names: str) -> list[FocusDatasetSpec]:
    requested = {name.strip() for name in names.split(",") if name.strip()}
    registry = {spec.name: spec for spec in default_focus_dataset_specs()}
    unknown = requested - registry.keys()
    if unknown:
        raise ValueError(f"Unknown local datasets: {sorted(unknown)}")
    return [registry[name] for name in registry if name in requested]


def _run_smoke(args: argparse.Namespace, protein_embeddings: dict) -> None:
    supported = _protein_supported_targets(protein_embeddings)
    outputs: dict[str, Path] = {}
    validation_targets: dict[str, str] = {}
    for offset, spec in enumerate(default_focus_dataset_specs()):
        indices, targets = _select_smoke_indices(
            spec, perturbations=args.perturbations, cells_per_group=args.cells_per_group,
            seed=args.seed + offset, supported_targets=supported,
        )
        if args.dry_run:
            print(f"{spec.name}: {len(indices)} cells; targets={targets}", flush=True)
            continue
        outputs[spec.name] = _materialize_local_subset(
            spec, indices, args.output_dir / f"{spec.name}.smoke.h5ad",
            {"source": str(spec.path), "mode": "smoke", "seed": args.seed + offset,
             "cells_per_group": args.cells_per_group, "selected_targets": targets},
            overwrite=args.overwrite,
        )
        validation_targets[spec.name] = targets[-1]
    if args.dry_run:
        return
    if args.run_se:
        outputs = _embed_with_se(outputs, protein_embeddings, args.se_batch_size, overwrite=args.overwrite)
    _write_smoke_toml(outputs, validation_targets, args.output_dir / ("vcc_smoke.toml" if args.run_se else "vcc_smoke_unembedded.toml"))


def _run_focused(args: argparse.Namespace, protein_embeddings: dict) -> None:
    vcc_targets = _load_vcc_targets(args.targets_file)
    supported = _protein_supported_targets(protein_embeddings)
    reports: list[pd.DataFrame] = []
    summaries: list[dict] = []
    plans: list[tuple[FocusDatasetSpec, np.ndarray, dict]] = []
    for spec in _selected_local_specs(args.datasets):
        indices, report, summary = _select_focused_indices(
            spec, vcc_targets=vcc_targets, supported_targets=supported,
            vcc_max_cells=args.vcc_max_cells, extra_max_cells=args.extra_max_cells,
            extra_min_cells=args.extra_min_cells, max_extra_targets=args.max_extra_targets,
            controls_per_batch=args.controls_per_batch,
            seed=args.seed + FOCUSED_SEED_OFFSETS[spec.name],
        )
        reports.append(report)
        summaries.append(summary)
        plans.append((spec, indices, summary))
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(reports, ignore_index=True).to_csv(args.output_dir / "selection_report.csv", index=False)
    (args.output_dir / "selection_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    selected_cells = sum(s["selected_cells"] for s in summaries)
    free = shutil.disk_usage(args.output_dir).free
    # Source-size scaling is an estimate, not a guaranteed CSR compression ratio.
    estimated_counts = sum(
        spec.path.stat().st_size * len(indices) / summary["source_cells"]
        for spec, indices, summary in plans
    )
    estimated_required = estimated_counts * (2 if args.run_se else 1) + selected_cells * 2058 * 4
    print(
        f"Selected {selected_cells:,} cells; X_state alone will use "
        f"{selected_cells * 2058 * 4 / 2**30:.2f} GiB; free disk is {free / 2**30:.2f} GiB.",
        flush=True,
    )
    print(f"Estimated output budget: {estimated_required / 2**30:.2f} GiB (source-size scaling).", flush=True)
    if args.dry_run:
        return
    if free < estimated_required * 1.5 + 5 * 2**30:
        raise OSError("Insufficient free disk for estimated outputs plus working headroom")
    outputs: dict[str, Path] = {}
    for spec, indices, summary in plans:
        print(f"Materializing {spec.name}: {len(indices):,} cells", flush=True)
        outputs[spec.name] = _materialize_local_subset(
            spec, indices, args.output_dir / f"{spec.name}.focused.h5ad",
            {"source": str(spec.path), "mode": "focused", "selection": summary,
             "vcc_max_cells": args.vcc_max_cells, "extra_max_cells": args.extra_max_cells,
             "extra_min_cells": args.extra_min_cells, "max_extra_targets": args.max_extra_targets,
             "controls_per_batch": args.controls_per_batch, "seed": args.seed},
            overwrite=args.overwrite,
        )
    if args.run_se:
        outputs = _embed_with_se(outputs, protein_embeddings, args.se_batch_size, overwrite=args.overwrite)
    _write_toml(outputs, args.output_dir / ("vcc_focused.toml" if args.run_se else "vcc_focused_unembedded.toml"))


def _hek_batch_from_barcode(barcode: str) -> str:
    match = re.search(r"-(HEK293T_Batch\d+)$", barcode)
    if not match:
        raise ValueError(f"Cannot parse HEK293T batch from barcode {barcode!r}")
    return match.group(1)


def _choose_hek_shards(frame: pd.DataFrame, vcc_targets: list[str], max_shards: int) -> list[str]:
    """Set-cover guarantees 300-target coverage, then extra shards improve depth."""
    vcc_set = set(vcc_targets)
    per_batch = {
        batch: Counter(group.loc[group["target_gene"].isin(vcc_set), "target_gene"])
        for batch, group in frame.groupby("batch", observed=True)
    }
    uncovered = set(vcc_targets)
    selected: list[str] = []
    while uncovered:
        remaining = [batch for batch in per_batch if batch not in selected]
        if not remaining:
            break
        best = max(remaining, key=lambda b: (len(set(per_batch[b]) & uncovered), -int(b.split("Batch")[-1])))
        covered = set(per_batch[best]) & uncovered
        if not covered:
            break
        selected.append(best)
        uncovered -= covered
    if uncovered:
        raise ValueError(f"HEK293T cannot cover VCC targets: {sorted(uncovered)}")
    desired = {target: 64 for target in vcc_targets}
    achieved = Counter()
    for batch in selected:
        achieved.update(per_batch[batch])
    while len(selected) < max_shards:
        remaining = [batch for batch in per_batch if batch not in selected]
        if not remaining:
            break
        def gain(batch: str) -> int:
            return sum(min(count, max(0, desired[target] - achieved[target])) for target, count in per_batch[batch].items())
        best = max(remaining, key=lambda b: (gain(b), -int(b.split("Batch")[-1])))
        if gain(best) == 0:
            break
        selected.append(best)
        achieved.update(per_batch[best])
    return sorted(selected, key=lambda value: int(value.split("Batch")[-1]))


def _plan_hek(args: argparse.Namespace, protein_embeddings: dict) -> tuple[pd.DataFrame, list[str], pd.DataFrame, dict]:
    vcc_targets = _load_vcc_targets(args.targets_file)
    supported = _protein_supported_targets(protein_embeddings)
    guides = pd.read_csv(
        args.hek_guides, usecols=["cell_barcode", "gene_target", "pass_guide_filter"],
        dtype={"cell_barcode": str, "gene_target": str},
    )
    guides = guides.loc[guides["pass_guide_filter"].astype(bool)].copy()
    guides["target_gene"] = guides["gene_target"].map(canonical_gene_name)
    guides["batch"] = guides["cell_barcode"].map(_hek_batch_from_barcode)
    shards = _choose_hek_shards(guides, vcc_targets, args.hek_max_shards)
    available = guides.loc[guides["batch"].isin(shards)].copy()
    counts = available["target_gene"].value_counts()
    global_counts = guides["target_gene"].value_counts()
    extra_candidates = [
        target for target, count in global_counts.items()
        if target not in set(vcc_targets) and target != "NON-TARGETING"
        and target in supported and count >= args.extra_min_cells and counts.get(target, 0) > 0
    ]
    extra_candidates.sort(key=lambda target: (-int(global_counts[target]), -int(counts[target]), target))
    extras = extra_candidates[: args.max_extra_targets]
    rng = np.random.default_rng(args.seed)
    selected_parts: list[pd.DataFrame] = []
    report_rows: list[dict] = []
    for category, targets, cap in (
        ("vcc", vcc_targets, args.vcc_max_cells), ("extra", extras, args.extra_max_cells),
    ):
        for target in targets:
            pool = available.loc[available["target_gene"] == target]
            if len(pool) > cap:
                pool = pool.iloc[np.sort(rng.choice(len(pool), size=cap, replace=False))]
            selected_parts.append(pool)
            report_rows.append({
                "dataset": "xatlas_hek293t", "category": category, "target_gene": target,
                "source_cells_in_streamed_shards": int(counts.get(target, 0)),
                "selected_cells": len(pool), "status": "selected" if len(pool) else "absent_in_selected_shards",
            })
    perturbation_cells = pd.concat(selected_parts, ignore_index=True)
    used_batches = set(perturbation_cells["batch"])
    controls = available.loc[(available["target_gene"] == "NON-TARGETING") & available["batch"].isin(used_batches)]
    control_parts = []
    for _batch, group in controls.groupby("batch", observed=True):
        if len(group) > args.controls_per_batch:
            group = group.iloc[np.sort(rng.choice(len(group), size=args.controls_per_batch, replace=False))]
        control_parts.append(group)
    selected = pd.concat([perturbation_cells, *control_parts], ignore_index=True).drop_duplicates("cell_barcode")
    report_rows.append({
        "dataset": "xatlas_hek293t", "category": "control", "target_gene": "NON-TARGETING",
        "source_cells_in_streamed_shards": len(controls),
        "selected_cells": sum(len(part) for part in control_parts), "status": "selected",
    })
    summary = {
        "dataset": "xatlas_hek293t", "streamed_shards": shards,
        "num_streamed_shards": int(len(shards)), "selected_cells": int(len(selected)),
        "vcc_targets_present": int(sum(counts.get(target, 0) > 0 for target in vcc_targets)),
        "extra_targets": int(len(extras)),
        "selected_controls": int(sum(len(part) for part in control_parts)),
    }
    return selected, shards, pd.DataFrame(report_rows), summary


def _materialize_hek_stream(selected: pd.DataFrame, shards: list[str], output: Path, summary: dict) -> Path:
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    with fs.open(f"datasets/{HF_REPO}/metadata/gene_metadata.parquet", "rb") as handle:
        genes = pq.read_table(handle).to_pandas().sort_values("gene_token_id")
    token_to_column = {int(token): i for i, token in enumerate(genes["gene_token_id"])}
    wanted = set(selected["cell_barcode"])
    metadata_by_barcode = selected.set_index("cell_barcode")
    matrices: list[sp.csr_matrix] = []
    obs_parts: list[pd.DataFrame] = []
    columns = ["gene_token_id", "gene_expression", "cell_barcode", "sample",
               "n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt"]
    cache_dir = output.parent / "stream_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    selection_hash = hashlib.sha256("\n".join(sorted(wanted)).encode()).hexdigest()[:16]

    def read_selected_shard(shard: str) -> pd.DataFrame:
        cache = cache_dir / f"{shard}.{selection_hash}.parquet"
        if cache.exists():
            return pq.read_table(cache).to_pandas()
        print(f"Streaming HEK shard: {shard}", flush=True)
        with HfFileSystem().open(
            f"datasets/{HF_REPO}/data/{shard}.parquet", "rb",
            block_size=8 * 2**20, cache_type="blockcache",
        ) as handle:
            frame = pq.read_table(handle, columns=columns).to_pandas()
        frame = frame.loc[frame["cell_barcode"].isin(wanted)]
        # Cache only selected rows; complete remote shards never reach disk.
        temporary = cache.with_suffix(".tmp.parquet")
        frame.to_parquet(temporary, index=False)
        temporary.replace(cache)
        return frame

    def selected_shards():
        # Bounded concurrency hides HTTP latency; map preserves source order.
        with ThreadPoolExecutor(max_workers=3) as pool:
            yield from pool.map(read_selected_shard, shards)

    for number, (shard, frame) in enumerate(zip(shards, selected_shards()), start=1):
        print(f"Materializing HEK shard {number}/{len(shards)}: {shard}", flush=True)
        if frame.empty:
            continue
        row_indices: list[int] = []
        col_indices: list[int] = []
        values: list[float] = []
        for row, (tokens, expression) in enumerate(zip(frame["gene_token_id"], frame["gene_expression"])):
            row_indices.extend([row] * len(tokens))
            col_indices.extend(token_to_column[int(token)] for token in tokens)
            values.extend(expression)
        matrix = sp.csr_matrix(
            (values, (row_indices, col_indices)), shape=(len(frame), len(genes)), dtype=np.float32
        )
        matrix.eliminate_zeros()
        matrices.append(matrix)
        obs = frame.drop(columns=["gene_token_id", "gene_expression"]).set_index("cell_barcode")
        obs["target_gene"] = metadata_by_barcode.loc[obs.index, "target_gene"].values
        obs["batch"] = "xatlas_hek293t::" + metadata_by_barcode.loc[obs.index, "batch"].values
        obs["cell_type"] = "HEK293T"
        obs["dataset"] = "xatlas_hek293t"
        obs_parts.append(obs)
    if not matrices:
        raise RuntimeError("No selected HEK293T barcodes were found in streamed shards")
    x = sp.vstack(matrices, format="csr")
    obs = pd.concat(obs_parts)
    missing = wanted - set(obs.index)
    if missing:
        raise RuntimeError(f"{len(missing)} selected HEK293T barcodes were absent from Parquet shards")
    obs.index = pd.Index([f"xatlas_hek293t::{value}" for value in obs.index])
    obs["target_gene"] = obs["target_gene"].replace({"NON-TARGETING": "non-targeting"})
    for column in ("target_gene", "batch", "cell_type", "dataset"):
        obs[column] = obs[column].astype("category")
    var = genes.set_index("ensembl_id")
    var["gene_name"] = [canonical_gene_name(value) for value in var["gene_name"]]
    var["gene_name"] = var["gene_name"].astype("category")
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata = _collapse_duplicate_gene_symbols(adata)
    _validate_raw_counts(adata, "xatlas_hek293t")
    adata.uns["vcc_preparation"] = {"mode": "hek293t-stream", **summary}
    temporary = output.with_suffix(".tmp.h5ad")
    adata.write_h5ad(temporary, compression="gzip")
    temporary.replace(output)
    return output


def _run_hek_stream(args: argparse.Namespace, protein_embeddings: dict) -> None:
    selected, shards, report, summary = _plan_hek(args, protein_embeddings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "hek293t_selection_report.csv", index=False)
    selected[["cell_barcode", "target_gene", "batch"]].to_csv(
        args.output_dir / "hek293t_selected_barcodes.csv.gz", index=False, compression="gzip"
    )
    (args.output_dir / "hek293t_streaming_plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    output = args.output_dir / "xatlas_hek293t.focused.h5ad"
    if output.exists() and not args.overwrite:
        _validate_hek_cached_selection(output, selected)
        print(f"Reusing existing {output}", flush=True)
    else:
        output = _materialize_hek_stream(selected, shards, output, summary)
    outputs = {"xatlas_hek293t": output}
    if args.run_se:
        outputs = _embed_with_se(outputs, protein_embeddings, args.se_batch_size, overwrite=args.overwrite)
    _write_toml(outputs, args.output_dir / ("vcc_hek293t.toml" if args.run_se else "vcc_hek293t_unembedded.toml"))


def _validate_hek_cached_selection(output: Path, selected: pd.DataFrame) -> None:
    cached = ad.read_h5ad(output, backed="r")
    try:
        actual = cached.obs[["target_gene"]].copy()
        actual.index = actual.index.str.removeprefix("xatlas_hek293t::")
        expected = selected.set_index("cell_barcode")[["target_gene"]].copy()
        actual["target_gene"] = actual["target_gene"].astype(str).str.upper()
        expected["target_gene"] = expected["target_gene"].astype(str).str.upper()
        if not actual.sort_index().equals(expected.sort_index()):
            raise ValueError(f"{output}: barcode/target selection differs; use a new output directory or --overwrite")
    finally:
        cached.file.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "focused", "hek293t-stream"), default="smoke")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--targets-file", type=Path, default=DEFAULT_VCC_TARGETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-se", action="store_true")
    parser.add_argument("--se-batch-size", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--perturbations", type=int, default=4)
    parser.add_argument("--cells-per-group", type=int, default=64)
    parser.add_argument("--datasets", default="arc_h1_train,replogle_k562_gwps,xatlas_hct116")
    parser.add_argument("--vcc-max-cells", type=int, default=256)
    parser.add_argument("--extra-max-cells", type=int, default=128)
    parser.add_argument("--extra-min-cells", type=int, default=128)
    parser.add_argument("--max-extra-targets", type=int, default=500)
    parser.add_argument("--controls-per-batch", type=int, default=128)
    parser.add_argument("--hek-guides", type=Path, default=DEFAULT_HEK_GUIDES)
    parser.add_argument("--hek-max-shards", type=int, default=12)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = ROOT / "data/prepared" / ("smoke" if args.mode == "smoke" else "focused")
    for name in ("cells_per_group", "vcc_max_cells", "extra_max_cells", "extra_min_cells",
                 "max_extra_targets", "controls_per_batch", "hek_max_shards", "se_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    protein_embeddings = torch.load(
        ROOT / "external/SE-600M/protein_embeddings.pt", map_location="cpu", weights_only=False
    )
    if args.mode == "smoke":
        _run_smoke(args, protein_embeddings)
    elif args.mode == "focused":
        _run_focused(args, protein_embeddings)
    else:
        _run_hek_stream(args, protein_embeddings)


if __name__ == "__main__":
    main()
