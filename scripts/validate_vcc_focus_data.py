#!/usr/bin/env python3
"""Validate complete prepared count matrices and selection-report alignment."""
import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def validate(path: Path, selection_report: Path) -> dict:
    a = ad.read_h5ad(path, backed="r")
    try:
        assert a.obs_names.is_unique and a.var_names.is_unique, f"Duplicate indices: {path}"
        obs = a.obs
        targets = obs["target_gene"].astype(str)
        controls = targets.eq("non-targeting")
        assert controls.any(), f"No recognized controls: {path}"
        missing_batches = set(obs.loc[~controls, "batch"]) - set(obs.loc[controls, "batch"])
        assert not missing_batches, f"Missing batch controls: {missing_batches}"
        assert all(key in obs for key in ("dataset", "cell_type", "batch", "target_gene"))
        assert "gene_name" in a.var
        if "pass_guide_filter" in obs:
            assert obs.pass_guide_filter.astype(bool).all(), f"Guide QC failed: {path}"
        dataset = str(obs.dataset.iloc[0])
        report = pd.read_csv(selection_report)
        report = report.loc[report.dataset.eq(dataset)]
        assert not report.empty, f"Dataset missing from selection report: {dataset}"
        expected = report.set_index("target_gene").selected_cells.to_dict()
        actual = targets.str.upper().value_counts().to_dict()
        assert {k: int(v) for k, v in expected.items() if v} == actual, "Target counts differ from plan"
        assert a.n_obs == report.selected_cells.sum()
        vcc = report.loc[report.category.eq("vcc"), "selected_cells"]
        assert len(vcc) == 300, "Report must describe all 300 VCC targets"
        summary = {
            "file": str(path.resolve()), "dataset": dataset, "cells": a.n_obs, "genes": a.n_vars,
            "controls": int(controls.sum()), "batches": int(obs.batch.nunique()),
            "vcc_targets_present": int((vcc > 0).sum()),
            "vcc_targets_at_least_32_cells": int((vcc >= 32).sum()),
            "extra_targets": int((report.category.eq("extra") & report.selected_cells.gt(0)).sum()),
            "bytes": path.stat().st_size, "has_X_state": "X_state" in a.obsm,
        }
    finally:
        a.file.close()
    with h5py.File(path, "r") as handle:
        matrix = handle["X"]
        assert matrix.attrs["encoding-type"] == "csr_matrix"
        data = matrix["data"]
        for start in range(0, len(data), 1_000_000):
            values = data[start:start + 1_000_000]
            assert np.isfinite(values).all() and (values >= 0).all(), "Invalid counts"
            assert np.equal(values, np.rint(values)).all(), "Non-integer counts"
        indptr = matrix["indptr"][:]
        assert len(indptr) == summary["cells"] + 1
        assert indptr[-1] == len(data) == len(matrix["indices"])
        assert np.diff(indptr).min() > 0, "Empty cells require additional QC"
    summary["all_stored_counts_validated"] = True
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/vcc_focused_validation.json")
    parser.add_argument("--scope", choices=("all", "local", "hek"), default="all")
    args = parser.parse_args()
    focused = ROOT / "data/prepared/focused"
    hek = ROOT / "data/prepared/hek293t"
    jobs = [
        (focused / f"{name}.focused.h5ad", focused / "selection_report.csv")
        for name in ("arc_h1_train", "replogle_k562_gwps", "xatlas_hct116")
    ] + [(hek / "xatlas_hek293t.focused.h5ad", hek / "hek293t_selection_report.csv")]
    if args.scope == "local":
        jobs = jobs[:3]
    elif args.scope == "hek":
        jobs = jobs[3:]
    summaries = []
    for path, report in jobs:
        summary = validate(path, report)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
