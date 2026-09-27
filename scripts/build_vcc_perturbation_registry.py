#!/usr/bin/env python3
"""Build stable trainable perturbation IDs from prepared VCC training files."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

from state.tx.vcc.data import canonical_gene_name


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    ROOT / "data/prepared/focused/arc_h1_train.focused.xstate.h5ad",
    ROOT / "data/prepared/focused/replogle_k562_gwps.focused.xstate.h5ad",
    ROOT / "data/prepared/focused/xatlas_hct116.focused.xstate.h5ad",
    ROOT / "data/prepared/hek293t/xatlas_hek293t.focused.xstate.h5ad",
)


def build_registry(inputs: list[Path], output: Path, pert_col: str, control: str) -> list[str]:
    control = canonical_gene_name(control)
    targets: set[str] = set()
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        data = ad.read_h5ad(path, backed="r")
        try:
            if pert_col not in data.obs:
                raise KeyError(f"{path}: missing obs/{pert_col}")
            targets.update(canonical_gene_name(value) for value in data.obs[pert_col].unique())
        finally:
            data.file.close()
    targets.discard(control)
    ordered = sorted(targets)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "assets/vcc_trainable_perturbations.txt"
    )
    parser.add_argument("--pert-col", default="target_gene")
    parser.add_argument("--control", default="non-targeting")
    args = parser.parse_args()
    targets = build_registry(args.inputs, args.output, args.pert_col, args.control)
    print(f"Wrote {len(targets):,} stable perturbation IDs to {args.output}")


if __name__ == "__main__":
    main()
