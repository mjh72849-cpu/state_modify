#!/usr/bin/env python3
"""Validate and package an existing STATE prediction H5AD with vcc-cli."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

from run_vcc_inference import ROOT, _pack_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, default=ROOT / "data/vcc_2026_controls")
    parser.add_argument("--scratch-dir", type=Path, default=ROOT / "runs/vcc_pack_scratch")
    parser.add_argument(
        "--debug-subset",
        action="store_true",
        help="Validate against targets present in the H5AD; output is not submittable",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.prediction.is_file():
        raise FileNotFoundError(args.prediction)
    if args.prediction.suffix != ".h5ad" or args.output.suffix != ".vcc":
        raise ValueError("Expected a .h5ad prediction and .vcc output")
    expected_targets = None
    if args.debug_subset:
        prediction = ad.read_h5ad(args.prediction, backed="r")
        try:
            if "target_gene" not in prediction.obs:
                raise KeyError("Prediction is missing obs['target_gene']")
            expected_targets = list(dict.fromkeys(prediction.obs["target_gene"].astype(str)))
        finally:
            prediction.file.close()
        if not expected_targets or len(expected_targets) >= 300:
            raise ValueError("--debug-subset requires between 1 and 299 targets")
        print("WARNING: packaging a debug target subset; this .vcc cannot be submitted")

    _pack_submission(
        args.prediction,
        args.output,
        args.controls_dir,
        args.scratch_dir,
        args.overwrite,
        expected_targets=expected_targets,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
