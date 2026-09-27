"""Explicit interfaces for H1, HCT116, K562, and streamed HEK293T resources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import h5py
import anndata as ad

from .data import canonical_gene_name

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "raw"


@dataclass(frozen=True)
class FocusDatasetSpec:
    """Schema and role of one on-disk perturbation dataset."""

    name: str
    context: Literal["H1", "HCT116", "K562", "HEK293T"]
    path: Path
    perturbation_key: str
    control_labels: tuple[str, ...]
    batch_key: str
    gene_symbol_key: str
    role: Literal["train", "validation"] = "train"
    qc_keys: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


def default_focus_dataset_specs(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    *,
    include_h1_validation: bool = False,
) -> list[FocusDatasetSpec]:
    """Return the conservative default training registry.

    H1 validation is opt-in and the concatenated ``preprocessed/h1.h5ad`` is
    deliberately omitted: it combines the original Training, Validation, and
    Test files.  This prevents accidental split leakage during architecture
    development.
    """

    root = Path(data_root) / "resource"
    specs = [
        FocusDatasetSpec(
            name="arc_h1_train",
            context="H1",
            path=root / "Arc_H1" / "adata_Training.h5ad",
            perturbation_key="target_gene",
            control_labels=("non-targeting",),
            batch_key="batch",
            gene_symbol_key="_index",
            qc_keys=(),
            notes="Raw integer CSR counts; 18,080 measured genes.",
        ),
        FocusDatasetSpec(
            name="replogle_k562_essential",
            context="K562",
            path=root / "Replogle_2022" / "K562_essential_raw_singlecell_01 (1).h5ad",
            perturbation_key="gene",
            control_labels=("non-targeting",),
            batch_key="gem_group",
            gene_symbol_key="gene_name",
            qc_keys=("UMI_count", "mitopercent"),
            notes="Legacy h5ad with dense raw integer X; 8,563 measured genes.",
        ),
        FocusDatasetSpec(
            name="replogle_k562_gwps",
            context="K562",
            path=root / "Replogle_2022" / "K562_gwps_raw_singlecell_01 (1).h5ad",
            perturbation_key="gene",
            control_labels=("non-targeting",),
            batch_key="gem_group",
            gene_symbol_key="gene_name",
            qc_keys=("UMI_count", "mitopercent"),
            notes="Legacy h5ad with dense raw integer X; 8,248 measured genes.",
        ),
        FocusDatasetSpec(
            name="xatlas_hct116",
            context="HCT116",
            path=root / "XAtlas" / "HCT116_filtered_dual_guide_cells.h5ad",
            perturbation_key="gene_target",
            control_labels=("Non-Targeting",),
            batch_key="sample",
            gene_symbol_key="_index",
            qc_keys=(
                "pass_guide_filter",
                "n_genes_by_counts",
                "total_counts",
                "pct_counts_mt",
                "num_features",
            ),
            notes=(
                "Raw integer CSR counts; 38,606 measured genes. CTRL is kept separate from "
                "Non-Targeting because its biological meaning must be confirmed."
            ),
        ),
    ]
    if include_h1_validation:
        specs.insert(
            1,
            FocusDatasetSpec(
                name="arc_h1_validation",
                context="H1",
                path=root / "Arc_H1" / "adata_Validation.h5ad",
                perturbation_key="target_gene",
                control_labels=("non-targeting",),
                batch_key="batch",
                gene_symbol_key="_index",
                role="validation",
                qc_keys=(),
                notes="Held-out original H1 validation split; do not merge into train by default.",
            ),
        )
    return specs


def _matrix_shape(node: h5py.Group | h5py.Dataset) -> tuple[int, int]:
    if isinstance(node, h5py.Dataset):
        return tuple(int(x) for x in node.shape)  # type: ignore[return-value]
    if "shape" not in node.attrs:
        raise ValueError("Sparse X group has no shape attribute")
    return tuple(int(x) for x in node.attrs["shape"])  # type: ignore[return-value]


def inspect_focus_dataset(spec: FocusDatasetSpec) -> dict:
    """Read only small HDF5 metadata; never materialize a count matrix."""

    if not spec.path.is_file():
        raise FileNotFoundError(spec.path)
    with h5py.File(spec.path, "r") as handle:
        if "X" not in handle or "obs" not in handle or "var" not in handle:
            raise ValueError(f"{spec.path} is missing X, obs, or var")
        obs = handle["obs"]
        missing_obs = [key for key in (spec.perturbation_key, spec.batch_key, *spec.qc_keys) if key not in obs]
        if missing_obs:
            raise ValueError(f"{spec.name} is missing obs keys: {missing_obs}")
        if spec.gene_symbol_key not in handle["var"]:
            raise ValueError(f"{spec.name} is missing var/{spec.gene_symbol_key}")
        x = handle["X"]
        encoding = x.attrs.get("encoding-type", "dense")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        return {
            **spec.to_dict(),
            "shape": _matrix_shape(x),
            "x_encoding": str(encoding),
            "x_dtype": str(x["data"].dtype if isinstance(x, h5py.Group) else x.dtype),
            "obsm_keys": sorted(handle.get("obsm", {}).keys()),
            "ready_for_state": "X_state" in handle.get("obsm", {}),
        }


def inspect_focus_datasets(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    *,
    include_h1_validation: bool = False,
) -> list[dict]:
    return [
        inspect_focus_dataset(spec)
        for spec in default_focus_dataset_specs(data_root, include_h1_validation=include_h1_validation)
    ]


def standardize_focus_metadata(
    adata: ad.AnnData,
    spec: FocusDatasetSpec,
    *,
    copy: bool = True,
) -> ad.AnnData:
    """Map dataset-specific metadata onto the shared VCC training schema.

    This function changes metadata only; it does not normalize counts, run QC,
    compute SE embeddings, or write data.  Dataset prefixes on batch values
    prevent unrelated H1/K562/HCT116 batches with the same label from being
    treated as one batch.
    """

    result = adata.copy() if copy else adata
    missing = [key for key in (spec.perturbation_key, spec.batch_key) if key not in result.obs]
    if missing:
        raise KeyError(f"{spec.name} is missing obs columns: {missing}")

    perturbations = result.obs[spec.perturbation_key].astype(str)
    for control in spec.control_labels:
        perturbations = perturbations.mask(perturbations == control, "non-targeting")
    result.obs["target_gene"] = perturbations
    result.obs["batch"] = spec.name + "::" + result.obs[spec.batch_key].astype(str)
    result.obs["cell_type"] = spec.context
    result.obs["dataset"] = spec.name

    if spec.gene_symbol_key == "_index":
        gene_names = result.var_names.astype(str)
    elif spec.gene_symbol_key in result.var:
        gene_names = result.var[spec.gene_symbol_key].astype(str)
    else:
        raise KeyError(f"{spec.name} is missing var column: {spec.gene_symbol_key}")
    result.var["gene_name"] = [canonical_gene_name(name) for name in gene_names]
    return result
