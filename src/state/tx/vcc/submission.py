"""VCC control pooling, baseline construction, and streaming H5AD output.

The control-selection contract follows the local
``Virtual-Cell-Challenge-2026/predict.py`` reference: sample 400 * 4 donors
without replacement, stable-sort them by raw library depth, and pool every
four adjacent donors. The model-specific inference path predicts the four
donors separately and pools their gene probabilities afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class VCCControlPool:
    donor_indices: np.ndarray
    library_sizes: np.ndarray

    def __post_init__(self) -> None:
        if self.donor_indices.ndim != 2 or not self.donor_indices.size:
            raise ValueError("donor_indices must have shape [output_cells, pool_k]")
        if self.library_sizes.shape != (self.donor_indices.shape[0],):
            raise ValueError("library_sizes must have one value per output cell")


def _validated_raw_csr(counts) -> sparse.csr_matrix:
    raw = sparse.csr_matrix(counts, dtype=np.float64)
    if not np.isfinite(raw.data).all() or (raw.data < 0).any() or (raw.data != np.floor(raw.data)).any():
        raise ValueError("Controls must contain finite non-negative integer raw counts")
    raw.eliminate_zeros()
    return raw


def build_vcc_control_pool(
    raw_counts,
    *,
    cells: int = 400,
    pool_k: int = 4,
    seed: int,
    expected_controls: int | None = 18_400,
) -> VCCControlPool:
    """Select and depth-sort control donors exactly once for one context."""
    if cells <= 0 or pool_k <= 0:
        raise ValueError("cells and pool_k must be positive")
    raw = _validated_raw_csr(raw_counts)
    if expected_controls is not None and raw.shape[0] != expected_controls:
        raise ValueError(f"Expected {expected_controls} context controls, got {raw.shape[0]}")
    depths = np.asarray(raw.sum(axis=1)).ravel()
    if (depths <= 0).any() or len(depths) < cells * pool_k:
        raise ValueError("Need positive-depth controls and at least cells * pool_k donors")
    selected = np.random.default_rng(seed).choice(len(depths), cells * pool_k, replace=False)
    selected = selected[np.argsort(depths[selected], kind="stable")]
    donor_indices = selected.reshape(cells, pool_k)
    pooled_depths = np.rint(depths[donor_indices].mean(axis=1)).astype(np.int64)
    if (pooled_depths < 1).any() or (pooled_depths > 1_000_000).any():
        raise ValueError("Pooled library sizes must be integers in [1, 1000000]")
    return VCCControlPool(donor_indices=donor_indices, library_sizes=pooled_depths)


def control_log_cp10k_baseline(raw_counts, *, target_sum: float = 10_000.0, block_size: int = 256) -> np.ndarray:
    """Mean per-cell log1p(CP10K), matching the decoder's training baseline."""
    if target_sum <= 0 or block_size <= 0:
        raise ValueError("target_sum and block_size must be positive")
    raw = _validated_raw_csr(raw_counts)
    depths = np.asarray(raw.sum(axis=1)).ravel()
    if (depths <= 0).any():
        raise ValueError("Control cells must have positive library sizes")
    total = np.zeros(raw.shape[1], dtype=np.float64)
    for start in range(0, raw.shape[0], block_size):
        stop = min(start + block_size, raw.shape[0])
        block = raw[start:stop].multiply((target_sum / depths[start:stop])[:, None]).tocsr()
        block.data = np.log1p(block.data)
        total += np.asarray(block.sum(axis=0)).ravel()
    return (total / raw.shape[0]).astype(np.float32)


def control_log_cp10k_rows(
    raw_counts,
    *,
    target_sum: float = 10_000.0,
    gene_indices: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """Per-cell log1p(CP10K) baselines, optionally restricted to genes."""
    if target_sum <= 0:
        raise ValueError("target_sum must be positive")
    raw = _validated_raw_csr(raw_counts)
    depths = np.asarray(raw.sum(axis=1)).ravel()
    if (depths <= 0).any():
        raise ValueError("Control cells must have positive library sizes")
    normalized = raw.multiply((target_sum / depths)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    if gene_indices is not None:
        normalized = normalized[:, np.asarray(gene_indices, dtype=np.int64)]
    return normalized.toarray().astype(np.float32, copy=False)


def control_log_cp10k_read_depth(
    raw_counts,
    *,
    target_sum: float = 10_000.0,
) -> np.ndarray:
    """Return the paper-style mean log1p(CP10K) depth for each control cell.

    The mean is taken over genes with positive raw counts, matching the scalar
    read-depth feature used by the paper's gene-expression decoder.
    """
    if target_sum <= 0:
        raise ValueError("target_sum must be positive")
    raw = _validated_raw_csr(raw_counts)
    depths = np.asarray(raw.sum(axis=1)).ravel()
    if (depths <= 0).any():
        raise ValueError("Control cells must have positive library sizes")
    normalized = raw.multiply((target_sum / depths)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    totals = np.asarray(normalized.sum(axis=1)).ravel()
    expressed = np.diff(normalized.indptr).astype(np.float32)
    return (totals / np.maximum(expressed, 1.0)).astype(np.float32)


class VCCPredictionWriter:
    """Append target-sized count blocks to an official-order sparse H5AD."""

    def __init__(
        self,
        path: str | Path,
        targets: list[str] | np.ndarray,
        genes: list[str] | np.ndarray,
        contexts: list[str] | tuple[str, ...] = ("A", "B", "C"),
        cells_per_target: int = 400,
    ):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.targets = np.asarray(targets, dtype=str)
        self.genes = np.asarray(genes, dtype=str)
        self.contexts = np.asarray(contexts, dtype=str)
        if len(set(self.targets)) != len(self.targets) or len(set(self.genes)) != len(self.genes):
            raise ValueError("Target and gene axes must be unique")
        if cells_per_target <= 0:
            raise ValueError("cells_per_target must be positive")
        self.cells_per_target = int(cells_per_target)
        self.nobs = len(self.targets) * len(self.contexts) * self.cells_per_target
        self.ngenes = len(self.genes)
        self.rows = 0
        self.nnz = 0
        obs = pd.DataFrame(
            {
                "target_gene": np.tile(np.repeat(self.targets, self.cells_per_target), len(self.contexts)),
                "context": np.repeat(self.contexts, len(self.targets) * self.cells_per_target),
            },
            index=[
                f"{context}_{target}_{cell}"
                for context in self.contexts
                for target in self.targets
                for cell in range(self.cells_per_target)
            ],
        )
        ad.AnnData(
            sparse.csr_matrix((self.nobs, self.ngenes), dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=self.genes),
        ).write_h5ad(self.path)
        self.file = h5py.File(self.path, "r+")
        group = self.file["X"]
        for name in ("data", "indices", "indptr"):
            del group[name]
        self.data = group.create_dataset(
            "data", shape=(0,), maxshape=(None,), dtype="float32",
            chunks=(1_048_576,), compression="lzf", shuffle=True,
        )
        self.indices = group.create_dataset(
            "indices", shape=(0,), maxshape=(None,), dtype="int32",
            chunks=(1_048_576,), compression="lzf", shuffle=True,
        )
        self.indptr = group.create_dataset("indptr", shape=(self.nobs + 1,), dtype="int64")
        self.indptr[0] = 0

    def append(self, counts) -> None:
        values = np.asarray(counts)
        if values.shape != (self.cells_per_target, self.ngenes):
            raise ValueError(
                f"Expected count block {(self.cells_per_target, self.ngenes)}, got {values.shape}"
            )
        if self.rows + len(values) > self.nobs:
            raise ValueError("Too many count blocks for declared axes")
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError("Expected finite non-negative integer counts")
        if (values.sum(axis=1, dtype=np.float64) > 1_000_000).any():
            raise ValueError("Cell depth exceeds 1,000,000")
        block = sparse.csr_matrix(values.astype(np.uint32))
        block.eliminate_zeros()
        end = self.nnz + block.nnz
        self.data.resize((end,))
        self.indices.resize((end,))
        self.data[self.nnz:end] = block.data
        self.indices[self.nnz:end] = block.indices
        block_rows = block.shape[0]
        self.indptr[self.rows + 1 : self.rows + block_rows + 1] = block.indptr[1:] + self.nnz
        self.rows += block_rows
        self.nnz = end

    def close(self) -> None:
        complete = self.rows == self.nobs and int(self.indptr[-1]) == self.nnz
        self.file.close()
        if not complete:
            raise ValueError(f"Incomplete prediction: wrote {self.rows}/{self.nobs} rows")

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.file.close()
            return False
        self.close()
        return False
