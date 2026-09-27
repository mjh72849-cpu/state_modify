"""Verify target retention, count caps, QC, and batch-matched controls."""
import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from state.tx.vcc.focus_data import FocusDatasetSpec

module_spec = importlib.util.spec_from_file_location(
    "prepare_vcc_focus_data", Path(__file__).resolve().parents[1] / "scripts/prepare_vcc_focus_data.py"
)
prep = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(prep)


def test_duplicate_gene_symbols_are_summed_without_changing_cell_depth():
    data = ad.AnnData(
        sp.csr_matrix([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        var=pd.DataFrame(
            {"gene_name": ["GeneA", "GENEA", "GeneB"]},
            index=["ENSG1", "ENSG2", "ENSG3"],
        ),
    )
    collapsed = prep._collapse_duplicate_gene_symbols(data)
    assert collapsed.var["gene_name"].tolist() == ["GENEA", "GENEB"]
    np.testing.assert_array_equal(collapsed.X.toarray(), [[3, 3], [9, 6]])
    np.testing.assert_array_equal(
        np.asarray(collapsed.X.sum(axis=1)), np.asarray(data.X.sum(axis=1))
    )
    assert collapsed.var.loc["GENEA", "source_feature_ids"] == "ENSG1;ENSG2"


def test_focused_selection_retains_vcc_and_filters_extras_with_matched_controls(tmp_path):
    targets = ["non-targeting"] * 8 + ["A"] * 7 + ["EXTRA"] * 6 + ["BAD"] * 6 + ["NOCTRL"] * 3
    batches = ["b1"] * 27 + ["b2"] * 3
    qc = np.ones(len(targets), dtype=bool)
    qc[8] = False
    obs = pd.DataFrame({"target": targets, "sample": batches, "pass_guide_filter": qc})
    source = tmp_path / "source.h5ad"
    ad.AnnData(sp.csr_matrix(np.ones((len(obs), 3))), obs=obs).write_h5ad(source)
    spec = FocusDatasetSpec(
        name="fixture", context="HCT116", path=source, perturbation_key="target",
        control_labels=("non-targeting",), batch_key="sample", gene_symbol_key="_index",
        qc_keys=("pass_guide_filter",),
    )
    kwargs = dict(
        vcc_targets=["A", "MISSING", "NOCTRL"], supported_targets={"EXTRA"},
        vcc_max_cells=4, extra_max_cells=3, extra_min_cells=5,
        max_extra_targets=1, controls_per_batch=2, seed=9,
    )
    indices, report, summary = prep._select_focused_indices(spec, **kwargs)
    repeated, _, _ = prep._select_focused_indices(spec, **kwargs)
    np.testing.assert_array_equal(indices, repeated)
    assert 8 not in indices  # Failed guide QC.
    selected = obs.iloc[indices]
    assert selected.target.value_counts().to_dict() == {"A": 4, "EXTRA": 3, "non-targeting": 2}
    assert summary["vcc_targets_present"] == 1
    assert summary["vcc_targets_absent"] == 2
    assert summary["control_batches"] == 1
    assert report.set_index("target_gene").loc["MISSING", "status"] == "absent_in_source"


def test_hek_shard_plan_covers_targets_before_optional_depth_budget():
    frame = pd.DataFrame({
        "batch": ["HEK293T_Batch1", "HEK293T_Batch1", "HEK293T_Batch2"],
        "target_gene": ["A", "B", "C"],
    })
    # Coverage wins over an unrealistically small optional depth budget.
    shards = prep._choose_hek_shards(frame, ["A", "B", "C"], 1)
    assert shards == ["HEK293T_Batch1", "HEK293T_Batch2"]


def test_hek_stream_preserves_token_count_alignment_and_control_label(tmp_path, monkeypatch):
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    genes = pa.table({
        "ensembl_id": ["ENSG2", "ENSG1", "ENSG3"],
        "gene_name": ["B", "A", "C"], "gene_token_id": [20, 10, 30],
    })
    pq.write_table(genes, tmp_path / "gene_metadata.parquet")
    cells = pa.table({
        "cell_barcode": ["ctrl", "pert", "discard"],
        "gene_token_id": [[30, 10], [20, 30], [10]],
        "gene_expression": [[7., 2.], [3., 5.], [1.]],
        "sample": ["HEK293T_Batch1"] * 3,
        "n_genes_by_counts": [2, 2, 1], "total_counts": [9., 8., 1.],
        "total_counts_mt": [0.] * 3, "pct_counts_mt": [0.] * 3,
    })
    pq.write_table(cells, tmp_path / "HEK293T_Batch1.parquet")

    class LocalRemote:
        def open(self, path, mode, **kwargs):
            return (tmp_path / Path(path).name).open(mode)

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", LocalRemote)
    selected = pd.DataFrame({
        "cell_barcode": ["pert", "ctrl"], "target_gene": ["B", "NON-TARGETING"],
        "batch": ["HEK293T_Batch1"] * 2,
    })
    output = tmp_path / "hek.h5ad"
    prep._materialize_hek_stream(selected, ["HEK293T_Batch1"], output, {"selected_cells": 2})
    result = ad.read_h5ad(output)
    assert result.var.gene_name.astype(str).tolist() == ["A", "B", "C"]
    np.testing.assert_array_equal(result.X.toarray(), [[2, 0, 7], [0, 3, 5]])
    assert result.obs.target_gene.astype(str).tolist() == ["non-targeting", "B"]
    assert result.obs_names.tolist() == ["xatlas_hek293t::ctrl", "xatlas_hek293t::pert"]
    assert set(result.obs.batch) == {"xatlas_hek293t::HEK293T_Batch1"}
    prep._validate_hek_cached_selection(output, selected)
    changed = selected.copy()
    changed.loc[0, "target_gene"] = "A"
    with pytest.raises(ValueError, match="selection differs"):
        prep._validate_hek_cached_selection(output, changed)
    cache = next((tmp_path / "stream_cache").glob("*.parquet"))
    assert pq.read_table(cache).num_rows == 2
    (tmp_path / "HEK293T_Batch1.parquet").unlink()
    # Retained-row cache permits a repeat even when the remote shard disappears.
    prep._materialize_hek_stream(selected, ["HEK293T_Batch1"], output, {"selected_cells": 2})
    np.testing.assert_array_equal(ad.read_h5ad(output).X.toarray(), result.X.toarray())
