import torch
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tomllib
from cell_load.data_modules import PerturbationDataModule
from scipy import sparse
from torch import nn

from state.tx.models.decoders import PanelFreeGeneDecoder
from state.tx.models.base import PerturbationModel
from state.tx.models.state_transition import StateTransitionPerturbationModel
from state.tx.vcc import (
    HeterogeneousGeneCollator,
    PanelFreePerturbationDataModule,
    build_gene_query_features,
    default_focus_dataset_specs,
    inspect_focus_datasets,
    load_gene_name_list,
    log_cp10k_to_counts,
    build_vcc_control_pool,
    control_log_cp10k_baseline,
    predict_vcc_counts,
    predict_vcc_pooled_counts,
    VCCPredictionWriter,
    standardize_focus_metadata,
)
from state.tx.vcc.sampler import BalancedPerturbationBatchSampler


@pytest.mark.parametrize("use_write_only_handle", [False, True])
def test_panel_free_data_module_save_state_supports_path_and_cli_handle(
    tmp_path, monkeypatch, use_write_only_handle
):
    module = PanelFreePerturbationDataModule.__new__(PanelFreePerturbationDataModule)
    module.max_decoder_genes = 1024
    module.decoder_target_sum = 10_000.0
    module.decoder_generator = torch.Generator().manual_seed(42)
    module.decoder_always_include = {"TP53"}
    module.decoder_fallback_gene_names_file = "fallback.txt"
    module.decoder_fallback_gene_to_id = {"MISSING": 0}
    module.trainable_perturbation_names_file = "perturbations.txt"
    module.trainable_perturbation_to_id = {"TP53": 0}
    module.balance_datasets = True
    module.balance_perturbations = True
    module.sets_per_dataset_per_epoch = 2048
    module.validation_sets_per_dataset = 256
    module.control_only_sets_per_epoch = 64

    def save_base_state(_self, destination):
        torch.save({"base_setting": "preserved"}, destination)

    monkeypatch.setattr(PerturbationDataModule, "save_state", save_base_state)
    output = tmp_path / "data_module.torch"
    if use_write_only_handle:
        with output.open("wb") as handle:
            module.save_state(handle)
    else:
        module.save_state(output)

    state = torch.load(output, weights_only=False)
    assert state["base_setting"] == "preserved"
    assert state["max_decoder_genes"] == 1024
    assert state["decoder_always_include"] == ["TP53"]
    assert state["decoder_fallback_gene_names"] == ["MISSING"]
    assert state["trainable_perturbation_names"] == ["TP53"]


def test_panel_free_decoder_accepts_arbitrary_panels_and_chunks():
    torch.manual_seed(1)
    decoder = PanelFreeGeneDecoder(
        latent_dim=5,
        gene_embedding_dim=7,
        hidden_dim=11,
        n_layers=2,
        dropout=0.0,
    )
    latent = torch.randn(2, 3, 5, requires_grad=True)
    genes = torch.randn(2, 3, 13, 7)

    full = decoder(latent, genes)
    chunked = decoder(latent, genes, chunk_size=4)

    assert full.shape == (2, 3, 13)
    torch.testing.assert_close(full, chunked)
    full.sum().backward()
    assert latent.grad is not None
    assert all(size != 13 for parameter in decoder.parameters() for size in parameter.shape)


def test_panel_free_decoder_uses_stable_trainable_fallback_ids():
    torch.manual_seed(7)
    decoder = PanelFreeGeneDecoder(
        latent_dim=5,
        gene_embedding_dim=7,
        hidden_dim=11,
        n_layers=2,
        dropout=0.0,
        num_fallback_genes=3,
    )
    latent = torch.randn(2, 3, 5)
    genes = torch.randn(3, 7)
    fallback_ids = torch.tensor([-1, 1, -1])

    expected = decoder(latent, genes, fallback_ids=fallback_ids, chunk_size=2)
    changed_placeholder = genes.clone()
    changed_placeholder[1] = 1000
    actual = decoder(latent, changed_placeholder, fallback_ids=fallback_ids, chunk_size=2)
    # The active fallback row replaces, rather than adds to, the placeholder protein vector.
    torch.testing.assert_close(expected, actual)

    expected[..., 1].sum().backward()
    grad = decoder.fallback_residual.weight.grad
    assert grad is not None
    assert grad[1].abs().sum() > 0
    assert grad[[0, 2]].abs().sum() == 0


def test_concat_residual_decoder_uses_control_gene_baseline_and_backpropagates():
    torch.manual_seed(11)
    decoder = PanelFreeGeneDecoder(
        latent_dim=5,
        gene_embedding_dim=7,
        hidden_dim=11,
        n_layers=2,
        dropout=0.0,
        fusion_mode="concat",
        use_gene_baseline=True,
        predict_residual=True,
        output_activation="identity",
    )
    latent = torch.randn(2, 3, 5, requires_grad=True)
    genes = torch.randn(2, 6, 7)
    baseline = torch.rand(2, 6)

    full = decoder(latent, genes, gene_baseline=baseline)
    chunked = decoder(latent, genes, gene_baseline=baseline, chunk_size=2)
    assert full.shape == (2, 3, 6)
    torch.testing.assert_close(full, chunked)
    full.sum().backward()
    assert latent.grad is not None and latent.grad.abs().sum() > 0

    last_linear = decoder.shared_head[-1]
    nn.init.zeros_(last_linear.weight)
    nn.init.zeros_(last_linear.bias)
    expected = baseline[:, None].expand(-1, latent.shape[1], -1)
    torch.testing.assert_close(decoder(latent, genes, gene_baseline=baseline), expected)


def test_gene_query_features_reject_non_vcc_missing_genes():
    proteins = {"A": torch.tensor([1.0, 2.0]), "B": torch.tensor([3.0, 4.0])}
    embeddings, fallback_ids = build_gene_query_features(["A", "VCC_MISSING"], proteins, ["VCC_MISSING"])
    assert embeddings.tolist() == [[1.0, 2.0], [0.0, 0.0]]
    assert fallback_ids.tolist() == [-1, 0]

    with np.testing.assert_raises(KeyError):
        build_gene_query_features(["OTHER_MISSING"], proteins, ["VCC_MISSING"])


def test_vcc_fallback_asset_has_343_stable_unique_genes():
    names = load_gene_name_list("assets/vcc_2026_se_fallback_genes.txt")
    assert len(names) == 343
    assert names == sorted(names)
    assert "AARS" in names


def test_trainable_perturbation_registry_is_stable_and_covers_vcc():
    names = load_gene_name_list("assets/vcc_trainable_perturbations.txt")
    vcc = pd.read_csv("data/vcc_2026_controls/pert_counts.csv")["target_gene"].str.upper().tolist()
    assert len(names) == 1484
    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert set(vcc).issubset(names)


def test_heterogeneous_collator_masks_padding_instead_of_treating_it_as_zero():
    lookup = {
        "TP53": torch.tensor([1.0, 0.0]),
        "BRCA1": torch.tensor([0.0, 1.0]),
        "ENSG000001.9": torch.tensor([0.5, 0.5]),
    }
    collate = HeterogeneousGeneCollator(lookup)
    batch = collate(
        [
            {"gene_names": ["TP53", "BRCA1"], "gene_targets": [0.0, 2.0]},
            {"gene_names": ["ENSG000001.2"], "gene_targets": [3.0]},
        ]
    )

    assert batch["gene_embeddings"].shape == (2, 2, 2)
    assert batch["gene_mask"].tolist() == [[True, True], [True, False]]
    # A measured zero remains supervised, while the padded zero is masked.
    assert batch["gene_targets"][0, 0] == 0
    assert batch["gene_targets"][1, 1] == 0


def test_log_cp10k_conversion_produces_exact_integer_library_sizes():
    expression = torch.log1p(torch.tensor([[2.0, 3.0, 5.0], [1.0, 1.0, 8.0]]) * 1000)
    sizes = torch.tensor([20, 31])
    counts = log_cp10k_to_counts(expression, sizes, generator=torch.Generator().manual_seed(4))

    assert counts.dtype == torch.int64
    assert torch.all(counts >= 0)
    assert counts.sum(-1).tolist() == [20, 31]


class _DummyTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.gene_decoder = PanelFreeGeneDecoder(4, 3, hidden_dim=6, dropout=0.0)
        self.seen_chunk_sizes = []

    def forward(self, batch, padded=True):
        assert not padded
        self.seen_chunk_sizes.append(batch["ctrl_cell_emb"].shape[0])
        return (batch["ctrl_cell_emb"] + batch["pert_emb"]).unsqueeze(0)


def test_vcc_inference_chunks_cells_and_genes_and_returns_counts():
    model = _DummyTransition()
    controls = torch.randn(10, 4)
    perturbation = torch.randn(4)
    genes = torch.randn(17, 3)
    counts = predict_vcc_counts(
        model,
        controls,
        perturbation,
        genes,
        library_sizes=23,
        cell_chunk_size=4,
        gene_chunk_size=5,
        generator=torch.Generator().manual_seed(2),
    )

    assert counts.shape == (10, 17)
    assert counts.sum(-1).tolist() == [23] * 10
    assert model.seen_chunk_sizes == [4, 4, 2]


def test_vcc_control_pool_matches_depth_sorted_four_donor_contract():
    depths = np.arange(1, 21, dtype=np.float32)
    raw = sparse.csr_matrix(np.stack([depths, np.zeros_like(depths)], axis=1))
    pool = build_vcc_control_pool(raw, cells=3, pool_k=4, seed=17, expected_controls=None)
    assert pool.donor_indices.shape == (3, 4)
    assert len(np.unique(pool.donor_indices)) == 12
    selected_depths = depths[pool.donor_indices.reshape(-1)]
    assert np.all(selected_depths[:-1] <= selected_depths[1:])
    np.testing.assert_array_equal(pool.library_sizes, np.rint(selected_depths.reshape(3, 4).mean(1)))


def test_control_baseline_is_mean_per_cell_log_cp10k():
    raw = sparse.csr_matrix([[1, 3], [3, 1]], dtype=np.float32)
    actual = control_log_cp10k_baseline(raw)
    expected = np.log1p(np.asarray([[2500, 7500], [7500, 2500]], dtype=np.float32)).mean(0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_pooled_inference_predicts_donors_then_returns_exact_group_depths():
    model = _DummyTransition()
    controls = torch.randn(12, 4)
    genes = torch.randn(7, 3)
    pool_indices = torch.arange(12).reshape(3, 4)
    depths = torch.tensor([21, 32, 43])
    counts = predict_vcc_pooled_counts(
        model,
        controls,
        torch.randn(4),
        genes,
        pool_indices,
        depths,
        cell_chunk_size=5,
        gene_chunk_size=3,
        generator=torch.Generator().manual_seed(4),
    )
    assert counts.shape == (3, 7)
    assert counts.sum(1).tolist() == depths.tolist()
    assert model.seen_chunk_sizes == [5, 5, 2]


def test_vcc_prediction_writer_emits_official_axis_order(tmp_path):
    output = tmp_path / "prediction.h5ad"
    targets = ["A", "B"]
    genes = ["G1", "G2", "G3"]
    with VCCPredictionWriter(output, targets, genes, contexts=("A", "B"), cells_per_target=2) as writer:
        for value in range(1, 5):
            writer.append(np.asarray([[value, 0, 1], [0, value, 1]], dtype=np.int64))
    result = ad.read_h5ad(output)
    assert result.shape == (8, 3)
    assert result.var_names.tolist() == genes
    assert result.obs["context"].tolist() == ["A"] * 4 + ["B"] * 4
    assert result.obs["target_gene"].tolist() == ["A", "A", "B", "B"] * 2
    assert sparse.isspmatrix_csr(result.X)
    assert np.all(result.X.data == np.floor(result.X.data))


def test_local_focus_registry_is_explicit_and_metadata_is_readable():
    specs = default_focus_dataset_specs()
    assert [spec.context for spec in specs] == ["H1", "K562", "K562", "HCT116"]
    assert specs[0].perturbation_key == "target_gene"
    assert specs[-1].control_labels == ("Non-Targeting",)

    manifests = inspect_focus_datasets()
    shapes = {item["name"]: tuple(item["shape"]) for item in manifests}
    assert shapes["arc_h1_train"] == (221273, 18080)
    assert shapes["replogle_k562_essential"] == (310385, 8563)
    assert shapes["replogle_k562_gwps"] == (1989578, 8248)
    assert shapes["xatlas_hct116"] == (3409169, 38606)
    assert not any(item["ready_for_state"] for item in manifests)


def test_vcc_training_manifests_define_loco_and_all_data_contracts():
    with open("configs/vcc/vcc_h1_loco.toml", "rb") as handle:
        loco = tomllib.load(handle)
    with open("configs/vcc/vcc_all_train.toml", "rb") as handle:
        full = tomllib.load(handle)
    expected = {"arc_h1_train", "replogle_k562_gwps", "xatlas_hct116", "xatlas_hek293t"}
    assert set(loco["datasets"]) == expected
    assert set(full["datasets"]) == expected
    assert loco["zeroshot"] == {"arc_h1_train.H1": "val"}
    assert full["zeroshot"] == {}
    assert set(loco["training"]) == expected
    assert set(full["training"]) == expected


def test_balanced_sampler_caps_a_control_only_loco_dataset():
    sampler = BalancedPerturbationBatchSampler.__new__(BalancedPerturbationBatchSampler)
    sampler.test = False
    sampler.balance_datasets = True
    sampler.balance_perturbations = True
    sampler.sets_per_dataset_per_epoch = 10
    sampler.control_perturbation = "non-targeting"
    sampler.control_only_sets_per_epoch = 2
    sampler._natural_sentences = [[index] for index in range(6)]
    identities = {
        0: ("held_out", "non-targeting"),
        1: ("held_out", "non-targeting"),
        2: ("train", "TP53"),
        3: ("train", "TP53"),
        4: ("train", "BRCA1"),
        5: ("train", "BRCA1"),
    }
    sampler._sentence_identity = lambda sentence: identities[sentence[0]]

    balanced = sampler._balanced_sentences(seed=7)
    held_out = [sentence for sentence in balanced if identities[sentence[0]][0] == "held_out"]
    train = [sentence for sentence in balanced if identities[sentence[0]][0] == "train"]
    assert len(held_out) == 2
    assert len(train) == 10


def test_focus_metadata_adapter_unifies_context_and_control_fields():
    spec = default_focus_dataset_specs()[-1]
    adata = ad.AnnData(
        X=np.ones((2, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"gene_target": ["Non-Targeting", "TP53"], "sample": ["Batch1", "Batch1"]},
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(index=["GeneA", "ENSG000001.7"]),
    )
    standardized = standardize_focus_metadata(adata, spec)

    assert standardized.obs["target_gene"].tolist() == ["non-targeting", "TP53"]
    assert standardized.obs["cell_type"].tolist() == ["HCT116", "HCT116"]
    assert standardized.obs["batch"].tolist() == ["xatlas_hct116::Batch1"] * 2
    assert standardized.var["gene_name"].tolist() == ["GENEA", "ENSG000001"]


def test_panel_free_data_module_collate_uses_native_full_library_size():
    module = PanelFreePerturbationDataModule.__new__(PanelFreePerturbationDataModule)
    module.max_decoder_genes = 2
    module.decoder_target_sum = 10_000.0
    module.decoder_generator = torch.Generator().manual_seed(3)
    module.decoder_always_include = {"A"}
    module.is_log1p = False
    module.cell_sentence_len = 2
    module._panel_names = {"k562": ["A", "B", "C"]}
    module.pert_onehot_map = {
        "A": torch.tensor([1.0, 0.0]),
        "B": torch.tensor([0.0, 1.0]),
        "C": torch.tensor([1.0, 1.0]),
    }

    def sample(counts):
        return {
            "pert_cell_emb": torch.ones(4),
            "ctrl_cell_emb": torch.zeros(4),
            "pert_emb": torch.ones(2),
            "pert_name": "TP53",
            "dataset_name": "k562",
            "batch_name": "b1",
            "batch": torch.ones(1),
            "cell_type": "K562",
            "cell_type_onehot": torch.ones(1),
            "pert_cell_counts": torch.tensor(counts, dtype=torch.float32),
        }

    batch = module._panel_free_collate([sample([1, 1, 8]), sample([2, 2, 16])])
    assert batch["gene_targets"].shape == (1, 2, 2)
    assert batch["gene_embeddings"].shape == (1, 2, 2)
    assert batch["gene_names"][0][0] == "A"
    # Gene A has CP10K=1000 for both cells because normalization used all 3 genes.
    expected = torch.full((2,), float(torch.log1p(torch.tensor(1000.0))))
    torch.testing.assert_close(batch["gene_targets"][0, :, 0], expected)


def test_panel_free_collate_emits_stable_trainable_perturbation_ids():
    module = PanelFreePerturbationDataModule.__new__(PanelFreePerturbationDataModule)
    module.max_decoder_genes = 1
    module.decoder_target_sum = 10_000.0
    module.decoder_generator = torch.Generator().manual_seed(3)
    module.decoder_always_include = set()
    module.decoder_fallback_gene_to_id = {}
    module.trainable_perturbation_to_id = {"TP53": 7}
    module.trainable_perturbation_names_file = "registry.txt"
    module.control_pert = "non-targeting"
    module.is_log1p = False
    module.cell_sentence_len = 2
    module._panel_names = {"k562": ["TP53"]}
    module.pert_onehot_map = {"TP53": torch.ones(2)}

    def sample():
        return {
            "pert_cell_emb": torch.ones(4), "ctrl_cell_emb": torch.zeros(4),
            "pert_emb": torch.ones(2), "pert_name": "TP53", "dataset_name": "k562",
            "batch_name": "b1", "batch": torch.ones(1), "cell_type": "K562",
            "cell_type_onehot": torch.ones(1), "pert_cell_counts": torch.ones(1),
        }

    batch = module._panel_free_collate([sample(), sample()])
    assert batch["perturbation_ids"].tolist() == [7, 7]


def test_panel_free_collate_builds_matched_control_cp10k_baseline():
    module = PanelFreePerturbationDataModule.__new__(PanelFreePerturbationDataModule)
    module.max_decoder_genes = 2
    module.decoder_target_sum = 10_000.0
    module.decoder_generator = torch.Generator().manual_seed(3)
    module.decoder_always_include = set()
    module.decoder_fallback_gene_to_id = {}
    module.decoder_control_residual = True
    module.is_log1p = False
    module.cell_sentence_len = 2
    module._panel_names = {"k562": ["A", "B"]}
    module.pert_onehot_map = {"A": torch.tensor([1.0]), "B": torch.tensor([2.0])}

    def sample(pert, ctrl):
        return {
            "pert_cell_emb": torch.ones(4), "ctrl_cell_emb": torch.zeros(4),
            "pert_emb": torch.ones(2), "pert_name": "TP53", "dataset_name": "k562",
            "batch_name": "b1", "batch": torch.ones(1), "cell_type": "K562",
            "cell_type_onehot": torch.ones(1),
            "pert_cell_counts": torch.tensor(pert, dtype=torch.float32),
            "ctrl_cell_counts": torch.tensor(ctrl, dtype=torch.float32),
        }

    batch = module._panel_free_collate([
        sample([1, 9], [2, 8]), sample([3, 7], [4, 6])
    ])
    expected = torch.stack([
        torch.log1p(torch.tensor([2000.0, 8000.0])),
        torch.log1p(torch.tensor([4000.0, 6000.0])),
    ]).mean(0)
    torch.testing.assert_close(batch["gene_baselines"][0], expected)


def test_panel_free_data_module_collates_mixed_dataset_sets():
    module = PanelFreePerturbationDataModule.__new__(PanelFreePerturbationDataModule)
    module.max_decoder_genes = 3
    module.decoder_target_sum = 10_000.0
    module.decoder_generator = torch.Generator().manual_seed(3)
    module.decoder_always_include = set()
    module.is_log1p = False
    module.cell_sentence_len = 2
    module._panel_names = {"h1": ["A", "B"], "k562": ["A", "B", "C"]}
    module.pert_onehot_map = {
        "A": torch.tensor([1.0, 0.0]),
        "B": torch.tensor([0.0, 1.0]),
        "C": torch.tensor([1.0, 1.0]),
    }

    def sample(dataset, counts):
        return {
            "pert_cell_emb": torch.ones(4),
            "ctrl_cell_emb": torch.zeros(4),
            "pert_emb": torch.ones(2),
            "pert_name": "TP53",
            "dataset_name": dataset,
            "batch_name": "b1",
            "batch": torch.ones(1),
            "cell_type": dataset,
            "cell_type_onehot": torch.ones(1),
            "pert_cell_counts": torch.tensor(counts, dtype=torch.float32),
        }

    batch = module._panel_free_collate(
        [
            sample("h1", [1, 9]),
            sample("h1", [2, 8]),
            sample("k562", [1, 2, 7]),
            sample("k562", [2, 3, 5]),
        ]
    )
    assert batch["ctrl_cell_emb"].shape == (4, 4)
    assert batch["gene_targets"].shape == (2, 2, 3)
    assert batch["gene_embeddings"].shape == (2, 3, 2)
    assert batch["gene_mask"][0, :, 2].tolist() == [False, False]
    assert batch["gene_mask"][1].all()


def test_panel_free_data_module_includes_only_configured_missing_genes():
    module = PanelFreePerturbationDataModule.__new__(PanelFreePerturbationDataModule)
    module.max_decoder_genes = 3
    module.decoder_target_sum = 10_000.0
    module.decoder_generator = torch.Generator().manual_seed(3)
    module.decoder_always_include = set()
    module.decoder_fallback_gene_to_id = {"VCC_MISSING": 0}
    module.is_log1p = False
    module.cell_sentence_len = 2
    module._panel_names = {"h1": ["A", "VCC_MISSING", "OTHER_MISSING"]}
    module.pert_onehot_map = {"A": torch.tensor([1.0, 2.0])}

    def sample(counts):
        return {
            "pert_cell_emb": torch.ones(4),
            "ctrl_cell_emb": torch.zeros(4),
            "pert_emb": torch.ones(2),
            "pert_name": "TP53",
            "dataset_name": "h1",
            "batch_name": "b1",
            "batch": torch.ones(1),
            "cell_type": "H1",
            "cell_type_onehot": torch.ones(1),
            "pert_cell_counts": torch.tensor(counts, dtype=torch.float32),
        }

    batch = module._panel_free_collate([sample([1, 2, 7]), sample([2, 3, 5])])
    assert batch["gene_names"] == [["A", "VCC_MISSING"]]
    assert batch["gene_fallback_ids"].tolist() == [[-1, 0]]
    assert batch["gene_embeddings"][0, 1].abs().sum() == 0
    # OTHER_MISSING remains excluded even though it is measured in this dataset.
    assert batch["gene_targets"].shape == (1, 2, 2)


def test_panel_free_data_module_canonicalizes_semantic_lookup_keys():
    mixed_case = {"C1orf112": torch.tensor([1.0, 2.0]), "C1ORF112": torch.zeros(2)}
    canonical = PanelFreePerturbationDataModule._canonicalize_feature_map(mixed_case)
    assert canonical["C1ORF112"].tolist() == [1.0, 2.0]


def test_decoder_gene_sampling_always_includes_set_perturbation_target():
    module = PanelFreePerturbationDataModule.__new__(PanelFreePerturbationDataModule)
    module.max_decoder_genes = 2
    module.decoder_generator = torch.Generator().manual_seed(9)
    module.decoder_always_include = set()
    module.decoder_fallback_gene_to_id = {}
    module.pert_onehot_map = {
        "A": torch.ones(2),
        "B": torch.ones(2),
        "TP53": torch.ones(2),
        "D": torch.ones(2),
    }
    names = ["A", "B", "TP53", "D"]
    selected = module._select_gene_indices(names, required_names={"TP53"})
    assert len(selected) == 2
    assert names[selected[0]] == "TP53"


def test_decoder_gene_sampling_reserves_de_ranked_fraction():
    module = PanelFreePerturbationDataModule.__new__(PanelFreePerturbationDataModule)
    module.max_decoder_genes = 10
    module.decoder_deg_fraction = 0.6
    module.decoder_generator = torch.Generator().manual_seed(9)
    module.decoder_always_include = set()
    module.decoder_fallback_gene_to_id = {}
    names = [f"G{i}" for i in range(20)] + ["TP53"]
    module.pert_onehot_map = {name: torch.ones(2) for name in names}
    # TP53 is deliberately not one of the top expression-shift genes, so the
    # result must contain 1 required target + 6 DE-ranked + 3 random genes.
    scores = torch.arange(len(names), dtype=torch.float32)
    scores[-1] = -1
    selected = module._select_gene_indices(
        names,
        required_names={"TP53"},
        differential_scores=scores,
    )
    assert len(selected) == 10
    assert names[selected[0]] == "TP53"
    assert {names[index] for index in selected[1:7]} == {
        "G14", "G15", "G16", "G17", "G18", "G19"
    }


def test_transfer_optimizer_uses_separate_learning_rate_groups():
    model = nn.Module()
    model.backbone = nn.Linear(3, 3)
    model.pert_encoder = nn.Sequential(nn.Linear(3, 3))
    model.perturbation_residual = nn.Embedding(5, 3)
    model.gene_decoder = PanelFreeGeneDecoder(3, 3, hidden_dim=4, num_fallback_genes=2)
    model.optimizer_group_lrs = {
        "backbone": 1e-6,
        "perturbation_encoder": 2e-5,
        "decoder": 3e-4,
        "fallback": 4e-4,
    }
    model.optimizer_weight_decay = 5e-4
    model.lr = 1e-4

    optimizer = PerturbationModel.configure_optimizers(model)
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert set(groups) == {"backbone", "perturbation_encoder", "decoder", "fallback"}
    assert groups["backbone"]["lr"] == 1e-6
    assert groups["fallback"]["lr"] == 4e-4
    assert sum(parameter.numel() for parameter in groups["fallback"]["params"]) == 12
    assert sum(parameter.numel() for parameter in groups["perturbation_encoder"]["params"]) == 27


def test_trainable_perturbation_residual_is_target_specific_and_control_is_zero():
    model = StateTransitionPerturbationModel.__new__(StateTransitionPerturbationModel)
    nn.Module.__init__(model)
    model.pert_encoder = nn.Identity()
    model.perturbation_residual = nn.Embedding(3, 4)
    nn.init.zeros_(model.perturbation_residual.weight)
    with torch.no_grad():
        model.perturbation_residual.weight[1] = 2.0

    semantic = torch.ones(1, 3, 4)
    result = model.encode_perturbation(semantic, torch.tensor([[-1, 1, 2]]))
    torch.testing.assert_close(result[0, 0], torch.ones(4))
    torch.testing.assert_close(result[0, 1], torch.full((4,), 3.0))
    torch.testing.assert_close(result[0, 2], torch.ones(4))
    result.sum().backward()
    gradient = model.perturbation_residual.weight.grad
    assert gradient is not None
    assert gradient[0].abs().sum() == 0
    assert gradient[1].abs().sum() > 0
    assert gradient[2].abs().sum() > 0


def test_state_panel_free_all_space_keeps_latent_residual_path():
    # Avoid constructing a transformer: exercise the forward branch with tiny
    # identity components and a minimal object initialized through nn.Module.
    model = StateTransitionPerturbationModel.__new__(StateTransitionPerturbationModel)
    nn.Module.__init__(model)
    model.cell_sentence_len = 2
    model.pert_dim = 2
    model.input_dim = 2
    model.output_dim = 2
    model.hidden_dim = 2
    model.predict_residual = True
    model.output_space = "all"
    model.gene_decoder = PanelFreeGeneDecoder(2, 3, hidden_dim=4)
    model.pert_encoder = nn.Identity()
    model.basal_encoder = nn.Identity()
    model.transformer_backbone = _IdentityBackbone()
    model.project_out = nn.Identity()
    model.final_down_then_up = nn.Sequential(nn.Linear(2, 2, bias=False))
    nn.init.zeros_(model.final_down_then_up[0].weight)
    model.batch_encoder = None
    model.use_batch_token = False
    model.batch_token = None
    model.confidence_token = None
    model._batch_token_cache = None
    model.hparams["mask_attn"] = False
    model.hparams["embed_key"] = "X_state"

    basal = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    output = model({"ctrl_cell_emb": basal, "pert_emb": torch.zeros_like(basal)})
    # The legacy all-space projection is zeroed; a non-zero result proves the
    # panel-free path preserved the normal latent residual calculation.
    torch.testing.assert_close(output, 2 * basal)


def test_state_predict_step_passes_panel_fallback_ids():
    model = StateTransitionPerturbationModel.__new__(StateTransitionPerturbationModel)
    nn.Module.__init__(model)
    model.cell_sentence_len = 2
    model.output_dim = 4
    model.confidence_token = None
    model.log1p_from_raw_counts = False
    model.gene_decoder = PanelFreeGeneDecoder(4, 3, hidden_dim=6, dropout=0.0, num_fallback_genes=1)
    model.hparams["gene_decoder_chunk_size"] = 2
    model.forward = lambda batch, padded=True: batch["ctrl_cell_emb"]

    batch = {
        "ctrl_cell_emb": torch.randn(2, 4),
        "pert_cell_emb": torch.randn(2, 4),
        "gene_embeddings": torch.randn(1, 3, 3),
        "gene_fallback_ids": torch.tensor([[-1, 0, -1]]),
        "gene_mask": torch.ones(1, 2, 3, dtype=torch.bool),
        "gene_names": [["A", "VCC_MISSING", "B"]],
    }
    result = model.predict_step(batch, batch_idx=0)
    assert result["pert_cell_counts_preds"].shape == (1, 2, 3)
    assert result["gene_names"] == batch["gene_names"]


class _IdentityBackbone(nn.Module):
    def forward(self, inputs_embeds, **kwargs):
        return type("Output", (), {"last_hidden_state": inputs_embeds})()
