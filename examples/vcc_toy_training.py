"""Executable toy example for the panel-free VCC/STATE training architecture.

All tensors and weights are synthetic. Run from the repository root with:

    external/state-env/bin/python examples/vcc_toy_training.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from state.tx.models.decoders import PanelFreeGeneDecoder  # noqa: E402


@dataclass
class ToySet:
    dataset: str
    context: str
    perturbation: str
    genes: list[str]
    control_counts: torch.Tensor
    target_counts: torch.Tensor
    latent_weight: float = 1.0


def make_gene_embeddings(names: list[str], embedding_dim: int = 5) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {name: torch.randn(embedding_dim, generator=generator) for name in names}


def make_toy_sets() -> list[ToySet]:
    """Make three sets with different native panels; every set has four cells."""

    h1_control = torch.tensor(
        [[8, 2, 4, 1, 5], [7, 3, 5, 1, 4], [9, 2, 3, 2, 4], [8, 4, 4, 1, 3]], dtype=torch.float32
    )
    h1_tp53 = torch.tensor(
        [[4, 2, 7, 1, 6], [3, 3, 8, 1, 5], [5, 2, 6, 2, 5], [4, 4, 7, 1, 4]], dtype=torch.float32
    )
    k562_control = torch.tensor([[3, 12, 5], [2, 13, 5], [4, 11, 5], [3, 10, 7]], dtype=torch.float32)
    k562_myc = torch.tensor([[2, 18, 4], [1, 19, 5], [3, 17, 4], [2, 16, 7]], dtype=torch.float32)
    vcc_control_input = torch.tensor(
        [[5, 2, 3, 1, 4, 5], [4, 3, 3, 2, 3, 5], [6, 2, 2, 1, 4, 5], [5, 3, 4, 1, 3, 4]],
        dtype=torch.float32,
    )
    # A disjoint synthetic control set is the identity target, not the same cells.
    vcc_control_target = torch.tensor(
        [[5, 3, 2, 1, 4, 5], [4, 2, 4, 2, 3, 5], [6, 1, 3, 1, 4, 5], [5, 2, 4, 1, 4, 4]],
        dtype=torch.float32,
    )
    return [
        ToySet("Arc_H1", "H1", "TP53", ["TP53", "GENEA", "GENEB", "GENEC", "GENED"], h1_control, h1_tp53),
        ToySet("Replogle", "K562", "MYC", ["MYC", "GENEA", "GENEX"], k562_control, k562_myc),
        ToySet(
            "VCC_controls",
            "A",
            "non-targeting",
            ["TP53", "MYC", "GENEA", "GENEB", "GENEX", "GENEY"],
            vcc_control_input,
            vcc_control_target,
            latent_weight=0.25,
        ),
    ]


def simulate_frozen_se(
    raw_counts: torch.Tensor,
    gene_embeddings: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """A small stand-in for frozen SE: variable-panel counts -> fixed state."""

    log_counts = torch.log1p(raw_counts)
    weights = log_counts / log_counts.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    pooled_gene_state = weights @ gene_embeddings
    return pooled_gene_state @ projection


def log_cp10k(raw_counts: torch.Tensor, library_sizes: torch.Tensor | None = None) -> torch.Tensor:
    """Create decoder targets without overwriting the raw count matrix."""

    if library_sizes is None:
        library_sizes = raw_counts.sum(dim=-1, keepdim=True)
    elif library_sizes.dim() == 1:
        library_sizes = library_sizes[:, None]
    scale = torch.where(library_sizes > 0, 10_000.0 / library_sizes, torch.zeros_like(library_sizes))
    return torch.log1p(raw_counts * scale)


def build_toy_batch(
    sets: list[ToySet],
    embedding_lookup: dict[str, torch.Tensor],
    state_dim: int = 8,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Encode each native panel and pad genes only at meta-batch collation."""

    embedding_dim = next(iter(embedding_lookup.values())).numel()
    generator = torch.Generator().manual_seed(11)
    se_projection = torch.randn(embedding_dim, state_dim, generator=generator) / embedding_dim**0.5
    se_projection.requires_grad_(False)

    control_states = []
    target_states = []
    perturbations = []
    decoder_targets = []
    decoder_baselines = []
    decoder_embeddings = []
    for cell_set in sets:
        panel_embeddings = torch.stack([embedding_lookup[gene] for gene in cell_set.genes])
        control_states.append(simulate_frozen_se(cell_set.control_counts, panel_embeddings, se_projection))
        target_states.append(simulate_frozen_se(cell_set.target_counts, panel_embeddings, se_projection))
        perturbation = (
            torch.zeros(embedding_dim)
            if cell_set.perturbation == "non-targeting"
            else embedding_lookup[cell_set.perturbation]
        )
        perturbations.append(perturbation)
        decoder_targets.append(log_cp10k(cell_set.target_counts))
        decoder_baselines.append(log_cp10k(cell_set.control_counts).mean(dim=0))
        decoder_embeddings.append(panel_embeddings)

    batch_size = len(sets)
    set_size = sets[0].control_counts.shape[0]
    max_genes = max(len(cell_set.genes) for cell_set in sets)
    gene_embeddings = torch.zeros(batch_size, max_genes, embedding_dim)
    gene_targets = torch.zeros(batch_size, set_size, max_genes)
    gene_baselines = torch.zeros(batch_size, max_genes)
    gene_mask = torch.zeros(batch_size, set_size, max_genes, dtype=torch.bool)
    for set_index, (targets, embeddings) in enumerate(zip(decoder_targets, decoder_embeddings)):
        width = targets.shape[-1]
        gene_embeddings[set_index, :width] = embeddings
        gene_targets[set_index, :, :width] = targets
        gene_baselines[set_index, :width] = decoder_baselines[set_index]
        gene_mask[set_index, :, :width] = True

    batch = {
        "ctrl_state": torch.stack(control_states),
        "target_state": torch.stack(target_states),
        "pert_embedding": torch.stack(perturbations),
        "gene_embeddings": gene_embeddings,
        "gene_targets": gene_targets,
        "gene_baselines": gene_baselines,
        "gene_mask": gene_mask,
    }
    latent_weights = torch.tensor([cell_set.latent_weight for cell_set in sets])
    return batch, latent_weights


class ToyStateModel(nn.Module):
    """Tiny random-weight analogue of genetic perturbation ST + shared decoder."""

    def __init__(self, state_dim: int = 8, perturbation_dim: int = 5, hidden_dim: int = 12):
        super().__init__()
        self.basal_encoder = nn.Linear(state_dim, hidden_dim)
        self.perturbation_encoder = nn.Linear(perturbation_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=3,
            dim_feedforward=24,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.project_out = nn.Linear(hidden_dim, state_dim)
        self.gene_decoder = PanelFreeGeneDecoder(
            latent_dim=state_dim,
            gene_embedding_dim=perturbation_dim,
            hidden_dim=10,
            n_layers=2,
            dropout=0.0,
            fusion_mode="concat",
            use_gene_baseline=True,
            predict_residual=True,
            output_activation="identity",
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        control_hidden = self.basal_encoder(batch["ctrl_state"])
        perturbation_hidden = self.perturbation_encoder(batch["pert_embedding"]).unsqueeze(1)
        # Shape is [B, S, H]. Transformer attention never crosses B/set boundaries.
        transformed = self.transformer(control_hidden + perturbation_hidden)
        predicted_state = self.project_out(transformed + control_hidden)
        predicted_expression = self.gene_decoder(
            predicted_state,
            batch["gene_embeddings"],
            gene_baseline=batch["gene_baselines"],
            chunk_size=3,
        )
        return predicted_state, predicted_expression


def energy_distance_per_set(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Biased empirical energy distance, independently for every set."""

    cross = torch.cdist(prediction, target).mean(dim=(1, 2))
    within_prediction = torch.cdist(prediction, prediction).mean(dim=(1, 2))
    within_target = torch.cdist(target, target).mean(dim=(1, 2))
    return 2 * cross - within_prediction - within_target


def compute_losses(
    predicted_state: torch.Tensor,
    predicted_expression: torch.Tensor,
    batch: dict[str, torch.Tensor],
    latent_weights: torch.Tensor,
    gene_loss_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    latent_per_set = energy_distance_per_set(predicted_state, batch["target_state"])
    latent_loss = (latent_per_set * latent_weights).sum() / latent_weights.sum()
    squared_error = (predicted_expression - batch["gene_targets"]).square()
    mask = batch["gene_mask"].to(squared_error.dtype)
    gene_per_set = (squared_error * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1)
    gene_loss = gene_per_set.mean()
    total_loss = latent_loss + gene_loss_weight * gene_loss
    return {
        "latent_per_set": latent_per_set,
        "gene_per_set": gene_per_set,
        "latent_loss": latent_loss,
        "gene_loss": gene_loss,
        "total_loss": total_loss,
    }


def gradient_norms(model: nn.Module) -> dict[str, float]:
    groups = {
        "basal_encoder": model.basal_encoder,
        "perturbation_encoder": model.perturbation_encoder,
        "transformer": model.transformer,
        "gene_decoder": model.gene_decoder,
    }
    return {
        name: float(torch.sqrt(sum((parameter.grad.detach() ** 2).sum() for parameter in module.parameters() if parameter.grad is not None)))
        for name, module in groups.items()
    }


def run_demo() -> dict:
    torch.manual_seed(23)
    sets = make_toy_sets()
    all_genes = sorted({gene for cell_set in sets for gene in cell_set.genes})
    embeddings = make_gene_embeddings(all_genes)
    batch, latent_weights = build_toy_batch(sets, embeddings)
    model = ToyStateModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    optimizer.zero_grad(set_to_none=True)
    predicted_state, predicted_expression = model(batch)
    losses = compute_losses(predicted_state, predicted_expression, batch, latent_weights)
    before = model.project_out.weight.detach().clone()
    losses["total_loss"].backward()
    norms = gradient_norms(model)
    optimizer.step()
    update_size = float((model.project_out.weight.detach() - before).norm())
    return {
        "sets": sets,
        "batch": batch,
        "model": model,
        "predicted_state": predicted_state,
        "predicted_expression": predicted_expression,
        "losses": losses,
        "gradient_norms": norms,
        "project_out_update_norm": update_size,
    }


if __name__ == "__main__":
    result = run_demo()
    print("batch shapes:")
    for key, value in result["batch"].items():
        print(f"  {key:18s} {tuple(value.shape)}")
    print("losses:")
    for key in ("latent_per_set", "gene_per_set", "latent_loss", "gene_loss", "total_loss"):
        print(f"  {key:18s} {result['losses'][key].detach()}")
    print("gradient norms:", result["gradient_norms"])
    print("project_out update norm:", result["project_out_update_norm"])
