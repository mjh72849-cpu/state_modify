"""Memory-bounded inference for a VCC target/context pair."""

from __future__ import annotations

import torch

from ..models.decoders import PanelFreeGeneDecoder
from .counts import log_cp10k_to_counts, log_cp10k_to_probabilities, probabilities_to_counts


@torch.no_grad()
def predict_vcc_log_expression(
    model,
    control_embeddings: torch.Tensor,
    perturbation_embeddings: torch.Tensor,
    query_gene_embeddings: torch.Tensor,
    *,
    perturbation_ids: torch.Tensor | int | None = None,
    query_gene_fallback_ids: torch.Tensor | None = None,
    query_gene_baseline: torch.Tensor | None = None,
    query_read_depth: torch.Tensor | None = None,
    cell_chunk_size: int = 256,
    gene_chunk_size: int = 512,
) -> torch.Tensor:
    """Return decoded log1p(CP10K) without assigning raw library sizes."""
    decoder = getattr(model, "gene_decoder", None)
    if not isinstance(decoder, PanelFreeGeneDecoder):
        raise TypeError("model.gene_decoder must be a PanelFreeGeneDecoder")
    if control_embeddings.dim() != 2:
        raise ValueError("control_embeddings must have shape [cells, latent_dim]")
    if perturbation_embeddings.dim() == 1:
        perturbation_embeddings = perturbation_embeddings.expand(control_embeddings.shape[0], -1)
    if perturbation_embeddings.dim() != 2 or perturbation_embeddings.shape[0] != control_embeddings.shape[0]:
        raise ValueError("perturbation_embeddings must have shape [cells, pert_dim] or [pert_dim]")
    if cell_chunk_size <= 0:
        raise ValueError("cell_chunk_size must be positive")
    if perturbation_ids is not None:
        perturbation_ids = torch.as_tensor(
            perturbation_ids, dtype=torch.long, device=control_embeddings.device
        )
        if perturbation_ids.numel() == 1:
            perturbation_ids = perturbation_ids.expand(control_embeddings.shape[0])
        if perturbation_ids.dim() != 1 or perturbation_ids.shape[0] != control_embeddings.shape[0]:
            raise ValueError("perturbation_ids must be a scalar or have shape [cells]")

    was_training = model.training
    model.eval()
    decoded = []
    try:
        for start in range(0, control_embeddings.shape[0], cell_chunk_size):
            stop = start + cell_chunk_size
            batch = {
                "ctrl_cell_emb": control_embeddings[start:stop],
                "pert_emb": perturbation_embeddings[start:stop],
            }
            if perturbation_ids is not None:
                batch["perturbation_ids"] = perturbation_ids[start:stop]
            latent = model(batch, padded=False)
            if isinstance(latent, tuple):
                latent = latent[0]
            control_latent = control_embeddings[start:stop].unsqueeze(0)
            if decoder.use_gene_baseline and query_gene_baseline is not None:
                values = decoder.forward_control_calibrated(
                    latent,
                    control_latent,
                    query_gene_embeddings,
                    fallback_ids=query_gene_fallback_ids,
                    gene_baseline=query_gene_baseline,
                    read_depth=query_read_depth[start:stop] if query_read_depth is not None else None,
                    chunk_size=gene_chunk_size,
                )
            else:
                values = decoder(
                    latent,
                    query_gene_embeddings,
                    fallback_ids=query_gene_fallback_ids,
                    gene_baseline=query_gene_baseline,
                    read_depth=query_read_depth[start:stop] if query_read_depth is not None else None,
                    chunk_size=gene_chunk_size,
                )
            decoded.append(values.squeeze(0))
    finally:
        model.train(was_training)
    return torch.cat(decoded, dim=0)


@torch.no_grad()
def predict_vcc_counts(
    model,
    control_embeddings: torch.Tensor,
    perturbation_embeddings: torch.Tensor,
    query_gene_embeddings: torch.Tensor,
    library_sizes: torch.Tensor | int,
    *,
    perturbation_ids: torch.Tensor | int | None = None,
    query_gene_fallback_ids: torch.Tensor | None = None,
    query_gene_baseline: torch.Tensor | None = None,
    query_read_depth: torch.Tensor | None = None,
    cell_chunk_size: int = 256,
    gene_chunk_size: int = 512,
    concentration: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run ST in cell chunks, query a gene panel, and sample integer counts.

    The function intentionally accepts already-computed SE embeddings.  SE is
    frozen/offline in the first-stage architecture, so inference never needs to
    load raw count matrices into the ST model.
    """

    log_expression = predict_vcc_log_expression(
        model,
        control_embeddings,
        perturbation_embeddings,
        query_gene_embeddings,
        perturbation_ids=perturbation_ids,
        query_gene_fallback_ids=query_gene_fallback_ids,
        query_gene_baseline=query_gene_baseline,
        query_read_depth=query_read_depth,
        cell_chunk_size=cell_chunk_size,
        gene_chunk_size=gene_chunk_size,
    )
    return log_cp10k_to_counts(
        log_expression,
        library_sizes,
        concentration=concentration,
        generator=generator,
    )


@torch.no_grad()
def predict_vcc_pooled_counts(
    model,
    control_embeddings: torch.Tensor,
    perturbation_embeddings: torch.Tensor,
    query_gene_embeddings: torch.Tensor,
    pool_indices: torch.Tensor,
    pooled_library_sizes: torch.Tensor,
    *,
    perturbation_ids: torch.Tensor | int | None = None,
    query_gene_fallback_ids: torch.Tensor | None = None,
    query_gene_baseline: torch.Tensor | None = None,
    query_read_depth: torch.Tensor | None = None,
    cell_chunk_size: int = 256,
    gene_chunk_size: int = 512,
    concentration: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Predict donors individually, then pool probabilities into VCC cells.

    ``pool_indices`` is ``[400, 4]`` for the official workflow. Keeping the
    four donor controls separate until after ST avoids feeding averaged latent
    vectors that were not present during training.
    """
    if pool_indices.dim() != 2 or not pool_indices.numel():
        raise ValueError("pool_indices must have shape [output_cells, pool_k]")
    flat_indices = pool_indices.reshape(-1).to(control_embeddings.device).long()
    if flat_indices.min() < 0 or flat_indices.max() >= control_embeddings.shape[0]:
        raise IndexError("pool_indices are outside control_embeddings")
    selected = control_embeddings.index_select(0, flat_indices)
    selected_perturbations = perturbation_embeddings
    selected_perturbation_ids = perturbation_ids
    selected_read_depth = query_read_depth
    if (
        perturbation_embeddings.dim() == 2
        and perturbation_embeddings.shape[0] == control_embeddings.shape[0]
    ):
        selected_perturbations = perturbation_embeddings.index_select(
            0, flat_indices.to(perturbation_embeddings.device)
        )
    if query_read_depth is not None:
        depth = torch.as_tensor(query_read_depth, device=control_embeddings.device)
        if depth.dim() > 0 and depth.shape[0] == control_embeddings.shape[0]:
            selected_read_depth = depth.index_select(0, flat_indices)
    if perturbation_ids is not None:
        selected_perturbation_ids = torch.as_tensor(
            perturbation_ids, dtype=torch.long, device=control_embeddings.device
        )
        if selected_perturbation_ids.numel() > 1:
            if selected_perturbation_ids.dim() != 1 or selected_perturbation_ids.shape[0] != control_embeddings.shape[0]:
                raise ValueError("perturbation_ids must be a scalar or have shape [control cells]")
            selected_perturbation_ids = selected_perturbation_ids.index_select(0, flat_indices)
    log_expression = predict_vcc_log_expression(
        model,
        selected,
        selected_perturbations,
        query_gene_embeddings,
        perturbation_ids=selected_perturbation_ids,
        query_gene_fallback_ids=query_gene_fallback_ids,
        query_gene_baseline=query_gene_baseline,
        query_read_depth=selected_read_depth,
        cell_chunk_size=cell_chunk_size,
        gene_chunk_size=gene_chunk_size,
    )
    probabilities = log_cp10k_to_probabilities(log_expression)
    probabilities = probabilities.reshape(*pool_indices.shape, -1).mean(dim=1)
    return probabilities_to_counts(
        probabilities,
        pooled_library_sizes,
        concentration=concentration,
        generator=generator,
    )
