"""Convert decoded log-CP10K profiles into submission-compatible counts."""

from __future__ import annotations

import torch


def log_cp10k_to_probabilities(log_expression: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert non-negative log1p(CP10K) values to compositional profiles."""

    if log_expression.dim() < 1:
        raise ValueError("log_expression must have at least one dimension")
    abundance = torch.expm1(log_expression.float().clamp_min(0)).clamp_min(0)
    totals = abundance.sum(dim=-1, keepdim=True)
    if torch.any(totals <= eps):
        raise ValueError("Cannot form a count profile from an all-zero decoded cell")
    return abundance / totals


def log_cp10k_to_counts(
    log_expression: torch.Tensor,
    library_sizes: torch.Tensor | int,
    *,
    concentration: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample integer counts using multinomial or Dirichlet-multinomial noise.

    Args:
        log_expression: ``[..., G]`` decoded log1p(CP10K) profiles.
        library_sizes: scalar or one size per flattened cell.
        concentration: if provided, first sample cell probabilities from a
            Dirichlet distribution.  Smaller values add more over-dispersion.
            Dirichlet sampling does not currently accept a custom generator.
    """

    probabilities = log_cp10k_to_probabilities(log_expression)
    return probabilities_to_counts(
        probabilities,
        library_sizes,
        concentration=concentration,
        generator=generator,
    )


def probabilities_to_counts(
    probabilities: torch.Tensor,
    library_sizes: torch.Tensor | int,
    *,
    concentration: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Convert cell-wise gene probabilities to exact-depth integer counts."""
    if probabilities.dim() < 1 or probabilities.shape[-1] == 0:
        raise ValueError("probabilities must have a nonempty gene dimension")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    totals = probabilities.sum(dim=-1, keepdim=True)
    if torch.any(totals <= 0):
        raise ValueError("Each probability row must have positive mass")
    probabilities = probabilities / totals
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    sizes = torch.as_tensor(library_sizes, device=flat.device)
    if sizes.numel() == 1:
        sizes = sizes.expand(flat.shape[0])
    sizes = sizes.reshape(-1)
    if sizes.numel() != flat.shape[0]:
        raise ValueError(f"Expected one library size per cell ({flat.shape[0]}), got {sizes.numel()}")
    if torch.any(sizes < 0) or torch.any(sizes != sizes.round()):
        raise ValueError("library_sizes must be non-negative integers")
    if concentration is not None:
        if concentration <= 0:
            raise ValueError("concentration must be positive")
        flat = torch.distributions.Dirichlet(flat * concentration + 1e-8).sample()

    rows = [
        torch.multinomial(p, int(n.item()), replacement=True, generator=generator).bincount(minlength=p.numel())
        for p, n in zip(flat, sizes)
    ]
    counts = torch.stack(rows).to(torch.int64)
    return counts.reshape(*probabilities.shape[:-1], probabilities.shape[-1])
