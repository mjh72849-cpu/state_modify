"""Data adapters for datasets with different measured gene panels."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def canonical_gene_name(name: str) -> str:
    """Return the repository's conservative gene-symbol normalization.

    Version suffixes on Ensembl identifiers are removed, while gene symbols
    are upper-cased.  Alias resolution should happen upstream against an
    explicit annotation table; silently guessing aliases here would make
    supervision ambiguous.
    """

    value = str(name).strip()
    if value.upper().startswith("ENSG"):
        return value.split(".", 1)[0].upper()
    return value.upper()


def load_gene_name_list(path: str | Path) -> list[str]:
    """Load a stable, canonical gene list from a one-column text or CSV file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Gene-name file does not exist: {source}")
    if source.suffix.lower() == ".csv":
        import csv

        with source.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Gene-name CSV has no header: {source}")
            column = "gene_name" if "gene_name" in reader.fieldnames else reader.fieldnames[0]
            names = [canonical_gene_name(row[column]) for row in reader if row.get(column)]
    else:
        names = [canonical_gene_name(line) for line in source.read_text().splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"Gene-name file contains duplicate canonical names: {source}")
    return names


def build_gene_query_features(
    gene_names: Sequence[str],
    protein_embeddings: Mapping[str, torch.Tensor],
    fallback_gene_names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build decoder embeddings and stable fallback IDs for an inference panel."""
    lookup = {
        canonical_gene_name(k): torch.as_tensor(v, dtype=torch.float32)
        for k, v in protein_embeddings.items()
    }
    if not lookup:
        raise ValueError("protein_embeddings cannot be empty")
    dimensions = {int(value.numel()) for value in lookup.values()}
    if len(dimensions) != 1:
        raise ValueError("protein_embeddings must all have the same one-dimensional size")
    embedding_dim = dimensions.pop()
    fallback_to_id = {canonical_gene_name(name): index for index, name in enumerate(fallback_gene_names)}

    embeddings = []
    fallback_ids = []
    unsupported = []
    for raw_name in gene_names:
        name = canonical_gene_name(raw_name)
        value = lookup.get(name)
        if value is not None and value.abs().sum().item() > 0:
            embeddings.append(value)
            fallback_ids.append(-1)
        elif name in fallback_to_id:
            embeddings.append(torch.zeros(embedding_dim, dtype=torch.float32))
            fallback_ids.append(fallback_to_id[name])
        else:
            unsupported.append(name)
    if unsupported:
        preview = ", ".join(unsupported[:10])
        raise KeyError(
            f"{len(unsupported)} genes have neither protein nor configured fallback embeddings: {preview}"
        )
    return torch.stack(embeddings), torch.tensor(fallback_ids, dtype=torch.long)


class HeterogeneousGeneCollator:
    """Add padded, masked gene targets to an existing cell collator.

    Each input sample must contain ``gene_names`` and ``gene_targets``.  Panels
    may differ between samples.  Padding is marked false in ``gene_mask`` and
    therefore can never be interpreted as measured zero expression.

    ``gene_embedding_lookup`` maps canonical gene names to one-dimensional
    embedding tensors.  Genes without embeddings are omitted (and a sample
    with no covered genes is rejected), because inventing an embedding would
    undermine zero-shot interpretation. Resolve aliases and add an explicit
    learned/non-coding embedding source upstream when full coverage is needed.
    """

    def __init__(
        self,
        gene_embedding_lookup: Mapping[str, torch.Tensor],
        *,
        base_collate: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
        max_genes: int | None = None,
        always_include: Sequence[str] = (),
        generator: torch.Generator | None = None,
    ):
        if not gene_embedding_lookup:
            raise ValueError("gene_embedding_lookup cannot be empty")
        self.lookup = {
            canonical_gene_name(k): torch.as_tensor(v, dtype=torch.float32)
            for k, v in gene_embedding_lookup.items()
        }
        if any(v.dim() != 1 for v in self.lookup.values()):
            raise ValueError("All gene embeddings must be one-dimensional and have the same size")
        dimensions = {int(v.numel()) for v in self.lookup.values()}
        if len(dimensions) != 1:
            raise ValueError("All gene embeddings must be one-dimensional and have the same size")
        self.embedding_dim = dimensions.pop()
        self.base_collate = base_collate
        self.max_genes = max_genes
        if max_genes is not None and max_genes <= 0:
            raise ValueError("max_genes must be positive")
        self.always_include = {canonical_gene_name(g) for g in always_include}
        self.generator = generator

    def _select(self, names: list[str]) -> list[int]:
        valid = [i for i, name in enumerate(names) if name in self.lookup]
        if not valid:
            raise ValueError("A sample has no genes with available embeddings")
        if self.max_genes is None or len(valid) <= self.max_genes:
            return valid

        required = [i for i in valid if names[i] in self.always_include]
        if len(required) > self.max_genes:
            raise ValueError("always_include contains more measured genes than max_genes")
        pool = torch.tensor([i for i in valid if i not in set(required)], dtype=torch.long)
        take = self.max_genes - len(required)
        order = torch.randperm(len(pool), generator=self.generator)[:take]
        return required + pool[order].tolist()

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        batch = self.base_collate(samples) if self.base_collate is not None else {}

        selected: list[tuple[list[str], torch.Tensor]] = []
        for sample in samples:
            if "gene_names" not in sample or "gene_targets" not in sample:
                raise KeyError("Each sample must contain gene_names and gene_targets")
            names = [canonical_gene_name(g) for g in sample["gene_names"]]
            targets = torch.as_tensor(sample["gene_targets"], dtype=torch.float32)
            if targets.dim() != 1 or targets.numel() != len(names):
                raise ValueError("gene_targets must be one-dimensional and aligned with gene_names")
            indices = self._select(names)
            selected.append(([names[i] for i in indices], targets[indices]))

        width = max(len(names) for names, _ in selected)
        embeddings = torch.zeros(len(samples), width, self.embedding_dim, dtype=torch.float32)
        targets = torch.zeros(len(samples), width, dtype=torch.float32)
        mask = torch.zeros(len(samples), width, dtype=torch.bool)
        names_out: list[list[str | None]] = []
        for row, (names, values) in enumerate(selected):
            size = len(names)
            embeddings[row, :size] = torch.stack([self.lookup[name] for name in names])
            targets[row, :size] = values
            mask[row, :size] = True
            names_out.append(names + [None] * (width - size))

        batch.update(
            {
                "gene_embeddings": embeddings,
                "gene_targets": targets,
                "gene_mask": mask,
                "gene_names": names_out,
            }
        )
        return batch
