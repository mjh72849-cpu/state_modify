"""Balanced set sampling for heterogeneous perturbation datasets."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterator

import numpy as np
from cell_load.data_modules.samplers import PerturbationBatchSampler


class BalancedPerturbationBatchSampler(PerturbationBatchSampler):
    """Sample datasets and perturbations uniformly at the cell-set level.

    ``cell-load`` normally creates one set per available block of cells, which
    makes large datasets and large perturbation groups dominate.  This sampler
    retains its grouping and distributed logic, then resamples complete sets.
    It never mixes cells from different dataset/perturbation groups.
    """

    def __init__(
        self,
        *args,
        balance_datasets: bool = True,
        balance_perturbations: bool = True,
        sets_per_dataset_per_epoch: int | None = 2048,
        control_perturbation: str | None = None,
        control_only_sets_per_epoch: int | None = 64,
        **kwargs,
    ):
        self.balance_datasets = bool(balance_datasets)
        self.balance_perturbations = bool(balance_perturbations)
        if sets_per_dataset_per_epoch is not None and sets_per_dataset_per_epoch <= 0:
            raise ValueError("sets_per_dataset_per_epoch must be positive or None")
        self.sets_per_dataset_per_epoch = sets_per_dataset_per_epoch
        self.control_perturbation = None if control_perturbation is None else str(control_perturbation).casefold()
        self.control_only_sets_per_epoch = control_only_sets_per_epoch
        if control_only_sets_per_epoch is not None and control_only_sets_per_epoch <= 0:
            raise ValueError("control_only_sets_per_epoch must be positive or None")
        super().__init__(*args, **kwargs)
        self._natural_sentences = list(self.sentences)
        self.sentences = self._balanced_sentences(self.seed + self.epoch)
        self.batches = self._create_batches()

    def _sentence_identity(self, sentence: list[int]) -> tuple[str, str]:
        if not sentence:
            raise ValueError("Cannot balance an empty cell set")
        global_index = int(sentence[0])
        subset_index = bisect_right(self.dataset.cumulative_sizes, global_index)
        previous = 0 if subset_index == 0 else self.dataset.cumulative_sizes[subset_index - 1]
        subset = self.dataset.datasets[subset_index]
        local_index = global_index - previous
        source_index = int(subset.indices[local_index])
        base = subset.dataset
        cache = base.metadata_cache
        perturbation = str(cache.pert_categories[int(cache.pert_codes[source_index])])
        dataset_name = str(getattr(base, "name", base.h5_path))
        return dataset_name, perturbation

    def _balanced_sentences(self, seed: int) -> list[list[int]]:
        if self.test or (not self.balance_datasets and not self.balance_perturbations):
            return list(self._natural_sentences)

        pools: dict[str, dict[str, list[list[int]]]] = defaultdict(lambda: defaultdict(list))
        for sentence in self._natural_sentences:
            dataset_name, perturbation = self._sentence_identity(sentence)
            pools[dataset_name][perturbation].append(sentence)
        if not pools:
            return []

        natural_counts = {name: sum(len(items) for items in perts.values()) for name, perts in pools.items()}
        if self.sets_per_dataset_per_epoch is not None:
            target_counts = {name: self.sets_per_dataset_per_epoch for name in pools}
        elif self.balance_datasets:
            # The median avoids both discarding nearly all large-dataset sets
            # and expanding every small dataset to the largest source.
            target = max(1, int(np.median(list(natural_counts.values()))))
            target_counts = {name: target for name in pools}
        else:
            target_counts = natural_counts

        # A LOCO split intentionally leaves the held-out context's controls in
        # the training pool. Without this cap, dataset balancing would make the
        # resulting control-only subset consume a full dataset share (25% for
        # four sources), overwhelming perturbation learning.
        if self.control_perturbation is not None and self.control_only_sets_per_epoch is not None:
            for dataset_name, perturbation_pools in pools.items():
                labels = {label.casefold() for label in perturbation_pools}
                if labels == {self.control_perturbation}:
                    target_counts[dataset_name] = min(
                        target_counts[dataset_name], self.control_only_sets_per_epoch
                    )

        rng = np.random.default_rng(seed)
        balanced: list[list[int]] = []
        for dataset_name in sorted(pools):
            perturbation_pools = pools[dataset_name]
            perturbations = sorted(perturbation_pools)
            target = target_counts[dataset_name]
            if self.balance_perturbations:
                chosen_perts = rng.choice(perturbations, size=target, replace=True)
                for perturbation in chosen_perts:
                    candidates = perturbation_pools[str(perturbation)]
                    balanced.append(candidates[int(rng.integers(len(candidates)))])
            else:
                candidates = [sentence for group in perturbation_pools.values() for sentence in group]
                replace = target > len(candidates)
                chosen = rng.choice(len(candidates), size=target, replace=replace)
                balanced.extend(candidates[int(index)] for index in chosen)
        rng.shuffle(balanced)
        return balanced

    def __iter__(self) -> Iterator[list[int]]:
        self.sentences = self._balanced_sentences(self.seed + self.epoch)
        self.batches = self._create_batches()
        yield from self.batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.sentences = self._balanced_sentences(self.seed + self.epoch)
        self.batches = self._create_batches()
