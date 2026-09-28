"""A cell-load adapter that supplies panel-free decoder supervision."""

from __future__ import annotations

import io
import os
from typing import BinaryIO

import torch
from cell_load.data_modules import PerturbationDataModule
from cell_load.dataset import MetadataConcatDataset, PerturbationDataset
from torch.utils.data import DataLoader

from .data import canonical_gene_name, load_gene_name_list
from .sampler import BalancedPerturbationBatchSampler


def _worker_init_fn(_worker_id: int) -> None:
    """Open HDF5 handles inside workers rather than inheriting parent handles."""
    from torch.utils.data import get_worker_info

    worker_info = get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "ensure_h5_open"):
        worker_info.dataset.ensure_h5_open()


class PanelFreePerturbationDataModule(PerturbationDataModule):
    """Use native gene panels while retaining cell-load's set sampler.

    The underlying sampler already keeps a sentence/set within one
    dataset/context/perturbation group.  This adapter replaces only collation:
    it samples one native-panel gene subset for the whole set, normalizes raw
    counts to log1p(CP10K), and emits the panel-free decoder batch keys.
    """

    def __init__(
        self,
        *args,
        max_decoder_genes: int = 1024,
        decoder_target_sum: float = 10_000.0,
        decoder_seed: int = 42,
        decoder_always_include: list[str] | None = None,
        decoder_deg_fraction: float = 0.0,
        decoder_deg_min_control_cpm: float = 5.0,
        decoder_fallback_gene_names_file: str | None = None,
        trainable_perturbation_names_file: str | None = None,
        decoder_control_residual: bool = False,
        decoder_read_depth: bool = False,
        decoder_shared_batch_panel: bool = False,
        decoder_exclude_gene_names_file: str | None = None,
        group_batches_by_dataset: bool = False,
        set_group_by_batch: bool | None = None,
        focus_perturbations_file: str | None = None,
        balance_datasets: bool = True,
        balance_perturbations: bool = True,
        sets_per_dataset_per_epoch: int | None = 2048,
        validation_sets_per_dataset: int | None = 256,
        control_only_sets_per_epoch: int | None = 64,
        **kwargs,
    ):
        if kwargs.get("output_space", "gene") != "all":
            raise ValueError("PanelFreePerturbationDataModule requires output_space='all' for native panels")
        if not kwargs.get("embed_key"):
            raise ValueError("PanelFreePerturbationDataModule requires an SE embed_key such as X_state")
        if not kwargs.get("perturbation_features_file"):
            raise ValueError("A semantic perturbation_features_file is required for genetic zero-shot transfer")
        if max_decoder_genes <= 0:
            raise ValueError("max_decoder_genes must be positive")
        if decoder_target_sum <= 0:
            raise ValueError("decoder_target_sum must be positive")
        if not 0.0 <= decoder_deg_fraction <= 1.0:
            raise ValueError("decoder_deg_fraction must be between 0 and 1")
        if decoder_deg_min_control_cpm < 0:
            raise ValueError("decoder_deg_min_control_cpm cannot be negative")
        self.max_decoder_genes = int(max_decoder_genes)
        self.decoder_target_sum = float(decoder_target_sum)
        self.decoder_deg_fraction = float(decoder_deg_fraction)
        self.decoder_deg_min_control_cpm = float(decoder_deg_min_control_cpm)
        self.pin_memory = bool(kwargs.get("pin_memory", True))
        self.decoder_generator = torch.Generator().manual_seed(decoder_seed)
        self.decoder_always_include = {
            canonical_gene_name(name) for name in (decoder_always_include or [])
        }
        self.decoder_fallback_gene_names_file = decoder_fallback_gene_names_file
        self.trainable_perturbation_names_file = trainable_perturbation_names_file
        self.decoder_control_residual = bool(decoder_control_residual)
        self.decoder_read_depth = bool(decoder_read_depth)
        self.decoder_shared_batch_panel = bool(decoder_shared_batch_panel)
        self.decoder_exclude_gene_names_file = decoder_exclude_gene_names_file
        self.decoder_exclude_gene_names = set(
            load_gene_name_list(decoder_exclude_gene_names_file)
            if decoder_exclude_gene_names_file else []
        )
        self.group_batches_by_dataset = bool(group_batches_by_dataset)
        # Keep control mapping batch-matched while optionally assembling a
        # perturbation Set across batches. Small H1 batch x target groups
        # otherwise repeat only a handful of cells up to cell_sentence_len.
        self.set_group_by_batch = (
            kwargs.get("basal_mapping_strategy") == "batch"
            if set_group_by_batch is None else bool(set_group_by_batch)
        )
        self.focus_perturbations_file = focus_perturbations_file
        self.focus_perturbations = (
            load_gene_name_list(focus_perturbations_file)
            if focus_perturbations_file else []
        )
        self._shared_decoder_names: list[str] = []
        self._shared_panel_indices: dict[str, list[int]] = {}
        self.balance_datasets = bool(balance_datasets)
        self.balance_perturbations = bool(balance_perturbations)
        self.sets_per_dataset_per_epoch = sets_per_dataset_per_epoch
        self.validation_sets_per_dataset = validation_sets_per_dataset
        self.control_only_sets_per_epoch = control_only_sets_per_epoch
        if validation_sets_per_dataset is not None and validation_sets_per_dataset <= 0:
            raise ValueError("validation_sets_per_dataset must be positive or None")
        if control_only_sets_per_epoch is not None and control_only_sets_per_epoch <= 0:
            raise ValueError("control_only_sets_per_epoch must be positive or None")
        fallback_names = (
            load_gene_name_list(decoder_fallback_gene_names_file) if decoder_fallback_gene_names_file else []
        )
        self.decoder_fallback_gene_to_id = {name: index for index, name in enumerate(fallback_names)}
        trainable_perturbations = (
            load_gene_name_list(trainable_perturbation_names_file)
            if trainable_perturbation_names_file
            else []
        )
        self.trainable_perturbation_to_id = {
            name: index for index, name in enumerate(trainable_perturbations)
        }
        self._panel_names: dict[str, list[str]] = {}
        super().__init__(*args, **kwargs)
        # Native panels are canonicalized below, so the semantic lookup must
        # use the same namespace. Prefer a non-zero protein vector if cell-load
        # also inserted a zero placeholder for a differently-cased target.
        # Retain exact metadata aliases too: cell-load performs its perturbation
        # lookup before collation, so e.g. the lower-case ``non-targeting``
        # category must remain addressable even though decoder genes are upper-case.
        raw_feature_map = dict(self.pert_onehot_map)
        canonical_features = self._canonicalize_feature_map(raw_feature_map)
        for raw_name, raw_value in raw_feature_map.items():
            canonical_features.setdefault(str(raw_name), torch.as_tensor(raw_value))
        self.pert_onehot_map = canonical_features

    @staticmethod
    def _canonicalize_feature_map(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        canonical_features: dict[str, torch.Tensor] = {}
        for raw_name, raw_value in features.items():
            name = canonical_gene_name(raw_name)
            value = torch.as_tensor(raw_value)
            current = canonical_features.get(name)
            if current is None or (current.abs().sum().item() == 0 and value.abs().sum().item() > 0):
                canonical_features[name] = value
            elif current.abs().sum().item() > 0 and value.abs().sum().item() > 0:
                if current.shape != value.shape or not torch.equal(current, value):
                    raise ValueError(f"Conflicting semantic embeddings canonicalize to {name!r}")
        return canonical_features

    def setup(self, stage: str | None = None):
        super().setup(stage)
        for dataset in (*self.train_datasets, *self.val_datasets, *self.test_datasets):
            source = dataset
            while not hasattr(source, "get_gene_names") and hasattr(source, "dataset"):
                source = source.dataset
            if not hasattr(source, "get_gene_names") or not hasattr(source, "name"):
                raise TypeError(f"Cannot resolve native gene panel from {type(dataset).__name__}")
            names = [canonical_gene_name(name) for name in source.get_gene_names(output_space="all")]
            if len(names) != len(set(names)):
                seen = set()
                duplicates = sorted({name for name in names if name in seen or seen.add(name)})
                raise ValueError(
                    f"Dataset {source.name!r} has duplicate canonical gene names: {duplicates[:10]}. "
                    "Collapse duplicate columns by summing raw counts before training."
                )
            previous = self._panel_names.setdefault(source.name, names)
            if previous != names:
                raise ValueError(f"Dataset {source.name!r} contains files with inconsistent gene panels")
        if self.decoder_shared_batch_panel:
            common = set.intersection(*(set(names) for names in self._panel_names.values()))
            common.difference_update(self.decoder_exclude_gene_names)
            eligible = [
                name for name in sorted(common)
                if (
                    name in self.decoder_fallback_gene_to_id
                    or (
                        name in self.pert_onehot_map
                        and torch.as_tensor(self.pert_onehot_map[name]).abs().sum().item() > 0
                    )
                )
            ]
            if len(eligible) < 2:
                raise ValueError("Shared decoder panel has fewer than two supported genes")
            panel_generator = torch.Generator().manual_seed(self.decoder_generator.initial_seed())
            order = torch.randperm(len(eligible), generator=panel_generator)
            self._shared_decoder_names = [eligible[int(index)] for index in order[:self.max_decoder_genes]]
            self._shared_panel_indices = {}
            for dataset_name, names in self._panel_names.items():
                lookup = {name: index for index, name in enumerate(names)}
                self._shared_panel_indices[dataset_name] = [
                    lookup[gene] for gene in self._shared_decoder_names
                ]

    def _select_gene_indices(
        self,
        names: list[str],
        *,
        required_names: set[str] | None = None,
        differential_scores: torch.Tensor | None = None,
    ) -> list[int]:
        # pert_onehot_map is the complete semantic feature dictionary loaded by
        # cell-load from perturbation_features_file, not merely a one-hot map.
        valid = []
        fallback_gene_to_id = getattr(self, "decoder_fallback_gene_to_id", {})
        for index, name in enumerate(names):
            value = self.pert_onehot_map.get(name)
            has_semantic_embedding = value is not None and torch.as_tensor(value).abs().sum().item() > 0
            if has_semantic_embedding or name in fallback_gene_to_id:
                valid.append(index)
        if not valid:
            raise ValueError("No native-panel genes have protein or configured fallback embeddings")
        if len(valid) <= self.max_decoder_genes:
            return valid
        required_lookup = self.decoder_always_include | (required_names or set())
        required = [index for index in valid if names[index] in required_lookup]
        if len(required) > self.max_decoder_genes:
            raise ValueError("decoder_always_include exceeds max_decoder_genes")
        required_set = set(required)
        remaining = [index for index in valid if index not in required_set]
        take = self.max_decoder_genes - len(required)

        # Enrich decoder supervision for genes with the largest matched-control
        # expression shifts. These are DE-ranked genes rather than genes called
        # significant by a hypothesis test: using a rank keeps the requested
        # fraction stable even for small perturbation groups.
        deg_selected: list[int] = []
        deg_fraction = float(getattr(self, "decoder_deg_fraction", 0.0))
        if differential_scores is not None and deg_fraction > 0 and take > 0:
            scores = torch.as_tensor(differential_scores, dtype=torch.float32).flatten()
            if scores.numel() != len(names):
                raise ValueError("differential_scores must align with the native gene panel")
            deg_take = min(round(self.max_decoder_genes * deg_fraction), take, len(remaining))
            if deg_take:
                remaining_tensor = torch.tensor(remaining, dtype=torch.long)
                candidate_scores = scores[remaining_tensor]
                finite = torch.isfinite(candidate_scores)
                deg_take = min(deg_take, int(finite.sum().item()))
                if deg_take:
                    eligible_indices = remaining_tensor[finite]
                    eligible_scores = candidate_scores[finite]
                    order = torch.topk(
                        eligible_scores, k=deg_take, largest=True, sorted=True
                    ).indices
                    deg_selected = eligible_indices[order].tolist()

        selected_set = required_set | set(deg_selected)
        random_pool = torch.tensor(
            [index for index in valid if index not in selected_set], dtype=torch.long
        )
        random_take = self.max_decoder_genes - len(required) - len(deg_selected)
        sampled = (
            random_pool[
                torch.randperm(len(random_pool), generator=self.decoder_generator)[:random_take]
            ].tolist()
            if random_take
            else []
        )
        return required + deg_selected + sampled

    def _panel_free_collate(self, samples: list[dict]) -> dict:
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        if len(samples) % self.cell_sentence_len:
            raise ValueError(
                f"Training meta-batch has {len(samples)} cells, which is not divisible by "
                f"cell_sentence_len={self.cell_sentence_len}"
            )

        set_batches = []
        set_targets = []
        set_embeddings = []
        set_fallback_ids = []
        set_baselines = []
        set_read_depths = []
        set_names = []
        set_perturbation_ids = []
        set_dataset_names = []
        set_pert_library_sizes = []
        set_ctrl_library_sizes = []
        for start in range(0, len(samples), self.cell_sentence_len):
            set_samples = samples[start : start + self.cell_sentence_len]
            dataset_names = {sample["dataset_name"] for sample in set_samples}
            if len(dataset_names) != 1:
                raise ValueError(
                    "Every cell set must come from one dataset; got " + ", ".join(sorted(dataset_names))
                )
            dataset_name = next(iter(dataset_names))
            set_dataset_names.append(dataset_name)
            if dataset_name not in self._panel_names:
                raise KeyError(f"No native gene panel registered for dataset {dataset_name!r}")

            base = PerturbationDataset.collate_fn(set_samples, exp_counts=False)
            raw_counts = base.pop("pert_cell_counts").float().clamp_min(0)
            ctrl_raw_counts = base.pop("ctrl_cell_counts", None)
            if self.is_log1p:
                raw_counts = torch.expm1(raw_counts).clamp_min(0)

            names = self._panel_names[dataset_name]
            perturbations = {canonical_gene_name(sample["pert_name"]) for sample in set_samples}
            if len(perturbations) != 1:
                raise ValueError(
                    "Every cell set must contain one perturbation; got " + ", ".join(sorted(perturbations))
                )
            perturbation = next(iter(perturbations))
            control_perturbation = canonical_gene_name(getattr(self, "control_pert", "non-targeting"))
            trainable_perturbation_to_id = getattr(self, "trainable_perturbation_to_id", {})
            if trainable_perturbation_to_id:
                if perturbation == control_perturbation:
                    perturbation_id = -1
                else:
                    try:
                        perturbation_id = trainable_perturbation_to_id[perturbation]
                    except KeyError as error:
                        raise KeyError(
                            f"Perturbation {perturbation!r} from {dataset_name!r} is absent from "
                            f"trainable_perturbation_names_file={self.trainable_perturbation_names_file!r}. "
                            "Regenerate the registry instead of silently sharing an unknown ID."
                        ) from error
                set_perturbation_ids.append(perturbation_id)
            required_names = set() if perturbation == control_perturbation else {perturbation}

            # All normalization denominators use the complete native panel.
            totals = raw_counts.sum(dim=-1, keepdim=True)
            set_pert_library_sizes.append(totals.squeeze(-1))
            scale = torch.where(
                totals > 0,
                self.decoder_target_sum / totals,
                torch.zeros_like(totals),
            )
            if ctrl_raw_counts is not None:
                ctrl_raw_counts = ctrl_raw_counts.float().clamp_min(0)
                if self.is_log1p:
                    ctrl_raw_counts = torch.expm1(ctrl_raw_counts).clamp_min(0)
                ctrl_totals = ctrl_raw_counts.sum(dim=-1, keepdim=True)
                set_ctrl_library_sizes.append(ctrl_totals.squeeze(-1))
                ctrl_scale = torch.where(
                    ctrl_totals > 0,
                    self.decoder_target_sum / ctrl_totals,
                    torch.zeros_like(ctrl_totals),
                )
            else:
                ctrl_scale = None

            if getattr(self, "decoder_read_depth", False):
                if ctrl_raw_counts is None or ctrl_scale is None:
                    raise KeyError(
                        "decoder_read_depth=True requires store_raw_basal=True "
                        "so matched ctrl_cell_counts are available"
                    )
                # Paper-compatible scalar: mean log1p(CP10K) over genes that
                # are expressed in the input/control cell.  It is supplied to
                # the decoder as a per-cell read-depth/context feature.
                ctrl_log = torch.log1p(ctrl_raw_counts * ctrl_scale)
                expressed = ctrl_raw_counts > 0
                set_read_depths.append(
                    (ctrl_log * expressed).sum(dim=-1)
                    / expressed.sum(dim=-1).clamp_min(1)
                )

            differential_scores = None
            if (
                not getattr(self, "decoder_shared_batch_panel", False)
                and float(getattr(self, "decoder_deg_fraction", 0.0)) > 0
                and perturbation != control_perturbation
            ):
                if ctrl_raw_counts is None or ctrl_scale is None:
                    raise KeyError(
                        "decoder_deg_fraction > 0 requires store_raw_basal=True "
                        "so matched ctrl_cell_counts are available"
                    )
                # Mirror the vcc2026 DE table as closely as is practical in a
                # streaming collator: rank by absolute log2 fold-change of
                # arithmetic mean normalized counts, and use the scorer's
                # control-only >5 CPM gene filter. Statistical significance
                # itself requires a cached whole-group Wilcoxon/BH pass and is
                # intentionally not approximated with a per-Set p-value here.
                pert_mean = (raw_counts * scale).mean(dim=0)
                ctrl_mean = (ctrl_raw_counts * ctrl_scale).mean(dim=0)
                epsilon = self.decoder_target_sum * 1.0e-15  # 1e-9 at CPM=1e6
                differential_scores = torch.abs(
                    torch.log2((pert_mean + epsilon) / (ctrl_mean + epsilon))
                )
                min_control = (
                    float(getattr(self, "decoder_deg_min_control_cpm", 5.0))
                    * self.decoder_target_sum
                    / 1_000_000.0
                )
                differential_scores = differential_scores.masked_fill(
                    ctrl_mean <= min_control, float("-inf")
                )

            indices = (
                self._shared_panel_indices[dataset_name]
                if getattr(self, "decoder_shared_batch_panel", False)
                else self._select_gene_indices(
                    names,
                    required_names=required_names,
                    differential_scores=differential_scores,
                )
            )
            index_tensor = torch.tensor(indices, dtype=torch.long)
            selected_counts = raw_counts.index_select(-1, index_tensor)
            set_targets.append(torch.log1p(selected_counts * scale))
            if getattr(self, "decoder_control_residual", False):
                if ctrl_raw_counts is None:
                    raise KeyError(
                        "decoder_control_residual=True requires store_raw_basal=True "
                        "so ctrl_cell_counts is available"
                    )
                ctrl_selected = ctrl_raw_counts.index_select(-1, index_tensor)
                assert ctrl_scale is not None
                # Keep one baseline per matched donor cell.  A Set-mean log
                # baseline destroys control-cell heterogeneity and, after
                # expm1/count reconstruction, creates a large perturbation-
                # independent pseudobulk shift (Jensen's inequality).
                set_baselines.append(torch.log1p(ctrl_selected * ctrl_scale))
            selected_names = [names[index] for index in indices]
            set_names.append(selected_names)
            embedding_dim = int(torch.as_tensor(next(iter(self.pert_onehot_map.values()))).numel())
            embeddings = []
            fallback_ids = []
            fallback_gene_to_id = getattr(self, "decoder_fallback_gene_to_id", {})
            for name in selected_names:
                value = self.pert_onehot_map.get(name)
                if value is not None and torch.as_tensor(value).abs().sum().item() > 0:
                    embeddings.append(torch.as_tensor(value, dtype=torch.float32))
                    fallback_ids.append(-1)
                else:
                    embeddings.append(torch.zeros(embedding_dim, dtype=torch.float32))
                    fallback_ids.append(fallback_gene_to_id[name])
            set_embeddings.append(torch.stack(embeddings))
            set_fallback_ids.append(torch.tensor(fallback_ids, dtype=torch.long))
            set_batches.append(base)

        # Common ST inputs remain flattened as [B*S, ...], matching the
        # original cell-load contract. Metadata lists are flattened likewise.
        merged = {}
        for key in set_batches[0]:
            values = [batch[key] for batch in set_batches]
            if torch.is_tensor(values[0]):
                merged[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], list):
                merged[key] = [item for value in values for item in value]
            else:
                merged[key] = values

        batch_sets = len(set_batches)
        max_genes = max(target.shape[-1] for target in set_targets)
        embedding_dim = set_embeddings[0].shape[-1]
        gene_targets = torch.zeros(batch_sets, self.cell_sentence_len, max_genes)
        gene_mask = torch.zeros(batch_sets, self.cell_sentence_len, max_genes, dtype=torch.bool)
        gene_embeddings = torch.zeros(batch_sets, max_genes, embedding_dim)
        gene_fallback_ids = torch.full((batch_sets, max_genes), -1, dtype=torch.long)
        gene_baselines = (
            torch.zeros(batch_sets, self.cell_sentence_len, max_genes)
            if getattr(self, "decoder_control_residual", False)
            else None
        )
        decoder_read_depth = (
            torch.zeros(batch_sets, self.cell_sentence_len, 1)
            if getattr(self, "decoder_read_depth", False)
            else None
        )
        padded_names: list[list[str | None]] = []
        for set_index, (targets, embeddings, fallback_ids, names) in enumerate(
            zip(set_targets, set_embeddings, set_fallback_ids, set_names)
        ):
            width = targets.shape[-1]
            gene_targets[set_index, :, :width] = targets
            gene_mask[set_index, :, :width] = True
            gene_embeddings[set_index, :width] = embeddings
            gene_fallback_ids[set_index, :width] = fallback_ids
            if gene_baselines is not None:
                gene_baselines[set_index, :, :width] = set_baselines[set_index]
            if decoder_read_depth is not None:
                decoder_read_depth[set_index, :, 0] = set_read_depths[set_index]
            padded_names.append(names + [None] * (max_genes - width))

        merged.update(
            {
                "gene_embeddings": gene_embeddings,
                "gene_fallback_ids": gene_fallback_ids,
                "gene_targets": gene_targets,
                "gene_mask": gene_mask,
                "gene_names": padded_names,
                "set_dataset_names": set_dataset_names,
                "decoder_pert_library_sizes": torch.stack(set_pert_library_sizes),
            }
        )
        if len(set_ctrl_library_sizes) == batch_sets:
            merged["decoder_ctrl_library_sizes"] = torch.stack(set_ctrl_library_sizes)
        if getattr(self, "trainable_perturbation_to_id", {}):
            # Match cell-load's flattened [B*S, ...] ST inputs. Every cell in a
            # Set receives the same target ID; control is the sentinel -1.
            merged["perturbation_ids"] = torch.tensor(
                set_perturbation_ids, dtype=torch.long
            ).repeat_interleave(self.cell_sentence_len)
        if gene_baselines is not None:
            merged["gene_baselines"] = gene_baselines
        if decoder_read_depth is not None:
            merged["decoder_read_depth"] = decoder_read_depth
        return merged

    def get_var_dims(self):
        dims = super().get_var_dims()
        dims["num_trainable_perturbations"] = len(self.trainable_perturbation_to_id)
        dims["trainable_perturbation_names"] = list(self.trainable_perturbation_to_id)
        return dims

    def _create_dataloader(self, datasets, test=False, batch_size=None, validation=False):
        dataset = MetadataConcatDataset(datasets)
        effective_batch_size = batch_size or (1 if test else self.batch_size)
        sampler = BalancedPerturbationBatchSampler(
            dataset=dataset,
            batch_size=effective_batch_size,
            drop_last=self.drop_last,
            cell_sentence_len=self.cell_sentence_len,
            test=test,
            use_batch=self.set_group_by_batch,
            use_consecutive_loading=self.use_consecutive_loading,
            downsample_cells=self.downsample_cells,
            seed=self.random_seed,
            balance_datasets=(self.balance_datasets and not test) or validation,
            balance_perturbations=(self.balance_perturbations and not test) or validation,
            sets_per_dataset_per_epoch=(
                self.validation_sets_per_dataset
                if validation
                else self.sets_per_dataset_per_epoch if not test else None
            ),
            control_perturbation=getattr(self, "control_pert", None),
            control_only_sets_per_epoch=self.control_only_sets_per_epoch,
            group_batches_by_dataset=self.group_batches_by_dataset and not test,
            focus_perturbations=self.focus_perturbations if not validation and not test else None,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._panel_free_collate,
            pin_memory=self.pin_memory,
            prefetch_factor=4 if not test and self.num_workers > 0 else None,
            persistent_workers=bool(self.num_workers > 0 and not test),
            worker_init_fn=_worker_init_fn if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        """Evaluate each natural validation Set once without train balancing."""
        datasets = self.val_datasets or self.test_datasets
        if not datasets:
            return []
        return self._create_dataloader(datasets, test=False, validation=True)

    def save_state(self, filepath: str | os.PathLike[str] | BinaryIO):
        """Persist the extra constructor settings alongside cell-load state."""
        # The parent intentionally stores a compact dict rather than pickling
        # the large semantic embedding table.  Serialize that base state to an
        # in-memory buffer first: callers in older STATE versions may pass a
        # write-only file handle, which cannot be read back with ``torch.load``.
        base_state = io.BytesIO()
        super().save_state(base_state)
        base_state.seek(0)
        state = torch.load(base_state, weights_only=False)
        state.update(
            {
                "max_decoder_genes": self.max_decoder_genes,
                "decoder_target_sum": self.decoder_target_sum,
                "decoder_seed": self.decoder_generator.initial_seed(),
                "decoder_always_include": sorted(self.decoder_always_include),
                "decoder_deg_fraction": float(getattr(self, "decoder_deg_fraction", 0.0)),
                "decoder_deg_min_control_cpm": float(
                    getattr(self, "decoder_deg_min_control_cpm", 5.0)
                ),
                "decoder_fallback_gene_names_file": self.decoder_fallback_gene_names_file,
                "decoder_shared_batch_panel": getattr(self, "decoder_shared_batch_panel", False),
                "decoder_exclude_gene_names_file": getattr(self, "decoder_exclude_gene_names_file", None),
                "group_batches_by_dataset": getattr(self, "group_batches_by_dataset", False),
                "set_group_by_batch": getattr(self, "set_group_by_batch", None),
                "focus_perturbations_file": getattr(self, "focus_perturbations_file", None),
                "decoder_fallback_gene_names": list(self.decoder_fallback_gene_to_id),
                "trainable_perturbation_names_file": self.trainable_perturbation_names_file,
                "trainable_perturbation_names": list(self.trainable_perturbation_to_id),
                "balance_datasets": self.balance_datasets,
                "balance_perturbations": self.balance_perturbations,
                "sets_per_dataset_per_epoch": self.sets_per_dataset_per_epoch,
                "validation_sets_per_dataset": self.validation_sets_per_dataset,
                "control_only_sets_per_epoch": self.control_only_sets_per_epoch,
            }
        )
        torch.save(state, filepath)
