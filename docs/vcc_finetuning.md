# VCC genetic-perturbation architecture

This repository contains an opt-in architecture for fine-tuning the Tahoe
state-transition backbone on genetic perturbations without forcing every
dataset onto the VCC 18,533-gene panel. It does not start training or download
model/data artifacts.

## Data contract

Keep raw integer counts in `X` (or an explicitly configured counts layer).
Compute log1p(CP10K) for gene
supervision and compute `obsm["X_state"]` once with a frozen SE model. Resolve
gene aliases upstream and retain the canonical ID alongside the symbol.

The ST batch retains the existing keys:

- `ctrl_cell_emb`: frozen SE embeddings, `[N, state_dim]`;
- `pert_cell_emb`: target SE embeddings, `[N, state_dim]`;
- `pert_emb`: semantic target-gene embeddings, `[N, pert_dim]`.

The panel-free decoder adds:

- `gene_embeddings`: `[N, sampled_genes, gene_embedding_dim]`;
- `gene_fallback_ids`: `[N, sampled_genes]`, `-1` for an SE protein
  embedding and `0..342` for the configured VCC fallback table;
- `gene_targets`: measured log1p(CP10K), `[N, sampled_genes]`;
- `gene_baselines`: matched-control Set mean log1p(CP10K),
  `[sets, sampled_genes]`;
- `gene_mask`: true only for measured genes, `[N, sampled_genes]`.

`HeterogeneousGeneCollator` constructs the last three fields and can wrap an
existing cell collator. It samples genes independently per cell when
`max_genes` is set, pads differing panels, and masks padding. An unmeasured gene
is therefore never used as a zero target.

For efficient training, group cells from the same dataset/context into each ST
set and sample 512--2,048 genes per step. Include the perturbation target and a
stable VCC anchor subset with `always_include`, then fill the remainder at
random. Balance datasets and perturbations in the sampler.

The panel-free data module now enforces the first rule dynamically: the current
Set's `pert_name` is included whenever that target is measured and has a protein
or configured fallback embedding. `decoder_always_include` remains available
for global anchor genes. `BalancedPerturbationBatchSampler` samples complete
Sets uniformly across datasets and perturbations; its epoch size is controlled
by `sets_per_dataset_per_epoch` and it can be disabled with the two `balance_*`
switches.

### Local focus-dataset registry

`default_focus_dataset_specs()` reads the repository-local `data/raw` link and
provides explicit, checked
schemas for the datasets that matter most for this project:

| context | source | perturbation | batch | native genes |
|---|---|---|---|---:|
| H1 | Arc `adata_Training.h5ad` | `target_gene` | `batch` | 18,080 |
| K562 | Replogle essential | `gene` | `gem_group` | 8,563 |
| K562 | Replogle GWPS | `gene` | `gem_group` | 8,248 |
| HCT116 | XAtlas | `gene_target` | `sample` | 38,606 |

The registry treats `non-targeting`/`Non-Targeting` as controls. XAtlas also
contains 171 cells labelled `CTRL`; these are deliberately not merged into the
control pool until their semantics are confirmed. HCT116 `pass_guide_filter`
is exposed as a required QC field.

The existing `cell_load` configuration has only one global `pert_col`,
`batch_col`, and `cell_type_key`, so these files cannot be mixed raw in one
TOML. `standardize_focus_metadata()` maps each source onto `target_gene`,
`batch`, `cell_type`, and `dataset`; batch values receive a dataset prefix.
Use these common columns in prepared files. The adapter changes metadata only
and intentionally does not apply hidden QC or alter counts.

H1's `preprocessed/h1.h5ad` concatenates its original Training, Validation, and
Test files, so it is not a default training source. Validation is opt-in via
`include_h1_validation=True`. `inspect_focus_datasets()` validates paths and
schemas using HDF5 metadata only, which is safe for the 195-GiB HCT116 file.
It also reports `ready_for_state`; the current local source files do not yet
contain `obsm["X_state"]`, so SE transformation is still a required preparation
step before any training command can be launched.

The local SE-600M `protein_embeddings.pt` contains 19,790 vectors of dimension
5,120. Direct symbol matching gives the following initial coverage (before
alias resolution): H1 expression 98.3% / targets 99.3%, K562 expression
94.7--95.2% / targets 97.6--98.4%, HCT116 expression 50.5% / targets 99.0%, and
VCC output genes 98.1%. The low HCT116 expression coverage is largely expected
from its 38,606-feature panel containing non-protein-coding features. Do not
silently replace arbitrary missing embeddings with one shared unknown vector.
The checked-in `assets/vcc_2026_se_fallback_genes.txt` defines exactly the 343
VCC-required output genes missing from SE. These genes alone receive stable
fallback IDs; other datasets' remaining missing genes stay excluded. All 300
VCC perturbation targets are present in the SE vocabulary.

## Model contract

Use `model=state_vcc`. `PanelFreeGeneDecoder` implements

```text
concat(cell_projection(H_perturbed), gene_projection(e_gene),
       matched_control_gene_log1p_CP10K)
    -> shared MLP -> signed delta
    -> matched control baseline + delta -> predicted log1p(CP10K)
```

Concatenation is used in `state_vcc` so cell and gene channels need not share
the same coordinate system. The additional scalar is deliberately the matched
control expression of that gene, not the perturbed cell's unknown raw library
depth. This follows SE's general concatenate-conditioning pattern while using
an inference-available covariate. `PanelFreeGeneDecoder` retains the older
add/direct modes for checkpoint compatibility.

No parameter depends on panel size. The same decoder can therefore train on
native panels and query all VCC genes at inference. The decoder evaluates genes
in chunks to avoid materializing a full cell-by-gene-by-hidden tensor.

### Trainable perturbation identity

The original local ST-SE-Tahoe checkpoint has a 1,138-dimensional perturbation
input and a trainable `pert_encoder` mapping it to the 768-dimensional ST hidden
space. That is consistent with the original categorical/one-hot treatment; it
does not contain the 5,120-dimensional SE protein vectors used by this genetic
transfer adaptation.

For VCC, the adapted perturbation path is deliberately hybrid:

```text
fixed 5120-D protein vector -> trainable pert_encoder -> 768-D semantic term
stable perturbation ID      -> trainable embedding   -> 768-D identity residual
                                                sum  -> ST perturbation token
```

The identity table is zero-initialized, so loading Tahoe starts from the
semantic path rather than random target offsets. It then receives gradients
jointly with `pert_encoder`. Control uses ID `-1` and always adds exactly zero.
An unknown non-control target raises an error instead of sharing an accidental
fallback ID. `assets/vcc_trainable_perturbations.txt` currently contains the
sorted union of 1,484 non-control targets in prepared H1, K562 GWPS, HCT116,
and HEK293T, including all 300 VCC targets. Regenerate it after changing the
training corpus:

```bash
PYTHONPATH=src external/state-env/bin/python \
  scripts/build_vcc_perturbation_registry.py
```

The ordered names are also saved in the model checkpoint hyperparameters and
the run's `data_module.torch`. At inference, prefer
`model.trainable_perturbation_to_id[target]`; do not pair a checkpoint with a
newly reordered registry.

For the 343 configured VCC genes without protein embeddings, the decoder
replaces the zero input placeholder after `gene_projection` with a shared
learnable base plus a stable per-gene 256-D residual. This adds 88,064 trainable
parameters. A configured fallback gene remains a measured, loss-bearing target
when it occurs in H1/K562/HCT116; it is not treated as padding or zero
expression. Of the 343 genes, 311 occur in at least one of those focus panels;
the remaining 32 retain their initialized fallback identity unless another
training dataset supplies supervision.

Use `data=vcc_panel_free` after writing standardized/embedded files. Its custom
data module reuses cell-load's set sampler, samples one native-panel gene subset
per cell set, normalizes with the *full native-panel* library size, and supplies
`gene_embeddings`, `gene_targets`, and `gene_mask`. The model raises an error if
this supervision is absent rather than silently training only the latent loss.

With the default `basal_mapping_strategy=batch`, one set contains 256 cells from
one underlying dataset and one `(batch, cell_type, perturbation)` group. A
training meta-batch contains `training.batch_size` such sets and may mix H1,
K562, and HCT116. ST attention remains within each set. Decoder genes are
sampled per set; panels are padded across the meta-batch and padding is masked.
Groups smaller than 256 are sampled with replacement unless `drop_last` is set.

CP10K is the decoder target, not the SE input transform. Source `X` stays as raw
integer counts. The official SE loader recognizes raw counts and uses its own
raw-count/log1p and expression-weight sampling path to produce `X_state`.
During ST collation, decoder targets are explicitly computed as
`log1p(raw_gene_count / full_native_panel_total * 10000)`. Matched control
counts are normalized independently with their own full-panel totals; their
Set mean becomes both a decoder input and the residual anchor.

## Repository layout

All commands should be run from the repository root. Large immutable resources
remain in their original storage locations and are exposed through links:

```text
state/
├── data/
│   ├── raw -> /sde/vcc/Data
│   ├── vcc_2026_controls -> /sde/vcc/vcc2026/vcc_2026_controls
│   └── prepared/                 # standardized/QC'd/X_state files
├── external/
│   ├── SE-600M -> .../Maojianhan/external/SE-600M
│   └── state-env -> .../miniconda3/envs/state
├── runs/                         # training and inference outputs
├── docs/
├── src/
└── tests/
```

Configurations and documentation use these repository-local paths. Do not copy
the 379-GiB raw collection or 12-GiB SE assets into Git.

Set `model.kwargs.init_from` to the Tahoe checkpoint for a real run. Tahoe's
drug one-hot perturbation encoder is not a target-gene encoder: the existing
shape-filtered checkpoint loader will skip/reinitialize it when `pert_dim`
changes. The ST backbone and compatible projections remain transferable.

The first run should keep SE offline/frozen. Warm up the new perturbation
encoder and decoder before unfreezing ST at a smaller learning rate. The
`state_vcc` configuration defines distinct Adam groups for the pretrained
backbone, perturbation encoder, decoder, and fallback table. For warm-up, set
`model.kwargs.freeze_pretrained_backbone=true`; for joint fine-tuning set it
back to false. The default group learning rates are respectively `5e-6`,
`5e-5`, `1e-4`, and `1e-4`.

## Preparing deterministic smoke data

The preparation command below reads the linked sources without modifying them,
selects three supported perturbations per source with batch-matched controls,
applies HCT116's `pass_guide_filter`, converts dense K562 matrices to CSR,
standardizes metadata, loads SE once, writes `X_state`, and creates a TOML with
one held-out validation perturbation per source:

```bash
external/state-env/bin/python scripts/prepare_vcc_focus_data.py \
  --output-dir data/prepared/smoke \
  --perturbations 3 \
  --cells-per-group 32 \
  --run-se \
  --se-batch-size 64
```

The resulting training manifest is `data/prepared/smoke/vcc_smoke.toml`.
These files are an interface/gradient smoke test, not a statistically useful
training collection.

## Disk-conscious production subsets

The extended preparation script provides two additional modes. Local sources
are linked under `data/raw`; outputs and selection reports stay in this
repository. The first command builds H1 Training, K562 GWPS, and HCT116:

```bash
external/state-env/bin/python scripts/prepare_vcc_focus_data.py \
  --mode focused --output-dir data/prepared/focused --dry-run
external/state-env/bin/python scripts/prepare_vcc_focus_data.py \
  --mode focused --output-dir data/prepared/focused
```

It retains every VCC target actually observed in a source, up to 256 cells per
target. Missing targets appear explicitly in `selection_report.csv`, rather
than being fabricated or treated as control. Additional perturbations require
a nonzero SE protein embedding and at least 128 eligible cells; the highest
cell-count targets are selected, up to 500, with up to 128 cells per target.
HCT116 additionally requires `pass_guide_filter`. These are eligibility proxies,
not a claim that knockdown efficiency has been experimentally validated.

Controls are capped at 128 per selected batch. A perturbation cell is eligible
only if its source batch contains controls. Sampling is deterministic and
keeps native gene panels intact: CP10K must use the full measured panel's
library size. `X` contains raw integer-valued counts, stored as CSR; prepared
files do not duplicate X into a counts layer.

Keep `drop_last=false` for these capped subsets. The Set sampler fills groups
smaller than `cell_sentence_len` by sampling cells with replacement; increasing
Set length does not increase the number of independent measured cells.

HEK293T uses its local guide-call table to choose barcodes, then reads selected
columns directly from remote Hugging Face Parquet files through range requests:

```bash
external/state-env/bin/python scripts/prepare_vcc_focus_data.py \
  --mode hek293t-stream --output-dir data/prepared/hek293t --dry-run
external/state-env/bin/python scripts/prepare_vcc_focus_data.py \
  --mode hek293t-stream --output-dir data/prepared/hek293t
```

The default `--hek-max-shards 12` first covers all 300 targets and then adds
shards that improve target depth toward 64 cells per target. Coverage takes
priority if more than the requested number is required. Three remote shards
are read concurrently. Only selected rows are cached under `stream_cache/`,
with a barcode-selection hash so interrupted downloads can resume safely.
Complete remote Parquet files are never cached on disk. Additional
targets are ranked by their **global** guide-QC cell counts, with only their
cells from selected shards materialized. Consequently HEK target depths can
be below 128 and are recorded separately in `hek293t_selection_report.csv`.
Increase `--hek-max-shards` when greater depth is needed; this increases network
traffic and materialized subset size. Remote expression must still pass the
raw-count check before a file is written.

Run either command with `--run-se --se-batch-size 128` to additionally compute
frozen SE embeddings. Completed raw subsets are reused, while the SE pass
writes `.xstate.h5ad` files and embedded TOML manifests. Without `--run-se`, the
`*_unembedded.toml` manifests are preparation inventories and cannot yet be
used by the ST trainer. Existing files are reused by default; use a new output
directory when changing selection parameters, or explicitly `--overwrite`.

The source-preparation manifests include all selected targets in training and
keep HEK separate from the primary three contexts. They are inventories rather
than leakage-free model-selection configurations. None of the preparation
commands starts ST training.

Ready-to-run combined manifests are provided under `configs/vcc/`:

- `vcc_h1_loco.toml` trains on K562 GWPS, HCT116, and HEK293T perturbations,
  holds out all H1 perturbed cells for validation, and retains H1 controls as
  available basal cells.
- `vcc_all_train.toml` trains on all four datasets for the final fit.

The staged launcher uses conservative one-Set microbatches with gradient
accumulation and never resumes an optimizer across the frozen/unfrozen
boundary:

```bash
# Inspect the exact command first.
external/state-env/bin/python scripts/train_vcc.py h1-loco-warmup --gpu 0 --dry-run

# Stage A: Tahoe backbone frozen; new perturbation/decoder parameters warm up.
external/state-env/bin/python scripts/train_vcc.py h1-loco-warmup --gpu 0

# Stage B: initialize a new run from Stage A best.ckpt and jointly fine-tune.
external/state-env/bin/python scripts/train_vcc.py h1-loco-joint --gpu 0

# Final fit on all four sources, initialized from the selected LOCO checkpoint.
external/state-env/bin/python scripts/train_vcc.py full --gpu 0
```

Validation uses 256 balanced natural H1 Sets per check instead of expanding
all thousands of small `(batch, perturbation)` groups. The all-train run has no
validation by design; use its `final.ckpt`, after selecting the schedule on the
LOCO run. H1 controls remain available during LOCO, but their control-only
training subset is capped at 64 Sets per epoch so dataset balancing cannot make
control-to-control examples consume 25% of training. The wrapper currently enforces one GPU because the custom balanced
sampler is not rank-sharded safely for multi-GPU training.

After both preparation modes finish, validate all stored count values, target
counts against the selection reports, full gene panels, and batch controls:

```bash
external/state-env/bin/python scripts/validate_vcc_focus_data.py
external/state-env/bin/python scripts/audit_vcc_training_interfaces.py \
  data/prepared/focused/*.xstate.h5ad
```

The checked results are written to `reports/vcc_focused_validation.json`.

## Inference

`predict_vcc_counts` accepts 400 control SE embeddings, one semantic target
embedding (or one per cell), gene embeddings in the exact official VCC
order. `build_gene_query_features` constructs the full `[18533, 5120]` query
matrix plus its fallback-ID vector; pass the latter as
`query_gene_fallback_ids`. The residual decoder also requires
`query_gene_baseline`: the official context-control mean log1p(CP10K), in the
same 18,533-gene order. It:

1. runs ST in cell chunks of at most 256;
2. queries genes in configurable chunks;
3. converts decoded log1p(CP10K) values to proportions;
4. samples integer counts at the supplied context-specific library sizes.

The caller remains responsible for choosing/calibrating the 400 library sizes,
writing a sparse AnnData matrix with no explicit zeros, and running the VCC
submission validator. Passing `concentration` enables a basic
Dirichlet-multinomial over-dispersion model; it should be calibrated on official
control cells rather than selected arbitrarily.

### Official-control pooling and submission alignment

Prepare the official A/B/C controls once:

```bash
CUDA_VISIBLE_DEVICES=0 external/state-env/bin/python \
  scripts/prepare_vcc_inference_controls.py --run-se --se-batch-size 128
```

For each context this follows the local `Virtual-Cell-Challenge-2026`
reference contract: sample 1,600 of the 18,400 controls without replacement,
stable-sort the selected donors by raw library depth, and reshape them into
400 adjacent groups of four. The assigned output depth is the rounded mean of
the four donor depths. The same deterministic grouping is reused for all 300
perturbations in that context.

`predict_vcc_pooled_counts` keeps the four real donor `X_state` vectors
separate through ST, converts each decoded profile to gene probabilities, and
only then averages the four probabilities. This avoids feeding an averaged SE
latent state that was absent from training while retaining the reference
method's noise reduction and depth-stratified 400-cell output. Integer counts
are sampled at the exact pooled depths. `VCCPredictionWriter` writes one
`[400,18533]` target block at a time in context-major, target-major official
order, so the full `[360000,18533]` matrix never needs to be dense in memory.

The per-target core call is:

```python
pool = np.load(f"data/prepared/vcc_inference/context_{context}.pool.npz")
controls = ad.read_h5ad(
    f"data/prepared/vcc_inference/context_{context}.xstate.h5ad"
)
counts = predict_vcc_pooled_counts(
    model,
    torch.from_numpy(controls.obsm["X_state"]).to(device),
    target_embedding.to(device),
    vcc_gene_embeddings.to(device),
    torch.from_numpy(pool["donor_indices"]).to(device),
    torch.from_numpy(pool["library_sizes"]).to(device),
    perturbation_ids=model.trainable_perturbation_to_id[target],
    query_gene_fallback_ids=vcc_fallback_ids.to(device),
    query_gene_baseline=torch.from_numpy(pool["log_cp10k_baseline"]).to(device),
)
```

Append the resulting `[400,18533]` blocks using `VCCPredictionWriter`, with
the loop ordered as context A/B/C outside and the official 300-target order
inside. Keep `concentration=None` for the first run; add Dirichlet
over-dispersion only after it is calibrated rather than guessed.

The complete streaming command is:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src external/state-env/bin/python \
  scripts/run_vcc_inference.py \
  --run-dir runs/vcc_full \
  --checkpoint final.ckpt \
  --output runs/vcc_full/prediction.h5ad \
  --pack \
  --vcc-output runs/vcc_full/prediction.vcc \
  --scratch-dir runs/vcc_full/pack_scratch
```

The script loads the checkpoint and its saved perturbation/fallback registries,
queries genes in official order, predicts the four selected control donors
before pooling, writes raw integer counts block-by-block, and runs `vcc prep
--dry-run` before creating the `.vcc`. Packaging sets `TMPDIR` to the supplied
scratch directory; the system `/tmp` is too small on this machine. Use
`--limit-targets 1` without `--pack` for a post-training inference smoke test.

## Executable walkthrough

`examples/vcc_toy_training_walkthrough.ipynb` walks through one complete mixed
H1/K562/VCC-control training step with tensor annotations, forward losses,
backward gradients, and an optimizer update. It uses random toy weights only.
The same logic is available as a non-notebook smoke test:

```bash
external/state-env/bin/python examples/vcc_toy_training.py
```
