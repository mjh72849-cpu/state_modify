# H1 AKT2/SIN3B overfit diagnostic

This experiment tests whether the existing identity-embedding → ST → calibrated
gene-conditioned decoder can distinguish two **training-seen** perturbations.
It is not a held-out-cell, LOCO, or VCC generalization result.

Both targets have 256 H1 cells. The decoder uses a fixed 1024-gene panel that
excludes official VCC target genes. All ST runs initialize from the same
ST-SE-Tahoe zero-shot checkpoint and use the same H1-only manifest, control
matching, 64-cell Sets, batch size 4, effect-retrieval loss, and 0.1 decoder
MSE. The only ablations are Set grouping and optimizer learning rates.

| run | Set grouping | median unique perturbed cells/Set | LR (ST / identity / decoder) | steps | final PDS surrogate | H1 in-sample PDS cosine |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `vcc_h1_two_target_batch_200` | within experimental batch | 4 | 1e-4 / 1e-3 / 5e-4 | 200 | 0.7707 | 0.0 |
| `vcc_h1_two_target_low_lr_200` | across batches, with per-cell matched-batch controls | 64 | 1e-5 / 5e-5 / 2e-5 | 200 | 0.00060 | 1.0 |
| `vcc_h1_two_target_st_300` | across batches, with per-cell matched-batch controls | 64 | 1e-4 / 1e-3 / 5e-4 | 300 | 0.00057 (0.00077 at step 200) | 1.0 |

The two-target random-retrieval cross entropy is `ln(2) ≈ 0.693`. The
within-batch run did not learn to rank even these two perturbations, whereas
both cross-batch runs did. Because the high-LR pair differs only in Set
grouping at step 200, and the low-LR cross-batch run also succeeds, the Set
construction is the leading cause of this **two-target** failure; the old low
learning rates are not sufficient to explain it.

An independent two-row `nn.Embedding` → 1024-gene effect lookup control was
trained on half of each target's cells and tested on the other half, with
disjoint 400-control pools. It reached 100% two-target retrieval accuracy by
step 10. Its output is in `runs/vcc_h1_two_target_lookup/metrics.json`.

The full ST result was scored by cell-eval2's `vcc2026` anndata preset using
the same real H1 reference for all three runs. PDS cosine is a binary rank
with only two targets, so 0.0/1.0 is expected and should not be interpreted
as competition performance. The full model trained on all H1 cells; only the
lookup control has a cell-disjoint test split. Expression MAE remains similar
across the three ST runs (~0.063-0.069), illustrating why expression error
alone did not reveal the discrimination failure.

Implementation and reproducibility:

- `src/state/tx/vcc/data_module.py`: `set_group_by_batch=false` decouples Set
  grouping from the unchanged `BatchMappingStrategy` for matched controls.
- `src/state/configs/data/vcc_h1_overfit_2.yaml`: cross-batch 64-unique-cell
  Set data profile; `_batch_grouped` is the comparator.
- `src/state/configs/model/state_vcc_h1_overfit_2.yaml`: high-LR 64-cell
  profile; `_low_lr` restores the earlier optimizer rates.
- `scripts/diagnose_h1_two_target_lookup.py`: small identity-effect control.
- `runs/vcc_h1_two_target_st_300/h1_two_metrics/agg_results.csv` and the
  analogous `_batch_200` and `_low_lr_200` metrics directories contain the
  cell-eval2 outputs. CSV training logs now include the **actual optimized**
  `train/total_loss` and `train/pds_surrogate_loss`; `train_loss` is only the
  detached latent diagnostic in this PDS-only profile.

Next test: extend the cross-batch grouping to the 21-target H1 diagnostic,
first using the original low learning rates and 64/128-cell Sets; split cells
by target into train and validation before interpreting PDS. A successful
two-target overfit proves the computational path works, not that the 21-target
or VCC problem is solved.

Disk cleanup: the two successful runs retain `final.ckpt`; the failed
within-batch run retains only its configuration, metrics, and log. Duplicate
`last.ckpt` files and generated prediction/reference H5AD files were removed.
Those H5AD files are reproducible from the H1 source and retained successful
checkpoints; the failed comparator checkpoint would require retraining.
