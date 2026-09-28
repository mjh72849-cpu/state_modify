# H1 21-target cross-batch, cell-held-out PDS diagnostic

This is a **cell-disjoint, within-H1 diagnostic**, not a held-out-target (LOCO)
or VCC generalization score. The same perturbations and experimental batches
are represented on both sides of the split.

## Data and training

- Original source: `data/prepared/focused/arc_h1_train.focused.xstate.h5ad`.
- `scripts/prepare_h1_cell_holdout.py` selected the 21 targets in
  `assets/vcc_h1_shared_21.txt` plus non-targeting controls, then split each
  target × experimental-batch group with seed `20260928`. Original raw counts,
  the 18,080-gene panel, and the existing 2,058-dimensional `X_state` were
  preserved. No SE recomputation or additional CP10K transform was applied
  when creating the files.
- Train: 8,155 cells, including 4,896 controls; validation: 2,236 cells,
  including 1,248 controls. Each target has at least 86 train and 35 held-out
  cells. `obs_names` do not overlap; gene panels match exactly. The row-level
  manifest is `data/prepared/h1_holdout_21/split.json`.
- Training used the existing PDS-effect-retrieval profile initialized from the
  ST-SE-Tahoe zero-shot checkpoint. It used 64-cell Sets grouped across
  experimental batches, while control matching remained within each cell's
  experimental batch; four Sets per optimizer step, 400 steps, low learning
  rates (ST 1e-5, identity table 5e-5, decoder 2e-5), and 0.1-weighted
  log(CP10K) decoder MSE. The gene query panel was the fixed 1,024-gene
  supported H1 panel excluding VCC target genes. The training source contained
  only train cells; no Lightning validation source was added, so cell-load did
  not silently feed held-out controls into training.
- Configs: `configs/vcc/vcc_h1_21_cell_holdout.toml`,
  `src/state/configs/data/vcc_h1_21_cross_batch.yaml`, and
  `src/state/configs/model/state_vcc_h1_21_cross_batch.yaml`.

## Result

Scored by cell-eval2 `vcc2026`/`anndata` on **all 21 held-out targets** and
the same held-out real reference. The evaluator read integer counts, used the
real control pool, and excluded each target gene from its PDS comparison.
There is no target-selection based on learned embedding magnitude.

| Metric, mean across 21 targets | Model, held-out cells | Training-cell empirical reference vs held-out cells |
| --- | ---: | ---: |
| PDS cosine | **0.6524** | 0.9952 |
| PDS L1 | 0.6143 | 0.9333 |
| PDS L2 | 0.6095 | 0.9762 |
| Delta Pearson | 0.0382 | 0.5057 |
| Expression MAE | 0.1558 | 0.0428 |

The empirical reference supplies the **observed training perturbation cells**
as predictions; it is a signal-reproducibility control, not a deployable
control-to-perturbation predictor. The ~0.5 PDS rank is the chance/tie level.
The model is above that level, but much weaker than the measured H1 signal.

| Target | Model PDS cosine | Target | Model PDS cosine | Target | Model PDS cosine |
| --- | ---: | --- | ---: | --- | ---: |
| AKT2 | 0.80 | CASP3 | 0.90 | DHX36 | 1.00 |
| DNMT1 | 0.90 | HIRA | 0.20 | HSBP1 | 1.00 |
| KLF10 | 0.15 | MED13 | 1.00 | MTA1 | 0.75 |
| NDUFB6 | 0.45 | NISCH | 0.55 | RNF2 | 0.95 |
| SHPRH | 0.60 | SIN3B | 1.00 | SMARCA5 | 0.85 |
| STAT6 | 0.75 | TARBP2 | 0.35 | TMSB10 | 0.40 |
| TRAPPC6A | 0.45 | TWF2 | 0.00 | ZNF714 | 0.65 |

Fourteen of 21 targets exceed 0.5; seven are at or below it. The single
logged training minibatch surrogate fell from 2.646 at step 50 to 2.072 at
step 400, but fluctuated substantially and is **not** the held-out PDS.
The model queried 18,077/18,080 H1 genes at inference. The remaining three
unsupported legacy symbols retained matched-control expression; no targets
or cells were dropped.

This result supports the earlier diagnosis that repeating a few batch-local
cells to fill a Set was harmful, but it does not establish that Set grouping
was the only issue. With 21 targets, the model's effect direction is still
weak (Delta Pearson 0.038); longer training, better effect supervision, and
evaluation-transform checks remain candidate next experiments. Do not infer
VCC performance from this within-H1, seen-target test.

Artifacts:

- Model: `runs/vcc_h1_21_cross_batch_holdout_400/checkpoints/final.ckpt`
- Training log: `logs/h1_21_cross_batch_holdout_400.log`
- Prediction log: `logs/h1_21_holdout_evaluation.log`
- Official-model metrics:
  `runs/vcc_h1_21_cross_batch_holdout_400/h1_21_holdout_metrics/`
- Empirical-reference metrics: `runs/h1_21_train_cell_reference_metrics/`

Reproduction commands:

```bash
PYTHONPATH=src external/state-env/bin/python scripts/prepare_h1_cell_holdout.py
PYTHONPATH=src external/state-env/bin/python scripts/train_vcc.py full \
  --name vcc_h1_21_cross_batch_holdout_400 \
  --toml configs/vcc/vcc_h1_21_cell_holdout.toml \
  --data-config vcc_h1_21_cross_batch \
  --model-config state_vcc_h1_21_cross_batch --gpu 2 \
  --init-from external/ST-SE-Tahoe/zeroshot/state_generalization_zeroshot_X_state/checkpoints/best.ckpt \
  --max-steps 400 --val-freq 100 --batch-size 4 \
  --gradient-accumulation 1 --num-workers 2
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src external/state-env/bin/python scripts/evaluate_h1_loco.py \
  --run-dir runs/vcc_h1_21_cross_batch_holdout_400 --checkpoint final.ckpt \
  --data data/prepared/h1_holdout_21/h1_21_val.xstate.h5ad \
  --targets-file assets/vcc_h1_shared_21.txt --no-shared-nonzero-only \
  --output runs/vcc_h1_21_cross_batch_holdout_400/h1_21_holdout_prediction.h5ad \
  --real-output runs/vcc_h1_21_cross_batch_holdout_400/h1_21_holdout_real.h5ad \
  --device cuda:0 --max-cells-per-pert 400
PYTHONPATH=/sde/vcc/vcc2026/cell-eval2/src external/state-env/bin/python \
  -c 'from cell_eval2.cli import main; main()' run \
  -ap runs/vcc_h1_21_cross_batch_holdout_400/h1_21_holdout_prediction.h5ad \
  -ar runs/vcc_h1_21_cross_batch_holdout_400/h1_21_holdout_real.h5ad \
  --preset vcc2026 --profile anndata --pert-col target_gene \
  -o runs/vcc_h1_21_cross_batch_holdout_400/h1_21_holdout_metrics
```
