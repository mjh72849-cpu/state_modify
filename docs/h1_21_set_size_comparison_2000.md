# H1 21-target Set-size comparison (2,000 steps)

Both runs use exactly the same cell-disjoint H1 split from
`data/prepared/h1_holdout_21/split.json`, the same ST-SE-Tahoe zero-shot
initialization, optimizer rates, PDS retrieval objective, 1,024-gene panel,
batch-aware control matching, and four Sets per optimizer step. The only model
change is `cell_set_len`.

The train partition has 8,155 cells and the held-out partition has 2,236
cells. Evaluation uses only the held-out H1 cells and all 21 target
perturbations. It is a seen-target, cell-held-out diagnostic, not LOCO or a VCC
submission score.

| run | Set size | steps | held-out PDS cosine | PDS L1 | PDS L2 | Delta Pearson | expression MAE | targets > 0.5 | targets >= 0.75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vcc_h1_21_cross_batch_set64_2000` | 64 | 2,000 | **0.7071** | 0.5690 | 0.5571 | 0.0412 | 0.1269 | 16/21 | 12/21 |
| `vcc_h1_21_cross_batch_set128_2000` | 128 | 2,000 | **0.8452** | 0.5500 | 0.5476 | 0.0378 | 0.0862 | 18/21 | 17/21 |

Set=128 is clearly better on held-out PDS and expression MAE in this controlled
experiment. The improvement is consistent with larger Sets producing a lower
variance estimate of the perturbation effect, while the cells can still be
sampled across native experimental batches. PDS is still not perfect: HIRA,
KLF10, and NISCH remain weak or unstable, so this does not establish VCC
generalization.

Per-target PDS cosine:

| target | Set=64 | Set=128 | target | Set=64 | Set=128 |
| --- | ---: | ---: | --- | ---: | ---: |
| AKT2 | 0.45 | 0.75 | CASP3 | 1.00 | 1.00 |
| DHX36 | 1.00 | 0.95 | DNMT1 | 0.95 | 1.00 |
| HIRA | 0.35 | 0.20 | HSBP1 | 0.85 | 1.00 |
| KLF10 | 0.00 | 0.45 | MED13 | 1.00 | 1.00 |
| MTA1 | 0.65 | 1.00 | NDUFB6 | 0.80 | 1.00 |
| NISCH | 0.65 | 0.35 | RNF2 | 0.25 | 0.95 |
| SHPRH | 0.35 | 0.80 | SIN3B | 1.00 | 1.00 |
| SMARCA5 | 1.00 | 1.00 | STAT6 | 1.00 | 0.95 |
| TARBP2 | 0.70 | 0.85 | TMSB10 | 0.55 | 0.95 |
| TRAPPC6A | 0.75 | 0.90 | TWF2 | 0.80 | 0.65 |
| ZNF714 | 0.75 | 1.00 |  |  |  |

Artifacts:

- Set=64 checkpoint: `runs/vcc_h1_21_cross_batch_set64_2000/checkpoints/final.ckpt`
- Set=128 checkpoint: `runs/vcc_h1_21_cross_batch_set128_2000/checkpoints/final.ckpt`
- Set=64 metrics: `runs/vcc_h1_21_cross_batch_set64_2000/h1_21_holdout_metrics/agg_results.csv`
- Set=128 metrics: `runs/vcc_h1_21_cross_batch_set128_2000/h1_21_holdout_metrics/agg_results.csv`
- Set=64 training log: `logs/h1_21_cross_batch_set64_2000.log`
- Set=128 training log: `logs/h1_21_cross_batch_set128_2000.log`

The duplicate Lightning `last.ckpt` files were removed after completion; the
final checkpoints are retained.
