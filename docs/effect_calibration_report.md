# Perturbation-effect calibration

The short calibration runs are diagnostic only; neither is a submission run.
The perturbation representation is intentionally unchanged: the existing
protein-semantic perturbation encoder plus its trainable residual rows.

| run | H1 panel | PDS cosine | note |
|---|---:|---:|---|
| original 8k best + old mean baseline | 21/144 | 0.5095 | control baseline confounded |
| original 8k best + donor-row baseline | 18 shared/non-zero | **0.5752** | calibration only |
| 20-step auxiliary-loss probe, mean baseline | 18 shared/non-zero | 0.5654 | legacy identity-mode diagnostic; weights too aggressive |
| 20-step auxiliary-loss probe, donor-row baseline | 18 shared/non-zero | 0.5196 | legacy identity-mode diagnostic; discarded |
| 20-step low-aux probe, donor-row baseline | 18 shared/non-zero | **0.5752** | legacy identity-mode diagnostic; coefficients retained only |
| 20-step low-aux probe, unchanged semantic encoder | 2 shared/non-zero | 0.5000 | too few residual rows updated for a reliable PDS estimate |

The current configuration therefore uses donor-specific log1p(CP10K) baselines,
paired control-calibrated decoding, deterministic decoder dropout, and the
conservative auxiliary weights in `configs/model/state_vcc.yaml`. H1 evaluation
defaults to the intersection of H1 targets with non-H1 training targets and
requires a non-zero learned perturbation row (the existing residual row in the
active semantic configuration).

The pipeline gate is set to `pds_cosine >= 0.60`; the best complete 18-target
diagnostic (`0.5752`) remains below the bar for expensive full VCC inference
and packaging.  The active semantic 20-step probe has only two valid targets,
so it is not used to pass or fail that gate.

These scores are local diagnostics only. The short unchanged-semantic run
updated only two of the 18 shared residual rows, so its `0.5000` PDS is not a
meaningful model-quality estimate. The retained weights need a longer run and
must first be evaluated with the same shared/non-zero gate before any VCC
inference or packaging.
