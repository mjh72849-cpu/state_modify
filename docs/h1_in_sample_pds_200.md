# H1-only perturbation discrimination diagnostic (200 steps)

This is an **in-sample diagnostic**, not a LOCO or VCC generalization estimate.
It asks whether the current ST + calibrated panel-free decoder can separate
perturbations that were explicitly sampled during training.

- Source: `data/prepared/focused/arc_h1_train.focused.xstate.h5ad` only.
- Split: `configs/vcc/vcc_h1_only_effect_pds_200.toml`; all H1 perturbation
  cells and controls are in the training partition. No validation partition.
- Targets: the 21 H1 targets listed in `assets/vcc_h1_shared_21.txt`.
- Training: 200 optimizer steps, four 256-cell Sets per step, one fixed 1024-gene
  H1 panel excluding official VCC target genes, control-centered PDS retrieval
  with a 64-entry real-effect memory bank plus 0.1 log(CP10K) MSE.
- Initialization: ST-SE-Tahoe zero-shot checkpoint; identity perturbation table
  and panel-free decoder trained from scratch. The pretrained ST backbone was
  trainable at 1e-5; perturbation table at 5e-5 and decoder at 2e-5.
- Replaying the deterministic Set sampler over 200 steps found **all 21 targets**
  sampled, 23-48 Sets each (800 Sets total).
- Evaluation: final checkpoint, 400 predicted cells at most per target,
  cell-eval2 `vcc2026` preset on the same H1 source. This is deliberately
  optimistic because training and evaluation use the same source cells.

| cell-eval2 mean metric | H1-only 200-step |
| --- | ---: |
| PDS cosine | 0.5167 |
| PDS L1 | 0.5071 |
| PDS L2 | 0.5000 |
| Delta Pearson | 0.0124 |
| Expression MAE | 0.2168 |

Source metrics: `runs/vcc_h1_effect_pds_200/h1_in_sample_metrics/agg_results.csv`.
The PDS scores are still close to 0.5 despite repeated in-sample exposure;
thus the earlier poor LOCO result cannot be explained solely by missing target
exposure or by mixing gene panels across datasets. A useful next diagnostic is
a much smaller two-target overfit experiment, checking predicted-versus-real
effect vectors and the exact evaluation transform before scaling training.

The training command was:

```bash
PYTHONPATH=src external/state-env/bin/python scripts/train_vcc.py full \
  --name vcc_h1_effect_pds_200 \
  --toml configs/vcc/vcc_h1_only_effect_pds_200.toml \
  --data-config vcc_panel_free_effect_pds_focused \
  --model-config state_vcc_effect_pds_focused \
  --gpu 2 \
  --init-from external/ST-SE-Tahoe/zeroshot/state_generalization_zeroshot_X_state/checkpoints/best.ckpt \
  --max-steps 200 --val-freq 50 --batch-size 4 \
  --gradient-accumulation 1 --num-workers 2
```

Historical runs retained only as a comparison summary after cleanup:

| Earlier run | H1 PDS cosine | Note |
| --- | ---: | --- |
| `vcc_paper_smoke_40_v2` | 0.5000 | 21-target H1 panel |
| `vcc_pds_only_500` | 0.5000 | 21-target H1 panel |
| `vcc_effect_pds_200` | 0.5000 | 21-target H1 LOCO panel |
| `vcc_h1_loco_joint_stageb-pds-8k` | 0.5095 | 21-target shared H1 panel; a separate per-cell calibration diagnostic reported 0.5752 on a different panel |

The prior runs' full predictions and redundant checkpoints were removed to
recover disk space. The last run above retains its final checkpoint because
its distinct per-cell result may still be worth revisiting.
