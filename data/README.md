# Data workspace

This directory is the single data entry point for the VCC/STATE work.

- `raw` is a symbolic link to `/sde/vcc/Data`. Treat it as read-only.
- `vcc_2026_controls` is a symbolic link to the official context A/B/C control
  files. Treat it as read-only.
- `prepared` is for standardized, QC'd AnnData files containing
  `obsm["X_state"]`, plus generated TOML manifests. Its large contents are
  ignored by Git.

Do not modify files through the two read-only links. Write all derived data to
`prepared/` so raw data and generated artifacts remain distinguishable.

Create the checked H1/K562/HCT116 smoke inputs and their real frozen SE
embeddings with:

```bash
external/state-env/bin/python scripts/prepare_vcc_focus_data.py \
  --output-dir data/prepared/smoke --perturbations 3 \
  --cells-per-group 32 --run-se --se-batch-size 64
```
