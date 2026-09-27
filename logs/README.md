# Runtime logs

`scripts/vcc_pipeline.py` creates timestamped, append-only logs here, grouped
under `training/`, `inference/`, `pipeline/`, and `submission/`. Generated log
files are ignored by Git. Machine-readable pipeline state and event history are
stored under `runs/pipelines/<pipeline-id>/`.

Do not place API tokens or copied VCC credential files in this directory. The
pipeline calls the installed `vcc` CLI, which uses its own credential store.
