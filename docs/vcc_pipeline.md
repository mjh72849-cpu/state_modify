# VCC pipeline operations

The repository provides one tracked pipeline for staged training, inference,
official packaging, and optional submission:

```text
H1-LOCO warm-up -> H1-LOCO joint -> four-dataset full
       -> 300-target inference -> vcc prep -> optional vcc submit
```

Configuration lives in `configs/vcc/vcc_pipeline.toml`. Runtime state is
written to `runs/pipelines/<pipeline-id>/state.json`, an append-only event
stream to `events.jsonl`, and complete command output to category-specific
files below `logs/`. Runtime files are ignored by Git.

After the low-PDS frozen warm-up probe, the recommended first rerun is
`configs/vcc/vcc_paper_like_submit.toml`. It trains the warm-up and the joint
fine-tuning stage, then infers from the joint stage's validation-selected
`best.ckpt`; it does not submit the frozen warm-up checkpoint by itself.

```bash
scripts/vcc_tmux_pipeline.sh start paper_like_v1 \
  --profile paper-like --gpu 0 --confirm-submit
```

## Inspect the production plan

This creates a local plan record but executes no training, inference, package,
or submission command:

```bash
external/state-env/bin/python scripts/vcc_pipeline.py run \
  --pipeline-id production_v1 --dry-run
```

## End-to-end smoke pipeline

Use a suffix so smoke checkpoints cannot overwrite production runs. Two steps
per stage and one official target are enough to exercise checkpoint transfer,
all three contexts, raw-count generation, H5AD writing, and debug `.vcc`
packaging. The resulting `.vcc` is deliberately not submittable.

```bash
scripts/vcc_pipeline_ctl.sh start smoke_v1 \
  --gpu 2 --run-suffix _smoke_v1 --max-steps 2 \
  --smoke-targets 1 --overwrite

scripts/vcc_pipeline_ctl.sh status smoke_v1
scripts/vcc_pipeline_ctl.sh tail smoke_v1
```

## Detached training through automatic VCC submission

For a warm-up probe that must keep running after the local computer disconnects,
use the tmux launcher. It trains only Stage A, infers from that run's
validation-selected `best.ckpt`, packages all 300 targets, submits through the
official CLI, waits for terminal scoring status, and records the submission ID:

```bash
scripts/vcc_tmux_pipeline.sh start warmup_probe_v1 \
  --profile warmup --gpu 2 \
  --model-name state-warmup-probe-v1 \
  --confirm-submit
```

Closing the SSH terminal or the local computer does not stop the remote tmux
server. The remote host itself must remain powered on.

```bash
scripts/vcc_tmux_pipeline.sh status warmup_probe_v1
scripts/vcc_tmux_pipeline.sh tail warmup_probe_v1
scripts/vcc_tmux_pipeline.sh attach warmup_probe_v1
```

If the remote process or host is interrupted, restart with the same ID and
`--resume`. Completed stages and artifacts are skipped; an incomplete training
stage resumes from `last.ckpt`:

```bash
scripts/vcc_tmux_pipeline.sh start warmup_probe_v1 \
  --profile warmup --gpu 2 --confirm-submit --resume
```

## Production training and packaging

Start in the background:

```bash
scripts/vcc_pipeline_ctl.sh start production_v1 --gpu 2
```

Resume after a host/process interruption. A completed stage is skipped; an
incomplete training stage resumes through its `checkpoints/last.ckpt`:

```bash
scripts/vcc_pipeline_ctl.sh start production_v1 --gpu 2 --resume
```

Start from or stop after a named phase when operating manually:

```bash
external/state-env/bin/python scripts/vcc_pipeline.py run \
  --pipeline-id joint_only \
  --from-stage h1-loco-joint --through-stage h1-loco-joint \
  --skip-inference --resume
```

`--overwrite` and `--resume` are mutually exclusive. Overwrite deletes the
selected training run through the existing STATE launcher, so use it only for
deliberately disposable runs.

## Submission safety

Packaging is enabled by default, but network submission is disabled. A real
submission requires both explicit flags:

```bash
external/state-env/bin/python scripts/vcc_pipeline.py run \
  --pipeline-id production_v1 --skip-training --skip-inference \
  --vcc-output runs/vcc_full/prediction.vcc \
  --submit --confirm-submit \
  --model-name state-vcc-production-v1
```

The pipeline first runs `vcc whoami`, then submits the existing `.vcc` and
records the returned entry ID from the official CLI output. The main
`logs/submission/<pipeline-id>__submission.log` preserves that human-readable
output verbatim, including upload percentage/speed/ETA, scoring status, final
rank, and metric values. Machine-readable state remains in
`runs/pipelines/<pipeline-id>/state.json`.
Authentication remains in the official VCC CLI credential store; tokens are
not copied into pipeline state or logs by this code.

Query and optionally follow the recorded submission:

```bash
external/state-env/bin/python scripts/vcc_pipeline.py track \
  --pipeline-id production_v1

external/state-env/bin/python scripts/vcc_pipeline.py track \
  --pipeline-id production_v1 --wait --poll-interval 30
```

An explicit ID can be supplied with `--entry-id` if a submission was started
on another machine.

## Local status format

```bash
external/state-env/bin/python scripts/vcc_pipeline.py status \
  --pipeline-id production_v1
```

Training, inference, packaging, and submission are separate tracked steps and
write separate logs. Each step records:

- exact command and Git commit;
- status (`planned`, `running`, `succeeded`, or `failed`);
- UTC start/end times, exit code, and elapsed seconds;
- the full log path;
- artifact path and byte size, packaged `.vcc` SHA-256, and VCC submission ID
  where applicable.

`scripts/vcc_pipeline_ctl.sh stop <pipeline-id>` sends `SIGTERM` to the
tracked pipeline process group, including its active training or inference
child. It does not delete checkpoints, logs, or partial output, so the run can
subsequently use `--resume`.
