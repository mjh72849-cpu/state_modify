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
records the returned entry ID when the installed CLI returns it as JSON.
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
