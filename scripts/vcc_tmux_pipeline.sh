#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/external/state-env/bin/python"
PIPELINE="$ROOT/scripts/vcc_pipeline.py"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/vcc_tmux_pipeline.sh start PIPELINE_ID [--confirm-submit] [options]
  scripts/vcc_tmux_pipeline.sh status PIPELINE_ID
  scripts/vcc_tmux_pipeline.sh attach PIPELINE_ID
  scripts/vcc_tmux_pipeline.sh tail PIPELINE_ID

Start options:
  --profile stageb-pds|warmup|paper-like|production   Workflow profile (default: stageb-pds)
  --gpu INDEX                  CUDA device (default: 2)
  --model-name NAME            VCC leaderboard model name
  --resume                     Resume checkpoints and completed pipeline steps
  --confirm-submit             Required: permit vcc submit after packaging
EOF
}

action="${1:-}"
pipeline_id="${2:-}"
if [[ -z "$action" || -z "$pipeline_id" ]]; then
  usage
  exit 2
fi
if [[ ! "$pipeline_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "PIPELINE_ID may contain only letters, digits, '-' and '_'" >&2
  exit 2
fi

session="state_${pipeline_id}"
launcher_log="$ROOT/logs/pipeline/${pipeline_id}__tmux.log"

case "$action" in
  status)
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "tmux=running session=$session"
    else
      echo "tmux=not-running session=$session"
    fi
    exec "$PYTHON" "$PIPELINE" status --pipeline-id "$pipeline_id"
    ;;
  attach)
    exec tmux attach-session -t "$session"
    ;;
  tail)
    exec tail -n 100 -f "$launcher_log"
    ;;
  start)
    shift 2
    profile="stageb-pds"
    gpu="2"
    model_name=""
    resume=false
    confirmed=false
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --profile)
          profile="${2:?--profile requires a value}"
          shift 2
          ;;
        --gpu)
          gpu="${2:?--gpu requires a value}"
          shift 2
          ;;
        --model-name)
          model_name="${2:?--model-name requires a value}"
          shift 2
          ;;
        --resume)
          resume=true
          shift
          ;;
        --confirm-submit)
          confirmed=true
          shift
          ;;
        *)
          echo "Unknown option: $1" >&2
          usage
          exit 2
          ;;
      esac
    done
    if [[ "$profile" != "stageb-pds" && "$profile" != "warmup" && "$profile" != "paper-like" && "$profile" != "production" ]]; then
      echo "--profile must be stageb-pds, warmup, paper-like, or production" >&2
      exit 2
    fi
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
      echo "--gpu must be one CUDA device index" >&2
      exit 2
    fi
    if [[ "$profile" != "stageb-pds" && "$confirmed" != true ]]; then
      echo "Refusing to schedule a network submission without --confirm-submit" >&2
      exit 2
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "tmux session already exists: $session" >&2
      exit 1
    fi
    if [[ "$profile" == "stageb-pds" ]]; then
      config="$ROOT/configs/vcc/vcc_stageb_pds.toml"
      default_model_name="state-stageb-pds-${pipeline_id}"
    elif [[ "$profile" == "warmup" ]]; then
      config="$ROOT/configs/vcc/vcc_warmup_submit.toml"
      default_model_name="state-warmup-probe-${pipeline_id}"
    elif [[ "$profile" == "paper-like" ]]; then
      config="$ROOT/configs/vcc/vcc_paper_like_submit.toml"
      default_model_name="state-paper-like-${pipeline_id}"
    else
      config="$ROOT/configs/vcc/vcc_pipeline.toml"
      default_model_name="state-production-${pipeline_id}"
    fi
    model_name="${model_name:-$default_model_name}"
    run_suffix="_${pipeline_id}"

    # Check the device before detaching. Submission profiles additionally check
    # credentials; stageb-pds intentionally stops after local .vcc packaging.
    "$PYTHON" - "$gpu" "$confirmed" <<'PY'
import json
import subprocess
import sys

gpu = int(sys.argv[1])
if sys.argv[2] == "true":
    payload = json.loads(subprocess.run(
        ["vcc", "whoami", "--json"], text=True, capture_output=True, check=True
    ).stdout)
    if not payload.get("identity", {}).get("can_submit", False):
        raise SystemExit("Current VCC profile is not allowed to submit")
try:
    import torch
    if gpu >= torch.cuda.device_count():
        raise SystemExit(f"GPU index {gpu} is outside available CUDA devices")
except ImportError:
    pass
PY

    command=(
      "$PYTHON" "$PIPELINE" run
      --pipeline-id "$pipeline_id"
      --config "$config"
      --gpu "$gpu"
      --run-suffix "$run_suffix"
      --model-name "$model_name"
    )
    if [[ "$confirmed" == true ]]; then
      command+=(--submit --confirm-submit --wait-submission)
    fi
    if [[ "$resume" == true ]]; then
      command+=(--resume)
    elif [[ -e "$ROOT/runs/pipelines/$pipeline_id/state.json" ]]; then
      echo "Pipeline state already exists; pass --resume or choose a new PIPELINE_ID" >&2
      exit 1
    fi

    # Validate the complete command graph before detaching.
    "${command[@]}" --dry-run
    mkdir -p "$(dirname "$launcher_log")"
    printf -v quoted '%q ' "${command[@]}"
    shell_command="cd $(printf '%q' "$ROOT") && exec ${quoted} >>$(printf '%q' "$launcher_log") 2>&1"
    tmux new-session -d -s "$session" "bash -lc $(printf '%q' "$shell_command")"
    echo "Started tmux session: $session"
    echo "Pipeline: $pipeline_id"
    echo "Profile: $profile"
    echo "GPU: $gpu"
    echo "Model name: $model_name"
    echo "Log: $launcher_log"
    echo "Status: scripts/vcc_tmux_pipeline.sh status $pipeline_id"
    echo "Attach: scripts/vcc_tmux_pipeline.sh attach $pipeline_id"
    ;;
  *)
    usage
    exit 2
    ;;
esac
