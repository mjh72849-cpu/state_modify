#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/external/state-env/bin/python"

usage() {
  echo "Usage: $0 start PIPELINE_ID [pipeline run args...]" >&2
  echo "       $0 status PIPELINE_ID" >&2
  echo "       $0 tail PIPELINE_ID" >&2
  echo "       $0 stop PIPELINE_ID" >&2
}

action="${1:-}"
pipeline_id="${2:-}"
if [[ -z "$action" || -z "$pipeline_id" ]]; then
  usage
  exit 2
fi
if [[ ! "$pipeline_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Invalid pipeline ID" >&2
  exit 2
fi

state_dir="$ROOT/runs/pipelines/$pipeline_id"
launcher_log="$ROOT/logs/pipeline/${pipeline_id}.log"
pid_file="$state_dir/launcher.pid"

case "$action" in
  start)
    shift 2
    mkdir -p "$state_dir" "$(dirname "$launcher_log")"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      echo "Pipeline is already running with PID $(cat "$pid_file")" >&2
      exit 1
    fi
    nohup setsid "$PYTHON" "$ROOT/scripts/vcc_pipeline.py" run \
      --pipeline-id "$pipeline_id" "$@" >>"$launcher_log" 2>&1 &
    pid=$!
    echo "$pid" >"$pid_file"
    echo "Started pipeline=$pipeline_id pid=$pid log=$launcher_log"
    ;;
  status)
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      echo "process=running pid=$(cat "$pid_file")"
    else
      echo "process=not-running"
    fi
    "$PYTHON" "$ROOT/scripts/vcc_pipeline.py" status --pipeline-id "$pipeline_id"
    ;;
  tail)
    test -f "$launcher_log"
    tail -n 100 -f "$launcher_log"
    ;;
  stop)
    test -f "$pid_file"
    pid="$(cat "$pid_file")"
    kill -TERM -- "-$pid"
    echo "Sent SIGTERM to pipeline=$pipeline_id process_group=$pid"
    ;;
  *)
    usage
    exit 2
    ;;
esac
