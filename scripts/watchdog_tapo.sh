#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere; resolve repo root from this script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

STATE_DIR="${REPO_ROOT}/captures"
LOG_FILE="${STATE_DIR}/tapo_watchdog.log"
IDLE_THRESHOLD_SECONDS=1800 # 30 minutes
CHECK_INTERVAL_SECONDS=60 # 1 minute
DOTENV_FILE="${REPO_ROOT}/.env"

mkdir -p "$STATE_DIR"

get_dotenv_value() {
  local key="$1"
  if [[ ! -f "$DOTENV_FILE" ]]; then
    return 0
  fi

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    local line
    line="$(echo "$raw_line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" != *=* ]] && continue

    local current_key="${line%%=*}"
    local value="${line#*=}"
    current_key="$(echo "$current_key" | sed -e 's/[[:space:]]*$//')"
    if [[ "$current_key" != "$key" ]]; then
      continue
    fi

    value="$(echo "$value" | sed -e 's/^[[:space:]]*//')"
    if [[ "${#value}" -ge 2 ]]; then
      if [[ "$value" == \"*\" && "$value" == *\" ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi
    echo "$value"
    return 0
  done <"$DOTENV_FILE"
}

is_tapo_running() {
  pgrep -f "python[0-9.]* scripts/tapo_opencv_test.py" >/dev/null 2>&1
}

get_windows_idle_seconds() {
  local idle
  idle="$(powershell.exe -NoProfile -ExecutionPolicy Bypass -File "${SCRIPT_DIR}/windows_idle_seconds.ps1" | tr -d '\r')"
  if [[ "$idle" =~ ^[0-9]+$ ]]; then
    echo "$idle"
  else
    echo "0"
  fi
}

start_tapo() {
  local watchdog_cmd
  watchdog_cmd="$(get_dotenv_value "WATCHDOG_TAPO_COMMAND")"
  if [[ -n "$watchdog_cmd" ]]; then
    nohup bash -lc "$watchdog_cmd" >>"$LOG_FILE" 2>&1 &
    echo "[$(date '+%F %T')] started command from .env: WATCHDOG_TAPO_COMMAND" >>"$LOG_FILE"
    return 0
  fi

  nohup python scripts/tapo_opencv_test.py \
    --zone-polygon "0.4158,0.5114;0.4863,0.5201;0.4814,0.6461;0.4102,0.6294" \
    --motion-threshold 1.4 \
    --process-fps 5 \
    --snapshot-cooldown 10 \
    --alert-seconds 4 \
    --save-clip-on-alert \
    --zone-edit \
    --clip-seconds 10 \
    --cat-model "models/yolov8m.pt" \
    --cat-confidence 0.08 \
    --cat-enter-frames 1 \
    --cat-hold-seconds 1.5 \
    --cat-detect-mode always \
    --cat-zone-overlap 0.3 \
    --cat-imgsz 1920 \
    --device cuda >>"$LOG_FILE" 2>&1 &
}

echo "[$(date '+%F %T')] watchdog started" >>"$LOG_FILE"

while true; do
  idle_seconds="$(get_windows_idle_seconds)"

  if is_tapo_running; then
    :
  elif (( idle_seconds >= IDLE_THRESHOLD_SECONDS )); then
    echo "[$(date '+%F %T')] windows idle ${idle_seconds}s and script not running -> start" >>"$LOG_FILE"
    start_tapo
  fi

  sleep "$CHECK_INTERVAL_SECONDS"
done
