#!/usr/bin/env bash
# divvy-observer local runner — foreground only.
#
# Usage:
#   ./run.sh             Start collector + automation + API + dashboard in
#                        this terminal. Ctrl+C stops everything.
#   ./run.sh stop        Kill anything left running on the divvy ports and
#                        unload any divvy LaunchAgent plists.
#   ./run.sh status      Show what (if anything) is currently running.
#   ./run.sh --help      Show this message.
#
# Notes:
#   - macOS LaunchAgents are NOT used by this script. If you previously ran
#     `uv run divvy start` (or `./run.sh --launchd` in the old version), the
#     plists are still in ~/Library/LaunchAgents and will auto-restart on
#     login. Use `./run.sh stop` to unload + remove them.

set -euo pipefail

cd "$(dirname "$0")"

LAUNCH_AGENT_DIR="${HOME}/Library/LaunchAgents"
LAUNCH_AGENT_LABELS=(divvy.collector divvy.automation divvy.api divvy.dashboard)
PORT_API=8000
PORT_DASHBOARD=8501

print_help() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
}

# --- helpers ----------------------------------------------------------------

require_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required. install with: brew install uv" >&2
    exit 1
  fi
}

loaded_launch_agents() {
  # Echoes labels of any divvy LaunchAgents currently loaded.
  local label
  for label in "${LAUNCH_AGENT_LABELS[@]}"; do
    if launchctl list 2>/dev/null | awk '{print $3}' | grep -Fxq "$label"; then
      echo "$label"
    fi
  done
}

unload_launch_agents() {
  local any=0
  local label plist
  for label in "${LAUNCH_AGENT_LABELS[@]}"; do
    plist="${LAUNCH_AGENT_DIR}/${label}.plist"
    if [[ -f "$plist" ]]; then
      any=1
      echo "==> unloading and removing ${label}.plist"
      launchctl bootout "gui/$(id -u)/${label}" 2>/dev/null \
        || launchctl unload "$plist" 2>/dev/null \
        || true
      rm -f "$plist"
    fi
  done
  # Also clean up the older thoughtbison-prefixed collector plist if present.
  local legacy="${LAUNCH_AGENT_DIR}/net.thoughtbison.divvy-collector.plist"
  if [[ -f "$legacy" ]]; then
    any=1
    echo "==> unloading and removing legacy net.thoughtbison.divvy-collector.plist"
    launchctl bootout "gui/$(id -u)/net.thoughtbison.divvy-collector" 2>/dev/null \
      || launchctl unload "$legacy" 2>/dev/null \
      || true
    rm -f "$legacy"
  fi
  if [[ "$any" -eq 0 ]]; then
    echo "==> no divvy LaunchAgent plists found"
  fi
}

pids_on_port() {
  lsof -ti:"$1" 2>/dev/null || true
}

kill_ports() {
  local port pids
  for port in "$PORT_API" "$PORT_DASHBOARD"; do
    pids="$(pids_on_port "$port")"
    if [[ -n "$pids" ]]; then
      echo "==> killing PID(s) on :${port}: ${pids}"
      # shellcheck disable=SC2086
      kill -TERM $pids 2>/dev/null || true
      sleep 1
      pids="$(pids_on_port "$port")"
      if [[ -n "$pids" ]]; then
        # shellcheck disable=SC2086
        kill -KILL $pids 2>/dev/null || true
      fi
    fi
  done
}

kill_divvy_processes() {
  # Catch any stray collector / automation / streamlit / uvicorn for divvy
  # that aren't bound to a port (e.g. background workers).
  local pids
  pids="$(pgrep -f 'divvy\.collector|divvy\.automation|divvy\.api|src/divvy/dashboard\.py' 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "==> killing divvy worker PID(s): ${pids}"
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
    sleep 1
    pids="$(pgrep -f 'divvy\.collector|divvy\.automation|divvy\.api|src/divvy/dashboard\.py' 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      # shellcheck disable=SC2086
      kill -KILL $pids 2>/dev/null || true
    fi
  fi
}

cmd_stop() {
  echo "==> stopping divvy-observer"
  local loaded
  loaded="$(loaded_launch_agents)"
  if [[ -n "$loaded" ]]; then
    echo "==> currently loaded LaunchAgents:"
    echo "$loaded" | sed 's/^/    /'
  fi
  unload_launch_agents
  kill_ports
  kill_divvy_processes
  echo "==> stopped"
}

cmd_status() {
  echo "== LaunchAgents loaded =="
  local loaded
  loaded="$(loaded_launch_agents)"
  if [[ -n "$loaded" ]]; then
    echo "$loaded" | sed 's/^/  /'
  else
    echo "  (none)"
  fi
  echo
  echo "== Plist files on disk =="
  local label plist any=0
  for label in "${LAUNCH_AGENT_LABELS[@]}" net.thoughtbison.divvy-collector; do
    plist="${LAUNCH_AGENT_DIR}/${label}.plist"
    if [[ -f "$plist" ]]; then
      echo "  $plist"
      any=1
    fi
  done
  [[ "$any" -eq 0 ]] && echo "  (none)"
  echo
  echo "== Port bindings =="
  local pids
  for port in "$PORT_API" "$PORT_DASHBOARD"; do
    pids="$(pids_on_port "$port")"
    if [[ -n "$pids" ]]; then
      echo "  :${port}  ${pids}"
    else
      echo "  :${port}  (free)"
    fi
  done
  echo
  echo "== Divvy worker processes =="
  if pgrep -fl 'divvy\.collector|divvy\.automation|divvy\.api|src/divvy/dashboard\.py' >/dev/null 2>&1; then
    pgrep -fl 'divvy\.collector|divvy\.automation|divvy\.api|src/divvy/dashboard\.py' | sed 's/^/  /'
  else
    echo "  (none)"
  fi
}

preflight() {
  # Refuse to start if anything is already in the way. Print actionable advice.
  local conflict=0

  local loaded
  loaded="$(loaded_launch_agents)"
  if [[ -n "$loaded" ]]; then
    conflict=1
    echo "error: divvy LaunchAgents are currently loaded:" >&2
    echo "$loaded" | sed 's/^/    /' >&2
    echo "       (these auto-restart and will collide with run.sh)" >&2
  fi

  local port pids
  for port in "$PORT_API" "$PORT_DASHBOARD"; do
    pids="$(pids_on_port "$port")"
    if [[ -n "$pids" ]]; then
      conflict=1
      echo "error: port :${port} is in use by PID(s) ${pids}" >&2
    fi
  done

  if [[ "$conflict" -ne 0 ]]; then
    echo >&2
    echo "Run ./run.sh stop to clear LaunchAgents + bound ports, then retry." >&2
    exit 1
  fi
}

cmd_start() {
  require_uv
  preflight
  mkdir -p data logs

  echo "==> preparing DuckDB read replica"
  uv run python -c "from divvy import db; conn = db.connect(read_only=False); db.init_schema(conn); conn.close(); db.refresh_read_replica()" \
    || echo "warning: could not prepare read replica; services will retry"

  echo "==> starting collector"
  uv run python -m divvy.collector &
  COLLECTOR_PID=$!

  # Give the collector a head start so it can grab the DuckDB writer lock
  # and finish its first tick before automation starts queuing write-jobs.
  sleep 15

  echo "==> starting automation supervisor"
  uv run python -m divvy.automation run &
  AUTOMATION_PID=$!

  echo "==> starting API on http://127.0.0.1:${PORT_API}/docs"
  uv run uvicorn divvy.api:app --host 127.0.0.1 --port "$PORT_API" &
  API_PID=$!

  echo "==> starting dashboard on http://127.0.0.1:${PORT_DASHBOARD}"
  uv run streamlit run src/divvy/dashboard.py \
    --server.address=127.0.0.1 \
    --server.port="$PORT_DASHBOARD" \
    --server.headless=true \
    --browser.gatherUsageStats=false &
  DASHBOARD_PID=$!

  cleanup() {
    echo
    echo "==> stopping Divvy local stack"
    kill -TERM "$COLLECTOR_PID" "$AUTOMATION_PID" "$API_PID" "$DASHBOARD_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      if ! kill -0 "$COLLECTOR_PID" 2>/dev/null &&
         ! kill -0 "$AUTOMATION_PID" 2>/dev/null &&
         ! kill -0 "$API_PID" 2>/dev/null &&
         ! kill -0 "$DASHBOARD_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    kill -KILL "$COLLECTOR_PID" "$AUTOMATION_PID" "$API_PID" "$DASHBOARD_PID" 2>/dev/null || true
    # Belt-and-suspenders for grandchildren (uv → python → streamlit).
    kill_ports
    kill_divvy_processes
  }
  trap cleanup EXIT INT TERM

  echo "==> dashboard: http://127.0.0.1:${PORT_DASHBOARD}"
  echo "==> API docs:  http://127.0.0.1:${PORT_API}/docs"
  echo "==> Ctrl+C to stop everything"
  wait "$COLLECTOR_PID" "$AUTOMATION_PID" "$API_PID" "$DASHBOARD_PID"
}

# --- dispatch ---------------------------------------------------------------

case "${1:-start}" in
  start)
    cmd_start
    ;;
  stop)
    cmd_stop
    ;;
  status)
    cmd_status
    ;;
  -h|--help|help)
    print_help
    ;;
  *)
    echo "unknown command: $1" >&2
    print_help
    exit 2
    ;;
esac
