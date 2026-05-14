#!/usr/bin/env bash
# One-command local runner. By default this starts collector, automation, API,
# and dashboard in the foreground so failures are visible in this terminal.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required. install with: brew install uv" >&2
  exit 1
fi

mkdir -p data logs

echo "==> preparing DuckDB read replica"
uv run python -c "from divvy import db; conn = db.connect(read_only=False); db.init_schema(conn); conn.close(); db.refresh_read_replica()" \
  || echo "warning: could not prepare read replica; services will retry"

if [[ "${1:-}" == "--launchd" ]]; then
  shift
  echo "==> installing and starting Divvy launchd services"
  exec uv run divvy start "$@"
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  ./run.sh             Start collector, automation, API, and dashboard in this terminal
  ./run.sh --launchd   Install/start the macOS LaunchAgent services

URLs:
  Dashboard: http://127.0.0.1:8501
  API docs:  http://127.0.0.1:8000/docs
EOF
  exit 0
fi

echo "==> starting collector"
uv run python -m divvy.collector &
COLLECTOR_PID=$!

echo "==> starting automation supervisor"
uv run python -m divvy.automation run &
AUTOMATION_PID=$!

echo "==> starting API on http://127.0.0.1:8000/docs"
uv run uvicorn divvy.api:app --host 127.0.0.1 --port 8000 &
API_PID=$!

echo "==> starting dashboard on http://127.0.0.1:8501"
uv run streamlit run src/divvy/dashboard.py \
  --server.address=127.0.0.1 \
  --server.port=8501 \
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
}
trap cleanup EXIT INT TERM

echo "==> dashboard: http://127.0.0.1:8501"
echo "==> API docs:  http://127.0.0.1:8000/docs"
wait "$COLLECTOR_PID" "$AUTOMATION_PID" "$API_PID" "$DASHBOARD_PID"
