#!/usr/bin/env bash
# Start only the prediction API and Streamlit UI. The collector should run separately.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required. install with: brew install uv" >&2
  exit 1
fi

mkdir -p data

API_ADDRESS="${DIVVY_API_ADDRESS:-0.0.0.0}"
API_PORT="${DIVVY_API_PORT:-8001}"
STREAMLIT_ADDRESS="${DIVVY_STREAMLIT_ADDRESS:-0.0.0.0}"
STREAMLIT_PORT="${DIVVY_STREAMLIT_PORT:-8503}"

echo "==> starting prediction API on http://localhost:${API_PORT}"
uv run uvicorn divvy.api:app \
  --host "${API_ADDRESS}" \
  --port "${API_PORT}" &
API_PID=$!

echo "==> starting dashboard on http://localhost:${STREAMLIT_PORT}"
uv run streamlit run src/divvy/dashboard.py \
  --server.address="${STREAMLIT_ADDRESS}" \
  --server.port="${STREAMLIT_PORT}" \
  --server.headless=true \
  --browser.gatherUsageStats=false &
STREAMLIT_PID=$!

cleanup() {
  echo
  echo "==> stopping api=$API_PID streamlit=$STREAMLIT_PID"
  kill -TERM "$API_PID" "$STREAMLIT_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    if ! kill -0 "$API_PID" 2>/dev/null && ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  kill -KILL "$API_PID" "$STREAMLIT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait "$API_PID" "$STREAMLIT_PID"
