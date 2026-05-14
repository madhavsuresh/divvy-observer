#!/usr/bin/env bash
# Foreground data collector. For always-on collection, install the launchd plist.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required. install with: brew install uv" >&2
  exit 1
fi

mkdir -p data
exec uv run python -m divvy.collector
