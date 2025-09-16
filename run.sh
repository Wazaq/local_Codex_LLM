#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Default to file-backed sessions unless explicitly overridden. If a session file
# already exists, prefer file backend even when an inherited env sets memory.
: "${SESSION_STORAGE_PATH:=data/sessions.json}"
if [ -z "${SESSION_BACKEND:-}" ] || { [ "${SESSION_BACKEND}" = "memory" ] && [ -f "${SESSION_STORAGE_PATH}" ]; }; then
  SESSION_BACKEND=file
fi
export SESSION_BACKEND SESSION_STORAGE_PATH

python3 main.py
