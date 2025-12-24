#!/bin/bash
# Start the LLM server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# This script snippet sets default values for the LLM server's host and port.
# It uses shell parameter expansion to assign '127.0.0.1' to LLM_SERVER_HOST
# and '8777' to LLM_SERVER_PORT if these environment variables are not already set.
# Existing environment variables will take precedence.

# Default settings (can be overridden via environment variables)
export LLM_SERVER_HOST="${LLM_SERVER_HOST:-127.0.0.1}"
export LLM_SERVER_PORT="${LLM_SERVER_PORT:-8777}"

echo "Starting LLM Server on ${LLM_SERVER_HOST}:${LLM_SERVER_PORT}"

# Run with uv
uv run llm-server --host "$LLM_SERVER_HOST" --port "$LLM_SERVER_PORT" ${LLM_SERVER_DEBUG}
