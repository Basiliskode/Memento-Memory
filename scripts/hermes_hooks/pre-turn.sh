#!/usr/bin/env bash
# Hermes pre_llm_call hook: capture user prompt to Memento.
# Reads JSON from stdin (Hermes passes hook context this way).
# Always exits 0 — capture failures must never block the turn.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$(cat 2>/dev/null || true)"

# Diagnostic: dump raw payload so we can see what Hermes actually sends.
echo "$PAYLOAD" > /tmp/memento-hook-payload.json

if [ -z "$PAYLOAD" ]; then
    exit 0
fi

# Pull session_id and message with python (handles JSON safely)
python3 "$SCRIPT_DIR/memento_write.py" prompt "_" --from-stdin <<< "$PAYLOAD" \
    > /tmp/memento-hook.log 2>&1 || true

exit 0
