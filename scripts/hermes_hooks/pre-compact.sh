#!/usr/bin/env bash
# Hermes on_session_end / pre-compact hook: persist session summary to Memento.
# Reads JSON from stdin. Tries to extract goal/accomplished/next from the payload.
# Always exits 0.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$(cat 2>/dev/null || true)"

if [ -z "$PAYLOAD" ]; then
    exit 0
fi

# If payload has structured fields (on_session_end), pass them; else empty.
SESSION_ID="$(python3 -c "
import json, sys
try:
    p = json.loads('''$PAYLOAD''')
    print(p.get('session_id') or p.get('conversation_id') or 'unknown')
except Exception:
    print('unknown')
")"

python3 "$SCRIPT_DIR/memento_write.py" summary "$SESSION_ID" \
    --goal "auto-captured at session boundary" \
    > /tmp/memento-summary-hook.log 2>&1 || true

exit 0
