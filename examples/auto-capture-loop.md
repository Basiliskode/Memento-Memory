# Auto-Capture Loop — End-to-End Example

This example shows a complete session loop with Memento auto-capture enabled. It demonstrates the three hooks (`prompt`, `summary`, `close`) and how they interact with the existing `mem_*` aliases from PR #23.

## Prerequisites

```bash
pip install memento-etch[mcp]
export MEMENTO_DB_PATH=~/.memento/etch.db
export MEMENTO_FAST_BUFFER=1   # zero-latency in-process path
```

The `MEMENTO_FAST_BUFFER=1` flag tells the host to use the in-process fast path. Without it, `memento-capture` shells out to `python -m memento.capture` (50-200ms per call).

## Setup

Create a session and export its ID:

```bash
export SESSION_ID="sess-$(date +%s)"
```

## The loop

### Step 1 — User says something meaningful

```bash
memento-capture prompt "$SESSION_ID" --text "We decided to use PostgreSQL for the runtime DB because the existing SQLite setup couldn't handle concurrent writes from the agents."
```

What happens:

1. The CLI checks `should_capture(text)` → returns `True` (not noise, has meaning).
2. On the fast path: `_prompt_buffer[SESSION_ID] = text` (microseconds).
3. The text is buffered for the next `mem_save` call to attach.

### Step 2 — Agent decides this is a decision worth saving

```python
# The host (or agent) calls mem_save with type="decision":
import json
from memento.mcp.server import mem_save

result = json.loads(mem_save(
    title="Use PostgreSQL for runtime DB",
    type="decision",
    content=(
        "## Why\n"
        "SQLite couldn't handle concurrent writes from the agents.\n\n"
        "## Where\n"
        "apps/runtime/db.py"
    ),
    project="basiliskode-runtime",
    session_id="$SESSION_ID",
))
# → {"id": 1, "status": "created"}
```

What happens:

1. `mem_save` pops the buffered prompt (the text from Step 1).
2. The fact is created with `metadata={"user_prompt": "..."}` attached.
3. Next time you search for "PostgreSQL", you'll find this fact with the original user prompt as context.

### Step 3 — User says noise

```bash
memento-capture prompt "$SESSION_ID" --text "ok"
```

What happens:

1. `should_capture("ok")` → `False` (too short, matches the `^ok$` drop pattern).
2. CLI returns `{"captured": false, "reason": "noise"}` and exits 1.
3. The buffer is **not** touched. Nothing reaches Memento.

### Step 4 — User forces a capture

```bash
memento-capture prompt "$SESSION_ID" --text "/remember the API key rotation happens every 90 days" --force
```

What happens:

1. The `--force` flag bypasses the noise filter.
2. Even though "the API key rotation..." is fine content, `/remember` would have already bypassed via the `force_prefixes` default. `--force` is the explicit override for arbitrary text.
3. Next `mem_save` will attach this as the user prompt.

### Step 5 — Context gets long, host calls summary

```bash
memento-capture summary "$SESSION_ID" \
  --goal "Decide on runtime DB and document the schema" \
  --accomplished "Picked PostgreSQL over SQLite" \
  --accomplished "Wrote schema migration plan" \
  --next "Implement connection pooling" \
  --next "Add integration tests" \
  --discovery "SQLite WAL mode + per-process locks was the bottleneck" \
  --file "apps/runtime/db.py" \
  --file "docs/migrations/postgres.md"
```

What happens:

1. `on_compact` calls `mem_session_summary` with the structured fields.
2. The fact is persisted with the structured session layout (`## Goal` / `## Accomplished` / `## Next Steps` / etc.).
3. The buffered prompt (if any) for this session is cleared.

### Step 6 — Verify the capture

```python
import json
from memento.mcp.server import mem_context

facts = json.loads(mem_context(session_id="$SESSION_ID"))
for f in facts:
    print(f["content"][:200])
```

You should see something like:

```
## Goal
Decide on runtime DB and document the schema

## Discoveries
- SQLite WAL mode + per-process locks was the bottleneck

## Accomplished
- Picked PostgreSQL over SQLite
- Wrote schema migration plan

## Next Steps
- Implement connection pooling
- Add integration tests

## Relevant Files
- apps/runtime/db.py
- docs/migrations/postgres.md
```

### Step 7 — Search across sessions

```python
from memento.mcp.server import mem_search

hits = json.loads(mem_search(query="PostgreSQL concurrency", limit=5))
for h in hits:
    print(h["id"], h["content"][:100], "…")
```

## End-to-end via the CLI only

For hosts that prefer pure-subprocess invocation (no Python imports):

```bash
# Buffer a prompt
memento-capture prompt "$SESSION_ID" --text "Use SQLite FTS5 for retrieval"

# Save a decision via the underlying MCP server
# (note: requires the MCP server to be running, e.g. via Claude Desktop)
# python -m memento.mcp --call mem_save --title "Wire FTS5" --type architecture ...

# Persist a session summary
memento-capture summary "$SESSION_ID" \
  --goal "Wire FTS5" \
  --accomplished "Added tokenizer config"

# Inspect the resolved config
memento-capture config
```

## What NOT to do

- **Don't** call `memento-capture prompt` from inside a tight loop with the same session ID — you'll overwrite the previous buffered prompt (one-shot semantics by design).
- **Don't** rely on `memento-capture` to capture tool outputs — that's noise. Use `mem_save` directly when a tool output is important.
- **Don't** put PII (API keys, passwords, customer data) in prompts you intend to capture. Memento is local-first, but it is **persistent**. Sanitise upstream.
