# Memento Persistent Memory — Protocol for AI Agents

You have access to **memento**, a persistent memory system that survives
across sessions and compactions. This protocol is **MANDATORY and ALWAYS
ACTIVE** — not something you activate on demand.

The tool names below are the same as Engram's, so skills written for
either system work against memento unchanged.

---

## PROACTIVE SAVE TRIGGERS (mandatory — do NOT wait for user to ask)

Call `mem_save` IMMEDIATELY and WITHOUT BEING ASKED after any of these:

- Architecture or design decision made
- Team convention documented or established
- Workflow change agreed upon
- Tool or library choice made with tradeoffs
- Bug fix completed (include root cause)
- Feature implemented with non-obvious approach
- Notion / Jira / GitHub artifact created or updated with significant content
- Configuration change or environment setup done
- Non-obvious discovery about the codebase
- Gotcha, edge case, or unexpected behavior found
- Pattern established (naming, structure, convention)
- User preference or constraint learned

Self-check after EVERY task: *"Did I make a decision, fix a bug, learn
something non-obvious, or establish a convention? If yes, call mem_save
NOW."*

### `mem_save` arguments

- `title` (required): Verb + what — short, searchable (e.g. *"Fixed N+1 in UserList"*).
- `type` (required): One of `bugfix | decision | architecture | discovery | pattern | config | preference`.
- `content` (required): Structured body covering **what**, **why**, **where**, **learned**.
- `project`: Optional. Namespace for facts. If you are working in a git repo, use the repo name (e.g. `basiliskode`).
- `session_id`: The current session id (same one you pass to `mem_save_prompt`).
- `topic_key`: Optional explicit key for upsert behaviour. Defaults to `mem:<type>` so same-type saves merge.
- `capture_prompt`: Defaults to `true`. Set to `false` ONLY for automated artifacts (CI reports, generated specs, scan output).

### `mem_save` content shape

Use this template inside `content`:

```
## What
One sentence — what was done.

## Why
What motivated it (user request, bug, performance, etc.).

## Where
Files or paths affected (one per line).

## Learned
Gotchas, edge cases, things that surprised you (omit if none).
```

---

## PROMPT CAPTURE (auto-attached to the next save)

Call `mem_save_prompt` as soon as you receive a user message — before
you start working on the answer. The prompt is buffered for that
session and attached automatically to the next `mem_save` call
(when `capture_prompt=True`).

```python
mem_save_prompt(session_id="<current>", prompt="<the user's message>")
```

Rules:

- One prompt per session. The buffer holds only the most recent prompt.
- The buffer is consumed (popped) on the next `mem_save`. It does not
  accumulate across saves.
- `mem_session_summary` clears the buffer on session close.
- If a `mem_save` is called with `capture_prompt=True` and there is no
  buffered prompt, the save still succeeds — it just omits the prompt.
- Do NOT set `capture_prompt=False` for normal decisions / fixes.
  Reserve it for CI artefacts, scan output, and auto-generated specs.

---

## WHEN TO SEARCH MEMORY

On any variation of *"remember"*, *"recall"*, *"what did we do"*,
*"how did we solve"*, or references to past work (in any language):

1. Call `mem_context` — checks recent session history (fast, cheap).
2. If not found, call `mem_search` with relevant keywords.
3. If found, use `get_fact` (the underlying memento tool) for the full
   untruncated content.

Also search PROACTIVELY when:

- Starting work on something that might have been done before
- The user mentions a topic you have no context on
- The user's FIRST message references the project, a feature, or a
  problem — call `mem_search` with keywords from their message to
  check for prior work before responding

---

## SESSION CLOSE PROTOCOL (mandatory)

Before ending a session or saying *"done"* / *"that's it"* (or the
equivalent in the user's language), call `mem_session_summary`:

```python
mem_session_summary(
    session_id="<current>",
    goal="What we were working on this session",
    accomplishments=["Completed item 1", "Completed item 2"],
    next_steps=["What remains for the next session"],
    discoveries=["Gotcha we hit", "Non-obvious thing we learned"],  # optional
    files_touched=["path/to/file.py — what it does"],               # optional
)
```

This is NOT optional. If you skip it, the next session starts blind.

---

## AFTER COMPACTION

If you see a compaction message or "FIRST ACTION REQUIRED":

1. IMMEDIATELY call `mem_session_summary` with the compacted summary
   content — this persists what was done before compaction.
2. Call `mem_context` to recover additional context from previous
   sessions.
3. Only THEN continue working.

---

## TOOL SURFACE (cheat sheet)

| Tool | Purpose |
|---|---|
| `mem_save_prompt(session_id, prompt)` | Buffer the user's latest prompt |
| `mem_save(title, type, content, ...)` | Save a decision / discovery / pattern |
| `mem_search(query, limit, project)` | Search memory (FTS5) |
| `mem_context(session_id, limit)` | Recover recent session history |
| `mem_session_summary(session_id, ...)` | Save session close summary |
| `mem_review(action="list", project)` | Lifecycle / staleness review |

All names map 1:1 to memento's underlying tools, which remain available
for advanced use (`add_fact`, `search_facts`, `get_fact`, `delete_fact`,
`get_timeline`, `search_similar`, `list_inbox`, `promote_fact`,
`reject_fact`, `read_map`, `list_maps`, `create_map`, `link_fact`,
`search_map`, `list_regions`, `get_version`).

---

## WHEN NOT TO SAVE

- Trivial restatements of code that is self-evident from a file read.
- Anything already in the project's README or docs (link instead).
- Transient state (current cursor position, scratch numbers).
- Secrets, tokens, or PII — never store these.
