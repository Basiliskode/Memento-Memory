#!/usr/bin/env python3
"""
Memento auto-capture writer — production pattern for Hermes shell hooks.

Uses the Memento plugin's `add_fact()` API so the full schema (facts, FTS5,
entities, fact_relations, etc.) is created correctly. This means the plugin
can read what the hook writes.

Also runs **automatic LLM-based fact extraction** with MiniMax-M3 after each
prompt is captured. The extraction runs in a background thread so the hook
returns immediately (capture is fire-and-forget from the agent's perspective).
Each prompt produces two facts:

1. `category=prompt` — the raw user message (what the agent normally sees)
2. `category=extracted_fact` x N — the structured facts extracted by M3

FTS5 search therefore returns BOTH the raw prompts and the structured facts,
so the user can query either way.

Hook integration (config.yaml):
  hooks.pre_llm_call: /home/hermes/.hermes/agent-hooks/pre-turn.sh
  hooks.on_session_end: /home/hermes/.hermes/agent-hooks/pre-compact.sh

Usage (called by shell hooks, not by humans directly):
  echo '{"session_id":"tg-12345","user_message":"hola"}' | \
      python3 memento_write.py prompt _ --from-stdin

  python3 memento_write.py prompt <session_id> --text "<message>"
  python3 memento_write.py prompt <session_id> --text "<message>" --force

  python3 memento_write.py summary <session_id> \
      --goal "..." --accomplished "..." --next "..." --file "..."

Features:
- Built-in Spanish/English noise filter (ok, dale, listo, etc.)
- topic_key-based upsert via plugin (same session:prompt -> update + revision++)
- Uses plugin's add_fact() so the full schema is created
- Idempotent via topic_key
- Automatic LLM extraction via MiniMax-M3 (async, fire-and-forget)
- Embedding cache: same content hash extracted only once per session
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DB = Path.home() / ".memento" / "etch.db"
DB_PATH = Path(os.environ.get("MEMENTO_DB_PATH", str(DEFAULT_DB)))
PROJECT = os.environ.get("MEMENTO_PROJECT", "basiliskode")
MIN_LENGTH = 12

HERMES_AGENT_DIR = Path("/home/hermes/hermes-agent")

# ---------------------------------------------------------------------------
# Noise filter
# ---------------------------------------------------------------------------

NOISE_PATTERNS = [
    r"^ok\.?$",
    r"^dale\.?$",
    r"^listo\.?$",
    r"^va\.?$",
    r"^sí\.?$",
    r"^si\.?$",
    r"^no\.?$",
    r"^gracias\.?$",
    r"^hola\.?$",
    r"^chau\.?$",
    r"^bye\.?$",
    r"^👍$",
    r"^🙏$",
    r"^👎$",
    r"^🚀$",
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)

CAPTURE_PREFIXES = (
    "/remember", "/note", "remember this", "no te olvides",
    "guarda esto", "guardá esto", "save this", "anota esto",
    "anotá esto",
)


def is_noise(text: str, force: bool = False) -> bool:
    """Return True if the text is noise that should be skipped."""
    if force:
        return False
    stripped = text.strip()
    if len(stripped) < MIN_LENGTH:
        return True
    if NOISE_RE.match(stripped):
        return True
    if any(stripped.lower().startswith(p) for p in CAPTURE_PREFIXES):
        return False
    return False


# ---------------------------------------------------------------------------
# Plugin loader (singleton per process)
# ---------------------------------------------------------------------------

_provider_singleton = None
_store_singleton = None
_session_id_singleton = None


def _ensure_path() -> None:
    """Add hermes-agent to sys.path so `import plugins.memory` works."""
    if str(HERMES_AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(HERMES_AGENT_DIR))


def _patch_hermes_home(home: Path) -> None:
    """Patch get_hermes_home() to use the profile's HOME."""
    import hermes_constants
    hermes_constants.get_hermes_home = lambda: home


def get_provider_and_store(session_id: str):
    """Load the memento plugin (singleton) and initialize for a session.

    Returns (provider, store). Both are reused across calls within the
    same Python process — the plugin opens the DB once.
    """
    global _provider_singleton, _store_singleton, _session_id_singleton

    # If the session changed, force re-init (different session_id)
    if _store_singleton is not None and _session_id_singleton == session_id:
        return _provider_singleton, _store_singleton

    _ensure_path()
    _patch_hermes_home(Path.home())

    # Make sure DB_PATH directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    from plugins.memory.memento import EtchMemoryProvider

    class HookConfig(dict):
        """Dict-based config that the EtchMemoryProvider accepts."""
        pass

    config = HookConfig(
        db_path=str(DB_PATH),
        project=PROJECT,
        extract_with_llm=False,
        embedding_provider="noop",
        auto_extract_llm=False,
    )

    provider = EtchMemoryProvider(config=config)
    # initialize() creates the full schema (FTS5, entities, etc.) on first call
    provider.initialize(session_id=session_id)
    _provider_singleton = provider
    _store_singleton = provider._store
    _session_id_singleton = session_id
    return provider, provider._store


# ---------------------------------------------------------------------------
# Capture functions (use plugin API)
# ---------------------------------------------------------------------------

def _find_by_topic_key(store, topic_key: str) -> Optional[dict]:
    """Find an existing fact by topic_key via direct SQL."""
    try:
        # Access the underlying sqlite3 connection via the store.
        # EtchStore wraps the connection; use the public conn attribute.
        rows = store.search_facts(query=topic_key, limit=1) if False else None
    except Exception:
        rows = None

    # Most reliable path: use search_facts with the topic_key as a free-text query
    # then filter exactly. Since topic_keys include ":" and dashes, FTS5 won't
    # match them as a phrase, so we fall back to SQL via the plugin's exec.
    if hasattr(store, "conn") and store.conn is not None:
        try:
            row = store.conn.execute(
                "SELECT fact_id, revision_count FROM facts "
                "WHERE topic_key = ? AND deleted = 0 LIMIT 1",
                (topic_key,),
            ).fetchone()
            if row:
                return {"fact_id": row[0], "revision_count": row[1]}
        except Exception:
            pass

    # Last resort: search by metadata (session_id + limit 100) and filter in Python
    try:
        candidates = store.search_by_metadata(limit=200)
        for f in candidates:
            if f.get("topic_key") == topic_key:
                return {
                    "fact_id": f["fact_id"],
                    "revision_count": f.get("revision_count", 0),
                }
    except Exception:
        pass

    return None


def capture_prompt(session_id: str, text: str, force: bool = False) -> dict:
    """Buffer a user prompt into Memento via the plugin's add_fact().

    Idempotent on topic_key (same session:prompt → update).

    Also schedules asynchronous LLM extraction via MiniMax-M3.
    """
    if is_noise(text, force):
        return {"captured": False, "reason": "noise"}

    provider, store = get_provider_and_store(session_id)

    topic_key = f"{session_id}:prompt"
    tags = "auto-capture,prompt"

    # Check if there's an existing fact for this topic_key
    existing = _find_by_topic_key(store, topic_key)

    if existing:
        fact_id = existing["fact_id"]
        rev = existing.get("revision_count", 0)
        # Update via plugin (handles revision_count atomically)
        from plugins.memory.memento.store._crud import update_fact
        update_fact(
            store,
            fact_id,
            content=text,
            tags=tags,
            revision_count=rev + 1,
        )
        capture_result = {
            "captured": True,
            "fact_id": fact_id,
            "action": "updated",
            "revision": rev + 1,
        }
    else:
        # Insert via plugin (handles all schema, FTS5, dedup by content hash)
        from plugins.memory.memento.store._crud import add_fact

        fact_id = add_fact(
            store,
            content=text,
            category="prompt",
            tags=tags,
            session_id=session_id,
            topic_key=topic_key,
            project=PROJECT,
            importance=0.4,
            source_harness="hermes-hook",
            source_agent="pre-turn",
            source_kind="user-prompt",
        )

        if isinstance(fact_id, dict):
            # add_fact returned metadata — extract fact_id
            fact_id = fact_id.get("fact_id", -1)

        capture_result = {
            "captured": True,
            "fact_id": fact_id,
            "action": "inserted",
        }

    # Schedule async LLM extraction (fire-and-forget; never blocks the hook)
    schedule_llm_extraction(session_id, text, force=force)

    return capture_result


def capture_summary(session_id: str, goal: str = "",
                   accomplished: list = None, next_steps: list = None,
                   discoveries: list = None, files: list = None,
                   force: bool = False) -> dict:
    """Persist a session summary to Memento via the plugin."""
    accomplished = accomplished or []
    next_steps = next_steps or []
    discoveries = discoveries or []
    files = files or []

    provider, store = get_provider_and_store(session_id)

    parts = []
    if goal:
        parts.append(f"Goal: {goal}")
    if accomplished:
        parts.append("Accomplished: " + "; ".join(accomplished))
    if next_steps:
        parts.append("Next: " + "; ".join(next_steps))
    if discoveries:
        parts.append("Discoveries: " + "; ".join(discoveries))
    if files:
        parts.append("Files: " + ", ".join(files))

    content = "\n".join(parts)
    if not content.strip():
        return {"captured": False, "reason": "empty"}

    topic_key = f"{session_id}:summary"
    tags = "session-summary"

    from plugins.memory.memento.store._crud import add_fact

    fact_id = add_fact(
        store,
        content=content,
        category="session_summary",
        tags=tags,
        session_id=session_id,
        topic_key=topic_key,
        project=PROJECT,
        importance=0.7,
        source_harness="hermes-hook",
        source_agent="session-end",
        source_kind="summary",
    )
    if isinstance(fact_id, dict):
        fact_id = fact_id.get("fact_id", -1)
    return {"captured": True, "fact_id": fact_id, "action": "summary"}


# ---------------------------------------------------------------------------
# LLM extraction (MiniMax-M3)
# ---------------------------------------------------------------------------

# Per-process cache: content_hash -> fact_ids already extracted
# Prevents the same prompt from being extracted multiple times if the hook
# is called repeatedly (e.g., during retries).
_extraction_cache: dict[str, list[int]] = {}
_extraction_cache_lock = threading.Lock()

# System prompt for M3 fact extraction. Tight, deterministic, JSON output.
_EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction assistant. Given a single user prompt, extract factual statements that should be remembered long-term.

Return ONLY valid JSON with this exact structure (no markdown, no commentary):
{
    "facts": [
        {"content": "concise fact statement", "category": "project|user_pref|tool|general", "importance": "critical|important|useful|trivial", "tags": "comma,separated,tags", "fact_type": "observation|reflection|decision|preference"}
    ]
}

Rules:
- Extract 0-5 facts per prompt. Skip if there's nothing meaningful (small talk, transient questions, greetings).
- Use present tense, third person.
- Be specific: include names, versions, decisions, tools, projects.
- Categories: project (specific project info), user_pref (user's stated preferences), tool (tool/tech decisions), general (everything else).
- importance: critical (decisions, must remember), important (preferences, key facts), useful (context), trivial (minor).
- fact_type: observation (default), reflection (user's introspective comments), decision (deliberate choice with rationale), preference (taste/habit/stylistic).
- Respond in the same language as the prompt (Spanish → Spanish facts).
- Skip the prompt itself if it's a command, a question about current state, or already captured by previous facts."""


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:16]


def _call_minimax_extract(user_prompt: str, timeout: int = 20) -> Optional[dict]:
    """Call MiniMax-M3 to extract structured facts from a prompt.

    Uses the chat/completions endpoint (OpenAI-compatible).
    Returns parsed JSON dict, or None if anything goes wrong.
    """
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        return None

    base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
    model = os.environ.get("EXTRACT_MODEL", "MiniMax-M3")

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Hermes-Agent/1.0 (memento-extract)",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"].strip()
            # Strip <think>...</think> reasoning block if present
            content = re.sub(r"<think>.*?</think>\s*", "", content,
                             flags=re.DOTALL)
            # Strip markdown fences if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*\n?", "", content)
                content = re.sub(r"\n?```\s*$", "", content)
            return json.loads(content.strip())
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
            json.JSONDecodeError, TimeoutError) as exc:
        sys.stderr.write(f"[memento-extract] LLM call failed: {exc}\n")
        return None


def _importance_to_float(imp: str) -> float:
    return {
        "critical": 0.95,
        "important": 0.75,
        "useful": 0.5,
        "trivial": 0.25,
    }.get(str(imp).lower(), 0.5)


def _store_extracted_facts(session_id: str, source_prompt: str,
                           parsed: dict) -> list[int]:
    """Persist extracted facts to the Memento DB. Returns list of fact_ids."""
    if not parsed or "facts" not in parsed:
        return []

    facts = parsed.get("facts") or []
    if not isinstance(facts, list):
        return []

    provider, store = get_provider_and_store(session_id)
    from plugins.memory.memento.store._crud import add_fact

    stored_ids = []
    source_hash = _content_hash(source_prompt)[:8]

    for idx, fact in enumerate(facts):
        if not isinstance(fact, dict):
            continue
        content = str(fact.get("content", "")).strip()
        if not content or len(content) < 5:
            continue
        category = str(fact.get("category", "general")).lower()
        if category not in ("project", "user_pref", "tool", "general",
                            "session_summary"):
            category = "general"
        importance = _importance_to_float(fact.get("importance", "useful"))
        tags = str(fact.get("tags", "")).strip()
        fact_type = str(fact.get("fact_type", "observation")).lower()
        if fact_type not in ("observation", "reflection", "decision", "preference"):
            fact_type = "observation"

        topic_key = f"{session_id}:extract:{source_hash}:{idx}"
        fact_id = add_fact(
            store,
            content=content,
            category=f"extracted_{category}",
            tags=f"auto-extract,m3,{fact_type},{tags}".strip(","),
            session_id=session_id,
            topic_key=topic_key,
            project=PROJECT,
            importance=importance,
            source_harness="hermes-hook-llm",
            source_agent="minimax-m3-extractor",
            source_kind="extracted-fact",
        )
        if isinstance(fact_id, dict):
            fact_id = fact_id.get("fact_id", -1)
        if fact_id and fact_id > 0:
            stored_ids.append(fact_id)

    return stored_ids


def _extraction_worker(session_id: str, prompt: str, content_h: str) -> None:
    """Background thread: run extraction and persist results."""
    try:
        parsed = _call_minimax_extract(prompt)
        if parsed is None:
            return
        fact_ids = _store_extracted_facts(session_id, prompt, parsed)
        with _extraction_cache_lock:
            _extraction_cache[content_h] = fact_ids
        if fact_ids:
            sys.stderr.write(
                f"[memento-extract] session={session_id[:8]} "
                f"prompt_hash={content_h} facts_added={len(fact_ids)}\n"
            )
    except Exception as exc:
        sys.stderr.write(f"[memento-extract] worker error: {exc}\n")


def schedule_llm_extraction(session_id: str, prompt: str,
                            force: bool = False) -> None:
    """Run LLM extraction synchronously (with a short timeout).

    Daemon-thread async doesn't work: the hook process exits before the
    thread completes, killing the LLM call mid-flight. So we run inline
    with an aggressive timeout (~6s) and a small token budget so the hook
    still returns quickly. If LLM is slow or fails, the raw prompt is
    still saved — extraction is best-effort, not blocking.

    Returns the list of fact_ids that were stored, or [] on failure.
    Skips if the same prompt (by content hash) has already been extracted
    in this Python process (idempotent within a single run).
    """
    if is_noise(prompt, force):
        return []
    if not os.environ.get("MINIMAX_API_KEY"):
        return []  # LLM not configured → skip silently

    content_h = _content_hash(prompt)

    # Cache check: skip if already extracted in this process
    with _extraction_cache_lock:
        if content_h in _extraction_cache:
            return _extraction_cache[content_h]

    # Mark in-flight (prevents duplicate spawns if hook fires twice quickly)
    with _extraction_cache_lock:
        if content_h not in _extraction_cache:
            _extraction_cache[content_h] = []

    try:
        # Short timeout so the hook returns quickly even if M3 is slow.
        # M3 typically takes 3-8s for extraction; 12s is the upper bound.
        # If it exceeds, the raw prompt is still saved — extraction is
        # best-effort.
        parsed = _call_minimax_extract(prompt, timeout=12)
        if parsed is None:
            with _extraction_cache_lock:
                _extraction_cache.pop(content_h, None)
            return []
        fact_ids = _store_extracted_facts(session_id, prompt, parsed)
        with _extraction_cache_lock:
            _extraction_cache[content_h] = fact_ids
        if fact_ids:
            sys.stderr.write(
                f"[memento-extract] session={session_id[:8]} "
                f"prompt_hash={content_h} facts_added={len(fact_ids)}\n"
            )
        return fact_ids
    except Exception as exc:
        sys.stderr.write(f"[memento-extract] sync error: {exc}\n")
        with _extraction_cache_lock:
            _extraction_cache.pop(content_h, None)
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_prompt(args) -> int:
    if args.from_stdin:
        try:
            payload = json.loads(sys.stdin.read())
        except json.JSONDecodeError:
            return 1
        session_id = args.session_id or payload.get("session_id") or "_unknown"
        user_message = payload.get("user_message") or ""
        force = bool(payload.get("force"))
        result = capture_prompt(session_id, user_message, force=force)
    else:
        result = capture_prompt(args.session_id, args.text, force=args.force)

    print(json.dumps(result))
    return 0 if result.get("captured") or not result.get("reason") == "error" else 1


def cmd_summary(args) -> int:
    session_id = args.session_id
    if args.from_stdin:
        try:
            payload = json.loads(sys.stdin.read())
            session_id = session_id or payload.get("session_id") or "_unknown"
        except json.JSONDecodeError:
            return 1

    result = capture_summary(
        session_id=session_id,
        goal=args.goal or "",
        accomplished=args.accomplished or [],
        next_steps=args.next or [],
        discoveries=args.discovery or [],
        files=args.file or [],
    )
    print(json.dumps(result))
    return 0


def cmd_extract(args) -> int:
    """Manual extraction (mostly for testing). Synchronous."""
    session_id = args.session_id
    text = args.text
    if args.from_stdin:
        try:
            payload = json.loads(sys.stdin.read())
            session_id = session_id or payload.get("session_id") or "_unknown"
            text = text or payload.get("user_message") or ""
        except json.JSONDecodeError:
            return 1

    if not text:
        print(json.dumps({"captured": False, "reason": "empty"}))
        return 1

    parsed = _call_minimax_extract(text)
    if parsed is None:
        print(json.dumps({"captured": False, "reason": "llm-failed"}))
        return 1

    fact_ids = _store_extracted_facts(session_id, text, parsed)
    print(json.dumps({
        "captured": True,
        "extracted": len(parsed.get("facts") or []),
        "stored": len(fact_ids),
        "fact_ids": fact_ids,
        "facts": parsed.get("facts", []),
    }))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Memento auto-capture writer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prompt = subparsers.add_parser("prompt", help="Capture a user prompt")
    p_prompt.add_argument("session_id", nargs="?", default=None)
    p_prompt.add_argument("--text", help="Prompt text")
    p_prompt.add_argument("--force", action="store_true", help="Bypass noise filter")
    p_prompt.add_argument("--from-stdin", action="store_true", help="Read JSON payload from stdin")
    p_prompt.set_defaults(func=cmd_prompt)

    p_summary = subparsers.add_parser("summary", help="Capture a session summary")
    p_summary.add_argument("session_id", nargs="?", default=None)
    p_summary.add_argument("--goal", default="")
    p_summary.add_argument("--accomplished", action="append", default=[])
    p_summary.add_argument("--next", action="append", default=[])
    p_summary.add_argument("--discovery", action="append", default=[])
    p_summary.add_argument("--file", action="append", default=[])
    p_summary.add_argument("--from-stdin", action="store_true")
    p_summary.set_defaults(func=cmd_summary)

    p_extract = subparsers.add_parser("extract",
        help="Run LLM extraction synchronously (testing)")
    p_extract.add_argument("session_id", nargs="?", default=None)
    p_extract.add_argument("--text", help="Prompt text to extract from")
    p_extract.add_argument("--from-stdin", action="store_true")
    p_extract.set_defaults(func=cmd_extract)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())