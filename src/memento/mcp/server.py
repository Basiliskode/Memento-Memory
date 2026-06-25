"""FastMCP stdio server for memento.

Exposes ``EtchStore`` as 16 MCP tools:

- ``get_version``
- ``add_fact``
- ``search_facts``
- ``get_fact``
- ``delete_fact``
- ``get_timeline``
- ``search_similar``
- ``list_inbox``
- ``promote_fact``
- ``reject_fact``
- ``read_map``
- ``list_maps``
- ``create_map``
- ``link_fact``
- ``search_map``
- ``list_regions``

The store is a module-level singleton initialized from the
``MEMENTO_DB_PATH`` environment variable (falls back to ``MEMORY_ETCH_DB_PATH``).

Usage:
    MEMENTO_DB_PATH=~/.memento/etch.db python -m memento.mcp
"""

import importlib.metadata
import json
import logging
import os
from collections import OrderedDict

from mcp.server.fastmcp import FastMCP

from memento import EtchStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton store
# ---------------------------------------------------------------------------

_store: EtchStore | None = None


def get_store() -> EtchStore:
    """Get or create the singleton EtchStore instance.

    The database path is read from the ``MEMENTO_DB_PATH`` environment
    variable (falls back to ``MEMORY_ETCH_DB_PATH`` for backward
    compatibility).  If neither is set, defaults to ``:memory:`` (useful
    for testing).
    For production, set it to ``~/.memento/etch.db``.

    The store is created once and cached for the lifetime of the process.
    """
    global _store
    if _store is not None:
        return _store

    db_path = os.environ.get("MEMENTO_DB_PATH") or os.environ.get("MEMORY_ETCH_DB_PATH")
    if not db_path:
        db_path = ":memory:"

    _store = EtchStore(db_path, auto_migrate=True)
    return _store


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

server = FastMCP("memento", log_level="WARNING")


@server.tool()
def get_version() -> str:
    """Get the installed memento-etch version.

    Returns:
        JSON string with ``{"version": "X.Y.Z"}``.
    """
    try:
        ver = importlib.metadata.version("memento-etch")
    except importlib.metadata.PackageNotFoundError:
        ver = "unknown"
    return json.dumps({"version": ver})


@server.tool()
def add_fact(
    content: str,
    project: str | None = None,
    session_id: str | None = None,
    topic_key: str | None = None,
    source: str | None = None,
    metadata: str | None = None,
    source_harness: str | None = None,
    source_agent: str | None = None,
    source_kind: str | None = None,
    scope: str | None = None,
) -> str:
    """Add a fact to the memory store.

    Args:
        content: Fact text content.
        project: Optional project namespace.
        session_id: Optional session identifier.
        topic_key: Optional topic key for upsert behavior.
        source: Optional source description (stored in ``what`` field).
        metadata: Optional JSON metadata string (stored in ``learned`` field).
        source_harness: Optional source harness identifier.
        source_agent: Optional source agent identifier.
        source_kind: Optional source kind (e.g. "provider", "conversation").
        scope: Optional fact scope (default: "canonical").

    Returns:
        JSON string with ``{"id": int, "status": "created"|"updated"}``.
    """
    store = get_store()
    what_text = source or ""
    learned_text = metadata or ""
    kwargs = dict(
        content=content,
        project=project or "",
        session_id=session_id or "",
        topic_key=topic_key or "",
        what=what_text,
        learned=learned_text,
    )
    if source_harness is not None:
        kwargs["source_harness"] = source_harness
    if source_agent is not None:
        kwargs["source_agent"] = source_agent
    if source_kind is not None:
        kwargs["source_kind"] = source_kind
    if scope is not None:
        kwargs["scope"] = scope
    fid = store.add_fact(**kwargs)
    # Determine status: if topic_key was provided and content differs → "updated"
    status = "updated" if (topic_key and store.get_fact(fid) and store.get_fact(fid)["content"] == content) else "created"  # noqa: E501
    # Simpler: always "created" unless we can detect upsert.  Use revision_count.
    fact = store.get_fact(fid)
    if fact and fact.get("revision_count", 0) > 0:
        status = "updated"
    return json.dumps({"id": fid, "status": status})


@server.tool()
def search_facts(
    query: str,
    limit: int = 10,
    project: str | None = None,
    mode: str = "auto",
    scope: str | None = None,
    source_harness: str | None = None,
    source_agent: str | None = None,
    source_kind: str | None = None,
) -> str:
    """Search facts by full-text query.

    Args:
        query: Full-text search query.
        limit: Max results (default: 10).
        project: Optional project filter.
        mode: Search mode (default: "auto").
        scope: Optional scope filter (default: "canonical").
        source_harness: Optional source harness filter.
        source_agent: Optional source agent filter.
        source_kind: Optional source kind filter.

    Returns:
        JSON array of fact dicts with ``id``, ``content``, ``score``,
        ``project``, ``summary`` keys.
    """
    store = get_store()
    results = store.search_facts(
        query=query, limit=limit, project=project or "",
        scope=scope or "canonical",
        source_harness=source_harness or "",
        source_agent=source_agent or "",
        source_kind=source_kind or "",
    )
    output = []
    for r in results:
        content = r.get("content", "")
        output.append({
            "id": r["fact_id"],
            "content": content,
            "score": r.get("trust_score", 0.0),
            "project": r.get("project", ""),
            "summary": content[:200] if content else "",
        })
    return json.dumps(output)


@server.tool()
def get_fact(fact_id: int) -> str:
    """Get a single fact by its ID.

    Args:
        fact_id: The fact ID to retrieve.

    Returns:
        JSON string with the full fact dict, or ``{"status": "not_found"}``.
    """
    store = get_store()
    fact = store.get_fact(fact_id)
    if fact is None:
        return json.dumps({"status": "not_found"})
    # Remove binary blobs for JSON serialisation
    fact.pop("hrr_vector", None)
    fact.pop("embedding", None)
    return json.dumps(fact, default=str)


@server.tool()
def delete_fact(fact_id: int) -> str:
    """Permanently delete a fact by its ID.

    Args:
        fact_id: The fact ID to delete.

    Returns:
        JSON string with ``{"status": "deleted"|"not_found"}``.
    """
    store = get_store()
    fact = store.get_fact(fact_id)
    if fact is None:
        return json.dumps({"status": "not_found"})
    store.remove_fact(fact_id)
    return json.dumps({"status": "deleted"})


@server.tool()
def get_timeline(project: str | None = None, limit: int = 20) -> str:
    """Get fact timeline, newest first.

    Args:
        project: Optional project filter.
        limit: Max entries (default: 20).

    Returns:
        JSON array of fact dicts.
    """
    store = get_store()
    results = store.list_facts(project=project or "", limit=limit)
    output = []
    for r in results:
        d = dict(r)
        d.pop("hrr_vector", None)
        d.pop("embedding", None)
        output.append(d)
    return json.dumps(output, default=str)


@server.tool()
def search_similar(query: str, limit: int = 5) -> str:
    """Search for facts similar to the given query text.

    Uses full-text search (FTS5) to find semantically related facts.

    Args:
        query: Text to find similar facts for.
        limit: Max results (default: 5).

    Returns:
        JSON array of fact dicts sorted by relevance.
    """
    store = get_store()
    results = store.search_facts(query=query, limit=limit)
    output = []
    for r in results:
        content = r.get("content", "")
        output.append({
            "id": r["fact_id"],
            "content": content,
            "score": r.get("trust_score", 0.0),
            "project": r.get("project", ""),
            "summary": content[:200] if content else "",
        })
    return json.dumps(output, default=str)


@server.tool()
def list_inbox(
    project: str | None = None,
    source_harness: str | None = None,
    limit: int = 50,
) -> str:
    """List inbox facts for review.

    Returns non-deleted facts where ``scope='inbox'``, optionally
    filtered by project and/or source_harness.

    Args:
        project: Optional project filter.
        source_harness: Optional source harness filter.
        limit: Max results (default: 50).

    Returns:
        JSON array of fact dicts.
    """
    store = get_store()
    results = store.list_inbox(
        project=project or "",
        source_harness=source_harness or "",
        limit=limit,
    )
    output = []
    for r in results:
        d = dict(r)
        d.pop("hrr_vector", None)
        d.pop("embedding", None)
        output.append(d)
    return json.dumps(output, default=str)


@server.tool()
def promote_fact(fact_id: int) -> str:
    """Promote an inbox fact to canonical scope.

    Changes ``scope`` from ``'inbox'`` to ``'canonical'`` and updates
    the timestamp. Only affects facts where ``scope='inbox'`` and not
    already deleted.

    Args:
        fact_id: ID of the inbox fact to promote.

    Returns:
        JSON string with ``{"status": "promoted"|"not_found"}``.
    """
    store = get_store()
    ok = store.promote_fact(fact_id)
    status = "promoted" if ok else "not_found"
    return json.dumps({"status": status})


@server.tool()
def read_map(map_id: int) -> str:
    """Get a full atlas map with its regions and edges.

    Args:
        map_id: The map ID to read.

    Returns:
        JSON string with the full map data, or ``{"status": "not_found"}``.
    """
    store = get_store()
    m = store.get_map(map_id)
    if m is None:
        return json.dumps({"status": "not_found"})
    m["regions"] = store.list_regions(map_id)
    m["edges"] = store.get_edges()
    return json.dumps(m, default=str)


@server.tool()
def list_maps(project: str = "") -> str:
    """List all atlas maps, optionally filtered by project.

    Args:
        project: Optional project filter.

    Returns:
        JSON array of map dicts.
    """
    store = get_store()
    maps = store.list_maps(project=project)
    return json.dumps(maps, default=str)


@server.tool()
def create_map(
    name: str,
    description: str = "",
    tags: str = "",
    project: str = "",
) -> str:
    """Create a new atlas map.

    Args:
        name: Map name.
        description: Optional description.
        tags: Comma-separated tags.
        project: Optional project namespace.

    Returns:
        JSON string with ``{"map_id": int}``.
    """
    store = get_store()
    mid = store.create_map(name=name, description=description,
                           tags=tags, project=project)
    return json.dumps({"map_id": mid})


@server.tool()
def link_fact(
    map_id: int,
    fact_id: int,
    relation_type: str = "contains",
    weight: float = 0.5,
) -> str:
    """Link a fact to an atlas map.

    Args:
        map_id: The map ID.
        fact_id: The fact ID.
        relation_type: Edge type (default: 'contains').
        weight: Edge weight 0.0–1.0 (default: 0.5).

    Returns:
        JSON string with ``{"edge_id": int}``.
    """
    store = get_store()
    eid = store.link_fact(map_id=map_id, fact_id=fact_id,
                           relation_type=relation_type, weight=weight)
    return json.dumps({"edge_id": eid})


@server.tool()
def search_map(
    query: str,
    limit: int = 20,
    project: str = "",
) -> str:
    """Full-text search across atlas maps.

    Args:
        query: Search query.
        limit: Max results (default: 20).
        project: Optional project filter.

    Returns:
        JSON array of map summary dicts.
    """
    store = get_store()
    results = store.search_map(query=query, limit=limit, project=project)
    return json.dumps(results, default=str)


@server.tool()
def list_regions(map_id: int) -> str:
    """List all regions in a map.

    Args:
        map_id: The parent map ID.

    Returns:
        JSON array of region dicts.
    """
    store = get_store()
    regions = store.list_regions(map_id)
    return json.dumps(regions, default=str)


@server.tool()
def reject_fact(fact_id: int, reason: str = "") -> str:
    """Reject an inbox fact (soft-delete with reason).

    Soft-deletes the fact and stores the rejection reason. Only affects
    non-deleted inbox facts.

    Args:
        fact_id: ID of the inbox fact to reject.
        reason: Optional rejection reason (default: "").

    Returns:
        JSON string with ``{"status": "rejected"|"not_found"}``.
    """
    store = get_store()
    ok = store.reject_fact(fact_id, reason=reason)
    status = "rejected" if ok else "not_found"
    return json.dumps({"status": status})


# ---------------------------------------------------------------------------
# Engram-compatible aliases (mem_save, mem_search, mem_context,
# mem_session_summary, mem_save_prompt, mem_review).
#
# These thin wrappers expose the same surface as Engram's agent tools so
# skills written for Engram (and the system-prompt protocol in
# examples/memento-protocol.md) work against memento unchanged. They
# delegate to the canonical tools above -- the underlying EtchStore
# storage and search behaviour is unchanged.
#
# The aliases also implement memento's auto-capture behaviour:
# `mem_save_prompt` buffers the most recent user prompt for a session,
# and `mem_save` (when `capture_prompt=True`) attaches that buffered
# prompt as metadata on the persisted fact.
# ---------------------------------------------------------------------------

# session_id -> most recent user prompt captured via mem_save_prompt.
# Module-level singleton because FastMCP runs single-threaded async;
# the dict is keyed by session_id and cleared at session close.
_PROMPT_BUFFER_MAX = 256
_prompt_buffer: "OrderedDict[str, str]" = OrderedDict()


def _topic_key_from_type(type_: str, explicit: str | None = None) -> str:
    """Derive a default topic_key from a mem_save type when none is supplied.

    Returns a stable key like ``mem:<type>`` so multiple saves of the same
    kind upsert instead of fragmenting into many one-off facts.
    """
    if explicit:
        return explicit
    return f"mem:{type_}" if type_ else ""


@server.tool()
def mem_save_prompt(session_id: str, prompt: str) -> str:
    """Buffer the user's most recent prompt for later attachment to mem_save.

    The buffer is keyed by ``session_id`` so concurrent sessions do not
    collide. The prompt is consumed (popped) by the next ``mem_save`` call
    with ``capture_prompt=True`` for the same session -- it is a one-shot
    handoff, not a multi-message log.

    Args:
        session_id: Active session identifier.
        prompt: The user's prompt text to buffer.

    Returns:
        JSON string with ``{"status": "buffered", "session_id": str}``.
    """
    _prompt_buffer[session_id] = prompt
    # Evict the oldest entry if we exceed the cap.
    while len(_prompt_buffer) > _PROMPT_BUFFER_MAX:
        _prompt_buffer.popitem(last=False)
    _prompt_buffer.move_to_end(session_id)
    return json.dumps({"status": "buffered", "session_id": session_id})


@server.tool()
def mem_save(
    title: str,
    type: str,
    content: str,
    project: str | None = None,
    session_id: str | None = None,
    topic_key: str | None = None,
    scope: str = "canonical",
    capture_prompt: bool = True,
) -> str:
    """Save a decision, discovery, or pattern (Engram-compatible).

    Combines ``title`` and structured ``content`` into a single fact
    namespaced under ``project``. If a prompt was previously buffered for
    ``session_id`` via :func:`mem_save_prompt` and ``capture_prompt`` is
    true, the prompt is attached as JSON metadata before being consumed.

    Args:
        title: Verb + what -- short, searchable (e.g. "Fixed N+1 in UserList").
        type: One of ``bugfix | decision | architecture | discovery |
            pattern | config | preference``. Drives the default topic_key.
        content: Structured body covering what / why / where / learned.
        project: Project namespace. Defaults to the value derived from
            the current working directory at MCP server boot.
        session_id: Optional session identifier (used for prompt capture
            and session-scoped retrieval).
        topic_key: Optional explicit topic key for upsert behaviour.
            Defaults to ``mem:<type>`` so same-type saves merge.
        scope: Fact scope (default ``"canonical"``).
        capture_prompt: When true (default) and a prompt was buffered
            for ``session_id``, attach it as metadata.

    Returns:
        JSON string with ``{"id": int, "status": "created"|"updated",
        "user_prompt_attached": bool}``.
    """
    full_content = f"{title}\n\n{content}".strip()

    metadata_str: str | None = None
    if capture_prompt and session_id and session_id in _prompt_buffer:
        metadata_str = json.dumps({"user_prompt": _prompt_buffer[session_id]})

    result = add_fact(
        content=full_content,
        project=project or "",
        session_id=session_id or "",
        topic_key=_topic_key_from_type(type, topic_key),
        metadata=metadata_str,
        scope=scope,
        source_kind="mem_save",
    )
    # Consume the buffered prompt only after the write succeeded.
    if metadata_str is not None and session_id:
        _prompt_buffer.pop(session_id, None)
    return result


@server.tool()
def mem_search(query: str, limit: int = 10, project: str | None = None) -> str:
    """Search memory (Engram-compatible wrapper around search_facts).

    Args:
        query: Full-text search query.
        limit: Max results (default 10).
        project: Optional project namespace filter.

    Returns:
        JSON array of fact dicts.
    """
    return search_facts(query=query, limit=limit, project=project or "")


@server.tool()
def mem_context(session_id: str, limit: int = 20) -> str:
    """Recover recent session history (Engram-compatible).

    Returns facts recorded for ``session_id``, newest first, capped at
    ``limit``. This is the first call an agent should make when a user
    asks "remember", "recall", "what did we do", or "how did we solve".

    Args:
        session_id: Session identifier to recover.
        limit: Max facts to return (default 20).

    Returns:
        JSON array of fact dicts ordered newest-first.
    """
    store = get_store()
    # Delegate to list_facts -- it already supports session_id via the
    # caller-supplied filter path; we hand-roll the SQL through the
    # store's connection to honour session scoping.
    with store._lock:  # type: ignore[attr-defined]
        cur = store._conn.cursor()  # type: ignore[attr-defined]
        cur.execute(
            "SELECT * FROM facts WHERE session_id = ? "
            "AND (deleted IS NULL OR deleted = 0) "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cur.fetchall()
    output = []
    for r in rows:
        d = dict(r)
        d.pop("hrr_vector", None)
        d.pop("embedding", None)
        output.append(d)
    return json.dumps(output, default=str)


@server.tool()
def mem_session_summary(
    session_id: str,
    goal: str,
    accomplishments: list[str],
    next_steps: list[str],
    discoveries: list[str] | None = None,
    files_touched: list[str] | None = None,
) -> str:
    """Save an end-of-session summary (Engram session close protocol).

    Persists a structured fact with the standard Engram layout (Goal /
    Instructions / Discoveries / Accomplished / Next Steps / Relevant
    Files). Clears any buffered prompt for the session so the next
    session starts clean.

    Args:
        session_id: Session being closed.
        goal: What we were working on this session.
        accomplishments: Completed items with key details.
        next_steps: What remains for the next session.
        discoveries: Technical findings, gotchas, non-obvious learnings.
        files_touched: Paths touched this session with a one-line gloss.

    Returns:
        JSON string with ``{"id": int, "status": "created"}``.
    """
    def _append_section(header: str, items: list[str] | None) -> None:
        if not items:
            return
        bullets = [f"- {x}" for x in items if x.strip()]
        if not bullets:
            return
        sections.append(f"## {header}")
        sections.extend(bullets)

    sections: list[str] = ["## Goal", goal.strip()]
    _append_section("Discoveries", discoveries)
    _append_section("Accomplished", accomplishments)
    _append_section("Next Steps", next_steps)
    _append_section("Relevant Files", files_touched)
    full_content = "\n\n".join(s for s in sections if s)

    # Clear buffered prompt on session close.
    _prompt_buffer.pop(session_id, None)

    return add_fact(
        content=full_content,
        project="",
        session_id=session_id,
        topic_key=f"mem:session-summary:{session_id}",
        metadata=json.dumps({
            "summary_kind": "session_close",
            "session_id": session_id,
        }),
        scope="canonical",
        source_kind="mem_session_summary",
    )


@server.tool()
def mem_review(action: str = "list", project: str | None = None) -> str:
    """Lifecycle management for stored memories (Engram-compatible).

    For memento's first release this only implements ``action="list"``,
    which returns facts whose ``learned`` metadata hints at staleness
    (e.g. ``needs_review`` markers). A future release will add
    ``mark_reviewed``; the surface is reserved here so skills do not
    break when that lands.

    Args:
        action: Currently only ``"list"`` is supported.
        project: Optional project namespace filter.

    Returns:
        JSON string with ``{"action": "list", "items": [...], "count": N}``.
    """
    if action != "list":
        return json.dumps({
            "status": "unsupported",
            "action": action,
            "supported": ["list"],
        })
    store = get_store()
    items = store.list_facts(project=project or "", limit=200)
    return json.dumps({
        "action": "list",
        "items": items,
        "count": len(items),
    }, default=str)
