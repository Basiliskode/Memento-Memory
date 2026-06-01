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

import json
import logging
import os
import sys
import importlib.metadata
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from memento import EtchStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton store
# ---------------------------------------------------------------------------

_store: Optional[EtchStore] = None


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
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    topic_key: Optional[str] = None,
    source: Optional[str] = None,
    metadata: Optional[str] = None,
    source_harness: Optional[str] = None,
    source_agent: Optional[str] = None,
    source_kind: Optional[str] = None,
    scope: Optional[str] = None,
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
    project: Optional[str] = None,
    mode: str = "auto",
    scope: Optional[str] = None,
    source_harness: Optional[str] = None,
    source_agent: Optional[str] = None,
    source_kind: Optional[str] = None,
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
def get_timeline(project: Optional[str] = None, limit: int = 20) -> str:
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
    project: Optional[str] = None,
    source_harness: Optional[str] = None,
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
