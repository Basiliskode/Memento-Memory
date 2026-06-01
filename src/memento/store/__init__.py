"""Store package — sub-modules for EtchStore.

This module defines the ``EtchStore`` class as a thin delegation layer.
All public and private methods are delegated to sub-module functions in
``_schema``, ``_event_log``, ``_hrr``, ``_embedding``, ``_workspaces``,
``_typed_facts``, ``_inbox``, ``_crud``, ``_sessions``, ``_relations``,
``_provenance``, ``_search``, ``_snapshots``, ``_export_import``,
``_ingest``, ``_sync``, ``_gc``, and ``_compat``.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import struct
import threading
import time
import uuid
import warnings
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional, Callable

from ..embedding import EmbeddingProvider, NoopProvider
from ..project import detect_project

# Import all sub-modules
from . import (
    _schema,
    _event_log,
    _hrr,
    _embedding,
    _workspaces,
    _typed_facts,
    _inbox,
    _crud,
    _sessions,
    _relations,
    _provenance,
    _search,
    _snapshots,
    _export_import,
    _ingest,
    _sync,
    _gc,
    _compat,
    _atlas,
)

logger = logging.getLogger(__name__)

# Valid scopes for Hive Memory governance
VALID_SCOPES: set[str] = {"canonical", "inbox", "scratch", "ephemeral"}


# ---------------------------------------------------------------------------
# Delegation helper
# ---------------------------------------------------------------------------


def _delegate(fn):
    """Wrap a module-level function into an EtchStore method."""
    def wrapper(self, *args, **kwargs):
        return fn(self, *args, **kwargs)
    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = f"EtchStore.{fn.__name__}"
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ---------------------------------------------------------------------------
# EtchStore class
# ---------------------------------------------------------------------------


class EtchStore:
    """SQLite-backed fact store.

    Thread-safe via RLock. Handles schema creation, migration, CRUD,
    FTS5 sync, HRR encoding, soft delete, and consolidation.

    Args:
        db_path: Path to the SQLite database file.
        hrr_dim: Dimension for HRR vectors (default: 256).
        auto_migrate: Whether to run schema migrations on init.
    """

    # Default configuration for garbage collection.
    # Each key maps to a GC phase parameter.
    _GC_DEFAULT_CONFIG = {
        "hard_delete_days": 30,           # hard-delete facts soft-deleted > N days ago
        "prune_event_log_days": 90,       # delete event_log entries older than N days
        "prune_turn_buffer_days": 30,     # delete turn_buffer entries older than N days
        "snapshot_keep": 10,              # keep last N snapshots per project, delete rest
        "snapshot_max_age_days": 365,     # delete snapshots older than this
        "vacuum_threshold_pct": 20,       # VACUUM if free pages exceed this %
    }

    def __init__(
        self,
        db_path: str,
        hrr_dim: int = 256,
        auto_migrate: bool = True,
        embedding_provider: Optional[EmbeddingProvider] = None,
        project: Optional[str] = None,
    ) -> None:
        """Initialize the EtchStore.

        Creates or opens the SQLite database, runs schema migrations, and
        starts the background HRR flush thread when NumPy is available.

        Args:
            db_path: Path to the SQLite database file.
            hrr_dim: Dimension for HRR vectors (default: 256).
            auto_migrate: Whether to run schema creation and migration
                on initialization (default: True).
            embedding_provider: Optional EmbeddingProvider for semantic
                search. If None, uses NoopProvider (no-op, no deps).
            project: Optional project name. ``"auto"`` calls
                ``detect_project()`` on cwd to auto-detect.

        Raises:
            sqlite3.Error: If the database cannot be opened or created.
        """
        if db_path is None or str(db_path).strip() == "":
            raise ValueError("db_path must be an explicit SQLite path or ':memory:'")

        self._db_path = db_path
        self._hrr_dim = hrr_dim
        self._lock = threading.RLock()
        self._embedding_provider = embedding_provider or NoopProvider()
        self._project = self._resolve_project(project)

        # HRR async flush
        self._pending_hrr: list[tuple[int, str]] = []  # (fact_id, content)
        self._hrr_ready = threading.Event()
        self._hrr_flush_signal = threading.Event()
        self._hrr_flush_stop = threading.Event()
        self._hrr_flush_thread: Optional[threading.Thread] = None

        # HRR vector cache (fact_id → np.ndarray, LRU max 500)
        self._hrr_vector_cache: dict[int, "np.ndarray"] = {}
        self._hrr_cache_max = 500

        # Connect
        self._closed = False
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-262144")  # 256 MB
        self._conn.execute("PRAGMA mmap_size=1073741824")  # 1 GB
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")

        if auto_migrate:
            self._ensure_schema()
            self._migrate_schema()
            self._register_default_schemas()
            self._start_hrr_flush()

    @staticmethod
    def _resolve_project(project: Optional[str]) -> Optional[str]:
        """Resolve the ``project`` parameter.

        If ``project`` is the literal string ``"auto"``, calls
        ``detect_project()`` on the current working directory.
        Otherwise returns the value as-is.
        """
        if project == "auto":
            return detect_project()
        return project

    def close(self) -> None:
        """Close the store and release resources.

        Stops the HRR flush thread and closes the database connection.
        Call this when done to avoid resource leaks.
        """
        with self._lock:
            if self._closed:
                return

        self._hrr_flush_stop.set()
        self._signal_flush()
        if self._hrr_flush_thread and self._hrr_flush_thread.is_alive():
            self._hrr_flush_thread.join()

        self._flush_pending_hrr_batch()
        with self._lock:
            self._conn.execute("PRAGMA optimize")
            self._conn.close()
            self._closed = True

    def _register_default_schemas(self) -> None:
        """Register built-in fact schemas for recognized fact types.

        These schemas are idempotent — calling them on an existing DB
        is a no-op (``INSERT OR REPLACE``).
        """
        _typed_facts.register_schema(
            self,
            fact_type="reflection",
            description="Reflective/self-analysis memory — agent introspecting on its own experiences, decisions, or learnings",
            required_fields=["what"],
            optional_fields=["why", "learned"],
        )
        _typed_facts.register_schema(
            self,
            fact_type="observation",
            description="General factual observation extracted from a conversation or interaction",
            required_fields=["what"],
            optional_fields=["why", "where_text"],
        )
        _typed_facts.register_schema(
            self,
            fact_type="decision",
            description="A decision made, with rationale and alternatives considered",
            required_fields=["what", "why"],
            optional_fields=["learned"],
        )
        _typed_facts.register_schema(
            self,
            fact_type="preference",
            description="A user preference, habit, or stylistic choice",
            required_fields=["what"],
            optional_fields=["why"],
        )


# ---------------------------------------------------------------------------
# Phase 1 delegation — Schema & Event Log & HRR
# ---------------------------------------------------------------------------

for _fn_name in ("_ensure_schema", "_migrate_schema", "_sanitize_fts5"):
    setattr(EtchStore, _fn_name, _delegate(getattr(_schema, _fn_name)))

for _fn_name in ("_log_event", "get_event_log"):
    setattr(EtchStore, _fn_name, _delegate(getattr(_event_log, _fn_name)))

for _fn_name in (
    "_start_hrr_flush",
    "_signal_flush",
    "_flush_pending_hrr_batch",
    "_get_effective_hrr_dim",
    "get_effective_hrr_dim",
    "compute_hrr_batch",
    "_get_hrr_cached",
    "_invalidate_hrr_cache",
):
    setattr(EtchStore, _fn_name, _delegate(getattr(_hrr, _fn_name)))

# ---------------------------------------------------------------------------
# Phase 1 delegation — Embedding
# ---------------------------------------------------------------------------

for _fn_name in ("_maybe_store_embedding", "_search_by_embedding"):
    setattr(EtchStore, _fn_name, _delegate(getattr(_embedding, _fn_name)))

# ---------------------------------------------------------------------------
# Phase 2 delegation — Workspaces, Typed Facts, Inbox
# ---------------------------------------------------------------------------

for _fn_name in (
    "_ensure_workspace",
    "_parse_workspace_row",
    "create_workspace",
    "get_workspace",
    "update_workspace",
    "delete_workspace",
    "list_workspaces",
    "workspace_stats",
):
    setattr(EtchStore, _fn_name, _delegate(getattr(_workspaces, _fn_name)))

for _fn_name in (
    "register_schema",
    "get_schema",
    "list_schemas",
    "delete_schema",
    "_validate_fact_type",
):
    setattr(EtchStore, _fn_name, _delegate(getattr(_typed_facts, _fn_name)))

for _fn_name in ("list_inbox", "promote_fact", "reject_fact"):
    setattr(EtchStore, _fn_name, _delegate(getattr(_inbox, _fn_name)))

# ---------------------------------------------------------------------------
# Phase 3 delegation — CRUD
# ---------------------------------------------------------------------------

for _fn_name in (
    "add_fact",
    "_detect_conflicts",
    "add_fact_with_consolidation",
    "soft_delete_fact",
    "restore_fact",
    "remove_fact",
    "get_fact",
    "get_fact_full",
    "list_facts",
    "update_fact",
    "purge_facts",
    "evict_stale",
    "_ensure_entity",
    "get_entities",
    "_reinforce_facts",
):
    setattr(EtchStore, _fn_name, _delegate(getattr(_crud, _fn_name)))

# ---------------------------------------------------------------------------
# Phase 4 delegation — Sessions, Relations, Provenance
# ---------------------------------------------------------------------------

for _fn_name in (
    "start_session",
    "end_session",
    "get_session",
    "generate_session_summary",
    "list_sessions",
):
    setattr(EtchStore, _fn_name, _delegate(getattr(_sessions, _fn_name)))

for _fn_name in (
    "add_relation",
    "judge_relation",
    "get_relations",
    "get_contradictions",
    "get_neighbors",
    "find_path",
    "get_ego_graph",
    "get_subgraph",
    "get_graph_stats",
):
    setattr(EtchStore, _fn_name, _delegate(getattr(_relations, _fn_name)))

for _fn_name in ("_add_derivation_link", "get_provenance", "get_derivation_tree"):
    setattr(EtchStore, _fn_name, _delegate(getattr(_provenance, _fn_name)))

# ---------------------------------------------------------------------------
# Phase 5 delegation — Search
# ---------------------------------------------------------------------------

for _fn_name in (
    "_search_facts_fts5",
    "search_facts",
    "search_by_metadata",
    "search_by_vector",
    "_rrf_merge",
    "search",
    "query",
):
    setattr(EtchStore, _fn_name, _delegate(getattr(_search, _fn_name)))

# ---------------------------------------------------------------------------
# Phase 6a delegation — Snapshots, Export/Import, Ingest
# ---------------------------------------------------------------------------

for _fn_name in (
    "create_snapshot",
    "get_snapshot",
    "list_snapshots",
    "delete_snapshot",
    "restore_snapshot",
    "snapshot_diff",
):
    setattr(EtchStore, _fn_name, _delegate(getattr(_snapshots, _fn_name)))

for _fn_name in ("export_memory", "import_memory", "stats", "projects"):
    setattr(EtchStore, _fn_name, _delegate(getattr(_export_import, _fn_name)))

for _fn_name in ("ingest",):
    setattr(EtchStore, _fn_name, _delegate(getattr(_ingest, _fn_name)))

# ---------------------------------------------------------------------------
# Phase 6b delegation — Atlas (semantic fact map)
# ---------------------------------------------------------------------------

for _fn_name in (
    "create_map",
    "get_map",
    "update_map",
    "delete_map",
    "list_maps",
    "create_region",
    "get_region",
    "list_regions",
    "link_fact",
    "link_region",
    "unlink",
    "get_edges",
    "search_map",
    "get_map_node",
    "traverse_path",
    "map_stats",
):
    setattr(EtchStore, _fn_name, _delegate(getattr(_atlas, _fn_name)))

# ---------------------------------------------------------------------------
# Phase 7 delegation — Sync, Garbage Collection
# ---------------------------------------------------------------------------

for _fn_name in (
    "register_peer",
    "unregister_peer",
    "list_peers",
    "sync_prepare",
    "sync_apply",
    "sync_to_file",
    "sync_from_file",
    "sync_with_peer",
    "get_sync_conflicts",
    "resolve_sync_conflict",
):
    setattr(EtchStore, _fn_name, _delegate(getattr(_sync, _fn_name)))

for _fn_name in ("gc",):
    setattr(EtchStore, _fn_name, _delegate(getattr(_gc, _fn_name)))

# ---------------------------------------------------------------------------
# Phase 8 delegation — Backward-compat aliases (overrides _relations.judge_relation
# and _search.search_facts with the compat versions)
# ---------------------------------------------------------------------------

for _fn_name in (
    "session_start",
    "session_end",
    "timeline",
    "get_timeline",
    "judge_relation",
    "get_recent_sessions",
    "search_facts",
):
    setattr(EtchStore, _fn_name, _delegate(getattr(_compat, _fn_name)))

# ---------------------------------------------------------------------------
# node_id property (was @property in old store.py)
# ---------------------------------------------------------------------------
EtchStore.node_id = property(lambda self: _sync.node_id(self))
