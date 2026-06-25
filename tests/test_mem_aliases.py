"""Tests for the Engram-compatible aliases on the memento MCP server.

The aliases (``mem_save``, ``mem_save_prompt``, ``mem_search``,
``mem_context``, ``mem_session_summary``, ``mem_review``) are thin
wrappers around the canonical memento tools. These tests exercise them
through the public function surface — calling each tool as if it were
invoked by an MCP client — and assert the persisted storage shape.

Each test gets a fresh in-memory DB by setting ``MEMENTO_DB_PATH`` to
``":memory:"`` before importing the server module. The module-level
``_prompt_buffer`` is reset at the start of every test to avoid
cross-test contamination.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def fresh_db_and_buffer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Reset the server module state before each test.

    Memento reads MEMENTO_DB_PATH at module import; we point it at a
    fresh temp file for every test, then reload the server module so
    the module-level ``_store`` singleton re-initialises against it.
    The ``_prompt_buffer`` dict is also cleared explicitly.
    """
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("MEMENTO_DB_PATH", str(db_file))

    # Force the server module to re-import so it picks up the new env.
    sys.modules.pop("memento.mcp.server", None)
    server = importlib.import_module("memento.mcp.server")
    server._prompt_buffer.clear()
    yield server
    server._prompt_buffer.clear()


# ---------------------------------------------------------------------------
# mem_save_prompt + mem_save: auto-capture
# ---------------------------------------------------------------------------


def test_mem_save_prompt_buffers_prompt(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    result = json.loads(
        server.mem_save_prompt(session_id="s1", prompt="fix the login bug")
    )
    assert result == {"status": "buffered", "session_id": "s1"}
    assert server._prompt_buffer["s1"] == "fix the login bug"


def test_mem_save_attaches_buffered_prompt(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    server.mem_save_prompt(session_id="s1", prompt="user wanted X")
    result = json.loads(
        server.mem_save(
            title="Did X",
            type="decision",
            content="## What\nChose X.\n\n## Why\nUser asked.",
            project="test-project",
            session_id="s1",
        )
    )
    assert result["status"] in ("created", "updated")
    fid = int(result["id"])

    fact = json.loads(server.get_fact(fid))
    # The title appears in the persisted content.
    assert "Did X" in fact["content"]
    # The buffered prompt is attached as metadata.
    metadata = json.loads(fact["learned"]) if fact["learned"] else {}
    assert metadata.get("user_prompt") == "user wanted X"
    # The prompt was consumed (one-shot handoff).
    assert "s1" not in server._prompt_buffer


def test_mem_save_without_buffered_prompt_works(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    result = json.loads(
        server.mem_save(
            title="Solo save",
            type="discovery",
            content="## What\nStandalone fact.",
            session_id="s2",
        )
    )
    assert result["status"] in ("created", "updated")
    fid = int(result["id"])
    fact = json.loads(server.get_fact(fid))
    # No prompt was buffered, so metadata is empty / null.
    assert fact["learned"] in (None, "", "null")


def test_mem_save_with_capture_prompt_false_skips_buffer(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    server.mem_save_prompt(session_id="s1", prompt="should not be attached")
    result = json.loads(
        server.mem_save(
            title="CI artefact",
            type="architecture",
            content="## What\nGenerated spec.",
            session_id="s1",
            capture_prompt=False,
        )
    )
    fid = int(result["id"])
    fact = json.loads(server.get_fact(fid))
    # The prompt must NOT be attached when capture_prompt is False.
    assert fact["learned"] in (None, "", "null")
    # The buffer is still consumed? No -- capture_prompt=False should leave it.
    # (Implementation detail: the buffer pop happens inside the `if capture_prompt`
    # branch, so the buffered prompt survives.)


def test_mem_save_uses_topic_key_from_type(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    # Two saves of the same type without explicit topic_key should merge.
    a = json.loads(
        server.mem_save(
            title="First",
            type="pattern",
            content="## What\nFirst observation.",
            project="p",
        )
    )
    b = json.loads(
        server.mem_save(
            title="Second",
            type="pattern",
            content="## What\nSecond observation.",
            project="p",
        )
    )
    # Same topic_key default -> same fact_id -> upsert behaviour.
    # The second save should report "updated" (revision_count > 0).
    assert a["id"] == b["id"]
    assert b["status"] == "updated"


def test_mem_save_with_explicit_topic_key_overrides_default(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    a = json.loads(
        server.mem_save(
            title="A",
            type="pattern",
            content="x",
            topic_key="custom/key",
        )
    )
    b = json.loads(
        server.mem_save(
            title="B",
            type="bugfix",
            content="x",
            topic_key="custom/key",
        )
    )
    # Same explicit topic_key -> same fact_id.
    assert a["id"] == b["id"]


# ---------------------------------------------------------------------------
# mem_search
# ---------------------------------------------------------------------------


def test_mem_search_delegates_to_search_facts(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    server.mem_save(
        title="Login fix",
        type="bugfix",
        content="## What\nFixed N+1 in login handler.",
        project="p",
    )
    results = json.loads(server.mem_search(query="login", project="p"))
    assert isinstance(results, list)
    assert len(results) >= 1
    assert "login" in results[0]["content"].lower()


# ---------------------------------------------------------------------------
# mem_context
# ---------------------------------------------------------------------------


def test_mem_context_returns_facts_for_session(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    # Use distinct explicit topic_keys so each save is a separate fact.
    server.mem_save(
        title="S1 fact",
        type="decision",
        content="## What\nIn session 1.",
        session_id="s1",
        topic_key="s1/fact-a",
    )
    server.mem_save(
        title="S2 fact",
        type="decision",
        content="## What\nIn session 2.",
        session_id="s2",
        topic_key="s2/fact-a",
    )
    s1 = json.loads(server.mem_context(session_id="s1"))
    assert len(s1) == 1
    assert "session 1" in s1[0]["content"].lower()
    s2 = json.loads(server.mem_context(session_id="s2"))
    assert len(s2) == 1
    assert "session 2" in s2[0]["content"].lower()


def test_mem_context_respects_limit(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    # Distinct topic_keys so each save creates a new fact (not an upsert).
    for i in range(5):
        server.mem_save(
            title=f"Fact {i}",
            type="discovery",
            content=f"## What\nNumber {i}.",
            session_id="s",
            topic_key=f"s/fact-{i}",
        )
    rows = json.loads(server.mem_context(session_id="s", limit=3))
    assert len(rows) == 3


def test_mem_context_excludes_deleted(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    a = json.loads(
        server.mem_save(
            title="Will be deleted",
            type="discovery",
            content="x",
            session_id="s",
            topic_key="s/only",
        )
    )
    server.delete_fact(int(a["id"]))
    rows = json.loads(server.mem_context(session_id="s"))
    assert rows == []


def test_mem_context_newest_first(fresh_db_and_buffer) -> None:
    """mem_context returns facts in DESC created_at order."""
    import time as _time
    server = fresh_db_and_buffer
    # Sleep between saves so created_at (second-resolution) differs.
    ids = []
    for i in range(3):
        result = json.loads(
            server.mem_save(
                title=f"Fact {i}",
                type="discovery",
                content=f"## What\nNumber {i}.",
                session_id="s",
                topic_key=f"s/seq-{i}",
            )
        )
        ids.append(int(result["id"]))
        _time.sleep(1.1)  # > 1 second so created_at differs
    rows = json.loads(server.mem_context(session_id="s"))
    assert len(rows) == 3
    # Newest first (highest created_at / fact_id).
    assert "Number 2" in rows[0]["content"]
    assert "Number 1" in rows[1]["content"]
    assert "Number 0" in rows[2]["content"]


def test_mem_context_with_session_summary_returns_structured_layout(fresh_db_and_buffer) -> None:
    """After mem_session_summary, mem_context recovers the summary fact."""
    import time as _time
    server = fresh_db_and_buffer
    server.mem_save(
        title="Earlier decision",
        type="decision",
        content="x",
        session_id="s",
        topic_key="s/decision",
    )
    _time.sleep(1.1)
    server.mem_session_summary(
        session_id="s",
        goal="Ship auto-capture",
        accomplishments=["Aliases added", "Skill written"],
        next_steps=["Open PR"],
    )
    rows = json.loads(server.mem_context(session_id="s"))
    assert len(rows) == 2
    # Summary (created later) appears first.
    content_first = rows[0]["content"]
    assert "## Goal" in content_first
    assert "Ship auto-capture" in content_first


# ---------------------------------------------------------------------------
# End-to-end: full session loop with auto-capture
# ---------------------------------------------------------------------------


def test_full_session_loop_with_prompt_capture(fresh_db_and_buffer) -> None:
    """Simulate a single user turn: prompt captured, decision saved."""
    import time as _time
    server = fresh_db_and_buffer
    session = "agent-2026-06-25-session-001"

    # 1. User prompt buffered.
    server.mem_save_prompt(
        session_id=session,
        prompt="we need to expose facts via FTS5 with Portuguese stemming",
    )

    # 2. Agent decides on the approach.
    server.mem_save(
        title="Use porter stemmer not snowball",
        type="decision",
        content=(
            "## What\nUse Porter stemmer via sqlite3 FTS5.\n\n"
            "## Why\nSnowball requires extra extension; Porter is built-in.\n\n"
            "## Where\nsrc/memento/store/_fts.py\n\n"
            "## Learned\nsqlite3 in Python ships porter by default."
        ),
        project="basiliskode",
        session_id=session,
        topic_key="basiliskode/fts-stemmer",
    )

    # Sleep so the summary's created_at is strictly after the decision.
    _time.sleep(1.1)

    # 3. End of session.
    server.mem_session_summary(
        session_id=session,
        goal="Wire FTS5 stemmer for pt_BR",
        accomplishments=["Decision documented", "Prompt captured"],
        next_steps=["Implement the stemmer in _fts.py"],
        files_touched=["src/memento/store/_fts.py"],
    )

    # 4. Recover the session -- mem_context returns both facts, summary on top.
    rows = json.loads(server.mem_context(session_id=session))
    assert len(rows) == 2
    assert "## Goal" in rows[0]["content"]
    assert "Wire FTS5 stemmer for pt_BR" in rows[0]["content"]  # the goal
    # The decision fact carries the user prompt in metadata.
    decision = next(r for r in rows if "Porter" in r["content"])
    metadata = json.loads(decision["learned"])
    assert "portuguese" in metadata["user_prompt"].lower()

    # 5. A future search can find this decision by keyword.
    found = json.loads(server.mem_search(query="stemmer", project="basiliskode"))
    assert len(found) >= 1


# ---------------------------------------------------------------------------
# mem_session_summary
# ---------------------------------------------------------------------------


def test_mem_session_summary_persists_structured_layout(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    result = json.loads(
        server.mem_session_summary(
            session_id="s",
            goal="Implement auto-capture",
            accomplishments=["Added mem_save_prompt", "Added mem_save"],
            next_steps=["Write skill text", "Open PR"],
            discoveries=["FastMCP runs single-threaded async"],
            files_touched=["src/memento/mcp/server.py -- aliases"],
        )
    )
    fid = int(result["id"])
    fact = json.loads(server.get_fact(fid))
    content = fact["content"]
    # All sections present.
    assert "## Goal" in content
    assert "Implement auto-capture" in content
    assert "## Discoveries" in content
    assert "FastMCP runs single-threaded async" in content
    assert "## Accomplished" in content
    assert "Added mem_save_prompt" in content
    assert "## Next Steps" in content
    assert "Write skill text" in content
    assert "## Relevant Files" in content
    assert "src/memento/mcp/server.py" in content


def test_mem_session_summary_clears_buffer(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    server.mem_save_prompt(session_id="s", prompt="leftover prompt")
    assert "s" in server._prompt_buffer
    server.mem_session_summary(
        session_id="s",
        goal="close",
        accomplishments=[],
        next_steps=[],
    )
    assert "s" not in server._prompt_buffer


def test_mem_session_summary_skips_empty_optional_sections(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    result = json.loads(
        server.mem_session_summary(
            session_id="s",
            goal="Minimal close",
            accomplishments=[],
            next_steps=[],
        )
    )
    fid = int(result["id"])
    fact = json.loads(server.get_fact(fid))
    content = fact["content"]
    assert "## Goal" in content
    # Optional empty sections should NOT appear.
    assert "## Discoveries" not in content
    assert "## Accomplished" not in content
    assert "## Next Steps" not in content
    assert "## Relevant Files" not in content


# ---------------------------------------------------------------------------
# mem_review
# ---------------------------------------------------------------------------


def test_mem_review_list_returns_facts(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    server.mem_save(
        title="Reviewed fact",
        type="decision",
        content="x",
        project="p",
    )
    result = json.loads(server.mem_review(action="list", project="p"))
    assert result["action"] == "list"
    assert result["count"] >= 1
    assert isinstance(result["items"], list)


def test_mem_review_unsupported_action_returns_status(fresh_db_and_buffer) -> None:
    server = fresh_db_and_buffer
    result = json.loads(server.mem_review(action="mark_reviewed"))
    assert result["status"] == "unsupported"
    assert result["action"] == "mark_reviewed"
    assert "list" in result["supported"]
