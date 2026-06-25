"""Tests for ``memento.capture`` — auto-capture helpers for the host.

Covers:
* Noise filter (``is_noise``, ``should_capture``)
* Force-prefix bypass
* Config loader (``~/.memento/capture.yaml``)
* In-process fast path (``on_user_prompt`` with ``MEMENTO_FAST_BUFFER=1``)
* Pre-compact hook (``on_compact``)
* CLI subcommands
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from memento import capture as cap


def _get_mcp_module():
    """Import the actual server *module* (not the FastMCP instance).

    ``from memento.mcp import server`` returns the FastMCP ``server``
    instance (re-exported from ``__init__``), not the module. We need
    the module to access ``_prompt_buffer`` and the tool functions.
    """
    return importlib.import_module("memento.mcp.server")


@pytest.fixture(autouse=True)
def _fresh_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Reload the MCP server module against a temp DB and clear the buffer."""
    monkeypatch.setenv("MEMENTO_DB_PATH", str(tmp_path / "capture-test.db"))
    importlib.reload(_get_mcp_module())
    _get_mcp_module()._prompt_buffer.clear()
    yield
    _get_mcp_module()._prompt_buffer.clear()


# ---------------------------------------------------------------------------
# is_noise / should_capture
# ---------------------------------------------------------------------------


class TestIsNoise:
    def test_empty_string_is_noise(self):
        assert cap.is_noise("") is True

    def test_whitespace_only_is_noise(self):
        assert cap.is_noise("   \n\t  ") is True

    def test_short_text_is_noise(self):
        # Default min_len=12 — "ok", "dale", "listo" should all be filtered.
        assert cap.is_noise("ok") is True
        assert cap.is_noise("dale") is True
        assert cap.is_noise("listo") is True

    def test_punctuation_only_is_noise(self):
        assert cap.is_noise("???") is True
        assert cap.is_noise("👍👍👍") is True
        assert cap.is_noise("---") is True

    def test_normal_sentence_is_not_noise(self):
        assert cap.is_noise("Use PostgreSQL for the runtime DB.") is False

    def test_technical_decision_is_not_noise(self):
        assert cap.is_noise("Fix N+1 query in UserList") is False


class TestShouldCapture:
    def test_force_bypasses_filter(self):
        # Even "ok" should be captured when force=True.
        assert cap.should_capture("ok", force=True) is True

    def test_noise_dropped_by_default(self):
        assert cap.should_capture("ok") is False
        assert cap.should_capture("dale") is False

    def test_normal_text_captured(self):
        assert cap.should_capture("Use SQLite for the runtime") is True

    def test_force_prefix_bypasses(self):
        # "/remember" prefix always captures, even for short text.
        assert cap.should_capture("/remember ok") is True

    def test_custom_config_respected(self):
        cfg = cap.CaptureConfig(
            min_length=4,
            drop_patterns=(r"^skip me$",),
            force_prefixes=("/mem",),
        )
        assert cap.should_capture("skip me", config=cfg) is False
        assert cap.should_capture("/mem anything", config=cfg) is True


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


class TestConfigLoader:
    def test_defaults_when_explicit_skip(self):
        cfg = cap.CaptureConfig.load(path=":skip:")
        assert cfg.min_length == cap.DEFAULT_MIN_LEN
        assert "/remember" in cfg.force_prefixes

    def test_missing_file_returns_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        cfg = cap.CaptureConfig.load()
        assert cfg.min_length == cap.DEFAULT_MIN_LEN

    def test_yaml_loaded(self, tmp_path: Path):
        pytest.importorskip("yaml", reason="PyYAML optional dep; install memento-etch[capture] or memento-etch[dev]")
        cfg_file = tmp_path / "capture.yaml"
        cfg_file.write_text(
            "min_length: 8\n"
            "drop_patterns:\n"
            "  - '^custom_drop$'\n"
            "custom_capture_prefixes:\n"
            "  - '/note'\n"
            "db_path: /tmp/custom.db\n"
        )
        cfg = cap.CaptureConfig.load(cfg_file)
        assert cfg.min_length == 8
        assert "^custom_drop$" in cfg.drop_patterns
        assert "/note" in cfg.force_prefixes
        assert cfg.db_path == "/tmp/custom.db"

    def test_invalid_pattern_skipped(self, tmp_path: Path):
        cfg_file = tmp_path / "capture.yaml"
        cfg_file.write_text("drop_patterns: ['[unclosed']\n")
        # Should fall back to defaults without raising.
        cfg = cap.CaptureConfig.load(cfg_file)
        assert cfg.min_length == cap.DEFAULT_MIN_LEN


# ---------------------------------------------------------------------------
# on_user_prompt — fast path (in-process)
# ---------------------------------------------------------------------------


class TestOnUserPromptFastPath:
    def test_noise_dropped_fast_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MEMENTO_FAST_BUFFER", "1")
        result = cap.on_user_prompt("s1", "ok")
        assert result["captured"] is False
        assert result["reason"] == "noise"
        assert "s1" not in _get_mcp_module()._prompt_buffer

    def test_normal_prompt_buffered_fast_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MEMENTO_FAST_BUFFER", "1")
        result = cap.on_user_prompt("s2", "Use SQLite FTS5 for search")
        assert result["captured"] is True
        assert _get_mcp_module()._prompt_buffer["s2"] == "Use SQLite FTS5 for search"
        assert result["transport"] == "in-process"

    def test_force_bypasses_filter_fast_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MEMENTO_FAST_BUFFER", "1")
        result = cap.on_user_prompt("s3", "ok", force=True)
        assert result["captured"] is True
        assert _get_mcp_module()._prompt_buffer["s3"] == "ok"


# ---------------------------------------------------------------------------
# on_compact — pre-compact hook
# ---------------------------------------------------------------------------


class TestOnCompact:
    def test_persists_summary_fast_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MEMENTO_FAST_BUFFER", "1")
        result = cap.on_compact(
            session_id="sess-1",
            goal="Wire FTS5 stemmer",
            accomplishments=["Added tokenizer config", "Updated schema"],
            next_steps=["Add tests", "Document in README"],
            discoveries=["Porter stemmer fails on Spanish"],
            files_touched=["src/memento/retrieval.py"],
        )
        assert result["captured"] is True
        # The underlying fact should be in the store now.
        ctx = json.loads(_get_mcp_module().mem_context(session_id="sess-1"))
        assert len(ctx) == 1
        content = ctx[0]["content"]
        assert "## Goal" in content
        assert "Wire FTS5 stemmer" in content
        assert "Added tokenizer config" in content
        assert "Porter stemmer fails on Spanish" in content
        assert "src/memento/retrieval.py" in content

    def test_on_session_close_alias(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MEMENTO_FAST_BUFFER", "1")
        result = cap.on_session_close(
            session_id="sess-2",
            goal="End-to-end test",
            accomplishments=["Done"],
            next_steps=["Nothing"],
        )
        assert result["captured"] is True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "memento.capture", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "memento-capture" in result.stdout

    def test_cli_prompt_drops_noise(self):
        # Without MEMENTO_FAST_BUFFER, the subprocess fallback runs.
        # Noise should still be dropped by the filter.
        result = subprocess.run(
            [
                sys.executable, "-m", "memento.capture", "prompt",
                "cli-test-noise", "--text", "ok",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        body = json.loads(result.stdout)
        assert body["captured"] is False
        assert body["reason"] == "noise"

    def test_cli_config_subcommand(self):
        result = subprocess.run(
            [sys.executable, "-m", "memento.capture", "config"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        body = json.loads(result.stdout)
        assert "config" in body
        assert "min_length" in body["config"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
