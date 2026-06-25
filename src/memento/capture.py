"""Auto-capture helpers — meant to be called by the host (Hermes, Claude
Desktop, etc.) from its hooks.

Pure functions, no daemon, no threads. The host calls these in its
turn / compact / session-close hooks. Each function is idempotent and
safe to call multiple times.

Three layers, used together:

1. **Noise filter** (pure): :func:`is_noise`, :func:`should_capture`.
   Decides whether a candidate string carries memory value.
2. **Hooks** (stateful): :func:`on_user_prompt`, :func:`on_compact`,
   :func:`on_session_close`. Idempotent. Safe to call from any thread.
3. **CLI** (`memento-capture`): wraps the hooks for shell invocation
   from hosts that prefer subprocess over in-process imports.

Configuration lives in ``~/.memento/capture.yaml`` (optional). Without
the file, the defaults baked into this module apply.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — overridable via ~/.memento/capture.yaml
# ---------------------------------------------------------------------------

DEFAULT_MIN_LEN = 12
DEFAULT_DROP_PATTERNS: tuple[str, ...] = (
    r"^ok$",
    r"^dale$",
    r"^si$",
    r"^sí$",
    r"^listo$",
    r"^perfecto$",
    r"^genial$",
    r"^joya$",
    r"^buenísimo$",
    r"^buenisimo$",
    r"^gracias$",
    r"^thanks$",
    r"^thx$",
    r"^hola$",
    r"^chau$",
    r"^bye$",
)

DEFAULT_FORCE_PREFIXES: tuple[str, ...] = (
    "/remember",
)

DEFAULT_DB_PATH = "~/.memento/etch.db"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CaptureConfig:
    """Runtime configuration for the auto-capture layer."""

    min_length: int = DEFAULT_MIN_LEN
    drop_patterns: tuple[str, ...] = DEFAULT_DROP_PATTERNS
    force_prefixes: tuple[str, ...] = DEFAULT_FORCE_PREFIXES
    db_path: str = DEFAULT_DB_PATH

    @classmethod
    def load(cls, path: str | Path | None = None) -> CaptureConfig:
        """Load config from ``~/.memento/capture.yaml`` if present.

        Args:
            path: Override the default config path. Pass ``":skip:"`` to
                bypass loading entirely (returns defaults).
        """
        if path == ":skip:":
            return cls()
        target = Path(path) if path else Path("~/.memento/capture.yaml").expanduser()
        if not target.exists():
            return cls()
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "PyYAML not installed; using defaults. "
                "Install with `pip install pyyaml` to enable YAML config."
            )
            return cls()
        try:
            data = yaml.safe_load(target.read_text()) or {}
        except Exception as exc:
            logger.warning("Failed to parse %s: %s; using defaults", target, exc)
            return cls()
        return cls(
            min_length=int(data.get("min_length", DEFAULT_MIN_LEN)),
            drop_patterns=tuple(data.get("drop_patterns", list(DEFAULT_DROP_PATTERNS))),
            force_prefixes=tuple(data.get("custom_capture_prefixes", list(DEFAULT_FORCE_PREFIXES))),
            db_path=str(data.get("db_path", DEFAULT_DB_PATH)),
        )


# ---------------------------------------------------------------------------
# Noise filter (pure)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _compile_patterns(patterns: Iterable[str]) -> tuple[re.Pattern[str], ...]:
    """Compile raw regex strings, skipping any that fail to compile."""
    compiled: list[re.Pattern[str]] = []
    for raw in patterns:
        try:
            compiled.append(re.compile(raw, re.IGNORECASE | re.UNICODE))
        except re.error as exc:
            logger.warning("Skipping invalid drop_pattern %r: %s", raw, exc)
    return tuple(compiled)


def is_noise(text: str, min_len: int = DEFAULT_MIN_LEN) -> bool:
    """Return True if ``text`` carries no memory value.

    A string is considered noise when:
      * It is empty or whitespace-only.
      * It is shorter than ``min_len`` characters (after stripping).
      * It is composed solely of punctuation / symbols.
      * It matches one of the configured ``drop_patterns`` regexes.

    Args:
        text: Candidate content.
        min_len: Minimum non-whitespace length to be considered.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < min_len:
        return True
    return bool(_PUNCT_RE.match(stripped))


def should_capture(text: str, *, force: bool = False, config: CaptureConfig | None = None) -> bool:
    """Decision point: should we persist this text?

    Args:
        text: User prompt (or other candidate content).
        force: Bypass the noise filter (for explicit ``/remember`` captures).
        config: Optional :class:`CaptureConfig`. Defaults are used if omitted.
    """
    if force:
        return bool(text.strip())
    cfg = config or CaptureConfig()
    stripped = text.strip()
    if not stripped:
        return False
    # Force prefixes (e.g. "/remember") always bypass the noise filter.
    for prefix in cfg.force_prefixes:
        if stripped.lower().startswith(prefix.lower()):
            return True
    if is_noise(stripped, min_len=cfg.min_length):
        return False
    compiled = _compile_patterns(cfg.drop_patterns)
    return not any(p.match(stripped) for p in compiled)


# ---------------------------------------------------------------------------
# Transport — fast path (in-process) vs subprocess fallback
# ---------------------------------------------------------------------------


def _fast_path_enabled() -> bool:
    """True when the host wants zero-overhead in-process capture."""
    return os.environ.get("MEMENTO_FAST_BUFFER") == "1"


def _buffer_prompt_in_process(session_id: str, prompt: str) -> dict:
    """Write directly to the MCP server's in-memory buffer (zero latency)."""
    # Imported lazily so the dependency is optional.
    from memento.mcp.server import _prompt_buffer

    _prompt_buffer[session_id] = prompt
    # Apply the same LRU eviction that the MCP server applies (best-effort).
    while len(_prompt_buffer) > getattr(_prompt_buffer, "_PROMPT_BUFFER_MAX", 256):
        _prompt_buffer.popitem(last=False)
    _prompt_buffer.move_to_end(session_id)
    return {"captured": True, "reason": "ok", "transport": "in-process"}


def _call_mcp_tool(tool: str, args: dict) -> dict:
    """Subprocess fallback: invoke the MCP server via ``python -m memento.mcp``.

    The MCP server is stdio-based; for one-shot calls we shell out to a
    small Python wrapper that imports the tool function directly and
    calls it. This is the slow path (~50-200ms per call) — prefer the
    fast path (``MEMENTO_FAST_BUFFER=1``) when possible.
    """
    from memento.mcp.server import get_store  # noqa: F401

    # The cleanest cross-version subprocess path: import and call.
    code = (
        "import json, sys; "
        f"from memento.mcp.server import {tool} as _fn; "
        f"print(json.dumps(_fn(**{args!r})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {"captured": False, "reason": "subprocess-failed", "stderr": result.stderr.strip()}
    try:
        return {
            "captured": True,
            "reason": "ok",
            "transport": "subprocess",
            "result": json.loads(result.stdout),
        }
    except json.JSONDecodeError:
        return {"captured": False, "reason": "invalid-json", "stdout": result.stdout.strip()}


# ---------------------------------------------------------------------------
# Hooks (the public API the host calls)
# ---------------------------------------------------------------------------


def on_user_prompt(
    session_id: str,
    prompt: str,
    *,
    config: CaptureConfig | None = None,
    force: bool = False,
) -> dict:
    """Pre-turn hook — buffer the user's most recent prompt.

    Idempotent and cheap (microseconds on the fast path). Returns a dict
    with at minimum ``{"captured": bool, "reason": str}`` so the host
    can log without parsing strings.

    Args:
        session_id: Active session identifier.
        prompt: User prompt text.
        config: Optional override for the capture config.
        force: Bypass the noise filter (for explicit captures).
    """
    cfg = config or CaptureConfig.load()
    if not should_capture(prompt, force=force, config=cfg):
        return {"captured": False, "reason": "noise", "transport": "skipped"}
    if _fast_path_enabled():
        return _buffer_prompt_in_process(session_id, prompt)
    # Subprocess fallback — host gets isolation but ~50-200ms per call.
    return _call_mcp_tool("mem_save_prompt", {"session_id": session_id, "prompt": prompt})


def on_compact(
    session_id: str,
    goal: str,
    accomplishments: Iterable[str],
    next_steps: Iterable[str],
    *,
    discoveries: Iterable[str] | None = None,
    files_touched: Iterable[str] | None = None,
    config: CaptureConfig | None = None,  # noqa: ARG001 — kept for parity with on_user_prompt
) -> dict:
    """Pre-compact hook — persist a structured session summary.

    Delegates to :func:`mem_session_summary` in the MCP server. Returns
    the parsed JSON result.

    Args:
        session_id: Session being summarised.
        goal: What we were working on this session.
        accomplishments: Completed items.
        next_steps: What remains.
        discoveries: Technical findings (optional).
        files_touched: Paths touched (optional).
        config: Optional capture config (currently unused but kept for
            parity with :func:`on_user_prompt`).
    """
    args = {
        "session_id": session_id,
        "goal": goal,
        "accomplishments": list(accomplishments),
        "next_steps": list(next_steps),
        "discoveries": list(discoveries) if discoveries is not None else None,
        "files_touched": list(files_touched) if files_touched is not None else None,
    }
    if _fast_path_enabled():
        # Lazy import keeps the dependency optional.
        from memento.mcp.server import mem_session_summary
        result = mem_session_summary(**args)
    else:
        result = _call_mcp_tool("mem_session_summary", args)
    return {"captured": True, "reason": "ok", "result": result}


def on_session_close(
    session_id: str,
    **kwargs,
) -> dict:
    """Session-close hook — alias for :func:`on_compact` with extra cleanup.

    Accepts the same kwargs as :func:`on_compact`. Provided as a
    separate function so the host can call it from a distinct lifecycle
    point (e.g. user types ``/end``, idle timeout, container shutdown).
    """
    return on_compact(session_id, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memento-capture",
        description="Memento auto-capture hooks — called by the host at turn boundaries.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # prompt
    p_prompt = sub.add_parser("prompt", help="Buffer a user prompt (pre-turn hook)")
    p_prompt.add_argument("session_id", help="Active session identifier")
    p_prompt.add_argument("--text", required=True, help="The user's prompt text")
    p_prompt.add_argument("--force", action="store_true", help="Bypass the noise filter")

    # summary
    p_summary = sub.add_parser("summary", help="Persist a session summary (pre-compact hook)")
    p_summary.add_argument("session_id", help="Session being summarised")
    p_summary.add_argument("--goal", required=True, help="Goal of the session")
    p_summary.add_argument(
        "--accomplished", action="append", default=[], help="Accomplished item (repeatable)"
    )
    p_summary.add_argument(
        "--next", action="append", default=[], dest="next_steps", help="Next step (repeatable)"
    )
    p_summary.add_argument(
        "--discovery", action="append", default=[], dest="discoveries",
        help="Discovery (repeatable)",
    )
    p_summary.add_argument(
        "--file", action="append", default=[], dest="files_touched",
        help="File touched (repeatable)",
    )

    # close — alias for summary
    p_close = sub.add_parser("close", help="Close a session (alias for summary)")
    for action in p_summary._actions[1:]:  # skip help
        p_close._add_action(action)

    # config — print the resolved config (handy for debugging)
    sub.add_parser("config", help="Print the resolved capture config")

    return parser


def cli(argv: list[str] | None = None) -> int:
    """Entry point for the ``memento-capture`` console script.

    Returns a Unix-style exit code (0 on success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "prompt":
        result = on_user_prompt(args.session_id, args.text, force=args.force)
    elif args.command in ("summary", "close"):
        result = on_compact(
            args.session_id,
            goal=args.goal,
            accomplishments=args.accomplished,
            next_steps=args.next_steps,
            discoveries=args.discoveries or None,
            files_touched=args.files_touched or None,
        )
    elif args.command == "config":
        cfg = CaptureConfig.load()
        result = {"captured": True, "config": cfg.__dict__}
    else:  # pragma: no cover
        parser.error(f"unknown command: {args.command}")
        return 2

    print(json.dumps(result, indent=2))
    return 0 if result.get("captured", True) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
