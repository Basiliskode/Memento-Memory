"""Ingest pipeline — parse structured data from files/text and import as facts.

Parser functions are module-level generators that yield ``(content, metadata)``
tuples.  The ``EtchStore.ingest()`` method dispatches to the appropriate parser
based on format detection or explicit user choice.

Parsers
-------
- ``parse_markdown`` — split by ``## `` headings
- ``parse_text`` — paragraph, line, or character-chunk delimiters
- ``parse_json`` — list / dict items
- ``parse_csv`` — rows with column headers
- ``detect_format`` — heuristic format detection
"""

import csv
import io
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Any, Generator, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def parse_markdown(text: str) -> Generator[tuple[str, dict], None, None]:
    """Split *text* by ``## `` (level-2+) headings.

    Each section below a heading becomes one fact.  The heading text is
    stored in metadata as ``{"heading": "..."}``.  Content before the first
    heading is treated as front-matter and **skipped**.

    If no ``## `` headings are found, the entire text is emitted as a single
    fact (fallback).
    """
    # Split on lines that start with ##  (level 2 or higher)
    parts = re.split(r"\n(?=##\s)", text)

    if len(parts) <= 1 and not parts[0].startswith("## "):
        # No headings found — fallback: whole text as one fact
        stripped = parts[0].strip()
        if stripped:
            yield stripped, {}
        return

    for part in parts:
        heading_match = re.match(r"##\s+(.+?)(?:\s*\n|$)", part)
        if heading_match is None:
            # Content before the first heading — front matter, skip
            continue
        heading = heading_match.group(1).strip()
        # Everything after the heading line
        body = part[heading_match.end() :].strip()
        if body:
            yield body, {"heading": heading}


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def parse_text(
    text: str,
    delimiter: Union[str, int] = "paragraph",
) -> Generator[tuple[str, dict], None, None]:
    """Split plain text into facts.

    Parameters
    ----------
    text:
        The input text.
    delimiter:
        - ``"paragraph"`` — split by one or more blank lines.
        - ``"line"`` — split by ``\\n``.
        - ``int`` *N* — chunk roughly *N* characters, preferring word
          boundaries.

    Metadata includes ``chunk_index`` and ``total_chunks`` for integer
    delimiters.
    """
    if not text or not text.strip():
        return

    if isinstance(delimiter, int):
        yield from _parse_text_chunks(text, delimiter)
        return

    if delimiter == "paragraph":
        # Split on one or more blank lines (handles \r\n as well)
        raw = re.split(r"\n\s*\n", text)
    elif delimiter == "line":
        raw = text.split("\n")
    else:
        # Unknown delimiter — treat as single block
        stripped = text.strip()
        if stripped:
            yield stripped, {}
        return

    for chunk in raw:
        stripped = chunk.strip()
        if stripped:
            yield stripped, {}


def _parse_text_chunks(
    text: str,
    chunk_size: int,
) -> Generator[tuple[str, dict], None, None]:
    """Split *text* into *chunk_size*-character segments.

    Uses stdlib ``textwrap.wrap`` for word-boundary-aware splitting.
    If a single word is longer than *chunk_size* it is emitted as-is.
    """
    chunks = textwrap.wrap(text, width=chunk_size, break_long_words=True, break_on_hyphens=False)
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        yield chunk, {"chunk_index": i, "total_chunks": total}


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def parse_json(
    data: Union[str, list, dict],
    content_key: Optional[str] = None,
) -> Generator[tuple[str, dict], None, None]:
    """Parse JSON data into facts.

    Parameters
    ----------
    data:
        JSON string, ``list``, or ``dict``.  Strings are parsed with
        ``json.loads`` first.
    content_key:
        If the input is a list of dicts, extract this key as content.
        For dict values, also uses *content_key* if set.

    Metadata includes source type info.
    """
    if isinstance(data, str):
        data = json.loads(data)

    if isinstance(data, list):
        for idx, item in enumerate(data):
            content, meta = _extract_json_item(item, content_key, idx)
            if content:
                yield content, meta

    elif isinstance(data, dict):
        for key, value in data.items():
            if key.startswith("_"):
                continue  # skip internal keys
            if content_key and isinstance(value, dict):
                content = value.get(content_key)
                if content:
                    yield str(content), {
                        "source_key": key,
                        "source_type": "dict_value",
                    }
            elif content_key and isinstance(value, list):
                for idx, sub in enumerate(value):
                    if isinstance(sub, dict) and content_key in sub:
                        yield str(sub[content_key]), {
                            "source_key": key,
                            "source_type": "dict_list_value",
                            "index": idx,
                        }
                    elif isinstance(sub, str):
                        yield sub, {
                            "source_key": key,
                            "source_type": "dict_list_value",
                            "index": idx,
                        }
            else:
                yield str(value), {
                    "source_key": key,
                    "source_type": "dict_value",
                }


def _extract_json_item(
    item: Any,
    content_key: Optional[str],
    idx: int,
) -> tuple[Optional[str], dict]:
    """Extract content + metadata from a single JSON list item."""
    meta: dict = {"index": idx}

    if isinstance(item, str):
        return item, {**meta, "source_type": "string"}

    if isinstance(item, dict):
        if content_key:
            content = item.get(content_key)
            if content:
                return str(content), {**meta, "source_type": "dict_item"}
            return None, {}
        return str(item), {**meta, "source_type": "dict_item"}

    # Numbers, bools, None, etc.
    if item is not None:
        return str(item), {**meta, "source_type": "scalar"}

    return None, {}


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def parse_csv(text: str) -> Generator[tuple[str, dict], None, None]:
    """Parse CSV text into facts.

    First row is treated as column headers.  Each subsequent row becomes
    one fact with content constructed as ``"column: value, col2: value2"``.

    Metadata includes ``row_index``, ``headers`` (list), and ``columns``
    (dict of original column values).
    """
    reader = csv.reader(io.StringIO(text))
    try:
        headers = next(reader, None)
    except StopIteration:
        return

    if not headers:
        return

    headers = [h.strip() for h in headers]

    for idx, row in enumerate(reader):
        if not row or all(cell.strip() == "" for cell in row):
            continue
        # Build content: "header1: val1, header2: val2"
        parts: list[str] = []
        columns: dict[str, str] = {}
        for i, cell in enumerate(row):
            col_name = headers[i] if i < len(headers) else f"col_{i}"
            val = cell.strip()
            if val:
                parts.append(f"{col_name}: {val}")
                columns[col_name] = val

        if parts:
            yield ", ".join(parts), {
                "row_index": idx,
                "headers": headers,
                "columns": columns,
            }


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_TEXT_FORMAT_HINTS = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".json": "json",
    ".csv": "csv",
    ".txt": "text",
}


def detect_format(
    source: Union[str, Path],
    text: Optional[str] = None,
) -> str:
    """Heuristically detect the ingest format from a file path or text content.

    1. If *source* has a recognised file extension (``.md``, ``.json``,
       ``.csv``, ``.txt``), return the corresponding format.
    2. Fall back to inspecting *text* content (starts with ``{`` or ``[`` →
       ``"json"``).
    3. Ultimate fallback: ``"text"``.
    """
    source_str = str(source)
    for ext, fmt in _TEXT_FORMAT_HINTS.items():
        if source_str.lower().endswith(ext):
            return fmt

    # Content-based heuristics
    if text is not None:
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                json.loads(stripped)
                return "json"
            except (json.JSONDecodeError, ValueError):
                pass

    return "text"
