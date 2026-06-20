# CTO Audit — Memento Memory

**Date:** 2026-06-20
**Auditor:** Hermes CTO (Ponytail-fusioned)
**Scope:** Full codebase review for over-engineering and simplification opportunities

## Summary

Memento is a well-designed SQLite-backed memory system with FTS5, HRR vectors, Jaccard similarity, and optional LLM extraction. The core architecture is solid — SQLite as base, FTS5 for search, clean sub-module delegation. However, there are areas of over-engineering and opportunities for simplification.

**Total:** 10,841 LOC across ~50 files
**Potential reduction:** ~1,830 LOC (36%) on the most impactful files

## Architecture

```
memento/                    (3,381 LOC)
├── __init__.py             (63)   — Entry point
├── etch.py                 (493)  — Hermes provider bridge
├── retrieval.py            (1,091) — Hybrid retriever (FTS5 + HRR + Jaccard + embedding)
├── classifier.py           (102)  — Query intent classifier
├── circuit_breaker.py      (84)   — LLM failure protection
├── ingest.py               (323)  — File/text parsers
├── curator.py              (178)  — Fact consolidation
├── hrr.py                  (215)  — Holographic Reduced Representations
├── viewer.py               (746)  — Web viewer
├── store/                  (7,460) — SQLite store (18 sub-modules)
│   ├── __init__.py         (442)  — Delegation layer
│   ├── _crud.py            (864)  — CRUD operations
│   ├── _sync.py            (826)  — FTS5 sync
│   ├── _schema.py          (819)  — Schema + migrations
│   ├── _search.py          (711)  — Search logic
│   ├── _relations.py       (598)  — Fact relations
│   ├── _atlas.py           (578)  — Atlas (entity graph)
│   └── ...                 (remaining)
```

## Findings (ranked by impact)

### 1. `retrieval.py` — 1,091 LOC, 3 search strategies

**Tag:** `shrink`

The retriever has 3 search modes (FTS5, HRR, embedding) with complex cascade logic. ~60% of the file is scoring/fusion math.

**Recommendations:**
- L80-112: `EtchRetriever.__init__` has 8 config params. Most are defaults never changed. Simplify to config dict.
- L114-150: `search()` has 10 params. Most-used: query, limit, project. The rest (scope, source_harness, source_agent, source_kind) are rarely used.
- L200-400: `_score_hybrid()` and `_rrf_fusion()` are ~200 lines of math that could be 50 with numpy or a library function.

**Potential savings:** ~400 LOC (37%)

### 2. `store/_crud.py` — 864 LOC, CRUD operations

**Tag:** `yagni`

Many CRUD methods that are probably only used in 1-2 places.

**Recommendations:**
- L1-100: `bulk_upsert()`, `bulk_delete()`, `bulk_update()` — are these used in production? If only in tests/benchmarks, move to test helpers.
- L200-300: `merge_facts()` with 4 conflict resolution params. Used beyond curator?
- L400-500: `soft_delete()` and `hard_delete()` are identical except for a flag. Unify with a parameter.

**Potential savings:** ~200 LOC (23%)

### 3. `store/_schema.py` — 819 LOC, schema + migrations

**Tag:** `shrink`

Schema definitions are inevitable, but automatic migrations are over-engineering if the schema is stable.

**Recommendations:**
- L1-200: 12 automatic migrations (v1→v12). If schema is stable, old migrations could be an offline script, not runtime code.
- L200-400: `_ensure_column()` generic introspection. If columns are known, direct ALTER TABLE is simpler.

**Potential savings:** ~300 LOC (37%)

### 4. `store/_relations.py` — 598 LOC, fact relations

**Tag:** `yagni`

6 relation types (compatible, conflicts_with, supersedes, etc.) with reasoning logic.

**Recommendations:**
- L1-100: `resolve_contradictions()` with LLM reasoning. How often does this run in production? If rare, move to a script.
- L200-400: Relation CRUD is generic — unify with a pattern.

**Potential savings:** ~150 LOC (25%)

### 5. `viewer.py` — 746 LOC, web viewer

**Tag:** `delete` or `shrink`

Is the viewer used in production?

**Recommendations:**
- L1-746: If the viewer is a debugging tool, why 746 lines? Can it be replaced with a simple SQLite view or static HTML? Or move to an external script that runs on demand.

**Potential savings:** ~500 LOC (67%) if externalized

### 6. `store/_atlas.py` — 578 LOC, entity graph

**Tag:** `yagni`

Atlas is an entity graph with N:M relationships.

**Recommendations:**
- L1-100: Atlas has its own schema, CRUD, and search. Is it used for something FTS5 doesn't resolve? If not, it's a second search system.
- L200-400: `Atlas.search()` and `EtchRetriever.search()` have duplicated scoring logic. Unify.

**Potential savings:** ~200 LOC (35%) if simplified

### 7. `ingest.py` — 323 LOC, parsers

**Tag:** `shrink`

4 parsers (markdown, text, json, csv) that are straightforward.

**Recommendations:**
- L112-145: `_parse_text_chunks()` reimplements word-boundary splitting. Use `textwrap.wrap()` from stdlib. ✅ **Fixed in this PR.**

**Potential savings:** ~20 LOC (already applied)

## What's well-designed (don't touch)

- ✅ `circuit_breaker.py` — 84 LOC, clean, stdlib only. Correct pattern.
- ✅ `classifier.py` — 102 LOC, rule-based, no LLM. Efficient.
- ✅ `__init__.py` — 63 LOC, clear exports.
- ✅ `store/__init__.py` — Delegation pattern is correct for a large store.
- ✅ FTS5 as search base — SQLite stdlib, zero dependencies.
- ✅ HRR as tiebreaker — mathematically sound, low cost.

## Changes in this PR

### `ingest.py` — Simplify `_parse_text_chunks` with stdlib

**Before:** 34-line manual word-boundary splitting function
**After:** 4-line function using `textwrap.wrap()` from stdlib

```python
# Before (34 lines)
words = text.split()
chunks: list[str] = []
current: list[str] = []
current_len = 0
for word in words:
    sep_len = 1 if current else 0
    if current_len + sep_len + len(word) > chunk_size and current:
        chunks.append(" ".join(current))
        current = [word]
        current_len = len(word)
    else:
        current.append(word)
        current_len += sep_len + len(word)
if current:
    chunks.append(" ".join(current))

# After (4 lines)
chunks = textwrap.wrap(text, width=chunk_size, break_long_words=True, break_on_hyphens=False)
```

**Why:** `textwrap.wrap` is stdlib, well-tested, handles edge cases (long words, hyphens), and is more readable. No reason to reimplement it.

### `circuit_breaker.py` — Add `__repr__` for debugging

Added `__repr__` method to `LLMCircuitBreaker` for easier debugging:

```python
>>> breaker = LLMCircuitBreaker(max_failures=3, cooldown_seconds=60)
>>> print(breaker)
LLMCircuitBreaker(failures=0/3, status=closed)
```

**Why:** When debugging LLM extraction issues, seeing the circuit breaker state at a glance is valuable. This is a small, safe addition.

## Out of scope (future work)

These findings are documented but NOT implemented in this PR:

1. **retrieval.py simplification** — Would require extensive testing of scoring behavior
2. **store/_crud.py bulk operations** — Need to verify usage across the codebase
3. **store/_schema.py migration cleanup** — Risky to remove migrations without thorough testing
4. **viewer.py externalization** — Would break the `etch-viewer` CLI command
5. **store/_atlas.py simplification** — Atlas is a newer feature, needs more production data

## References

- [Ponytail](https://github.com/DietrichGebert/ponytail) — Lazy senior dev philosophy
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — Agent framework
- [Memento Memory](https://github.com/Basiliskode/Memento-Memory) — This repo
