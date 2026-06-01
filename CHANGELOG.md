# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] — 2026-06-01

### Added

- **Reflection fact types** — nuevos schemas built-in `observation`, `reflection`, `decision`, `preference`. El extractor LLM los clasifica automáticamente durante la extracción desde el buffer.
- **Recency boost** — `recency_weight=0.15` y `recency_half_life=7.0` en `EtchRetriever`. Facts recientes rankean más alto vía decay exponencial `2^(-days/half_life)`.
- **`get_version()` MCP tool** — Hermes y otros agentes pueden consultar la versión instalada de Memento vía `importlib.metadata`.
- **`PRAGMA optimize` en close** — mantenimiento automático del índice SQLite al cerrar `EtchStore`.

## [1.2.0] — 2026-05-27

### Added

- **Memento Atlas** — nueva capa estructural/source-map con `atlas_maps`, `atlas_regions`, `atlas_edges`.
- Navegación jerárquica: `add_map`, `add_region`, `traverse_path`, `get_subtree` con CTE recursivo.
- Búsqueda FTS5 sobre regiones de Atlas con lazy-loading de contenido.
- Puente `atlas_fact_links`: vincula regiones de Atlas con facts existentes (bidireccional).
- 6 herramientas MCP para Atlas: `create_map`, `read_map`, `list_maps`, `search_map`, `list_regions`, `link_fact`.
- Acciones Hermes `atlas` en `EtchMemoryProvider.handle_tool_call`.
- Export/import v2 con soporte completo de Atlas (v1 backward compatible).
- Snapshots incluyen estado de Atlas.
- 41 tests de Atlas + 5 tests E2E de dispatch Hermes; coverage `_atlas.py` 94%.

## [1.1.0]

### Added

- Hive Memory provenance fields with governed scopes and inbox review lifecycle.
- MCP inbox review tools: `list_inbox`, `promote_fact`, and `reject_fact`.

## [1.0.0]

### Added

- MCP stdio server integration for agent workflows.
- Structured fact fields (`what`, `why`, `where`, `learned`) and automatic project detection.
- Pluggable embedding providers, expanded search fallback, HRR multi-query search, dynamic RRF, exact deduplication, conflict surfacing, circuit breaker protection, auto-eviction, session summaries, and progressive disclosure.

## [0.2.0]

### Added

- SQLite-backed memory store with FTS5 search and optional HRR vector support.
- Retrieval and query classification helpers for local-first agent memory workflows.
- Optional BGE-M3/fastembed integration behind extras.

### Changed

- Package metadata, CI, and quality tooling prepared for broader open-source contribution.

## [0.1.0]

### Added

- Initial Memory Etch package foundation.
- Basic persistent memory APIs and local development workflow.
