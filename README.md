<p align="center">
    <img width="2172" height="724" alt="ChatGPT Image 31 may 2026, 21_02_15" src="https://github.com/user-attachments/assets/46a959b1-7347-41cb-933e-5a32d9f362fd" />
<p/>
    
# Memento Memory

**Memoria persistente local-first para agentes AI.** SQLite + FTS5 + HRR vectors + embeddings opcionales.
Sin servicios externos, sin GPU, sin API keys.

---

## Tabla de Contenidos

- [¿Por qué memento?](#por-qué-memento)
- [Instalación](#instalación)
- [Primeros pasos](#primeros-pasos)
- [Hermes Agent — Instalación en 30 segundos](#hermes-agent--instalación-en-30-segundos)
- [Arquitectura](#arquitectura)
- [Características](#características)
- [Embedding Providers](#embedding-providers)
- [MCP Server](#mcp-server)
- [Web Viewer](#web-viewer)
- [Memento Atlas](#memento-atlas-v12)
- [Benchmarks](#benchmarks)
- [API](#api)
- [Contribuir](#contribuir)
- [Licencia](#licencia)

---

## ¿Por qué memento?

Los agentes AI necesitan memoria persistente para ser útiles. Pero las opciones existentes implicaban elegir entre:

- **Dependencia de APIs externas** (Pinecone, OpenAI embeddings) — tu agente deja de funcionar sin internet.
- **Infraestructura pesada** (Chroma, Qdrant, AgentMemory con iii-engine) — 2GB+ de descarga, runtimes externos, config compleja.
- **Archivos JSON artesanales** — crecen como plaga, sin búsqueda, sin estructura.

memento es el punto medio: **SQLite embedded, sin servidores, sin dependencias obligatorias, sin llamadas externas.** Tu información nunca sale de tu máquina.

```
pip install memento-etch
python -c "from memento import EtchStore; s = EtchStore('memory.db'); print('anda')"
```

Eso es todo lo que necesitás para arrancar.

---

## Instalación

```bash
# Mínimo: FTS5 + Jaccard (solo stdlib de Python)
pip install memento-etch

# Recomendado: FTS5 + HRR vectors (necesita numpy)
pip install "memento-etch[hrr]"
pip install "memento-etch[embeddings]"
pip install "memento-etch[mcp]"
pip install "memento-etch[all]"
```

**Requisitos:** Python 3.10-3.12 | Sin GPU | Sin CUDA | Sin runtime externo.

---

## Primeros pasos

```python
from memento import EtchStore, EtchRetriever

# Crear o abrir la base de datos
store = EtchStore("memory.db")

# Guardar hechos
store.add_fact("Python es un lenguaje interpretado", category="tech")
store.add_fact("SQLite soporta FTS5 para búsqueda de texto completo", category="tech")
store.add_fact("FastAPI está construido sobre Starlette", category="tech")

# Guardar con campos estructurados (v1.0)
store.add_fact(
    content="Usar httpx para llamadas HTTP asincrónicas en Python",
    what="Decisión técnica",
    why="httpx tiene mejor soporte de async/await que requests",
    where="src/http_client.py",
    learned="httpx funciona con anyio y trio, no solo asyncio",
)

# Buscar
retriever = EtchRetriever(store)
results = retriever.search("búsqueda de texto completo")
for r in results:
    print(f"[{r['_score']:.2f}] {r['content']}")

# Búsqueda inteligente con fallback automático (v1.0)
results = retriever.search(
    "¿cómo hago requests HTTP en Python?",
    mode="auto",  # FTS5 → HRR multi-query → embeddings (si están configurados)
    limit=5,
)

# Detección automática de proyecto (v1.0)
# Si estás en un repo git, el proyecto se detecta solo del remote origin
store = EtchStore("project.db", project="auto")
```

---

## Hermes Agent — Instalación en 30 segundos

Si usás [Hermes Agent](https://github.com/NousResearch/hermes-agent), tenés
un instalador que configura todo automáticamente: hooks, plugin, variables
de entorno, y (opcionalmente) extracción LLM con MiniMax-M3.

```bash
pip install "memento-etch[hrr,embeddings]"
git clone https://github.com/Basiliskode/Memento-Memory
cd Memento-Memory
./scripts/install_hermes.sh        # detecta profiles y configura cada uno
hermes gateway restart              # una sola línea y listo
```

El script es **idempotente** (lo podés correr varias veces sin romper
nada), **multi-profile** (configura cada `~/.hermes/profiles/<name>/`
por separado), y **zero-config** si ya tenés `MINIMAX_API_KEY` en tu
ambiente. Sin LLM, también funciona — solo guarda el prompt crudo en
lugar de facts extraídos.

Para configuración detallada (instalación manual, troubleshooting,
variables de entorno), ver [`docs/integrations/hermes-agent.md`](docs/integrations/hermes-agent.md).

---

## Arquitectura

```
┌─────────────────────────────────────────────────────┐
│                    Tu Agente AI                       │
├─────────────────────────────────────────────────────┤
│         MCP Server (stdio)  │  Python API            │
├─────────────────────────────────────────────────────┤
│  EtchRetriever                                        │
│  ┌─────────┬──────────┬───────────┬──────────────┐   │
│  │  FTS5   │   HRR    │  Jaccard  │  Embeddings  │   │
│  │ (exact) │(vectors) │ (n-gram)  │ (semántico)  │   │
│  └────┴────┴────┴─────┴────┴──────┴──────┴───────┘   │
│              │           │                            │
│         Reciprocal Rank Fusion (RRF)                  │
│              │                                        │
│  Memento Atlas — navegación estructural por mapas     │
│  (árbol, regiones, edges, búsqueda FTS5 sobre nodos)  │
│              │                                        │
│  EtchStore — SQLite + FTS5 + triggers automáticos     │
│  ┌────────────────────────────────────────────────┐   │
│  │  Facts   │  Atlas   │  Sessions  │  Relations  │   │
│  └────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**Tres capas de búsqueda, sin dependencias externas por defecto:**

| Capa | Qué hace | Costo | Dependencia |
|---|---|---|---|
| **FTS5** | Búsqueda exacta por palabras clave | ~0.05ms | stdlib |
| **HRR** | Similaridad semántica holográfica | ~0.8ms | numpy (opt-in) |
| **Jaccard** | Re-ranking por n-gramas | incluido en HRR | numpy (opt-in) |
| **Embeddings** | Búsqueda semántica densa | ~185ms | fastembed (opt-in) |

Por defecto usa solo FTS5 + Jaccard. Con `pip install memento-etch[hrr]` ganás HRR.
Con `pip install memento-etch[embeddings]` ganás embeddings densos.
Cada nivel es opcional, aditivo, y retrocompatible.

---

## Características

### Core (v0.x)

| Feature | Descripción |
|---|---|
| **FTS5** | Búsqueda de texto completo con triggers auto-sincronizados |
| **HRR vectors** | Representaciones holográficas sin modelos, sin GPU |
| **Jaccard re-rank** | Overlap de n-gramas para ordenar resultados |
| **Soft delete** | Los hechos no se borran, se ocultan |
| **Consolidación activa** | LLM decide ante hechos duplicados o contradictorios |
| **Entity tracking** | N:M entre entidades con tipos y alias |
| **Fact relations** | compatible, conflicts_with, supersedes |
| **Session timeline** | Contexto cronológico por sesión |
| **Web viewer** | SPA en puerto :9120 |
| **Trust scoring** | Puntuación de confianza que se refuerza con retrievals |
| **Topic upsert** | Hechos que evolucionan: mismo topic_key, se actualizan |

### v1.0

| Feature | Descripción |
|---|---|
| **MCP Server** | 15 tools vía stdio (facts, atlas, search, timeline, similar, inbox review) |
| **Structured facts** | Campos what/why/where/learned para memorias disciplinadas |
| **Project detection** | Detecta el proyecto desde git remote automáticamente |
| **Embedding providers** | Pluggable: NoopProvider, FastembedProvider, OllamaProvider |
| **Search expanded** | FTS5 con expansión progresiva (full query → OR → single terms) |
| **HRR multi-query** | Búsqueda paralela con variaciones semánticas de la query |
| **Dynamic RRF** | k adaptativo según cantidad de resultados |
| **Fallback chain** | Modo "auto" que cascada FTS5 → HRR → embeddings |
| **SHA-256 dedup** | Deduplicación exacta con ventana de 60s |
| **Conflict surfacing** | Detecta hechos similares al insertar y muestra conflictos |
| **Circuit breaker** | Protege contra fallos en cadena de LLM externos (3 fallos, 60s cooldown) |
| **Auto-eviction** | Elimina facts stale (trust < 0.1 o 30 días sin retrieve) |
| **Session summaries** | Genera resúmenes estructurados de sesiones |
| **Progressive disclosure** | Search devuelve resumen (200 chars), get_fact_full() da el contenido completo |

---

## Proveedores de Embeddings

Tres modos de búsqueda semántica, plug and play:

```python
# 1. Sin embeddings (FTS5 + HRR, cero overhead)
store = EtchStore("memory.db")  # NoopProvider por defecto

# 2. Con fastembed (local, ONNX, sin API key)
#    pip install memento-etch[embeddings]
from memento.embedding import FastembedProvider
store = EtchStore("memory.db", embedding_provider=FastembedProvider())

# 3. Con Ollama (si ya tenés Ollama corriendo)
from memento.embedding import OllamaProvider
store = EtchStore("memory.db", embedding_provider=OllamaProvider(
    base_url="http://localhost:11434",
    model="nomic-embed-text",
))
```

Cada provider se puede usar en cualquier combinación con el MCP server.

---

## MCP Server

Para integrar memento con cualquier agente que soporte MCP (Claude Code, Codex, Gemini CLI, etc.):

```bash
pip install "memento-etch[mcp]"

# Con variable de entorno
set MEMENTO_DB_PATH=./memory.db
python -m memento.mcp
```

Herramientas MCP expuestas (15 tools):

| Grupo | Tools |
|-------|-------|
| **Facts** | `add_fact`, `search_facts`, `get_fact`, `delete_fact`, `get_timeline`, `search_similar` |
| **Inbox** | `list_inbox`, `promote_fact`, `reject_fact` |
| **Atlas** | `create_map`, `read_map`, `list_maps`, `search_map`, `list_regions`, `link_fact` |

Configuración vía `MEMENTO_DB_PATH`. Si no está definida, el servidor usa `:memory:` como default; para uso persistente, seteá una ruta explícita como `./memory.db` o `~/.memento/etch.db`.

## Hive Memory (v1.1)

Facts con provenance y scopes gobernados, más un ciclo de revisión vía inbox. Cada fact puede llevar identidad de origen (`source_harness`, `source_agent`, `source_kind`) y un scope que controla su descubribilidad.

### Ámbitos

| Scope | Búsqueda por defecto | Caso de uso |
|-------|---------------------|-------------|
| `canonical` | ✅ Incluido | Memoria de proyecto confiable |
| `inbox` | ❌ Excluido | Escrituras no confiables / de subagentes esperando revisión |
| `personal` | ❌ Excluido | Facts privados del usuario |
| `ephemeral` | ❌ Excluido | Datos transitorios |

**Ejemplo de provenance:**

```python
store.add_fact(
    "FastMCP reintenta en timeout",
    source_harness="opencode",
    source_agent="worker-1",
    source_kind="manual",
    scope="canonical",
)
```

### Flujo de inbox

```python
# Listar facts pendientes en inbox
inbox = store.list_inbox(project="mi-proyecto", limit=20)

# Promover a canonical (se vuelve buscable)
store.promote_fact(inbox[0]["fact_id"])

# Rechazar (soft-delete, oculto de búsqueda por defecto)
store.reject_fact(inbox[3]["fact_id"], reason="baja calidad")
```

Sin dependencias nuevas. Funciona con los callers existentes de `add_fact` — los args de provenance son opcionales, el scope por defecto es `canonical`. Ver [`docs/api/store.md`](docs/api/store.md) para la API completa.

---

## Memento Atlas (v1.2)

**Atlas es la capa estructural de Memento.** Mientras los facts capturan *qué* se dijo/decidió (memoria operacional atómica), Atlas captura *dónde* vive esa información — estructura de documentos, jerarquías de proyectos, mapas de conocimiento navegables.

Atlas **complementa** facts, no los reemplaza. Ambos conviven en la misma DB y pueden vincularse entre sí.

### Conceptos

| Concepto | Descripción |
|----------|-------------|
| **Map** | Contenedor top-level (ej: "README Memento", "Especificación API", "Arquitectura del proyecto") |
| **Region** | Nodo jerárquico dentro de un mapa — puede tener hijos, forma un árbol |
| **Edge** | Relación estructurada entre regiones (contiene, importa, extiende, referencia) |
| **Fact link** | Puente entre una región de Atlas y un fact de Memento |

### Inicio rápido

```python
from memento import EtchStore, EtchRetriever

store = EtchStore("memory.db")
retriever = EtchRetriever(store)

# Crear un mapa (ej: documentación del proyecto)
store.add_map("Memento Docs", description="Documentación técnica de Memento")

# Agregar regiones jerárquicas
store.add_region("Arquitectura", map_id=1, parent_id=None)
store.add_region("API Reference", map_id=1, parent_id=1)  # hija de Arquitectura
store.add_region("EtchStore", map_id=1, parent_id=2)      # sub-región

# Vincular una región con un fact existente
store.link_fact(region_id=3, fact_id=42, relationship="annotates")

# Navegar el árbol
tree = retriever.traverse_path(start_region_id=1, end_region_id=3)
for r in tree:
    print(f"{'  ' * r['depth']}{r['name']} — {r.get('summary', '')}")

# Buscar en Atlas (FTS5 sobre nombres + summaries de regiones)
results = retriever.search_map("arquitectura API")
for r in results:
    print(f"[{r['_score']:.2f}] {r['name']} ({r['kind']})")
```

### Herramientas MCP de Atlas

| Tool | Propósito |
|------|-----------|
| `create_map(name, description, project)` | Crear un nuevo mapa |
| `read_map(map_id)` | Obtener mapa con todas sus regiones |
| `list_maps(project, limit)` | Listar mapas existentes |
| `search_map(query, project, limit)` | Buscar en regiones vía FTS5 |
| `list_regions(map_id)` | Listar regiones de un mapa |
| `link_fact(region_id, fact_id, relationship)` | Vincular región con fact |

### ¿Cuándo usar Atlas vs Facts?

| Situación | Usar |
|-----------|------|
| "¿Qué decidimos sobre X?" | Facts (memoria operacional) |
| "¿Dónde está documentada la función Y?" | Atlas (navegación estructural) |
| "¿Por qué elegimos SQLite y dónde está explicado?" | Facts + Atlas (fact con link a región del README) |
| "Mostrame la estructura del proyecto" | Atlas (árbol de regiones) |

---

Benchmark integrado para medir recall@k con dataset sintético y juez Gemini:

```bash
# Requiere GEMINI_API_KEY
export GEMINI_API_KEY="..."

# Benchmark memento (FTS5 + HRR)
python -m memento.benchmark --verbose

# Benchmark contra baseline JSON (para comparar)
python -m memento.benchmark --provider json-baseline --verbose

# Personalizar dataset
python -m memento.benchmark --n-docs 500 --seed 42 --output results.json
```

Para benchmarkear OTRO sistema de memoria contra el mismo benchmark,
implementá ``MemoryProvider``:

```python
from memento.benchmark import MemoryProvider, BenchmarkRunner

class MyMemory(MemoryProvider):
    name = "mi-sistema"
    def ingest(self, documents): ...
    def retrieve(self, query, k=10, user_id=None): ...

runner = BenchmarkRunner(MyMemory())
results = runner.run(verbose=True)
print(f"Accuracy: {results['accuracy']:.1%}")
```

Resultado de referencia (100 docs, 18 queries):

| Provider | Accuracy | Avg retrieve |
|---|---|---|
| memento (FTS5 + HRR) | **94.4%** (17/18) | **5.2ms** |
| JSON baseline (word overlap) | ~40% | ~0.1ms |

---

## Web Viewer
<p align="center">
<img width="1080" height="1080" alt="Diseño sin título (3)" src="https://github.com/user-attachments/assets/297c461c-b7dc-4fe3-9ace-aed647b774ca" />
<p/>
Visualizá toda la memoria de tu agente en un SPA local, sin servidores, sin config.

```bash
python -m memento.viewer --db ./memory.db
# http://127.0.0.1:9120
```

**Qué ves:**

| Feature | Para qué sirve |
|---|---|
| **Buscador** | Buscá facts por contenido, proyecto, o categoría |
| **Timeline** | Cronología por sesión — qué pasó y cuándo |
| **Relaciones** | Facts conectados: compatible, conflicts_with, supersedes |
| **Metadata** | trust_score, retrieval_count, categoría, proyecto |
| **Soft delete** | Facts archivados no se pierden, se ocultan |

**Combinado con la DB versionable:**

```bash
# Compartí la misma memoria con tu equipo
git add memory.db
git commit -m "seed data: 500 facts de referencia"
git push

# Otro dev hace pull y abre el viewer
git pull
python -m memento.viewer --db memory.db
# → ve exactamente los mismos facts, relaciones, timeline
```

Útil para debuggear el estado de un agente, revisar qué facts acumuló, o compartir datasets de prueba con el equipo.

---

## Benchmarks

### Benchmark sintético (100 documentos, 18 queries)

| Modo | Recall | Latencia | Dependencias |
|---|---|---|---|
| FTS5 + HRR (search_expanded + re-score) | **94.4%** (17/18) | **5.2ms** | numpy |
| Solo FTS5 raw | ~5% | ~0.05ms | stdlib |
| Con embeddings (BGE-small) | ~72% | ~185ms | fastembed + 65MB |

Benchmark reproducible:

```bash
set GEMINI_API_KEY=...
pip install "memento-etch[hrr]"
python scripts/run_amb_benchmark.py --n-docs 100 --verbose
```

### Benchmarks en producción (VPS con facts reales de agente)

| Métrica | FTS5 solo | FTS5 + HRR | Embeddings densos |
|---|---|---|---|
| Coverage @100 facts | 39.2% | **69.7%** | 72% |
| Latencia por query | ~0.05ms | **~0.8ms** | ~185ms |
| Dependencias extra | ninguna | numpy | fastembed + ONNX |

HRR es 200-400x más rápido que embeddings densos con ~97% de su cobertura.

---

## API

Documentación detallada en [`docs/api/`](docs/api/):

- **[EtchStore](docs/api/store.md)** — Core SQLite: CRUD, FTS5, HRR, sesiones, relaciones, consolidación.
- **[EtchRetriever](docs/api/retrieval.md)** — Búsqueda híbrida: FTS5 + HRR + Jaccard + embeddings con RRF.
- **[QueryClassifier](docs/api/classifier.md)** — Clasificador rule-based para rutear estrategias de búsqueda.

---

## Auto-Capture (v1.3)

Wire Memento into your host's turn lifecycle so prompts and pre-compact summaries are saved automatically. The agent still calls `mem_save` for explicit decisions; auto-capture fills in the rest.

### Tres hooks

| When | What to call | CLI equivalent |
|---|---|---|
| Every user message | `on_user_prompt(session_id, text)` | `memento-capture prompt <s> --text "..."` |
| Before context compaction | `on_compact(session_id, goal=..., accomplishments=..., next_steps=...)` | `memento-capture summary <s> --goal ... --accomplished ...` |
| Session close | `on_session_close(session_id, ...)` | `memento-capture close <s> ...` |

### Inicio rápido (Python)

```python
import os
os.environ["MEMENTO_FAST_BUFFER"] = "1"  # zero-latency in-process path

from memento.capture import on_user_prompt, on_compact, on_session_close

# Every turn — buffer the user's prompt
on_user_prompt("sess-1", user_text)

# Before compacting — persist a structured summary
on_compact(
    "sess-1",
    goal="Decide on runtime DB",
    accomplishments=["Picked PostgreSQL", "Wrote migration plan"],
    next_steps=["Implement connection pooling"],
    discoveries=["SQLite WAL was the bottleneck"],
    files_touched=["apps/runtime/db.py"],
)

# Session end — same args, different lifecycle intent
on_session_close("sess-1", goal="...", accomplishments=["..."])
```

### Inicio rápido (CLI)

```bash
# Pre-turn
memento-capture prompt sess-1 --text "Use SQLite FTS5 for retrieval"

# Pre-compact
memento-capture summary sess-1 \
  --goal "Wire FTS5 stemmer" \
  --accomplished "Added tokenizer config" \
  --next "Add tests" \
  --discovery "Porter fails on Spanish" \
  --file src/memento/retrieval.py

# Inspect resolved config
memento-capture config
```

### Filtro de ruido

By default, short confirmations (`ok`, `dale`, `listo`, emojis-only) are dropped before reaching Memento. Override via `~/.memento/capture.yaml`:

```yaml
min_length: 12
drop_patterns:
  - "^ok$"
  - "^dale$"
custom_capture_prefixes:    # bypass filter when prompt starts with:
  - "/remember"
  - "/note"
```

See `examples/auto-capture-loop.md` for the end-to-end walk-through.

---

## Contribuir

```bash
git clone https://github.com/Basiliskode/Memento-Memory
cd Memento-Memory
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Todos los PRs son bienvenidos. Usamos conventional commits y TDD estricto.

---

## Licencia

MIT. Construí algo útil.

---

> memento nació dentro de un agente AI real que necesitaba acordarse de las cosas sin depender de servicios externos. Hoy corre en producción y está probado con miles de facts.
>
> Si estás construyendo un agente que necesite memoria, probalo. Son 30 segundos.
>
> ```bash
> pip install "memento-etch[hrr]"
> python -c "from memento import EtchStore; s = EtchStore('test.db'); print('anda')"
> ```
