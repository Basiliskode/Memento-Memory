# Integración con Hermes Agent

Esta guía explica cómo configurar Memento como memory provider de cualquier
instalación de Hermes Agent (multi-profile o single-profile).

## TL;DR — instalación automática

```bash
pip install memento-etch[hrr]
git clone https://github.com/Basiliskode/Memento-Memory
cd Memento-Memory
./scripts/install_hermes.sh
```

El script:

1. Verifica que `memento` esté instalado
2. Copia los hooks a `~/.hermes/agent-hooks/`
3. Instala el plugin manifest en cada profile
4. Parchea `config.yaml` (agrega `memory.provider`, `hooks.pre_llm_call`, `hooks.on_session_end`)
5. Parchea `.env` (agrega `MEMENTO_DB_PATH`, `MEMENTO_PROJECT`)
6. Si detecta `MINIMAX_API_KEY` en el ambiente, configura extracción LLM

Después del script, una sola línea:

```bash
hermes gateway restart
```

Y listo. Cada conversación va a guardar su prompt en Memento automáticamente.

## ¿Qué hace cada cosa?

```
Tu mensaje en Telegram
        ↓
Hermes Agent
        ↓ (pre_llm_call hook)
pre-turn.sh → memento_write.py prompt
        ↓
plugin memento → EtchMemoryProvider → SQLite (FTS5 + 34 tablas)
        ↓
[MiniMax-M3 opcional] extrae facts estructurados
        ↓
~/.hermes/memento_active.db  (o por-profile)
```

## Configuración por profile

Cada profile de Hermes Agent tiene su propia DB Memento. El script
detecta profiles automáticamente bajo `~/.hermes/profiles/<name>/` y
configura cada uno con:

- DB aislada: `~/.hermes/profiles/<name>/memento_active.db`
- `MEMENTO_PROJECT=<name>` (tag en cada fact)
- Hooks independientes
- Config independiente

## Variables de entorno (todas opcionales)

| Var | Default | Qué hace |
|---|---|---|
| `MEMENTO_DB_PATH` | `~/.memento/etch.db` | Path al SQLite |
| `MEMENTO_PROJECT` | `basiliskode` | Tag de proyecto |
| `MINIMAX_API_KEY` | (none) | Habilita extracción LLM. Sin esto, solo se guarda el prompt crudo |
| `MINIMAX_BASE_URL` | `https://api.minimax.io/v1` | Endpoint |
| `EXTRACT_MODEL` | `MiniMax-M3` | Modelo a usar |

## Lo que se guarda sin LLM

Sin `MINIMAX_API_KEY`, Memento solo guarda el **prompt crudo** del usuario
como un fact con `category=prompt`. La búsqueda es por FTS5 keyword
matching. Sirve para:

- "¿Qué le dije al agente sobre X?" → buscar por keyword
- Historial completo de conversaciones

## Lo que se agrega con LLM

Con `MINIMAX_API_KEY` configurada, además del prompt crudo, M3 extrae
**facts estructurados** con:

- `category` (project/user_pref/tool/general)
- `importance` (critical/important/useful/trivial)
- `fact_type` (observation/reflection/decision/preference)
- `tags` (separados por coma)

Esto habilita:

- Búsqueda semántica tipo "todo lo que hablamos sobre pricing"
- Facts categorizados y rankeados por importance
- Consolidación automática (LLM detecta duplicados/contradicciones)

Costo aproximado: ~$0.001-0.003 por prompt.

## Instalación manual (paso a paso)

Si preferís hacer cada paso a mano en vez de usar el script:

### 1. Instalar Memento

```bash
pip install "memento-etch[hrr,embeddings]"
```

### 2. Copiar los hooks

```bash
mkdir -p ~/.hermes/agent-hooks
cp scripts/hermes_hooks/* ~/.hermes/agent-hooks/
chmod +x ~/.hermes/agent-hooks/*
```

### 3. Instalar el plugin

```bash
mkdir -p ~/.hermes/plugins/memento
cp plugins/memory/etch/plugin.yaml ~/.hermes/plugins/memento/
```

(El código Python real del plugin vive en el paquete `memento` instalado
via pip — el `plugin.yaml` es solo el manifest que Hermes usa para
descubrirlo.)

### 4. Configurar `config.yaml`

Agregá al final de `~/.hermes/config.yaml` (o por profile):

```yaml
memory:
  provider: memento

hooks:
  pre_llm_call:
    - name: memento-pre-turn
      command: ~/.hermes/agent-hooks/pre-turn.sh
      timeout: 3
      enabled: true
      description: "Capture user prompts to Memento for cross-session memory."
  on_session_end:
    - name: memento-session-end
      command: ~/.hermes/agent-hooks/pre-compact.sh
      timeout: 5
      enabled: true
      description: "Persist session summary to Memento on session boundary."
```

### 5. Configurar `.env`

En `~/.hermes/.env` (o por profile):

```bash
MEMENTO_DB_PATH=/home/TU_USUARIO/.hermes/memento_active.db
MEMENTO_PROJECT=default

# Opcional: extracción LLM
MINIMAX_API_KEY=sk-cp-...
MINIMAX_BASE_URL=https://api.minimax.io/v1
EXTRACT_MODEL=MiniMax-M3
```

### 6. Reiniciar gateway

```bash
hermes gateway restart
```

### 7. Verificar

```bash
echo '{"session_id":"smoke","user_message":"test del installer"}' | \
    python3 ~/.hermes/agent-hooks/memento_write.py prompt _ --from-stdin

sqlite3 ~/.hermes/memento_active.db \
    "SELECT category, content FROM facts ORDER BY fact_id DESC LIMIT 5"
```

Deberías ver tu prompt y (si configuraste LLM) los facts extraídos.

## Solución de problemas

### El hook no dispara

1. Verificá que el gateway se reinició después del cambio
2. Chequeá `/tmp/memento-hook.log` para ver errores del script
3. Verificá que `MEMENTO_DB_PATH` apunta a un directorio escribible

### FTS5 search devuelve 0 resultados

1. SQLite tiene que ser >= 3.9 (todas las distros modernas)
2. Verificá que el path del DB es el mismo entre hook y query
3. Si guardaste con un writer custom, asegurate que haya insertado en
   la tabla virtual `facts_fts` (no solo `facts`)

### Extracción LLM falla

1. Verificá que `MINIMAX_API_KEY` esté bien
2. M3 puede tardar hasta 12s — si la API está sobrecargada, timeout
3. Mientras tanto, el prompt crudo igual se guarda (es best-effort)

### Cada profile se mezcla con otro

Verificá que `MEMENTO_PROJECT` esté configurado distinto en cada
profile. El plugin memento usa este tag para filtrar.

## Tests

Para correr los tests de integración con Hermes (excluidos por default):

```bash
pip install memento-etch[dev]
python -m pytest tests/test_etch_e2e.py tests/test_etch_contract.py -v
```

Requieren tener Hermes Agent instalado y un `HERMES_HOME` válido.