# Hermes Hooks — Memento

Estos scripts se instalan automáticamente con `scripts/install_hermes.sh`. Se
copian a `~/.hermes/agent-hooks/` y se registran en `config.yaml` como
`hooks.pre_llm_call` y `hooks.on_session_end`.

## Archivos

| Archivo | Hook | Función |
|---|---|---|
| `pre-turn.sh` | `pre_llm_call` | Captura el prompt crudo del usuario y lo guarda en Memento. |
| `pre-compact.sh` | `on_session_end` | Persiste un resumen de sesión al cerrar la conversación. |
| `memento_write.py` | (helper) | Escritor Python: usa la API del plugin memento (`add_fact`, `update_fact`) y dispara extracción LLM opcional. |

## Variables de entorno

| Var | Default | Función |
|---|---|---|
| `MEMENTO_DB_PATH` | `~/.memento/etch.db` | Path al SQLite donde se guardan los facts. |
| `MEMENTO_PROJECT` | `basiliskode` | Tag de proyecto para aislar facts entre profiles. |
| `MINIMAX_API_KEY` | (none) | Si está presente, se hace extracción de facts con LLM. |
| `MINIMAX_BASE_URL` | `https://api.minimax.io/v1` | Endpoint OpenAI-compatible. |
| `EXTRACT_MODEL` | `MiniMax-M3` | Modelo a usar para extracción. |

## Uso manual (testing)

```bash
# Capturar un prompt manualmente
echo '{"session_id":"smoke","user_message":"hola mundo test"}' | \
    python3 ~/.hermes/agent-hooks/memento_write.py prompt _ --from-stdin

# Forzar extracción LLM (sin esperar al hook)
python3 ~/.hermes/agent-hooks/memento_write.py extract test-session \
    --text "El usuario prefiere voseo rioplatense"

# Capturar un resumen de sesión
python3 ~/.hermes/agent-hooks/memento_write.py summary test-session \
    --goal "Test del installer" \
    --accomplished "Copiamos hooks" \
    --next "Reiniciar gateway"
```

## Cómo funciona la extracción LLM

`memento_write.py` llama a MiniMax-M3 (o el modelo configurado en
`EXTRACT_MODEL`) **sincrónicamente** después de cada prompt. La llamada
tiene timeout 12s y budget 400 tokens. Si M3 falla, el prompt crudo
igual queda guardado — la extracción es best-effort.

Por cada prompt:
1. Se guarda el prompt crudo (`category=prompt`, importance 0.4)
2. M3 extrae 0-5 facts estructurados
3. Cada fact se guarda con `category=extracted_<topic>`, `importance` mapeada, `tags` enriquecidos

Costo aproximado: ~$0.001-0.003 por prompt.