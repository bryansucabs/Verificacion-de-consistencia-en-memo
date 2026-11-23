# Verificación de Consistencia en Mem0 con NLI

Este es el repositorio de mi tesis para el bachillerato en Ciencia de la
Computación en la UCSP. La idea es agregarle a Mem0 un módulo que detecte
cuando el usuario dice algo que contradice información que ya guardó antes,
usando un modelo de Inferencia de Lenguaje Natural (DeBERTa-v3).

Sin el módulo, Mem0 delega la detección de contradicciones al LLM, que no
es muy bueno detectándolas. Con el módulo, un clasificador NLI decide con un
umbral concreto sobre P(contradicción) y la memoria vieja queda marcada como
desactualizada, en lugar de borrarse.

## Qué agregué al código

Todo lo que agregué o modifiqué está dentro de la carpeta `mem0/` (fork
completo del sistema Mem0 original).

**Archivos nuevos:**
- `mem0/nli/verificador_consistencia.py`: la clase principal. Forma los
  pares, infiere con DeBERTa y decide la acción, más la fase de
  consolidación. Tiene tres mejoras que se pueden activar por separado:
  - **Filtro de entidades:** descarta pares de temas distintos sin llamar
    a DeBERTa.
  - **Marcado `[OUTDATED]`:** la memoria vieja no se borra, se conserva
    marcada como histórica. Así se pueden responder preguntas sobre el
    valor actual y sobre el anterior.
  - **Verificación bidireccional:** una contradicción solo se confirma si
    supera el umbral en las dos direcciones del par.
- `mem0/nli/filtro_entidades.py`: el filtro de entidades. Usa spaCy
  (`en_core_web_sm`) para comparar entidades y sustantivos de los dos textos.
- `mem0/nli/__init__.py`: exporta la clase.

**Archivos modificados:**
- `mem0/memory/main.py`: se crea el verificador al iniciar `Memory` y se
  llama antes de que el LLM decida qué guardar. Las memorias en conflicto
  se marcan o se eliminan según la configuración del verificador.
- `mem0/configs/llms/ollama.py` y `mem0/llms/ollama.py`: se agregan las
  opciones `num_ctx`, `think` y `seed`. Sin `num_ctx`, Ollama recorta el
  prompt de extracción de Mem0, que mide más de 8000 tokens.

## Evaluadores

| Script | Modelo | Embeddings | Qdrant | Variables de entorno |
|---|---|---|---|---|
| `evaluacion_qwen35.py` | Qwen 2.5 7B o Qwen 3.5 4B (Ollama, local) | `nomic-embed-text` | 6333 | ninguna |
| `evaluacion_chatgpt.py` | `gpt-6-luna` | `text-embedding-3-small` | 6335 | `OPENAI_API_KEY` |
| `evaluacion_claude.py` | `claude-haiku-5-5` | `text-embedding-3-small` | 6336 | `ANTHROPIC_API_KEY` y `OPENAI_API_KEY` |

`evaluacion_chatgpt.py` y `evaluacion_claude.py` solo eligen el proveedor;
el trabajo lo hace `evaluacion_api.py`, que es común a los dos.

Todos siguen el mismo protocolo: el historial de Mem0 vive en memoria (ninguna
pregunta hereda mensajes de otra), la respuesta se genera con temperatura 0 y
el propio modelo califica con las plantillas oficiales de LongMemEval, fuera
del tiempo medido.

**Cómo correrlos:**

```
python evaluacion_qwen35.py <oracle|small> <base|modulo> [numero] [--modelo=qwen35|qwen25]
python evaluacion_chatgpt.py <oracle|small> <base|modulo> [numero] [--modelo=...] [--tope=USD] [--puerto=N]
python evaluacion_claude.py  <oracle|small> <base|modulo> [numero] [--modelo=...] [--tope=USD] [--puerto=N]
```

- `base` corre Mem0 sin el módulo y `modulo` con el módulo.
- `numero` es opcional: corre una prueba con ese número de preguntas y la
  guarda en `prueba/`.
- `--tope` es el gasto máximo en dólares (15 por defecto). Si se supera, la
  corrida se detiene y se puede retomar después.

Cada evaluador necesita su propio Qdrant, por ejemplo con Docker:

```
docker run -d -p 6333:6333 qdrant/qdrant
docker run -d -p 6335:6333 qdrant/qdrant
docker run -d -p 6336:6333 qdrant/qdrant
```

## Juez independiente

Para comparar modelos con la misma vara, todas las corridas se califican
también con un juez externo: **Llama 3.3 70B**, servido por DeepInfra, con
las plantillas, la temperatura y el límite de tokens del script oficial de
LongMemEval. Necesita la variable `DEEPINFRA_API_KEY`.

- `juez.py`: arma el prompt y pide el veredicto (1 = correcta, 0 = incorrecta).
- `recalificar.py`: califica una corrida ya terminada y guarda los veredictos
  en la subcarpeta `juez/`, junto a la corrida. No modifica el archivo original
  y, si se interrumpe, continúa donde se quedó.
- `comparar_jueces.py`: compara la calificación del propio modelo con la del
  juez independiente (acuerdo, kappa de Cohen y prueba de McNemar). No tiene
  costo.

```
python recalificar.py <archivo_de_resultados.json>
python comparar_jueces.py
```

## Resultados

Las corridas están en `resultados_pruebas/`:

| Carpeta | Contenido |
|---|---|
| `resultados_local_qwen25_corregido/` | Qwen 2.5 7B con el protocolo actual |
| `resultados_local_qwen_pfc3/` | Qwen 2.5 7B con la versión PFC3 del módulo y el protocolo anterior |
| `resultados_local_qwen35/` | Qwen 3.5 4B con el protocolo actual |
| `resultados_chatgpt/` | `gpt-6-luna`, variante oracle |
| `resultados_openai/` | GPT-5.4-nano con versiones anteriores del módulo (mayo de 2026) |
| `resultados_local_qwen/` y `resultados_qwen/` | Corridas preliminares con Qwen |

En las carpetas del protocolo actual, cada variante (`oracle/` y `small/`)
tiene:
- un JSON por corrida, sin y con módulo;
- `txt/` con la salida de la consola;
- `juez/` con los veredictos del juez independiente.

La exactitud de cada corrida está en el campo `accuracy` del JSON y en el
resumen `.txt` de `juez/`.

## Dataset

El benchmark usado es **LongMemEval** (Wu et al., ICLR 2025), en la categoría
knowledge-update (78 preguntas), con sus variantes oracle y small. Como es
pesado, no lo subo al repo. Se puede descargar de su repositorio oficial.

## Modelo NLI

El módulo usa `cross-encoder/nli-deberta-v3-base` de Hugging Face. Se
descarga automáticamente la primera vez que se ejecuta.

## Parámetros

- `top_k = 10`: memorias candidatas que Qdrant recupera.
- Umbral de similitud `0.70`: descarta pares de temas distintos.
- Umbral de contradicción `0.85`: marca la memoria vieja como desactualizada.
- `top_k = 6` en la consolidación: vecinas semánticas por memoria.

---

Bryan Edward Suca Jaramillo · UCSP · 2026
Asesora: Dra. Graciela Meza Lovón
