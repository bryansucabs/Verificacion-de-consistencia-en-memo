# Verificación de Consistencia en Mem0 con NLI

Este es el repositorio de mi tesis para el bachillerato en Ciencia de la
Computación en la UCSP. La idea es agregarle a Mem0 un módulo que detecte
cuando el usuario dice algo que contradice información que ya guardó antes,
usando un modelo de Inferencia de Lenguaje Natural (DeBERTa-v3).

Sin el módulo, Mem0 delega la detección de contradicciones al LLM, que no
es muy bueno detectándolas. Con el módulo, uso un clasificador NLI para
decidir con un umbral concreto sobre P(contradicción).

## Qué agregué al código

Todo lo que agregué o modifiqué está dentro de la carpeta `mem0/` (fork
completo del sistema Mem0 original).

**Archivos nuevos:**
- `mem0/nli/verificador_consistencia.py` — la clase principal, tiene los
  tres componentes del módulo (formar pares, inferir con DeBERTa, decidir
  la acción) más la fase de consolidación.
- `mem0/nli/__init__.py` — exporta la clase.

**Archivos modificados:**
- `mem0/memory/main.py` — dos cambios:
  1. En el `__init__` de la clase `Memory` (líneas 387-395) se importa y
     se instancia el verificador.
  2. En el ciclo de escritura, después de que Qdrant devuelve las 10
     memorias candidatas y antes de que el LLM decida (líneas 737-770),
     se llama a `verificar()` para eliminar las memorias contradictorias.

## Cómo correr los evaluadores

Hay dos scripts en la raíz. Los dos hacen lo mismo pero con distinta
configuración de LLM y embeddings.

- `evaluacion_nli.py` — usa Qwen-2.5-7B via Ollama con nomic-embed-text.
  Corre en local. Es la primera configuración que probé.
- `evaluacion_nli_mem0.py` — usa GPT-5.4-nano via OpenAI API con
  text-embedding-3-small. Es la configuración final que usé para los
  resultados oficiales.

Antes de correr, necesitas Qdrant corriendo en `localhost:6333`
(fácil con Docker):

```
docker run -p 6333:6333 qdrant/qdrant
```

Para el evaluador remoto también hace falta exportar la variable de
entorno `OPENAI_API_KEY`.

La bandera `USE_NLI` al inicio de cada script decide si se corre con el
módulo o el baseline sin módulo.

## Resultados

Los resultados están en dos carpetas separadas:

**`resultados_openai/`** — los oficiales de la tesis, con GPT-5.4-nano.
Los que importan son:
- `oracle_gpt_baseline_78.json` — Mem0 sin módulo sobre oracle (65.38%)
- `oracle_gpt_consolidacion_78.json` — Mem0 con el módulo completo sobre
  oracle (71.79%, +6.41 puntos)
- `longmemeval_s_memo_78.json` — Mem0 sin módulo sobre small (67.95%)
- `longmemeval_s_nli_78.json` — Mem0 con el módulo completo sobre small
  (65.38%, −2.57 puntos)

Los demás archivos de esa carpeta son corridas intermedias (con solo el
módulo antiguo sin consolidación, con menos preguntas, sin filtro, etc.)
que dejé como evidencia del proceso.

**`resultados_local_qwen/`** — corridas preliminares con Qwen local. Las
que corresponden a la Tabla 5.1 de la tesis (comparación Qwen vs GPT):
- `resultados_oracle_sin_nli.json` — Qwen baseline (64.10%)
- `resultados_oracle_nli_v2.json` — Qwen con módulo (65.38%, +1.28
  puntos, 97 contradicciones)

Para la variante small con Qwen no llegué a correr la evaluación completa
porque el tiempo estimado era prohibitivo (más de 115 horas). Los demás
archivos de esa carpeta son pruebas exploratorias más chicas.

## Dataset

El benchmark usado fue **LongMemEval** (Wu et al., ICLR 2025). Como es
pesado, no lo subo al repo. Se puede descargar de su repositorio oficial.

## Modelo NLI

El módulo usa `cross-encoder/nli-deberta-v3-base` de Hugging Face. Se
descarga automáticamente la primera vez que se ejecuta.

## Parámetros

Los 4 parámetros del módulo se definen como constantes en el código:
- `top_k = 10` — memorias candidatas que Qdrant recupera
- `umbral de similitud = 0.70` — para descartar pares de temas distintos
- `umbral de contradicción = 0.85` — para eliminar la memoria
- `top_k = 6` en consolidación — vecinas semánticas por memoria

---

Bryan Edward Suca Jaramillo — UCSP · 2026
Asesora: Dra. Graciela Meza Lovón
