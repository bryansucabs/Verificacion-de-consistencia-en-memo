"""Evaluacion de Mem0 con Qwen 3.5 (4B) local sobre LongMemEval (knowledge-update).

Uso:
    python evaluacion_qwen35.py <oracle|small> <base|modulo> [numero_de_preguntas] [--modelo=qwen35|qwen25]

    oracle | small : variante del benchmark
    base | modulo  : Mem0 sin el Verificador de Consistencia, o con el (version 3)
    numero         : opcional. Si se indica, es una prueba: evalua solo ese numero
                     de preguntas y guarda en la carpeta prueba/, sin mezclarse
                     con las corridas completas.
    --modelo       : opcional. qwen35 (por defecto) usa Qwen 3.5 de 4B y guarda en
                     resultados_local_qwen35/. qwen25 usa Qwen 2.5 de 7B con el mismo
                     protocolo corregido y guarda en resultados_local_qwen25_corregido/.

Diferencias con evaluacion_nli.py (corridas de Qwen 2.5):
  1. El historial de mensajes de Mem0 vive en memoria (":memory:"), de modo que
     ninguna pregunta hereda mensajes de corridas anteriores.
  2. La ventana de contexto se fija en NUM_CTX y se desactiva el modo de
     razonamiento, para que el prompt de extraccion de Mem0 entre completo.
  3. La respuesta se genera con temperatura 0 y semilla fija.
  4. El propio modelo califica con las plantillas oficiales de LongMemEval
     (incluida la de abstencion) y fuera del tiempo medido. El juez
     independiente se aplica despues con recalificar.py.
  5. memorias_guardadas cuenta solo la coleccion principal de Mem0.
"""
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime

# Modelos disponibles. Todos corren con el mismo protocolo corregido; solo cambia
# el modelo y la carpeta de resultados. Se elige con --modelo=<clave> (por defecto qwen35).
#   think: False apaga el modo de razonamiento; None no envia el parametro (modelos sin ese modo).
MODELOS = {
    "qwen35": {"tag": "qwen3.5:4b", "carpeta": "resultados_local_qwen35", "think": False},
    "qwen25": {"tag": "qwen2.5:7b-instruct-q4_K_M", "carpeta": "resultados_local_qwen25_corregido", "think": None},
}

# Configuracion del experimento
_args     = [a for a in sys.argv[1:] if not a.startswith("--")]
_opciones = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
MODELO_CLAVE = _opciones.get("modelo", "qwen35")
if (len(_args) < 2 or _args[0] not in ("oracle", "small") or _args[1] not in ("base", "modulo")
        or MODELO_CLAVE not in MODELOS):
    raise SystemExit(__doc__)

VARIANTE  = _args[0]
# True  -> Mem0 + Verificador de Consistencia (incremental + consolidacion)
# False -> Mem0 baseline sin NLI
USE_NLI   = _args[1] == "modulo"
ES_PRUEBA = len(_args) > 2
MAX_PREGUNTAS = int(_args[2]) if ES_PRUEBA else 78

DATASETS = {
    "oracle": "E:/proyecto/LongMemEval/data/longmemeval_oracle",
    "small":  "E:/proyecto/LongMemEval/data/longmemeval_s",
}
DATASET_PATH = DATASETS[VARIANTE]
CATEGORIA    = "knowledge-update"

MODELO_LLM        = MODELOS[MODELO_CLAVE]["tag"]
THINK             = MODELOS[MODELO_CLAVE]["think"]
_THINK_KW         = {} if THINK is None else {"think": THINK}   # para las llamadas directas a Ollama
CARPETA_RESULTADOS = "E:/proyecto/" + MODELOS[MODELO_CLAVE]["carpeta"]
MODELO_EMBEDDINGS = "nomic-embed-text"
NUM_CTX           = 12288   # el prompt de extraccion de Mem0 mide unos 8250 tokens
SEMILLA           = 42

# Mejoras PFC3 (flags de ablacion, solo aplican si USE_NLI=True):
USE_NER           = True   # Mejora A: filtro de entidades previo al NLI
USE_MARKING       = True   # Mejora B: marcado [OUTDATED] en lugar de borrado
USE_BIDIRECTIONAL = True   # Mejora C: contradiccion confirmada en ambas direcciones

_nombre = f"{VARIANTE}_{'con' if USE_NLI else 'sin'}_modulo_{MODELO_CLAVE}"
if ES_PRUEBA:
    RESULTADOS_PATH = f"{CARPETA_RESULTADOS}/prueba/{_nombre}_{MAX_PREGUNTAS}.json"
else:
    RESULTADOS_PATH = f"{CARPETA_RESULTADOS}/{VARIANTE}/{_nombre}.json"

# Opciones de las llamadas directas a Ollama (respuesta y calificacion propia).
# Se usa el mismo num_ctx que en Mem0 para que Ollama no recargue el modelo.
OPCIONES_RESPUESTA = {"temperature": 0, "seed": SEMILLA, "num_ctx": NUM_CTX}
OPCIONES_JUEZ      = {"temperature": 0, "seed": SEMILLA, "num_ctx": NUM_CTX, "num_predict": 10}

# Un prompt se considera recortado si Ollama leyo muchos menos tokens de los que
# corresponden a su longitud (el prompt de Mem0 ronda los 4.1 caracteres por token).
CARACTERES_POR_TOKEN_MAXIMO = 5.5

# Guarda todo lo que aparece en la terminal (print + logs) en un .txt con el
# mismo nombre del json, dentro de la subcarpeta txt/ del directorio de resultados
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

# Si la consola de Windows no puede mostrar un caracter, lo reemplaza en vez de detener la corrida
for _flujo in (sys.__stdout__, sys.__stderr__):
    try:
        _flujo.reconfigure(errors="replace")
    except Exception:
        pass

_txt_dir = os.path.join(os.path.dirname(RESULTADOS_PATH), "txt")
os.makedirs(_txt_dir, exist_ok=True)
_txt_path = os.path.join(_txt_dir, os.path.basename(RESULTADOS_PATH).replace(".json", ".txt"))
_log_file = open(_txt_path, "a", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

# Sin telemetria de Mem0: la corrida no depende de internet
os.environ.setdefault("MEM0_TELEMETRY", "False")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("mem0.nli").setLevel(logging.INFO)
logging.getLogger("mem0.memory.main").setLevel(logging.INFO)
logging.getLogger("mem0.vector_stores.qdrant").setLevel(logging.ERROR)
try:
    from mem0 import Memory
except ImportError:
    raise SystemExit(
        f"Este Python no tiene Mem0 instalado: {sys.executable}\n"
        f"Usa el Python 3.12 del proyecto:  py -3.12 evaluacion_qwen35.py {' '.join(sys.argv[1:])}"
    )
from qdrant_client import QdrantClient
import ollama

from juez import construir_prompt, leer_veredicto

# Cuenta los errores que Mem0 registra por dentro (por ejemplo, cuando el LLM
# devuelve un JSON invalido). Mem0 no los lanza como excepcion: solo los anota
# y descarta el mensaje, asi que sin este contador pasarian desapercibidos.
class _ContadorErrores(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.total = 0
    def emit(self, record):
        if record.name.startswith("mem0"):
            self.total += 1

contador_errores = _ContadorErrores()
logging.getLogger().addHandler(contador_errores)

# Configuracion de Mem0: LLM=Qwen3.5-4B (Ollama), embedder=nomic-embed-text, vector_store=Qdrant
config = {
    # El historial de mensajes vive en memoria: cada pregunta empieza sin mensajes heredados
    "history_db_path": ":memory:",
    "llm": {
        "provider": "ollama",
        "config": {
            "model": MODELO_LLM,
            "ollama_base_url": "http://localhost:11434",
            "num_ctx": NUM_CTX,
            "think": THINK,
            "seed": SEMILLA,
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": MODELO_EMBEDDINGS,
            "ollama_base_url": "http://localhost:11434",
            "embedding_dims": 768
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768
        }
    }
}

# Borra todas las colecciones de Qdrant para empezar limpio cada pregunta
def limpiar_memorias():
    c = QdrantClient("localhost", port=6333)
    for col in c.get_collections().collections:
        c.delete_collection(col.name)

# Cuenta las llamadas que Mem0 hace al LLM y sus tokens. No cambia lo que se envia.
def medir_llamadas(m, uso):
    chat_original = m.llm.client.chat
    def chat(*args, **kwargs):
        respuesta = chat_original(*args, **kwargs)
        entrada = getattr(respuesta, "prompt_eval_count", None) or 0
        caracteres = sum(len(msg.get("content") or "") for msg in kwargs.get("messages", []))
        uso["llamadas"] += 1
        uso["tokens_entrada"] += entrada
        uso["tokens_salida"] += getattr(respuesta, "eval_count", None) or 0
        if 0 < entrada < caracteres / CARACTERES_POR_TOKEN_MAXIMO:
            uso["llamadas_recortadas"] += 1
        return respuesta
    m.llm.client.chat = chat

# Carga en Mem0 todos los mensajes role=user de todas las sesiones.
# Devuelve cuantos mensajes se enviaron a Mem0 y cuantos fallaron.
def cargar_sesiones(m, sesiones, user_id):
    cargados = 0
    errores = 0
    for sesion in sesiones:
        if isinstance(sesion, list):
            mensajes = sesion
        elif isinstance(sesion, dict):
            mensajes = sesion.get("messages", [])
        else:
            continue

        for msg in mensajes:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "").strip()
            if not content or len(content) < 10:
                continue
            cargados += 1
            try:
                m.add([{"role": "user", "content": content}], user_id=user_id)
            except Exception as e:
                errores += 1
                logging.warning(f"Error al agregar mensaje: {e}")
    return cargados, errores

# Busca memorias en Qdrant y le pregunta a Qwen la respuesta
def responder_pregunta(m, pregunta, user_id):
    t_inicio = time.time()
    memorias = m.search(
        query=pregunta,
        filters={"user_id": user_id},
        top_k=10
    )
    contexto = "\n".join([r["memory"] for r in memorias["results"]])
    usadas = [
        {"memoria": r["memory"], "puntaje": round(r.get("score", 0) or 0, 4)}
        for r in memorias["results"]
    ]

    print(f"  Memorias recuperadas para responder:")
    for r in memorias["results"]:
        print(f"    - [{r.get('score', 0):.3f}] {r['memory'][:80]}")

    if not contexto.strip():
        return "No relevant memories found.", round(time.time() - t_inicio, 2), usadas

    prompt = (
        "Based on these user memories:\n"
        + contexto
        + "\n\nIMPORTANT:\n"
        + "- Memories marked [OUTDATED ...] contain PREVIOUS values that were later "
        + "updated. Use them ONLY if the question asks about a past, previous or "
        + "original value.\n"
        + "- For questions about the CURRENT value, use the memories WITHOUT the "
        + "[OUTDATED] marker. If several current memories conflict, always use the "
        + "MOST RECENT value based on the context and dates.\n\n"
        + "Answer this question concisely:\n"
        + pregunta
        + "\n\nAnswer:"
    )

    respuesta = ollama.chat(
        model=MODELO_LLM,
        messages=[{"role": "user", "content": prompt}],
        options=OPCIONES_RESPUESTA,
        **_THINK_KW,
    )
    return respuesta["message"]["content"], round(time.time() - t_inicio, 2), usadas

# Calificacion del propio modelo, con las mismas plantillas oficiales que usa el
# juez independiente (juez.py). Devuelve el veredicto y la salida textual.
def juez_propio(question_id, pregunta, respuesta_correcta, respuesta_generada):
    prompt, _ = construir_prompt(question_id, pregunta, respuesta_correcta, respuesta_generada)
    resultado = ollama.chat(
        model=MODELO_LLM,
        messages=[{"role": "user", "content": prompt}],
        options=OPCIONES_JUEZ,
        **_THINK_KW,
    )
    texto = (resultado["message"]["content"] or "").strip()
    veredicto = leer_veredicto(texto)
    if veredicto is None:
        # Formato inesperado: se aplica la regla del script oficial
        veredicto = 1 if "yes" in texto.lower() else 0
    return veredicto, texto

# Devuelve la linea de `ollama ps` del modelo: memoria, procesador y contexto cargados
def estado_ollama():
    try:
        salida = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=20).stdout
        for linea in salida.splitlines():
            if linea.startswith(MODELO_LLM):
                return " ".join(linea.split())
    except Exception:
        pass
    return ""

def version_ollama():
    try:
        return subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""

# Limpiar Qdrant antes de empezar para garantizar estado limpio
print("Limpiando Qdrant...")
limpiar_memorias()
print("Qdrant limpio.")

# Evaluacion principal
print("Cargando dataset...")
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

preguntas_ku = [
    d for d in data
    if d['question_type'] == CATEGORIA
][:MAX_PREGUNTAS]

print(f"Evaluando {len(preguntas_ku)} preguntas de '{CATEGORIA}' ({VARIANTE})")
print(f"Modelo: {MODELO_LLM} | num_ctx: {NUM_CTX} | think: {THINK} | semilla: {SEMILLA}")
print(f"Modulo NLI: {'ACTIVADO' if USE_NLI else 'desactivado'}"
      + (f" (NER={USE_NER}, marcado={USE_MARKING}, bidireccional={USE_BIDIRECTIONAL})" if USE_NLI else ""))
print(f"Resultados en: {RESULTADOS_PATH}")
print("="*60)

# Retoma desde donde se quedo si el JSON ya existe
resultados = []
ya_procesados = set()
if os.path.exists(RESULTADOS_PATH):
    try:
        with open(RESULTADOS_PATH, 'r', encoding='utf-8') as f:
            guardado = json.load(f)
        if isinstance(guardado, list):
            resultados = guardado
        elif isinstance(guardado, dict) and "detalle" in guardado:
            resultados = guardado["detalle"]
        ya_procesados = {r["question_id"] for r in resultados}
        print(f"Retomando: {len(ya_procesados)} preguntas ya procesadas, continuando desde donde se quedo.")
    except Exception:
        resultados = []
        ya_procesados = set()

correctas              = sum(r["score"] for r in resultados)
tiempos                = [r["tiempo_segundos"] for r in resultados]
tiempos_carga          = [r.get("tiempo_carga_seg", 0) for r in resultados]
tiempos_respuesta      = [r.get("tiempo_respuesta_seg", 0) for r in resultados]
ollama_ps              = ""

# Calentamiento: carga los modelos (Qwen, embeddings, entidades y NLI) antes de
# medir, para que la primera pregunta no cargue con ese tiempo. Se repite cada
# vez que la corrida se retoma. No deja nada guardado: Qdrant se limpia al final.
def calentar():
    limpiar_memorias()
    m = Memory.from_config(config)
    if not USE_NLI:
        m._verificador_nli = None
    for texto in ("I adopted a dog named Max last week.", "Actually, my dog is called Rocky, not Max."):
        m.add([{"role": "user", "content": texto}], user_id="calentamiento")
    m.search(query="What is the name of my dog?", filters={"user_id": "calentamiento"}, top_k=10)
    ollama.chat(model=MODELO_LLM, messages=[{"role": "user", "content": "Say OK."}],
                options=OPCIONES_RESPUESTA, **_THINK_KW)
    limpiar_memorias()

if len(ya_procesados) < len(preguntas_ku):
    print("Cargando modelos (calentamiento, este tiempo no se mide)...")
    _t_calentamiento = time.time()
    calentar()
    print(f"Modelos listos en {time.time() - _t_calentamiento:.0f} s.")

for i, item in enumerate(preguntas_ku):
    if item['question_id'] in ya_procesados:
        print(f"Pregunta {i+1}/{len(preguntas_ku)}: ya procesada, saltando.")
        continue
    tiempo_inicio = time.time()
    print(f"\nPregunta {i+1}/{len(preguntas_ku)}: {item['question'][:80]}")

    # Limpiar y reiniciar Mem0 para cada pregunta
    limpiar_memorias()
    m = Memory.from_config(config)
    # Si USE_NLI=False desactiva el verificador para correr el baseline
    if not USE_NLI:
        m._verificador_nli = None
    elif m._verificador_nli is not None:
        # Aplicar flags de ablacion PFC3
        m._verificador_nli.usar_filtro_entidades = USE_NER
        m._verificador_nli.usar_marcado = USE_MARKING
        m._verificador_nli.usar_bidireccional = USE_BIDIRECTIONAL

    uso = {"llamadas": 0, "tokens_entrada": 0, "tokens_salida": 0, "llamadas_recortadas": 0}
    medir_llamadas(m, uso)
    errores_mem0_inicio = contador_errores.total

    user_id = f"eval_{item['question_id']}"
    n_mensajes = sum(
        1 for s in item['haystack_sessions']
        for msg in (s if isinstance(s, list) else s.get("messages", []))
        if isinstance(msg, dict) and msg.get("role") == "user"
    )
    print(f"  Mensajes user a cargar: {n_mensajes}")

    # Medir tiempo de carga de sesiones
    t_carga_inicio = time.time()
    mensajes_cargados, errores_carga = cargar_sesiones(m, item['haystack_sessions'], user_id)
    tiempo_carga = round(time.time() - t_carga_inicio, 2)

    # Fase de consolidacion post-carga: limpia contradicciones que el flujo
    # incremental pudo perder, viendo el estado completo de la memoria
    tiempo_consolidacion = 0.0
    if USE_NLI and m._verificador_nli is not None:
        t_consolidacion_inicio = time.time()
        procesadas = m._verificador_nli.consolidar(
            vector_store=m.vector_store,
            embedding_model=m.embedding_model,
            search_filters={"user_id": user_id},
        )
        tiempo_consolidacion = round(time.time() - t_consolidacion_inicio, 2)
        if procesadas:
            accion = "marcadas" if USE_MARKING else "eliminadas"
            print(f"  [Consolidacion] {len(procesadas)} memorias obsoletas {accion}")

    # Contar memorias guardadas: solo la coleccion principal de Mem0.
    # puntos_qdrant_total suma todas las colecciones (incluye la de entidades),
    # que es lo que contaban las corridas anteriores.
    c_qdrant = QdrantClient("localhost", port=6333)
    memorias_guardadas = c_qdrant.get_collection(m.collection_name).points_count
    puntos_qdrant_total = sum(
        c_qdrant.get_collection(col.name).points_count
        for col in c_qdrant.get_collections().collections
    )

    # Contar contradicciones detectadas por el modulo NLI durante la carga
    contradicciones = 0
    memorias_marcadas = 0
    pares_descartados = 0
    if USE_NLI and m._verificador_nli is not None:
        contradicciones = m._verificador_nli.contradicciones
        memorias_marcadas = m._verificador_nli.marcadas
        pares_descartados = m._verificador_nli.pares_descartados_filtro

    # Responder pregunta y medir latencia real
    respuesta, tiempo_respuesta, memorias_usadas = responder_pregunta(m, item['question'], user_id)

    # Tiempo total de la pregunta: carga + consolidacion + respuesta, sin la calificacion
    tiempo_pregunta = round(time.time() - tiempo_inicio, 2)
    errores_mem0 = contador_errores.total - errores_mem0_inicio

    # Calificacion del propio modelo (fuera del tiempo medido)
    t_juez_inicio = time.time()
    score, salida_juez = juez_propio(item['question_id'], item['question'], item['answer'], respuesta)
    tiempo_juez = round(time.time() - t_juez_inicio, 2)
    correctas += score

    tiempos.append(tiempo_pregunta)
    tiempos_carga.append(tiempo_carga)
    tiempos_respuesta.append(tiempo_respuesta)

    # Guardar resultado parcial por si se interrumpe
    resultado = {
        "question_id":          item['question_id'],
        "question":             item['question'],
        "answer_correcta":      item['answer'],
        "respuesta_generada":   respuesta,
        "score":                score,
        "salida_juez_propio":   salida_juez,
        "tiempo_segundos":      tiempo_pregunta,
        "tiempo_carga_seg":     tiempo_carga,
        "tiempo_consolidacion_seg": tiempo_consolidacion,
        "tiempo_respuesta_seg": tiempo_respuesta,
        "tiempo_juez_propio_seg": tiempo_juez,
        "mensajes_cargados":    mensajes_cargados,
        "errores_carga":        errores_carga,
        "errores_mem0":         errores_mem0,
        "memorias_guardadas":   memorias_guardadas,
        "puntos_qdrant_total":  puntos_qdrant_total,
        "contradicciones":      contradicciones,
        "memorias_marcadas":    memorias_marcadas,
        "pares_descartados_filtro": pares_descartados,
        "llamadas_llm":         uso["llamadas"],
        "tokens_entrada":       uso["tokens_entrada"],
        "tokens_salida":        uso["tokens_salida"],
        "llamadas_recortadas":  uso["llamadas_recortadas"],
        "memorias_usadas":      memorias_usadas,
    }
    resultados.append(resultado)

    with open(RESULTADOS_PATH, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    accuracy        = correctas / len(resultados)
    seg_por_mensaje = tiempo_carga / mensajes_cargados if mensajes_cargados else 0
    print(f"  Correcta:   {item['answer']}")
    print(f"  Generada:   {respuesta[:100]}")
    print(f"  Memorias:   {memorias_guardadas} | Contradicciones: {contradicciones} | "
          f"Mensajes: {mensajes_cargados} (errores: {errores_carga}, errores internos de Mem0: {errores_mem0})")
    print(f"  LLM:        {uso['llamadas']} llamadas | {uso['tokens_entrada']} tokens entrada | "
          f"{uso['tokens_salida']} salida | recortadas: {uso['llamadas_recortadas']}")
    print(f"  Score (propio): {score} [{salida_juez[:20]}] | Accuracy: {accuracy:.2%} | "
          f"Carga: {tiempo_carga:.1f}s ({seg_por_mensaje:.1f}s/mensaje) | "
          f"Respuesta: {tiempo_respuesta:.1f}s | Total: {tiempo_pregunta:.1f}s")

    # Estado del modelo en Ollama: se anota una vez para verificar GPU y contexto
    if not ollama_ps:
        ollama_ps = estado_ollama()
        print(f"  Ollama:     {ollama_ps}")
        if "100% GPU" not in ollama_ps or str(NUM_CTX) not in ollama_ps:
            print("  AVISO: el modelo no esta 100% en GPU o el contexto no es el esperado.")

# Resumen final
tiempo_total           = sum(tiempos)
accuracy_final         = correctas / len(preguntas_ku)
tiempo_promedio        = sum(tiempos) / len(tiempos)
tiempo_carga_promedio  = sum(tiempos_carga) / len(tiempos_carga) if tiempos_carga else 0
tiempo_resp_promedio   = sum(tiempos_respuesta) / len(tiempos_respuesta) if tiempos_respuesta else 0
total_contradicciones  = sum(r.get("contradicciones", 0) for r in resultados)
total_marcadas         = sum(r.get("memorias_marcadas", 0) for r in resultados)
total_descartados      = sum(r.get("pares_descartados_filtro", 0) for r in resultados)
promedio_memorias      = sum(r.get("memorias_guardadas", 0) for r in resultados) / len(resultados)
total_mensajes         = sum(r.get("mensajes_cargados", 0) for r in resultados)
total_errores_carga    = sum(r.get("errores_carga", 0) for r in resultados)
total_errores_mem0     = sum(r.get("errores_mem0", 0) for r in resultados)
total_llamadas        = sum(r.get("llamadas_llm", 0) for r in resultados)
total_recortadas       = sum(r.get("llamadas_recortadas", 0) for r in resultados)
total_tokens_entrada   = sum(r.get("tokens_entrada", 0) for r in resultados)
total_tokens_salida    = sum(r.get("tokens_salida", 0) for r in resultados)
seg_por_mensaje_prom   = sum(tiempos_carga) / total_mensajes if total_mensajes else 0

resumen = {
    "total_preguntas":              len(preguntas_ku),
    "correctas":                    correctas,
    "accuracy":                     round(accuracy_final, 4),
    "accuracy_pct":                 f"{accuracy_final:.2%}",
    "juez_de_score":                f"propio ({MODELO_LLM}, plantillas oficiales, temperatura 0)",
    "tiempo_promedio_seg":          round(tiempo_promedio, 2),
    "tiempo_carga_promedio_seg":    round(tiempo_carga_promedio, 2),
    "tiempo_respuesta_promedio_seg": round(tiempo_resp_promedio, 2),
    "tiempo_por_mensaje_guardado_seg": round(seg_por_mensaje_prom, 2),
    "tiempo_total_min":             round(tiempo_total / 60, 2),
    "total_contradicciones":        total_contradicciones,
    "total_memorias_marcadas":      total_marcadas,
    "total_pares_descartados_filtro": total_descartados,
    "promedio_memorias_guardadas":  round(promedio_memorias, 2),
    "total_mensajes_cargados":      total_mensajes,
    "total_errores_carga":          total_errores_carga,
    "total_errores_mem0":           total_errores_mem0,
    "total_llamadas_llm":          total_llamadas,
    "total_llamadas_recortadas":    total_recortadas,
    "total_tokens_entrada":         total_tokens_entrada,
    "total_tokens_salida":          total_tokens_salida,
    "use_nli":                      USE_NLI,
    "use_ner":                      USE_NER,
    "use_marking":                  USE_MARKING,
    "use_bidirectional":            USE_BIDIRECTIONAL,
    "categoria":                    CATEGORIA,
    "variante":                     VARIANTE,
    "configuracion": {
        "modelo":                   MODELO_LLM,
        "embeddings":               MODELO_EMBEDDINGS,
        "num_ctx":                  NUM_CTX,
        "think":                    THINK,
        "semilla":                  SEMILLA,
        "temperatura_respuesta":    OPCIONES_RESPUESTA["temperature"],
        "temperatura_extraccion":   "0.1 (valor por defecto de Mem0)",
        "top_k_busqueda":           10,
        "history_db_path":          config["history_db_path"],
        "dataset":                  DATASET_PATH,
        "ollama_version":           version_ollama(),
        "ollama_ps":                ollama_ps,
        "fecha_fin":                datetime.now().isoformat(timespec="seconds"),
    },
    "detalle":                      resultados,
}

with open(RESULTADOS_PATH, 'w', encoding='utf-8') as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)
print(f"Modelo                 : {MODELO_LLM} ({VARIANTE}, {'con' if USE_NLI else 'sin'} modulo)")
print(f"Preguntas evaluadas    : {len(preguntas_ku)}")
print(f"Respuestas correctas   : {correctas}  (calificacion del propio modelo)")
print(f"Accuracy               : {accuracy_final:.2%}")
print(f"Tiempo carga promedio  : {tiempo_carga_promedio:.1f} segundos")
print(f"Tiempo por mensaje     : {seg_por_mensaje_prom:.2f} segundos por mensaje guardado")
print(f"Tiempo respuesta prom. : {tiempo_resp_promedio:.1f} segundos")
print(f"Tiempo total promedio  : {tiempo_promedio:.1f} segundos")
print(f"Total contradicciones  : {total_contradicciones}")
print(f"Memorias marcadas      : {total_marcadas}")
print(f"Pares descartados NER  : {total_descartados}")
print(f"Memorias prom/pregunta : {promedio_memorias:.1f}")
print(f"Mensajes cargados      : {total_mensajes} (errores: {total_errores_carga}, "
      f"errores internos de Mem0: {total_errores_mem0})")
print(f"Llamadas al LLM (Mem0) : {total_llamadas} (recortadas: {total_recortadas})")
print(f"Tiempo total           : {tiempo_total/60:.1f} minutos")
print(f"Resultados guardados   : {RESULTADOS_PATH}")
print("="*60)
