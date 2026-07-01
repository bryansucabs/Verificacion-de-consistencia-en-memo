import json
import logging
import time
import warnings
import os
import sys

# Configuracion del experimento (configuracion remota: OpenAI + text-embedding-3-small)
DATASET_PATH    = "E:/proyecto/LongMemEval/data/longmemeval_oracle"
RESULTADOS_PATH = "E:/proyecto/resultados_openai/oracle_gpt_nli_78.json"
CATEGORIA       = "knowledge-update"
MAX_PREGUNTAS   = 78
# True  -> Mem0 + Verificador de Consistencia (incremental + consolidacion)
# False -> Mem0 baseline sin NLI
USE_NLI = True

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

_txt_dir = os.path.join(os.path.dirname(RESULTADOS_PATH), "txt")
os.makedirs(_txt_dir, exist_ok=True)
_txt_path = os.path.join(_txt_dir, os.path.basename(RESULTADOS_PATH).replace(".json", ".txt"))
_log_file = open(_txt_path, "a", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("mem0.nli").setLevel(logging.INFO)
logging.getLogger("mem0.memory.main").setLevel(logging.INFO)
logging.getLogger("mem0.vector_stores.qdrant").setLevel(logging.ERROR)
from mem0 import Memory
from qdrant_client import QdrantClient
from openai import OpenAI

# API key de OpenAI (leer de variable de entorno OPENAI_API_KEY)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta variable de entorno OPENAI_API_KEY")

_openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Configuracion de Mem0: LLM=GPT-5.4-nano, embedder=text-embedding-3-small, vector_store=Qdrant
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-5.4-nano",
            "api_key": OPENAI_API_KEY,
            "temperature": 0
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": OPENAI_API_KEY,
            "embedding_dims": 1536
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 1536
        }
    }
}

# Borra todas las colecciones de Qdrant para empezar limpio cada pregunta
def limpiar_memorias():
    c = QdrantClient("localhost", port=6333)
    for col in c.get_collections().collections:
        c.delete_collection(col.name)

# Carga en Mem0 todos los mensajes role=user de todas las sesiones
def cargar_sesiones(m, sesiones, user_id):
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
            try:
                m.add([{"role": "user", "content": content}], user_id=user_id)
            except Exception as e:
                logging.warning(f"Error al agregar mensaje: {e}")

# Busca memorias en Qdrant y le pregunta a GPT-5.4-nano la respuesta
def responder_pregunta(m, pregunta, user_id):
    t_inicio = time.time()
    memorias = m.search(
        query=pregunta,
        filters={"user_id": user_id},
        top_k=10
    )
    contexto = "\n".join([r["memory"] for r in memorias["results"]])

    print(f"  Memorias recuperadas para responder:")
    for r in memorias["results"]:
        print(f"    - [{r.get('score', 0):.3f}] {r['memory'][:80]}")

    if not contexto.strip():
        return "No relevant memories found.", round(time.time() - t_inicio, 2)

    prompt = (
        "Based on these user memories:\n"
        + contexto
        + "\n\nIMPORTANT: If the memories contain conflicting values for the same fact "
        + "(e.g. an old value and an updated one), always use the MOST RECENT value "
        + "based on the context and dates.\n\n"
        + "Answer this question concisely:\n"
        + pregunta
        + "\n\nAnswer:"
    )

    respuesta = _openai_client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return respuesta.choices[0].message.content, round(time.time() - t_inicio, 2)

# Evalua si la respuesta generada contiene el mismo hecho que la respuesta correcta
def llm_judge(pregunta, respuesta_correcta, respuesta_generada):
    prompt = (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response contains some previous information along with an updated answer, "
        "the response should be considered as correct as long as the updated answer is the required answer.\n\n"
        "Question: " + str(pregunta) + "\n\n"
        "Correct Answer: " + str(respuesta_correcta) + "\n\n"
        "Model Response: " + str(respuesta_generada) + "\n\n"
        "Is the model response correct? Answer yes or no only."
    )

    resultado = _openai_client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    texto = resultado.choices[0].message.content.strip().lower()
    return 1 if "yes" in texto else 0

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

print(f"Evaluando {len(preguntas_ku)} preguntas de '{CATEGORIA}'")
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
tiempo_inicio_total    = time.time()

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

    user_id = f"eval_{item['question_id']}"
    n_mensajes = sum(
        1 for s in item['haystack_sessions']
        for msg in (s if isinstance(s, list) else s.get("messages", []))
        if isinstance(msg, dict) and msg.get("role") == "user"
    )
    print(f"  Mensajes user a cargar: {n_mensajes}")

    # Medir tiempo de carga de sesiones
    t_carga_inicio = time.time()
    cargar_sesiones(m, item['haystack_sessions'], user_id)
    tiempo_carga = round(time.time() - t_carga_inicio, 2)

    # Fase de consolidacion post-carga: limpia contradicciones que el flujo
    # incremental pudo perder, viendo el estado completo de la memoria
    if USE_NLI and m._verificador_nli is not None:
        eliminados = m._verificador_nli.consolidar(
            vector_store=m.vector_store,
            embedding_model=m.embedding_model,
            search_filters={"user_id": user_id},
        )
        if eliminados:
            print(f"  [Consolidacion] {len(eliminados)} memorias obsoletas eliminadas")

    # Contar memorias guardadas en Qdrant
    c_qdrant = QdrantClient("localhost", port=6333)
    memorias_guardadas = sum(
        c_qdrant.get_collection(col.name).points_count
        for col in c_qdrant.get_collections().collections
    )

    # Contar contradicciones detectadas por el modulo NLI durante la carga
    contradicciones = 0
    if USE_NLI and m._verificador_nli is not None:
        contradicciones = m._verificador_nli.contradicciones

    # Responder pregunta y medir latencia real
    respuesta, tiempo_respuesta = responder_pregunta(m, item['question'], user_id)

    # Evaluar con LLM-as-Judge
    score = llm_judge(item['question'], item['answer'], respuesta)
    correctas += score

    # Medir tiempo total de la pregunta
    tiempo_pregunta = round(time.time() - tiempo_inicio, 2)
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
        "tiempo_segundos":      tiempo_pregunta,
        "tiempo_carga_seg":     tiempo_carga,
        "tiempo_respuesta_seg": tiempo_respuesta,
        "memorias_guardadas":   memorias_guardadas,
        "contradicciones":      contradicciones,
    }
    resultados.append(resultado)

    with open(RESULTADOS_PATH, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    accuracy        = correctas / (i + 1)
    tiempo_promedio = sum(tiempos) / len(tiempos)
    print(f"  Correcta:   {item['answer']}")
    print(f"  Generada:   {respuesta[:100]}")
    print(f"  Memorias:   {memorias_guardadas} | Contradicciones: {contradicciones}")
    print(f"  Score: {score} | Accuracy: {accuracy:.2%} | "
          f"Carga: {tiempo_carga:.1f}s | Respuesta: {tiempo_respuesta:.1f}s | Total: {tiempo_pregunta:.1f}s")

# Resumen final
tiempo_total           = time.time() - tiempo_inicio_total
accuracy_final         = correctas / len(preguntas_ku)
tiempo_promedio        = sum(tiempos) / len(tiempos)
tiempo_carga_promedio  = sum(tiempos_carga) / len(tiempos_carga) if tiempos_carga else 0
tiempo_resp_promedio   = sum(tiempos_respuesta) / len(tiempos_respuesta) if tiempos_respuesta else 0
total_contradicciones  = sum(r.get("contradicciones", 0) for r in resultados)
promedio_memorias      = sum(r.get("memorias_guardadas", 0) for r in resultados) / len(resultados)

resumen = {
    "total_preguntas":              len(preguntas_ku),
    "correctas":                    correctas,
    "accuracy":                     round(accuracy_final, 4),
    "accuracy_pct":                 f"{accuracy_final:.2%}",
    "tiempo_promedio_seg":          round(tiempo_promedio, 2),
    "tiempo_carga_promedio_seg":    round(tiempo_carga_promedio, 2),
    "tiempo_respuesta_promedio_seg": round(tiempo_resp_promedio, 2),
    "tiempo_total_min":             round(tiempo_total / 60, 2),
    "total_contradicciones":        total_contradicciones,
    "promedio_memorias_guardadas":  round(promedio_memorias, 2),
    "use_nli":                      USE_NLI,
    "categoria":                    CATEGORIA,
    "detalle":                      resultados,
}

with open(RESULTADOS_PATH, 'w', encoding='utf-8') as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)
print(f"Preguntas evaluadas    : {len(preguntas_ku)}")
print(f"Respuestas correctas   : {correctas}")
print(f"Accuracy               : {accuracy_final:.2%}")
print(f"Tiempo carga promedio  : {tiempo_carga_promedio:.1f} segundos")
print(f"Tiempo respuesta prom. : {tiempo_resp_promedio:.1f} segundos")
print(f"Tiempo total promedio  : {tiempo_promedio:.1f} segundos")
print(f"Total contradicciones  : {total_contradicciones}")
print(f"Memorias prom/pregunta : {promedio_memorias:.1f}")
print(f"Tiempo total           : {tiempo_total/60:.1f} minutos")
print(f"Resultados guardados   : {RESULTADOS_PATH}")
print("="*60)
