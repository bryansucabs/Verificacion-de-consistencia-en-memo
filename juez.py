"""Juez independiente para las respuestas de LongMemEval (knowledge-update).

Un solo juez para todas las corridas: Llama 3.3 70B (Meta), servido por DeepInfra.
Las plantillas, la temperatura y el limite de tokens son los del script oficial
LongMemEval/src/evaluation/evaluate_qa.py.

La clave se lee de la variable DEEPINFRA_API_KEY y nunca se escribe en un archivo.
"""
import hashlib
import os
import re
import sys
import threading
from datetime import datetime, timezone

from openai import OpenAI

NOMBRE_JUEZ = "llama33"
MODELO = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
PROVEEDOR = "DeepInfra"
BASE_URL = "https://api.deepinfra.com/v1/openai"
VARIABLE_CLAVE = "DEEPINFRA_API_KEY"

# Iguales al script oficial de LongMemEval.
TEMPERATURA = 0
MAX_TOKENS = 10

# Solo se reintenta cuando el juez no contesta ni "yes" ni "no".
MAX_INTENTOS = 3

# Plantilla oficial para las preguntas de knowledge-update.
PLANTILLA_KU = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, "
    "the response should be considered as correct as long as the updated answer is the required answer."
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

# Plantilla oficial para las preguntas de abstencion (question_id con "_abs").
PLANTILLA_ABS = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. "
    "The model could say that the information is incomplete, or some other information is given "
    "but the asked information is not."
    "\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes or no only."
)

_cliente = None
_candado = threading.Lock()


def _leer_clave():
    """Busca la clave en el entorno y, si no esta, en las variables de usuario de Windows."""
    clave = os.environ.get(VARIABLE_CLAVE)
    if not clave and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as entorno:
                clave = winreg.QueryValueEx(entorno, VARIABLE_CLAVE)[0]
        except OSError:
            clave = None
    if not clave:
        raise SystemExit(f"Falta la variable {VARIABLE_CLAVE}. Se crea con: setx {VARIABLE_CLAVE} \"la-clave\"")
    return clave.strip()


def _obtener_cliente():
    global _cliente
    with _candado:
        if _cliente is None:
            # El modelo puede tardar mas de 30 s cuando DeepInfra esta cargado.
            _cliente = OpenAI(api_key=_leer_clave(), base_url=BASE_URL, max_retries=4, timeout=180)
    return _cliente


def construir_prompt(question_id, pregunta, correcta, generada):
    """Devuelve el prompt oficial y el nombre de la plantilla usada."""
    if "_abs" in question_id:
        return PLANTILLA_ABS.format(pregunta, correcta, generada), "abstencion"
    return PLANTILLA_KU.format(pregunta, correcta, generada), "knowledge-update"


def leer_veredicto(texto):
    """Lectura estricta: la primera palabra debe ser yes o no. Si no lo es, devuelve None."""
    palabras = re.findall(r"[a-z]+", (texto or "").lower())
    if not palabras:
        return None
    return {"yes": 1, "no": 0}.get(palabras[0])


def juzgar(question_id, pregunta, correcta, generada):
    """Pide un veredicto al juez y devuelve todo lo necesario para poder auditarlo."""
    cliente = _obtener_cliente()
    prompt, plantilla = construir_prompt(question_id, pregunta, correcta, generada)

    for intento in range(1, MAX_INTENTOS + 1):
        respuesta = cliente.chat.completions.create(
            model=MODELO,
            messages=[{"role": "user", "content": prompt}],
            n=1,
            temperature=TEMPERATURA,
            max_tokens=MAX_TOKENS,
        )
        crudo = (respuesta.choices[0].message.content or "").strip()
        veredicto = leer_veredicto(crudo)
        if veredicto is not None:
            break

    uso = respuesta.usage
    return {
        "question_id": question_id,
        "veredicto": veredicto,
        # Regla del script oficial: basta con que aparezca "yes" en la salida.
        "veredicto_regla_oficial": 1 if "yes" in crudo.lower() else 0,
        "salida_cruda": crudo,
        "intentos": intento,
        "plantilla": plantilla,
        "juez": NOMBRE_JUEZ,
        "modelo_pedido": MODELO,
        "modelo_devuelto": respuesta.model,
        "proveedor": PROVEEDOR,
        "temperatura": TEMPERATURA,
        "max_tokens": MAX_TOKENS,
        "sha256_prompt": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "tokens_entrada": getattr(uso, "prompt_tokens", None),
        "tokens_salida": getattr(uso, "completion_tokens", None),
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
