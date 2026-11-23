"""Evaluacion de Mem0 con un modelo remoto (OpenAI o Anthropic) sobre LongMemEval.

Usa el mismo protocolo corregido de evaluacion_qwen35.py: historial de Mem0 en
memoria, misma carga de mensajes, mismo texto para pedir la respuesta, juez
propio con las plantillas oficiales fuera del tiempo medido y los mismos campos
en el JSON. No se ejecuta directamente: se lanza con evaluacion_chatgpt.py o
evaluacion_claude.py.

    py -3.12 evaluacion_chatgpt.py <oracle|small> <base|modulo> [numero] [opciones]
    py -3.12 evaluacion_claude.py  <oracle|small> <base|modulo> [numero] [opciones]

    numero          : opcional. Prueba con ese numero de preguntas; guarda en prueba/.
    --modelo=NOMBRE : otro modelo del mismo proveedor (debe tener precios en PRECIOS).
    --tope=USD      : gasto maximo de la corrida en dolares (por defecto 15). Al
                      superarlo, la corrida se detiene y se puede retomar despues.
    --puerto=N      : puerto de Qdrant. Cada proveedor usa su propio Qdrant para
                      no interferir con las corridas de Qwen, que usan el 6333.

Mem0 no se modifica: despues de crear cada Memory se reemplaza su funcion
generate_response por la de este archivo, que controla los parametros de cada
proveedor y registra tokens y costo.
"""
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime

PROVEEDORES = {
    "chatgpt": {"modelo": "gpt-6-luna",       "carpeta": "resultados_chatgpt", "clave": "OPENAI_API_KEY",    "puerto": 6335},
    "claude":  {"modelo": "claude-haiku-5-5", "carpeta": "resultados_claude",  "clave": "ANTHROPIC_API_KEY", "puerto": 6336},
}

# Dolares por millon de tokens. Consultados en noviembre de 2026: VERIFICAR antes de gastar.
PRECIOS = {
    "gpt-6-luna":       {"entrada": 0.10, "cache_lectura": 0.01, "cache_escritura": 0.125, "salida": 0.50},
    "gpt-5.4-nano":     {"entrada": 0.20, "cache_lectura": 0.02, "cache_escritura": 0.20,  "salida": 1.25},
    "claude-haiku-5-5": {"entrada": 0.10, "cache_lectura": 0.01, "cache_escritura": 0.125, "salida": 0.50},
}

MODELO_EMBEDDINGS = "text-embedding-3-small"   # de OpenAI, tambien para Claude
DIMENSIONES       = 1536
CATEGORIA         = "knowledge-update"
SEMILLA           = 42
MAX_ERRORES_SEGUIDOS = 5
DATASETS = {
    "oracle": "E:/proyecto/LongMemEval/data/longmemeval_oracle",
    "small":  "E:/proyecto/LongMemEval/data/longmemeval_s",
}
# Mejoras PFC3 (flags de ablacion, solo aplican con el modulo activo)
USE_NER = True
USE_MARKING = True
USE_BIDIRECTIONAL = True


class CorridaDetenida(BaseException):
    """Detiene la corrida completa. No hereda de Exception para que Mem0 no la absorba."""


def leer_clave(nombre):
    """Busca la clave en el entorno y, si no esta, en las variables de usuario de Windows."""
    clave = os.environ.get(nombre)
    if not clave and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as entorno:
                clave = winreg.QueryValueEx(entorno, nombre)[0]
        except OSError:
            clave = None
    if not clave:
        raise SystemExit(f"Falta la variable {nombre}. Se crea con: setx {nombre} \"la-clave\"")
    os.environ[nombre] = clave.strip()
    return os.environ[nombre]


class ClienteLLM:
    """Llama al modelo remoto, cuenta tokens y costo, y corta la corrida si algo va mal."""

    # Parametros que algunos modelos rechazan. Si la API devuelve un error que
    # nombra uno de ellos, se deja de enviar y se anota en `omitidos`.
    OPCIONALES = ("temperature", "top_p", "reasoning_effort", "thinking", "response_format")

    def __init__(self, proveedor, modelo, tope_usd, gastado_antes):
        self.proveedor, self.modelo = proveedor, modelo
        self.precio = PRECIOS[modelo]
        self.tope_usd, self.gastado = tope_usd, gastado_antes
        self.omitidos, self.errores_seguidos, self.uso = set(), 0, None
        if proveedor == "chatgpt":
            import openai
            self.sdk = openai
            self.cliente = openai.OpenAI(api_key=leer_clave("OPENAI_API_KEY"), max_retries=6, timeout=120)
        else:
            try:
                import anthropic
            except ImportError:
                raise SystemExit("Falta el paquete de Anthropic. Se instala con: py -3.12 -m pip install anthropic")
            self.sdk = anthropic
            self.cliente = anthropic.Anthropic(api_key=leer_clave("ANTHROPIC_API_KEY"), max_retries=6, timeout=120)

    def _enviar(self, crear, parametros):
        """Envia la peticion; si el modelo rechaza un parametro opcional, lo quita y reintenta."""
        for _ in range(len(self.OPCIONALES) + 1):
            for nombre in self.omitidos:
                parametros.pop(nombre, None)
            try:
                return crear(**parametros)
            except self.sdk.BadRequestError as error:
                texto = str(error).lower()
                culpable = next((n for n in self.OPCIONALES if n in parametros and n in texto), None)
                if culpable is None:
                    raise
                self.omitidos.add(culpable)
                print(f"  [API] {self.modelo} no acepta '{culpable}': se deja de enviar.", flush=True)
        raise RuntimeError("La API rechazo la peticion incluso sin los parametros opcionales")

    def _openai(self, mensajes, max_tokens, temperatura, top_p, formato_json):
        parametros = {"model": self.modelo, "messages": mensajes, "max_completion_tokens": max_tokens,
                      "reasoning_effort": "none"}   # sin razonamiento previo
        if temperatura is not None:
            parametros["temperature"] = temperatura
        if top_p is not None:
            parametros["top_p"] = top_p
        if formato_json:
            parametros["response_format"] = {"type": "json_object"}
        r = self._enviar(self.cliente.chat.completions.create, parametros)
        u = r.usage
        lectura = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        return (r.choices[0].message.content or ""), u.prompt_tokens - lectura, lectura, 0, u.completion_tokens

    def _anthropic(self, mensajes, max_tokens, temperatura, top_p, formato_json):
        sistema = [m["content"] for m in mensajes if m["role"] == "system"]
        resto = [dict(m) for m in mensajes if m["role"] != "system"]
        if formato_json and resto and resto[-1]["role"] == "user":
            resto[-1]["content"] += "\n\nPlease respond with valid JSON only."
        parametros = {"model": self.modelo, "max_tokens": max_tokens, "messages": resto,
                      "thinking": {"type": "disabled"}}   # sin razonamiento previo
        if sistema:   # la instruccion fija de Mem0 se marca para la cache del proveedor
            parametros["system"] = [{"type": "text", "text": sistema[0], "cache_control": {"type": "ephemeral"}}]
        if temperatura is not None:
            parametros["temperature"] = temperatura
        r = self._enviar(self.cliente.messages.create, parametros)
        texto = next((b.text for b in r.content if getattr(b, "type", "") == "text"), "")
        u = r.usage
        lectura = getattr(u, "cache_read_input_tokens", 0) or 0
        escritura = getattr(u, "cache_creation_input_tokens", 0) or 0
        return texto, u.input_tokens, lectura, escritura, u.output_tokens

    def llamar(self, mensajes, max_tokens, temperatura=None, top_p=None, formato_json=False):
        """Devuelve el texto de la respuesta. Actualiza el uso de la pregunta en curso."""
        funcion = self._openai if self.proveedor == "chatgpt" else self._anthropic
        try:
            texto, sin_cache, lectura, escritura, salida = funcion(mensajes, max_tokens, temperatura, top_p, formato_json)
        except self.sdk.AuthenticationError as error:
            raise CorridaDetenida(f"La clave de la API fue rechazada: {error}")
        except Exception:
            self.errores_seguidos += 1
            if self.uso is not None:
                self.uso["errores_llm"] += 1
            if self.errores_seguidos >= MAX_ERRORES_SEGUIDOS:
                raise CorridaDetenida(f"{MAX_ERRORES_SEGUIDOS} errores seguidos de la API")
            raise
        self.errores_seguidos = 0
        p = self.precio
        costo = (sin_cache * p["entrada"] + lectura * p["cache_lectura"]
                 + escritura * p["cache_escritura"] + salida * p["salida"]) / 1e6
        self.gastado += costo
        if self.uso is not None:
            self.uso["llamadas"] += 1
            self.uso["tokens_entrada"] += sin_cache + lectura + escritura
            self.uso["tokens_cache_lectura"] += lectura
            self.uso["tokens_cache_escritura"] += escritura
            self.uso["tokens_salida"] += salida
            self.uso["costo_usd"] += costo
        if self.tope_usd and self.gastado > self.tope_usd:
            raise CorridaDetenida(f"Se alcanzo el tope de gasto: {self.gastado:.2f} de {self.tope_usd:.2f} dolares")
        return texto


def ejecutar(proveedor):
    datos = PROVEEDORES[proveedor]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opciones = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if len(args) < 2 or args[0] not in DATASETS or args[1] not in ("base", "modulo"):
        raise SystemExit(__doc__)
    variante, use_nli = args[0], args[1] == "modulo"
    es_prueba = len(args) > 2
    max_preguntas = int(args[2]) if es_prueba else 78
    modelo = opciones.get("modelo", datos["modelo"])
    tope_usd = float(opciones.get("tope", 15))
    puerto = int(opciones.get("puerto", datos["puerto"]))
    if modelo not in PRECIOS:
        raise SystemExit(f"El modelo {modelo} no tiene precios en PRECIOS (evaluacion_api.py). Agregalos antes de correr.")
    # Las claves se comprueban antes de crear carpetas o gastar nada
    leer_clave("OPENAI_API_KEY")   # los embeddings son de OpenAI en ambos proveedores
    leer_clave(datos["clave"])

    carpeta = "E:/proyecto/" + datos["carpeta"]
    nombre = f"{variante}_{'con' if use_nli else 'sin'}_modulo_{proveedor}"
    ruta = f"{carpeta}/prueba/{nombre}_{max_preguntas}.json" if es_prueba else f"{carpeta}/{variante}/{nombre}.json"

    # Guarda todo lo que aparece en la terminal en un .txt junto al json
    class _Tee:
        def __init__(self, *flujos):
            self.flujos = flujos
        def write(self, texto):
            for f in self.flujos:
                f.write(texto)
                f.flush()
        def flush(self):
            for f in self.flujos:
                f.flush()
    for flujo in (sys.__stdout__, sys.__stderr__):
        try:
            flujo.reconfigure(errors="replace")
        except Exception:
            pass
    carpeta_txt = os.path.join(os.path.dirname(ruta), "txt")
    os.makedirs(carpeta_txt, exist_ok=True)
    registro = open(os.path.join(carpeta_txt, os.path.basename(ruta).replace(".json", ".txt")), "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, registro)
    sys.stderr = _Tee(sys.__stderr__, registro)

    os.environ.setdefault("MEM0_TELEMETRY", "False")
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("mem0.nli").setLevel(logging.INFO)
    logging.getLogger("mem0.memory.main").setLevel(logging.INFO)
    logging.getLogger("mem0.vector_stores.qdrant").setLevel(logging.ERROR)
    for ruidoso in ("httpx", "openai", "anthropic"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    try:
        from mem0 import Memory
    except ImportError:
        raise SystemExit(f"Este Python no tiene Mem0 instalado: {sys.executable}\nUsa: py -3.12 ...")
    from qdrant_client import QdrantClient
    from juez import construir_prompt, leer_veredicto

    # Errores que Mem0 anota por dentro sin lanzarlos (por ejemplo, un JSON invalido)
    class _ContadorErrores(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.ERROR)
            self.total = 0
        def emit(self, record):
            if record.name.startswith("mem0"):
                self.total += 1
    contador_errores = _ContadorErrores()
    logging.getLogger().addHandler(contador_errores)

    try:
        QdrantClient("localhost", port=puerto, timeout=5).get_collections()
    except Exception:
        raise SystemExit(f"No hay un Qdrant en el puerto {puerto}. Se crea una vez con:\n"
                         f"  docker run -d --name qdrant_{proveedor} -p {puerto}:6333 qdrant/qdrant\n"
                         f"y despues se enciende con: docker start qdrant_{proveedor}")

    config = {
        "history_db_path": ":memory:",   # cada pregunta empieza sin mensajes heredados
        # El proveedor "openai" solo sirve para que Mem0 se construya: la llamada
        # real al modelo la hace ClienteLLM (ver conectar).
        "llm": {"provider": "openai", "config": {"model": "gpt-5.4-nano"}},
        "embedder": {"provider": "openai", "config": {"model": MODELO_EMBEDDINGS, "embedding_dims": DIMENSIONES}},
        "vector_store": {"provider": "qdrant", "config": {"host": "localhost", "port": puerto,
                                                          "embedding_model_dims": DIMENSIONES}},
    }

    def limpiar_memorias():
        c = QdrantClient("localhost", port=puerto)
        for col in c.get_collections().collections:
            c.delete_collection(col.name)

    # Retoma desde donde se quedo si el JSON ya existe
    resultados = []
    if os.path.exists(ruta):
        try:
            guardado = json.load(open(ruta, encoding="utf-8"))
            resultados = guardado["detalle"] if isinstance(guardado, dict) else guardado
        except Exception:
            resultados = []
    ya_procesados = {r["question_id"] for r in resultados}
    cliente = ClienteLLM(proveedor, modelo, tope_usd, sum(r.get("costo_usd", 0) for r in resultados))

    def conectar(m):
        """Hace que Mem0 use ClienteLLM para la extraccion (temperatura por defecto de Mem0)."""
        def generar(messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
            return cliente.llamar(messages, max_tokens=2000, temperatura=0.1,
                                  top_p=0.1 if proveedor == "chatgpt" else None,
                                  formato_json=bool(response_format))
        m.llm.generate_response = generar

    def nueva_memoria():
        m = Memory.from_config(config)
        if not use_nli:
            m._verificador_nli = None
        elif m._verificador_nli is not None:
            m._verificador_nli.usar_filtro_entidades = USE_NER
            m._verificador_nli.usar_marcado = USE_MARKING
            m._verificador_nli.usar_bidireccional = USE_BIDIRECTIONAL
        conectar(m)
        return m

    def cargar_sesiones(m, sesiones, user_id):
        cargados = errores = 0
        for sesion in sesiones:
            mensajes = sesion if isinstance(sesion, list) else sesion.get("messages", []) if isinstance(sesion, dict) else []
            for msg in mensajes:
                if not isinstance(msg, dict) or msg.get("role") != "user":
                    continue
                contenido = msg.get("content", "").strip()
                if not contenido or len(contenido) < 10:
                    continue
                cargados += 1
                try:
                    m.add([{"role": "user", "content": contenido}], user_id=user_id)
                except Exception as e:
                    errores += 1
                    logging.warning(f"Error al agregar mensaje: {e}")
        return cargados, errores

    def responder_pregunta(m, pregunta, user_id):
        t_inicio = time.time()
        memorias = m.search(query=pregunta, filters={"user_id": user_id}, top_k=10)
        contexto = "\n".join(r["memory"] for r in memorias["results"])
        usadas = [{"memoria": r["memory"], "puntaje": round(r.get("score", 0) or 0, 4)} for r in memorias["results"]]
        print("  Memorias recuperadas para responder:")
        for r in memorias["results"]:
            print(f"    - [{r.get('score', 0):.3f}] {r['memory'][:80]}")
        if not contexto.strip():
            return "No relevant memories found.", round(time.time() - t_inicio, 2), usadas
        prompt = (
            "Based on these user memories:\n" + contexto + "\n\nIMPORTANT:\n"
            + "- Memories marked [OUTDATED ...] contain PREVIOUS values that were later "
            + "updated. Use them ONLY if the question asks about a past, previous or "
            + "original value.\n"
            + "- For questions about the CURRENT value, use the memories WITHOUT the "
            + "[OUTDATED] marker. If several current memories conflict, always use the "
            + "MOST RECENT value based on the context and dates.\n\n"
            + "Answer this question concisely:\n" + pregunta + "\n\nAnswer:"
        )
        texto = cliente.llamar([{"role": "user", "content": prompt}], max_tokens=1000, temperatura=0)
        return texto, round(time.time() - t_inicio, 2), usadas

    def juez_propio(question_id, pregunta, correcta, generada):
        prompt, _ = construir_prompt(question_id, pregunta, correcta, generada)
        texto = cliente.llamar([{"role": "user", "content": prompt}], max_tokens=10, temperatura=0).strip()
        veredicto = leer_veredicto(texto)
        if veredicto is None:   # formato inesperado: regla del script oficial
            veredicto = 1 if "yes" in texto.lower() else 0
        return veredicto, texto

    print("Limpiando Qdrant...")
    limpiar_memorias()
    print("Cargando dataset...")
    with open(DATASETS[variante], encoding="utf-8") as f:
        preguntas_ku = [d for d in json.load(f) if d["question_type"] == CATEGORIA][:max_preguntas]
    print(f"Evaluando {len(preguntas_ku)} preguntas de '{CATEGORIA}' ({variante})")
    print(f"Modelo: {modelo} | embeddings: {MODELO_EMBEDDINGS} | Qdrant en el puerto {puerto} | tope: {tope_usd} USD")
    print(f"Modulo NLI: {'ACTIVADO' if use_nli else 'desactivado'}")
    print(f"Resultados en: {ruta}")
    if ya_procesados:
        print(f"Retomando: {len(ya_procesados)} preguntas ya procesadas, {cliente.gastado:.2f} USD gastados.")
    print("=" * 60)

    correctas = sum(r["score"] for r in resultados)
    detenida = ""
    try:
        if len(ya_procesados) < len(preguntas_ku):
            # Calentamiento: carga el modelo NLI y comprueba las claves antes de medir
            print("Comprobando la conexion (calentamiento, no se mide)...")
            m = nueva_memoria()
            for texto in ("I adopted a dog named Max last week.", "Actually, my dog is called Rocky, not Max."):
                m.add([{"role": "user", "content": texto}], user_id="calentamiento")
            m.search(query="What is the name of my dog?", filters={"user_id": "calentamiento"}, top_k=10)
            limpiar_memorias()
            print(f"Conexion correcta. Gastado hasta ahora: {cliente.gastado:.4f} USD.")

        for i, item in enumerate(preguntas_ku):
            if item["question_id"] in ya_procesados:
                continue
            tiempo_inicio = time.time()
            print(f"\nPregunta {i+1}/{len(preguntas_ku)}: {item['question'][:80]}")
            limpiar_memorias()
            m = nueva_memoria()
            uso = {"llamadas": 0, "tokens_entrada": 0, "tokens_cache_lectura": 0, "tokens_cache_escritura": 0,
                   "tokens_salida": 0, "costo_usd": 0.0, "errores_llm": 0}
            cliente.uso = uso
            errores_mem0_inicio = contador_errores.total
            user_id = f"eval_{item['question_id']}"

            t_carga = time.time()
            mensajes_cargados, errores_carga = cargar_sesiones(m, item["haystack_sessions"], user_id)
            tiempo_carga = round(time.time() - t_carga, 2)

            tiempo_consolidacion = 0.0
            if use_nli and m._verificador_nli is not None:
                t_cons = time.time()
                procesadas = m._verificador_nli.consolidar(vector_store=m.vector_store,
                                                           embedding_model=m.embedding_model,
                                                           search_filters={"user_id": user_id})
                tiempo_consolidacion = round(time.time() - t_cons, 2)
                if procesadas:
                    print(f"  [Consolidacion] {len(procesadas)} memorias obsoletas marcadas")

            c_qdrant = QdrantClient("localhost", port=puerto)
            memorias_guardadas = c_qdrant.get_collection(m.collection_name).points_count
            contradicciones = marcadas = descartados = 0
            if use_nli and m._verificador_nli is not None:
                contradicciones = m._verificador_nli.contradicciones
                marcadas = m._verificador_nli.marcadas
                descartados = m._verificador_nli.pares_descartados_filtro

            respuesta, tiempo_respuesta, memorias_usadas = responder_pregunta(m, item["question"], user_id)
            tiempo_pregunta = round(time.time() - tiempo_inicio, 2)   # sin la calificacion
            errores_mem0 = contador_errores.total - errores_mem0_inicio

            t_juez = time.time()
            score, salida_juez = juez_propio(item["question_id"], item["question"], item["answer"], respuesta)
            tiempo_juez = round(time.time() - t_juez, 2)
            correctas += score

            resultados.append({
                "question_id": item["question_id"], "question": item["question"],
                "answer_correcta": item["answer"], "respuesta_generada": respuesta,
                "score": score, "salida_juez_propio": salida_juez,
                "tiempo_segundos": tiempo_pregunta, "tiempo_carga_seg": tiempo_carga,
                "tiempo_consolidacion_seg": tiempo_consolidacion, "tiempo_respuesta_seg": tiempo_respuesta,
                "tiempo_juez_propio_seg": tiempo_juez,
                "mensajes_cargados": mensajes_cargados, "errores_carga": errores_carga,
                "errores_mem0": errores_mem0, "errores_llm": uso["errores_llm"],
                "memorias_guardadas": memorias_guardadas,
                "contradicciones": contradicciones, "memorias_marcadas": marcadas,
                "pares_descartados_filtro": descartados,
                "llamadas_llm": uso["llamadas"], "tokens_entrada": uso["tokens_entrada"],
                "tokens_cache_lectura": uso["tokens_cache_lectura"],
                "tokens_cache_escritura": uso["tokens_cache_escritura"],
                "tokens_salida": uso["tokens_salida"], "costo_usd": round(uso["costo_usd"], 6),
                "memorias_usadas": memorias_usadas,
            })
            cliente.uso = None
            with open(ruta, "w", encoding="utf-8") as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False)

            en_cache = uso["tokens_cache_lectura"] / uso["tokens_entrada"] if uso["tokens_entrada"] else 0
            print(f"  Correcta:   {item['answer']}")
            print(f"  Generada:   {respuesta[:100]}")
            print(f"  Memorias:   {memorias_guardadas} | Contradicciones: {contradicciones} | Mensajes: "
                  f"{mensajes_cargados} (errores: {errores_carga}, de Mem0: {errores_mem0}, de la API: {uso['errores_llm']})")
            print(f"  API:        {uso['llamadas']} llamadas | {uso['tokens_entrada']} tokens entrada "
                  f"({en_cache:.0%} desde cache) | {uso['tokens_salida']} salida | {uso['costo_usd']:.4f} USD "
                  f"| acumulado {cliente.gastado:.2f} USD")
            print(f"  Score (propio): {score} [{salida_juez[:20]}] | Accuracy: {correctas / len(resultados):.2%} | "
                  f"Carga: {tiempo_carga:.1f}s | Respuesta: {tiempo_respuesta:.1f}s | Total: {tiempo_pregunta:.1f}s")
    except CorridaDetenida as motivo:
        detenida = str(motivo)

    if detenida or len(resultados) < len(preguntas_ku):
        print("\n" + "=" * 60)
        print(f"CORRIDA DETENIDA: {detenida or 'incompleta'}")
        print(f"Quedaron guardadas {len(resultados)} de {len(preguntas_ku)} preguntas. Gastado: {cliente.gastado:.2f} USD.")
        print("Para continuar, corrige la causa y vuelve a lanzar el mismo comando"
              " (con un --tope mayor si se detuvo por el tope de gasto).")
        raise SystemExit(1)

    # Resumen final
    n = len(resultados)
    suma = lambda campo: sum(r.get(campo, 0) for r in resultados)
    tiempo_total = suma("tiempo_segundos")
    entrada, lectura = suma("tokens_entrada"), suma("tokens_cache_lectura")
    resumen = {
        "total_preguntas": n, "correctas": correctas,
        "accuracy": round(correctas / n, 4), "accuracy_pct": f"{correctas / n:.2%}",
        "juez_de_score": f"propio ({modelo}, plantillas oficiales)",
        "tiempo_promedio_seg": round(tiempo_total / n, 2),
        "tiempo_carga_promedio_seg": round(suma("tiempo_carga_seg") / n, 2),
        "tiempo_respuesta_promedio_seg": round(suma("tiempo_respuesta_seg") / n, 2),
        "tiempo_por_mensaje_guardado_seg": round(suma("tiempo_carga_seg") / max(1, suma("mensajes_cargados")), 2),
        "tiempo_consolidacion_promedio_seg": round(suma("tiempo_consolidacion_seg") / n, 2),
        "tiempo_total_min": round(tiempo_total / 60, 2),
        "total_contradicciones": suma("contradicciones"), "total_memorias_marcadas": suma("memorias_marcadas"),
        "total_pares_descartados_filtro": suma("pares_descartados_filtro"),
        "promedio_memorias_guardadas": round(suma("memorias_guardadas") / n, 2),
        "maximo_memorias_guardadas": max(r["memorias_guardadas"] for r in resultados),
        "total_mensajes_cargados": suma("mensajes_cargados"), "total_errores_carga": suma("errores_carga"),
        "total_errores_mem0": suma("errores_mem0"), "total_errores_llm": suma("errores_llm"),
        "total_llamadas_llm": suma("llamadas_llm"), "total_tokens_entrada": entrada,
        "total_tokens_cache_lectura": lectura, "fraccion_desde_cache": round(lectura / entrada, 4) if entrada else 0,
        "total_tokens_salida": suma("tokens_salida"), "costo_total_usd": round(suma("costo_usd"), 4),
        "use_nli": use_nli, "use_ner": USE_NER, "use_marking": USE_MARKING, "use_bidirectional": USE_BIDIRECTIONAL,
        "categoria": CATEGORIA, "variante": variante,
        "configuracion": {
            "proveedor": proveedor, "modelo": modelo, "embeddings": MODELO_EMBEDDINGS,
            "temperatura_respuesta": 0, "temperatura_extraccion": 0.1, "razonamiento": "desactivado",
            "parametros_que_el_modelo_rechazo": sorted(cliente.omitidos),
            "top_k_busqueda": 10, "history_db_path": ":memory:", "dataset": DATASETS[variante],
            "precios_usd_por_millon": PRECIOS[modelo], "costo_no_incluye": "embeddings",
            "fecha_fin": datetime.now().isoformat(timespec="seconds"),
        },
        "detalle": resultados,
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"Modelo                 : {modelo} ({variante}, {'con' if use_nli else 'sin'} modulo)")
    print(f"Respuestas correctas   : {correctas} de {n}  (calificacion del propio modelo)")
    print(f"Accuracy               : {correctas / n:.2%}")
    print(f"Tiempo por mensaje     : {resumen['tiempo_por_mensaje_guardado_seg']:.2f} segundos por mensaje guardado")
    print(f"Consolidacion promedio : {resumen['tiempo_consolidacion_promedio_seg']:.1f} segundos por pregunta")
    print(f"Tiempo respuesta prom. : {resumen['tiempo_respuesta_promedio_seg']:.1f} segundos")
    print(f"Tiempo total promedio  : {resumen['tiempo_promedio_seg']:.1f} segundos")
    print(f"Total contradicciones  : {resumen['total_contradicciones']}")
    print(f"Memorias prom/pregunta : {resumen['promedio_memorias_guardadas']:.1f} (maximo {resumen['maximo_memorias_guardadas']})")
    print(f"Mensajes cargados      : {resumen['total_mensajes_cargados']} (errores: {resumen['total_errores_carga']}, "
          f"de Mem0: {resumen['total_errores_mem0']}, de la API: {resumen['total_errores_llm']})")
    print(f"Tokens de entrada      : {entrada} ({resumen['fraccion_desde_cache']:.0%} desde cache)")
    print(f"Parametros rechazados  : {sorted(cliente.omitidos) or 'ninguno'}")
    print(f"Costo estimado         : {resumen['costo_total_usd']:.2f} USD (sin embeddings)")
    print(f"Tiempo total           : {tiempo_total / 60:.1f} minutos")
    print(f"Resultados guardados   : {ruta}")
    print("=" * 60)
