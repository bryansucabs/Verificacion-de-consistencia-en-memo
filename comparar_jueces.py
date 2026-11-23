"""Compara el juez independiente con la calificacion del propio modelo en cada corrida.

Para cada corrida muestra la exactitud segun cada juez, cuanto coinciden
(acuerdo y kappa de Cohen) y las preguntas donde difieren. Para cada par
base/modulo aplica la prueba exacta de McNemar. Las corridas que todavia no
existen o no han sido calificadas por el juez independiente se omiten.

Lee los veredictos que recalificar.py dejo en la subcarpeta juez/ de cada
corrida. No llama al juez ni tiene costo: se puede ejecutar las veces que haga falta.

Uso:
    python comparar_jueces.py [repeticion]

Escribe resumen.txt, resumen.json y desacuerdos.md en la carpeta comparacion_jueces/.
"""
import json
import os
import sys
from math import comb

from juez import NOMBRE_JUEZ
from recalificar import RAIZ, cargar_resultados, ruta_veredictos

# La tabla que junta todas las corridas se guarda aqui
CARPETA_COMPARACION = os.path.join(RAIZ, "comparacion_jueces")

# (nombre, corrida base, corrida con modulo). Rutas relativas a E:/proyecto.
PARES = [
    ("GPT-5.4-nano, modulo v2, oracle",
     "resultados_openai/oracle_gpt_baseline_78.json",
     "resultados_openai/oracle_gpt_consolidacion_78.json"),
    ("GPT-5.4-nano, modulo v2, small",
     "resultados_openai/longmemeval_s_memo_78.json",
     "resultados_openai/longmemeval_s_nli_78.json"),
    ("Qwen 2.5, modulo v3, oracle",
     "resultados_local_qwen_pfc3/oracle/oracle_sin_modulo.json",
     "resultados_local_qwen_pfc3/oracle/oracle_con_modulo.json"),
    ("Qwen 2.5, modulo v3, small",
     "resultados_local_qwen_pfc3/small/small_sin_modulo.json",
     "resultados_local_qwen_pfc3/small/small_con_modulo.json"),
    ("Qwen 2.5 corregido, modulo v3, oracle",
     "resultados_local_qwen25_corregido/oracle/oracle_sin_modulo_qwen25.json",
     "resultados_local_qwen25_corregido/oracle/oracle_con_modulo_qwen25.json"),
    ("Qwen 2.5 corregido, modulo v3, small",
     "resultados_local_qwen25_corregido/small/small_sin_modulo_qwen25.json",
     "resultados_local_qwen25_corregido/small/small_con_modulo_qwen25.json"),
    ("ChatGPT, modulo v3, oracle",
     "resultados_chatgpt/oracle/oracle_sin_modulo_chatgpt.json",
     "resultados_chatgpt/oracle/oracle_con_modulo_chatgpt.json"),
    ("ChatGPT, modulo v3, small",
     "resultados_chatgpt/small/small_sin_modulo_chatgpt.json",
     "resultados_chatgpt/small/small_con_modulo_chatgpt.json"),
    ("Claude, modulo v3, oracle",
     "resultados_claude/oracle/oracle_sin_modulo_claude.json",
     "resultados_claude/oracle/oracle_con_modulo_claude.json"),
    ("Claude, modulo v3, small",
     "resultados_claude/small/small_sin_modulo_claude.json",
     "resultados_claude/small/small_con_modulo_claude.json"),
    ("Qwen 3.5, modulo v3, oracle",
     "resultados_local_qwen35/oracle/oracle_sin_modulo_qwen35.json",
     "resultados_local_qwen35/oracle/oracle_con_modulo_qwen35.json"),
    ("Qwen 3.5, modulo v3, small",
     "resultados_local_qwen35/small/small_sin_modulo_qwen35.json",
     "resultados_local_qwen35/small/small_con_modulo_qwen35.json"),
]


def kappa(a, b):
    """Kappa de Cohen entre dos listas de veredictos 0/1."""
    n = len(a)
    acuerdo = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    azar = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if azar == 1 else (acuerdo - azar) / (1 - azar)


def mcnemar(base, modulo):
    """Prueba exacta de McNemar. Devuelve (gana el modulo, pierde el modulo, valor p)."""
    gana = sum(x == 0 and y == 1 for x, y in zip(base, modulo))
    pierde = sum(x == 1 and y == 0 for x, y in zip(base, modulo))
    n, k = gana + pierde, min(gana, pierde)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return gana, pierde, p


def leer_corrida(ruta, repeticion):
    """Une cada respuesta con la calificacion propia y la del juez independiente.

    Devuelve None si la corrida no existe o el juez independiente aun no la califico.
    """
    ruta_resultados = os.path.join(RAIZ, ruta)
    ruta_juez = ruta_veredictos(ruta_resultados, repeticion)
    if not os.path.exists(ruta_resultados) or not os.path.exists(ruta_juez):
        return None
    datos, _ = cargar_resultados(ruta_resultados)
    detalle = datos["detalle"] if isinstance(datos, dict) else datos
    with open(ruta_juez, encoding="utf-8") as archivo:
        veredictos = {v["question_id"]: v for v in json.load(archivo)["veredictos"]}
    if any(item["question_id"] not in veredictos for item in detalle):
        return None
    filas = []
    for item in detalle:
        v = veredictos[item["question_id"]]
        filas.append({
            "question_id": item["question_id"],
            "pregunta": item["question"],
            "correcta": item["answer_correcta"],
            "generada": item["respuesta_generada"],
            "propio": int(item["score"]),
            "independiente": v["veredicto"] if v["veredicto"] is not None else 0,
            "invalido": v["veredicto"] is None,
            "difiere_regla_oficial": v["veredicto"] != v["veredicto_regla_oficial"],
            "salida_cruda": v["salida_cruda"],
        })
    return filas


def resumir(nombre, filas):
    n = len(filas)
    propio = [f["propio"] for f in filas]
    independiente = [f["independiente"] for f in filas]
    abstencion = [f for f in filas if "_abs" in f["question_id"]]
    return {
        "corrida": nombre,
        "preguntas": n,
        "correctas_propio": sum(propio),
        "correctas_independiente": sum(independiente),
        "exactitud_propio": round(100 * sum(propio) / n, 2),
        "exactitud_independiente": round(100 * sum(independiente) / n, 2),
        "coinciden": sum(x == y for x, y in zip(propio, independiente)),
        "kappa": round(kappa(propio, independiente), 3),
        "propio_si_independiente_no": sum(x == 1 and y == 0 for x, y in zip(propio, independiente)),
        "propio_no_independiente_si": sum(x == 0 and y == 1 for x, y in zip(propio, independiente)),
        "abstencion_propio": sum(f["propio"] for f in abstencion),
        "abstencion_independiente": sum(f["independiente"] for f in abstencion),
        "abstencion_total": len(abstencion),
        "invalidos": sum(f["invalido"] for f in filas),
        "difieren_de_regla_oficial": sum(f["difiere_regla_oficial"] for f in filas),
    }


def main(repeticion="r1"):
    carpeta = CARPETA_COMPARACION if repeticion == "r1" else os.path.join(CARPETA_COMPARACION, repeticion)
    os.makedirs(carpeta, exist_ok=True)
    resumen = {"juez_independiente": NOMBRE_JUEZ, "repeticion": repeticion, "corridas": [], "mcnemar": []}
    desacuerdos = [f"# Preguntas donde el propio modelo y el juez independiente ({NOMBRE_JUEZ}) difieren\n"]
    lineas = []

    def mostrar(texto=""):
        print(texto)
        lineas.append(texto)

    mostrar("Correctas de 78 y exactitud (%) segun cada juez")
    mostrar(f"{'corrida':<46}{'juez propio':>13}{'juez indep.':>13}{'coinciden':>11}{'kappa':>7}{'abstencion':>12}")
    pendientes = []
    for nombre, ruta_base, ruta_modulo in PARES:
        corridas = {}
        for etiqueta, ruta in (("base", ruta_base), ("con modulo", ruta_modulo)):
            filas = leer_corrida(ruta, repeticion)
            if filas is None:
                pendientes.append(f"{nombre}, {etiqueta}")
                continue
            corridas[etiqueta] = filas
            r = resumir(f"{nombre}, {etiqueta}", filas)
            r["archivo"] = ruta
            resumen["corridas"].append(r)
            mostrar(f"{r['corrida']:<46}"
                    f"{r['correctas_propio']:>6} {r['exactitud_propio']:>6.2f}"
                    f"{r['correctas_independiente']:>6} {r['exactitud_independiente']:>6.2f}"
                    f"{r['coinciden']:>8}/{r['preguntas']}"
                    f"{r['kappa']:>7.3f}"
                    f"{r['abstencion_propio']:>6}/{r['abstencion_independiente']} de {r['abstencion_total']}")

            distintas = [f for f in filas if f["propio"] != f["independiente"]]
            desacuerdos.append(f"\n## {r['corrida']} ({len(distintas)} preguntas)\n")
            for f in distintas:
                desacuerdos.append(
                    f"### {f['question_id']}\n\n"
                    f"- Pregunta: {f['pregunta']}\n"
                    f"- Respuesta correcta: {f['correcta']}\n"
                    f"- Respuesta del sistema: {f['generada']}\n"
                    f"- Propio modelo: {'correcta' if f['propio'] else 'incorrecta'}\n"
                    f"- Juez independiente: {'correcta' if f['independiente'] else 'incorrecta'} "
                    f"(salida: {f['salida_cruda']!r})\n"
                )

        if len(corridas) == 2:
            for juez in ("propio", "independiente"):
                gana, pierde, p = mcnemar([f[juez] for f in corridas["base"]],
                                          [f[juez] for f in corridas["con modulo"]])
                resumen["mcnemar"].append({"par": nombre, "juez": juez, "gana_modulo": gana,
                                           "pierde_modulo": pierde, "p": round(p, 4)})

    mostrar()
    mostrar("Base contra modulo: preguntas que el modulo gana y pierde (McNemar exacto)")
    mostrar(f"{'par':<46}{'juez':>15}{'gana':>7}{'pierde':>8}{'p':>8}")
    for m in resumen["mcnemar"]:
        mostrar(f"{m['par']:<46}{m['juez']:>15}{m['gana_modulo']:>7}{m['pierde_modulo']:>8}{m['p']:>8.4f}")

    invalidos = sum(r["invalidos"] for r in resumen["corridas"])
    difieren = sum(r["difieren_de_regla_oficial"] for r in resumen["corridas"])
    mostrar()
    mostrar(f"Salidas del juez independiente con formato invalido: {invalidos}. "
            f"Veredictos que cambiarian con la regla del script oficial: {difieren}.")
    if pendientes:
        mostrar("Corridas sin calificar todavia: " + "; ".join(pendientes))
    resumen["pendientes"] = pendientes

    with open(os.path.join(carpeta, "resumen.json"), "w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, indent=2, ensure_ascii=False)
    with open(os.path.join(carpeta, "resumen.txt"), "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")
    with open(os.path.join(carpeta, "desacuerdos.md"), "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(desacuerdos))
    print(f"Guardado en {carpeta}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "r1")
