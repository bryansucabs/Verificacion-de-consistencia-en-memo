"""Califica con el juez independiente las respuestas ya guardadas de una corrida.

No modifica el archivo de resultados. Los veredictos se guardan junto a la
corrida, en una subcarpeta juez/, con el mismo nombre del archivo:

    resultados_local_qwen35/oracle/oracle_sin_modulo_qwen35.json        <- la corrida
    resultados_local_qwen35/oracle/juez/oracle_sin_modulo_qwen35.json   <- veredictos del juez
    resultados_local_qwen35/oracle/juez/oracle_sin_modulo_qwen35.txt    <- resumen legible

Si se interrumpe, al volver a ejecutarlo continua desde donde se quedo.

Uso:
    python recalificar.py <archivo_de_resultados.json> [repeticion]

La repeticion es opcional (r2, r3...). Sirve para calificar lo mismo otra vez
y comprobar que el juez es estable; se guarda con ese sufijo en el nombre.
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from juez import BASE_URL, MAX_TOKENS, MODELO, NOMBRE_JUEZ, PROVEEDOR, TEMPERATURA, juzgar

RAIZ = os.path.dirname(os.path.abspath(__file__))
EN_PARALELO = 20


def cargar_resultados(ruta):
    """Lee un archivo de resultados. Algunos de mayo estan en cp1252 y no en utf-8."""
    with open(ruta, "rb") as archivo:
        crudo = archivo.read()
    huella = hashlib.sha256(crudo).hexdigest()
    for codificacion in ("utf-8", "cp1252"):
        try:
            return json.loads(crudo.decode(codificacion)), huella
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer {ruta}")


def ruta_veredictos(ruta, repeticion="r1"):
    """Archivo de veredictos de una corrida: subcarpeta juez/ junto al archivo de resultados."""
    carpeta = os.path.join(os.path.dirname(os.path.abspath(ruta)), "juez")
    nombre = os.path.basename(ruta)
    if repeticion != "r1":
        nombre = nombre.replace(".json", f"_{repeticion}.json")
    return os.path.join(carpeta, nombre)


def ruta_relativa(ruta):
    """Ruta relativa a la carpeta del proyecto, para que el registro no dependa del equipo."""
    try:
        return os.path.relpath(os.path.abspath(ruta), RAIZ).replace("\\", "/")
    except ValueError:
        return os.path.abspath(ruta).replace("\\", "/")


def guardar(destino, contenido):
    """Escribe primero en un archivo temporal para no dejar un JSON a medias."""
    temporal = destino + ".tmp"
    with open(temporal, "w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, indent=2, ensure_ascii=False)
    os.replace(temporal, destino)


def resumen_legible(ruta, detalle, hechos):
    """Texto con el resultado del juez independiente y su comparacion con el propio modelo."""
    n = len(detalle)
    filas = []
    for item in detalle:
        v = hechos[item["question_id"]]
        filas.append({
            "id": item["question_id"],
            "pregunta": item["question"],
            "correcta": item["answer_correcta"],
            "generada": item["respuesta_generada"],
            "propio": int(item["score"]) if item.get("score") is not None else None,
            "independiente": v["veredicto"],
            "salida": v["salida_cruda"],
        })
    indep = sum(1 for f in filas if f["independiente"] == 1)
    invalidos = sum(1 for f in filas if f["independiente"] is None)
    con_propio = [f for f in filas if f["propio"] is not None]
    abstencion = [f for f in filas if "_abs" in f["id"]]

    lineas = [
        f"Juez independiente : {NOMBRE_JUEZ} ({MODELO}, {PROVEEDOR})",
        f"Corrida            : {ruta_relativa(ruta)}",
        f"Fecha              : {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Correctas segun el juez independiente : {indep} de {n}  ({100 * indep / n:.2f} %)",
    ]
    if con_propio:
        propio = sum(f["propio"] for f in con_propio)
        ambos_si = sum(1 for f in con_propio if f["propio"] == 1 and f["independiente"] == 1)
        ambos_no = sum(1 for f in con_propio if f["propio"] == 0 and f["independiente"] != 1)
        solo_propio = sum(1 for f in con_propio if f["propio"] == 1 and f["independiente"] != 1)
        solo_indep = sum(1 for f in con_propio if f["propio"] == 0 and f["independiente"] == 1)
        lineas += [
            f"Correctas segun el propio modelo      : {propio} de {n}  ({100 * propio / n:.2f} %)",
            "",
            f"Los dos jueces coinciden en           : {ambos_si + ambos_no} de {n}",
            f"  correcta para los dos               : {ambos_si}",
            f"  incorrecta para los dos             : {ambos_no}",
            f"  correcta solo para el propio modelo : {solo_propio}",
            f"  correcta solo para el independiente : {solo_indep}",
        ]
    if abstencion:
        linea = (f"Preguntas de abstencion ({len(abstencion)})           : "
                 f"independiente {sum(1 for f in abstencion if f['independiente'] == 1)}")
        if con_propio:
            linea += f", propio {sum(f['propio'] for f in abstencion if f['propio'] is not None)}"
        lineas.append(linea)
    lineas.append(f"Salidas del juez con formato invalido : {invalidos}")

    distintas = [f for f in con_propio if f["propio"] != (1 if f["independiente"] == 1 else 0)]
    lineas += ["", f"Preguntas donde los dos jueces difieren: {len(distintas)}"]
    for f in distintas:
        lineas += [
            "",
            f"  [{f['id']}] {f['pregunta']}",
            f"    Respuesta correcta    : {f['correcta']}",
            f"    Respuesta del sistema : {str(f['generada']).strip()}",
            f"    Propio modelo         : {'correcta' if f['propio'] else 'incorrecta'}",
            f"    Juez independiente    : {'correcta' if f['independiente'] == 1 else 'incorrecta'} (salida: {f['salida']!r})",
        ]

    incorrectas = [f for f in filas if f["independiente"] != 1]
    lineas += ["", "", f"Respuestas incorrectas segun el juez independiente: {len(incorrectas)}"]
    for f in incorrectas:
        lineas += [
            "",
            f"  [{f['id']}] {f['pregunta']}",
            f"    Respuesta correcta    : {f['correcta']}",
            f"    Respuesta del sistema : {str(f['generada']).strip()}",
        ]
    return "\n".join(lineas) + "\n"


def recalificar(ruta, repeticion="r1"):
    datos, huella = cargar_resultados(ruta)
    if not isinstance(datos, dict) or "detalle" not in datos:
        raise SystemExit(f"La corrida todavia no ha terminado (falta el resumen final): {ruta}")
    detalle = datos["detalle"]

    destino = ruta_veredictos(ruta, repeticion)
    os.makedirs(os.path.dirname(destino), exist_ok=True)

    hechos = {}
    if os.path.exists(destino):
        with open(destino, encoding="utf-8") as archivo:
            previo = json.load(archivo)
        if previo["sha256_origen"] != huella:
            raise SystemExit(f"El archivo de resultados cambio desde la calificacion anterior: {ruta}")
        hechos = {v["question_id"]: v for v in previo["veredictos"]}

    # Las llamadas se envian en paralelo porque el juez puede tardar mas de 30 s por respuesta.
    # Cada veredicto es independiente de los demas, asi que el orden de llegada no cambia nada.
    pendientes = [item for item in detalle if item["question_id"] not in hechos]
    with ThreadPoolExecutor(EN_PARALELO) as grupo:
        futuros = {
            grupo.submit(juzgar, item["question_id"], item["question"],
                         item["answer_correcta"], item["respuesta_generada"]): item
            for item in pendientes
        }
        for futuro in as_completed(futuros):
            item = futuros[futuro]
            veredicto = futuro.result()
            veredicto["score_propio"] = item.get("score")
            hechos[item["question_id"]] = veredicto
            guardar(destino, {
                "archivo_origen": ruta_relativa(ruta),
                "sha256_origen": huella,
                "juez": {"nombre": NOMBRE_JUEZ, "modelo": MODELO, "proveedor": PROVEEDOR,
                         "base_url": BASE_URL, "temperatura": TEMPERATURA, "max_tokens": MAX_TOKENS},
                "repeticion": repeticion,
                "veredictos": [hechos[d["question_id"]] for d in detalle if d["question_id"] in hechos],
            })
            print(f"  [{len(hechos)}/{len(detalle)}] {item['question_id']}: "
                  f"juez={veredicto['salida_cruda']!r} propio={item.get('score')}", flush=True)

    texto = resumen_legible(ruta, detalle, hechos)
    destino_txt = destino.replace(".json", ".txt")
    with open(destino_txt, "w", encoding="utf-8") as archivo:
        archivo.write(texto)

    print("\n" + "=" * 60)
    print("RESULTADO DEL JUEZ INDEPENDIENTE")
    print("=" * 60)
    print(texto.split("\nPreguntas donde los dos jueces difieren")[0].rstrip())
    print("=" * 60)
    print(f"Veredictos : {destino}")
    print(f"Resumen    : {destino_txt}")
    print("El resumen incluye la lista de respuestas incorrectas y las preguntas donde los jueces difieren.")
    return destino


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    recalificar(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "r1")
