"""
Modulo de Verificacion de Consistencia mediante Inferencia de Lenguaje Natural.

Este modulo es el componente principal de la propuesta de tesis. Su funcion
es detectar cuando el usuario actualiza un hecho previo y eliminar la memoria
obsoleta antes de que Qwen decida que guardar.

"""


import logging
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

UMBRAL_CONTRADICCION = 0.85
MODELO_NLI = "cross-encoder/nli-deberta-v3-base"
IDX_CONTRADICCION = 0
IDX_IMPLICACION = 1
IDX_NEUTRALIDAD = 2


class VerificadorConsistencia:
    """
    Verifica si el mensaje nuevo del usuario contradice alguna memoria
    ya guardada en Qdrant. Si detecta una contradiccion, elimina la
    memoria obsoleta antes de que Qwen decida que guardar.

    Tiene tres pasos:
    1. Formar pares: filtra las memorias mas similares al mensaje nuevo
    2. Inferir relaciones: DeBERTa compara cada par y calcula probabilidades
    3. Determinar accion: si P(contradiccion) > 0.85, elimina la memoria vieja
    """


    def __init__(self, umbral: float = UMBRAL_CONTRADICCION):
        self.umbral = umbral
        self._tokenizer = None
        self._modelo = None
        self.contradicciones = 0
        logger.info(f"VerificadorConsistencia inicializado con umbral={umbral}")

    def _cargar_modelo(self):
        if self._modelo is None:
            logger.info(f"Cargando modelo NLI: {MODELO_NLI}")
            self._tokenizer = AutoTokenizer.from_pretrained(MODELO_NLI)
            self._modelo = AutoModelForSequenceClassification.from_pretrained(MODELO_NLI)
            self._modelo.eval()
            logger.info("Modelo NLI cargado correctamente en CPU")
    #premisa = hecho nuevo, hipotesis = memoria candidata
    def _inferir_par(self, premisa: str, hipotesis: str) -> Dict:
        self._cargar_modelo()
        inputs = self._tokenizer(
            premisa,
            hipotesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = self._modelo(**inputs).logits
        probabilidades = torch.softmax(logits, dim=-1)[0]
        return {
            "p_contradiccion": float(probabilidades[IDX_CONTRADICCION]),
            "p_implicacion":   float(probabilidades[IDX_IMPLICACION]),
            "p_neutralidad":   float(probabilidades[IDX_NEUTRALIDAD]),
        }

    def formar_pares(self,hecho_nuevo: str,memorias_candidatas: List[Dict]) -> List[Tuple[str, str, str]]:
        """
        Componente 1: Formacion de pares de evaluacion.

        Toma el mensaje nuevo y las memorias encontradas por Qdrant, y forma
        pares (mensaje_nuevo, memoria) solo con las memorias que tienen
        score > 0.70. Las memorias con score menor se ignoran porque son de
        temas muy distintos y podrian causar falsas contradicciones.
        """

        pares = []
        for memoria in memorias_candidatas:
            id_memoria = memoria.get("id", "")
            texto = memoria.get("text", "").strip()
            score = memoria.get("score", 0.0)
            if texto and score > 0.70:
                pares.append((id_memoria, hecho_nuevo, texto))

        logger.debug(f"Formados {len(pares)} pares de evaluación relevantes")
        return pares

    def inferir_relaciones(self,pares: List[Tuple[str, str, str]]) -> List[Dict]:
        """
        Componente 2: Inferencia de relaciones entre pares.

        Para cada par formado en el Componente 1, llama a DeBERTa y obtiene
        tres probabilidades: P(contradiccion), P(implicacion), P(neutralidad).
        Solo P(contradiccion) se usa para decidir si hay un conflicto.
        """

        if not pares:
            return []

        resultados = []
        for id_memoria, fi, mj in pares:
            try:
                probs = self._inferir_par(premisa=fi, hipotesis=mj)
                resultado = {
                    "id_memoria": id_memoria,
                    "memoria_texto": mj,
                    "hecho_nuevo": fi,
                    **probs,
                }
                resultados.append(resultado)
                logger.debug(
                    f"Par evaluado — "
                    f"P(contradicción)={probs['p_contradiccion']:.3f} | "
                    f"P(implicación)={probs['p_implicacion']:.3f} | "
                    f"P(neutralidad)={probs['p_neutralidad']:.3f}"
                )
            except Exception as e:
                logger.warning(f"Error al evaluar par id={id_memoria}: {e}")

        return resultados

    def determinar_accion(self, resultados: List[Dict]) -> Tuple[bool, List[Dict]]:
        """
        Componente 3: Determinacion de la accion.

        Devuelve TODAS las memorias con P(contradiccion) > umbral, no solo la maxima.
        Esto evita que memorias duplicadas sobre el mismo hecho sobrevivan cuando
        Qwen las almaceno en multiples entradas durante sesiones anteriores.

        Caso 1 : Si alguna P(contradiccion) > 0.85:
            Hay contradiccion : devuelve (True, [lista de conflictos])
            El modulo borrara todas las memorias en conflicto de Qdrant.

        Caso 2 : Si ninguna P(contradiccion) > 0.85:
            No hay contradiccion : devuelve (False, [])
            Mem0 continua su flujo normal sin cambios.
        """
        if not resultados:
            return False, []

        en_conflicto = [r for r in resultados if r["p_contradiccion"] > self.umbral]

        if en_conflicto:
            self.contradicciones += len(en_conflicto)
            for r in en_conflicto:
                logger.info(
                    f"CONTRADICCIÓN DETECTADA — "
                    f"P(contradicción)={r['p_contradiccion']:.3f} > umbral={self.umbral}\n"
                    f"  Memoria en conflicto: '{r['memoria_texto'][:80]}'\n"
                    f"  Hecho nuevo:          '{r['hecho_nuevo'][:80]}'"
                )
            return True, en_conflicto

        p_max = max(r["p_contradiccion"] for r in resultados)
        logger.debug(
            f"Sin contradicción — "
            f"P(contradicción) máx={p_max:.3f} ≤ umbral={self.umbral}"
        )
        return False, []

    def verificar(self, hecho_nuevo: str, memorias_candidatas: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Metodo principal. Usa las memorias que Mem0 ya recupero (top_k=10)
        y compara con DeBERTa. Retorna (True, [lista de ids]) si hay contradicciones.
        """
        pares = self.formar_pares(hecho_nuevo, memorias_candidatas)
        if not pares:
            return False, []

        resultados = self.inferir_relaciones(pares)
        if not resultados:
            return False, []

        hay_contradiccion, conflictos = self.determinar_accion(resultados)

        if hay_contradiccion:
            ids = [r["id_memoria"] for r in conflictos]
            return True, ids

        return False, []

    def consolidar(self, vector_store, embedding_model, search_filters: dict) -> List[str]:
        """
        Fase de consolidacion post-carga.

        Despues de cargar TODAS las memorias, recorre cada una, busca sus
        memorias similares (score > 0.70 para evitar comparar temas distintos)
        y cuando detecta una contradiccion (P > umbral) elimina la MAS ANTIGUA
        segun created_at, conservando la actualizacion mas reciente.

        A diferencia de verificar() que corre durante la carga y depende del
        orden de llegada, esta pasada ve el estado completo de la memoria, por
        lo que detecta contradicciones que el flujo incremental podria perder.

        Retorna lista de UUIDs eliminados.
        """
        try:
            raw = vector_store.list(filters=search_filters, top_k=1000)
        except Exception as e:
            logger.warning(f"[NLI-Consolidacion] Error al listar memorias: {e}")
            return []

        # vector_store.list puede devolver lista de puntos o tupla envuelta
        if isinstance(raw, tuple) and len(raw) > 0 and isinstance(raw[0], (list, tuple)):
            puntos = raw[0]
        else:
            puntos = raw

        memorias = {}
        for p in puntos:
            texto = p.payload.get("data", "")
            if texto:
                memorias[p.id] = {
                    "text": texto,
                    "created_at": p.payload.get("created_at", ""),
                }

        if len(memorias) < 2:
            return []

        eliminados = set()
        for mid, info in list(memorias.items()):
            if mid in eliminados:
                continue
            try:
                emb = embedding_model.embed(info["text"], "search")
                similares = vector_store.search(
                    query=info["text"],
                    vectors=emb,
                    top_k=6,
                    filters=search_filters,
                )
            except Exception as e:
                logger.warning(f"[NLI-Consolidacion] Error buscando similares: {e}")
                continue

            for s in similares:
                if s.id == mid or s.id in eliminados or s.id not in memorias:
                    continue
                if s.score < 0.70:
                    continue
                probs = self._inferir_par(premisa=info["text"], hipotesis=memorias[s.id]["text"])
                if probs["p_contradiccion"] > self.umbral:
                    # eliminar la mas antigua por created_at
                    if info["created_at"] <= memorias[s.id]["created_at"]:
                        viejo_id, nuevo_id = mid, s.id
                    else:
                        viejo_id, nuevo_id = s.id, mid
                    eliminados.add(viejo_id)
                    self.contradicciones += 1
                    logger.info(
                        f"[NLI-Consolidacion] CONTRADICCIÓN P={probs['p_contradiccion']:.3f}\n"
                        f"  Eliminada (vieja): '{memorias[viejo_id]['text'][:70]}'\n"
                        f"  Conservada (nueva): '{memorias[nuevo_id]['text'][:70]}'"
                    )
                    if viejo_id == mid:
                        break

        for vid in eliminados:
            try:
                vector_store.delete(vector_id=vid)
            except Exception as e:
                logger.warning(f"[NLI-Consolidacion] Error al borrar {vid}: {e}")

        return list(eliminados)