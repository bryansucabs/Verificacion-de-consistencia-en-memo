"""
Modulo de Verificacion de Consistencia mediante Inferencia de Lenguaje Natural.

Este modulo es el componente principal de la propuesta de tesis. Su funcion
es detectar cuando el usuario actualiza un hecho previo y resolver la
memoria obsoleta antes de que el modelo de lenguaje decida que guardar.

Version PFC3: incorpora tres mejoras sobre la version evaluada en PFC2,
cada una activable de forma independiente para experimentos de ablacion:

  A. Filtro de entidades previo al NLI (usar_filtro_entidades):
     descarta pares de temas distintos sin invocar a DeBERTa, reduciendo
     las falsas contradicciones observadas en almacenamientos extensos.

  B. Marcado de memorias obsoletas (usar_marcado):
     en lugar de eliminar la memoria obsoleta, la conserva marcada como
     historica ([OUTDATED ...]), lo que permite responder tanto preguntas
     de valor vigente como de valor previo. Inspirado en la invalidacion
     bi-temporal de hechos de Zep/Graphiti.

  C. Verificacion bidireccional (usar_bidireccional):
     una contradiccion solo se confirma si P(contradiccion) supera el
     umbral en AMBAS direcciones del par, reduciendo falsos positivos
     derivados de la asimetria del modelo NLI.
"""


import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from mem0.nli.filtro_entidades import FiltroEntidades

logger = logging.getLogger(__name__)

UMBRAL_CONTRADICCION = 0.85
MODELO_NLI = "cross-encoder/nli-deberta-v3-base"
IDX_CONTRADICCION = 0
IDX_IMPLICACION = 1
IDX_NEUTRALIDAD = 2
MARCA_OBSOLETA = "[OUTDATED"


class VerificadorConsistencia:
    """
    Verifica si el mensaje nuevo del usuario contradice alguna memoria
    ya guardada en Qdrant. Si detecta una contradiccion, marca (o elimina)
    la memoria obsoleta antes de que el LLM decida que guardar.

    Tiene tres pasos:
    1. Formar pares: filtra las memorias mas similares al mensaje nuevo
       (umbral de similitud 0.70 + filtro de entidades si esta activo)
    2. Inferir relaciones: DeBERTa compara cada par y calcula probabilidades
       (con confirmacion bidireccional si esta activa)
    3. Determinar accion: si P(contradiccion) > 0.85, marca o elimina
       la memoria vieja
    """

    def __init__(
        self,
        umbral: float = UMBRAL_CONTRADICCION,
        usar_filtro_entidades: bool = True,
        usar_bidireccional: bool = True,
        usar_marcado: bool = True,
    ):
        self.umbral = umbral
        self.usar_filtro_entidades = usar_filtro_entidades
        self.usar_bidireccional = usar_bidireccional
        self.usar_marcado = usar_marcado
        self._tokenizer = None
        self._modelo = None
        self._filtro = FiltroEntidades()
        self.contradicciones = 0
        self.marcadas = 0
        logger.info(
            f"VerificadorConsistencia inicializado con umbral={umbral} | "
            f"filtro_entidades={usar_filtro_entidades} | "
            f"bidireccional={usar_bidireccional} | "
            f"marcado={usar_marcado}"
        )

    @property
    def pares_descartados_filtro(self) -> int:
        return self._filtro.pares_descartados

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

    def _p_contradiccion_confirmada(self, texto_a: str, texto_b: str) -> Tuple[float, Dict]:
        """
        Calcula P(contradiccion) del par. Con la Mejora C activa, si la
        direccion (a,b) supera el umbral se evalua tambien la direccion
        inversa (b,a) y se usa el minimo de ambas: una contradiccion
        genuina debe sostenerse en las dos direcciones.
        """
        probs = self._inferir_par(premisa=texto_a, hipotesis=texto_b)
        p_contra = probs["p_contradiccion"]
        if self.usar_bidireccional and p_contra > self.umbral:
            probs_inv = self._inferir_par(premisa=texto_b, hipotesis=texto_a)
            p_contra = min(p_contra, probs_inv["p_contradiccion"])
            logger.debug(
                f"Bidireccional — directa={probs['p_contradiccion']:.3f} | "
                f"inversa={probs_inv['p_contradiccion']:.3f} | min={p_contra:.3f}"
            )
        return p_contra, probs

    def formar_pares(self, hecho_nuevo: str, memorias_candidatas: List[Dict]) -> List[Tuple[str, str, str]]:
        """
        Componente 1: Formacion de pares de evaluacion.

        Toma el mensaje nuevo y las memorias encontradas por Qdrant, y forma
        pares (mensaje_nuevo, memoria) solo con las memorias que tienen
        score > 0.70. Las memorias con score menor se ignoran porque son de
        temas muy distintos y podrian causar falsas contradicciones.

        Version PFC3:
        - Las memorias ya marcadas como historicas ([OUTDATED ...]) no
          participan: ya fueron invalidadas.
        - Mejora A: si el filtro de entidades esta activo, los pares cuyos
          textos no comparten ningun termino tematico se descartan sin
          invocar al NLI.
        """

        pares = []
        for memoria in memorias_candidatas:
            id_memoria = memoria.get("id", "")
            texto = memoria.get("text", "").strip()
            score = memoria.get("score", 0.0)
            if not texto or score <= 0.70:
                continue
            if texto.startswith(MARCA_OBSOLETA):
                continue
            if self.usar_filtro_entidades and not self._filtro.par_es_comparable(hecho_nuevo, texto):
                continue
            pares.append((id_memoria, hecho_nuevo, texto))

        logger.debug(f"Formados {len(pares)} pares de evaluación relevantes")
        return pares

    def inferir_relaciones(self, pares: List[Tuple[str, str, str]]) -> List[Dict]:
        """
        Componente 2: Inferencia de relaciones entre pares.

        Para cada par formado en el Componente 1, llama a DeBERTa y obtiene
        tres probabilidades: P(contradiccion), P(implicacion), P(neutralidad).
        Solo P(contradiccion) se usa para decidir si hay un conflicto.

        Version PFC3 (Mejora C): cuando la direccion directa supera el
        umbral, la direccion inversa tambien debe superarlo; se usa el
        minimo de ambas como P(contradiccion) final.
        """

        if not pares:
            return []

        resultados = []
        for id_memoria, fi, mj in pares:
            try:
                p_contra, probs = self._p_contradiccion_confirmada(fi, mj)
                resultado = {
                    "id_memoria": id_memoria,
                    "memoria_texto": mj,
                    "hecho_nuevo": fi,
                    **probs,
                    "p_contradiccion": p_contra,
                }
                resultados.append(resultado)
                logger.debug(
                    f"Par evaluado — "
                    f"P(contradicción)={p_contra:.3f} | "
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
        el LLM las almaceno en multiples entradas durante sesiones anteriores.

        Caso 1 : Si alguna P(contradiccion) > 0.85:
            Hay contradiccion : devuelve (True, [lista de conflictos])
            El modulo marcara (o borrara) todas las memorias en conflicto.

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

    def marcar_obsoleta(self, vector_store, vector_id, reemplazo: str = "") -> bool:
        """
        Mejora B: marcado de la memoria obsoleta en lugar de eliminacion.

        Conserva la memoria en Qdrant pero:
        - Prefija su texto con [OUTDATED as of <fecha>] para que el modelo
          de lenguaje la reconozca como valor previo al responder.
        - Registra en el payload: status=historical, superseded_by (el hecho
          que la reemplaza) e invalidated_at (fecha de invalidacion).

        El vector de embedding NO se recalcula: la memoria sigue siendo
        recuperable por su contenido original, lo que permite responder
        preguntas de valor previo.

        Retorna True si la memoria quedo marcada.
        """
        try:
            punto = vector_store.get(vector_id=vector_id)
            payload = getattr(punto, "payload", None) if punto is not None else None
            if not payload:
                return False
            texto_original = payload.get("data", "")
            if not texto_original:
                return False
            if texto_original.startswith(MARCA_OBSOLETA):
                return True  # ya estaba marcada

            fecha = datetime.now().strftime("%Y-%m-%d")
            nuevo_payload = {
                "data": f"{MARCA_OBSOLETA} as of {fecha}] {texto_original}",
                "status": "historical",
                "superseded_by": (reemplazo or "")[:300],
                "invalidated_at": fecha,
            }
            # set_payload fusiona claves: created_at, user_id, etc. se preservan
            vector_store.update(vector_id=vector_id, payload=nuevo_payload)
            self.marcadas += 1
            logger.info(
                f"[NLI] Memoria marcada como historica id={vector_id}: "
                f"'{texto_original[:70]}'"
            )
            return True
        except Exception as e:
            logger.warning(f"[NLI] Error al marcar memoria {vector_id}: {e}")
            return False

    def aplicar_accion(self, vector_store, vector_id, reemplazo: str = "") -> str:
        """
        Aplica la accion sobre una memoria en conflicto segun la
        configuracion: marcado (Mejora B, por defecto) o eliminacion
        (comportamiento PFC2). Retorna "marcada" o "eliminada".
        """
        if self.usar_marcado and self.marcar_obsoleta(vector_store, vector_id, reemplazo):
            return "marcada"
        vector_store.delete(vector_id=vector_id)
        return "eliminada"

    def consolidar(self, vector_store, embedding_model, search_filters: dict) -> List[str]:
        """
        Fase de consolidacion post-carga.

        Despues de cargar TODAS las memorias, recorre cada una, busca sus
        memorias similares (score > 0.70 para evitar comparar temas distintos)
        y cuando detecta una contradiccion (P > umbral) resuelve la MAS ANTIGUA
        segun created_at, conservando la actualizacion mas reciente.

        A diferencia de verificar() que corre durante la carga y depende del
        orden de llegada, esta pasada ve el estado completo de la memoria, por
        lo que detecta contradicciones que el flujo incremental podria perder.

        Version PFC3:
        - Las memorias ya marcadas como historicas no participan.
        - Mejora A: el filtro de entidades descarta pares de temas distintos.
        - Mejora C: la contradiccion se confirma en ambas direcciones.
        - Mejora B: la memoria mas antigua se marca como historica en lugar
          de eliminarse (si usar_marcado esta activo).

        Retorna lista de IDs procesados (marcados o eliminados).
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
            # las memorias ya invalidadas no vuelven a compararse
            if texto and not texto.startswith(MARCA_OBSOLETA):
                memorias[p.id] = {
                    "text": texto,
                    "created_at": p.payload.get("created_at", ""),
                }

        if len(memorias) < 2:
            return []

        procesados = set()
        reemplazos = {}  # viejo_id -> texto de la memoria conservada
        for mid, info in list(memorias.items()):
            if mid in procesados:
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
                if s.id == mid or s.id in procesados or s.id not in memorias:
                    continue
                if s.score < 0.70:
                    continue
                texto_vecina = memorias[s.id]["text"]
                # Mejora A: descartar pares de temas distintos sin invocar NLI
                if self.usar_filtro_entidades and not self._filtro.par_es_comparable(info["text"], texto_vecina):
                    continue
                p_contra, _ = self._p_contradiccion_confirmada(info["text"], texto_vecina)
                if p_contra > self.umbral:
                    # resolver la mas antigua por created_at
                    if info["created_at"] <= memorias[s.id]["created_at"]:
                        viejo_id, nuevo_id = mid, s.id
                    else:
                        viejo_id, nuevo_id = s.id, mid
                    procesados.add(viejo_id)
                    reemplazos[viejo_id] = memorias[nuevo_id]["text"]
                    self.contradicciones += 1
                    accion = "marcada" if self.usar_marcado else "eliminada"
                    logger.info(
                        f"[NLI-Consolidacion] CONTRADICCIÓN P={p_contra:.3f}\n"
                        f"  {accion.capitalize()} (vieja): '{memorias[viejo_id]['text'][:70]}'\n"
                        f"  Conservada (nueva): '{memorias[nuevo_id]['text'][:70]}'"
                    )
                    if viejo_id == mid:
                        break

        for vid in procesados:
            try:
                self.aplicar_accion(vector_store, vid, reemplazo=reemplazos.get(vid, ""))
            except Exception as e:
                logger.warning(f"[NLI-Consolidacion] Error al procesar {vid}: {e}")

        return list(procesados)
