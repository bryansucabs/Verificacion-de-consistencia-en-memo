"""
Filtro de entidades previo al NLI (Mejora A - PFC3).

Antes de pasar un par de textos a DeBERTa, verifica que ambos hablen
del mismo tema comparando sus entidades nombradas y sustantivos clave.
Si ambos textos tienen terminos tematicos y no comparten ninguno (ni
tienen entidades del mismo tipo), el par se descarta sin invocar al
NLI: son temas distintos y evaluarlos solo produciria falsas
contradicciones.

Reglas de decision (en orden):
  1. Si alguno de los textos no tiene terminos extraibles -> comparable
     (decision conservadora: se deja que el NLI decida).
  2. Si comparten al menos un termino tematico -> comparable.
  3. Si ambos tienen entidades nombradas del MISMO TIPO (p.ej. ambos
     mencionan una ORG, o ambos un lugar GPE) -> comparable. Esta regla
     cubre las actualizaciones genuinas, donde el valor viejo y el nuevo
     son entidades DISTINTAS del mismo tipo ("works at Google" vs
     "works at Microsoft").
  4. En caso contrario -> NO comparable (se descarta sin NLI).

Usa spaCy (en_core_web_sm) para NER y lematizacion. Si spaCy no esta
disponible, cae a una heuristica basada en palabras significativas.

Los numeros y fechas NO se usan para la interseccion: en una
actualizacion genuina el numero es justo lo que cambia
("record is 25:50" vs "record is 23:40" comparten tema pero no numero).
"""

import logging
import re
from typing import Set, Tuple

logger = logging.getLogger(__name__)

# Etiquetas de spaCy que representan entidades tematicas (identifican
# de QUE se habla). Se excluyen las de valor (DATE, TIME, CARDINAL,
# QUANTITY, MONEY, PERCENT, ORDINAL) porque el valor es lo que cambia
# en una actualizacion.
ETIQUETAS_TEMATICAS = {
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "FAC", "NORP", "LANGUAGE", "LAW",
}

# Stopwords minimas propias. Incluye palabras que aparecen en casi
# todas las memorias extraidas por Mem0 ("user") y sustantivos
# genericos que no discriminan tema.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "may", "might", "my", "your", "his", "her", "its",
    "our", "their", "i", "you", "he", "she", "it", "we", "they", "me",
    "him", "them", "this", "that", "these", "those", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "up", "about", "into",
    "and", "or", "but", "not", "no", "so", "if", "as", "than", "too",
    "very", "just", "now", "then", "there", "here", "user", "also",
    "new", "old", "currently", "recently", "lot", "thing", "things",
    "time", "way", "day", "today", "yesterday", "week", "month", "year",
    # verbos genericos que no discriminan tema
    "like", "get", "go", "make", "take", "know", "think", "see",
    "come", "want", "say", "tell", "ask", "give", "find", "use",
}

_RE_NUMERO = re.compile(r"^\d+([.,:]\d+)*\w*$")


class FiltroEntidades:
    """
    Decide si un par de textos es comparable (mismo tema) antes de
    invocar al modelo NLI.
    """

    def __init__(self):
        self._nlp = None
        self._spacy_disponible = None
        self.pares_descartados = 0

    def _cargar(self):
        if self._spacy_disponible is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm", disable=["parser"])
                self._spacy_disponible = True
                logger.info("[FiltroEntidades] spaCy en_core_web_sm cargado")
            except Exception as e:
                self._spacy_disponible = False
                logger.warning(
                    f"[FiltroEntidades] spaCy no disponible ({e}), "
                    "usando heuristica de palabras significativas"
                )

    def _extraer(self, texto: str) -> Tuple[Set[str], Set[str]]:
        """
        Extrae (terminos, tipos_de_entidad) de un texto:
        - terminos: entidades nombradas tematicas + lemas de sustantivos
          y nombres propios. Numeros y fechas excluidos.
        - tipos_de_entidad: etiquetas NER tematicas presentes (ORG, GPE...).
        """
        self._cargar()
        texto = texto.strip()
        if not texto:
            return set(), set()

        if self._spacy_disponible:
            doc = self._nlp(texto)
            terminos = set()
            tipos = set()
            # Entidades nombradas tematicas (multi-palabra, normalizadas).
            # Las entidades precedidas por "named"/"called" son nombres
            # propios de mascotas u objetos ("a cat named Toby") que el
            # NER de spaCy suele etiquetar mal; su TIPO no se cuenta
            # para la regla 3, aunque el termino si se conserva.
            for ent in doc.ents:
                if ent.label_ in ETIQUETAS_TEMATICAS:
                    terminos.add(ent.text.lower().strip())
                    tok_previo = doc[ent.start - 1] if ent.start > 0 else None
                    if tok_previo is not None and tok_previo.lemma_.lower() in ("name", "call"):
                        continue
                    tipos.add(ent.label_)
            # Lemas de sustantivos, nombres propios y verbos con contenido
            # (el verbo indica el atributo del hecho: work, live, study...)
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN", "VERB") and not token.is_stop:
                    lema = token.lemma_.lower().strip()
                    if (
                        len(lema) > 1
                        and lema not in _STOPWORDS
                        and not _RE_NUMERO.match(lema)
                    ):
                        terminos.add(lema)
            return terminos, tipos

        # Fallback sin spaCy: palabras significativas; las palabras
        # capitalizadas (no al inicio de oracion) aproximan entidades
        palabras = re.findall(r"[A-Za-z][A-Za-z\-']+", texto)
        terminos = {
            p.lower() for p in palabras
            if p.lower() not in _STOPWORDS and len(p) > 2
        }
        tipos = {"CAP"} if any(
            p[0].isupper() for p in palabras[1:]
        ) else set()
        return terminos, tipos

    def extraer_terminos(self, texto: str) -> Set[str]:
        """Devuelve solo el conjunto de terminos tematicos del texto."""
        terminos, _ = self._extraer(texto)
        return terminos

    def par_es_comparable(self, texto_a: str, texto_b: str) -> bool:
        """
        True si ambos textos pueden tratar el mismo tema:
        comparten terminos, o tienen entidades del mismo tipo (posible
        actualizacion de valor), o alguno no tiene terminos extraibles.
        False si ambos tienen terminos y ninguna condicion se cumple.
        """
        terminos_a, tipos_a = self._extraer(texto_a)
        terminos_b, tipos_b = self._extraer(texto_b)

        # Regla 1: sin evidencia suficiente, no descartar (conservador)
        if not terminos_a or not terminos_b:
            return True

        # Regla 2: comparten al menos un termino tematico
        if terminos_a & terminos_b:
            return True

        # Regla 3: entidades del mismo tipo = posible actualizacion
        # de valor ("works at Google" vs "works at Microsoft": ambas ORG)
        if tipos_a & tipos_b:
            return True

        # Regla 4: temas distintos, descartar sin invocar NLI
        self.pares_descartados += 1
        logger.debug(
            f"[FiltroEntidades] Par descartado sin NLI — "
            f"terminos A={terminos_a} | terminos B={terminos_b}"
        )
        return False
