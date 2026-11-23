"""Evaluacion de Mem0 con un modelo de Anthropic (por defecto claude-haiku-5-5).

Uso:
    py -3.12 evaluacion_claude.py <oracle|small> <base|modulo> [numero] [--modelo=...] [--tope=USD] [--puerto=N]

Ejemplos:
    py -3.12 evaluacion_claude.py oracle base 2           (prueba pagada de 2 preguntas)
    py -3.12 evaluacion_claude.py oracle base

Necesita las variables ANTHROPIC_API_KEY y OPENAI_API_KEY (los embeddings son de
OpenAI), el paquete anthropic y un Qdrant propio en el puerto 6336.
El detalle de las opciones esta en evaluacion_api.py.
"""
from evaluacion_api import ejecutar

ejecutar("claude")
