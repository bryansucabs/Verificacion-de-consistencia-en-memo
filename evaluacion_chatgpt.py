"""Evaluacion de Mem0 con un modelo de OpenAI (por defecto gpt-6-luna).

Uso:
    py -3.12 evaluacion_chatgpt.py <oracle|small> <base|modulo> [numero] [--modelo=...] [--tope=USD] [--puerto=N]

Ejemplos:
    py -3.12 evaluacion_chatgpt.py oracle base 2          (prueba pagada de 2 preguntas)
    py -3.12 evaluacion_chatgpt.py oracle base
    py -3.12 evaluacion_chatgpt.py oracle modulo --modelo=gpt-5.4-nano

Necesita la variable OPENAI_API_KEY y un Qdrant propio en el puerto 6335.
El detalle de las opciones esta en evaluacion_api.py.
"""
from evaluacion_api import ejecutar

ejecutar("chatgpt")
