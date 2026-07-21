"""
Paquete local del proyecto.

Nota:
Este proyecto usa el nombre de paquete `gym`, lo que puede "sombrar" (shadow)
al paquete externo OpenAI Gym cuando otras librerías hacen `import gym`.
Stable-Baselines3 consulta `gym.__version__` al guardar modelos.

Solución mínima:
exponer `__version__` para evitar AttributeError.
"""

__version__ = "0.0.0-local"