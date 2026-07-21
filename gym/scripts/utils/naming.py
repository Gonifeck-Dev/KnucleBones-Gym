# gym/scripts/utils/naming.py
"""
Utilidades centralizadas de naming para artefactos y carpetas generadas.

Todos los scripts de entrenamiento y evaluación deben usar estas funciones
en lugar de definir sus propias versiones.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone


def utc_stamp() -> str:
    """Timestamp UTC compacto para metadata (NO para nombres de archivo)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def safe_name(s: str, max_len: int = 140) -> str:
    """
    Convierte un string arbitrario en un nombre seguro para archivos/carpetas.

    - Reemplaza caracteres no alfanuméricos (excepto - y _) por _
    - Colapsa múltiples _ consecutivos
    - Recorta a max_len caracteres
    """
    s = s.strip().lower().replace("\\", "_").replace("/", "_").replace(":", "__")
    s = re.sub(r"[^a-z0-9_\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len]


