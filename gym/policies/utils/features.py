# gym/policies/utils/features.py
"""
Vector de features canónico para el estado del juego.

Todos los modelos (DT, RL, NEAT) deben usar extract_features()
para garantizar consistencia entre entrenamiento e inferencia.
"""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

# Incrementar si se cambia la composición o el orden del vector.
# Todos los modelos entrenados con una versión anterior son incompatibles.
FEATURES_VERSION = 1
FEATURES_DIM = 21


def flatten_board(cols: List[List[int]]) -> List[int]:
    """
    Aplana tablero 3x3 en orden por columna (c0 r0..r2, c1..., c2...).
    """
    flat: List[int] = []
    for c in range(3):
        flat.extend(int(v) for v in cols[c])
    return flat


def extract_features(obs: Dict[str, Any]) -> np.ndarray:
    """
    Extrae un vector de features NUMÉRICO y estable.

    Diseño:
    - Mantenerlo simple y reproducible para tesis.
    - Evitar features "mágicas" sin justificación.

    Features (21):
    [ dice_value,
      my_board_flat(9),
      op_board_flat(9),
      score_my,
      score_op
    ]
    """
    dice = obs.get("dice_value", None)
    if dice is None:
        raise ValueError("obs missing dice_value")

    my_cols = obs.get("my_cols", None)
    op_cols = obs.get("op_cols", None)
    if my_cols is None or op_cols is None:
        raise ValueError("obs missing my_cols/op_cols")

    score_my = obs.get("score_my", None)
    score_op = obs.get("score_op", None)
    if score_my is None or score_op is None:
        raise ValueError("obs missing score_my/score_op")

    x = [int(dice)]
    x += flatten_board(my_cols)
    x += flatten_board(op_cols)
    x += [int(score_my), int(score_op)]

    return np.array(x, dtype=np.float32)