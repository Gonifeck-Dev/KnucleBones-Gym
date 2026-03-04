# gym/policies/kmeans/profile_features_v2.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List
import numpy as np


@dataclass(frozen=True)
class OppMoveV2:
    action: int
    n_legal: int
    is_first_legal: int
    is_last_legal: int
    repeat_col: int


class OpponentProfileWindowV2:
    """
    Perfil v2 (ventana deslizante):
    Captura patrón de decisión del oponente relativo a acciones legales.

    Vector (10 dims):
      - hist_cols (3)
      - hist_n_legal (3)    # n_legal in {1,2,3}
      - mean_is_first_legal (1)
      - mean_is_last_legal  (1)
      - mean_repeat_col     (1)
      - entropy_cols        (1)
    """

    def __init__(self, window: int = 6):
        if window <= 0:
            raise ValueError("window must be > 0")
        self.window = int(window)
        self.buf: Deque[OppMoveV2] = deque(maxlen=self.window)
        self._prev_action: int | None = None

    def reset(self) -> None:
        self.buf.clear()
        self._prev_action = None

    def ready(self) -> bool:
        return len(self.buf) >= self.window

    def update(self, action: int, legal_actions: List[int]) -> None:
        if not legal_actions:
            return
        a = int(action)
        legal = [int(x) for x in legal_actions]
        n_legal = len(legal)
        first = min(legal)
        last = max(legal)

        is_first = 1 if a == first else 0
        is_last = 1 if a == last else 0
        repeat = 1 if (self._prev_action is not None and a == self._prev_action) else 0

        # clamp n_legal to 1..3
        if n_legal < 1:
            n_legal = 1
        if n_legal > 3:
            n_legal = 3

        self.buf.append(OppMoveV2(
            action=a,
            n_legal=n_legal,
            is_first_legal=is_first,
            is_last_legal=is_last,
            repeat_col=repeat,
        ))
        self._prev_action = a

    def vector(self) -> np.ndarray:
        if len(self.buf) == 0:
            return np.zeros((10,), dtype=np.float32)

        actions = np.array([m.action for m in self.buf], dtype=np.int64)
        n_legal = np.array([m.n_legal for m in self.buf], dtype=np.int64)
        firsts = np.array([m.is_first_legal for m in self.buf], dtype=np.float32)
        lasts = np.array([m.is_last_legal for m in self.buf], dtype=np.float32)
        reps = np.array([m.repeat_col for m in self.buf], dtype=np.float32)

        # hist columnas
        hist_cols = np.bincount(actions, minlength=3).astype(np.float32)
        hist_cols = hist_cols / max(1.0, float(hist_cols.sum()))

        # hist n_legal (1..3) -> bins [1,2,3]
        hist_n = np.zeros((3,), dtype=np.float32)
        for v in n_legal:
            hist_n[v - 1] += 1.0
        hist_n = hist_n / max(1.0, float(hist_n.sum()))

        # entropía de columnas (0..log(3))
        eps = 1e-8
        entropy = float(-np.sum(hist_cols * np.log(hist_cols + eps)))

        z = np.array([
            hist_cols[0], hist_cols[1], hist_cols[2],
            hist_n[0], hist_n[1], hist_n[2],
            float(firsts.mean()),
            float(lasts.mean()),
            float(reps.mean()),
            float(entropy),
        ], dtype=np.float32)
        return z