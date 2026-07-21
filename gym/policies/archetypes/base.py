# gym/policies/archetypes/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import random

from gym.env.knucklebones_env import (
    Board,
    deep_copy_board,
    first_empty_row,
    destroy_in_column,
    score_board,
)



@dataclass(frozen=True)
class SimResult:
    action: int
    delta_my: int
    delta_op: int
    delta_diff: int
    destroyed: int
    my_score_after: int
    op_score_after: int
    dup_count_after: int
    empties_after: int
    col_full_after: int


def simulate_action(obs: Dict[str, Any], action: int) -> SimResult:
    """
    Simula colocar dice_value en una columna y aplicar regla de destrucción.
    Obs usa el contrato del env:
      - my_cols, op_cols, dice_value, score_my, score_op
    """
    die = int(obs["dice_value"])
    my_before = deep_copy_board(obs["my_cols"])
    op_before = deep_copy_board(obs["op_cols"])

    score_my_before = int(obs["score_my"])
    score_op_before = int(obs["score_op"])

    row = first_empty_row(my_before[action])
    if row is None:
        raise ValueError(f"Illegal action {action} (column full)")

    # Place die
    my_before[action][row] = die

    # Destruction: remove opp cells == die in same column
    destroyed = sum(1 for x in op_before[action] if x == die)
    destroy_in_column(op_before, action, die)

    my_after = my_before
    op_after = op_before

    score_my_after = score_board(my_after)
    score_op_after = score_board(op_after)

    delta_my = score_my_after - score_my_before
    delta_op = score_op_after - score_op_before
    delta_diff = (score_my_after - score_op_after) - (score_my_before - score_op_before)

    # Defensive metrics
    dup_count_after = sum(1 for x in my_after[action] if x == die)
    empties_after = sum(1 for c in range(3) for r in range(3) if my_after[c][r] == 0)
    col_full_after = 1 if first_empty_row(my_after[action]) is None else 0

    return SimResult(
        action=action,
        delta_my=delta_my,
        delta_op=delta_op,
        delta_diff=delta_diff,
        destroyed=destroyed,
        my_score_after=score_my_after,
        op_score_after=score_op_after,
        dup_count_after=dup_count_after,
        empties_after=empties_after,
        col_full_after=col_full_after,
    )


class ArchetypePolicyBase:
    """
    Base simple para arquetipos: usa RNG reproducible.
    """

    def __init__(self, name: str, seed: int = 123):
        self.name = name
        self._rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = random.Random(int(seed))