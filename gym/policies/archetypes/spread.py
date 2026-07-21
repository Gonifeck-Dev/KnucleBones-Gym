# gym/policies/archetypes/spread.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from gym.policies.utils.base_policy import BasePolicy, PolicyStep
from gym.policies.archetypes.base import ArchetypePolicyBase, simulate_action


class SpreadPolicy(BasePolicy, ArchetypePolicyBase):
    """
    Arquetipo SPREAD (defensivo):
    Heurística práctica para Knucklebones:
    - Minimiza duplicados del dado en la columna (reduce "exposición" a destrucción futura).
    - Prefiere columnas con más espacio (mantener flexibilidad).
    - Desempata por delta_diff (si no cuesta defensa).
    """

    def __init__(self, seed: int = 123):
        BasePolicy.__init__(self, name="heuristic_spread")
        ArchetypePolicyBase.__init__(self, name="heuristic_spread", seed=seed)

    def reset(self, seed: Optional[int] = None) -> None:
        ArchetypePolicyBase.reset(self, seed)

    def select_action(self, obs: Dict[str, Any], legal_actions: Optional[List[int]] = None) -> PolicyStep:
        if not legal_actions:
            raise ValueError("legal_actions required")

        sims = [simulate_action(obs, a) for a in legal_actions]

        # Prefer:
        # - lower dup_count_after
        # - lower col_full_after (evitar cerrar columnas si no es necesario)
        # - higher empties_after
        # tie: higher delta_diff
        sims.sort(
            key=lambda s: (
                -s.dup_count_after,        # menor dup mejor
                -s.col_full_after,         # no cerrar columna mejor
                s.empties_after,           # más vacíos mejor
                s.delta_diff,              # si no afecta, mejorar diff
                -s.action,
            ),
            reverse=True,
        )
        best = sims[0]

        info = {
            "archetype": "spread",
            "dup_count_after": best.dup_count_after,
            "col_full_after": best.col_full_after,
            "empties_after": best.empties_after,
            "delta_diff": best.delta_diff,
        }
        return PolicyStep(action=int(best.action), info=info)