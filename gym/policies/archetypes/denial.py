# gym/policies/archetypes/denial.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from gym.policies.utils.base_policy import BasePolicy, PolicyStep
from gym.policies.archetypes.base import ArchetypePolicyBase, simulate_action


class DenialPolicy(BasePolicy, ArchetypePolicyBase):
    """
    Arquetipo DENIAL (agresivo/anti-oponente):
    - Maximiza daño al rival: (score_op_before - score_op_after) => -delta_op
    - Equivalente: maximiza (-delta_op) y/o destroyed.
    - Desempata por delta_diff y luego delta_my.
    """

    def __init__(self, seed: int = 123):
        BasePolicy.__init__(self, name="heuristic_denial")
        ArchetypePolicyBase.__init__(self, name="heuristic_denial", seed=seed)

    def reset(self, seed: Optional[int] = None) -> None:
        ArchetypePolicyBase.reset(self, seed)

    def select_action(self, obs: Dict[str, Any], legal_actions: Optional[List[int]] = None) -> PolicyStep:
        if not legal_actions:
            raise ValueError("legal_actions required")

        sims = [simulate_action(obs, a) for a in legal_actions]

        # max opponent damage = -delta_op, tie: destroyed, tie: delta_diff, tie: delta_my
        sims.sort(key=lambda s: (-s.delta_op, s.destroyed, s.delta_diff, s.delta_my, -s.action), reverse=True)
        best = sims[0]

        info = {
            "archetype": "denial",
            "opp_damage": -best.delta_op,
            "destroyed": best.destroyed,
            "delta_diff": best.delta_diff,
            "delta_my": best.delta_my,
        }
        return PolicyStep(action=int(best.action), info=info)