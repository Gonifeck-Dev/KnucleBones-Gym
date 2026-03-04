# gym/policies/archetypes/greedy.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from gym.policies.utils.base_policy import BasePolicy, PolicyStep
from gym.policies.archetypes.base import ArchetypePolicyBase, simulate_action


class GreedyPolicy(BasePolicy, ArchetypePolicyBase):
    """
    Arquetipo GREEDY:
    - Maximiza ganancia inmediata de puntaje propio (delta_my).
    - Desempata por delta_diff y destrucción.
    """

    def __init__(self, seed: int = 123):
        BasePolicy.__init__(self, name="heuristic_greedy")
        ArchetypePolicyBase.__init__(self, name="heuristic_greedy", seed=seed)

    def reset(self, seed: Optional[int] = None) -> None:
        ArchetypePolicyBase.reset(self, seed)

    def select_action(self, obs: Dict[str, Any], legal_actions: Optional[List[int]] = None) -> PolicyStep:
        if not legal_actions:
            raise ValueError("legal_actions required")

        sims = [simulate_action(obs, a) for a in legal_actions]

        # max delta_my, tie: delta_diff, tie: destroyed, tie: lowest action
        sims.sort(key=lambda s: (s.delta_my, s.delta_diff, s.destroyed, -s.action), reverse=True)
        best = sims[0]

        info = {
            "archetype": "greedy",
            "delta_my": best.delta_my,
            "delta_op": best.delta_op,
            "delta_diff": best.delta_diff,
            "destroyed": best.destroyed,
        }
        return PolicyStep(action=int(best.action), info=info)