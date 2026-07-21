# gym/policies/archetypes/random_style.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from gym.policies.utils.base_policy import BasePolicy, PolicyStep
from gym.policies.archetypes.base import ArchetypePolicyBase


class RandomArchetypePolicy(BasePolicy, ArchetypePolicyBase):
    """
    Arquetipo RANDOM:
    - Selecciona una acción legal al azar (reproducible con seed).
    """

    def __init__(self, seed: int = 123):
        BasePolicy.__init__(self, name="heuristic_random")
        ArchetypePolicyBase.__init__(self, name="heuristic_random", seed=seed)

    def reset(self, seed: Optional[int] = None) -> None:
        ArchetypePolicyBase.reset(self, seed)

    def select_action(self, obs: Dict[str, Any], legal_actions: Optional[List[int]] = None) -> PolicyStep:
        if not legal_actions:
            raise ValueError("legal_actions required")

        a = self._rng.choice(list(legal_actions))
        return PolicyStep(action=int(a), info={"archetype": "random"})