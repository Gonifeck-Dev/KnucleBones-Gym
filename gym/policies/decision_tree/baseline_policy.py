# gym/policies/decision_tree/baseline_policy.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import random

from gym.policies.utils.base_policy import BasePolicy, PolicyStep


class BaselinePolicy(BasePolicy):
    """
    Baseline simple (determinista o random) para:
    - sanity checks
    - generar dataset por imitación
    """

    def __init__(self, name: str = "baseline", mode: str = "first", seed: int = 123):
        super().__init__(name=name)
        if mode not in ("first", "random"):
            raise ValueError("mode must be 'first' or 'random'")
        self.mode = mode
        self._rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = random.Random(int(seed))

    def select_action(
        self,
        obs: Dict[str, Any],
        legal_actions: Optional[List[int]] = None,
    ) -> PolicyStep:
        if legal_actions is None:
            raise ValueError("BaselinePolicy requires legal_actions")

        if len(legal_actions) == 0:
            raise ValueError("No legal actions available")

        if self.mode == "first":
            a = legal_actions[0]
        else:
            a = self._rng.choice(legal_actions)

        return PolicyStep(action=int(a), info={"policy": self.name, "mode": self.mode})