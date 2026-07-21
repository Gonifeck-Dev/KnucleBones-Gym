# gym/policies/rl/sb3_policy.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from gym.policies.utils.base_policy import BasePolicy, PolicyStep
from gym.policies.utils.features import extract_features


class SB3Policy(BasePolicy):
    """
    Wrapper de inferencia para modelos SB3 guardados (PPO/DQN/etc.).
    """

    def __init__(self, model_path: str, algo: str = "PPO", name: str = "rl_sb3", deterministic: bool = True):
        super().__init__(name=name)
        self.model_path = model_path
        self.algo = algo.upper()
        self.deterministic = bool(deterministic)
        self.model = self._load()

    def _load(self):
        # Forzar CPU para inferencia: la red MLP es tan pequeña (21→64→64→3)
        # que el overhead de transferencia CPU↔GPU es mayor que el cálculo.
        try:
            if self.algo == "PPO":
                from stable_baselines3 import PPO
                return PPO.load(self.model_path, device="cpu")
            if self.algo == "DQN":
                from stable_baselines3 import DQN
                return DQN.load(self.model_path, device="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load SB3 model ({self.algo}) from {self.model_path}: {e}") from e

        raise ValueError(f"Unsupported SB3 algo: {self.algo}")

    def select_action(self, obs: Dict[str, Any], legal_actions: Optional[List[int]] = None) -> PolicyStep:
        if legal_actions is None or len(legal_actions) == 0:
            raise ValueError("SB3Policy requires non-empty legal_actions")

        x = extract_features(obs)  # shape (21,)
        action, _ = self.model.predict(x, deterministic=self.deterministic)
        a = int(action)

        if a not in legal_actions:
            a = int(legal_actions[0])

        return PolicyStep(action=a, info={"policy": self.name, "algo": self.algo, "deterministic": self.deterministic})