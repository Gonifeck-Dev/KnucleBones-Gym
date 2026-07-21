# gym/policies/decision_tree/sklearn_tree_policy.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import joblib


from gym.policies.utils.base_policy import BasePolicy, PolicyStep
from gym.policies.utils.features import extract_features


class SklearnTreePolicy(BasePolicy):
    """
    Policy de inferencia para Decision Tree entrenado con scikit-learn.

    Regla A:
    - no entrena
    - no escribe archivos
    - solo inferencia

    Requiere:
    - un modelo guardado con joblib (DecisionTreeClassifier o compatible)
    """

    def __init__(
        self,
        model_path: str,
        name: str = "dt_sklearn",
        fallback: str = "first",
    ):
        super().__init__(name=name)
        if fallback not in ("first", "random"):
            raise ValueError("fallback must be 'first' or 'random'")
        self.model_path = model_path
        self.fallback = fallback
        self.model = self._load_model(model_path)

    def _load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        return joblib.load(path)

    def select_action(
        self,
        obs: Dict[str, Any],
        legal_actions: Optional[List[int]] = None,
    ) -> PolicyStep:
        if legal_actions is None or len(legal_actions) == 0:
            raise ValueError("SklearnTreePolicy requires non-empty legal_actions")

        x = extract_features(obs).reshape(1, -1)  # (1, n_features)
        pred = self.model.predict(x)
        a = int(pred[0])

        # Legalidad + fallback
        if a not in legal_actions:
            if self.fallback == "first":
                a_fb = int(legal_actions[0])
            else:
                # Fallback random reproducible: derive from obs
                # (no usamos RNG global; generamos una pseudo-seed estable)
                s = int(obs.get("turn", 0)) + 1000 * int(obs.get("episode_id", 0))
                a_fb = int(legal_actions[s % len(legal_actions)])
            return PolicyStep(
                action=a_fb,
                info={
                    "policy": self.name,
                    "pred_action": a,
                    "used_fallback": True,
                    "fallback": self.fallback,
                    "n_features": int(x.shape[1]),
                },
            )

        return PolicyStep(
            action=a,
            info={
                "policy": self.name,
                "pred_action": a,
                "used_fallback": False,
                "n_features": int(x.shape[1]),
            },
        )