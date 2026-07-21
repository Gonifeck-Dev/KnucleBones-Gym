# gym/policies/kmeans/kmeans_adapter_policy.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from gym.policies.utils.base_policy import BasePolicy, PolicyStep
from gym.policies.kmeans.profile_features import OpponentProfileWindowV2


def _build_policy_lazy(spec: str) -> BasePolicy:
    """
    Import lazy para evitar circular imports:
    policy_factory puede importar SystemPolicy, que importa este módulo.
    """
    from gym.policies.utils.policy_factory import build_policy  # noqa: WPS433

    return build_policy(spec)


class KMeansAdapterPolicyV2(BasePolicy):
    """
    Adaptación online (sin entrenar pesos):
    - Observa acciones del oponente + legal_actions
    - Predice cluster (KMeans)
    - Selecciona policy respuesta (response table: cluster -> policy_spec)

    Notas:
    - fallback_policy_spec configurable (para warmup antes de tener cluster)
    - cachea subpolicies por cluster
    """

    def __init__(
        self,
        kmeans_artifact_path: str,
        response_table_path: str,
        name: str = "kmeans_adapter_v2",
        my_player_id: int = 0,
        fallback_policy_spec: str = "baseline:first",
    ):
        super().__init__(name=name)
        self.my_player_id = int(my_player_id)

        art = joblib.load(kmeans_artifact_path)
        if int(art.get("profile_version", 0)) != 2:
            raise ValueError("KMeansAdapterPolicyV2 requires profile_version=2")

        self.window = int(art["window"])
        self.profile = OpponentProfileWindowV2(window=self.window)

        self.scaler = art["scaler"]
        self.kmeans = art["kmeans"]

        rt = json.loads(Path(response_table_path).read_text(encoding="utf-8-sig"))
        raw_map = rt.get("cluster_to_policy_spec", {})
        self.cluster_to_policy_spec = {int(k): v for k, v in raw_map.items() if v is not None}

        # cache policies: cluster -> policy instance
        self._policies: Dict[int, BasePolicy] = {}

        # fallback
        self.fallback_policy_spec = fallback_policy_spec
        self.fallback_policy = _build_policy_lazy(fallback_policy_spec)

        # state
        self.current_cluster: int = -1
        self.switch_count: int = 0

    def reset(self, seed: Optional[int] = None) -> None:
        self.profile.reset()
        self.current_cluster = -1
        self.switch_count = 0

        self.fallback_policy.reset(seed=seed)
        for p in self._policies.values():
            p.reset(seed=seed)

    def on_turn_end(self, record: Dict[str, Any]) -> None:
        player = int(record.get("player"))
        if player == self.my_player_id:
            return

        action = int(record.get("action"))
        legal = record.get("legal_actions") or []
        self.profile.update(action=action, legal_actions=legal)

    def _predict_cluster(self) -> int:
        if not self.profile.ready():
            return -1
        z = self.profile.vector().reshape(1, -1)
        zs = self.scaler.transform(z)
        return int(self.kmeans.predict(zs)[0])

    def _policy_for_cluster(self, c: int) -> BasePolicy:
        if c in self._policies:
            return self._policies[c]

        spec = self.cluster_to_policy_spec.get(c, self.fallback_policy_spec)
        pol = _build_policy_lazy(spec)
        self._policies[c] = pol
        return pol

    def select_action(self, obs: Dict[str, Any], legal_actions: Optional[List[int]] = None) -> PolicyStep:
        if not legal_actions:
            raise ValueError("KMeansAdapterPolicyV2 requires legal_actions")

        predicted = self._predict_cluster()
        switched = False

        if predicted != -1 and predicted != self.current_cluster:
            self.current_cluster = predicted
            self.switch_count += 1
            switched = True

        if self.current_cluster == -1:
            step = self.fallback_policy.select_action(obs=obs, legal_actions=legal_actions)
            info = dict(step.info)
            info.update({
                "meta": self.name,
                "cluster_ready": False,
                "cluster": -1,
                "switched": False,
                "switch_count": self.switch_count,
                "selected_policy": self.fallback_policy.name,
            })
            return PolicyStep(action=int(step.action), info=info)

        pol = self._policy_for_cluster(self.current_cluster)
        step = pol.select_action(obs=obs, legal_actions=legal_actions)

        info = dict(step.info)
        info.update({
            "meta": self.name,
            "cluster_ready": True,
            "cluster": int(self.current_cluster),
            "switched": switched,
            "switch_count": self.switch_count,
            "selected_policy": pol.name,
        })
        return PolicyStep(action=int(step.action), info=info)