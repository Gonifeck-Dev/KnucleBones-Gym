# gym/policies/systems/system_policy.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

from gym.policies.utils.base_policy import BasePolicy, PolicyStep
from gym.policies.kmeans.kmeans_adapter_policy import KMeansAdapterPolicyV2


@dataclass(frozen=True)
class SystemConfig:
    name: str
    kmeans_artifact: str
    response_table: str
    my_player_id: int = 0
    fallback_policy_spec: str = "baseline:first"

    @staticmethod
    def from_json(path: Path) -> "SystemConfig":
        # utf-8-sig elimina BOM si el editor guardó con BOM (Windows)
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return SystemConfig(
            name=str(data["name"]),
            kmeans_artifact=str(data["kmeans_artifact"]),
            response_table=str(data["response_table"]),
            my_player_id=int(data.get("my_player_id", 0)),
            fallback_policy_spec=str(data.get("fallback_policy_spec", "baseline:first")),
        )


class SystemPolicy(BasePolicy):
    """
    'Sistema' por familia: encapsula KMeansAdapter configurado mediante JSON.
    El response_table define qué especialistas se eligen por cluster.
    """

    def __init__(self, cfg: SystemConfig):
        super().__init__(name=cfg.name)
        self.cfg = cfg
        self.adapter = KMeansAdapterPolicyV2(
            kmeans_artifact_path=cfg.kmeans_artifact,
            response_table_path=cfg.response_table,
            name=cfg.name,
            my_player_id=cfg.my_player_id,
            fallback_policy_spec=cfg.fallback_policy_spec,
        )

    @staticmethod
    def from_json(path: Path) -> "SystemPolicy":
        return SystemPolicy(SystemConfig.from_json(path))

    def reset(self, seed: Optional[int] = None) -> None:
        self.adapter.reset(seed=seed)

    def on_turn_end(self, record: Dict[str, Any]) -> None:
        self.adapter.on_turn_end(record)

    def select_action(self, obs: Dict[str, Any], legal_actions: Optional[List[int]] = None) -> PolicyStep:
        return self.adapter.select_action(obs=obs, legal_actions=legal_actions)