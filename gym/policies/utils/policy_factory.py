# gym/policies/utils/policy_factory.py
from __future__ import annotations

from pathlib import Path

from gym.policies.utils.base_policy import BasePolicy

from gym.policies.decision_tree import BaselinePolicy, SklearnTreePolicy
from gym.policies.rl import SB3Policy
from gym.policies.neat import NEATPolicy

from gym.policies.archetypes import (
    GreedyPolicy,
    DenialPolicy,
    SpreadPolicy,
    RandomArchetypePolicy,
)


def build_policy(spec: str) -> BasePolicy:
    """
    Única fuente de verdad para parsear specs de policies.

    Soporta:
      - baseline:first | baseline:random
      - heuristic:greedy | heuristic:denial | heuristic:spread | heuristic:random
      - dt:<path.joblib>
      - rl:PPO:<path.zip> | rl:DQN:<path.zip>
      - neat:<genome.pkl>:<config.ini>    (split por último ":" para soportar C:\\...)
      - system:<config.json>             (import lazy para evitar circular imports)
    """
    if ":" not in spec:
        raise ValueError("Policy spec must contain ':' (e.g., baseline:first)")

    kind, rest = spec.split(":", 1)
    kind = kind.strip().lower()
    rest = rest.strip()

    if kind == "baseline":
        return BaselinePolicy(name=f"baseline_{rest}", mode=rest)

    if kind == "heuristic":
        key = rest.lower()
        if key == "greedy":
            return GreedyPolicy()
        if key == "denial":
            return DenialPolicy()
        if key == "spread":
            return SpreadPolicy()
        if key == "random":
            return RandomArchetypePolicy()
        raise ValueError(f"Unknown heuristic archetype: {rest}")

    if kind in ("dt", "decision_tree", "sklearn"):
        return SklearnTreePolicy(model_path=rest, name="dt_sklearn", fallback="first")

    if kind == "rl":
        # rl:PPO:path.zip
        algo, path = rest.split(":", 1)
        algo = algo.strip()
        path = path.strip()
        return SB3Policy(model_path=path, algo=algo, name=f"rl_{algo.lower()}")

    if kind == "neat":
        # neat:<genome.pkl>:<config.ini>  (split por último ":" para soportar C:\...)
        genome_path, cfg_path = rest.rsplit(":", 1)
        return NEATPolicy(genome_path=genome_path.strip(), config_path=cfg_path.strip(), name="neat")

    if kind == "system":
        # IMPORT LAZY: evita circular import
        from gym.policies.systems.system_policy import SystemPolicy  # noqa: WPS433

        cfg_path = Path(rest)
        return SystemPolicy.from_json(cfg_path)

    raise ValueError(f"Unknown policy kind: {kind}")