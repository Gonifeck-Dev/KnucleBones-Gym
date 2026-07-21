# gym/policies/neat/neat_policy.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import neat
import numpy as np

from gym.policies.utils.base_policy import BasePolicy, PolicyStep
from gym.policies.utils.features import extract_features


class NEATPolicy(BasePolicy):
    def __init__(self, genome_path: str, config_path: str, name: str = "neat"):
        super().__init__(name=name)
        self.genome_path = genome_path
        self.config_path = config_path

        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path,
        )

        with open(self.genome_path, "rb") as f:
            self.genome = pickle.load(f)

        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

    def select_action(self, obs: Dict[str, Any], legal_actions: Optional[List[int]] = None) -> PolicyStep:
        if legal_actions is None or len(legal_actions) == 0:
            raise ValueError("NEATPolicy requires non-empty legal_actions")

        x = extract_features(obs).astype(np.float32)
        out = self.net.activate(x.tolist())  # length 3

        # elegir argmax entre legales
        best_a = legal_actions[0]
        best_v = float("-inf")
        for a in legal_actions:
            v = float(out[a])
            if v > best_v:
                best_v = v
                best_a = a

        return PolicyStep(action=int(best_a), info={"policy": self.name, "best_value": best_v})