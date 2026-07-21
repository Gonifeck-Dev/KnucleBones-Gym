# gym/policies/utils/base_policy.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PolicyStep:
    action: int
    info: Dict[str, Any]


class BasePolicy(ABC):
    """
    Contrato mínimo para policies enchufables.
    - No entrenan
    - No escriben archivos
    - Deciden acción por turno
    """

    def __init__(self, name: str):
        self.name = name

    def reset(self, seed: Optional[int] = None) -> None:
        return None

    def on_turn_end(self, record: Dict[str, Any]) -> None:
        """
        Hook opcional: el evaluador llama esto al final de cada turno,
        con el record del turno (die, action, player, etc.).
        Por defecto no hace nada.
        """
        return None

    @abstractmethod
    def select_action(
        self,
        obs: Dict[str, Any],
        legal_actions: Optional[List[int]] = None,
    ) -> PolicyStep:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"