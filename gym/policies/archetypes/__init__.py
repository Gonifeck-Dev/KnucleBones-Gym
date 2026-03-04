# gym/policies/archetypes/__init__.py
from .greedy import GreedyPolicy
from .denial import DenialPolicy
from .spread import SpreadPolicy
from .random_style import RandomArchetypePolicy

__all__ = [
    "GreedyPolicy",
    "DenialPolicy",
    "SpreadPolicy",
    "RandomArchetypePolicy",
]