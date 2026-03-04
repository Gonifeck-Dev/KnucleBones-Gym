"""
gym.policies — Módulos de inferencia de IA (Regla A).

Cada sub-paquete implementa una familia de policies:
  - archetypes: heurísticas manuales (greedy, denial, spread, random)
  - decision_tree: baseline y sklearn DT
  - kmeans: adaptación online con clustering
  - neat: neuroevolución (NEAT-Python)
  - rl: aprendizaje por refuerzo (Stable-Baselines3)
  - systems: meta-sistemas que combinan KMeans + especialistas
  - utils: BasePolicy, features, policy_factory
"""

