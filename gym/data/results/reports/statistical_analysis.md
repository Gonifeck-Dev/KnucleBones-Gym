# Análisis Estadístico Consolidado

## Ablation Study — ¿Aporta la Adaptación KMeans?

| Familia | Peor Generalista | Mejor Generalista | Sistema Adaptativo | Oráculo |
|---------|:----------------:|:-----------------:|:------------------:|:-------:|
| DT | 0.5845 | 0.6176 | 0.6079 | 0.6192 |
| PPO | 0.5719 | 0.6096 | 0.6044 | 0.6174 |
| NEAT | 0.5431 | 0.6512 | 0.6527 | 0.6512 |

**Interpretación**: Si Sistema ≈ Mejor Generalista, la adaptación KMeans no aporta valor significativo.
Si Sistema > Mejor Generalista, la adaptación sí mejora el rendimiento.

## Dominancia en Response Tables

| Sistema | Clusters | Especialista Dominante | Cobertura |
|---------|:--------:|:----------------------:|:---------:|
| system_dt | 4 | sklearn_tree__vs_heuristic_denial | 2/4 (50%) |
| system_ppo | 4 | PPO__vs_heuristic_denial | 2/4 (50%) |
| system_neat | 4 | neat_config | 4/4 (100%) |

