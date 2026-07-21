# Evaluación Cruzada — Generalización vs Overfitting

> 2000 partidas por matchup | seed=123 | 197s total

★ = oponente de entrenamiento (matchup esperado mejor)

## NEAT Especialistas

| Especialista | vs denial | vs spread | vs greedy | Δ train vs otros |
|-------------|-----------|-----------|-----------|-----------------|
| neat → denial | 0.635 ★ | 0.692 | 0.627 | -0.025 |
| neat → spread | 0.496 | 0.634 ★ | 0.499 | +0.136 |
| neat → greedy | 0.635 | 0.692 | 0.627 ★ | -0.036 |

## PPO Especialistas

| Especialista | vs denial | vs spread | vs greedy | Δ train vs otros |
|-------------|-----------|-----------|-----------|-----------------|
| ppo → denial | 0.598 ★ | 0.625 | 0.589 | -0.009 |
| ppo → spread | 0.531 | 0.650 ★ | 0.535 | +0.117 |
| ppo → greedy | 0.600 | 0.626 | 0.602 ★ | -0.011 |

## DT Especialistas

| Especialista | vs denial | vs spread | vs greedy | Δ train vs otros |
|-------------|-----------|-----------|-----------|-----------------|
| dt → denial | 0.589 ★ | 0.633 | 0.589 | -0.022 |
| dt → spread | 0.549 | 0.656 ★ | 0.548 | +0.107 |
| dt → greedy | 0.601 | 0.651 | 0.601 ★ | -0.025 |

## Análisis de Generalización

- **NEAT**: Δ promedio = +0.025 → 🔶 Ligera especialización (2-5%)
- **PPO**: Δ promedio = +0.032 → 🔶 Ligera especialización (2-5%)
- **DT**: Δ promedio = +0.020 → ✅ Generaliza bien (Δ < 2%)

**Interpretación del Δ (delta)**:
- Δ ≈ 0: El especialista rinde igual contra todos → buena generalización
- Δ > 0: El especialista rinde mejor contra su oponente de entrenamiento → especialización
- Δ > 0.05: El especialista rinde significativamente mejor solo contra su oponente → overfitting
