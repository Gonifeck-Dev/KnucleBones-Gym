# Documento 4 — Referencia de Scripts y Datos

> Qué hace cada script, qué produce, qué métricas captura.
> Todos los comandos están escritos para **Windows 11 PowerShell**.

---

## 1. Resumen del pipeline

```
PASO 1 ─ Entrenar PPO especialistas          (train_sb3.py)
PASO 2 ─ Entrenar NEAT especialistas         (train_neat.py)
PASO 3 ─ Generar datasets behavior cloning   (generate_dataset_teacher.py)
PASO 4 ─ Destilar a Decision Trees           (train_tree.py)
PASO 5 ─ Entrenar KMeans para perfilado      (train_kmeans.py)
PASO 6 ─ Construir response tables           (build_response_table.py)
PASO 7 ─ Evaluar cualquier matchup           (evaluate_any.py)
PASO 8 ─ Generar gráficos y tablas           (analysis_toolkit.py)
PASO 9 ─ Evaluación cruzada                  (cross_evaluate.py)
```

---

## 2. Scripts de entrenamiento

### 2.1 `train_sb3.py` — Entrenar PPO/DQN

**Comando**:
```powershell
python -m gym.scripts.rl.train_sb3 `
  --algo PPO `
  --timesteps 400000 `
  --opponent heuristic:denial `
  --n-envs 4 `
  --seed 123
```

**Parámetros**:

- `--algo` — Algoritmo: `PPO` o `DQN` (default: `PPO`)
- `--timesteps` — Pasos totales de entrenamiento (default: `200000`)
- `--seed` — Semilla para reproducibilidad (default: `123`)
- `--opponent` — Spec del oponente (default: `baseline:first`)
- `--reward-mode` — Modo de recompensa: `diff_delta` o `outcome` (default: `diff_delta`)
- `--out` — Nombre personalizado del modelo (default: generado automáticamente)
- `--n-envs` — Environments paralelos (default: `1`, recomendado: `4`)

> **Nota GPU/CPU**: PPO con MlpPolicy se ejecuta forzosamente en CPU. La GPU no aporta speedup con redes pequeñas (21→64→64→3) — el overhead de transferencia CPU↔GPU es mayor que el cálculo. El paralelismo se logra con `--n-envs` (SubprocVecEnv).

**Archivos generados** (en `gym/data/models/rl/`):

- `PPO__vs_heuristic_denial.zip` — Modelo SB3 serializado
- `PPO__vs_heuristic_denial.meta.json` — Metadata + métricas
- `PPO__vs_heuristic_denial.training_log.jsonl` — Curva de aprendizaje

**Métricas en meta.json**:
```json
{
  "algo": "PPO",
  "timesteps": 400000,
  "seed": 123,
  "opponent": "heuristic:denial",
  "wall_time_seconds": 1234.5,
  "model_size_bytes": 45678,
  "peak_memory_mb": 512.3,
  "training_log": "gym/data/models/rl/PPO__vs_heuristic_denial.training_log.jsonl"
}
```

**Métricas en training_log.jsonl** (una línea por rollout):
```json
{
  "timestep": 2048,
  "elapsed_seconds": 12.5,
  "ep_reward_mean": 0.65,
  "ep_len_mean": 22.1,
  "policy_loss": -0.032,
  "value_loss": 0.156,
  "entropy_loss": -1.23,
  "approx_kl": 0.008,
  "clip_fraction": 0.12,
  "explained_variance": 0.45
}
```

---

### 2.2 `train_neat.py` — Entrenar NEAT

**Comando** (single-seed con rotación):
```powershell
python -m gym.scripts.neat.train_neat `
  --opponent heuristic:denial `
  --generations 100 `
  --episodes-per-genome 50 `
  --seed 123
```

**Comando** (multi-semilla real — recomendado):
```powershell
python -m gym.scripts.neat.train_neat `
  --opponent heuristic:denial `
  --generations 300 `
  --episodes-per-genome 34 `
  --seeds "123,456,789"
```

**Parámetros**:

- `--opponent` — Spec del oponente (default: `baseline:first`)
- `--generations` — Generaciones evolutivas (default: `30`, recomendado: `100-300`)
- `--episodes-per-genome` — Partidas por seed para evaluar cada genoma (default: `20`, recomendado: `34-50`)
- `--seed` — Semilla única legacy (default: `123`)
- `--seeds` — Semillas múltiples separadas por coma (ej. `"123,456,789"`). Activa multi-semilla real: cada genoma se evalúa contra todas las seeds simultáneamente, promediando fitness. Total episodios = episodes × num\_seeds.
- `--config` — Archivo de configuración NEAT (default: `gym/config/neat/neat_config.ini`)
- `--workers` — Workers paralelos (default: `0` = auto = cpu\_cores - 2, `1` = secuencial)
- `--out` — Nombre personalizado del modelo de salida

> **Multi-semilla real**: Con `--seeds "123,456,789"` y `--episodes-per-genome 34`, cada genoma juega 34 × 3 = 102 episodios totales, evaluado contra 3 secuencias de dados distintas. Esto previene sobreajuste a una secuencia determinista particular.

> **NOTA importante**: Con pocos episodios por genoma (ej. 20), la señal de fitness es ruidosa y NEAT puede seleccionar genomas que ganaron por suerte. Se recomienda mínimo 50 episodios totales para genomas robustos.

> **Paralelismo**: Por defecto usa todos los cores menos 2 para evaluar genomas en paralelo con `multiprocessing.Pool`. Con 12 cores lógicos → 10 workers → speedup ~5-6x.

**Archivos generados** (en `gym/data/models/neat/`):

- `neat__vs_heuristic_denial.pkl` — Genoma ganador (pickle)
- `neat__vs_heuristic_denial.meta.json` — Metadata + métricas
- `neat__vs_heuristic_denial.generation_log.jsonl` — Curva evolutiva

**Métricas en generation_log.jsonl** (una línea por generación):
```json
{
  "generation": 5,
  "elapsed_seconds": 120.3,
  "best_fitness": 0.87,
  "avg_fitness": 0.64,
  "std_fitness": 0.12,
  "min_fitness": 0.31,
  "num_species": 3,
  "num_genomes": 150
}
```

---

### 2.3 `generate_dataset_teacher.py` — Dataset por Behavior Cloning

**Comando**:
```powershell
python -m gym.scripts.decision_tree.generate_dataset_teacher `
  --teacher "rl:PPO:gym/data/models/rl/PPO__vs_heuristic_denial.zip" `
  --opponent heuristic:denial `
  --episodes 50000 `
  --seed 123
```

**Parámetros**:

- `--teacher` — Spec del agente maestro **(requerido)**
- `--opponent` — Spec del oponente **(requerido)**
- `--episodes` — Episodios a generar (default: `50000`)
- `--seed` — Semilla (default: `123`)
- `--out-dir` — Directorio de salida (default: `gym/data/datasets/raw`)

**Archivos generados** (en `gym/data/datasets/raw/<nombre>/`):

- `samples.npz` — Arrays X (features) e y (acciones)
- `preview.csv` — Preview de 2000 muestras
- `meta.json` — Metadata + métricas

**Métricas en meta.json**:
```json
{
  "n_samples": 534210,
  "wall_time_seconds": 89.3,
  "episodes_per_second": 560.0,
  "dataset_size_bytes": 12345678,
  "action_distribution": {"0": 178070, "1": 178070, "2": 178070},
  "action_balance_ratio": 0.98
}
```

---

### 2.4 `train_tree.py` — Destilar a Decision Tree

**Comando**:
```powershell
python -m gym.scripts.decision_tree.train_tree `
  --dataset "gym/data/datasets/raw/bc__teacher_..." `
  --max-depth 10 `
  --seed 123
```

**Parámetros**:

- `--dataset` — Carpeta del dataset (contiene `samples.npz`) **(requerido)**
- `--out` — Nombre del modelo de salida (default: generado automáticamente)
- `--seed` — Semilla (default: `123`)
- `--test-size` — Fracción de test (default: `0.2`)
- `--max-depth` — Profundidad máxima del árbol (default: `10`)
- `--min-samples-leaf` — Muestras mínimas por hoja (default: `5`)
- `--criterion` — Criterio de división: `gini`, `entropy` o `log_loss` (default: `gini`)

**Archivos generados** (en `gym/data/models/sklearn/`):

- `dt__*.joblib` — Modelo scikit-learn serializado
- `dt__*.meta.json` — Metadata + métricas

**Métricas en meta.json**:
```json
{
  "wall_time_seconds": 0.45,
  "model_size_bytes": 12345,
  "tree_depth": 8,
  "n_leaves": 47,
  "feature_importances": [0.05, 0.12, "...21 valores..."],
  "metrics": {
    "accuracy": 0.934,
    "confusion_matrix": [[120, 5, 3], [8, 110, 7], [2, 6, 118]],
    "classification_report": {"0": {"precision": 0.92}, "...": "..."}
  }
}
```

---

### 2.5 `train_kmeans.py` — Entrenar KMeans

**Comando**:
```powershell
python -m gym.scripts.kmeans.train_kmeans `
  --opponents heuristic:denial heuristic:spread heuristic:greedy heuristic:random `
  --episodes-per-style 5000 `
  --window 6 `
  --seed 123
```

**Parámetros**:

- `--opponents` — Lista de specs de oponentes **(requerido, acepta múltiples valores)**
- `--episodes-per-style` — Episodios por cada estilo (default: `5000`)
- `--window` — Ventana temporal del perfil (default: `6`)
- `--k` — Número de clusters, 0 = len(opponents) (default: `0`)
- `--seed` — Semilla (default: `123`)

**Archivos generados** (en `gym/data/models/kmeans/`):

- `kmeans__v2__*.joblib` — Artefacto (scaler + kmeans + mappings)
- `kmeans__v2__*.meta.json` — Metadata + métricas

**Métricas en meta.json**:
```json
{
  "k": 4,
  "wall_time_seconds": 456.7,
  "inertia": 12345.6,
  "silhouette_score": 0.6234,
  "artifact_size_bytes": 89012,
  "cluster_to_style": {"0": "heuristic__denial", "1": "heuristic__spread"},
  "cluster_purity": {"0": {"counts_per_style": [4800, 120, 50, 30], "total": 5000}}
}
```

---

### 2.6 `build_response_table.py` — Response Table por Estilo

**Comando**:
```powershell
python -m gym.scripts.kmeans.build_response_table `
  --kmeans gym/data/models/kmeans/kmeans__v2__k4__win6__seed123.joblib `
  --candidates "dt:gym/data/models/sklearn/dt__ppo_vs_denial.joblib" `
               "dt:gym/data/models/sklearn/dt__ppo_vs_spread.joblib" `
               "dt:gym/data/models/sklearn/dt__ppo_vs_greedy.joblib" `
  --games-per-cluster 3000 `
  --swap-roles `
  --out gym/data/models/systems/response_table__system_dt.json
```

**Parámetros**:

- `--kmeans` — Path al artefacto KMeans v2 **(requerido)**
- `--candidates` — Lista de specs candidatos **(requerido, acepta múltiples valores)**
- `--games-per-cluster` — Juegos por cluster (default: `3000`)
- `--metric` — Métrica de selección: `winrate_decided` o `score_diff` (default: `winrate_decided`)
- `--swap-roles` — Flag: evaluar como p0 y como p1 para reducir sesgo
- `--seed` — Semilla (default: `123`)
- `--out` — Path del JSON de salida **(requerido)**

**Métricas en JSON de salida**:
```json
{
  "wall_time_seconds": 890.1,
  "total_games_evaluated": 36000,
  "cluster_to_policy_spec": {"0": "dt:...", "1": "dt:...", "2": "dt:..."}
}
```

---

## 3. Scripts de evaluación

### 3.1 `evaluate_any.py` — Evaluar cualquier matchup

**Comando**:
```powershell
python -m gym.scripts.utils.evaluate_any `
  --p0 "heuristic:greedy" `
  --p1 "heuristic:denial" `
  --games 5000 `
  --seed 123
```

**Parámetros**:

- `--p0` — Spec del jugador 0 **(requerido)**
- `--p1` — Spec del jugador 1 **(requerido)**
- `--games` — Número de partidas (default: `2000`)
- `--seed` — Semilla (default: `123`)
- `--algo-tag` — Etiqueta descriptiva para la corrida (default: `evaluate_any`)

**Archivos generados** (en `gym/data/results/runs/<stamp>/`):

- `config.json` — Configuración de la corrida
- `turns.jsonl` — Registro turno a turno (incluye `latency_ms`)
- `games.jsonl` — Registro partida a partida
- `summary.json` — Resumen con todas las métricas

**Métricas en summary.json**:
```json
{
  "games": 5000,
  "wins_p0": 2847,
  "wins_p1": 2109,
  "ties": 44,
  "winrate_p0": 0.5694,
  "winrate_p0_decided": 0.5745,
  "winrate_p0_decided_se": 0.007,
  "avg_turns": 22.1,
  "avg_score_diff_p0_minus_p1": 4.23,
  "std_score_diff": 8.92,
  "score_diff_percentiles": {
    "p5": -18.2,
    "p25": -6.1,
    "p50": 0.8,
    "p75": 9.4,
    "p95": 22.3
  },
  "wall_time_seconds": 12.3,
  "avg_wall_time_per_game": 0.0025
}
```

---

## 4. Script de análisis

### 4.1 `analysis_toolkit.py` — Gráficos y tablas

**Comando**:
```powershell
python -m gym.scripts.utils.analysis_toolkit
```

**Lee de**:

- `gym/data/models/rl/*.training_log.jsonl` — generado por train_sb3.py
- `gym/data/models/neat/*.generation_log.jsonl` — generado por train_neat.py
- `gym/data/models/*/*.meta.json` — generado por todos los scripts de entrenamiento
- `gym/data/results/runs/*/summary.json` — generado por evaluate_any.py

**Genera** (en `gym/data/results/`):

- `plots/ppo_training_*.png` — Curva de aprendizaje PPO
- `plots/neat_evolution_*.png` — Curva evolutiva NEAT
- `plots/training_time_comparison.png` — Barras: tiempo de entrenamiento por modelo
- `plots/model_size_comparison.png` — Barras: tamaño en disco por modelo
- `plots/evaluation_winrates.png` — Barras: winrate por evaluación (con error estándar)
- `plots/score_diff_boxplot.png` — Box plot: distribución del score diff
- `plots/dt_feature_importances_*.png` — Importancia de cada feature del DT
- `tables/models_summary.md` — Tabla Markdown con todos los modelos
- `tables/evaluations_summary.md` — Tabla Markdown con todas las evaluaciones

---

## 5. Scripts utilitarios

### 5.1 `test_env_sanity.py` — Verificar reglas del entorno

```powershell
python -m gym.scripts.utils.test_env_sanity
```

### 5.2 `artifacts_doctor.py` — Diagnosticar artefactos faltantes

```powershell
python -m gym.scripts.utils.artifacts_doctor
```

### 5.3 `cross_evaluate.py` — Evaluación cruzada (generalización vs overfitting)

**Comando**:
```powershell
python -m gym.scripts.utils.cross_evaluate --games 2000 --seed 123
```

Evalúa cada especialista (NEAT/PPO/DT × 3 oponentes = 9 especialistas) contra **todos** los oponentes (3), generando una tabla 9×3 de winrates. Calcula el Δ (delta) entre rendimiento contra el oponente de entrenamiento vs oponentes no vistos para detectar overfitting.

**Parámetros**:

- `--games` — Partidas por matchup (default: `2000`)
- `--seed` — Semilla (default: `123`)

**Genera** (en `gym/data/results/reports/`):

- `cross_evaluation.json` — Resultados completos en JSON
- `cross_evaluation.md` — Tabla Markdown con análisis de generalización

---

## 6. Gramática de specs (policy_factory)

Todos los scripts que reciben `--opponent`, `--teacher`, `--p0`, `--p1` o `--candidates`
usan esta gramática para identificar políticas:

- `baseline:first` — Siempre elige primera columna disponible
- `baseline:random` — Elige columna aleatoria
- `heuristic:greedy` — Maximiza puntaje inmediato
- `heuristic:denial` — Prioriza destruir columnas del rival
- `heuristic:spread` — Distribuye fichas entre columnas
- `heuristic:random` — Acciones completamente aleatorias
- `dt:<path.joblib>` — Decision Tree desde archivo
- `rl:PPO:<path.zip>` — Modelo PPO desde archivo
- `neat:<genome.pkl>:<config.ini>` — NEAT desde genoma + config
- `system:<config.json>` — Sistema adaptativo (KMeans + especialistas)

---

## 7. Estructura de datos generados

```
gym/data/
├── datasets/raw/
│   └── bc__teacher_*__vs_*/
│       ├── samples.npz
│       ├── preview.csv
│       └── meta.json
│
├── models/
│   ├── rl/
│   │   ├── PPO__vs_*.zip
│   │   ├── PPO__vs_*.meta.json
│   │   └── PPO__vs_*.training_log.jsonl
│   ├── neat/
│   │   ├── neat__vs_*.pkl
│   │   ├── neat__vs_*.meta.json
│   │   └── neat__vs_*.generation_log.jsonl
│   ├── sklearn/
│   │   ├── dt__*.joblib
│   │   └── dt__*.meta.json
│   ├── kmeans/
│   │   ├── kmeans__*.joblib
│   │   └── kmeans__*.meta.json
│   └── systems/
│       └── response_table__*.json
│
└── results/
    ├── runs/<stamp>__<tag>__<p0>_vs_<p1>/
    │   ├── config.json
    │   ├── turns.jsonl
    │   ├── games.jsonl
    │   └── summary.json
    ├── plots/*.png
    └── tables/*.md
```

---

## 8. Los 3 sistemas adaptativos

Cada **sistema** es la combinación de:

```
KMeans (perfilador, compartido) + Response Table (por sistema) + Especialistas (por tipo de IA)
```

| Sistema | Especialistas | Response Table |
|---------|--------------|----------------|
| system\_dt | 3 Decision Trees destilados de PPO | response\_table\_\_system\_dt.json |
| system\_ppo | 3 modelos PPO directos | response\_table\_\_system\_ppo.json |
| system\_neat | 3 genomas NEAT | response\_table\_\_system\_neat.json |

El KMeans detecta el estilo del oponente → consulta la response table → elige el especialista más fuerte para ese estilo.

---

## 9. Pipeline completo — Comandos en orden

> Copiar y pegar directamente en PowerShell de Windows 11.
> Cada línea es independiente — se puede ejecutar una por una.

```powershell
# ═══ PASO 1: Entrenar PPO especialistas (×3) ═══
python -m gym.scripts.rl.train_sb3 --algo PPO --timesteps 400000 --opponent heuristic:denial --n-envs 4 --seed 123
python -m gym.scripts.rl.train_sb3 --algo PPO --timesteps 400000 --opponent heuristic:spread --n-envs 4 --seed 123
python -m gym.scripts.rl.train_sb3 --algo PPO --timesteps 400000 --opponent heuristic:greedy --n-envs 4 --seed 123

# ═══ PASO 2: Entrenar NEAT especialistas (×3, multi-semilla real) ═══
python -m gym.scripts.neat.train_neat --opponent heuristic:denial --generations 300 --episodes-per-genome 34 --seeds "123,456,789"
python -m gym.scripts.neat.train_neat --opponent heuristic:spread --generations 300 --episodes-per-genome 34 --seeds "123,456,789"
python -m gym.scripts.neat.train_neat --opponent heuristic:greedy --generations 300 --episodes-per-genome 34 --seeds "123,456,789"

# ═══ PASO 3: Generar datasets BC — PPO como teacher (×3) ═══
python -m gym.scripts.decision_tree.generate_dataset_teacher --teacher "rl:PPO:gym/data/models/rl/PPO__vs_heuristic_denial.zip" --opponent heuristic:denial --episodes 50000
python -m gym.scripts.decision_tree.generate_dataset_teacher --teacher "rl:PPO:gym/data/models/rl/PPO__vs_heuristic_spread.zip" --opponent heuristic:spread --episodes 50000
python -m gym.scripts.decision_tree.generate_dataset_teacher --teacher "rl:PPO:gym/data/models/rl/PPO__vs_heuristic_greedy.zip" --opponent heuristic:greedy --episodes 50000

# ═══ PASO 4: Destilar a Decision Trees (×3) ═══
# NOTA: reemplazar <carpeta_*> con el nombre real generado en el paso 3
python -m gym.scripts.decision_tree.train_tree --dataset "gym/data/datasets/raw/<carpeta_denial>" --out dt__ppo_vs_denial.joblib
python -m gym.scripts.decision_tree.train_tree --dataset "gym/data/datasets/raw/<carpeta_spread>" --out dt__ppo_vs_spread.joblib
python -m gym.scripts.decision_tree.train_tree --dataset "gym/data/datasets/raw/<carpeta_greedy>" --out dt__ppo_vs_greedy.joblib

# ═══ PASO 5: Entrenar KMeans (perfilador compartido) ═══
python -m gym.scripts.kmeans.train_kmeans --opponents heuristic:denial heuristic:spread heuristic:greedy heuristic:random --seed 123

# ═══ PASO 6a: Response Table — Sistema DT ═══
python -m gym.scripts.kmeans.build_response_table `
  --kmeans gym/data/models/kmeans/kmeans__v2__k4__win6__seed123.joblib `
  --candidates "dt:gym/data/models/sklearn/dt__ppo_vs_denial.joblib" `
               "dt:gym/data/models/sklearn/dt__ppo_vs_spread.joblib" `
               "dt:gym/data/models/sklearn/dt__ppo_vs_greedy.joblib" `
  --games-per-cluster 3000 --swap-roles --seed 123 `
  --out gym/data/models/systems/response_table__system_dt.json

# ═══ PASO 6b: Response Table — Sistema PPO ═══
python -m gym.scripts.kmeans.build_response_table `
  --kmeans gym/data/models/kmeans/kmeans__v2__k4__win6__seed123.joblib `
  --candidates "rl:PPO:gym/data/models/rl/PPO__vs_heuristic_denial.zip" `
               "rl:PPO:gym/data/models/rl/PPO__vs_heuristic_spread.zip" `
               "rl:PPO:gym/data/models/rl/PPO__vs_heuristic_greedy.zip" `
  --games-per-cluster 3000 --swap-roles --seed 123 `
  --out gym/data/models/systems/response_table__system_ppo.json

# ═══ PASO 6c: Response Table — Sistema NEAT ═══
python -m gym.scripts.kmeans.build_response_table `
  --kmeans gym/data/models/kmeans/kmeans__v2__k4__win6__seed123.joblib `
  --candidates "neat:gym/data/models/neat/neat__vs_heuristic_denial.pkl:gym/config/neat/neat_config.ini" `
               "neat:gym/data/models/neat/neat__vs_heuristic_spread.pkl:gym/config/neat/neat_config.ini" `
               "neat:gym/data/models/neat/neat__vs_heuristic_greedy.pkl:gym/config/neat/neat_config.ini" `
  --games-per-cluster 3000 --swap-roles --seed 123 `
  --out gym/data/models/systems/response_table__system_neat.json

# ═══ PASO 7: Evaluar los 3 sistemas vs los 3 heurísticos (9 evaluaciones) ═══
# --- System DT ---
python -m gym.scripts.utils.evaluate_any --p0 "system:gym/config/systems/system_dt.json" --p1 heuristic:denial --games 5000 --seed 123
python -m gym.scripts.utils.evaluate_any --p0 "system:gym/config/systems/system_dt.json" --p1 heuristic:spread --games 5000 --seed 123
python -m gym.scripts.utils.evaluate_any --p0 "system:gym/config/systems/system_dt.json" --p1 heuristic:greedy --games 5000 --seed 123

# --- System PPO ---
python -m gym.scripts.utils.evaluate_any --p0 "system:gym/config/systems/system_ppo.json" --p1 heuristic:denial --games 5000 --seed 123
python -m gym.scripts.utils.evaluate_any --p0 "system:gym/config/systems/system_ppo.json" --p1 heuristic:spread --games 5000 --seed 123
python -m gym.scripts.utils.evaluate_any --p0 "system:gym/config/systems/system_ppo.json" --p1 heuristic:greedy --games 5000 --seed 123

# --- System NEAT ---
python -m gym.scripts.utils.evaluate_any --p0 "system:gym/config/systems/system_neat.json" --p1 heuristic:denial --games 5000 --seed 123
python -m gym.scripts.utils.evaluate_any --p0 "system:gym/config/systems/system_neat.json" --p1 heuristic:spread --games 5000 --seed 123
python -m gym.scripts.utils.evaluate_any --p0 "system:gym/config/systems/system_neat.json" --p1 heuristic:greedy --games 5000 --seed 123

# ═══ PASO 8: Generar gráficos y tablas ═══
python -m gym.scripts.utils.analysis_toolkit

# ═══ PASO 9: Evaluación cruzada (generalización vs overfitting) ═══
python -m gym.scripts.utils.cross_evaluate --games 2000 --seed 123
```

---

## 10. Nota sobre PowerShell en Windows 11

En PowerShell, el carácter de continuación de línea es el **backtick** (`` ` ``), no la barra invertida `\` de bash/Linux.

```powershell
# ✗ Esto NO funciona en PowerShell (sintaxis bash)
python -m script \
  --arg valor

# ✓ Esto SÍ funciona en PowerShell
python -m script `
  --arg valor

# ✓ Esto también funciona (todo en una línea)
python -m script --arg valor
```

El backtick debe ir **al final de la línea**, sin espacios después de él.
