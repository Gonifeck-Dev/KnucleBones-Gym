# Documento 2 — Arquitectura Técnica y Pipeline

---

## 1. Principios rectores

### 1.1 Fuente de verdad
`KnucklebonesEnv` (`gym/env/knucklebones_env.py`) es la **única fuente de verdad** de reglas del juego. Cualquier entrenamiento/evaluación sigue el mismo protocolo de turno:
```python
die = env.roll_die()
obs = env._get_obs(dice_value=die)
action = policy.select_action(obs, legal_actions)
env.step(action, dice_value=die)
```

### 1.2 Reproducibilidad
Semillas (`--seed`) para: reset del env, reset de políticas, generación de episodios, comparaciones justas. Resultados registrados en: `summary.json`, `config.json`, `games.jsonl`, `turns.jsonl`, `*.meta.json`.

### 1.3 Gramática de specs
Fábrica única `policy_factory.build_policy(spec)`:
```
baseline:first | baseline:random
heuristic:greedy | heuristic:denial | heuristic:spread | heuristic:random
dt:<path.joblib>
rl:PPO:<path.zip>
neat:<genome.pkl>:<config.ini>
system:<config.json>
```

---

## 2. Pipeline completo

```
Arquetipos heurísticos (greedy/denial/spread/random)
        ↓ entrenar contra cada uno
PPO especialistas (400k timesteps c/u)
        ↓ behavior cloning (50k episodios)
DT destilados (accuracy ~0.93-0.95)
        ↓ perfilar oponentes
KMeans v2 (k=4, window=6, ~486k samples)
        ↓ evaluar mejor respuesta por cluster
Response Table (cluster → style → mejor DT)
        ↓ orquestar
SystemPolicy = KMeans + DT especialistas
```

### 2.1 Paso 1 — Arquetipos heurísticos
Estilos controlados que sirven como "ground truth":
- **Greedy**: maximiza puntaje inmediato
- **Denial**: prioriza destruir columnas del rival
- **Spread**: distribución segura entre columnas
- **Random**: acciones aleatorias uniformes

### 2.2 Paso 2 — PPO especialistas
Entrenamiento RL offline contra cada arquetipo:
```bash
python -m gym.scripts.rl.train_sb3 --algo PPO --timesteps 400000 --opponent heuristic:denial
python -m gym.scripts.rl.train_sb3 --algo PPO --timesteps 400000 --opponent heuristic:spread
python -m gym.scripts.rl.train_sb3 --algo PPO --timesteps 400000 --opponent heuristic:greedy
```

### 2.3 Paso 3 — Destilación PPO → DT
Behavior cloning: el PPO juega como teacher, el DT aprende por imitación:
```bash
python -m gym.scripts.decision_tree.generate_dataset_teacher \
  --teacher rl:PPO:gym/data/models/rl/PPO__vs_denial.zip \
  --opponent heuristic:denial --episodes 50000
python -m gym.scripts.decision_tree.train_tree --dataset <carpeta>
```

### 2.4 Paso 4 — KMeans
Perfilado online con ventana temporal:
```bash
python -m gym.scripts.kmeans.train_kmeans \
  --opponents heuristic:denial heuristic:spread heuristic:greedy heuristic:random
```

### 2.5 Paso 5 — Response Table
Mapeo cluster → mejor especialista, evaluando por estilo:
```bash
python -m gym.scripts.kmeans.build_response_table \
  --kmeans gym/data/models/kmeans/kmeans__v2__archetypes.joblib \
  --candidates dt:...denial.joblib dt:...spread.joblib dt:...greedy.joblib \
  --swap-roles --out gym/data/models/systems/response_table__system_dt.json
```

### 2.6 Paso 6 — NEAT especialistas (para Sistema NEAT comparativo)

Entrenamiento con multi-seed real (3 seeds simultáneas en fitness, 300 generaciones):
```bash
python -m gym.scripts.neat.train_neat --opponent heuristic:denial --generations 300 --episodes-per-genome 34 --seeds "123,456,789"
python -m gym.scripts.neat.train_neat --opponent heuristic:spread --generations 300 --episodes-per-genome 34 --seeds "123,456,789"
python -m gym.scripts.neat.train_neat --opponent heuristic:greedy --generations 300 --episodes-per-genome 34 --seeds "123,456,789"
```

> **Multi-seed real**: Cada genoma se evalúa contra 3 seeds simultáneamente (34 × 3 = 102 episodios), promediando fitness. Esto previene sobreajuste a una secuencia determinista y fue la técnica que permitió a NEAT alcanzar WR = 0.653 (vs 0.499 con seed fija).

---

## 3. Los 3 Sistemas

Cada sistema combina KMeans + especialistas de una familia:

| Sistema | Especialistas | Inferencia | Costo |
|---------|--------------|------------|-------|
| **Sistema_DT** | DT destilados de PPO | ~μs | Muy bajo |
| **Sistema_PPO** | PPO especialistas directos | ~ms | Medio |
| **Sistema_NEAT** | NEAT especialistas directos | ~μs-ms | Bajo-Medio |

Todos comparten el mismo KMeans y la misma lógica de gating (SystemPolicy). Solo difieren en los especialistas y la response table.

---

## 4. Estructura del proyecto

```
gym/
├── config/
│   ├── neat/neat_config.ini
│   └── systems/
│       ├── system_dt.json
│       ├── system_ppo.json
│       └── system_neat.json
├── data/
│   ├── datasets/raw/           # BC datasets
│   ├── models/
│   │   ├── kmeans/             # KMeans artefacto
│   │   ├── neat/               # Genomas NEAT
│   │   ├── rl/                 # PPO modelos
│   │   ├── sklearn/            # DT destilados
│   │   └── systems/            # Response tables
│   └── results/runs/           # Corridas experimentales
├── env/
│   ├── knucklebones_env.py     # Entorno (fuente de verdad)
│   └── knucklebones_sb3_env.py # Wrapper SB3
├── policies/                   # Solo inferencia (Regla A)
│   ├── archetypes/             # greedy, denial, spread, random
│   ├── decision_tree/          # baseline + sklearn DT
│   ├── kmeans/                 # Adapter (perfilado + selección)
│   ├── neat/                   # NEAT policy
│   ├── rl/                     # SB3 policy
│   ├── systems/                # SystemPolicy
│   └── utils/                  # base_policy, features, policy_factory
└── scripts/                    # Entrenamiento y evaluación (Regla B)
    ├── decision_tree/          # generate_dataset_teacher, train_tree
    ├── kmeans/                 # train_kmeans, build_response_table
    ├── neat/                   # train_neat
    ├── rl/                     # train_sb3
    └── utils/                  # evaluate_any, run_manager, naming, etc.
```

### 4.1 Reglas de oro

| Regla | Aplica en | Puede | No puede |
|-------|-----------|-------|----------|
| **A** | `policies/` | Inferir acciones | Entrenar, escribir archivos |
| **B** | `scripts/` | Entrenar, evaluar, generar datos | Duplicar lógica de features |
| **C** | `data/` | Almacenar todo lo generado | Contener código |

---

## 5. Convenciones de artefactos

Nombres semánticos, metadata en `.meta.json`:
```
gym/data/models/rl/PPO__vs_denial.zip           + .meta.json
gym/data/models/sklearn/dt__ppo_vs_denial.joblib + .meta.json
gym/data/models/neat/neat__vs_denial.pkl         + .meta.json
gym/data/models/kmeans/kmeans__v2__archetypes.joblib + .meta.json
```

---

## 6. Vector de features (21 dimensiones)

```python
# gym/policies/utils/features.py
FEATURES_VERSION = 1
FEATURES_DIM = 21

[dice, *flatten(my_cols), *flatten(op_cols), score_my, score_op]
# 1 + 9 + 9 + 2 = 21
```

Todos los modelos (DT, RL, NEAT) usan `extract_features()` para garantizar consistencia.

---

## 8. Dependencias

```
numpy>=1.24
scikit-learn>=1.3
joblib>=1.3
neat-python>=0.92
stable-baselines3>=2.1
gymnasium>=0.29
```

