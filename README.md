# Knucklebones AI Gym

> **Tesis**: *Rendimiento y desempeño del aprendizaje de distintas inteligencias artificiales en un entorno de videojuego offline*

Plataforma de entrenamiento y evaluación de IAs adaptativas para el juego de mesa **Knucklebones** (Cult of the Lamb). Ejecutado 100% offline en hardware local.

## Hipótesis

El *aprendizaje online pleno* (entrenar mientras se juega) es costoso e inestable en hardware local. Sin embargo, la **adaptación online por selección** (player modeling + gating) puede entregar adaptabilidad real con menor costo, combinando:

1. Agentes fuertes entrenados offline (**PPO**)
2. Destilación a modelos livianos e interpretables (**DT**)
3. Perfilado del oponente con **KMeans** (ventana temporal)
4. Selección del mejor especialista por cluster (**Response Table**)

## Pipeline

```
Arquetipos heurísticos (greedy/denial/spread/random)
        ↓ entrenar contra cada uno
PPO especialistas (PPO vs denial, vs spread, vs greedy)
        ↓ behavior cloning
DT destilados (dt__ppo_vs_denial, dt__ppo_vs_spread, dt__ppo_vs_greedy)
        ↓ perfilar oponentes
KMeans v2 (clustering de estilos online)
        ↓ evaluar mejor respuesta por cluster
Response Table (cluster → mejor DT)
        ↓ orquestar
SystemPolicy = KMeans + DT especialistas = adaptación online sin entrenamiento
```

## Estructura

```
gym/
├── config/
│   ├── neat/neat_config.ini              # Config NEAT (solo inferencia comparativa)
│   └── systems/                          # Configs JSON de sistemas
│       ├── system_dt.json
│       ├── system_ppo.json
│       └── system_neat.json
├── data/
│   ├── datasets/raw/                     # Datasets de behavior cloning
│   ├── models/
│   │   ├── kmeans/                       # KMeans + response tables
│   │   ├── neat/                         # Genomas NEAT (.pkl)
│   │   ├── rl/                           # PPO especialistas (.zip + .meta.json)
│   │   ├── sklearn/                      # DT destilados (.joblib + .meta.json)
│   │   └── systems/                      # Response tables por sistema
│   └── results/runs/                     # Corridas auto-generadas
├── env/
│   ├── knucklebones_env.py               # Entorno (fuente de verdad de reglas)
│   └── knucklebones_sb3_env.py           # Wrapper Gymnasium para SB3
├── policies/                             # Solo inferencia (Regla A)
│   ├── archetypes/                       # greedy, denial, spread, random
│   ├── decision_tree/                    # baseline + sklearn DT
│   ├── kmeans/                           # KMeans adapter (perfilado + selección)
│   ├── neat/                             # NEAT (solo inferencia comparativa)
│   ├── rl/                               # SB3 wrapper (PPO/DQN)
│   ├── systems/                          # SystemPolicy (KMeans + especialistas)
│   └── utils/
│       ├── base_policy.py                # Contrato abstracto
│       ├── features.py                   # extract_features() — 21 dims
│       └── policy_factory.py             # build_policy(spec) — fábrica única
└── scripts/                              # Entrenamiento y evaluación (Regla B)
    ├── decision_tree/
    │   ├── generate_dataset_teacher.py   # BC dataset desde cualquier teacher
    │   └── train_tree.py                 # Entrenar DT sklearn
    ├── kmeans/
    │   ├── train_kmeans.py               # Entrenar KMeans con arquetipos
    │   └── build_response_table.py       # Response table por estilo
    ├── neat/
    │   └── train_neat.py                 # Entrenar NEAT especialista
    ├── rl/
    │   └── train_sb3.py                  # Entrenar PPO/DQN
    └── utils/
        ├── evaluate_any.py               # Evaluación universal p0 vs p1
        ├── run_manager.py                # Gestión de runs (Regla C)
        ├── naming.py                     # utc_stamp(), safe_name()
        ├── artifacts_doctor.py           # Validar/reparar artefactos
        └── test_env_sanity.py            # Test de sanidad del entorno
```

## Reglas de oro

| Regla | Aplica en | ✅ Puede | ❌ No puede |
|-------|-----------|----------|-------------|
| **A** | `policies/` | Inferir acciones, estado por partida | Entrenar, escribir archivos |
| **B** | `scripts/` | Entrenar, evaluar, generar datos | Duplicar lógica de features/heurísticas |
| **C** | `data/` | Almacenar todo lo generado | Contener código |

## Policy specs (fábrica única)

```
baseline:first | baseline:random
heuristic:greedy | heuristic:denial | heuristic:spread | heuristic:random
dt:<path.joblib>
rl:PPO:<path.zip>
neat:<genome.pkl>:<config.ini>
system:<config.json>
```

## Ejecución rápida

```bash
# 1. Validar entorno
python -m gym.scripts.utils.test_env_sanity

# 2. Entrenar PPO especialista
python -m gym.scripts.rl.train_sb3 --algo PPO --timesteps 400000 --opponent heuristic:denial

# 2b. Entrenar NEAT especialista
python -m gym.scripts.neat.train_neat --opponent heuristic:denial --generations 50

# 3. Generar dataset BC (destilación)
python -m gym.scripts.decision_tree.generate_dataset_teacher --teacher rl:PPO:gym/data/models/rl/PPO__vs_denial.zip --opponent heuristic:denial --episodes 50000

# 4. Entrenar DT destilado
python -m gym.scripts.decision_tree.train_tree --dataset gym/data/datasets/raw/<carpeta>

# 5. Entrenar KMeans
python -m gym.scripts.kmeans.train_kmeans --opponents heuristic:denial heuristic:spread heuristic:greedy heuristic:random

# 6. Construir response table
python -m gym.scripts.kmeans.build_response_table --kmeans gym/data/models/kmeans/kmeans__v2__archetypes.joblib --candidates dt:gym/data/models/sklearn/dt__ppo_vs_denial.joblib dt:gym/data/models/sklearn/dt__ppo_vs_spread.joblib dt:gym/data/models/sklearn/dt__ppo_vs_greedy.joblib --games-per-cluster 3000 --swap-roles --out gym/data/models/systems/response_table__system_dt.json

# 7. Evaluar sistema
python -m gym.scripts.utils.evaluate_any --p0 "system:gym/config/systems/system_dt.json" --p1 "heuristic:denial" --games 5000

# 8. Generar gráficos y tablas
python -m gym.scripts.utils.analysis_toolkit

# 9. Validar artefactos
python -m gym.scripts.utils.artifacts_doctor --fix
```

## Documentación

| Documento | Contenido |
|-----------|-----------|
| `docs/01_marco_teorico.md` | Problema, hipótesis, antecedentes, técnicas, glosario |
| `docs/02_arquitectura_pipeline.md` | Pipeline, estructura, reglas, features |
| `docs/03_resultados_experimentales.md` | Hallazgos, tabla de runs, interpretaciones |
| `docs/04_referencia_scripts.md` | Referencia de cada script: parámetros, salidas, métricas |

## Dependencias

```bash
pip install -r requirements.txt
```
