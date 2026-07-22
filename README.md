# Knucklebones AI Gym

> **Tesis**: *Rendimiento y desempeño del aprendizaje de distintas inteligencias artificiales en un entorno de videojuego offline*
> **Estado**: Entrenamiento completado, marzo 2026

Plataforma de entrenamiento y evaluación de IAs adaptativas para el juego de mesa **Knucklebones** (Cult of the Lamb). Ejecutado 100% offline en hardware local (sin GPU dedicada de datacenter).

**Sobre Knucklebones no existe tratamiento académico publicado**: solo proyectos aislados de MCTS/Q-Learning en GitHub. Este proyecto compara tres paradigmas de RL (PPO, NEAT, destilación a árboles), cuantifica por qué el aprendizaje online pleno *no* es viable con datos de un solo humano, y propone y valida una alternativa: adaptación online por selección de especialistas. El valor no está en cada pieza por separado, todas existen en la literatura, sino en la **integración cuantificada de las tres, en hardware de consumo, replicada end-to-end**.

## Resultados Finales

| Sistema | Winrate | IC 95% | Significancia |
|---------|:-------:|:------:|:-------------:|
| **System NEAT** | **0.653** | [0.645, 0.660] | Superior (p < 0.003) |
| System DT | 0.608 | [0.600, 0.616] | Equivalente a PPO |
| System PPO | 0.604 | [0.597, 0.612] | Equivalente a DT |

NEAT con multi-seed real (3 seeds × 300 gen) es significativamente superior a PPO y DT destilado. DT es la mejor opción práctica si se prioriza interpretabilidad.

## Hipótesis

El *aprendizaje online pleno* (entrenar mientras se juega) es costoso e inestable en hardware local. Sin embargo, la **adaptación online por selección** (player modeling + gating) puede entregar adaptabilidad real con menor costo, combinando:

1. Agentes fuertes entrenados offline (**PPO**, **NEAT**)
2. Destilación a modelos livianos e interpretables (**DT**)
3. Perfilado del oponente con **KMeans** (ventana temporal)
4. Selección del mejor especialista por cluster (**Response Table**)

## Related Work: posicionamiento frente a la literatura

**Sobre Knucklebones específicamente**: sin tratamiento académico previo (confianza baja: ausencia en una búsqueda dedicada, no prueba de inexistencia). La integración tri-paradigma, la cuantificación de la brecha de datos y la adaptación por selección, todo en hardware local sin GPU, no aparece replicada en ninguna fuente encontrada.

**PPO vs NEAT.** La literatura comparativa coincide con el patrón observado acá: la neuroevolución converge más rápido al inicio; RL con gradientes gana en rendimiento final con suficiente entrenamiento (caso self-driving NPC[^1]; Flappy Bird, NEAT vs DQN[^2]). Que NEAT multi-seed gane acá (WR 0.653) en un dominio pequeño con presupuesto acotado es consistente: con cómputo limitado, la ventaja inicial de la neuroevolución domina.

**Destilación PPO → árboles de decisión.** Línea fundacional: VIPER[^3], árboles destilados desde DNN+Q-function que retienen el rendimiento del teacher y habilitan verificación formal. Continuaciones: MAVIPER, destilación multi-agente[^4]; árboles programáticos editables[^5]; optimización directa de árboles[^6]. Este proyecto replica el hallazgo central (fidelidad >93%, WR del alumno ≈ teacher) con behavior cloning directo, más simple que VIPER.

**Perfilado de oponente + selección de especialistas.** Patrón ancestral: Missura & Gärtner[^7], clustering de tipos de jugador y predicción desde trazas cortas para ajustar dificultad, junto con el survey de Bakkes/Spronck[^8]. Precedente directo del "switching": controladores rule-based conmutables en fighting games (Ishihara et al.). Estado del arte 2026: StratFormer, mismo patrón en versión continua (GTO → best-response)[^9].

**Viabilidad del aprendizaje online, el hallazgo central.** El contrapunto obligado: rtNEAT / NERO[^10] demostró aprendizaje real durante el gameplay, pero con experiencia generada por muchos agentes simultáneos, en un juego diseñado alrededor del entrenamiento. No hay contradicción con este proyecto, hay una frontera: NERO funciona cuando el diseño multiplica la generación de datos; acá, con datos de un solo humano (180-1,800 por sesión vs. 400K-4.6M requeridas), el aprendizaje online pleno no alcanza, y la selección de especialistas es la salida viable. Validación industrial del mismo patrón offline-first: EA entrena offline con datos pre-recolectados y despliega política congelada en producción para EA FC 25[^11].

**Trabajo futuro (meta-learning).** Cita canónica: MAML[^12], adaptación rápida con pocas muestras vía meta-entrenamiento. La línea meta-RL es la vía publicada para cerrar la brecha de datos que este proyecto mide.

[^1]: Comparación de comportamiento de NPCs self-driving (NEAT vs RL con gradientes). Springer, 2021.
[^2]: NEAT vs DQN en Flappy Bird. [arXiv:2207.14140](https://arxiv.org/abs/2207.14140)
[^3]: Bastani, Pu & Solar-Lezama. *Verifiable Reinforcement Learning via Policy Extraction* (VIPER). NeurIPS 2018. [arXiv:1805.08328](https://arxiv.org/abs/1805.08328)
[^4]: MAVIPER: destilación multi-agente. [arXiv:2205.12449](https://arxiv.org/abs/2205.12449)
[^5]: Árboles programáticos editables por búsqueda local. [arXiv:2405.14956](https://arxiv.org/abs/2405.14956)
[^6]: Optimización directa de árboles de decisión. [arXiv:2408.11632](https://arxiv.org/abs/2408.11632)
[^7]: Missura & Gärtner. Clustering de tipos de jugador para ajuste de dificultad. 2009.
[^8]: Bakkes & Spronck. Survey de player modeling y dynamic difficulty adjustment. 2012.
[^9]: StratFormer: opponent modeling adaptativo (GTO → best-response). 2026.
[^10]: Stanley et al. rtNEAT / NERO: aprendizaje en tiempo real durante el gameplay. 2005.
[^11]: Sestini et al. / EA SEED. RL sample-efficient en EA FC 25. Oct 2025.
[^12]: Finn, Abbeel & Levine. *Model-Agnostic Meta-Learning* (MAML). 2017. [arXiv:1703.03400](https://arxiv.org/abs/1703.03400)

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

## Reglas de oro

| Regla | Aplica en | ✅ Puede | ❌ No puede |
|-------|-----------|----------|-------------|
| **A** | `policies/` | Inferir acciones, estado por partida | Entrenar, escribir archivos |
| **B** | `scripts/` | Entrenar, evaluar, generar datos | Duplicar lógica de features/heurísticas |
| **C** | `data/` | Almacenar todo lo generado | Contener código |

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
| `docs/05_handoff_unity.md` | Especificación técnica para integración con Unity |

## Dependencias

```bash
pip install -r requirements.txt
```

<details>
<summary><strong>Mecánica de turnos y reproducibilidad</strong> (click para expandir)</summary>

Reglas completas en `docs/05_handoff_unity.md` (§2). Resumen para poder replicar el experimento:

1. El jugador activo **tira un dado** (1-6).
2. Elige una **columna** (0, 1 o 2) con al menos una fila vacía.
3. El dado se coloca en la primera fila vacía de esa columna (de abajo hacia arriba).
4. **Regla de destrucción**: todos los dados del oponente con el mismo valor en la misma columna se eliminan.
5. Se recalculan puntajes: `Score(columna) = Σ v × (cantidad de v en la columna)²`.
6. La partida termina cuando algún jugador completa su tablero (9 celdas); gana el mayor puntaje total.

`KnucklebonesEnv` (`gym/env/knucklebones_env.py`) es la única fuente de verdad de estas reglas: todo entrenamiento/evaluación sigue el mismo protocolo (`roll_die → _get_obs → select_action → step`).

**Registro turno a turno**: cada corrida de `evaluate_any.py` genera, en `gym/data/results/runs/<stamp>/`:

| Archivo | Contenido |
|---------|-----------|
| `config.json` | Configuración de la corrida |
| `turns.jsonl` | Registro turno a turno (incluye `latency_ms`), el log grande que permite auditar cada decisión |
| `games.jsonl` | Registro partida a partida |
| `summary.json` | Resumen con todas las métricas |

**Reproducibilidad**: todas las corridas usan `--seed` para fijar reset del entorno, reset de políticas y generación de episodios, de modo que las comparaciones entre sistemas sean justas y el experimento sea replicable con los mismos artefactos (`*.meta.json`) documentados en `docs/04_referencia_scripts.md`.

</details>

<details>
<summary><strong>Estructura del proyecto</strong> (click para expandir)</summary>

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
│       ├── features.py                   # extract_features(), 21 dims
│       └── policy_factory.py             # build_policy(spec), fábrica única
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

</details>

<details>
<summary><strong>Policy specs</strong> (fábrica única, click para expandir)</summary>

```
baseline:first | baseline:random
heuristic:greedy | heuristic:denial | heuristic:spread | heuristic:random
dt:<path.joblib>
rl:PPO:<path.zip>
neat:<genome.pkl>:<config.ini>
system:<config.json>
```

</details>
