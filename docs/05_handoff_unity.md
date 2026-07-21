# Documento 5 — Handoff para Integración con Unity

> **Destinatario**: Agente/desarrollador encargado del videojuego en Unity.
> **Fecha**: 2026-03-06
> **Estado del entrenamiento**: COMPLETADO — Todos los modelos están entrenados y evaluados.

---

## 1. Contexto del proyecto

### 1.1 Objetivo de la tesis

**Título**: *Rendimiento y desempeño del aprendizaje de distintas inteligencias artificiales en un entorno de videojuego offline*

Se evalúa la factibilidad técnica de integrar y comparar enfoques de IA adaptativa en un videojuego ejecutado 100% en hardware local (sin nube). El juego es **Knucklebones** (del videojuego Cult of the Lamb): juego de mesa por turnos, 2 jugadores, tablero 3×3 cada uno, dados de 6 caras.

### 1.2 Qué se ha completado (lado Python/entrenamiento)

- **9 modelos especialistas** entrenados: 3 PPO, 3 NEAT, 3 DT destilados
- **1 perfilador KMeans** (k=4, ventana=6 turnos, 10 dimensiones)
- **3 sistemas adaptativos** (KMeans + Response Table + Especialistas)
- **Evaluación estadística completa** con z-tests y Bonferroni
- **Resultado**: NEAT es el mejor (WR=0.653), DT es práctico (WR=0.608), PPO equivalente a DT (WR=0.604)

### 1.3 Qué falta (lado Unity)

- Implementar el juego Knucklebones en Unity
- Implementar un **middleware local** que conecte Unity con los modelos Python
- Permitir que la IA juegue contra un humano o contra otra IA
- Registrar métricas durante gameplay real

---

## 2. Reglas del juego — Knucklebones

### 2.1 Estructura

- 2 jugadores, cada uno con un tablero de **3 columnas × 3 filas**
- Cada celda puede estar vacía (0) o contener un dado (1-6)
- Los jugadores alternan turnos

### 2.2 Flujo de un turno

1. El jugador activo **tira un dado** (valor aleatorio 1-6)
2. El jugador **elige una columna** (0, 1 o 2) que tenga al menos una fila vacía
3. El dado se coloca en la **primera fila vacía** de esa columna (de abajo hacia arriba)
4. **Regla de destrucción**: todos los dados del **oponente** con el mismo valor en la **misma columna** se eliminan (se ponen en 0 y las filas se compactan)
5. Se recalculan puntajes

### 2.3 Puntuación por columna

```
Score(columna) = Σ para cada valor v presente: v × (cantidad de v en la columna)²
```

**Ejemplos**:
- Columna `[3, 0, 0]` → 3 × 1² = 3
- Columna `[3, 3, 0]` → 3 × 2² = 12
- Columna `[3, 3, 3]` → 3 × 3² = 27
- Columna `[2, 3, 5]` → 2×1² + 3×1² + 5×1² = 10

**Puntaje total** = Score(col0) + Score(col1) + Score(col2)

### 2.4 Condición de fin

La partida termina cuando **cualquier** jugador tiene su tablero completo (9 celdas ocupadas). El ganador es quien tiene mayor puntaje total.

### 2.5 Formalización matemática

```
Estado: s = (B₁, B₂, d, t)
  B_p ∈ {0,1,2,3,4,5,6}^{3×3}  (tablero del jugador p)
  d ∈ {1,...,6}                  (dado actual)
  t = turno

Transición: T(s, a) = s'  donde a ∈ {0, 1, 2} (columna elegida)

Destrucción: B_oponente[i,c] = 0 si B_oponente[i,c] == d, para i ∈ {0,1,2}
```

**CRÍTICO**: El simulador Python y Unity DEBEN producir exactamente la misma transición para el mismo estado+acción. Se recomienda validar con golden logs (trazas de referencia).

---

## 3. Arquitectura del middleware

### 3.1 Diseño general

```
┌─────────────────┐     HTTP/Socket/Pipe     ┌─────────────────────┐
│                 │ ◄──── estado del turno ─── │                     │
│  Servidor Python│                           │   Unity (C#)         │
│  (Middleware)   │ ───── acción elegida ────► │                     │
│                 │                           │                     │
│  - Carga modelos│                           │  - Ejecuta reglas    │
│  - Extrae feat. │                           │  - UI/Rendering      │
│  - Infiere      │                           │  - Input humano      │
│  - Registra     │                           │  - Envía estado      │
└─────────────────┘                           └─────────────────────┘
```

### 3.2 Protocolo por turno

```
1. Unity tira dado → obtiene dice_value
2. Unity construye estado y lo envía al middleware:
   {
     "dice_value": 4,
     "current_player": 0,
     "my_cols": [[3,0,0], [5,2,0], [0,0,0]],
     "op_cols": [[1,1,0], [0,0,0], [6,0,0]],
     "score_my": 28,
     "score_op": 15,
     "legal_actions": [0, 2],
     "turn": 7
   }
3. Middleware:
   a) Extrae features (21 dims) del estado
   b) Si usa sistema adaptativo: actualiza perfil KMeans del oponente
   c) Selecciona acción con la policy activa
   d) Registra métricas (latencia, policy usada, etc.)
   e) Retorna:
   {
     "action": 2,
     "info": {
       "policy_name": "neat_vs_denial",
       "latency_ms": 0.3,
       "cluster": 1
     }
   }
4. Unity aplica la acción al tablero
5. Unity llama on_turn_end con el resultado del turno
```

### 3.3 Comunicación recomendada

**Opción A — HTTP local (recomendado para prototipo)**:
- Servidor Flask/FastAPI en Python (localhost:5000)
- Unity hace POST `/decide` con el estado JSON
- Simplicidad máxima, debugging fácil

**Opción B — Named Pipes / Sockets**:
- Menor latencia que HTTP
- Más complejo de implementar
- Recomendado si se necesita <1ms por decisión

**Opción C — Exportar modelos a ONNX y cargar en Unity**:
- Solo viable para PPO (redes neuronales estándar)
- NO viable para NEAT (topología variable) ni DT sklearn
- Elimina la dependencia de Python en runtime

---

## 4. Estructura del estado (obs)

El estado que Unity debe enviar al middleware en cada turno:

```json
{
  "dice_value": 4,
  "current_player": 0,
  "my_cols": [[3, 0, 0], [5, 2, 0], [0, 0, 0]],
  "op_cols": [[1, 1, 0], [0, 0, 0], [6, 0, 0]],
  "score_my": 28,
  "score_op": 15,
  "legal_actions": [0, 2],
  "turn": 7
}
```

### 4.1 Campos obligatorios

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `dice_value` | int (1-6) | Valor del dado tirado este turno |
| `current_player` | int (0 o 1) | Qué jugador toma la decisión |
| `my_cols` | int[3][3] | Tablero del jugador activo, 3 columnas de 3 filas. `0 = vacío` |
| `op_cols` | int[3][3] | Tablero del oponente, misma estructura |
| `score_my` | int | Puntaje total actual del jugador activo |
| `score_op` | int | Puntaje total actual del oponente |
| `legal_actions` | int[] | Columnas válidas (con espacio libre), subconjunto de [0,1,2] |
| `turn` | int | Número de turno (0-indexed) |

### 4.2 Convenciones

- **Orden de las columnas**: `my_cols[c][r]` donde `c` = columna (0-2), `r` = fila (0-2, de abajo hacia arriba)
- **Vacío = 0**: una celda sin dado tiene valor 0
- **Perspectiva**: `my_cols` siempre es el tablero del jugador que decide, `op_cols` el del rival. Unity debe rotar según quién juega.

---

## 5. Vector de features (21 dimensiones)

Los modelos entrenados esperan un vector numérico de **exactamente 21 dimensiones** (`float32`):

```
[dice_value, my_board_flat(9), op_board_flat(9), score_my, score_op]
```

### 5.1 Descomposición

| Índice | Feature | Origen |
|--------|---------|--------|
| 0 | dice_value | `obs["dice_value"]` |
| 1-3 | my_col0 (fila 0,1,2) | `obs["my_cols"][0]` |
| 4-6 | my_col1 (fila 0,1,2) | `obs["my_cols"][1]` |
| 7-9 | my_col2 (fila 0,1,2) | `obs["my_cols"][2]` |
| 10-12 | op_col0 (fila 0,1,2) | `obs["op_cols"][0]` |
| 13-15 | op_col1 (fila 0,1,2) | `obs["op_cols"][1]` |
| 16-18 | op_col2 (fila 0,1,2) | `obs["op_cols"][2]` |
| 19 | score_my | `obs["score_my"]` |
| 20 | score_op | `obs["score_op"]` |

### 5.2 Código de referencia (Python)

```python
def extract_features(obs):
    x = [obs["dice_value"]]
    for c in range(3):
        x.extend(obs["my_cols"][c])  # 3 valores por col
    for c in range(3):
        x.extend(obs["op_cols"][c])  # 3 valores por col
    x.append(obs["score_my"])
    x.append(obs["score_op"])
    return np.array(x, dtype=np.float32)  # shape (21,)
```

### 5.3 Versión

`FEATURES_VERSION = 1`. Si se modifica la composición del vector, todos los modelos previos son incompatibles.

---

## 6. Perfil de oponente — KMeans (10 dimensiones)

El sistema adaptativo utiliza un perfilador KMeans para detectar el estilo de juego del oponente en tiempo real (sin reentrenamiento).

### 6.1 Ventana deslizante

Se observan las **últimas 6 acciones del oponente** y se construye un vector de 10 dimensiones:

| Índice | Feature | Descripción |
|--------|---------|-------------|
| 0-2 | hist_cols[3] | Histograma normalizado de columnas elegidas |
| 3-5 | hist_n_legal[3] | Histograma normalizado de nº de acciones legales (1,2,3) |
| 6 | mean_is_first_legal | Proporción de veces que eligió la primera acción legal |
| 7 | mean_is_last_legal | Proporción de veces que eligió la última acción legal |
| 8 | mean_repeat_col | Proporción de veces que repitió la columna anterior |
| 9 | entropy_cols | Entropía de la distribución de columnas (0 = siempre la misma, ~1.1 = uniforme) |

### 6.2 Hook on_turn_end

Después de cada turno del **oponente**, se debe llamar `on_turn_end` con la información del turno para actualizar la ventana:

```json
{
  "player": 1,
  "action": 2,
  "die": 4,
  "legal_actions": [0, 1, 2]
}
```

Solo se actualizan los turnos del oponente (el jugador contrario al controlado por la IA).

### 6.3 Activación

El perfilador necesita al menos 6 turnos del oponente para clasificar. Antes de eso, se usa la **fallback policy** especificada en la configuración del sistema.

---

## 7. Modelos entrenados — Formatos y paths

### 7.1 Tabla de artefactos

| Tipo | Formato | Path relativo | Tamaño |
|------|---------|---------------|--------|
| PPO vs denial | .zip (SB3) | `gym/data/models/rl/PPO__vs_heuristic_denial.zip` | ~162 KB |
| PPO vs spread | .zip (SB3) | `gym/data/models/rl/PPO__vs_heuristic_spread.zip` | ~162 KB |
| PPO vs greedy | .zip (SB3) | `gym/data/models/rl/PPO__vs_heuristic_greedy.zip` | ~162 KB |
| NEAT vs denial | .pkl (pickle) | `gym/data/models/neat/neat__vs_heuristic_denial.pkl` | ~2.4 KB |
| NEAT vs spread | .pkl (pickle) | `gym/data/models/neat/neat__vs_heuristic_spread.pkl` | ~4.2 KB |
| NEAT vs greedy | .pkl (pickle) | `gym/data/models/neat/neat__vs_heuristic_greedy.pkl` | ~3.6 KB |
| DT vs denial | .joblib | `gym/data/models/sklearn/dt__ppo_vs_denial.joblib` | ~134 KB |
| DT vs spread | .joblib | `gym/data/models/sklearn/dt__ppo_vs_spread.joblib` | ~153 KB |
| DT vs greedy | .joblib | `gym/data/models/sklearn/dt__ppo_vs_greedy.joblib` | ~147 KB |
| KMeans | .joblib | `gym/data/models/kmeans/kmeans_profiler` | — |
| NEAT config | .ini | `gym/config/neat/neat_config.ini` | — |
| Response Table DT | .json | `response_table_dt` | — |
| Response Table NEAT | .json | `response_table_neat` | — |
| Response Table PPO | .json | `response_table_ppo` | — |
| System Config DT | .json | `gym/config/systems/system_dt.json` | — |
| System Config NEAT | .json | `gym/config/systems/system_neat.json` | — |
| System Config PPO | .json | `gym/config/systems/system_ppo.json` | — |

### 7.2 Cómo cargar cada tipo

**PPO (Stable-Baselines3)**:
```python
from stable_baselines3 import PPO
model = PPO.load("gym/data/models/rl/PPO__vs_heuristic_denial.zip")
action, _ = model.predict(features_vector, deterministic=True)
```

**NEAT (neat-python)**:
```python
import pickle, neat
genome = pickle.load(open("gym/data/models/neat/neat__vs_heuristic_denial.pkl", "rb"))
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     "gym/config/neat/neat_config.ini")
net = neat.nn.FeedForwardNetwork.create(genome, config)
output = net.activate(features_vector)
action = max(range(3), key=lambda i: output[i])
```

**DT (scikit-learn)**:
```python
import joblib
model = joblib.load("gym/data/models/sklearn/dt__ppo_vs_denial.joblib")
action = model.predict(features_vector.reshape(1, -1))[0]
```

**KMeans (artefacto compuesto)**:
```python
import joblib
artifact = joblib.load("gym/data/models/kmeans/kmeans_profiler")
# artifact contiene: scaler, kmeans, cluster_to_style
profile_vector = ...  # 10 dims del OpponentProfileWindowV2
scaled = artifact["scaler"].transform(profile_vector.reshape(1, -1))
cluster = artifact["kmeans"].predict(scaled)[0]
style = artifact["cluster_to_style"][str(cluster)]
```

---

## 8. Contrato de la policy (interfaz)

Todas las policies implementan la misma interfaz abstracta:

```python
class BasePolicy:
    def reset(self, seed=None) -> None
    def select_action(self, obs: dict, legal_actions: list[int]) -> PolicyStep
    def on_turn_end(self, record: dict) -> None

@dataclass
class PolicyStep:
    action: int       # columna elegida (0, 1 o 2)
    info: dict        # metadata de la decisión
```

### 8.1 Gramática de specs (policy_factory)

Para instanciar cualquier policy desde un string:

```
baseline:first              → siempre elige primera columna disponible
baseline:random             → columna aleatoria
heuristic:greedy            → maximiza puntaje inmediato
heuristic:denial            → prioriza destruir columnas del rival
heuristic:spread            → distribuye fichas entre columnas
heuristic:random            → acciones completamente aleatorias
dt:<path.joblib>            → Decision Tree desde archivo
rl:PPO:<path.zip>           → Modelo PPO desde archivo
neat:<genome.pkl>:<cfg.ini> → NEAT desde genoma + config
system:<config.json>        → Sistema adaptativo completo
```

---

## 9. Sistemas adaptativos — Estructura

Cada sistema combina 3 componentes:

```
KMeans (perfilador, compartido) → detecta estilo oponente
Response Table (por sistema)    → mapea cluster → mejor especialista
Especialistas (por familia)     → ejecutan la acción
```

### 9.1 Configuración JSON de un sistema

```json
{
  "name": "system_neat",
  "kmeans_artifact": "gym/data/models/kmeans/kmeans_profiler",
  "response_table": "response_table_neat",
  "my_player_id": 0,
  "fallback_policy_spec": "neat:gym/data/models/neat/neat__vs_heuristic_denial.pkl:gym/config/neat/neat_config.ini"
}
```

### 9.2 Response Table

```json
{
  "cluster_to_policy_spec": {
    "0": "neat:gym/data/models/neat/neat__vs_heuristic_denial.pkl:gym/config/neat/neat_config.ini",
    "1": "neat:gym/data/models/neat/neat__vs_heuristic_denial.pkl:gym/config/neat/neat_config.ini",
    "2": "neat:gym/data/models/neat/neat__vs_heuristic_denial.pkl:gym/config/neat/neat_config.ini",
    "3": "neat:gym/data/models/neat/neat__vs_heuristic_denial.pkl:gym/config/neat/neat_config.ini"
  }
}
```

### 9.3 Flujo adaptativo en runtime

```
Turno 1-5: usar fallback_policy (KMeans aún no tiene datos suficientes)
Turno 6+:
  1. Ventana deslizante tiene 6 acciones del oponente
  2. Calcular profile_vector (10 dims)
  3. KMeans → cluster
  4. Response Table[cluster] → policy_spec
  5. policy_factory.build_policy(spec) → policy
  6. policy.select_action(obs, legal_actions) → acción
```

---

## 10. Recomendación de sistema

| Criterio | Recomendación |
|----------|:-------------:|
| Mejor rendimiento competitivo | **System NEAT** (WR = 0.653) |
| Mejor opción práctica / interpretable | System DT (WR = 0.608) |
| Modelo más liviano | NEAT (3.4 KB promedio) |
| Inferencia más rápida | DT (~μs por decisión) |
| Sin dependencia de Python en runtime | DT (exportable a reglas) |

**Recomendación principal**: usar **System NEAT** como oponente principal en el videojuego. Mantener **System DT** como fallback si se necesita depurar o exportar a C# nativo.

---

## 11. Consideraciones para Unity

### 11.1 Equivalencia funcional (CRÍTICO)

El tablero Unity **DEBE** implementar exactamente las mismas reglas que `KnucklebonesEnv`:

- La destrucción elimina TODAS las ocurrencias del valor en la columna del oponente, no solo una
- El dado se coloca en la primera fila vacía (de abajo hacia arriba)
- El score se calcula como Σ v × count(v)² por columna
- `0 = vacío`, no null ni -1

Se recomienda generar **golden logs** desde Python y validarlos en Unity:

```python
python -m gym.scripts.utils.test_env_sanity
```

### 11.2 Latencia esperada

| Componente | Latencia típica |
|------------|:--------------:|
| Extract features | <0.01 ms |
| KMeans predict | <0.1 ms |
| NEAT inference | <0.1 ms |
| DT inference | <0.01 ms |
| PPO inference | ~1-5 ms |
| HTTP roundtrip | ~1-5 ms |

Total esperado con middleware HTTP: **< 10 ms por turno** (imperceptible para el jugador humano).

### 11.3 Formato de tablero — Python vs Unity

En Python, el tablero se almacena como `list[list[int]]` donde `board[col][row]`:
- `board[0]` = columna izquierda
- `board[1]` = columna central
- `board[2]` = columna derecha
- `board[c][0]` = fila inferior, `board[c][2]` = fila superior

Unity debe mantener esta misma convención o mapear explícitamente al enviar/recibir estados.

### 11.4 Métricas a registrar en Unity

Para la tesis, Unity debería registrar (o facilitar el registro desde el middleware):

- Latencia de decisión por turno (ms)
- Policy activa por turno (nombre del especialista)
- Cluster detectado por turno
- Resultado final de cada partida (winner, scores, turnos)
- Uso de CPU/RAM del proceso completo

---

## 12. Arquetipos heurísticos (para testing en Unity)

Pueden implementarse directamente en C# para testing sin middleware:

### 12.1 Greedy
Elegir la columna que maximiza el puntaje inmediato.

### 12.2 Denial
Elegir la columna que maximiza la destrucción de dados del oponente.

### 12.3 Spread
Elegir la columna con menos dados (distribuir uniformemente).

### 12.4 Random
Elegir una columna aleatoria de las disponibles.

---

## 13. Checklist de integración

- [ ] Implementar reglas Knucklebones en Unity (tablero, scoring, destrucción)
- [ ] Validar con golden logs vs simulador Python
- [ ] Implementar middleware HTTP (Flask/FastAPI en Python)
- [ ] Endpoint `/decide` que recibe estado y retorna acción
- [ ] Endpoint `/reset` para reiniciar partida y perfilador
- [ ] Endpoint `/on_turn_end` para actualizar perfil del oponente
- [ ] Cargar modelos al inicio del servidor
- [ ] Implementar selector de sistema (NEAT/DT/PPO) en UI
- [ ] Registrar métricas por turno y por partida
- [ ] Testing con heurísticos implementados en Unity
- [ ] Testing con modelos Python vía middleware
- [ ] Medir latencia end-to-end

---

## 14. Archivos que Unity necesita del repositorio

```
gym/
├── config/
│   ├── neat/neat_config.ini
│   └── systems/
│       ├── system_dt.json
│       ├── system_neat.json
│       └── system_ppo.json
├── data/models/
│   ├── kmeans/kmeans_profiler
│   ├── neat/*.pkl
│   ├── rl/*.zip
│   ├── sklearn/*.joblib
│   └── systems/response_table__*.json
├── env/knucklebones_env.py       ← referencia de reglas
├── policies/                     ← todas las policies
└── scripts/utils/evaluate_any.py ← para generar golden logs
```

Los response tables también están en la raíz:
```
response_table_dt
response_table_neat
response_table_ppo
```

