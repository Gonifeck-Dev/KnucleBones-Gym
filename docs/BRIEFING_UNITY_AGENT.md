# Briefing Completo para Integración Unity — Knucklebones AI

> **Documento preparado para**: Agente Claude Opus 4.6 (Unity)
> **Preparado por**: Agente Claude (Pipeline de entrenamiento Python)
> **Fecha**: 2026-03-06
> **Estado**: Entrenamiento COMPLETADO — Todos los modelos finales listos

---

## RESUMEN EJECUTIVO

Se ha completado el pipeline completo de entrenamiento de 3 sistemas de IA adaptativa para el juego **Knucklebones** (Cult of the Lamb). El lado Python está cerrado y listo para conectarse con Unity mediante un middleware local.

**Lo que necesitamos de Unity**:
1. Implementación del juego Knucklebones con reglas exactas
2. Middleware HTTP para comunicación con el servidor Python de IA
3. Registro de métricas durante gameplay
4. UI para que un humano juegue contra las IAs

---

## PARTE 1: REGLAS DEL JUEGO

### Estructura
- 2 jugadores, tablero **3 columnas × 3 filas** cada uno
- Celdas: vacío = 0, o dado 1-6
- Turnos alternados entre jugador 0 y jugador 1

### Flujo de un turno
```
1. Tirar dado → dice_value (1-6, aleatorio)
2. Elegir columna (0, 1 o 2) — solo columnas con espacio
3. Colocar dado en primera fila vacía (de abajo hacia arriba)
4. DESTRUCCIÓN: eliminar TODOS los dados iguales del OPONENTE en la MISMA columna
5. Recalcular puntajes
```

### Fórmula de puntuación
```
Score(columna) = Σ para cada valor v: v × (count(v) en la columna)²

Ejemplos:
  [3, 0, 0] → 3 × 1² = 3
  [3, 3, 0] → 3 × 2² = 12
  [3, 3, 3] → 3 × 3² = 27
  [2, 3, 5] → 2 + 3 + 5 = 10
  [4, 4, 6] → 4×2² + 6×1² = 22

Score_total = Score(col0) + Score(col1) + Score(col2)
```

### Destrucción — DETALLE CRÍTICO
Al colocar dado `d` en columna `c`:
- Se eliminan **TODAS** las ocurrencias de `d` en la columna `c` del **oponente**
- No solo una, **TODAS**
- Las celdas quedan en 0
- Las filas se compactan (los valores restantes bajan)

Ejemplo: si el oponente tiene `col[1] = [3, 5, 3]` y se coloca un `3` en columna 1:
→ `col[1]` del oponente queda `[5, 0, 0]`

### Fin de partida
- Cuando **cualquier** jugador llena su tablero (9 celdas ocupadas)
- Ganador = mayor puntaje total
- Empate si puntajes iguales

### Código de referencia Python (función de transición)
```python
def apply_action(my_board, op_board, chosen_col, die):
    my2 = deep_copy_board(my_board)
    op2 = deep_copy_board(op_board)
    r = first_empty_row(my2[chosen_col])  # primera fila vacía
    my2[chosen_col][r] = die
    # Destrucción: eliminar TODOS los dados iguales en la misma columna del oponente
    for r in range(3):
        if op2[chosen_col][r] == die:
            op2[chosen_col][r] = 0
    return my2, op2

def score_column(col):
    s = 0
    for v in range(1, 7):
        cnt = sum(1 for x in col if x == v)
        s += v * (cnt * cnt)
    return s

def score_board(b):
    return score_column(b[0]) + score_column(b[1]) + score_column(b[2])
```

---

## PARTE 2: ARQUITECTURA DEL SISTEMA

### Diagrama general
```
┌──────────────────────────┐
│       Unity (C#)         │
│                          │
│  ┌──────────────────┐    │
│  │  KnucklebonesGame │   │
│  │  (reglas + estado)│   │
│  └──────────┬───────┘    │
│             │             │
│  ┌──────────▼───────┐    │
│  │  Middleware Client│    │     HTTP POST (localhost)
│  │  (C# HttpClient) │────────────────────────┐
│  └──────────────────┘    │                    │
│                          │                    ▼
│  ┌──────────────────┐    │    ┌──────────────────────────┐
│  │  Metrics Logger   │   │    │   Servidor Python        │
│  │  (registra datos) │   │    │   (Flask/FastAPI)         │
│  └──────────────────┘    │    │                          │
│                          │    │  ┌─────────────────────┐ │
│  ┌──────────────────┐    │    │  │ PolicyFactory        │ │
│  │  UI / Input       │   │    │  │ → SystemPolicy       │ │
│  │  (humano o IA)    │   │    │  │ → KMeans + RT + AI   │ │
│  └──────────────────┘    │    │  └─────────────────────┘ │
└──────────────────────────┘    └──────────────────────────┘
```

### Endpoints del middleware Python

**POST `/decide`** — Solicitar acción de la IA
```json
// Request
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

// Response
{
  "action": 2,
  "info": {
    "policy_name": "neat_vs_denial",
    "latency_ms": 0.3,
    "cluster": 1,
    "system": "system_neat"
  }
}
```

**POST `/on_turn_end`** — Notificar resultado del turno (para actualizar perfil KMeans)
```json
{
  "player": 1,
  "action": 2,
  "die": 4,
  "legal_actions": [0, 1, 2]
}
```

**POST `/reset`** — Reiniciar para nueva partida
```json
{
  "system": "system_neat",
  "seed": 123
}
```

**GET `/status`** — Verificar que el servidor está activo
```json
{
  "status": "ok",
  "loaded_system": "system_neat",
  "models_loaded": true
}
```

### Formato del tablero
```
Python: board[columna][fila]  — board[0][0] = celda inferior izquierda
Unity debe usar la misma convención.

Ejemplo visual:
  Col 0  Col 1  Col 2
  ─────  ─────  ─────
  [2]=0  [2]=0  [2]=0    ← fila 2 (arriba)
  [1]=5  [1]=3  [1]=0    ← fila 1 (medio)
  [0]=3  [0]=2  [0]=6    ← fila 0 (abajo)

JSON: [[3,5,0], [2,3,0], [6,0,0]]
```

---

## PARTE 3: RESULTADOS EXPERIMENTALES (para contexto)

### Los 3 Sistemas de IA

| Sistema | Qué es | WR promedio | Significancia |
|---------|--------|:-----------:|:-------------:|
| **System NEAT** | Neuroevolución (NEAT-Python) | **0.653** | Mejor |
| System DT | Árboles de Decisión destilados de PPO | 0.608 | Equivalente a PPO |
| System PPO | Reinforcement Learning (PPO, Stable-Baselines3) | 0.604 | Equivalente a DT |

### Cómo funciona cada sistema
```
KMeans (perfilador compartido)
  ↓ detecta estilo del oponente (6 turnos de observación)
Response Table (por sistema)
  ↓ mapea cluster → mejor especialista
Especialistas (3 por sistema)
  ↓ selecciona acción
```

### Oponentes heurísticos (para testing)
- **Greedy**: maximiza puntaje inmediato
- **Denial**: prioriza destruir dados del rival
- **Spread**: distribuye dados uniformemente
- **Random**: acción aleatoria

---

## PARTE 4: VECTOR DE FEATURES (21 dimensiones)

Todos los modelos esperan un vector `float32[21]`:

```
Índice  Qué contiene
──────  ─────────────────────────
  0     dice_value (1-6)
  1-3   my_col0 [fila0, fila1, fila2]
  4-6   my_col1 [fila0, fila1, fila2]
  7-9   my_col2 [fila0, fila1, fila2]
 10-12  op_col0 [fila0, fila1, fila2]
 13-15  op_col1 [fila0, fila1, fila2]
 16-18  op_col2 [fila0, fila1, fila2]
  19    score_my
  20    score_op
```

Los valores van en orden: columna 0 fila 0, columna 0 fila 1, columna 0 fila 2, columna 1 fila 0, etc.

---

## PARTE 5: PERFIL DE OPONENTE (KMeans, 10 dimensiones)

El KMeans necesita 6 turnos del oponente para activarse. Antes de eso, se usa un fallback (el especialista vs denial para NEAT).

### Vector de perfil (10 dims)
```
Índice  Qué contiene
──────  ─────────────────────────
  0-2   hist_cols[3] — histograma normalizado de columnas elegidas por el oponente
  3-5   hist_n_legal[3] — histograma normalizado de acciones legales disponibles (1,2,3)
  6     mean_is_first_legal — proporción de veces que eligió la primera acción legal
  7     mean_is_last_legal — proporción de veces que eligió la última acción legal
  8     mean_repeat_col — proporción de veces que repitió la columna del turno anterior
  9     entropy_cols — entropía de la distribución de columnas (0=siempre misma, ~1.1=uniforme)
```

Este vector se calcula sobre una ventana deslizante de 6 turnos del oponente.

---

## PARTE 6: MODELOS ENTRENADOS

### Inventario de artefactos

| Archivo | Tipo | Para qué |
|---------|------|----------|
| `gym/data/models/rl/PPO__vs_heuristic_*.zip` | SB3 model | PPO especialistas |
| `gym/data/models/neat/neat__vs_heuristic_*.pkl` | Pickle genome | NEAT especialistas |
| `gym/data/models/sklearn/dt__ppo_vs_*.joblib` | Sklearn model | DT destilados |
| `gym/data/models/kmeans/kmeans_profiler` | Joblib artifact | Perfilador KMeans |
| `gym/config/neat/neat_config.ini` | Config NEAT | Necesario para cargar genomas |
| `response_table_neat` (raíz) | JSON | Mapping cluster→especialista NEAT |
| `response_table_dt` (raíz) | JSON | Mapping cluster→especialista DT |
| `response_table_ppo` (raíz) | JSON | Mapping cluster→especialista PPO |
| `gym/config/systems/system_*.json` | JSON | Config de cada sistema |

### Cómo cargar modelos (referencia Python)

**PPO**: `PPO.load("path.zip")` → `model.predict(features, deterministic=True)`

**NEAT**: `pickle.load("path.pkl")` → `neat.nn.FeedForwardNetwork.create(genome, config)` → `net.activate(features)` → `argmax(output)`

**DT**: `joblib.load("path.joblib")` → `model.predict(features.reshape(1,-1))[0]`

**KMeans**: `joblib.load("kmeans_profiler")` → contiene `scaler`, `kmeans`, `cluster_to_style` → `scaler.transform(profile)` → `kmeans.predict(scaled)` → cluster

---

## PARTE 7: MÉTRICAS A REGISTRAR EN UNITY

### Por turno
- `timestamp` (ISO 8601)
- `turn_number`
- `current_player` (0 o 1)
- `dice_value`
- `action_chosen` (columna)
- `policy_name` (nombre del especialista usado)
- `cluster_detected` (si sistema adaptativo)
- `decision_latency_ms`
- `score_p0`, `score_p1` (después de aplicar)

### Por partida
- `game_id`
- `system_used` (system_neat, system_dt, etc.)
- `winner` (0, 1 o null=empate)
- `final_score_p0`, `final_score_p1`
- `total_turns`
- `total_time_seconds`
- `avg_latency_ms`

### Por sesión
- `cpu_usage_percent`
- `ram_usage_mb`
- `total_games`
- `total_time_minutes`

---

## PARTE 8: VALIDACIÓN DE EQUIVALENCIA (CRÍTICO)

Para garantizar que Unity y Python implementan las mismas reglas:

1. **Generar golden logs desde Python**:
```powershell
python -m gym.scripts.utils.test_env_sanity
```
Esto produce trazas de estado→acción→estado_nuevo verificadas.

2. **Implementar las mismas transiciones en Unity C#**

3. **Validar**: para cada transición en el golden log, verificar que Unity produce el mismo estado resultante.

**Puntos de fallo comunes**:
- La destrucción NO compacta filas en Python (deja 0s en su lugar). Verificar que Unity haga lo mismo.
- El orden de las filas: fila 0 = abajo, fila 2 = arriba.
- El score se recalcula DESPUÉS de la destrucción.
- `first_empty_row` busca la primera fila con valor 0 de abajo hacia arriba.

---

## PARTE 9: RECOMENDACIONES DE IMPLEMENTACIÓN

### Prioridad 1: Implementar juego + middleware básico
1. Juego Knucklebones funcional en Unity con reglas validadas
2. Servidor Python Flask con endpoint `/decide`
3. Un humano puede jugar contra System NEAT

### Prioridad 2: Selector de sistema
4. UI para elegir entre System NEAT / DT / PPO
5. Mostrar métricas en tiempo real (latencia, cluster, policy activa)

### Prioridad 3: Registro completo
6. Logger de métricas por turno y por partida
7. Exportar datos para análisis posterior

### Latencias esperadas
| Componente | Tiempo |
|------------|:------:|
| Extracción de features | < 0.01 ms |
| Inferencia NEAT | < 0.1 ms |
| Inferencia DT | < 0.01 ms |
| Inferencia PPO | ~ 1-5 ms |
| HTTP roundtrip local | ~ 1-5 ms |
| **Total por turno** | **< 10 ms** |

---

## PARTE 10: DEPENDENCIAS PYTHON (para el middleware)

```
numpy>=1.24
scikit-learn>=1.3
joblib>=1.3
neat-python>=0.92
stable-baselines3>=2.1
gymnasium>=0.29
flask>=3.0  (o fastapi + uvicorn)
torch>=2.0  (requerido por SB3)
```

Instalar: `pip install -r requirements.txt`

---

## CONTACTO / PREGUNTAS

Si el agente Unity tiene dudas sobre:
- **Reglas del juego** → ver `gym/env/knucklebones_env.py`
- **Features** → ver `gym/policies/utils/features.py`
- **Perfil KMeans** → ver `gym/policies/kmeans/profile_features.py`
- **Cómo funciona un sistema** → ver `gym/policies/systems/system_policy.py`
- **Configs** → ver `gym/config/systems/system_*.json`
- **Documentación completa** → ver `docs/01` a `docs/05`

