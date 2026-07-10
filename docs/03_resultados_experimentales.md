# Documento 3 — Resultados Experimentales y Evidencia

> **Datos finales** — Actualizado 2026-03-06. Entrenamiento cerrado.
> Re-entrenamiento completo con multi-seed real NEAT (3 seeds × 300 gen) y PPO 4-envs paralelos.
> Análisis estadístico formal ejecutado con `statistical_analysis.py`.

---

## 1. Entrenamiento de modelos

### 1.1 Tiempos de entrenamiento

| Familia | Modelo | Oponente | Tiempo (s) | Tamaño (KB) | Método |
|---------|--------|----------|----------:|------------:|--------|
| PPO | PPO\_\_vs\_denial | heuristic:denial | 1369 | 162 | 4 envs paralelos |
| PPO | PPO\_\_vs\_spread | heuristic:spread | 390 | 162 | 4 envs paralelos |
| PPO | PPO\_\_vs\_greedy | heuristic:greedy | 400 | 162 | 4 envs paralelos |
| NEAT | neat\_\_vs\_denial | heuristic:denial | 1995 | 2.4 | Multi-seed real (3 seeds, 300 gen) |
| NEAT | neat\_\_vs\_spread | heuristic:spread | 1868 | 4.2 | Multi-seed real (3 seeds, 300 gen) |
| NEAT | neat\_\_vs\_greedy | heuristic:greedy | 1818 | 3.6 | Multi-seed real (3 seeds, 300 gen) |
| DT | sklearn\_tree\_\_vs\_denial | (destilado) | 2.2 | 134 | Behavior Cloning |
| DT | sklearn\_tree\_\_vs\_spread | (destilado) | ~2 | 153 | Behavior Cloning |
| DT | sklearn\_tree\_\_vs\_greedy | (destilado) | ~2 | 147 | Behavior Cloning |
| KMeans | kmeans\_profiler | 4 estilos | ~50 | — | k=4, win=6 |

**Observaciones**:
- PPO con 4 envs paralelos: ~6-23 min por especialista (400k timesteps)
- NEAT multi-seed real: ~30 min por especialista (300 gen × 150 pop × 102 eps/genoma con 3 seeds)
- Destilación PPO→DT: instantánea (<3s), reduce tamaño ~100x
- KMeans se entrena en menos de 1 minuto

### 1.2 Calidad de destilación (DT accuracy)

| DT destilado | Accuracy | Profundidad | Hojas |
|-------------|--------:|----------:|------:|
| sklearn\_tree\_\_vs\_denial | 93.79% | 10 | 773 |
| sklearn\_tree\_\_vs\_spread | ~94% | 10 | ~800 |
| sklearn\_tree\_\_vs\_greedy | ~94% | 10 | ~800 |

La destilación conserva >92% de fidelidad al teacher PPO.

### 1.3 KMeans — perfilado de estilos

- **k = 4**, ventana = 6 turnos, perfil = 10 dimensiones
- **Silhouette score**: 0.232 (separación moderada, esperable en estilos de juego)
- Mapping cluster→estilo:
  - Cluster 0 → heuristic\_greedy
  - Cluster 1 → heuristic\_denial
  - Cluster 2 → heuristic\_denial
  - Cluster 3 → heuristic\_spread

---

## 2. Evaluación de los 3 sistemas adaptativos

Cada sistema combina: **KMeans (perfilador compartido) + Response Table + Especialistas del tipo**.

### 2.1 Tabla comparativa — Winrate de cada sistema vs cada oponente

| Sistema | vs denial | vs spread | vs greedy | Promedio | IC 95% global |
|---------|----------:|----------:|----------:|---------:|:-------------:|
| **system\_neat** (NEAT directos) | **0.635** | **0.697** | **0.626** | **0.653** | [0.645, 0.660] |
| **system\_dt** (DT destilados) | 0.592 | 0.641 | 0.591 | 0.608 | [0.600, 0.616] |
| **system\_ppo** (PPO directos) | 0.588 | 0.641 | 0.584 | 0.604 | [0.597, 0.612] |

### 2.2 Test estadístico entre sistemas

Se aplicó un **z-test de dos proporciones** (bilateral) a cada par de sistemas por oponente (9 comparaciones), con corrección de Bonferroni para comparaciones múltiples.

| Par comparado | vs denial (p) | vs spread (p) | vs greedy (p) |
|---------------|:-------------:|:--------------:|:-------------:|
| DT vs PPO | 1.000 | 1.000 | 1.000 |
| DT vs NEAT | **0.0001** | **<0.0001** | **0.003** |
| PPO vs NEAT | **<0.0001** | **<0.0001** | **0.0002** |

**Resultado: 6/9 comparaciones son significativas** tras corrección de Bonferroni (α = 0.05).

**Conclusión estadística**: NEAT es significativamente superior a DT y PPO en los 3 oponentes. DT y PPO son estadísticamente equivalentes.

### 2.3 Resultados detallados por sistema

#### System DT (KMeans + DT especialistas destilados de PPO)

| Oponente | Games | Winrate decided | IC 95% | Avg diff | Wall (s) |
|----------|------:|:--------:|:------:|---------:|---------:|
| heuristic:denial | 5,000 | 0.592 | [0.578, 0.606] | +4.80 | 30.4 |
| heuristic:spread | 5,000 | 0.641 | [0.628, 0.654] | +8.06 | 40.0 |
| heuristic:greedy | 5,000 | 0.591 | [0.577, 0.605] | +4.68 | 28.5 |

#### System PPO (KMeans + PPO especialistas directos)

| Oponente | Games | Winrate decided | IC 95% | Avg diff | Wall (s) |
|----------|------:|:--------:|:------:|---------:|---------:|
| heuristic:denial | 5,000 | 0.588 | [0.574, 0.602] | +4.70 | 45.2 |
| heuristic:spread | 5,000 | 0.641 | [0.628, 0.655] | +8.00 | 44.3 |
| heuristic:greedy | 5,000 | 0.584 | [0.571, 0.598] | +4.50 | 43.1 |

#### System NEAT (KMeans + NEAT especialistas directos)

| Oponente | Games | Winrate decided | IC 95% | Avg diff | Wall (s) |
|----------|------:|:--------:|:------:|---------:|---------:|
| heuristic:denial | 5,000 | 0.635 | [0.621, 0.648] | +6.40 | 28.8 |
| heuristic:spread | 5,000 | 0.697 | [0.684, 0.710] | +11.0 | 27.4 |
| heuristic:greedy | 5,000 | 0.626 | [0.613, 0.640] | +6.00 | 27.1 |

---

## 3. Ablation Study — ¿Aporta la adaptación KMeans?

Para cada familia se comparan 4 condiciones usando datos de la evaluación cruzada (2,000 partidas por matchup):

### 3.1 Resultados del ablation

| Familia | Peor generalista | Mejor generalista | Sistema adaptativo | Oráculo | Δ Sist. vs Mejor Gen. |
|---------|:----------------:|:-----------------:|:------------------:|:-------:|:---------------------:|
| **DT** | 0.585 (dt→spread) | **0.618** (dt→greedy) | 0.608 | 0.618 | **-0.010** |
| **PPO** | 0.572 (ppo→spread) | **0.610** (ppo→greedy) | 0.604 | 0.610 | **-0.005** |
| **NEAT** | 0.543 (neat→spread) | **0.651** (neat→denial) | **0.653** | 0.651 | **+0.002** |

**Definiciones**:
- **Peor generalista**: El especialista con menor WR promedio, usado para todo.
- **Mejor generalista**: El especialista con mayor WR promedio, usado para todo.
- **Sistema adaptativo**: El sistema completo (KMeans + response table + selección).
- **Oráculo**: Selección perfecta del mejor especialista por oponente (límite teórico).

### 3.2 Hallazgo clave

**El sistema adaptativo NO mejora sobre el mejor generalista fijo**. En DT y PPO, el sistema adaptativo incluso **empeora** ligeramente (-0.9% y -1.1%) respecto a usar el mejor generalista solo. Esto se debe a:

1. El KMeans identifica 2/4 clusters como "denial" → no distingue bien entre estilos.
2. El Oráculo ≈ Mejor Generalista → un solo especialista (entrenado vs greedy) es bueno contra todo.
3. La response table a veces selecciona un especialista subóptimo para un cluster.

### 3.3 Dominancia en Response Tables

| Sistema | Especialista dominante | Cobertura | Especialistas únicos usados |
|---------|:----------------------:|:---------:|:---------------------------:|
| system\_dt | dt\_\_ppo\_vs\_greedy | **3/4** (75%) | 2 |
| system\_ppo | PPO\_\_vs\_greedy | **3/4** (75%) | 2 |
| system\_neat | neat\_\_vs\_denial | **4/4** (100%) | 1 |

**Interpretación para la tesis**: El marco de adaptación es válido arquitectónicamente, pero en Knucklebones los estilos no son suficientemente distintos para beneficiarse. Los agentes aprenden una política generalista fuerte que domina contra todos los heurísticos. Juegos con mayor diversidad estratégica (RTS, MOBA) se beneficiarían más de este enfoque.

---

## 4. Evaluación Cruzada — Generalización vs Overfitting

> Cada especialista se evaluó contra **todos** los oponentes (2,000 partidas por matchup).
> ★ = oponente contra el que fue entrenado.

### 4.1 NEAT Especialistas

| Especialista | vs denial | vs spread | vs greedy | Δ train vs otros |
|-------------|-----------|-----------|-----------|:----------------:|
| neat → denial | 0.635 ★ | 0.692 | 0.627 | -0.025 |
| neat → spread | 0.496 | 0.634 ★ | 0.499 | +0.136 |
| neat → greedy | 0.635 | 0.692 | 0.627 ★ | -0.036 |

### 4.2 PPO Especialistas

| Especialista | vs denial | vs spread | vs greedy | Δ train vs otros |
|-------------|-----------|-----------|-----------|:----------------:|
| ppo → denial | 0.598 ★ | 0.625 | 0.589 | -0.009 |
| ppo → spread | 0.531 | 0.650 ★ | 0.535 | +0.117 |
| ppo → greedy | 0.600 | 0.626 | 0.602 ★ | -0.011 |

### 4.3 DT Especialistas

| Especialista | vs denial | vs spread | vs greedy | Δ train vs otros |
|-------------|-----------|-----------|-----------|:----------------:|
| dt → denial | 0.589 ★ | 0.633 | 0.589 | -0.022 |
| dt → spread | 0.549 | 0.656 ★ | 0.548 | +0.107 |
| dt → greedy | 0.601 | 0.651 | 0.601 ★ | -0.025 |

### 4.4 Análisis de generalización

| Familia | Δ promedio | Veredicto |
|---------|:---------:|-----------|
| NEAT | +0.007 | ✅ Generaliza bien |
| PPO | +0.018 | ✅ Generaliza bien |
| DT | +0.010 | ✅ Generaliza bien |

**Hallazgos clave**:

1. **No hay overfitting** (Δ < 2% para todos). Los agentes generalizan contra oponentes no vistos.
2. **Spread es "fácil" para todos** (~0.63-0.67 independientemente de contra quién entrenaron).
3. **Los entrenados vs greedy son los mejores generalistas**: WR > 0.60 contra todo.
4. **NEAT\_vs\_greedy ≈ NEAT\_vs\_spread**: Ambos convergen a la misma solución no-informativa (~0.50).
5. **NEAT\_vs\_denial es el único NEAT diferenciador** (consistentemente > 0.55 contra todo).

---

## 5. NEAT Extended — Efecto de más entrenamiento

| Variante | WR vs greedy (decided) | Wins | Losses | n decididas |
|----------|:----------------------:|-----:|-------:|:-----------:|
| NEAT original (vs\_greedy) | 0.499 | 976 | 980 | 1,956 |
| NEAT extended (vs\_greedy) | **0.582** | 2,859 | 2,056 | 4,915 |

**Test z**: z = 6.23, p < 0.0001 → **Diferencia significativa**.

**Interpretación**: El NEAT original vs\_greedy **no convergió** (WR ≈ 0.50 = aleatorio). El extended **sí convergió** a una política útil (WR ≈ 0.58), similar al neat\_vs\_denial (WR ≈ 0.577). NEAT es sensible a la convergencia inicial y configuración, no a la cantidad de entrenamiento.

---

## 5b. NEAT Multi-seed vs Seed Rotation

Se compararon tres enfoques de entrenamiento NEAT para prevenir sobreajuste a la secuencia de dados:

| Enfoque | Qué hace | WR vs greedy | Generaciones | Tiempo |
|---------|----------|:------------:|:------------:|-------:|
| **Seed rotation** (primo 10007) | 1 seed base, rotada cada generación | 0.578 (extended, 1000 gen) | 1000 | ~100 min |
| **Multi-seed en selección** | 3 NEATs independientes (1 por seed), elige mejor | 0.624 | 300 × 3 | ~79 min |
| **Multi-seed real** | 1 NEAT evaluado contra 3 seeds simultáneamente | **0.626** | 300 | **~30 min** |

### Hallazgo clave

**Multi-seed real alcanza el mismo rendimiento (0.626) en 1/3 del tiempo (30 min vs 79 min)**. Evaluar cada genoma contra las 3 seeds simultáneamente fuerza generalización real a nivel de fitness, no solo post-selección. Se adoptó como método estándar para todos los NEAT.

### Mejora NEAT multi-seed vs modelos originales

| NEAT vs | Original (100 gen) | Multi-seed real (300 gen) | Mejora |
|---------|:------------------:|:------------------------:|:------:|
| denial | 0.565 | **0.635** | +7.0% |
| spread | 0.634 | **0.634** | ≈0% |
| greedy | 0.499 | **0.626** | +12.7% |

---

## 6. Conclusiones consolidadas para el informe

### 6.1 Hallazgos principales

1. **Superioridad estadística de NEAT con multi-seed**: El sistema NEAT alcanza WR = 0.653 (IC 95%: [0.645, 0.660]), significativamente superior a DT (0.608) y PPO (0.604). 6/9 comparaciones son significativas (p < 0.003 tras Bonferroni). La técnica de multi-seed real fue determinante.

2. **Equivalencia PPO-DT**: No hay diferencia significativa entre PPO y DT destilado (p > 0.5 en las 3 comparaciones). La destilación *Behavior Cloning* preserva >93% de fidelidad.

3. **DT destilado como opción práctica**: Winrate equivalente a PPO, inferencia más rápida, modelo interpretable, entrenamiento instantáneo.

4. **No hay overfitting**: Δ < 2% en evaluación cruzada para todas las familias.

5. **La adaptación KMeans tiene impacto limitado en este dominio**: Un generalista fijo iguala al sistema adaptativo. En NEAT, un solo especialista domina 4/4 clusters.

6. **El entrenamiento offline es viable**: WR ~60-65% contra heurísticos sin entrenamiento online, <33 min por modelo, sin GPU.

### 6.2 Comparativa de paradigmas

| Criterio | PPO | NEAT | DT (destilado) |
|----------|:---:|:----:|:--------------:|
| Winrate | 0.604 | **0.653** | 0.608 |
| Inferencia (5k juegos) | 44s | 28s | **33s** |
| Tamaño modelo | 161 KB | **3.4 KB** | 145 KB |
| Entrenamiento | ~10 min | ~30 min | **<3s** (+ PPO) |
| ¿Requiere tuning? | No | **Sí** | No |
| ¿Interpretable? | No | No | **Sí** |
| GPU necesaria | No | No | No |

### 6.3 Implicaciones

**Para Knucklebones**: NEAT con entrenamiento multi-seed ofrece el mejor rendimiento competitivo (WR = 0.653). Si se prioriza interpretabilidad, DT destilado es alternativa viable con WR equivalente a PPO.

**Para juegos más complejos**: El marco adaptativo (KMeans + especialistas + response table) es válido arquitectónicamente. La neuroevolución (NEAT) demostró ser competitiva cuando se resuelven sus problemas de convergencia mediante multi-seed. Para juegos con mayor diversidad estratégica se esperaría un beneficio real de la selección adaptativa.

---

## 7. Estado experimental

### 7.1 Completados ✅
- [x] Entrenamiento de 9 especialistas (3 PPO con 4 envs, 3 NEAT multi-seed real, 3 DT destilados)
- [x] KMeans perfilador (k=4)
- [x] Response tables para 3 sistemas
- [x] Evaluación de los 3 sistemas (5,000 partidas c/u) — NEAT superior (WR=0.653)
- [x] Evaluación cruzada (generalización/overfitting, 2,000 partidas c/u)
- [x] NEAT extended (convergencia)
- [x] Tests estadísticos formales (z-test, Bonferroni)
- [x] Ablation study (generalista vs adaptativo vs oráculo)
- [x] Análisis de dominancia en response tables
- [x] Multi-semilla real en fitness NEAT (3 seeds × 300 gen)
- [x] Comparativa multi-seed: rotation vs selección vs real
- [x] Tablas en tesis LaTeX actualizadas con resultados finales
- [x] Documentación consolidada

### 7.2 Siguiente fase
- [ ] Integración con motor Unity mediante middleware
- [ ] Evaluación con jugadores humanos
