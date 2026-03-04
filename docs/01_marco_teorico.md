# Documento 1 — Marco Teórico y Justificación
> **Tesis**: *Rendimiento y desempeño del aprendizaje de distintas inteligencias artificiales en un entorno de videojuego offline*

---

## 1. Problema de investigación

La mayoría de flujos de IA en videojuegos dependen de comportamientos estáticos (máquinas de estados, scripts) o de infraestructura externa (servidores cloud, entrenamiento distribuido). El "aprendizaje online pleno" — actualizar parámetros mientras se juega — es costoso, inestable y difícil de reproducir en hardware local.

**Pregunta central**: ¿Es posible lograr adaptabilidad real de la IA en un videojuego ejecutado completamente offline, combinando entrenamiento previo con adaptación en tiempo de ejecución?

## 2. Hipótesis operativa

La **adaptación online por selección** (player modeling + gating) puede entregar adaptabilidad real con menor costo que el aprendizaje online pleno, mediante:

1. Agentes fuertes entrenados offline (PPO, NEAT)
2. Destilación a modelos livianos e interpretables (Decision Trees)
3. Perfilado del oponente con KMeans (ventana temporal)
4. Selección del mejor especialista por cluster (Response Table)

## 3. Antecedentes en la industria

### 3.1 El caso ARC Raiders
Se promocionó que sus enemigos exhibían comportamientos emergentes, dando la impresión de "aprender en tiempo real". La investigación reveló que Embark Studios usó ML **durante el desarrollo** para entrenar locomoción y navegación, pero la IA final no aprende durante gameplay. Los comportamientos ya fueron aprendidos previo al lanzamiento.

### 3.2 Alien: Isolation (2014)
El Xenomorfo no entrena un modelo, pero detecta patrones del jugador y desbloquea contra-medidas predefinidas. Si el jugador se esconde mucho bajo mesas, el Alien busca más debajo. Adaptación por reglas, no por ML.

### 3.3 Metal Gear Solid V (2015)
Soldados reaccionan a tendencias del jugador: muchos headshots → cascos; jugar de noche → visores nocturnos. Reglas adaptativas sin ML.

### 3.4 Conclusión de antecedentes
El aprendizaje en tiempo real no es viable en hardware local para juegos comerciales. La adaptación por reglas funciona pero está limitada a situaciones previstas. **Nuestro enfoque combina lo mejor de ambos mundos**: agentes entrenados offline + selección adaptativa en runtime.

## 4. Técnicas de IA seleccionadas

| Nº | Técnica | Implementación | Rol | Motivo académico |
|----|---------|----------------|-----|------------------|
| 1 | Árboles de Decisión | scikit-learn | Inferencia barata, interpretable, baseline | Referencia a IA clásica |
| 2 | Aprendizaje por Refuerzo | Stable Baselines3 (PPO) | Agente fuerte, teacher para destilación | Demuestra RL aplicado |
| 3 | Neuroevolución | NEAT-Python | Paradigma alternativo, estrategias emergentes | Explora evolución vs gradientes |
| 4 | Clustering | scikit-learn (KMeans) | Perfilado de oponente, selección de especialista | Adaptación sin reentrenamiento |

### 4.1 ¿Por qué estas 3 familias?

- **DT (Decision Tree)**: Inferencia en microsegundos, explicable, ideal para deployment. Limitación: no aprende solo, necesita teacher.
- **PPO (Reinforcement Learning)**: Aprende políticas fuertes por refuerzo. Estable, well-documented. Costo: entrenamiento largo, modelo pesado.
- **NEAT (Neuroevolución)**: Evoluciona estructura + pesos sin gradientes. Puede encontrar soluciones creativas. Costo: convergencia lenta, CPU intensivo.

### 4.2 ¿Por qué KMeans?
- No requiere datos etiquetados
- Computacionalmente ligero (milisegundos)
- Permite player modeling online: inferir estilo del oponente durante una sesión
- Habilita gating: cluster → mejor respuesta

## 5. Distinción clave: Aprendizaje vs Adaptación

### 5.1 Aprendizaje online pleno
Actualización de parámetros/pesos **mientras se juega**. Ventaja: mejora sostenida. Riesgo: costo computacional, inestabilidad, irreproducibilidad.

### 5.2 Adaptación online (nuestro enfoque)
No se actualizan pesos, pero se ajusta el comportamiento en runtime mediante perfilado (ventana de acciones), clustering (KMeans), y selección de especialista (response table).

## 6. Entorno experimental: Knucklebones

Juego de mesa de Cult of the Lamb. Elegido por:
- Reglas simples pero con profundidad estratégica
- Turnos discretos (fácil de simular)
- Estado completamente observable
- Espacio de acciones pequeño (3 columnas)
- Permite miles de partidas por segundo en simulación

### 6.1 Reglas
- 2 jugadores, tablero 3×3 cada uno
- Cada turno: tirar dado (1-6), elegir columna (0-2)
- Score por columna: Σ(v × count(v)²)
- Destrucción: colocar v en columna c elimina v del rival en su columna c
- Fin: cuando un jugador llena su tablero (9 casillas)

## 7. Glosario

| Término | Definición |
|---------|-----------|
| Policy (política) | Función que mapea observación → acción |
| Spec | String que describe una policy (ej. `heuristic:denial`) |
| Behavior Cloning | Entrenamiento supervisado imitando un teacher |
| Destilación | Transferencia de comportamiento de modelo complejo a liviano |
| Gating | Selección de especialista en runtime según perfil del oponente |
| Response Table | Mapeo cluster → policy_spec |
| Swap roles | Evaluación como p0 y p1 para reducir sesgo |
| Features | Vector numérico de 21 dimensiones que describe el estado del juego |
| PPO | Proximal Policy Optimization (algoritmo de RL) |
| NEAT | NeuroEvolution of Augmenting Topologies |
| DT | Decision Tree (Árbol de Decisión) |
| BC | Behavior Cloning |
| SB3 | Stable-Baselines3 |

## 8. Referencias

[1] Unity Technologies, "ML-Agents Toolkit Documentation," 2025. https://github.com/Unity-Technologies/ml-agents
[2] K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through Augmenting Topologies," *Evolutionary Computation*, vol. 10, no. 2, pp. 99–127, 2002.
[3] Scikit-learn Developers, "User Guide: Clustering," 2023. https://scikit-learn.org
[4] GamerPulse, "AI Machine Learning in Arc Raiders," Medium, 2025.
[5] Reddit r/ArcRaiders, "Will the ARC be learning off the players," 2025.
[6] Kraj, N., "Designing an AI Decision Tree," GDKeys, 2023.
[7] Malte Skarupke, "Why Video Game AI does not use Machine Learning," 2020.
[8] Smartico.ai, "IA predictiva y adaptación en tiempo real," 2023.

