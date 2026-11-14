# Documento 6 — Trabajo relacionado

Posicionamiento del proyecto frente a la literatura. Los resultados propios están en [`03_resultados_experimentales.md`](03_resultados_experimentales.md).

**Sobre Knucklebones específicamente**: sin tratamiento académico previo (confianza baja: ausencia en una búsqueda dedicada, no prueba de inexistencia). La integración tri-paradigma, la cuantificación de la brecha de datos y la adaptación por selección, todo en hardware local sin GPU, no aparece replicada en ninguna fuente encontrada.

**Fuera de la academia** sí hay trabajo previo sobre el juego: un solver que indica la mejor jugada y la probabilidad de ganar bajo juego perfecto[^13], y un proyecto que compara MCTS, Q-Learning y DQN sin reportar resultados[^14]. Ninguno aborda la brecha de datos ni la adaptación por selección; el solver sí ofrece una cota de juego perfecto contra la cual medir a estos agentes, lo que queda como trabajo futuro.

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
[^13]: CyberShadow. Solver de Knucklebones (Cult of the Lamb). [cy.md/ta/cult-of-the-lamb/knucklebones](https://cy.md/ta/cult-of-the-lamb/knucklebones/), [código](https://gitlab.com/CyberShadow/cult-of-the-lamb-knucklebones)
[^14]: Mia Müßig. Knucklebones-AI. [github.com/PhoenixSmaug/Knucklebones-AI](https://github.com/PhoenixSmaug/Knucklebones-AI)
