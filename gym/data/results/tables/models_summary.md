| Family   | Artifact                      | Opponent         |   Wall Time (s) |   Size (KB) | Accuracy   | Fitness   |
|:---------|:------------------------------|:-----------------|----------------:|------------:|:-----------|:----------|
| rl       | PPO__vs_heuristic_denial      | heuristic:denial |           783.6 |       161.5 | -          | -         |
| rl       | PPO__vs_heuristic_greedy      | heuristic:greedy |           650.5 |       161.5 | -          | -         |
| rl       | PPO__vs_heuristic_spread      | heuristic:spread |           734.6 |       161.5 | -          | -         |
| neat     | neat__vs_heuristic_denial     | heuristic:denial |          1600.6 |         1.4 | -          | 0.8802    |
| neat     | neat__vs_heuristic_greedy     | heuristic:greedy |          1624.9 |         3.2 | -          | 0.8729    |
| neat     | neat__vs_heuristic_spread     | heuristic:spread |          2104   |         1.3 | -          | 0.9349    |
| sklearn  | dt__ppo_vs_denial             | nan              |             3.4 |       139.5 | 0.9476     | -         |
| sklearn  | dt__ppo_vs_greedy             | nan              |             3.5 |       139.7 | 0.9275     | -         |
| sklearn  | dt__ppo_vs_spread             | nan              |             4.3 |       162   | 0.9362     | -         |
| kmeans   | kmeans__v2__k4__win6__seed123 | nan              |            52.3 |       477.3 | -          | -         |