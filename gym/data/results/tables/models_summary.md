| Family   | Artifact                                        | Opponent         |   Wall Time (s) |   Size (KB) | Accuracy   | Fitness   |
|:---------|:------------------------------------------------|:-----------------|----------------:|------------:|:-----------|:----------|
| rl       | PPO__vs_heuristic_denial                        | heuristic:denial |          1369.4 |       161.9 | -          | -         |
| rl       | PPO__vs_heuristic_greedy                        | heuristic:greedy |           399.8 |       161.9 | -          | -         |
| rl       | PPO__vs_heuristic_spread                        | heuristic:spread |           390.3 |       161.9 | -          | -         |
| neat     | neat__vs__heuristic_greedy__extended            | heuristic:greedy |          5973.6 |         2   | -          | 0.9610    |
| neat     | neat__vs_heuristic_denial                       | heuristic:denial |          1995.4 |         2.4 | -          | 0.8529    |
| neat     | neat__vs_heuristic_denial__multiseed_real       | heuristic:denial |          1995.4 |         2.4 | -          | 0.8529    |
| neat     | neat__vs_heuristic_denial__old_100gen           | heuristic:denial |           610.6 |         2.1 | -          | 0.8943    |
| neat     | neat__vs_heuristic_greedy                       | heuristic:greedy |          1818.1 |         3.6 | -          | 0.8598    |
| neat     | neat__vs_heuristic_greedy__multiseed_real       | heuristic:greedy |          1818.1 |         3.6 | -          | 0.8598    |
| neat     | neat__vs_heuristic_greedy__old_multiseed_select | heuristic:greedy |          1671.8 |         6.2 | -          | 0.8753    |
| neat     | neat__vs_heuristic_greedy__seed123              | heuristic:greedy |          1671.8 |         6.2 | -          | 0.8753    |
| neat     | neat__vs_heuristic_greedy__seed456              | heuristic:greedy |          1526.9 |         3.7 | -          | 0.8407    |
| neat     | neat__vs_heuristic_greedy__seed789              | heuristic:greedy |          1531   |         4.2 | -          | 0.8688    |
| neat     | neat__vs_heuristic_spread                       | heuristic:spread |          1867.9 |         4.2 | -          | 0.8741    |
| neat     | neat__vs_heuristic_spread__multiseed_real       | heuristic:spread |          1867.9 |         4.2 | -          | 0.8741    |
| neat     | neat__vs_heuristic_spread__old_100gen           | heuristic:spread |           748.2 |         2   | -          | 0.9464    |
| sklearn  | dt__ppo_vs_denial                               | nan              |             3.4 |       139.5 | 0.9476     | -         |
| sklearn  | dt__ppo_vs_greedy                               | nan              |             3.5 |       139.7 | 0.9275     | -         |
| sklearn  | dt__ppo_vs_spread                               | nan              |             4.3 |       162   | 0.9362     | -         |
| sklearn  | sklearn_tree__vs_heuristic_denial               | nan              |             2.2 |       134   | 0.9379     | -         |
| sklearn  | sklearn_tree__vs_heuristic_greedy               | nan              |             2   |       147.1 | 0.9442     | -         |
| sklearn  | sklearn_tree__vs_heuristic_spread               | nan              |             2   |       153.5 | 0.9238     | -         |
| kmeans   | kmeans__v2__k4__win6__seed123                   | nan              |            52.3 |       477.3 | -          | -         |
| kmeans   | kmeans_profiler                                 | nan              |            19.9 |       191   | -          | -         |