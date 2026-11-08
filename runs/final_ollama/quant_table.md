| | Q8_0 | Q4_K_M |
|---|---|---|
| Correct | 17/20 | 15/20 |
| Status | ok 20 | ok 19, max_steps 1 |
| Median wall s / question | 30.0 | 23.0 |
| Median model-wait s / question | 30.0 | 22.9 |
| Mean model calls / question | 5.50 | 6.45 |
| Mean prompt / output tokens | 32,918 / 1,005 | 45,071 / 1,189 |
| Output tok/s seen by the agent | 28.8 | 37.2 |
| Prefill tok/s (native API, cache defeated) | 1552.9 | 1458.8 |
| Decode tok/s (native API) | 32.8 | 42.8 |
| Load time s | 4.6 | 4.0 |
| Server memory GB (weights + KV cache, /api/ps) | 7.25 | 5.61 |
| Of which on GPU GB | 7.25 | 5.61 |
| nvidia-smi used GB | [7.52, 6.0] | [6.0, 6.0] |
| One call, cold prefix / warm prefix (s) | 7.37 / 4.01 | 4.89 / 1.3 |

Paired on 20 questions: only Q8_0 right 3, only Q4_K_M right 1, exact McNemar p = 0.625 (20 questions in the Q8_0 run).
