| Decoder | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|
| BP-OSD | 60 · 131 ms | 472 · 302 ms | 2,433 · 581 ms |
| relay-BP | 5 · 85.3 ms | 83 · 319 ms | 845 · 1,471 ms |
| router (teammates) | 6 · 41.0 ms | 87 · 315 ms | 807 · 1,928 ms |
| PriorTilt (IonQ) | 4 · 22.7 ms | 74 · 87.3 ms | 601 · 264 ms |
| ours, 1 core | 3 · 45.7 ms | 44 · 216 ms | 479 · 825 ms |
| ours, 2 cores | 3 · 28.9 ms | 44 · 150 ms | 479 · 608 ms |
| TCBP-JOBS (GPU) | 3 · 3.9 ms | 41 · 7.8 ms | 360 · 16.9 ms |
| ours, ensemble | 2 · offline | 16 · offline | 360 · offline |

Failed pairs of 10,000 test pairs · mean time per pair. "Offline": combined from saved answers, no time. TCBP-JOBS runs on a GPU.
