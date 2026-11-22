**Exploration shots, not test shots: the predecoder's time per X shot.** The team's server, one process and one thread (server jobs cpu/050 and cpu/060), the first 1,000 X shots of `data/p<p>/validation_X.npz` (so no test file was decoded twice); the head and the final X network pinned by sha256 as in the test run. Per X shot, not per pair; ms: mean / median / 99th percentile.

| p | The network and the head, on the CPU | Relay-BP with the head's priors, built and run per shot | Relay-BP converged (mean ms) | The two together, mean ms per X shot |
|---|---|---|---|---|
| 0.003 | 26.4 / 26.3 / 27.5 | 26.5 / 19.0 / 56.7 | 999 of 1,000 (21.0 ms); not converged 1 (5,515 ms) | 52.9 |
| 0.004 | 26.7 / 26.6 / 27.3 | 67.3 / 23.1 / 631 | 997 of 1,000 (51.0 ms); not converged 3 (5,512 ms) | 94.0 |
| 0.005 | 26.8 / 26.7 / 28.0 | 185 / 30.1 / 5,467 | 992 of 1,000 (142 ms); not converged 8 (5,544 ms) | 212 |

For scale only, other shots and so no paired comparison, on the test X shots: p = 0.003: relay-BP's timed row 44.3 ms per X shot (its 85.3 ms per pair is an X shot plus a Z shot), TCBP-JOBS 2.0 ms per X shot; p = 0.004: relay-BP's timed row 156 ms per X shot (its 319 ms per pair is an X shot plus a Z shot), TCBP-JOBS 3.9 ms per X shot; p = 0.005: relay-BP's timed row 760 ms per X shot (its 1,471 ms per pair is an X shot plus a Z shot), TCBP-JOBS 8.7 ms per X shot (TCBP-JOBS: a GPU decode call alone, not comparable with a CPU time).

Source: `report/data/exploration/predecoder/latency_validation_X_p<p>/` (`summary.json`, `per_shot.npz`, `latency.txt`) and the jobs' printouts `job_cpu_050-predecoder-latency.out`, `job_cpu_060-predecoder-latency-p0.003-p0.004.out`.
