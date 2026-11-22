Checks made by `python -m report.make_tables` on this run:

- **Rows on the test shots:** 6 of 27 registered rows have a file under `report/data/test/`; the others print "not measured".
- **Same shots for every row:** each measured row names the sha256 of the test files it decoded, and the run stops unless they are the team's `data/p<p>/test_X.npz` and `test_Z.npz` (the values are in `report/rows.py`). A paired count is only made between such rows.
- **The rate uses pairs and 12 cycles:** for each evaluator row the per-shot file is re-counted (X failures, Z failures, pair = X or Z) and the rate per cycle is recomputed as 1 - (1 - P)^(1/12); the run stops if this differs from the evaluator's `summary.json`.
- **Times are per pair:** mean, median and p99 are recomputed from `seconds_total_pair` (the whole decode of one X shot plus one Z shot, every stage and rerun included) and must equal the evaluator's summary.
- **The paired count agrees with the team's `evaluation/compare_gain.py`:** `report/test_report.py` recomputes the two committed comparisons (`results/icrn/gain_p0.003_portable`, `results/mac/gain_p0.004`) from the per-shot files and gets the same counts.
- **Cross-check with the teammates' earlier runs.** p = 0.003: relay-BP's timed row and the teammates' earlier run (ICRN server, before the quiet window) give the same outcome on every one of the 10,000 pairs.
- **Cross-check with the teammates' earlier runs.** p = 0.004: relay-BP's timed row and the teammates' earlier run (a teammate's Mac (arm64)) give a different outcome on 65 of the 10,000 pairs (83 against 84 failed pairs).
