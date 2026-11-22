# Every scoring of a validation file by the selector

One line per run of `selector.evaluate` on validation shots, in the order they ran. A model is
scored on validation once per frozen fit; a re-run of the same model on the same files is listed too.

| UTC | model file | model sha256 | shots | log | note |
|---|---|---|---|---|---|
| 2026-11-22 08:34 | `models/p0.004.json` | `164990fd...9b5b31` | p = 0.004 validation_X, validation_Z | `validation_p0.004.log` | first run 08:34:15, re-run 08:34:52 only to fix the log's header (identical numbers) |
| 2026-11-22 08:37 | `models/p0.004.json` | `164990fd...9b5b31` | p = 0.003 validation_X, validation_Z (the p = 0.004 model unchanged) | `validation_p0.003.log` | 1 failed pair, as T and f: says nothing either way |
| 2026-11-22 09:09 | `models/p0.005.json` | `696bf544...698ce6` | p = 0.005 validation_X, validation_Z | `validation_p0.005.log` | the pre-registered single scoring: kill line passed (172 against 187 failed pairs; 5 / 20, p = 0.004) |
| 2026-11-22 09:10 | `models/p0.005.json` | `696bf544...698ce6` | the same | `shuffle_check.log` | leakage check (truth moved between shots): no feature, choice or answer changed; also for `models/p0.004.json` on p = 0.004 validation_X |
| 2026-11-22 09:10 | `models/p0.005.json` | `696bf544...698ce6` | the same | `validation_p0.005.log` (replaced) | re-run after adding comparison lines to evaluate.py (gw and f against T; per shot; double failures): every earlier number unchanged |

## Which code made each log (the review's finding 1c)

A log's header names the commit HEAD was at when it ran. Where the code that ran was not yet committed,
this says where it was committed. The numbers do not depend on it: the review re-ran every log at
`b8608de` and got them back.

| log | header | the code that ran |
|---|---|---|
| `fit_p0.004.log`, `validation_p0.004.log` | bc3b7d4 | `selector/` as committed in 161d3da a minute later |
| `validation_p0.003.log` | 161d3da | 161d3da (`evaluate.py` gained its comparison lines later, in 0d8f38e) |
| `check_data.log`, `fit_p0.005.log` | 9ed2447 | 9ed2447 |
| `shuffle_check.log` | cdeef70 | cdeef70, with `shuffle_check.py` as committed in 0d8f38e |
| `validation_p0.005.log` | cdeef70 | cdeef70, with `evaluate.py` as committed in 0d8f38e (the first run, 09:09:23, was cdeef70 itself) |
| `cv_p0.004_without_J.log`, `pair_wilson_p0.005.log` | b8608de | b8608de, the frozen code (the fit's model went to scratch, not to `models/`) |

## The run on the test shots

| UTC | model file | model sha256 | shots | log | note |
|---|---|---|---|---|---|
| 2026-11-22 12:27 | `models/p0.005.json` | `696bf544...698ce6` | p = 0.005 test_X, test_Z (10,000 each) | `test_p0.005.out` | run once by the orchestrator (server job ctl/110, from b8608de): 353 against 360 failed pairs, 23 / 30, p = 0.41: the kill line is not met |
