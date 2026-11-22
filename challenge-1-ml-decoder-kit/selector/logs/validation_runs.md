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
