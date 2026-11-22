# Every scoring of a validation file by the selector

One line per run of `selector.evaluate` on validation shots, in the order they ran. A model is
scored on validation once per frozen fit; a re-run of the same model on the same files is listed too.

| UTC | model file | model sha256 | shots | log | note |
|---|---|---|---|---|---|
| 2026-11-22 08:34 | `models/p0.004.json` | `164990fd...9b5b31` | p = 0.004 validation_X, validation_Z | `validation_p0.004.log` | first run 08:34:15, re-run 08:34:52 only to fix the log's header (identical numbers) |
