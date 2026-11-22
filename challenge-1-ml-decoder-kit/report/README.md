# report: the comparison, made by scripts from committed result files

Start with `REPORT.md`. Run every command from `challenge-1-ml-decoder-kit/`.

| file | what it is |
|------|------------|
| `REPORT.md` | the write-up: the question, the approach, the results on the test shots, loss, what did not work, resources |
| `APPENDIX.md` | the full tables behind the write-up |
| `SLIDES.md` | the material for a five-minute talk, and the questions a judge will ask |
| `REPRODUCE.md`, `reproduce_ours.sh` | every command from a fresh clone to our rows of the table, with what was run where |
| `make_tables.py` | writes `tables/*.md` and `*.csv`, and the tables and numbers inside `REPORT.md` and `SLIDES.md` |
| `make_plots.py` | writes `figures/*.png` and `*.svg` |
| `rows.py` | the list of decoders (name, colour, whose work) and where each result file is expected |
| `loaders.py`, `stats.py` | reading and checking the result files; the rate per cycle, the intervals, the paired counts |
| `sync_data.py` | the team's tool that copied the result files into `data/` and wrote `data/SOURCES.md` |
| `test_report.py` | tests of the arithmetic and of both scripts |
| `data/` | the copied result files; `data/SOURCES.md` says where each came from, with its sha256 |
| `tables/`, `figures/` | what the two scripts write; `tables/facts.md` lists every single number quoted in the prose |

```sh
python -m report.make_tables      # the tables
python -m report.make_plots       # the figures
python -m pytest -p no:cacheprovider report
```

Rules the scripts enforce:

- A row whose file is not under `data/` is printed as "not measured". No number comes from anywhere else.
- A row of a test table must name the team's saved test files by sha256, or the run stops.
- Each evaluator row is re-counted from its per-shot file (failures of X, of Z, pairs; the rate per cycle; the times
  per pair), and the run stops if that disagrees with the evaluator's own summary.
- Test shots and exploration shots never share a table.
