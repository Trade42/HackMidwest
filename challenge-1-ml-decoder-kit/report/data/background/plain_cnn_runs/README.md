# Plain-CNN runs on the laptop GPU (2026-11-21)

These are `ml_decoder/train_logical_cnn.py` from branch MLDecoder at commit 0eab7dd, X memory, on the RTX 4060 Laptop
GPU, run from `challenge-1-ml-decoder-kit/`. Each run has its own files:

- `<run>.log`: the trainer's stdout. Report lines are on 20,000 fresh DEM-sampled shots per p.
- `<run>.summary.txt`: wall time, exit code, GPU memory peak and utilisation.
- `<run>.gpu.csv`: nvidia-smi samples every 2 s. Columns: timestamp, memory.used MiB, utilisation %, temperature C, power W.

`RESULTS_laptop.md` is the laptop's running record of every run, in time order. It also covers the later cascade runs.

| Run | Command (after `python ml_decoder/train_logical_cnn.py`) | Parameters | Outcome |
| --- | --- | --- | --- |
| `x_check3min` | `--basis X --minutes 3 --width 32 --depth 2 --batch 256` | 0.7 M | 5.8 M shots in 180 s; at p = 0.003, 54.72% of bits right (always-0 50.51%) |
| `x_4h` (run 1) | `--basis X --hours 4 --out model_X.pt` (defaults: width 128, depth 6, batch 2048) | 5.6 M | Stopped at 0.93 h. It equalled the always-0 guess at every p, also at p = 0.0005 (`x_4h.ckpt_0.83h.eval_lowp.log`). About 2,500 shots/s, about 3,600 optimizer steps by 0.83 h |
| `x_1h_w32` (run 2) | `--basis X --hours 1 --width 32 --depth 2 --batch 256 --p-report 0.0005,...,0.005` | 0.7 M | 117.3 M shots; at p = 0.003, 90.55% of shots wrong and 67.76% of bits right |
| `x_3h_w32_resume` (3B) | resume of run 2, `--p-train 0.003,0.004,0.005 --hours 3 --batch 256` | 0.7 M | Stopped at 0.39 h (replaced by the cascade network); at p = 0.003, 91.24% wrong |
| `x_1h_w64` (3A) | `--basis X --hours 1 --width 64 --depth 2 --batch 256 --p-report ...` | 1.0 M | Stopped at 0.70 h; at p = 0.003, 88.96% wrong and 69.50% of bits right |

`x_check3min.eval_lowp.log` is the 3-minute model evaluated at p = 0.0005 to 0.003 with `--eval` (`--p-report
0.0005,0.001,0.002,0.003 --shots 20000`). It gets 96.26% of bits right at p = 0.0005, against 73.20% for the always-0
guess.

The model checkpoints of these runs stay on the laptop (`runs\*.pt`) and are not in this branch.