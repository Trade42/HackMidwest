# Cascade network, X memory, seed 1: kept snapshots and their scores on the kit's Stim shots

## Training

- Code: branch `cascade` of this repo, `python -m neural.train` from `challenge-1-ml-decoder-kit/`.
- Laptop GPU: RTX 4060 Laptop, 8 GB, torch 2.14.1+cu130, fp16.
- Training history:
  - f0a95fa: steps 0 to 10,000. The run was resumed once from the step-2,000 checkpoint after a display-driver reset.
  - 7ff6a68: from step 10,000, with `--steps 230000 --lr-decay-steps 230000 --report-shots 5000`.
- `train.log` is the run's own log up to the time of this commit.
- Speed: about 720 shots/s alone on the GPU at 7ff6a68, about 620 at f0a95fa.

| File | Bytes | sha256 |
| --- | --- | --- |
| `checkpoint_step20000.pt` | 14,667,276 | 478423b5500659db079e40b7b37ccacea584fbccd6a92c607bf0dc0c08d612ef |
| `checkpoint_step30000.pt` | 14,669,196 | d1f54ea15ef611f60f8e068046555a3c0c6ffb70167e77e857f91a95e20aecb9 |

`SHA256SUMS` lists the same. Load the decoder with `neural.train.load_decoder(path)`. It uses the moving-average
weights; the `config` in the file gives width 128, depth 8 and detectors=all.

## Scores on Stim shots: `eval/`

- Made with `python -m neural.evaluate_stim --checkpoint <snapshot> --data <file> --save eval/step<N>_<label>`.
- Run from the measurement worktree at commit 4d6ee8c, with the training venv, one file at a time beside the training.
- The run script is `cascade_eval_one.ps1`. It checks each data file's sha256 and starts only with at least 2 GiB of RAM
  available.

Per scoring: `<prefix>.log` is the printout, `<prefix>.json` the numbers, and `<prefix>.npz` the per-shot results in
file order (logits, predictions, observables, failed, confidence).

| Label | Data file | Shots | sha256 of the data |
| --- | --- | --- | --- |
| `validation_X_p0.004` | repo-main data/p0.004/validation_X.npz | 5,000 | b00ee09c35ed088a12bcba60b8b704b528c47fb156c7d09cf0f171f90a181f3d |
| `train_X_p0.004` | repo-main data/p0.004/train_X.npz | 20,000 | 98c0afeb2a0b6b27f3b689e832160a060a4c0aeb37d229f84daa1995a9a0f7cc |
| `gen_X_seed9203_p0.003` | laptop only, see ../priortilt/README.md | 50,000 | 4625e704682b3a6ef5c465eec33479d04f3d6f9a62912d4e4ce176e51bf7725e |
| `gen_X_seed9205_p0.005` | laptop only, see ../priortilt/README.md | 20,000 | 45a9761e688088c0e502e80a5569258985242e40231eea7f474f7887188213eb |

Failures per shot, X memory:

| Step | p = 0.003 | p = 0.004 validation | p = 0.004 train | p = 0.005 |
| --- | --- | --- | --- | --- |
| 20,000 | 15.97% | 42.12% | - | 73.03% |
| 30,000 | 8.31% | 26.60% | 27.00% | 55.89% |

- Batch-one latency was about 18-20 ms, measured beside the training, so it is inflated.
- Any confidence threshold read from these tables is fitted to that file. Freeze it on one file and report it on
  another.
- No test_* or go_* file was scored.
