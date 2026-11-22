# hierarchy: two fast decodes, and a tie-breaker where they disagree

Work in progress: the candidates tool, the offline scorer and the file comparison are here.
The plugin for the team's evaluator, the tests and the numbers follow in this folder.

Run every command from `challenge-1-ml-decoder-kit/`.

## Install

The kit's environment (Python 3.12, `requirements.txt`), plus PriorTilt (Ye and Delfosse, IonQ,
arXiv:2611.13373, https://github.com/ionq-publications/PriorTiltDecoder). PriorTilt's licence is
CC BY-NC-SA 4.0. None of its code is in this repository: it is installed as a dependency from
a clone, and its tuned schedule is read from that clone at run time.

```sh
git clone https://github.com/ionq-publications/PriorTiltDecoder.git ~/PriorTiltDecoder
git -C ~/PriorTiltDecoder checkout 8eefd37d694b5942fe60ec6d5c82ce97f3d2a450
python3.12 -m venv ~/hierarchy-venv
~/hierarchy-venv/bin/pip install -r requirements.txt ~/PriorTiltDecoder/decoder    # needs a Rust toolchain (cargo)
export PRIORTILT_REPO=~/PriorTiltDecoder
export PYTHONDONTWRITEBYTECODE=1
```

## Decode shot files on any machine

```sh
~/hierarchy-venv/bin/python -m hierarchy.candidates --data <folder with shot files> --out <output folder> --processes 12
```

- `--data` takes shot files of `make_dataset.py`, or folders. In a folder, files named
  `test_*` or `go_*` are left alone. Named directly, they are refused without `--final`.
- Every shot is decoded by relay-BP (the kit's settings) and by PriorTilt under two seeds, in
  chunks of 500 shots on fresh decoders. The answers do not depend on `--processes`.
- Every finished chunk is saved under `<output folder>/parts/`. A stopped run continues where
  it stopped when the same command runs again.
- Per shot file the tool writes `<output folder>/<name>.chunks.npz` and prints two tables: the
  three decoders, and the rules of the hierarchy that need no network.

Score the rules again later, or with the network's logits:

```sh
~/hierarchy-venv/bin/python -m hierarchy.select --candidates <output folder>/<name>.chunks.npz
```

Check that two machines got the same answers:

```sh
~/hierarchy-venv/bin/python -m hierarchy.compare <one>.npz <other>.npz
```
