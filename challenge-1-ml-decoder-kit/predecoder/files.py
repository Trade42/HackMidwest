"""Small helpers about files, without torch, so that the decoding worker processes stay small.

compare.py and priors.py start one worker process per core; each imports the module that
started it. Keeping torch out of these imports saves about 600 MB per worker.
"""
import json
from pathlib import Path

HELD_OUT = ("test_", "go_")       # the team's held-out shot files, by the start of their name


def read_logits(folder, data):
    """(meta, path of logits.npy) of predecoder.evaluate's output, after checking it belongs to this shot file."""
    folder = Path(folder)
    with open(folder / "meta.json") as file:
        meta = json.load(file)
    if meta["data"]["sha256"] != data.sha256:
        raise SystemExit(f"{folder} holds the logits of another shot file ({meta['data']['path']})")
    return meta, folder / "logits.npy"
