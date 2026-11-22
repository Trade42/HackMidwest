"""Sample shots of the kit's memory circuit under the neutral-atom-shaped model, and save them.

  python -m neutral_atom_noise.make_shots --basis X --factor 0.3 --shots 2000 --seed 11 --out OUTDIR
  python -m neutral_atom_noise.make_shots --basis X --uniform-p 0.004 --shots 2000 --seed 11 --out OUTDIR
  python -m neutral_atom_noise.make_shots --basis X --factor 0.1 --loss --shots 2000 --seed 11 --out OUTDIR

Run it from challenge-1-ml-decoder-kit/. Without --loss the shots come from Stim's detector sampler on
`model.model_circuit`. With --loss the same circuit also loses atoms (model.Noise.loss per atom per CX) and is
simulated by Clifft through `loss/`'s circuit text and readout model; the file then also holds the heralds.
--uniform-p samples the kit's own uniform model instead, as a reference.

The file holds `detectors` (all 1728 per shot, bit-packed), `observables` (12 per shot) and `meta`. A seed gives
the same shots only on the same machine and library versions, so the sha256 of the file is printed: quote it.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import stim

from circuit import sample
from neutral_atom_noise.model import Noise, kit_uniform, model_circuit, neutral_atom


def save_shots(path, detectors, observables, meta, heralds=None):
    """Write a shot file and return its sha256."""
    arrays = {"detectors": np.packbits(detectors.astype(bool), axis=1), "observables": observables.astype(np.uint8),
              "meta": np.array(json.dumps(meta))}
    if heralds is not None:
        arrays["heralds"] = np.packbits(heralds.astype(bool), axis=1)
    np.savez_compressed(path, **arrays)
    return sha256(path)


def load_shots(path):
    """Read a shot file: a dict with detectors, observables, meta and, for loss shots, heralds."""
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        shots = {"meta": meta, "observables": f["observables"],
                 "detectors": np.unpackbits(f["detectors"], axis=1, count=meta["num_detectors"])}
        if "heralds" in f.files:
            shots["heralds"] = np.unpackbits(f["heralds"], axis=1, count=meta["num_records"])
    shots["noise"] = Noise(**meta["noise"])
    return shots


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_loss(circuit, loss, shots, seed, chunk=2000):
    """Clifft shots of `circuit` with LOSS(loss) on both atoms after every CX, in chunks; chunk k uses seed + k.

    Uses loss/ as it is: `clifft_text` writes the circuit, `sample_chunk` runs it with loss/'s readout model
    (a lost atom is heralded at its readout and reads a random bit; a reset restores a lost check atom).
    Returns detectors, observables and the heralds in the kit's record order.
    """
    from loss.lossmodel import clifft_text, sample_chunk
    text, _, records = clifft_text(circuit, loss, probes=False)
    parts = [sample_chunk(text, min(chunk, shots - start), seed + k) for k, start in enumerate(range(0, shots, chunk))]
    return (np.concatenate([np.asarray(part.detectors, np.uint8) for part in parts]),
            np.concatenate([np.asarray(part.observables, np.uint8) for part in parts]),
            np.concatenate([np.asarray(part.heralds, np.uint8)[:, records] for part in parts]))


def shot_name(basis, factor, uniform_p, loss, seed):
    noise = f"p{uniform_p:g}" if uniform_p is not None else f"f{factor:g}"
    return f"{basis}_{noise}{'_loss' if loss else ''}_seed{seed}"


def make(basis, factor, uniform_p, loss, shots, seed, out):
    """Sample and save one shot file in the folder `out`, unless it is already there. Returns its path."""
    out = Path(out)
    path = out / (shot_name(basis, factor, uniform_p, loss, seed) + ".npz")
    if path.exists():
        kept = load_shots(path)["meta"]
        if kept["shots"] != shots:
            raise SystemExit(f"{path} holds {kept['shots']} shots, not {shots}: use another seed or folder")
        print(json.dumps({"file": path.name, "sha256": sha256(path), "kept": True}), flush=True)
        return path
    noise = kit_uniform(uniform_p) if uniform_p is not None else neutral_atom(factor)
    circuit = model_circuit(basis, noise)
    meta = {"basis": basis, "factor": factor, "uniform_p": uniform_p, "noise": noise.as_dict(), "loss": bool(loss),
            "shots": shots, "seed": seed, "num_detectors": circuit.num_detectors,
            "num_records": circuit.num_measurements, "machine": platform.node(), "stim": stim.__version__}
    heralds = None
    if loss:
        import clifft
        meta["sampler"], meta["clifft"] = "clifft noncomp, chunks of 2000 shots, chunk k uses seed + k", clifft.__version__
        detectors, observables, heralds = sample_loss(circuit, noise.loss, shots, seed)
    else:
        meta["sampler"] = "stim detector sampler"
        detectors, observables = sample(circuit, shots, seed)
    out.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    save_shots(partial, detectors, observables, meta, heralds)
    partial.replace(path)
    report = {"file": path.name, "sha256": sha256(path),
              **{k: meta[k] for k in ("basis", "factor", "uniform_p", "loss", "shots", "seed", "machine")},
              "detection_events_per_shot": round(float(detectors.sum(axis=1).mean()), 2)}
    if heralds is not None:
        report["heralds_per_shot"] = round(float(heralds.sum(axis=1).mean()), 2)
    print(json.dumps(report), flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--basis", choices=["X", "Z"], default="X", help="memory basis")
    parser.add_argument("--factor", type=float, default=None, help="scale of the model; 1 = the published figures")
    parser.add_argument("--uniform-p", type=float, default=None, help="sample the kit's uniform model at this p instead")
    parser.add_argument("--loss", action="store_true", help="also lose atoms at the model's loss rate (Clifft)")
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True, help="output folder")
    args = parser.parse_args()
    if (args.factor is None) == (args.uniform_p is None):
        parser.error("give exactly one of --factor and --uniform-p")
    if args.loss and args.factor is None:
        parser.error("--loss needs the model (--factor): the kit's uniform model has no loss")
    make(args.basis, args.factor, args.uniform_p, args.loss, args.shots, args.seed, args.out)


if __name__ == "__main__":
    main()
