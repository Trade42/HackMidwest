"""The TCBP pipeline on the neutral-atom-shaped noise: TCBP-JOBS with generic, matched and head priors.

Exploration, after the main test results, on the branch infleqtion-noise. The arms, the pairs and the stopping rule
are declared in tcbp_check_declared.md, written before any number of this tool existed.

Three steps, run from challenge-1-ml-decoder-kit/:

  convert   on luminaty (Stim, this branch): the shot file of make_shots.py in the kit's format (what
            predecoder.evaluate and evaluation.data.load_dataset read), the generic and the matched prior vectors,
            and transfer.json (the hash of H and L, the noise, the vectors' ranges).
              python -m neutral_atom_noise.tcbp_check convert --shots <X_f0.8_seed4100.npz> --p 0.0326371 --out <folder>

  decode    on the server, in the teammates' TCBP environment, with predecoder-tcbp's challenge-1-ml-decoder-kit/
            first on PYTHONPATH (it imports predecoder.tcbp; nothing of neutral_atom_noise). Blocks of 100 shots, the
            arms m, g, h one after the other inside each block; no new block after --stop-at.
              python -m neutral_atom_noise.tcbp_check decode --data <converted.npz> --generic <generic.npy> \\
                  --matched <matched.npy> --transfer <transfer.json> --logits <predecoder.evaluate folder> \\
                  --settings predecoder/logs/fit_tcbp_X_p0.005.json --stop-at 2026-11-22T17:12:00Z --out <folder>

  score     on luminaty, from decode's per_shot.npz and this branch's BP-OSD and relay-BP result files: failures with
            Wilson intervals on the scored prefix, and the exact two-sided sign tests of the declared pairs.
              python -m neutral_atom_noise.tcbp_check score --per-shot <per_shot.npz> --shots <X_f0.8_seed4100.npz> \\
                  --results <run folder>/results --out <summary.json>
"""
import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ARMS = ("m", "g", "h")                    # the declared order inside each block
BLOCK = 100
PRIOR_CAP = 0.5                           # TCBP-JOBS refuses a prior above 0.5 (predecoder/tcbp.py)
FLOOR_FALLBACK = 1e-6                     # declared: used for the matched arm only if TCBP-JOBS refuses 1e-9
FROZEN = {"mode": "logodds", "alpha": 0.4, "eps": 1e-4}
FIELDS = ("failed", "reproduced", "flag", "n_faults", "seconds_build", "seconds_decode", "predicted_code")
BITS = 1 << np.arange(12)


def matrix_hash(H, L):
    """sha256 of H and L in CSC form (shape, column pointers, row indices): it fixes the column order."""
    digest = hashlib.sha256()
    for matrix in (H, L):
        csc = matrix.tocsc()
        csc.sort_indices()
        digest.update(np.asarray(csc.shape, np.int64).tobytes())
        digest.update(np.asarray(csc.indptr, np.int64).tobytes())
        digest.update(np.asarray(csc.indices, np.int64).tobytes())
    return digest.hexdigest()


def cap(priors):
    """The priors with every value above PRIOR_CAP set to it, and how many columns that changed."""
    priors = np.asarray(priors, np.float64)
    return np.minimum(priors, PRIOR_CAP), int((priors > PRIOR_CAP).sum())


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------- convert (luminaty) ----------

def convert_command(args):
    from circuit import GROSS_CIRCUITS, detector_ids, load_circuit
    from neutral_atom_noise.make_shots import load_shots
    from neutral_atom_noise.priors import kit_problem, matched_priors, uniform_priors

    shots = load_shots(args.shots)
    meta = shots["meta"]
    basis = meta["basis"]
    if meta["loss"]:
        raise SystemExit("a loss shot file: this check is for question 1's files without atom loss")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.shots).stem
    converted = out / f"{stem}_kitformat_p{args.p:g}.npz"
    circuit = load_circuit(GROSS_CIRCUITS[basis], args.p)
    ids = detector_ids(circuit, basis)
    detectors = np.asarray(shots["detectors"], np.uint8)
    if detectors.shape[1] != circuit.num_detectors:
        raise SystemExit(f"{detectors.shape[1]} detectors per shot; the kit's circuit has {circuit.num_detectors}")
    np.savez_compressed(converted, p=args.p, seed=meta["seed"], basis=basis, detectors=detectors,
                        observables=np.asarray(shots["observables"], np.uint8), basis_detectors=ids)
    H, L, _ = kit_problem(basis)
    H_p, L_p, generic = kit_problem(basis, args.p)
    if matrix_hash(H, L) != matrix_hash(H_p, L_p):
        raise SystemExit("the kit's H and L at this p differ from those at p = 0.001: the columns would not line up")
    if not np.array_equal(generic, uniform_priors(basis, args.p)):
        raise SystemExit("generic priors differ between two builds")
    matched, fired = matched_priors(basis, shots["noise"])
    generic, generic_capped = cap(generic)
    matched, matched_capped = cap(matched)
    np.save(out / "priors_generic.npy", generic)
    np.save(out / "priors_matched.npy", matched)
    record = {"shots_file": str(args.shots), "shots_sha256": file_sha256(args.shots), "shots": int(len(detectors)),
              "seed": meta["seed"], "basis": basis, "noise": meta["noise"], "factor": meta["factor"],
              "converted": converted.name, "converted_sha256": file_sha256(converted), "p": args.p,
              "columns": int(H.shape[1]), "rows": int(H.shape[0]), "h_l_sha256": matrix_hash(H, L),
              "generic": {"what": f"kit uniform model at p = {args.p:g}", "max": float(generic.max()),
                          "min": float(generic.min()), "sum": float(generic.sum()), "capped_columns": generic_capped,
                          "sha256": file_sha256(out / "priors_generic.npy")},
              "matched": {"what": "priors.matched_priors(basis, noise of the file's meta), floor 1e-9",
                          "max": float(matched.max()), "min": float(matched.min()), "sum": float(matched.sum()),
                          "columns_fired": int(fired.sum()), "capped_columns": matched_capped,
                          "sha256": file_sha256(out / "priors_matched.npy")}}
    with open(out / "transfer.json", "w") as file:
        json.dump(record, file, indent=1)
    print(json.dumps(record, indent=1))


# ---------- decode (server, TCBP environment) ----------

def save_results(out, mirror, done, rows, hashes, truth, extra):
    """per_shot.npz and summary.json of the first `done` shots (whole blocks every arm finished)."""
    from predecoder.tcbp import summarise

    per_shot = {"truth": truth[:done], "shots_scored": np.int64(done)}
    arms = {}
    for arm in ARMS:
        block = rows[arm][:done]
        for i, field in enumerate(FIELDS):
            per_shot[f"{field}_{arm}"] = block[:, i]
        per_shot[f"predicted_{arm}"] = ((block[:, 6].astype(np.int64)[:, None] & BITS) > 0).astype(np.uint8)
        per_shot[f"logits_hash_{arm}"] = hashes[arm][:done]
        arms[arm] = summarise(block) if done else None
    np.savez_compressed(out / "per_shot.npz", **per_shot)
    with open(out / "summary.json", "w") as file:
        json.dump(extra | {"shots_scored": done, "arms": arms, "saved_utc": datetime.now(timezone.utc).isoformat()},
                  file, indent=1)
    if mirror:
        for name in ("per_shot.npz", "summary.json"):
            shutil.copy(out / name, Path(mirror) / name)


def decode_command(args):
    from evaluation.data import load_dataset
    from predecoder import tcbp
    from predecoder.decode import kit_problem
    from predecoder.files import read_logits
    from predecoder.priors import load_settings

    with open(args.transfer) as file:
        transfer = json.load(file)
    data = load_dataset(args.data)
    if data.sha256 != transfer["converted_sha256"]:
        raise SystemExit("the shot file is not the converted file of transfer.json")
    problem = kit_problem(data.basis, data.p)
    if matrix_hash(problem.H, problem.L) != transfer["h_l_sha256"]:
        raise SystemExit("H and L rebuilt here differ from luminaty's: the prior columns would not line up")
    generic, matched = np.load(args.generic), np.load(args.matched)
    for name, vector in (("generic", generic), ("matched", matched)):
        if file_sha256(getattr(args, name)) != transfer[name]["sha256"]:
            raise SystemExit(f"{getattr(args, name)} is not the {name} vector of transfer.json")
        if vector.shape != (problem.H.shape[1],) or not (0 < vector).all() or (vector > PRIOR_CAP).any():
            raise SystemExit(f"the {name} vector has the wrong shape or values outside (0, {PRIOR_CAP}]")
    if not np.allclose(generic, problem.priors, rtol=1e-12, atol=0):
        raise SystemExit("the generic vector differs from the kit's priors rebuilt here at the file's p")
    setting = load_settings(args.settings)
    if setting != FROZEN:
        raise SystemExit(f"the knobs are {setting}, not the frozen {FROZEN}")
    meta, logits_path = read_logits(args.logits, data)
    logits = np.load(logits_path, mmap_mode="r")
    if logits.shape != (len(data.detectors), problem.H.shape[1]):
        raise SystemExit(f"logits of shape {logits.shape}")
    version = tcbp.check_environment(args.decoder)
    stop_at = datetime.fromisoformat(args.stop_at.replace("Z", "+00:00")).timestamp()
    shots = len(data.detectors) if args.limit is None else min(args.limit, len(data.detectors))
    syndromes = np.ascontiguousarray(data.basis_input[:shots], dtype=np.uint8)
    truth = np.asarray(data.observables[:shots], np.uint8)
    out = Path(args.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    if args.mirror:
        Path(args.mirror).mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(f"{data.path.name}: {shots} shots, p={data.p:g} (the network's p input and the mixing base), decoder "
          f"{args.decoder} ({version}), preset {args.preset}; head knobs {setting}, cap {PRIOR_CAP}; stop at "
          f"{args.stop_at}; {tcbp.where(args.decoder)}", flush=True)
    builder = tcbp.Builder(problem, args.decoder, args.preset)
    print(f"  model fitted in {builder.model_seconds:.1f} s; GPU memory {tcbp.gpu_memory()}", flush=True)
    floor = "1e-9"
    try:                                          # does TCBP-JOBS take the matched vector's 1e-9 floor?
        tcbp.decode_one(builder, matched, syndromes[0])
    except Exception as error:                    # noqa: BLE001  the declared fallback, logged
        print(f"  the matched vector was refused ({type(error).__name__}: {error}); its floor is raised to "
              f"{FLOOR_FALLBACK:g}, as declared", flush=True)
        matched, floor = np.maximum(matched, FLOOR_FALLBACK), f"{FLOOR_FALLBACK:g}"
    problems = {"m": SimpleNamespace(H=problem.H, L=problem.L, priors=matched),
                "g": SimpleNamespace(H=problem.H, L=problem.L, priors=generic),
                "h": SimpleNamespace(H=problem.H, L=problem.L, priors=generic)}
    if tuple(tcbp.FIELDS[:len(FIELDS)]) != FIELDS:
        raise SystemExit(f"predecoder.tcbp's row fields are {tcbp.FIELDS}, not {FIELDS} first")
    rows = {arm: np.zeros((shots, len(tcbp.FIELDS))) for arm in ARMS}
    hashes = {arm: np.zeros(shots, np.uint64) for arm in ARMS}
    capped = {arm: 0 for arm in ARMS}
    extra = {"data": data.identity(), "transfer": transfer, "decoder": args.decoder, "version": version,
             "preset": args.preset, "head_knobs": setting, "prior_cap": PRIOR_CAP, "matched_floor": floor,
             "logits": meta, "stop_at": args.stop_at, "block": BLOCK, "arm_order": list(ARMS),
             "model_seconds": builder.model_seconds, "machine": tcbp.where(args.decoder),
             "timer": "build: get_decoder with the shot's priors; decode: decode() and the cut to 0/1; one process"}
    done, warm = 0, True
    for first in range(0, shots, BLOCK):
        if time.time() > stop_at:
            print(f"  stop: {args.stop_at} passed before block {first}; scored shots {done}", flush=True)
            break
        last = min(first + BLOCK, shots)
        for arm in ARMS:
            path = out / "parts" / f"{arm}_{first:06d}_{last:06d}.npz"
            if not path.exists():
                before = tcbp.CAPPED["columns"]
                part, part_hashes = tcbp.decode_shots(builder, problems[arm], syndromes[first:last], truth[first:last],
                                                      logits[first:last] if arm == "h" else None,
                                                      setting if arm == "h" else None, f"{arm} {first}-{last - 1}", warm)
                capped[arm] += tcbp.CAPPED["columns"] - before
                warm = False
                np.savez(path, rows=part, hashes=part_hashes)
            with np.load(path) as file:
                rows[arm][first:last], hashes[arm][first:last] = file["rows"], file["hashes"]
        done = last
        extra["capped_columns"] = dict(capped)
        extra["seconds"] = time.time() - start
        save_results(out, args.mirror, done, rows, hashes, truth, extra)
        print(f"  block {first}-{last - 1} done, {time.time() - start:.0f} s; failures so far "
              + ", ".join(f"{arm} {int(rows[arm][:done, 0].sum())}" for arm in ARMS)
              + f" of {done}; GPU memory {tcbp.gpu_memory()}", flush=True)
    save_results(out, args.mirror, done, rows, hashes, truth, extra | {"seconds": time.time() - start})
    print(f"saved {out / 'per_shot.npz'}: {done} shots scored; {time.time() - start:.0f} s", flush=True)


# ---------- score (luminaty) ----------

def wilson(failures, shots, z=1.959963984540054):
    if shots == 0:
        return [0.0, 1.0]
    share = failures / shots
    centre = (share + z * z / (2 * shots)) / (1 + z * z / shots)
    half = z * np.sqrt(share * (1 - share) / shots + z * z / (4 * shots * shots)) / (1 + z * z / shots)
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def sign_test(only_a, only_b):
    """Exact two-sided sign test: the chance of a split at least this uneven if each discordant shot is a coin."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def branch_rows(results, stem, decoder, spec):
    """(failed per shot, shots) of one of this branch's rows on this file, from its part files in order."""
    parts = sorted(Path(results).glob(f"{stem}__{decoder}__{spec}__*.npz"))
    if not parts:
        return None
    failed, index = [], []
    for part in parts:
        with np.load(part) as file:
            failed.append(np.asarray(file["failed"], bool))
            index.append(np.asarray(file["shots"], np.int64))
    index = np.concatenate(index)
    if not np.array_equal(index, np.arange(len(index))):
        raise SystemExit(f"the parts of {stem} {decoder} {spec} do not cover shots 0 to {len(index) - 1} in order")
    return np.concatenate(failed)


def score_command(args):
    with np.load(args.per_shot) as file:
        per_shot = {key: file[key] for key in file.files}
    done = int(per_shot["shots_scored"])
    with np.load(args.shots) as file:
        observables = file["observables"][:done]
    if not np.array_equal(per_shot["truth"], observables):
        raise SystemExit(f"{args.per_shot} was not made from {args.shots}")
    failed = {arm: per_shot[f"failed_{arm}"] > 0 for arm in ARMS}
    stem = Path(args.shots).stem
    for name, decoder, spec in (("BP-OSD matched", "bposd", "matched"), ("BP-OSD generic (fitted p)", "bposd", None),
                                ("relay-BP matched", "relay", "matched"), ("relay-BP generic (fitted p)", "relay", None)):
        if spec is None:
            with open(Path(args.results).parent / "fit.json") as file:
                p = json.load(file)[Path(args.shots).name][decoder]["p"]
            spec = f"uniform{p:g}"
        row = branch_rows(args.results, stem, decoder, spec)
        if row is not None:
            failed[name] = row
    summary = {"shots_file": str(args.shots), "shots_sha256": file_sha256(args.shots), "tcbp_shots_scored": done,
               "label": "exploration, after the main test results; a separate branch", "rows": {}, "pairs": []}
    for name, row in failed.items():
        n = min(len(row), done)
        summary["rows"][name] = {"shots": n, "failures": int(row[:n].sum()), "wilson95": wilson(int(row[:n].sum()), n)}
    pairs = [("h", "g"), ("h", "m"), ("m", "g")]
    pairs += [(arm, other) for arm in ARMS for other in failed if other not in ARMS]
    for a, b in pairs:
        n = min(len(failed[a]), len(failed[b]), done)
        only_a = int((failed[a][:n] & ~failed[b][:n]).sum())
        only_b = int((failed[b][:n] & ~failed[a][:n]).sum())
        summary["pairs"].append({"a": a, "b": b, "shots": n, "failures_a": int(failed[a][:n].sum()),
                                 "failures_b": int(failed[b][:n].sum()), "only_a": only_a, "only_b": only_b,
                                 "sign_test_p": sign_test(only_a, only_b)})
    with open(args.out, "w") as file:
        json.dump(summary, file, indent=1)
    for name, row in summary["rows"].items():
        print(f"{name:28s} {row['failures']:5d} of {row['shots']:5d}  [{row['wilson95'][0]:.4f}, {row['wilson95'][1]:.4f}]")
    for pair in summary["pairs"]:
        print(f"{pair['a']} vs {pair['b']}: {pair['failures_a']} vs {pair['failures_b']} of {pair['shots']}; only "
              f"{pair['a']} {pair['only_a']}, only {pair['b']} {pair['only_b']}; sign test p = {pair['sign_test_p']:.3g}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert")
    convert.add_argument("--shots", required=True)
    convert.add_argument("--p", type=float, required=True, help="the generic arm's uniform p (BP-OSD's fitted p)")
    convert.add_argument("--out", required=True)
    decode = sub.add_parser("decode")
    for name in ("data", "generic", "matched", "transfer", "logits", "settings", "stop_at", "out"):
        decode.add_argument(f"--{name.replace('_', '-')}", required=True)
    decode.add_argument("--mirror", help="a folder that gets a copy of per_shot.npz and summary.json after each block")
    decode.add_argument("--decoder", default="tcbp-jobs", choices=("tcbp-jobs", "relay"))
    decode.add_argument("--preset", default="tcbp-jobs")
    decode.add_argument("--limit", type=int, help="only the first this many shots (dry runs)")
    score = sub.add_parser("score")
    for name in ("per_shot", "shots", "results", "out"):
        score.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args(argv)
    {"convert": convert_command, "decode": decode_command, "score": score_command}[args.command](args)


if __name__ == "__main__":
    main()
