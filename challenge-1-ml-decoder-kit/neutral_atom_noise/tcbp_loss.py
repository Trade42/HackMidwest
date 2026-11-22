"""The TCBP pipeline with atom loss: TCBP-JOBS with matched or head priors, with and without loss/'s flag raise.

Exploration, after the main test results, on the branch infleqtion-noise. The arms, the pairs and the stopping rule
are declared in tcbp_loss_declared.md, pushed before any number of this tool existed.

  bm  matched priors, loss-blind              fm  matched priors + loss/'s 'starter' flag raise
  fh  head priors (frozen TCBP knobs) + raise  bh  head priors, loss-blind (pass 2, if time allows)

The raise XOR-combines bumps with a base: P = (1 - (1 - 2 base) * F) / 2, F = prod(1 - 2 v) over the shot's bumps
(loss.lossmap._combine). F does not depend on the base, so it is computed here once per shot (convert) and shipped
as a sparse matrix; the server applies it to the matched vector (fm) and to the head's vector (fh) alike.

Three steps, run from challenge-1-ml-decoder-kit/:

  convert   on luminaty: each loss shot file's first --prefix shots in the kit's format at its generic-2q p, the
            matched vector, the flag factors F, and transfer.json.
              python -m neutral_atom_noise.tcbp_loss convert --shots <X_f0.4_loss_seed6100.npz> <X_f0.6...> \\
                  --prefix 1000 --out <folder>
  decode    on the server, TCBP environment, predecoder-tcbp's kit first on PYTHONPATH. One process, one model.
              python -m neutral_atom_noise.tcbp_loss decode --transfer <transfer.json> --logits <dir f0.4> <dir f0.6> \\
                  --settings predecoder/logs/fit_tcbp_X_p0.005.json --stop-at 2026-11-22T17:35:00Z --out <folder>
  score     on luminaty, from decode's per_shot_<tag>.npz and the branch's relay-BP part files.
              python -m neutral_atom_noise.tcbp_loss score --decoded <folder> --shots <X_f0.4...> <X_f0.6...> \\
                  --results <run folder>/results --out <summary.json>
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from neutral_atom_noise.tcbp_check import FROZEN, PRIOR_CAP, branch_rows, file_sha256, matrix_hash, sign_test, wilson

MAIN_ARMS = ("bm", "fm", "fh")            # pass 1, in this order inside each block
LATE_ARMS = ("bh",)                       # pass 2
ARMS = MAIN_ARMS + LATE_ARMS
BLOCK = 100
FIELDS = ("failed", "reproduced", "flag", "n_faults", "seconds_build", "seconds_decode", "predicted_code")
BITS = 1 << np.arange(12)


def combine(base, columns, factors):
    """The flag raise on top of a base vector: P = (1 - (1 - 2 base) F) / 2 on the given columns, then capped."""
    priors = np.array(base, np.float64)
    priors[columns] = (1 - (1 - 2 * priors[columns]) * factors) / 2
    return np.minimum(priors, PRIOR_CAP)


def shot_factors(flags, k):
    """(columns, F) of shot k from the CSR arrays of a flags file."""
    lo, hi = flags["indptr"][k], flags["indptr"][k + 1]
    return flags["columns"][lo:hi], flags["factors"][lo:hi]


def tag_of(path):
    return Path(path).stem.split("_")[1]           # X_f0.4_loss_seed6100 -> f0.4


# ---------- convert (luminaty) ----------

def convert_command(args):
    from circuit import GROSS_CIRCUITS, detector_ids, load_circuit
    from evaluation.data import load_dataset
    from loss.lossmap import LossMap, PriorBuilder, parse_condition
    from neutral_atom_noise.make_shots import load_shots
    from neutral_atom_noise.priors import kit_problem, matched_priors, uniform_priors

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = {"label": "exploration, after the main test results; a separate branch", "prefix": args.prefix,
              "condition": "starter", "files": []}
    loss_maps = {}
    for shots_path in args.shots:
        shots = load_shots(shots_path)
        meta, noise = shots["meta"], shots["noise"]
        basis, tag = meta["basis"], tag_of(shots_path)
        if not meta["loss"] or "heralds" not in shots:
            raise SystemExit(f"{shots_path} is not a loss shot file")
        n = min(args.prefix, len(shots["detectors"]))
        p = float(f"{noise.two_qubit_error():.5g}")              # generic-2q p, as declared
        circuit = load_circuit(GROSS_CIRCUITS[basis], p)
        ids = detector_ids(circuit, basis)
        detectors = np.asarray(shots["detectors"][:n], np.uint8)
        converted = out / f"{Path(shots_path).stem}_first{n}_kitformat_p{p:g}.npz"
        np.savez_compressed(converted, p=p, seed=meta["seed"], basis=basis, detectors=detectors,
                            observables=np.asarray(shots["observables"][:n], np.uint8), basis_detectors=ids)
        data = load_dataset(converted)
        if not np.array_equal(data.basis_input, detectors[:, ids]):
            raise SystemExit("the converted file's basis detectors differ from the shot file's")
        H, L, _ = kit_problem(basis)
        H_p, L_p, generic = kit_problem(basis, p)
        if matrix_hash(H, L) != matrix_hash(H_p, L_p):
            raise SystemExit("the kit's H and L at this p differ from those at p = 0.001")
        if not np.array_equal(generic, uniform_priors(basis, p)):
            raise SystemExit("generic priors differ between two builds")
        matched_raw, fired = matched_priors(basis, noise)
        matched = np.minimum(matched_raw, PRIOR_CAP)
        if basis not in loss_maps:
            loss_maps[basis] = LossMap(basis, 0.001)
        loss_map = loss_maps[basis]
        if (loss_map.H != H).nnz or (loss_map.L != L).nnz:
            raise SystemExit("loss/'s matrices are not the kit's")
        builder = PriorBuilder(loss_map, noise.loss)
        raw = dict(parse_condition("starter"), cap=1.0)           # the bumps alone, no cap: F = 1 - 2 P(base 0)
        starter = parse_condition("starter")
        heralds = shots["heralds"][:n]
        indptr, columns, factors, flagged_atoms = [0], [], [], []
        worst = 0.0
        for k in range(n):
            loss_map.pri = np.zeros(H.shape[1])
            alone = builder.priors(raw, heralds[k])
            cols = np.nonzero(alone > 0)[0]
            f = 1 - 2 * alone[cols]
            columns.append(cols.astype(np.int32)); factors.append(f)
            indptr.append(indptr[-1] + len(cols))
            loss_map.pri = matched_raw                            # run._flag_setup's base: the matched vector
            reference = builder.priors(starter, heralds[k])
            ours = combine(matched_raw, cols, f)
            worst = max(worst, float(np.abs(ours - reference).max()))
            flagged_atoms.append(int(heralds[k].sum()))
        if worst > 1e-12:
            raise SystemExit(f"the shipped raise differs from loss/'s starter priors by {worst:g}")
        blind = builder.priors(parse_condition("blind"), heralds[0])
        if not np.array_equal(blind, matched):
            raise SystemExit("bm's vector is not loss/'s blind vector")
        flags_path = out / f"flags_{tag}.npz"
        np.savez_compressed(flags_path, indptr=np.asarray(indptr, np.int64), columns=np.concatenate(columns),
                            factors=np.concatenate(factors))
        matched_path = out / f"priors_matched_{tag}.npy"
        np.save(matched_path, matched)
        record["files"].append({
            "tag": tag, "shots_file": str(shots_path), "shots_sha256": file_sha256(shots_path),
            "shots_in_file": int(len(shots["detectors"])), "shots": n, "seed": meta["seed"], "basis": basis,
            "factor": meta["factor"], "noise": meta["noise"], "p": p, "p_what": "generic-2q: noise.two_qubit_error()",
            "converted": converted.name, "converted_sha256": file_sha256(converted),
            "columns": int(H.shape[1]), "h_l_sha256": matrix_hash(H, L),
            "generic_sum": float(generic.sum()),
            "matched": {"file": matched_path.name, "sha256": file_sha256(matched_path), "max": float(matched.max()),
                        "min": float(matched.min()), "columns_fired": int(fired.sum())},
            "flags": {"file": flags_path.name, "sha256": file_sha256(flags_path), "bumped_entries": int(indptr[-1]),
                      "shots_with_flags": int(sum(a > 0 for a in flagged_atoms)),
                      "mean_heralds_per_shot": float(np.mean(flagged_atoms)),
                      "largest_gap_to_loss_starter": worst}})
        print(f"{converted.name}: {n} shots, p {p:g}; matched max {matched.max():.4g}; flags on "
              f"{record['files'][-1]['flags']['shots_with_flags']} shots, {indptr[-1]} bumped entries; largest gap "
              f"to loss/'s starter vector {worst:.2g}", flush=True)
    with open(out / "transfer.json", "w") as file:
        json.dump(record, file, indent=1)


# ---------- decode (server, TCBP environment) ----------

def decode_command(args):
    from evaluation.data import load_dataset
    from predecoder import tcbp
    from predecoder.decode import kit_problem
    from predecoder.files import read_logits
    from predecoder.priors import load_settings, shot_priors

    transfer_dir = Path(args.transfer).parent
    with open(args.transfer) as file:
        transfer = json.load(file)
    if len(args.logits) != len(transfer["files"]):
        raise SystemExit("one --logits folder per file of transfer.json, in its order")
    setting = load_settings(args.settings)
    if setting != FROZEN:
        raise SystemExit(f"the knobs are {setting}, not the frozen {FROZEN}")
    version = tcbp.check_environment(args.decoder)
    files = []
    for entry, logits_dir in zip(transfer["files"], args.logits, strict=True):
        data = load_dataset(transfer_dir / entry["converted"])
        if data.sha256 != entry["converted_sha256"]:
            raise SystemExit(f"{entry['converted']} is not the file of transfer.json")
        problem = kit_problem(data.basis, data.p)
        if matrix_hash(problem.H, problem.L) != entry["h_l_sha256"]:
            raise SystemExit("H and L rebuilt here differ from luminaty's: the prior columns would not line up")
        if not np.isclose(problem.priors.sum(), entry["generic_sum"], rtol=1e-12, atol=0):
            raise SystemExit("the kit's priors at the file's p differ from luminaty's")
        for key in ("matched", "flags"):
            if file_sha256(transfer_dir / entry[key]["file"]) != entry[key]["sha256"]:
                raise SystemExit(f"{entry[key]['file']} is not the file of transfer.json")
        matched = np.load(transfer_dir / entry["matched"]["file"])
        if matched.shape != (problem.H.shape[1],) or not (0 < matched).all() or (matched > PRIOR_CAP).any():
            raise SystemExit("the matched vector has the wrong shape or values outside (0, 0.5]")
        with np.load(transfer_dir / entry["flags"]["file"]) as file:
            flags = {key: file[key] for key in file.files}
        meta, logits_path = read_logits(logits_dir, data)
        logits = np.load(logits_path, mmap_mode="r")
        shots = len(data.detectors) if args.limit is None else min(args.limit, len(data.detectors))
        if logits.shape != (len(data.detectors), problem.H.shape[1]) or len(flags["indptr"]) != len(data.detectors) + 1:
            raise SystemExit(f"logits of shape {logits.shape}, flags of {len(flags['indptr']) - 1} shots")
        files.append(SimpleNamespace(tag=entry["tag"], data=data, problem=problem, matched=matched, flags=flags,
                                     logits=logits, logits_meta=meta, shots=shots,
                                     syndromes=np.ascontiguousarray(data.basis_input[:shots], dtype=np.uint8),
                                     truth=np.asarray(data.observables[:shots], np.uint8)))
    if len({matrix_hash(f.problem.H, f.problem.L) for f in files}) != 1:
        raise SystemExit("the files' H and L differ: one model cannot serve both")
    stop_at = datetime.fromisoformat(args.stop_at.replace("Z", "+00:00")).timestamp()
    out = Path(args.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(f"files {[f.data.path.name for f in files]}, {[f.shots for f in files]} shots, p {[f.data.p for f in files]} "
          f"(the network's p input and the head's mixing base); decoder {args.decoder} ({version}), preset "
          f"{args.preset}; head knobs {setting}, cap {PRIOR_CAP}; stop at {args.stop_at}; {tcbp.where(args.decoder)}",
          flush=True)
    builder = tcbp.Builder(files[0].problem, args.decoder, args.preset)
    print(f"  model fitted in {builder.model_seconds:.1f} s", flush=True)
    floor = "1e-9"
    try:                                          # does TCBP-JOBS take the matched vector's 1e-9 floor?
        tcbp.decode_one(builder, files[0].matched, files[0].syndromes[0])
    except Exception as error:                    # noqa: BLE001  logged; the run carries on at 1e-6
        print(f"  the matched vector was refused ({type(error).__name__}: {error}); floor raised to 1e-6", flush=True)
        for f in files:
            f.matched = np.maximum(f.matched, 1e-6)
        floor = "1e-6"

    def priors_of(f, arm, k):
        if arm == "bm":
            return f.matched
        if arm == "fm":
            return combine(f.matched, *shot_factors(f.flags, k))
        head = shot_priors(f.logits[k], f.problem.priors, setting["alpha"], setting["eps"], setting["mode"])
        return combine(head, *shot_factors(f.flags, k)) if arm == "fh" else np.minimum(head, PRIOR_CAP)

    def decode_block(f, arm, first, last):
        path = out / "parts" / f"{f.tag}_{arm}_{first:06d}_{last:06d}.npz"
        if not path.exists():
            rows = np.zeros((last - first, len(FIELDS)))
            hashes = np.zeros(last - first, np.uint64)
            for i, k in enumerate(range(first, last)):
                priors = tcbp.capped(priors_of(f, arm, k))
                error, flag, build, decode = tcbp.decode_one(builder, priors, f.syndromes[k])
                predicted = (f.problem.L @ error) % 2
                reproduced = np.array_equal((f.problem.H @ error) % 2, f.syndromes[k])
                rows[i] = (bool((predicted != f.truth[k]).any()), reproduced, flag, int(error.sum()), build, decode,
                           int(np.asarray(predicted, np.int64) @ BITS))
                hashes[i] = tcbp.row_hash(f.logits[k]) if arm in ("fh", "bh") else 0
            np.savez(path, rows=rows, hashes=hashes)
        with np.load(path) as file:
            f.rows[arm][first:last], f.hashes[arm][first:last] = file["rows"], file["hashes"]

    for f in files:
        f.rows = {arm: np.zeros((f.shots, len(FIELDS))) for arm in ARMS}
        f.hashes = {arm: np.zeros(f.shots, np.uint64) for arm in ARMS}
        f.done = {"main": 0, "late": 0}
    extra = {"transfer": transfer, "decoder": args.decoder, "version": version, "preset": args.preset,
             "head_knobs": setting, "prior_cap": PRIOR_CAP, "matched_floor": floor, "stop_at": args.stop_at,
             "block": BLOCK, "pass_1": list(MAIN_ARMS), "pass_2": list(LATE_ARMS),
             "model_seconds": builder.model_seconds, "machine": tcbp.where(args.decoder),
             "timer": "build: get_decoder with the shot's priors; decode: decode() and the cut to 0/1; one process"}
    stopped = False
    for passname, arms in (("main", MAIN_ARMS), ("late", LATE_ARMS)):
        for first in range(0, max(f.shots for f in files), BLOCK):
            for f in files:
                if first >= f.shots or stopped:
                    continue
                if time.time() > stop_at:
                    print(f"  stop: {args.stop_at} passed before {passname} block {f.tag} {first}", flush=True)
                    stopped = True
                    continue
                last = min(first + BLOCK, f.shots)
                for arm in arms:
                    decode_block(f, arm, first, last)
                f.done[passname] = last
                save(out, args.mirror, files, extra | {"seconds": time.time() - start, "capped": dict(tcbp.CAPPED)})
                print(f"  {passname} block {f.tag} {first}-{last - 1} done, {time.time() - start:.0f} s; failures so far "
                      + ", ".join(f"{arm} {int(f.rows[arm][:f.done[passname], 0].sum())}" for arm in arms)
                      + f" of {f.done[passname]}", flush=True)
    save(out, args.mirror, files, extra | {"seconds": time.time() - start, "capped": dict(tcbp.CAPPED)})
    print(f"saved {out}: " + ", ".join(f"{f.tag} main {f.done['main']} late {f.done['late']}" for f in files)
          + f"; {time.time() - start:.0f} s", flush=True)


def save(out, mirror, files, extra):
    """per_shot_<tag>.npz (the scored prefixes only) and summary.json; copies to the mirror folder."""
    import shutil

    summary = dict(extra, saved_utc=datetime.now(timezone.utc).isoformat(), files={})
    for f in files:
        per_shot = {"truth": f.truth, "scored_main": np.int64(f.done["main"]), "scored_late": np.int64(f.done["late"])}
        for arm in ARMS:
            n = f.done["main" if arm in MAIN_ARMS else "late"]
            block = f.rows[arm][:n]
            for i, field in enumerate(FIELDS):
                per_shot[f"{field}_{arm}"] = block[:, i]
            per_shot[f"predicted_{arm}"] = ((block[:, 6].astype(np.int64)[:, None] & BITS) > 0).astype(np.uint8)
            per_shot[f"logits_hash_{arm}"] = f.hashes[arm][:n]
        np.savez_compressed(out / f"per_shot_{f.tag}.npz", **per_shot)
        summary["files"][f.tag] = {"data": f.data.identity(), "logits": f.logits_meta, "scored": dict(f.done),
                                   "failures": {arm: int(f.rows[arm][:f.done["main" if arm in MAIN_ARMS else "late"], 0].sum())
                                                for arm in ARMS},
                                   "mean_ms": {arm: float(1e3 * f.rows[arm][:f.done["main" if arm in MAIN_ARMS else "late"], 4:6].sum(1).mean())
                                               if f.done["main" if arm in MAIN_ARMS else "late"] else None for arm in ARMS}}
    with open(out / "summary.json", "w") as file:
        json.dump(summary, file, indent=1)
    if mirror:
        Path(mirror).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(out / f"per_shot_{f.tag}.npz", Path(mirror) / f"per_shot_{f.tag}.npz")
        shutil.copy(out / "summary.json", Path(mirror) / "summary.json")


# ---------- score (luminaty) ----------

RELAY_ROWS = {"relay matched + starter": "matched__starter", "relay matched blind": "matched__blind",
              "relay generic-2q + starter": "generic-2q__starter"}
PAIRS = [("fm", "bm"), ("fh", "fm"), ("fh", "bh"),
         ("fm", "relay matched + starter"), ("bm", "relay matched blind"), ("fh", "relay generic-2q + starter")]


def score_command(args):
    summary = {"label": "exploration, after the main test results; a separate branch", "files": {}}
    for shots_path in args.shots:
        tag = tag_of(shots_path)
        with np.load(Path(args.decoded) / f"per_shot_{tag}.npz") as file:
            per_shot = {key: file[key] for key in file.files}
        with np.load(shots_path) as file:
            observables = file["observables"][:len(per_shot["truth"])]
        if not np.array_equal(per_shot["truth"], observables):
            raise SystemExit(f"per_shot_{tag}.npz was not made from {shots_path}")
        for arm in ARMS:                                   # the stored failure flags agree with the predictions
            n = len(per_shot[f"failed_{arm}"])
            again = (per_shot[f"predicted_{arm}"] != observables[:n]).any(1)
            if not np.array_equal(again, per_shot[f"failed_{arm}"] > 0):
                raise SystemExit(f"{tag} {arm}: failed does not match the predicted bits")
        failed = {arm: per_shot[f"failed_{arm}"] > 0 for arm in ARMS}
        for name, spec in RELAY_ROWS.items():
            row = branch_rows(args.results, Path(shots_path).stem, "relay", spec)
            if row is not None:                            # on the shots TCBP's main arms scored, as declared
                failed[name] = row[:int(per_shot["scored_main"])]
        result = {"shots_file": str(shots_path), "shots_sha256": file_sha256(shots_path),
                  "scored_main": int(per_shot["scored_main"]), "scored_late": int(per_shot["scored_late"]),
                  "rows": {}, "pairs": []}
        for name, row in failed.items():
            n = len(row)
            result["rows"][name] = {"shots": n, "failures": int(row.sum()), "wilson95": wilson(int(row.sum()), n)}
        for a, b in PAIRS:
            if a not in failed or b not in failed:
                continue
            n = min(len(failed[a]), len(failed[b]))
            if n == 0:
                continue
            only_a = int((failed[a][:n] & ~failed[b][:n]).sum())
            only_b = int((failed[b][:n] & ~failed[a][:n]).sum())
            result["pairs"].append({"a": a, "b": b, "shots": n, "failures_a": int(failed[a][:n].sum()),
                                    "failures_b": int(failed[b][:n].sum()), "only_a": only_a, "only_b": only_b,
                                    "sign_test_p": sign_test(only_a, only_b)})
        summary["files"][tag] = result
        print(f"{tag}: TCBP scored {result['scored_main']} shots (bh {result['scored_late']})")
        for name, row in result["rows"].items():
            print(f"  {name:28s} {row['failures']:5d} of {row['shots']:5d}  "
                  f"[{row['wilson95'][0]:.4f}, {row['wilson95'][1]:.4f}]")
        for pair in result["pairs"]:
            print(f"  {pair['a']} vs {pair['b']}: {pair['failures_a']} vs {pair['failures_b']} of {pair['shots']}; "
                  f"only {pair['a']} {pair['only_a']}, only {pair['b']} {pair['only_b']}; sign test p = "
                  f"{pair['sign_test_p']:.3g}")
    with open(args.out, "w") as file:
        json.dump(summary, file, indent=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert")
    convert.add_argument("--shots", nargs="+", required=True)
    convert.add_argument("--prefix", type=int, default=1000)
    convert.add_argument("--out", required=True)
    decode = sub.add_parser("decode")
    decode.add_argument("--logits", nargs="+", required=True)
    for name in ("transfer", "settings", "stop_at", "out"):
        decode.add_argument(f"--{name.replace('_', '-')}", required=True)
    decode.add_argument("--mirror")
    decode.add_argument("--decoder", default="tcbp-jobs", choices=("tcbp-jobs", "relay"))
    decode.add_argument("--preset", default="tcbp-jobs")
    decode.add_argument("--limit", type=int, help="only the first this many shots of each file (dry runs)")
    score = sub.add_parser("score")
    score.add_argument("--shots", nargs="+", required=True)
    for name in ("decoded", "results", "out"):
        score.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    {"convert": convert_command, "decode": decode_command, "score": score_command}[args.command](args)


if __name__ == "__main__":
    main()
