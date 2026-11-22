"""The TCBP pipeline with a head retrained on the neutral-atom-shaped noise: five arms of TCBP-JOBS on the same shots.

Exploration, task tcbp-retrain. The arms, the pairs and the stopping rule are declared in tcbp_retrain_declared.md,
written before any number of this tool existed. Every arm decodes the kit's H and L (X memory, 936 x 8,784); an arm
is a base prior vector, plus, for the three head arms, one logits file of predecoder.evaluate and the frozen knobs
(log-odds, alpha 0.4, eps 1e-4, cap 0.5) that mix the head's logits with that same base:

  m    matched priors (the shots' own error model on the kit's columns), no head
  g    generic: the kit's uniform priors at the converted file's p, no head
  h    the kit-trained head (38ab667e...), its input and base the generic vector: phase 1's head arm
  hm   the kit-trained head, its input and base the matched vector (evaluate --base-priors)
  r    the head retrained on the model (predecoder.train --noise-factor 0.8), its input and base the matched vector,
       the vector it was trained with (evaluate checks the sha256)
  rf   r with the knobs refitted for it on train shots (`fit`, below; tcbp_retrain_refit_declared.md), the same logits

The refit (`fit`): on one train task of a plan, TCBP-JOBS decodes arm m once and then r's logits at every setting of
FIT_GRID (log-odds mixing with the matched base, the vector the head was trained with; predecoder.tcbp's own fit
cannot do that, it mixes with the kit's priors at the file's p). The chosen setting has the fewest train failures;
ties go to fewer corrections that do not reproduce the syndrome, then fewer shots flagged not converged, then the
smaller alpha, then the larger eps. `decode --refit <fit.json>` gives that setting to the rf arm.

Three steps, run from challenge-1-ml-decoder-kit/:

  plan     on luminaty, after neutral_atom_noise.tcbp_check convert made each task's folder: plan.json names the
           tasks, their files, their arms and the sha256 of the two heads.
             python -m neutral_atom_noise.tcbp_retrain plan --root <input folder> \\
                 --task f0.8-seed4800:seed4800:m,g,h,hm,r --task f0.8-seed4100:seed4100:m,h,hm,r \\
                 --kit-head <sha256> --retrained-head <sha256> --out <input folder>/plan.json
  decode   on the server, in the teammates' TCBP environment. Logits in <logits root>/<task>/<kit|kit-matched|retrained>/.
           One TCBP-JOBS model for every task (one H). Blocks of 100 shots, tasks in turn, the arms in the order
           m, g, h, hm, r inside each block; no new block after --stop-at.
             python -m neutral_atom_noise.tcbp_retrain decode --plan <plan.json> --root <input folder> \\
                 --logits-root <folder> --settings predecoder/logs/fit_tcbp_X_p0.005.json --stop-at <UTC> --out <folder>
  fit      on the server: python -m neutral_atom_noise.tcbp_retrain fit --plan <train plan.json> --root <inputs> \
               --logits-root <folder> --stop-at <UTC> --out <fit.json>
  score    on luminaty: failures with Wilson intervals on each task's scored prefix, and the declared pairs.
             python -m neutral_atom_noise.tcbp_retrain score --plan <plan.json> --root <input folder> \\
                 --decoded <decode folder> --out <score.json>
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from neutral_atom_noise.tcbp_check import (BITS, BLOCK, FIELDS, FROZEN, PRIOR_CAP, file_sha256, matrix_hash, sign_test,
                                           wilson)

ORDER = ("m", "g", "h", "hm", "r", "rf")                             # the declared order inside each block
BASE = {"m": "matched", "g": "generic", "h": "generic", "hm": "matched", "r": "matched", "rf": "matched"}
LOGITS = {"h": "kit", "hm": "kit-matched", "r": "retrained", "rf": "retrained"}   # sub-folder of the logits root
HEAD = {"h": "kit", "hm": "kit", "r": "retrained", "rf": "retrained"}  # which head's sha256 the logits must carry
PRIMARY = [("r", "m"), ("r", "h")]                                   # a plan may name its own (plan --primary)
SECONDARY = [("r", "hm"), ("hm", "m"), ("h", "m"), ("m", "g"), ("r", "g"), ("h", "g"), ("rf", "r"), ("r", "m")]
FIT_GRID = [{"mode": "logodds", "alpha": alpha, "eps": eps} for eps in (1e-4, 1e-6)
            for alpha in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)]        # declared in tcbp_retrain_refit_declared.md


# ---------- plan (luminaty) ----------

def plan_command(args):
    root, tasks = Path(args.root), []
    for spec in args.task:
        name, folder, arms = spec.split(":")
        arms = arms.split(",")
        if any(arm not in ORDER for arm in arms) or "m" not in arms:
            raise SystemExit(f"{spec}: arms are among {ORDER}, and m is always one of them")
        with open(root / folder / "transfer.json") as file:
            transfer = json.load(file)
        tasks.append({"name": name, "data": f"{folder}/{transfer['converted']}", "transfer": f"{folder}/transfer.json",
                      "generic": f"{folder}/priors_generic.npy", "matched": f"{folder}/priors_matched.npy",
                      "arms": [arm for arm in ORDER if arm in arms], "shots": transfer["shots"], "seed": transfer["seed"],
                      "factor": transfer["factor"], "p": transfer["p"]})
    plan = {"label": "exploration, task tcbp-retrain", "declared": args.declared,
            "tasks": tasks, "heads": {"kit": args.kit_head, "retrained": args.retrained_head}, "knobs": FROZEN,
            "prior_cap": PRIOR_CAP, "block": BLOCK, "arm_order": list(ORDER)}
    if args.primary:
        plan["primary"] = [pair.split(":") for pair in args.primary.split(",")]
    if args.role:
        plan["role"] = args.role
    with open(args.out, "w") as file:
        json.dump(plan, file, indent=1)
    print(json.dumps(plan, indent=1))


# ---------- decode (server, TCBP environment) ----------

def load_task(task, plan, root, logits_root, first_problem):
    """One task: its shots, problem, base vectors, and per head arm its logits, after every check."""
    from evaluation.data import load_dataset
    from predecoder.decode import kit_problem
    from predecoder.files import read_logits, vector_sha256

    root = Path(root)
    with open(root / task["transfer"]) as file:
        transfer = json.load(file)
    data = load_dataset(root / task["data"])
    if data.sha256 != transfer["converted_sha256"]:
        raise SystemExit(f"{task['name']}: the shot file is not the one of its transfer.json")
    problem = kit_problem(data.basis, data.p)
    if matrix_hash(problem.H, problem.L) != transfer["h_l_sha256"]:
        raise SystemExit(f"{task['name']}: H and L rebuilt here differ from luminaty's")
    if first_problem is not None and matrix_hash(problem.H, problem.L) != matrix_hash(first_problem.H, first_problem.L):
        raise SystemExit(f"{task['name']}: another H than the first task's: one TCBP model cannot serve both")
    vectors = {}
    for name in ("generic", "matched"):
        path = root / task[name]
        if file_sha256(path) != transfer[name]["sha256"]:
            raise SystemExit(f"{task['name']}: {path} is not the {name} vector of its transfer.json")
        vector = np.load(path)
        if vector.shape != (problem.H.shape[1],) or not (0 < vector).all() or (vector > PRIOR_CAP).any():
            raise SystemExit(f"{task['name']}: the {name} vector has the wrong shape or values outside (0, {PRIOR_CAP}]")
        vectors[name] = vector
    if not np.allclose(vectors["generic"], problem.priors, rtol=1e-12, atol=0):
        raise SystemExit(f"{task['name']}: the generic vector differs from the kit's priors at the file's p")
    arms, metas = {}, {}
    for arm in task["arms"]:
        base = vectors[BASE[arm]]
        logits = None
        if arm in LOGITS:
            meta, path = read_logits(Path(logits_root) / task["name"] / LOGITS[arm], data)
            if meta["model"]["head_sha256"] != plan["heads"][HEAD[arm]]:
                raise SystemExit(f"{task['name']} {arm}: the logits are of head {meta['model']['head_sha256']}, not "
                                 f"the {HEAD[arm]} head {plan['heads'][HEAD[arm]]}")
            if meta.get("base_priors", {}).get("sha256") != vector_sha256(base):
                raise SystemExit(f"{task['name']} {arm}: the head's base was not the {BASE[arm]} vector")
            logits = np.load(path, mmap_mode="r")
            if logits.shape != (len(data.detectors), problem.H.shape[1]):
                raise SystemExit(f"{task['name']} {arm}: logits of shape {logits.shape}")
            metas[arm] = meta
        arms[arm] = SimpleNamespace(problem=SimpleNamespace(H=problem.H, L=problem.L, priors=base), logits=logits)
    return SimpleNamespace(name=task["name"], data=data, problem=problem, transfer=transfer, arms=arms, metas=metas)


def save_results(state, extra):
    from predecoder.tcbp import summarise

    done = state.done
    per_shot = {"truth": state.truth[:done], "shots_scored": np.int64(done)}
    arms = {}
    for arm, rows in state.rows.items():
        block = rows[:done]
        for i, field in enumerate(FIELDS):
            per_shot[f"{field}_{arm}"] = block[:, i]
        per_shot[f"predicted_{arm}"] = ((block[:, 6].astype(np.int64)[:, None] & BITS) > 0).astype(np.uint8)
        per_shot[f"logits_hash_{arm}"] = state.hashes[arm][:done]
        arms[arm] = summarise(block) if done else None
    np.savez_compressed(state.out / "per_shot.npz", **per_shot)
    with open(state.out / "summary.json", "w") as file:
        json.dump(extra | {"shots_scored": done, "arms": arms, "saved_utc": datetime.now(timezone.utc).isoformat()},
                  file, indent=1)


def decode_command(args):
    from predecoder import tcbp
    from predecoder.priors import load_settings

    with open(args.plan) as file:
        plan = json.load(file)
    setting = load_settings(args.settings)
    if setting != FROZEN or plan["knobs"] != FROZEN:
        raise SystemExit(f"the knobs are {setting} (plan: {plan['knobs']}), not the frozen {FROZEN}")
    if tuple(tcbp.FIELDS[:len(FIELDS)]) != FIELDS:
        raise SystemExit(f"predecoder.tcbp's row fields are {tcbp.FIELDS}, not {FIELDS} first")
    knobs = {arm: setting for arm in LOGITS}
    refit = None
    if args.refit:
        with open(args.refit) as file:
            refit = json.load(file)
        if refit.get("chosen") is None:
            raise SystemExit(f"{args.refit} chose nothing: the rf arm cannot run")
        knobs["rf"] = refit["chosen"]
    tasks, first = [], None
    for task in plan["tasks"]:
        tasks.append(load_task(task, plan, args.root, args.logits_root, first))
        first = first or tasks[-1].problem
        if "rf" in task["arms"] and refit is None:
            raise SystemExit(f"{task['name']} has the arm rf: give the refit with --refit")
    version = tcbp.check_environment(args.decoder)
    stop_at = datetime.fromisoformat(args.stop_at.replace("Z", "+00:00")).timestamp()
    start = time.time()
    print(f"plan {args.plan}: tasks {[(t.name, list(t.arms)) for t in tasks]}; decoder {args.decoder} ({version}), preset "
          f"{args.preset}; head knobs {setting}" + (f", rf {knobs['rf']} (from {args.refit})" if refit else "")
          + f", cap {PRIOR_CAP}; stop at {args.stop_at}; {tcbp.where(args.decoder)}", flush=True)
    builder = tcbp.Builder(first, args.decoder, args.preset)
    print(f"  model fitted in {builder.model_seconds:.1f} s (one model, shared by every task); GPU memory {tcbp.gpu_memory()}",
          flush=True)
    states = {}
    for t in tasks:
        shots = len(t.data.detectors) if args.limit is None else min(args.limit, len(t.data.detectors))
        out = Path(args.out) / t.name
        (out / "parts").mkdir(parents=True, exist_ok=True)
        states[t.name] = SimpleNamespace(
            shots=shots, out=out, done=0, syndromes=np.ascontiguousarray(t.data.basis_input[:shots], dtype=np.uint8),
            truth=np.asarray(t.data.observables[:shots], np.uint8),
            rows={arm: np.zeros((shots, len(tcbp.FIELDS))) for arm in t.arms},
            hashes={arm: np.zeros(shots, np.uint64) for arm in t.arms}, capped={arm: 0 for arm in t.arms},
            extra={"task": t.name, "data": t.data.identity(), "transfer": t.transfer, "plan": plan, "decoder": args.decoder,
                   "version": version, "preset": args.preset, "head_knobs": setting, "prior_cap": PRIOR_CAP,
                   "knobs_per_arm": {arm: knobs[arm] for arm in t.arms if arm in LOGITS},
                   "refit": None if refit is None else {"file": args.refit, "chosen": refit["chosen"]},
                   "logits": t.metas, "stop_at": args.stop_at, "block": BLOCK, "arm_order": list(t.arms),
                   "model_seconds": builder.model_seconds, "machine": tcbp.where(args.decoder),
                   "timer": "build: get_decoder with the shot's priors; decode: decode() and the cut to 0/1; one process"})
    warm, stopped = True, False
    for begin in range(0, max(s.shots for s in states.values()), BLOCK):
        for t in tasks:
            s = states[t.name]
            if begin >= s.shots:
                continue
            if time.time() > stop_at:
                stopped = True
                break
            end = min(begin + BLOCK, s.shots)
            for arm, spec in t.arms.items():
                path = s.out / "parts" / f"{arm}_{begin:06d}_{end:06d}.npz"
                if not path.exists():
                    before = tcbp.CAPPED["columns"]
                    part, part_hashes = tcbp.decode_shots(
                        builder, spec.problem, s.syndromes[begin:end], s.truth[begin:end],
                        None if spec.logits is None else spec.logits[begin:end], None if spec.logits is None else knobs[arm],
                        f"{t.name} {arm} {begin}-{end - 1}", warm)
                    s.capped[arm] += tcbp.CAPPED["columns"] - before
                    warm = False
                    np.savez(path, rows=part, hashes=part_hashes)
                with np.load(path) as file:
                    s.rows[arm][begin:end], s.hashes[arm][begin:end] = file["rows"], file["hashes"]
            s.done = end
            s.extra |= {"capped_columns": dict(s.capped), "seconds": time.time() - start}
            save_results(s, s.extra)
            print(f"  {t.name} block {begin}-{end - 1} done, {time.time() - start:.0f} s; failures so far "
                  + ", ".join(f"{arm} {int(s.rows[arm][:s.done, 0].sum())}" for arm in t.arms) + f" of {s.done}", flush=True)
        if stopped:
            print(f"  stop: {args.stop_at} passed; scored shots " + ", ".join(f"{n} {s.done}" for n, s in states.items()),
                  flush=True)
            break
    for name, s in states.items():
        save_results(s, s.extra | {"seconds": time.time() - start})
        print(f"saved {s.out / 'per_shot.npz'}: {s.done} shots scored", flush=True)
    print(f"done; {time.time() - start:.0f} s", flush=True)


# ---------- fit (server, TCBP environment) ----------

def fit_key(row):
    """The declared choice rule: fewest failures, then not reproduced, then flagged, then smaller alpha, larger eps."""
    return (row["failures"], row["not_reproduced"], row["flag_not_converged"], row["alpha"], -row["eps"])


def fit_command(args):
    from predecoder import tcbp

    with open(args.plan) as file:
        plan = json.load(file)
    if len(plan["tasks"]) != 1 or plan["tasks"][0]["arms"] != ["m", "r"] or plan.get("role") != "train":
        raise SystemExit("the fit's plan has one train task (plan --role train) with the arms m and r")
    t = load_task(plan["tasks"][0], plan, args.root, args.logits_root, None)
    version = tcbp.check_environment(args.decoder)
    stop_at = datetime.fromisoformat(args.stop_at.replace("Z", "+00:00")).timestamp()
    shots = len(t.data.detectors) if args.limit is None else min(args.limit, len(t.data.detectors))
    syndromes = np.ascontiguousarray(t.data.basis_input[:shots], dtype=np.uint8)
    truth = np.asarray(t.data.observables[:shots], np.uint8)
    out = Path(args.out)
    parts = out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(f"fit on {t.name} ({t.data.path.name}, {shots} train shots); base matched; grid {FIT_GRID}; decoder "
          f"{args.decoder} ({version}), preset {args.preset}; stop at {args.stop_at}; {tcbp.where(args.decoder)}", flush=True)
    builder = tcbp.Builder(t.problem, args.decoder, args.preset)
    print(f"  model fitted in {builder.model_seconds:.1f} s", flush=True)
    rows, warm, complete = [], True, True
    for setting in [None] + FIT_GRID:                    # None: arm m, the control
        if time.time() > stop_at:
            complete = False
            print(f"  stop: {args.stop_at} passed before {setting}; nothing is chosen", flush=True)
            break
        name = "m" if setting is None else f"alpha{setting['alpha']:g}_eps{setting['eps']:g}"
        path = parts / f"{name}.npz"
        arm = t.arms["m" if setting is None else "r"]
        if not path.exists():
            before = tcbp.CAPPED["columns"]
            part, hashes = tcbp.decode_shots(builder, arm.problem, syndromes, truth,
                                             None if setting is None else arm.logits[:shots], setting, f"fit {name}", warm)
            warm = False
            np.savez(path, rows=part, hashes=hashes, capped=tcbp.CAPPED["columns"] - before)
        with np.load(path) as file:
            part, capped_columns = file["rows"], int(file["capped"])
        flags = np.nan_to_num(part[:, 2], nan=1.0)
        row = {"arm": "m" if setting is None else "r", **(setting or {"mode": None, "alpha": 0.0, "eps": None}),
               "shots": shots, "failures": int(part[:, 0].sum()), "not_reproduced": int((part[:, 1] == 0).sum()),
               "flag_not_converged": int((flags == 0).sum()), "capped_columns": capped_columns,
               "build_decode_ms_mean": float((part[:, 4] + part[:, 5]).mean() * 1e3)}
        rows.append(row)
        print(f"  {name}: failures {row['failures']} of {shots}, not reproduced {row['not_reproduced']}, flagged "
              f"{row['flag_not_converged']}, capped {capped_columns}, {row['build_decode_ms_mean']:.0f} ms; "
              f"{time.time() - start:.0f} s", flush=True)
    candidates = [row for row in rows if row["arm"] == "r"]
    chosen = min(candidates, key=fit_key) if complete and len(candidates) == len(FIT_GRID) else None
    record = {"label": "exploration, task tcbp-retrain, the refit", "declared": plan["declared"], "plan": plan,
              "data": t.data.identity(), "logits": t.metas.get("r"), "decoder": args.decoder, "version": version,
              "preset": args.preset, "machine": tcbp.where(args.decoder), "model_seconds": builder.model_seconds,
              "grid": FIT_GRID, "rule": "fewest train failures; then fewer not reproduced; then fewer flagged not "
              "converged; then the smaller alpha; then the larger eps", "rows": rows, "complete": complete,
              "chosen": None if chosen is None else {key: chosen[key] for key in ("mode", "alpha", "eps")},
              "seconds": time.time() - start}
    with open(out, "w") as file:
        json.dump(record, file, indent=1)
    print(f"chosen: {record['chosen']}; saved {out}; {time.time() - start:.0f} s", flush=True)


# ---------- score (luminaty) ----------

def score_command(args):
    with open(args.plan) as file:
        plan = json.load(file)
    root, summary = Path(args.root), {"label": plan["label"], "declared": plan["declared"], "tasks": {}}
    for task in plan["tasks"]:
        with np.load(Path(args.decoded) / task["name"] / "per_shot.npz") as file:
            per_shot = {key: file[key] for key in file.files}
        done = int(per_shot["shots_scored"])
        with np.load(root / task["data"]) as file:
            observables = file["observables"][:done]
        if not np.array_equal(per_shot["truth"], observables):
            raise SystemExit(f"{task['name']}: the decoded truth is not the shot file's")
        failed = {arm: per_shot[f"failed_{arm}"] > 0 for arm in task["arms"]}
        rows = {arm: {"shots": done, "failures": int(failed[arm].sum()), "wilson95": wilson(int(failed[arm].sum()), done),
                      "not_reproduced": int((per_shot[f"reproduced_{arm}"] == 0).sum())} for arm in task["arms"]}
        pairs = []
        primary = [tuple(pair) for pair in plan.get("primary", PRIMARY)]
        for kind, listed in (("primary", primary), ("secondary", [q for q in SECONDARY if q not in primary])):
            for a, b in listed:
                if a in failed and b in failed:
                    only_a, only_b = int((failed[a] & ~failed[b]).sum()), int((failed[b] & ~failed[a]).sum())
                    pairs.append({"kind": kind, "a": a, "b": b, "shots": done, "failures_a": int(failed[a].sum()),
                                  "failures_b": int(failed[b].sum()), "only_a": only_a, "only_b": only_b,
                                  "sign_test_p": sign_test(only_a, only_b)})
        summary["tasks"][task["name"]] = {"shots_file": task["data"], "shots_sha256": file_sha256(root / task["data"]),
                                          "shots_scored": done, "of": task["shots"], "rows": rows, "pairs": pairs}
        print(f"{task['name']}: {done} of {task['shots']} shots scored")
        for arm, row in rows.items():
            print(f"  {arm:3s} {row['failures']:5d}  [{row['wilson95'][0]:.4f}, {row['wilson95'][1]:.4f}]  "
                  f"not reproduced {row['not_reproduced']}")
        for pair in pairs:
            print(f"  {pair['kind']:9s} {pair['a']} vs {pair['b']}: {pair['failures_a']} vs {pair['failures_b']}; only "
                  f"{pair['a']} {pair['only_a']}, only {pair['b']} {pair['only_b']}; sign test p = {pair['sign_test_p']:.3g}")
    with open(args.out, "w") as file:
        json.dump(summary, file, indent=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--root", required=True)
    plan.add_argument("--task", action="append", required=True, help="name:folder:arms, the arms comma-separated")
    plan.add_argument("--kit-head", required=True, help="sha256 of the kit-trained head")
    plan.add_argument("--retrained-head", required=True, help="sha256 of the retrained head")
    plan.add_argument("--out", required=True)
    plan.add_argument("--declared", default="neutral_atom_noise/tcbp_retrain_declared.md")
    plan.add_argument("--primary", help="the plan's primary pairs, a:b comma-separated (default: r:m,r:h)")
    plan.add_argument("--role", help="'train' for the plan of a fit")
    decode = sub.add_parser("decode")
    for name in ("plan", "root", "logits_root", "settings", "stop_at", "out"):
        decode.add_argument(f"--{name.replace('_', '-')}", required=True)
    decode.add_argument("--decoder", default="tcbp-jobs", choices=("tcbp-jobs", "relay"))
    decode.add_argument("--preset", default="tcbp-jobs")
    decode.add_argument("--limit", type=int, help="only the first this many shots of each task (dry runs)")
    decode.add_argument("--refit", help="fit.json of `fit`: the knobs of the rf arm")
    fit = sub.add_parser("fit")
    for name in ("plan", "root", "logits_root", "stop_at", "out"):
        fit.add_argument(f"--{name.replace('_', '-')}", required=True)
    fit.add_argument("--decoder", default="tcbp-jobs", choices=("tcbp-jobs", "relay"))
    fit.add_argument("--preset", default="tcbp-jobs")
    fit.add_argument("--limit", type=int, help="only the first this many train shots (dry runs)")
    score = sub.add_parser("score")
    for name in ("plan", "root", "decoded", "out"):
        score.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args(argv)
    {"plan": plan_command, "decode": decode_command, "fit": fit_command, "score": score_command}[args.command](args)


if __name__ == "__main__":
    main()
