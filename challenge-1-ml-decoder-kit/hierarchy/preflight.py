"""Preflight: every tool of this folder on a few VALIDATION shots, before a run on held-out shots.

Run from challenge-1-ml-decoder-kit/, in the environment of part 1 (PRIORTILT_REPO set):

  python -m hierarchy.preflight --options runs/hierarchy_options.json \\
      --x data/p0.004/validation_X.npz --z data/p0.004/validation_Z.npz --rule f \\
      --tcbp-python /home/blakew3/envs/tcbp-py312/bin/python --out runs/preflight

It exits with status 0 and the last line "PREFLIGHT PASSED" only if every step worked. Any
failure stops it at once with a non-zero status. A file named test_* or go_* is refused: the
preflight never opens held-out shots.

The steps, on the first --shots shots (default 16) of each file:

  1. the plugin, made through the evaluator's own create_session with exactly the given
     options file: every shot is decoded and its result checked. Both bases must have a
     checkpoint (unless --allow-weight-rule), and each network is loaded through the plugin
     and asked once, also for a basis whose chosen rule does not use it;
  2. hierarchy.candidates on the slices (P1, P2, R);
  3. neural.evaluate_stim --save with each checkpoint of the options file (the logits);
  4. hierarchy.tcbp_member in the Python given by --tcbp-python, as a subprocess (T). On a
     machine without a GPU, --tcbp-stand-in runs its plumbing mode instead;
  5. hierarchy.ensemble --write-selection, then hierarchy.joint_member on shot 0 of each slice (J);
  6. hierarchy.ensemble with every member, the logits and --rule, and the pair score from
     its two saved files. The marked rule must have been measured for both bases.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from evaluation.adapters import create_session, validate_shot_result
from hierarchy import candidates, ensemble, joint_member, tcbp_member
from hierarchy.rules import bits_for_network, most_likely
from hierarchy.stages import kit_problem, load_shots

PLUGIN = "hierarchy.plugin:make_decoder"


def step(text):
    print(f"\n--- {text}", flush=True)


def write_slice(shots, count, path):
    """The first shots of a shot file, in the same format."""
    np.savez_compressed(path, p=shots.p, seed=shots.seed, basis=shots.basis, detectors=shots.detectors[:count],
                        observables=shots.observables[:count], basis_detectors=shots.basis_detectors)
    return path


def check_plugin(slices, options, allow_weight_rule):
    for basis, path in slices.items():
        data = load_shots(path)
        session = create_session(PLUGIN, kit_problem(basis, data.p), options)
        asked, used = session.settings["rule_asked"], session.settings["rule"]
        has_checkpoint = basis in (options.get("checkpoints") or {})
        if (asked != used or not has_checkpoint) and not allow_weight_rule:
            raise SystemExit(f"{basis}: no network can be asked for this basis: the options hold no checkpoint for it "
                             f"(rule asked: {asked}, rule used: {used}). Pass --allow-weight-rule only if that is meant.")
        stages = []
        for full, part in zip(data.detectors, data.basis_input, strict=True):
            result = session.decode_shot(full, part)
            validate_shot_result(result)
            stages.append(result.metrics["stage"])
        json.dumps(session.settings)                      # the evaluator writes the settings as JSON
        line = f"{basis}: rule {used}, {len(stages)} shots decoded, stages {sorted(set(stages))}, set-up {session.setup_seconds:.1f} s"
        if session.network is None and has_checkpoint:    # the chosen rule does not use the network: load and ask it all the same
            session = create_session(PLUGIN, kit_problem(basis, data.p), options | {"rule": "network"})
            line += "; the chosen rule does not use the network, which was loaded through the plugin and asked once all the same"
        if session.network is not None:                   # few shots rarely reach the network: ask it once directly
            logits, answer = session.network(data.detectors[0], data.basis_input[0])
            three = np.stack([answer, 1 - answer, np.roll(answer, 1)])[:, None, :]
            own = bits_for_network(three, (logits > 0)[None], answer[None], session.network.model_from_kit)
            if logits.shape != (12,) or not np.isfinite(logits).all() or int(most_likely(own, logits[None])[0]) != 0:
                raise SystemExit(f"{basis}: the network's answer does not score highest among its own candidates: logits {logits}")
            record = session.settings["network"]
            line += (f"; NETWORK ASKED: step {record['step']}, sha256 {record['sha256'][:12]}..., operators {record['logical_operators']}, "
                     f"on {record['device']}")
        elif not allow_weight_rule:
            raise SystemExit(f"{basis}: the network was not asked")
        print(line, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--options", required=True, help="the plugin options JSON of the real run")
    parser.add_argument("--x", required=True, help="a VALIDATION shot file of X memory")
    parser.add_argument("--z", required=True, help="a VALIDATION shot file of Z memory, at the same p")
    parser.add_argument("--out", required=True, help="a new or empty folder for the preflight's files")
    parser.add_argument("--rule", default="a", choices=[rule[0] for rule in ensemble.RULES], help="the ensemble rule of the real run")
    parser.add_argument("--shots", type=int, default=16, help="shots per basis (default 16; neural.evaluate_stim needs 12 or more)")
    parser.add_argument("--tcbp-python", help="the Python of the TCBP environment, for hierarchy.tcbp_member")
    parser.add_argument("--tcbp-stand-in", action="store_true", help="no GPU here: run tcbp_member's plumbing mode with relay-BP instead")
    parser.add_argument("--tcbp-batch", action="store_true", help="run tcbp_member with --batch (the real run on held-out shots is serial)")
    parser.add_argument("--allow-weight-rule", action="store_true", help="accept a basis without a checkpoint")
    parser.add_argument("--processes", type=int, default=2)
    args = parser.parse_args(argv)
    if bool(args.tcbp_python) == args.tcbp_stand_in:
        raise SystemExit("give --tcbp-python (the real decoder) or --tcbp-stand-in (no GPU), one of them")
    if args.shots < 12:
        raise SystemExit("--shots must be 12 or more: neural.evaluate_stim leaves its first 10 shots out of the latency it reports")

    start = time.perf_counter()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: the preflight wants a new folder, so that nothing old can pass for new")
    out.mkdir(parents=True, exist_ok=True)
    options = json.loads(Path(args.options).read_text(encoding="utf-8"))

    step(f"0. the first {args.shots} shots of each file")
    files = {"X": load_shots(args.x), "Z": load_shots(args.z)}     # refuses held-out files
    if files["X"].basis != "X" or files["Z"].basis != "Z" or files["X"].p != files["Z"].p:
        raise SystemExit("--x must be an X memory file and --z a Z memory file, at the same p")
    slices = {basis: write_slice(shots, args.shots, out / f"preflight_{basis}.npz") for basis, shots in files.items()}
    print(f"p = {files['X'].p:g}; from {args.x} (sha256 {files['X'].sha256}) and {args.z} (sha256 {files['Z'].sha256})")

    step("1. the plugin through the evaluator's create_session, with the given options")
    check_plugin(slices, options, args.allow_weight_rule)

    step("2. hierarchy.candidates")
    candidates.main(["--data", *map(str, slices.values()), "--out", str(out / "cand"), "--processes", str(args.processes)])

    step("3. neural.evaluate_stim with the checkpoints of the options")
    logits = {}
    for basis, checkpoint in (options.get("checkpoints") or {}).items():
        from neural import evaluate_stim

        evaluate_stim.main(["--checkpoint", checkpoint, "--data", str(slices[basis]), "--device", options.get("device", "cpu"),
                            "--latency-shots", "3", "--save", str(out / f"net_{basis}")])
        logits[basis] = str(out / f"net_{basis}.npz")

    step("4. hierarchy.tcbp_member" + (" (the plumbing stand-in: relay-BP, NOT TCBP-JOBS)" if args.tcbp_stand_in else f" in {args.tcbp_python}"))
    command = ["--data", *map(str, slices.values()), "--out", str(out / "tcbp")] + (["--batch"] if args.tcbp_batch else [])
    if args.tcbp_stand_in:
        tcbp_member.main([*command, "--decoder", "relay"])
    else:
        subprocess.run([args.tcbp_python, "-m", "hierarchy.tcbp_member", *command], check=True)
    suffix = tcbp_member.SUFFIX["relay" if args.tcbp_stand_in else "tcbp-jobs"]

    step("5. hierarchy.ensemble --write-selection, and hierarchy.joint_member on shot 0 of each slice")
    members = {}
    for basis, path in slices.items():
        members[basis] = [str(out / "cand" / f"{path.stem}.chunks.npz"), str(out / "tcbp" / f"{path.stem}.{suffix}.npz")]
        selection = out / f"{path.stem}.selection.npz"
        ensemble.main(["--members", *members[basis], "--write-selection", str(selection)])
        joint_member.listed_shots(selection, load_shots(path), path)       # the selection file is readable and belongs to the slice
        joint_member.main(["--data", str(path), "--shots", "0", "--out", str(out / "joint"), "--processes", "1"])
        members[basis].append(str(out / "joint" / f"{path.stem}.joint.npz"))

    step(f"6. hierarchy.ensemble with every member and --rule {args.rule}, then the pair score")
    for basis, path in slices.items():
        ensemble.main(["--members", *members[basis], *(["--logits", logits[basis]] if basis in logits else []),
                       "--rule", args.rule, "--save", str(out / f"ensemble_{basis}.json")])
        saved = json.loads((out / f"ensemble_{basis}.json").read_text(encoding="utf-8"))
        row = next(row for row in saved["rows"] if row["key"] == args.rule)
        if "failures" not in row:
            raise SystemExit(f"{basis}: the rule {args.rule} was not measured: {row['note']}")
    ensemble.main(["--pair", str(out / "ensemble_X.npz"), str(out / "ensemble_Z.npz")])

    print(f"\nPREFLIGHT PASSED in {time.perf_counter() - start:.0f} s: the plugin, candidates, evaluate_stim, tcbp_member"
          + (" (STAND-IN ONLY)" if args.tcbp_stand_in else "") + f", joint_member and ensemble all ran on {args.shots} validation shots per basis")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(f"PREFLIGHT FAILED: {error}")
