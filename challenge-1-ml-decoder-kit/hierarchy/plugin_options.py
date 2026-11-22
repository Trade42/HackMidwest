"""Write the plugin's options file: per memory basis, the network rule only if it earns its place.

The frozen procedure: per basis, the plugin's rule is "network" if, with the given checkpoint,
the network rule has no more failures than the rule "converged" (the best rule without
learning) on the given exploration shots of that basis; otherwise it is "converged". The shots
are the p = 0.004 train and validation files; no held-out file is involved.

Run from challenge-1-ml-decoder-kit/:

  python -m hierarchy.plugin_options --out runs/hierarchy_options.json \\
      --x-checkpoint <X checkpoint> --x-candidates lap/selector_validation_X.npz lap/selector_train_X.npz \\
      --x-logits netX/validation_X.npz netX/train_X.npz \\
      --z-checkpoint <Z checkpoint> --z-candidates lap/selector_validation_Z.npz lap/selector_train_Z.npz \\
      --z-logits netZ/validation_Z.npz netZ/train_Z.npz

The candidates are files of hierarchy.candidates (chunk mode) or the laptop's selector files;
the logits are what `python -m neural.evaluate_stim --checkpoint <the same checkpoint> --save`
wrote for the same shot files, in the same order. Where the .json beside a logits file names
the step of its checkpoint, that step must be the given checkpoint's, or the command stops:
logits of another checkpoint would choose the rule for the wrong network. Per basis it prints
one line to parse:

  PLUGIN_RULE X network    (network 62 failures, converged 67, on 25000 shots; logits of checkpoint step 60000)

and it writes the options file for hierarchy.plugin with those rules, both checkpoints, the
seeds and the device. A basis given without logits gets the rule "converged".

--network-first also fixes, per basis, the threshold of the plugin's network-first stage (the
network answers alone where its confidence, the smallest |logit| of its 12 outputs, is at
least the threshold). The procedure: on the same shots, the smallest threshold at which that
rule has no more failures than the basis' chosen rule; none if no threshold does. It prints

  NETWORK_FIRST X 4.123    (the network answers 41.2% of 25000 shots; failures 59 with it, 59 without)

and writes {"network_first": {"X": 4.123, "Z": null}} into the options file. The frozen
rule's failures here come from the candidate files (chunk mode), the network's from the
logits files' own answers ("predictions", the kit's 12 bits).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from hierarchy import select
from hierarchy.stages import DEFAULT_SEEDS, sha256_of


def logits_steps(logits):
    """The checkpoint step each logits file names in the .json beside it (None where there is no .json)."""
    steps = []
    for path in logits:
        numbers = Path(path).with_suffix(".json")
        steps.append(json.loads(numbers.read_text(encoding="utf-8")).get("checkpoint_step") if numbers.is_file() else None)
    return steps


def checkpoint_step(checkpoint):
    import torch

    return int(torch.load(checkpoint, map_location="cpu", weights_only=True)["step"])


def choose(basis, candidates, logits, checkpoint=None, per_shot=None):
    """(rule, line) for one basis: "network" if it fails no more often than "converged" on these shots.

    per_shot, a dict, receives the chosen rule's failed mask, the network's own failed mask and
    its confidence, per shot, for network_first_threshold.
    """
    if not candidates or not logits:
        return "converged", f"PLUGIN_RULE {basis} converged    (no candidates or logits were given for this basis)"
    steps = logits_steps(logits)
    if checkpoint and any(step is not None for step in steps):
        step = checkpoint_step(checkpoint)
        if any(found is not None and found != step for found in steps):
            raise SystemExit(f"{basis}: the logits come from checkpoint step(s) {steps}, the given checkpoint is at step {step}. "
                             "Make the logits again with this checkpoint.")
    select.refuse_held_out_fit(candidates, logits)
    pool, sources = select.load_candidates(candidates, basis)
    if any(source["basis"] != basis for source in sources):
        raise SystemExit(f"a candidates file given for {basis} holds another basis")
    by_key = {row.key: row for row in select.score(pool, select.load_logits(logits, pool, sources))}
    network, converged = int(by_key["network"].failed.sum()), int(by_key["weight_converged"].failed.sum())
    rule = "network" if network <= converged else "converged"
    if per_shot is not None:
        answers, confidence = [], []
        for path in logits:
            with np.load(path, allow_pickle=False) as file:
                answers.append(file["predictions"].astype(np.uint8))
                confidence.append(np.abs(file["logits"].astype(np.float64)).min(axis=1))
        per_shot |= {"frozen_failed": by_key["network" if rule == "network" else "weight_converged"].failed,
                     "network_failed": (np.concatenate(answers) != pool["truth"]).any(axis=1),
                     "confidence": np.concatenate(confidence)}
    known = sorted({step for step in steps if step is not None})
    return rule, (f"PLUGIN_RULE {basis} {rule}    (network {network} failures, converged {converged}, on {len(pool['truth'])} shots; "
                  f"logits of checkpoint step {', '.join(map(str, known)) if known else 'not recorded'})")


def network_first_threshold(frozen_failed, network_failed, confidence):
    """(threshold or None, share of shots the network answers, failures with the stage).

    The smallest threshold c such that answering with the network where confidence >= c, and
    with the frozen rule elsewhere, fails no more often than the frozen rule alone on these
    shots. None when no threshold does (the network then never answers alone).
    """
    frozen_failed, network_failed = np.asarray(frozen_failed, bool), np.asarray(network_failed, bool)
    confidence = np.asarray(confidence, np.float64)
    order = np.argsort(-confidence, kind="stable")
    ranked = confidence[order]
    change = np.cumsum(network_failed[order].astype(int) - frozen_failed[order].astype(int))   # failures added by covering the first k + 1
    last_of_value = np.append(ranked[:-1] > ranked[1:], True)       # a threshold covers all shots of equal confidence
    allowed = np.flatnonzero((change <= 0) & last_of_value)
    if not len(allowed):
        return None, 0.0, int(frozen_failed.sum())
    threshold = float(ranked[allowed.max()])
    covered = confidence >= threshold
    return threshold, float(covered.mean()), int(np.where(covered, network_failed, frozen_failed).sum())


def apply_threshold(threshold, frozen_failed, network_failed, confidence):
    """(share answered by the network, failures) of a given threshold on other shots."""
    covered = np.zeros(len(confidence), bool) if threshold is None else np.asarray(confidence) >= threshold
    return float(covered.mean()), int(np.where(covered, network_failed, frozen_failed).sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for basis in "xz":
        parser.add_argument(f"--{basis}-checkpoint", help=f"the {basis.upper()} network's checkpoint")
        parser.add_argument(f"--{basis}-candidates", nargs="+", default=[], help=f"candidate files of {basis.upper()} exploration shots")
        parser.add_argument(f"--{basis}-logits", nargs="+", default=[], help="the logits of the same shots, from the same checkpoint")
    parser.add_argument("--out", required=True, help="the options file to write")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs=2, default=list(DEFAULT_SEEDS))
    parser.add_argument("--parallel", action="store_true", help="also set the plugin's option for two PriorTilt decodes side by side")
    parser.add_argument("--network-first", action="store_true", help="also fix the threshold of the plugin's network-first stage per basis")
    args = parser.parse_args(argv)

    rules, checkpoints, thresholds = {}, {}, {}
    for basis in "XZ":
        checkpoint, per_shot = getattr(args, f"{basis.lower()}_checkpoint"), {}
        if checkpoint:
            checkpoints[basis] = checkpoint
            print(f"{basis} checkpoint {checkpoint}, sha256 {sha256_of(checkpoint)}")
            rules[basis], line = choose(basis, getattr(args, f"{basis.lower()}_candidates"), getattr(args, f"{basis.lower()}_logits"),
                                        checkpoint, per_shot)
        else:
            rules[basis], line = "converged", f"PLUGIN_RULE {basis} converged    (no checkpoint was given for this basis)"
        print(line, flush=True)
        if args.network_first:
            if per_shot:
                thresholds[basis], share, failures = network_first_threshold(**per_shot)
                shown = "none" if thresholds[basis] is None else f"{thresholds[basis]:.6g}"
                print(f"NETWORK_FIRST {basis} {shown}    (the network answers {share:.1%} of {len(per_shot['confidence'])} shots; "
                      f"failures {failures} with it, {int(per_shot['frozen_failed'].sum())} without)", flush=True)
            else:
                thresholds[basis] = None
                print(f"NETWORK_FIRST {basis} none    (no checkpoint, candidates or logits for this basis)", flush=True)
    options = {"rule": rules, "seeds": args.seeds, "checkpoints": checkpoints, "device": args.device, "torch_threads": args.torch_threads}
    if args.network_first:
        options["network_first"] = thresholds
    if args.parallel:
        options["parallel"] = True
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(options, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {json.dumps(options)}")
    return options


if __name__ == "__main__":
    main()
