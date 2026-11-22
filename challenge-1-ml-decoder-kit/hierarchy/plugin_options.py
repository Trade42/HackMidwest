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
"""
import argparse
import json
from pathlib import Path

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


def choose(basis, candidates, logits, checkpoint=None):
    """(rule, line) for one basis: "network" if it fails no more often than "converged" on these shots."""
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
    known = sorted({step for step in steps if step is not None})
    return rule, (f"PLUGIN_RULE {basis} {rule}    (network {network} failures, converged {converged}, on {len(pool['truth'])} shots; "
                  f"logits of checkpoint step {', '.join(map(str, known)) if known else 'not recorded'})")


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
    args = parser.parse_args(argv)

    rules, checkpoints = {}, {}
    for basis in "XZ":
        checkpoint = getattr(args, f"{basis.lower()}_checkpoint")
        if checkpoint:
            checkpoints[basis] = checkpoint
            print(f"{basis} checkpoint {checkpoint}, sha256 {sha256_of(checkpoint)}")
            rules[basis], line = choose(basis, getattr(args, f"{basis.lower()}_candidates"), getattr(args, f"{basis.lower()}_logits"), checkpoint)
        else:
            rules[basis], line = "converged", f"PLUGIN_RULE {basis} converged    (no checkpoint was given for this basis)"
        print(line, flush=True)
    options = {"rule": rules, "seeds": args.seeds, "checkpoints": checkpoints, "device": args.device, "torch_threads": args.torch_threads}
    if args.parallel:
        options["parallel"] = True
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(options, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {json.dumps(options)}")
    return options


if __name__ == "__main__":
    main()
