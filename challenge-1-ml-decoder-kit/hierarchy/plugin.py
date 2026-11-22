"""The hierarchy as a decoder for the team's evaluator.

Run from challenge-1-ml-decoder-kit/:

  python -m evaluation.evaluate --test-x <X shots> --test-z <Z shots> \\
      --decoder hierarchy.plugin:make_decoder --plugin-options <options.json> --out <folder>

What one shot costs, in order, all inside decode_shot (the evaluator's timer wraps all of it):

  1. PriorTilt under the first seed            -> P1
  2. PriorTilt under the second seed           -> P2
  3. if P1 and P2 predict the same 12 bits: that is the answer
  4. otherwise relay-BP at the kit's settings  -> R
  5. the rule chooses among R, P1 and P2:
       "weight"   the lowest weight (the sum of log((1 - prior) / prior) over a correction's faults)
       "converged"  the lowest weight among the candidates that converged (among all three if
                  none did): the best rule without learning on the exploration shots
       "network"  the candidate the neural network finds most likely: the largest sum over the
                  12 bits of log sigmoid(logit) for a predicted 1 and log sigmoid(-logit) for a
                  predicted 0. With a constant C, the score is that sum minus weight / C.

The three decoders read the 936 basis bits. The network reads the shot's detector bits (all
1,728, or the 936 if the checkpoint was trained on those) and nothing else. It is loaded once,
when the session is made. The truth never reaches a session.

A network trained with `--logicals light` (neural/, used for Z memory) predicts 12 other
logical operators than the kit's. Its 12 logits are then compared with each candidate's bits
in those operators (rules.bits_for_network), so the rule is the same for both kinds of network.

Options (a JSON object; every key is optional):

  rule            "network" (default), "weight" or "converged"; or one per basis: {"X": "network", "Z": "converged"}
  checkpoints     {"X": <checkpoint of neural.train>, "Z": <...>}. A basis without a checkpoint
                  uses the "weight" rule, and the session's recorded settings say so.
  seeds           PriorTilt's two seeds, default [1, 1001]
  constant        C of the "network" rule, default none. Fit it on other shots (hierarchy.select).
  device          for the network: "cpu" (default), "cuda" or "auto"
  torch_threads   CPU threads for the network (default: torch's own choice)
  priortilt_repo  a clone of PriorTilt's repository (default: the environment variable PRIORTILT_REPO)
  priortilt_max_rounds, relay_sets   smaller budgets, for quick tests only

A session decodes its shots in order on decoders built once, and both decoders carry state.
So relay-BP sees only the shots where P1 and P2 differ, and the answers are those of
`python -m hierarchy.candidates --mode session`, not those of its chunk mode.
"""
from time import perf_counter

import numpy as np

from evaluation.adapters import ShotResult
from hierarchy.rules import ORDER, bits_for_network, lowest_weight, most_likely
from hierarchy.stages import DEFAULT_SEEDS, Hierarchy, decoder_settings, problem_sha256, sha256_of

OPTIONS = {"rule", "checkpoints", "seeds", "constant", "device", "torch_threads", "priortilt_repo",
           "priortilt_max_rounds", "relay_sets"}
RULES = ("network", "weight", "converged")


class Network:
    """The neural decoder of one memory basis, loaded once.

    Calling it with one shot gives (logits, kit bits): the 12 logits of the network's own
    logical operators, and the kit's 12 observable bits of the network's answer.
    model_from_kit turns kit bits into the network's bits (rules.bits_for_network); it is None
    when the network was trained on the kit's observables.
    """

    def __init__(self, checkpoint, problem, device="cpu", threads=None):
        import torch

        from neural.graph import gf2_inverse
        from neural.train import load_decoder     # the checkpoint is read by neural/'s own loader

        if threads:
            torch.set_num_threads(int(threads))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch, self.device = torch, torch.device(device)
        self.model, saved = load_decoder(checkpoint, self.device)
        config = saved["config"]
        if config["basis"] != problem.basis:
            raise ValueError(f"{checkpoint} decodes {config['basis']} memory, the session decodes {problem.basis} memory")
        self.all_bits, self.basis_bits = problem.circuit.num_detectors, problem.H.shape[0]
        if self.model.num_inputs not in (self.all_bits, self.basis_bits):
            raise ValueError(f"{checkpoint} reads {self.model.num_inputs} detector bits; the shots have "
                             f"{self.all_bits}, of which {self.basis_bits} on {problem.basis} checks")
        self.model_from_kit = None
        if config["logicals"] != "kit":
            self.model_from_kit = gf2_inverse(self.model.kit_from_model.cpu().numpy().astype(np.uint8))
        self.record = {"checkpoint": str(checkpoint), "sha256": sha256_of(checkpoint), "step": saved["step"],
                       "commit": saved["commit"], "training_seed": config["seed"], "reads_detectors": config["detectors"],
                       "logical_operators": config["logicals"], "width": config["width"], "depth": config["depth"],
                       "device": str(self.device), "torch": torch.__version__, "torch_threads": torch.get_num_threads()}
        self(np.zeros(self.all_bits, np.uint8), np.zeros(self.basis_bits, np.uint8))   # so the first real call pays no one-time cost

    def __call__(self, full, basis):
        bits = full if self.model.num_inputs == self.all_bits else basis
        with self.torch.no_grad():
            shot = self.torch.from_numpy(np.ascontiguousarray(bits, dtype=np.uint8)[None]).to(self.device)
            logits = self.model(shot).float()
            kit_bits = self.model.kit_bits(shot, logits > 0)      # neural/'s own change to the kit's observables
        return logits.cpu().numpy()[0], kit_bits.cpu().numpy()[0]


class HierarchySession:
    """One memory basis: PriorTilt twice; where the two differ, relay-BP joins and a rule chooses."""

    def __init__(self, problem, options, hierarchy=None, network=None):
        """hierarchy and network replace the real stages and the real network. Tests use them."""
        start = perf_counter()
        unknown = set(options) - OPTIONS
        if unknown:
            raise ValueError(f"unknown options {sorted(unknown)}; known: {sorted(OPTIONS)}")
        asked = options.get("rule", "network")
        if isinstance(asked, dict):                       # one rule per memory basis
            asked = asked.get(problem.basis, "network")
        if asked not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {asked!r}")
        self.problem = problem
        seeds = list(options.get("seeds", DEFAULT_SEEDS))
        decoders = None
        if hierarchy is None:
            relay_changes = {"num_sets": options["relay_sets"]} if options.get("relay_sets") is not None else None
            decoders = decoder_settings(problem.p, seeds, options.get("priortilt_repo"),
                                        options.get("priortilt_max_rounds"), relay_changes)
            hierarchy = Hierarchy(problem, decoders)
        self.hierarchy = hierarchy

        self.rule, note = asked, None
        checkpoint = (options.get("checkpoints") or {}).get(problem.basis)
        if asked == "network" and network is None:
            if checkpoint:
                network = Network(checkpoint, problem, options.get("device", "cpu"), options.get("torch_threads"))
            else:
                self.rule = "weight"
                note = f"no checkpoint was given for {problem.basis} memory, so the weight rule is used in place of the network rule"
        self.network = network if self.rule == "network" else None
        self.constant = options.get("constant") if self.rule == "network" else None
        self.settings = {
            "decoder": "hierarchy: PriorTilt under two seeds; where their predictions differ, relay-BP joins and the rule chooses",
            "rule": self.rule, "rule_asked": asked, "rule_note": note, "constant": self.constant, "seeds": seeds,
            "tie_order": list(ORDER), "decoders": decoders, "network": getattr(self.network, "record", None),
            "problem_sha256": problem_sha256(problem),
            "state": "decoders are built once and keep their state: relay-BP sees only the shots where P1 and P2 differ",
        }
        self.setup_seconds = perf_counter() - start

    def decode_shot(self, full, basis):
        syndrome = np.ascontiguousarray(basis, dtype=np.uint8)
        if syndrome.shape != (self.problem.H.shape[0],):
            raise ValueError("the basis syndrome does not have one bit per row of H")
        first, second = self.hierarchy.first_two(syndrome)
        if np.array_equal(first.predicted, second.predicted):
            return ShotResult(first.predicted, self._metrics("agree", first, [first, second]))

        third = self.hierarchy.third(syndrome)
        by_name = {candidate.name: candidate for candidate in (first, second, third)}
        candidates = [by_name[name] for name in ORDER]
        # A later stage for the shots where no candidate converged belongs here: it would take
        # the shot when `not any(candidate.converged for candidate in candidates)`.
        predicted = np.stack([candidate.predicted for candidate in candidates])[:, None, :]   # [candidates, 1 shot, 12]
        weight = np.array([[candidate.weight] for candidate in candidates])
        extra = {f"predicted_{candidate.name}": candidate.predicted.tolist() for candidate in candidates}
        if self.rule == "network":
            start = perf_counter()
            logits, network_answer = self.network(full, syndrome)
            logits, network_answer = np.asarray(logits, np.float64), np.asarray(network_answer, np.uint8)
            extra |= {"seconds_network": perf_counter() - start, "logits": logits.tolist(),
                      "network_answer": network_answer.tolist()}
            in_its_bits = bits_for_network(predicted, (logits > 0)[None], network_answer[None], self.network.model_from_kit)
            choice = int(most_likely(in_its_bits, logits[None], weight, self.constant)[0])
        else:
            converged = np.array([[candidate.converged] for candidate in candidates]) if self.rule == "converged" else None
            choice = int(lowest_weight(weight, converged)[0])
        return ShotResult(candidates[choice].predicted, self._metrics(self.rule, candidates[choice], candidates, extra))

    @staticmethod
    def _metrics(stage, winner, candidates, extra=None):
        """What happened on one shot. The evaluator averages converged, iterations and fallback_used."""
        metrics = {"stage": stage,                       # which stage answered: "agree", or the rule's name
                   "agreed": stage == "agree",           # P1 and P2 predicted the same bits
                   "fallback_used": stage != "agree",    # relay-BP and the rule ran
                   "winner": winner.name,
                   "converged": bool(winner.converged),
                   "any_converged": any(candidate.converged for candidate in candidates),
                   "iterations": int(sum(candidate.iterations for candidate in candidates))}
        for candidate in candidates:
            metrics |= {f"converged_{candidate.name}": bool(candidate.converged),
                        f"iterations_{candidate.name}": int(candidate.iterations),
                        f"weight_{candidate.name}": float(candidate.weight),
                        f"seconds_{candidate.name}": float(candidate.seconds)}
        return metrics | (extra or {})


def make_decoder(problem, options):
    """The factory the evaluator calls, once per memory basis."""
    return HierarchySession(problem, options)
