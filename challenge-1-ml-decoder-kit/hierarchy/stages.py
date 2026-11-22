"""The decoders of the hierarchy on the kit's problem, and the order they run in.

Three decoders, each on the kit's own H, L and priors of one memory basis:

  P1, P2   PriorTilt (Ye and Delfosse, arXiv:2611.13373) under two seeds, with the paper's
           schedule for this code at the shots' p
  R        relay-BP, the kit's relay_decoder at the kit's settings

PriorTilt is a dependency, not part of this repository (licence CC BY-NC-SA 4.0). Its compiled
core is the Python module priortilt_rs. Its tuned schedule is read at run time from a clone of
https://github.com/ionq-publications/PriorTiltDecoder, named by the option priortilt_repo or the
environment variable PRIORTILT_REPO. README.md has the install commands.

Both decoders keep state from one decode to the next (relay-BP its random memory strengths,
PriorTilt its random stream), so an answer depends on which shots the decoder saw before.
"""
import hashlib
import importlib.util
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit
from decoders import RELAY_SETTINGS, relay_decoder
from evaluation.adapters import Problem
from evaluation.data import load_dataset

HELD_OUT = ("test_", "go_")       # the team's held-out shot files, by the start of their name
DEFAULT_SEEDS = (1, 1001)         # PriorTilt's two seeds
PRIORTILT_ENV = "PRIORTILT_REPO"
PRIORTILT_COMMIT = "8eefd37d694b5942fe60ec6d5c82ce97f3d2a450"   # the commit every number in README.md was made with
PRIORTILT_CODE = "BB144"          # PriorTilt's name for the Gross code


# ---------- the shots and the problem ----------

def load_shots(path, final=False):
    """A shot file of the kit's make_dataset.py. A held-out file is refused, by its name, before it is opened."""
    path = Path(path)
    if path.name.startswith(HELD_OUT) and not final:
        raise SystemExit(f"{path} is a held-out file (test_* or go_*). It is decoded once, after every choice is "
                         "frozen: pass the flag that says this run is final.")
    return load_dataset(path)


def kit_problem(basis, p):
    """The kit's decoding problem of one memory basis at noise strength p: H, L, priors and the circuit."""
    circuit = load_circuit(GROSS_CIRCUITS[basis], p)
    H, L, priors = dem_matrices(circuit, basis)
    return Problem(basis, H, L, priors, p, circuit)


def syndromes(shots, problem):
    """[shots, 936] uint8: the detector bits the three decoders read, in the order of the rows of H."""
    if not np.array_equal(shots.basis_detectors, detector_ids(problem.circuit, problem.basis)):
        raise ValueError(f"{shots.path}: basis_detectors are not the {problem.basis} checks of the kit's circuit")
    return np.ascontiguousarray(shots.basis_input, dtype=np.uint8)


def problem_sha256(problem):
    """One hash of H, L and the priors, to show that two machines decoded the same problem."""
    digest = hashlib.sha256()
    for matrix in (problem.H.tocsc().sorted_indices(), problem.L.tocsc().sorted_indices()):
        digest.update(np.asarray(matrix.shape, np.int64).tobytes())
        digest.update(matrix.indptr.astype(np.int64).tobytes())
        digest.update(matrix.indices.astype(np.int64).tobytes())
    digest.update(np.asarray(problem.priors, np.float64).tobytes())
    return digest.hexdigest()


# ---------- the settings of the three decoders ----------

def priortilt_settings(p, repo=None, max_rounds=None):
    """PriorTilt's tuned schedule for the Gross code at p, read from a clone of its repository.

    max_rounds replaces the paper's budget of 3,000 rounds. It exists for quick tests: every
    measured number uses the paper's budget.
    """
    repo = repo or os.environ.get(PRIORTILT_ENV)
    params_file = Path(repo) / "priortilt" / "params.py" if repo else None
    if params_file is None or not params_file.is_file():
        raise SystemExit("PriorTilt's tuned schedule is read from a clone of its repository. Give the clone's folder "
                         f"with the option priortilt_repo or the environment variable {PRIORTILT_ENV} (README.md has "
                         f"the commands). Looked for: {params_file}")
    spec = importlib.util.spec_from_file_location("_priortilt_params", params_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # its dataclasses look the module up while it loads
    spec.loader.exec_module(module)
    config = module.tuned_config(PRIORTILT_CODE, p, max_rounds=max_rounds or module.DEFAULT_MAX_ROUNDS)
    ends, prob_positive, pos_coef, neg_coef, threshold_factor = config.schedule_arrays()
    commit = _git_commit(repo)
    if commit != PRIORTILT_COMMIT:
        warnings.warn(f"the PriorTilt clone is at commit {commit}; the numbers in README.md were made with {PRIORTILT_COMMIT}",
                      stacklevel=2)
    return {"decoder": "PriorTilt, three-phase mode, f64 messages (priortilt_rs.PriorTiltDecoder on the kit's H and priors)",
            "code": PRIORTILT_CODE, "p_tuned": float(config.p_used),
            "initial_iters": int(config.initial_iters), "round_len": int(config.round_len), "max_rounds": int(config.max_rounds),
            "ends": list(ends), "prob_positive": list(prob_positive), "pos_coef": list(pos_coef),
            "neg_coef": list(neg_coef), "threshold_factor": list(threshold_factor),
            "priortilt_commit": commit, "params_sha256": hashlib.sha256(params_file.read_bytes()).hexdigest(),
            "priortilt_rs": metadata.version("priortilt_rs")}


def _git_commit(folder):
    try:
        return subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def decoder_settings(p, seeds=DEFAULT_SEEDS, priortilt_repo=None, priortilt_max_rounds=None, relay_changes=None):
    """Everything that fixes the three decoders at noise strength p, as one dict that JSON can hold.

    seeds: PriorTilt's seed for P1 and for P2. relay_changes replaces single settings of the
    kit's relay-BP; like priortilt_max_rounds it exists for quick tests.
    """
    seeds = [int(seed) for seed in seeds]
    if len(seeds) != 2 or seeds[0] == seeds[1]:
        raise ValueError("PriorTilt runs under two different seeds")
    relay_changes = dict(relay_changes or {})
    tilt = priortilt_settings(p, priortilt_repo, priortilt_max_rounds)
    return {"R": {"decoder": "relay-BP (the kit's decoders.relay_decoder, called through decode_detailed)",
                  "settings": json.loads(json.dumps(RELAY_SETTINGS | relay_changes)), "changes_to_the_kit": relay_changes,
                  "relay_bp": metadata.version("relay-bp")},
            "P1": tilt | {"seed": seeds[0]},
            "P2": tilt | {"seed": seeds[1]}}


# ---------- one decoder, one answer ----------

class _Relay:
    def __init__(self, problem, settings):
        self.decoder = relay_decoder(problem.H, problem.priors, **settings["changes_to_the_kit"])

    def decode(self, syndrome):
        result = self.decoder.decode_detailed(syndrome)
        return result.decoding, bool(result.success), int(result.iterations)


class _PriorTilt:
    def __init__(self, problem, settings, seed):
        import priortilt_rs   # built from PriorTilt's repository; see README.md

        self.decoder = priortilt_rs.PriorTiltDecoder(
            problem.H, np.asarray(problem.priors, np.float64), initial_iters=settings["initial_iters"],
            round_len=settings["round_len"], max_rounds=settings["max_rounds"], seed=int(seed))
        self.decoder.set_round_schedule(
            np.asarray(settings["ends"], np.int64), np.asarray(settings["prob_positive"], np.float64),
            np.asarray(settings["pos_coef"], np.float64), np.asarray(settings["neg_coef"], np.float64),
            np.asarray(settings["threshold_factor"], np.float64))

    def decode(self, syndrome):
        errors = self.decoder.decode(syndrome)
        return errors, bool(self.decoder.converge), int(self.decoder.iterations)


@dataclass
class Candidate:
    """One decoder's answer to one shot."""
    name: str               # "R", "P1" or "P2"
    predicted: np.ndarray   # the 12 observable bits of its correction: L e mod 2
    converged: bool         # the correction reproduces the syndrome
    iterations: int
    n_faults: int           # faults in the correction
    weight: float           # the sum of log((1 - prior) / prior) over those faults
    seconds: float          # the decode call alone


class Stage:
    """One decoder, built once. It keeps its state from shot to shot."""

    def __init__(self, name, problem, settings, seed_offset=0):
        self.name, self.L = name, problem.L
        self.llr = np.log((1 - problem.priors) / problem.priors)
        if name == "R":
            self.decoder = _Relay(problem, settings[name])
        else:
            self.decoder = _PriorTilt(problem, settings[name], settings[name]["seed"] + seed_offset)

    def run(self, syndrome):
        """Decode one shot. The timer wraps the decode call alone, as the kit's decoders.score does."""
        start = time.perf_counter()
        errors, converged, iterations = self.decoder.decode(syndrome)
        seconds = time.perf_counter() - start
        errors = np.asarray(errors, np.uint8)
        on = errors.astype(bool)
        predicted = np.asarray((self.L @ errors) % 2, np.uint8)   # uint8 sums wrap at 256, which keeps the parity
        return Candidate(self.name, predicted, converged, iterations, int(on.sum()), float(self.llr[on].sum()), seconds)


class Hierarchy:
    """The stages of one memory basis, in the order they run. Built once per session.

    first_two runs on every shot. third runs only where the first two predictions differ, so
    relay-BP sees those shots alone, in their order.
    """

    def __init__(self, problem, settings):
        self.p1, self.p2, self.relay = (Stage(name, problem, settings) for name in ("P1", "P2", "R"))

    def first_two(self, syndrome):
        return self.p1.run(syndrome), self.p2.run(syndrome)

    def third(self, syndrome):
        return self.relay.run(syndrome)


def _second_seed_worker(connection, basis, H, L, priors, p, settings):
    """The helper process of ParallelHierarchy: PriorTilt under the second seed, one answer per syndrome received."""
    stage = Stage("P2", Problem(basis, H, L, priors, p, None), settings)
    connection.send("ready")
    while True:
        try:
            syndrome = connection.recv()
        except EOFError:
            return
        if syndrome is None:
            return
        connection.send(stage.run(syndrome))


class ParallelHierarchy(Hierarchy):
    """The same stages, with the two PriorTilt decodes of a shot side by side on two cores.

    PriorTilt's Python module keeps the interpreter lock while it decodes, so threads would
    not help. P2 therefore lives in a helper process that is started once, with the session,
    and stays alive: it holds the second-seed decoder and sees every shot in order, so its
    answers are those of the sequential Hierarchy, bit for bit. A shot then takes the longer
    of the two decodes instead of their sum, in wall time; the CPU time is unchanged.
    """

    def __init__(self, problem, settings):
        self.p1, self.relay = Stage("P1", problem, settings), Stage("R", problem, settings)
        context = multiprocessing.get_context("spawn")
        self.connection, other_end = context.Pipe()
        self.process = context.Process(target=_second_seed_worker, daemon=True,
                                       args=(other_end, problem.basis, problem.H, problem.L, problem.priors, problem.p, settings))
        self.process.start()
        other_end.close()
        if self.connection.recv() != "ready":
            raise RuntimeError("the helper process for the second PriorTilt decode did not start")

    def first_two(self, syndrome):
        self.connection.send(syndrome)        # P2 starts in the helper ...
        first = self.p1.run(syndrome)         # ... while P1 runs here
        return first, self.connection.recv()  # an EOFError here means the helper died: the run stops

    def close(self):
        if self.process.is_alive():
            self.connection.send(None)
            self.process.join(timeout=10)

    def __del__(self):
        try:
            self.close()
        except (OSError, ValueError, AttributeError):
            pass


# ---------- the record of a run ----------

def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def processor():
    """The CPU's name, as far as the system tells it."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def machine():
    """Where a run happened, for the times it reports."""
    return {"platform": platform.platform(), "processor": processor(),
            "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "numpy": metadata.version("numpy"), "scipy": metadata.version("scipy"), "stim": metadata.version("stim")}
