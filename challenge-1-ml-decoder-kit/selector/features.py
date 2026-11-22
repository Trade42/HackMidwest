"""What the selector reads about a shot: the members' answers and how each member got there. Never the truth.

A member is one decoder's answer to a saved shot (12 predicted logical bits), read from the
per-shot files of hierarchy/ with hierarchy.ensemble.load_members, which also checks that every
file was made from the same data file (sha256) and lines up shot for shot:

  P1, P2   PriorTilt under two seeds   every shot      hierarchy.candidates (or the laptop's selector files)
  T        TCBP-JOBS                   every shot      hierarchy.tcbp_member
  R        relay-BP, the kit's settings                hierarchy.candidates
  J        joint relay-BP              listed shots    hierarchy.joint_member (p = 0.003 and 0.004 only)

P1, P2 and T are "always on". Where all three give the same 12 bits, that answer stands, as in
the ensemble's rule f. Elsewhere (a "choice shot") the selector looks at the distinct answers
the members gave, and scores each one. An answer is one 12-bit string; several members may give
it. At most one answer of a shot is right, because they are all different.

The features of one answer (one row of the choice table), all from the members' own outputs:

  has_T               TCBP-JOBS gives this answer
  n_priortilt         how many of the two PriorTilt seeds give it (0, 1, 2)
  has_R               relay-BP gives it
  J_converged         joint relay-BP gives it and its correction reproduces the syndrome
  J_unconverged       joint relay-BP gives it, but its correction does not reproduce the syndrome
  pt_converged        a PriorTilt seed that gives it converged
  R_converged         relay-BP gives it and converged
  pt_weight_minus_T   for an answer TCBP-JOBS does not give: the lightest converged PriorTilt correction
                      behind it, minus TCBP-JOBS's correction on the same shot, in units of 10, clipped
                      to [-5, 5] (0 for TCBP-JOBS's own answer and where no converged seed gives it)
  R_weight_minus_T    the same for relay-BP
  pt_log_iterations   log10 of PriorTilt's iterations, averaged over the seeds that give it (0 if none)
  R_log_iterations    log10 of relay-BP's iterations (0 if relay-BP does not give it)

The weight of a correction is the sum of log((1 - prior) / prior) over its faults: lower is
more probable. Weights from different decoders are not directly comparable (TCBP-JOBS finds
lighter corrections than PriorTilt or relay-BP for the same answer), so each source's weight is
taken relative to TCBP-JOBS's weight on that shot, with its own coefficient.

With the network's logits (optional, model-dependent):

  network_gap         the network's log-probability of this answer's 12 bits minus the best
                      answer's of the same shot (0 or below; 0 for every answer without logits)

The truth, the files' `failed_*` masks and the logits' `observables`, `failed` and
`confidence` are never read here. `labels` (below) is the only function that sees the truth,
and it is used only to fit and to score.
"""
from dataclasses import dataclass

import numpy as np

from hierarchy.ensemble import Member, load_members, load_network, unanimous
from hierarchy.rules import bits_for_network, network_score

POOL = ("T", "P1", "P2", "R", "J")      # the order of the answers in a shot; on a tie the earlier answer wins
ALWAYS_ON = ("P1", "P2", "T")
PRIORTILT = ("P1", "P2")
BASE_FEATURES = ("has_T", "n_priortilt", "has_R", "J_converged", "J_unconverged", "pt_converged", "R_converged",
                 "pt_weight_minus_T", "R_weight_minus_T", "pt_log_iterations", "R_log_iterations")
NETWORK_FEATURES = ("network_gap",)
WEIGHT_UNIT = 10.0        # weight gaps are in units of 10 (a factor e^10 in probability)
WEIGHT_CLIP = 5.0         # and clipped to [-5, 5], i.e. +-50
NETWORK_CLIP = -30.0      # the network's log-probability gap, clipped below


@dataclass
class View:
    """The members of one data file, without the truth: what a decoder may see."""
    members: dict             # name -> Member (hierarchy.ensemble), predictions only
    iterations: dict          # name -> [shots] float, NaN where the file holds none
    n_faults: dict            # name -> [shots] float, NaN where the file holds none
    logits: np.ndarray = None           # [shots, 12] the network's logits, or None
    network_answer: np.ndarray = None   # [shots, 12] the kit's bits of the network's own answer
    model_from_kit: np.ndarray = None   # see hierarchy.rules.bits_for_network

    @property
    def shots(self):
        return len(next(iter(self.members.values())).covers)


@dataclass
class Info:
    """What the files say about the shots, the truth included (for labels and scoring only)."""
    truth: np.ndarray
    data_sha256: str
    data_name: str
    basis: str
    p: float
    paths: list


def load(paths, logits_path=None):
    """(View, Info) of the member files of ONE data file, and optionally the network's logits of the same shots.

    hierarchy.ensemble.load_members checks the data sha256 and the alignment; this adds the
    iteration and fault counts, which its Member does not carry.
    """
    members, shots = load_members(paths)
    count = len(shots.truth)
    iterations, n_faults = {}, {}
    for path in paths:
        with np.load(path, allow_pickle=False) as file:
            rows = file["shot"].astype(int) if "shot" in file.files else np.arange(count)
            for name in [key[len("predicted_"):] for key in file.files if key.startswith("predicted_")]:
                for field, into in (("iterations", iterations), ("n_faults", n_faults)):
                    values = np.full(count, np.nan)
                    if f"{field}_{name}" in file.files:
                        values[rows] = file[f"{field}_{name}"]
                    into[name] = values
    for name in ALWAYS_ON:
        if name not in members or not members[name].covers.all():
            raise SystemExit(f"the selector needs {name} on every shot: give its file among the member files ({', '.join(map(str, paths))})")
    view = View(members, iterations, n_faults)
    if logits_path is not None:
        network = load_network(logits_path, shots)
        view.logits, view.network_answer, view.model_from_kit = network.logits, network.answer, network.model_from_kit
    info = Info(shots.truth, shots.data_sha256, shots.data_name or "", shots.basis, shots.p, [str(path) for path in paths])
    return view, info


def empty_member(name, count):
    """A member that decoded no shot (joint decoding not run for this file)."""
    return Member(name, np.zeros((count, 12), np.uint8), np.zeros(count, bool), np.full(count, np.nan), np.full(count, np.nan),
                  np.ones(count, bool), "no file", "none")


def choice_shots(view):
    """[shots] bool: the shots where P1, P2 and T are not unanimous."""
    return ~unanimous(view.members)


@dataclass
class Table:
    """One row per distinct answer of every choice shot."""
    shot: np.ndarray          # [rows] the shot's index in the data file
    bits: np.ndarray          # [rows, 12] the answer
    gives: np.ndarray         # [rows, len(POOL)] bool: which members give it
    features: np.ndarray      # [rows, len(names)]
    names: tuple
    starts: np.ndarray        # [choice shots + 1] the rows of choice shot i are starts[i]:starts[i + 1]
    shots: np.ndarray         # [choice shots] their indices in the data file


def _log_iterations(values):
    return float(np.log10(max(values, 1.0)))


def table(view, pool=POOL, network=False):
    """The choice table of a View: the distinct answers of every choice shot and their features.

    pool: the members that may be chosen (a model fitted without J uses a pool without J).
    network: add the network block (needs view.logits; without logits it is 0 on every row).
    """
    names = BASE_FEATURES + (NETWORK_FEATURES if network else ())
    present = [name for name in pool if name in view.members]
    members = view.members
    shots = np.flatnonzero(choice_shots(view))
    rows_shot, rows_bits, rows_gives, rows_features, starts = [], [], [], [], [0]
    for shot in shots:
        here = [name for name in present if members[name].covers[shot]]
        answers = []            # (bits, [member names]) in POOL order of their first member
        for name in here:
            bits = members[name].predicted[shot]
            for known in answers:
                if np.array_equal(known[0], bits):
                    known[1].append(name)
                    break
            else:
                answers.append((bits, [name]))
        weight_T = members["T"].weight[shot]
        for bits, givers in answers:
            row = dict.fromkeys(names, 0.0)
            row["has_T"] = float("T" in givers)
            pts = [name for name in givers if name in PRIORTILT]
            row["n_priortilt"] = float(len(pts))
            row["has_R"] = float("R" in givers)
            if "J" in givers:
                row["J_converged" if members["J"].converged[shot] else "J_unconverged"] = 1.0
            converged_pts = [name for name in pts if members[name].converged[shot]]
            row["pt_converged"] = float(bool(converged_pts))
            other_than_T = "T" not in givers      # the weight gaps compare another answer's correction with T's
            if converged_pts and other_than_T and np.isfinite(weight_T):
                lightest = min(members[name].weight[shot] for name in converged_pts)
                row["pt_weight_minus_T"] = float(np.clip((lightest - weight_T) / WEIGHT_UNIT, -WEIGHT_CLIP, WEIGHT_CLIP))
            if pts:
                row["pt_log_iterations"] = float(np.mean([_log_iterations(view.iterations[name][shot]) for name in pts]))
            if "R" in givers:
                row["R_converged"] = float(members["R"].converged[shot])
                if members["R"].converged[shot] and other_than_T and np.isfinite(weight_T):
                    row["R_weight_minus_T"] = float(np.clip((members["R"].weight[shot] - weight_T) / WEIGHT_UNIT, -WEIGHT_CLIP, WEIGHT_CLIP))
                row["R_log_iterations"] = _log_iterations(view.iterations["R"][shot])
            rows_shot.append(shot)
            rows_bits.append(bits)
            rows_gives.append([name in givers for name in POOL])
            rows_features.append([row[name] for name in names])
        starts.append(len(rows_shot))
    result = Table(np.array(rows_shot, int), np.array(rows_bits, np.uint8).reshape(-1, 12), np.array(rows_gives, bool).reshape(-1, len(POOL)),
                   np.array(rows_features, float).reshape(-1, len(names)), names, np.array(starts, int), shots)
    if network and view.logits is not None and len(result.shot):
        own = bits_for_network(result.bits[None], view.logits[result.shot] > 0, view.network_answer[result.shot], view.model_from_kit)
        score = network_score(own, view.logits[result.shot])[0]
        column = names.index("network_gap")
        for i in range(len(shots)):
            rows = slice(result.starts[i], result.starts[i + 1])
            result.features[rows, column] = np.maximum(score[rows] - score[rows].max(), NETWORK_CLIP)
    return result


def labels(choices, truth):
    """[rows] bool: this answer is the shot's truth (all 12 bits). Used to fit and to score, never as a feature."""
    return (choices.bits == np.asarray(truth)[choices.shot]).all(axis=1)


def answers(view, choices, pick):
    """[shots, 12]: P1's answer where P1, P2, T are unanimous; on choice shot i the answer at row pick[i] of the table."""
    result = view.members["P1"].predicted.copy()
    if len(choices.shots):
        result[choices.shots] = choices.bits[pick]
    return result
