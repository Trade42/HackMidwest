"""Separate one-qubit parity-oblivious memory demonstration."""

import argparse
from itertools import product
from pathlib import Path
import re

import numpy as np
import pennylane as qml
from scipy.optimize import linprog
from scipy.stats import binomtest


BITS = np.array(list(product((0, 1), repeat=2)))
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
I = np.eye(2, dtype=complex)
SENTENCE = re.compile(r"Ava is (taller|shorter) than Mia but (older|younger) than Mia\.")
PLOT = Path(__file__).resolve().parent / "output/contextuality.png"


def encode(sentence: str) -> np.ndarray:
    match = SENTENCE.fullmatch(sentence)
    if match is None:
        raise ValueError("Unsupported contextuality sentence")
    return np.array([match[1] == "shorter", match[2] == "younger"], dtype=int)


def sentence_for(bits) -> str:
    return (f"Ava is {'shorter' if bits[0] else 'taller'} than Mia but "
            f"{'younger' if bits[1] else 'older'} than Mia.")


def prepare(bits, visibility: float = 1.0, dephase: bool = False):
    signs = 1 - 2 * np.asarray(bits)
    return (I + visibility * ((0 if dephase else signs[0]) * X + signs[1] * Z) / np.sqrt(2)) / 2


def probabilities(state, query: int):
    expectation = np.trace(state @ (X if query == 0 else Z)).real
    return np.array([(1 + expectation) / 2, (1 - expectation) / 2])


def classical_bound() -> float:
    """LP over all parity-oblivious four-message classical encodings."""
    reward = (BITS[:, None, :] == BITS[None, :, :]).mean(axis=2) / 4
    normalization = np.kron(np.eye(4), np.ones((1, 4)))
    parity_sign = (-1.0) ** BITS.sum(axis=1)
    parity_constraints = np.concatenate([sign * np.eye(4) for sign in parity_sign], axis=1)
    result = linprog(-reward.ravel(), A_eq=np.vstack((normalization, parity_constraints)),
                     b_eq=np.r_[np.ones(4), np.zeros(4)], bounds=(0, 1), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return float(-result.fun)


def circuit_error(states, visibility: float) -> float:
    device = qml.device("default.mixed", wires=1)

    @qml.qnode(device)
    def circuit(angle, query):
        qml.RY(angle, wires=0)
        qml.DepolarizingChannel(3 * (1 - visibility) / 4, wires=0)
        if query == 0:
            qml.Hadamard(wires=0)
        return qml.probs(wires=0)

    errors = []
    for bits, state in zip(BITS, states):
        signs = 1 - 2 * bits
        angle = np.arctan2(signs[0], signs[1])
        for query in (0, 1):
            errors.append(np.max(np.abs(circuit(angle, query) - probabilities(state, query))))
    return float(max(errors))


def plot_results(bound, expected, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([0, 1], [bound, expected], width=0.55,
                  color=["#9aa7b4", "#287e96"])
    ax.bar_label(bars, labels=[f"{bound:.1%}", f"{expected:.1%}"], padding=4, fontsize=11)
    ax.set(xticks=[0, 1],
           xticklabels=["Classical", "Qubit"],
           ylim=(0, 1.02), ylabel="Accuracy", title="Classical vs quantum memory")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=241123)
    parser.add_argument("--visibility", type=float, default=1.0)
    parser.add_argument("--plot", type=Path, default=PLOT)
    args = parser.parse_args()
    if args.trials < 1 or not 0 <= args.visibility <= 1:
        parser.error("Use positive trials and visibility in [0, 1]")
    states = np.array([prepare(encode(sentence_for(bits)), args.visibility) for bits in BITS])
    table = np.array([[probabilities(state, query) for query in (0, 1)] for state in states])
    bound = classical_bound()
    parity_states = [states[BITS.sum(axis=1) % 2 == parity].mean(axis=0) for parity in (0, 1)]
    parity_guess = 0.5 + np.abs(np.linalg.eigvalsh(parity_states[0] - parity_states[1])).sum() / 4
    error = circuit_error(states, args.visibility)
    if parity_guess > 0.5 + 1e-10 or error > 1e-10:
        raise RuntimeError("Parity obliviousness or PennyLane circuit check failed")
    expected = np.mean([table[i, q, BITS[i, q]] for i in range(4) for q in range(2)])
    dephased = np.array([prepare(bits, args.visibility, dephase=True) for bits in BITS])
    commuting = np.mean([probabilities(dephased[i], q)[BITS[i, q]]
                         for i in range(4) for q in range(2)])
    rng = np.random.default_rng(args.seed)
    inputs = rng.integers(4, size=args.trials)
    queries = rng.integers(2, size=args.trials)
    answers = (rng.random(args.trials) >= table[inputs, queries, 0]).astype(int)
    successes = int(np.sum(answers == BITS[inputs, queries]))
    interval = binomtest(successes, args.trials).proportion_ci(confidence_level=0.95)
    print(f"Classical limit with parity hidden: {bound:.2%}; "
          f"one-qubit prediction: {expected:.2%}")
    print(f"Simulation check: {successes / args.trials:.2%} correct in {args.trials:,} trials "
          f"(95% interval {interval.low:.2%}–{interval.high:.2%})")
    print(f"Parity remains hidden (best guess {parity_guess:.2%}); "
          f"dephased control {commuting:.2%}; circuit error {error:.1e}")
    print("This is a simulated, constrained memory protocol; it does not test the NLP classifier.")
    plot_results(bound, expected, args.plot)
    print(f"Plot: {args.plot}")


if __name__ == "__main__":
    main()
