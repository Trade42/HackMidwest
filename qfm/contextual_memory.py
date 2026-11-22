"""A small, simulated preparation-contextuality experiment with sentence cues.

This is a delayed-query memory task, not the unrestricted learning benchmark.
The encoder receives two independent facts before the decoder receives its
question. One qubit is sent once and measured once. No expectation vector,
statevector, original text, or second copy is accessible to the decoder.

Reference: Spekkens et al., https://arxiv.org/abs/0805.1463 .
"""

import argparse
import csv
from itertools import product
import json
from pathlib import Path
import re
import time

import numpy as np
import pennylane as qml
from scipy.optimize import linprog
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

BITS = np.array(list(product((0, 1), repeat=2)), dtype=int)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1., -1.]).astype(complex)
I = np.eye(2, dtype=complex)
PATTERN = re.compile(r"Ava is (taller|shorter) than Mia but (older|younger) than Mia\.")


def parse_sentence(sentence):
    """Candidate order is fixed: Ava=0, Mia=1. No question is supplied here."""
    match = PATTERN.fullmatch(sentence)
    if match is None:
        raise ValueError('Expected the controlled taller/older sentence grammar')
    return np.array([match[1] == 'shorter', match[2] == 'younger'], dtype=int)


def sentence_for(bits):
    return (f"Ava is {'shorter' if bits[0] else 'taller'} than Mia but "
            f"{'younger' if bits[1] else 'older'} than Mia.")


def prepare(bits, visibility=1., *, dephase=False):
    """One-qubit density matrix; visibility v means rho -> v*rho+(1-v)*I/2.

    The query is deliberately absent. Dephasing removes X coherence while
    retaining the Z component, producing a commuting-state ablation.
    """
    bits = np.asarray(bits)
    if bits.shape != (2,) or not np.isin(bits, [0, 1]).all():
        raise ValueError('Supply two binary meaning cues')
    if not 0 <= visibility <= 1:
        raise ValueError('Visibility must be in [0, 1]')
    signs = 1 - 2 * bits
    return (I + visibility / np.sqrt(2) *
            ((0 if dephase else signs[0]) * X + signs[1] * Z)) / 2


def response_probabilities(state, query):
    """Born probabilities used by the simulator, not features sent to a model."""
    if query not in (0, 1):
        raise ValueError('Query 0 asks taller; query 1 asks older')
    observable = X if query == 0 else Z
    expectation = np.trace(state @ observable).real
    return np.clip(np.array([1 + expectation, 1 - expectation]) / 2, 0, 1)


def measure_once(state, query, rng):
    """Consume one prepared system and return one candidate, never an argmax."""
    return int(rng.choice(2, p=response_probabilities(state, query)))


def parity_guess_probability(states):
    """Helstrom optimum over ALL measurements, not merely the two task queries."""
    even = states[BITS.sum(axis=1) % 2 == 0].mean(axis=0)
    odd = states[BITS.sum(axis=1) % 2 == 1].mean(axis=0)
    return float(.5 + np.abs(np.linalg.eigvalsh(even - odd)).sum() / 4)


def classical_noncontextual_optimum():
    """Optimize every parity-oblivious classical message using a tiny LP.

    A message can be reduced to the deterministic responses it would yield
    for the two questions: lambda=(answer0, answer1), four possibilities.
    Stochastic decoders are mixtures of these response assignments. Optimize
    p(lambda|x), with normalization and equal even/odd-parity mixtures for
    every lambda. This is stronger than comparison with one trained network.
    """
    reward = (BITS[:, None, :] == BITS[None, :, :]).mean(axis=2) / len(BITS)
    normalization = np.kron(np.eye(4), np.ones((1, 4)))
    parity_sign = (-1.) ** BITS.sum(axis=1)
    equivalence = np.concatenate([sign * np.eye(4) for sign in parity_sign], axis=1)
    with threadpool_limits(limits=1):
        result = linprog(-reward.ravel(),
                         A_eq=np.vstack([normalization, equivalence]),
                         b_eq=np.r_[np.ones(4), np.zeros(4)], bounds=(0, 1), method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    return float(-result.fun), result.x.reshape(4, 4)


def probability_table(states):
    return np.array([[response_probabilities(state, query) for query in range(2)]
                     for state in states])


def expected_success(table):
    return float(np.mean([table[i, q, BITS[i, q]] for i in range(4) for q in range(2)]))


def circuit_crosscheck(visibility):
    """Verify analytical probabilities with eight actual PennyLane circuits."""
    device = qml.device('default.mixed', wires=1)

    @qml.qnode(device)
    def circuit(angle, query):
        qml.RY(angle, wires=0)
        # PennyLane's Pauli-error parameter p contracts the Bloch vector by
        # 1-4p/3, so p=3(1-v)/4 implements precisely the documented channel.
        qml.DepolarizingChannel(3 * (1 - visibility) / 4, wires=0)
        if query == 0:
            qml.Hadamard(wires=0)
        return qml.probs(wires=0)

    discrepancies = []
    for bits in BITS:
        signs = 1 - 2 * bits
        angle = np.arctan2(signs[0], signs[1])
        for query in range(2):
            discrepancies.append(np.max(np.abs(
                circuit(angle, query) - response_probabilities(prepare(bits, visibility), query))))
    return float(max(discrepancies))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=241123)
    parser.add_argument('--visibility', type=float, default=1.)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'results/contextual_memory')
    args = parser.parse_args()
    if args.trials < 1 or not 0 <= args.visibility <= 1:
        parser.error('Use positive trials and visibility in [0, 1]')
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        'task': 'Two independent sentence facts, uniformly random delayed query',
        'preparations': 4, 'queries': ['Choose her: taller.', 'Choose her: older.'],
        'candidate_order': ['Ava', 'Mia'], 'word_count': 9,
        'semantic_bits': 2, 'quantum_message_qubits': 1, 'copies_per_trial': 1,
        'measurements_per_trial': 1, 'query_available_to_encoder': False,
        'decoder_input': 'one physical qubit and query; no text, amplitudes or expectation vector',
        'classical_constraint': 'Every message must hide the XOR of the two independent facts',
        'unrestricted_classical_control': 'Both facts retained; 100% correct, reveals parity',
        'training': 'None; analytic protocol, fixed before sampling',
        'scope': 'Classical simulation of a contextual memory protocol with an explicit sentence parser',
        'does_not_establish': ['NLP generalization', 'learning sample efficiency', 'computational speedup',
                               'an unrestricted classical accuracy advantage', 'a hardware violation'],
        'primary_statistic': 'One-sided binomial test against the exact classical bound',
        'uncertainty': '95% exact binomial interval; simulated measurement noise only',
        'visibility': args.visibility, 'trials': args.trials, 'seed': args.seed,
        'reference': 'https://arxiv.org/abs/0805.1463',
    }
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    states = np.array([prepare(parse_sentence(sentence_for(bits)), args.visibility) for bits in BITS])
    table = probability_table(states)
    bound, classical_channel = classical_noncontextual_optimum()
    parity = parity_guess_probability(states)
    discrepancy = circuit_crosscheck(args.visibility)
    if parity > .5 + 1e-10 or discrepancy > 1e-10:
        raise RuntimeError('Parity equivalence or independent circuit verification failed')
    # Every run is a fresh message, with a fresh independent input and question.
    # Sampling the Born table is equivalent to a single circuit shot per run.
    # The table is simulator instrumentation; it is not decoder-accessible data.
    rng = np.random.default_rng(args.seed)
    inputs = rng.integers(4, size=args.trials)
    queries = rng.integers(2, size=args.trials)
    outputs = (rng.random(args.trials) >= table[inputs, queries, 0]).astype(int)
    correct = outputs == BITS[inputs, queries]
    observed = float(correct.mean())
    test = binomtest(int(correct.sum()), args.trials, p=bound, alternative='greater')
    interval = binomtest(int(correct.sum()), args.trials).proportion_ci(confidence_level=.95)
    dephased = probability_table(np.array([prepare(b, args.visibility, dephase=True) for b in BITS]))
    expected = expected_success(table)
    report = {
        'quantum_expected_accuracy': expected, 'quantum_sampled_accuracy': observed,
        'quantum_accuracy_95pct_interval': [float(interval.low), float(interval.high)],
        'classical_parity_oblivious_optimum': bound,
        'classical_unrestricted_accuracy': 1.,
        'quantum_expected_gain_percentage_points': 100 * (expected - bound),
        'quantum_sampled_p_value_vs_bound': float(test.pvalue),
        'parity_guess_probability_best_possible_measurement': parity,
        'parity_information_trace_distance': 2 * (parity - .5),
        'commuting_dephased_ablation_accuracy': expected_success(dephased),
        'commuting_optimal_encoding_accuracy': .75,
        'pennylane_max_probability_error': discrepancy,
        'visibility_threshold_for_advantage': float(1 / np.sqrt(2)),
        'mixture_with_maximally_mixed_threshold': float(1 - 1 / np.sqrt(2)),
        'clear_simulated_violation': bool(interval.low > bound and parity <= .5 + 1e-10),
        'elapsed_seconds_before_plot': time.perf_counter() - started,
    }
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.output / 'simulation.npz', states=states, probabilities=table,
                        classical_channel=classical_channel, inputs=inputs, queries=queries,
                        outputs=outputs, correct=correct)
    with (args.output / 'sentences.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sentence', 'delayed_query', 'candidate_0',
                                                   'candidate_1', 'answer', 'quantum_success_probability'])
        writer.writeheader()
        for i, bits in enumerate(BITS):
            for q in range(2):
                writer.writerow(dict(sentence=sentence_for(bits), delayed_query=protocol['queries'][q],
                                     candidate_0='Ava', candidate_1='Mia', answer=int(bits[q]),
                                     quantum_success_probability=table[i, q, bits[q]]))
    plot(report, args.output, args.trials)
    print(json.dumps(report, indent=2))
    print(f'Saved contextual memory demo to {args.output}')


def plot(report, output, trials):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, (ax, noise) = plt.subplots(1, 2, figsize=(12, 4.8))
    values = [report['classical_parity_oblivious_optimum'], report['quantum_expected_accuracy'], 1.]
    bars = ax.bar(['Classical\nparity hidden', 'One qubit\nparity hidden', 'Classical\nfull information'],
                  values, color=['#747c8c', '#187f9c', '#c0c4cc'])
    ax.bar_label(bars, labels=[f'{100*v:.2f}%' for v in values], padding=5)
    lo, hi = report['quantum_accuracy_95pct_interval']
    observed = report['quantum_sampled_accuracy']
    ax.errorbar(1, observed, yerr=[[observed-lo], [hi-observed]], fmt='o', color='#111111', capsize=4,
                label=f'{trials:,} simulated single-shot trials, 95% CI')
    ax.set(ylim=(0, 1.12), ylabel='Probability of answering the delayed question',
           title='Contextual memory: explicit parity constraint')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='lower center', fontsize=8)
    noise_fraction = np.linspace(0, 1, 201)
    noise.plot(noise_fraction, .5 + (1-noise_fraction)/(2*np.sqrt(2)),
               color='#187f9c', label='One-qubit protocol (exact)')
    noise.axhline(.75, color='#747c8c', ls='--', label='Optimal parity-oblivious classical')
    noise.axvline(1-1/np.sqrt(2), color='#999999', ls=':', label='Advantage disappears at 29.29%')
    noise.set(xlabel='Mixture weight of the maximally mixed state', ylabel='Expected accuracy',
              title='Sensitivity to a specified noise model', ylim=(.45, .95), xlim=(0, 1))
    noise.xaxis.set_major_formatter(PercentFormatter(1))
    noise.yaxis.set_major_formatter(PercentFormatter(1))
    noise.legend(fontsize=8)
    fig.suptitle('Simulated resource-constrained protocol; no training or hardware speedup', fontsize=11)
    fig.tight_layout()
    fig.savefig(output / 'contextual_memory.png', dpi=180)
    fig.savefig(output / 'contextual_memory.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
