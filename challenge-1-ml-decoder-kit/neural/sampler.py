"""Draw training shots from the detector error model, on the training device.

A shot is drawn in three steps:
1. every error mechanism fires on its own, with the kit's prior at the chosen noise strength p;
2. the detectors are H times the fired mechanisms, mod 2;
3. the labels (the logical observables) are L times the fired mechanisms, mod 2.
H, L and the priors are the kit's circuit.dem_matrices of the kit's memory circuit. Nothing is
read from disk.

detectors="all" uses the full model: every detector, about 68,000 mechanisms for the Gross code.
detectors="basis" uses the model of the memory basis alone: 936 detectors, 8,784 mechanisms.

Check against Stim's own circuit sampler, and time both models:

  python -m neural.sampler --p 0.004 --shots 200000
"""
import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch

from circuit import dem_matrices, detector_ids, sample
from make_circuit import memory_experiment
from neural.graph import build_graph, check_parity_matrix, gf2_inverse


class DemSampler:
    """Samples (detectors, observables) at any of a fixed list of noise strengths."""

    def __init__(self, code="gross", basis="X", rounds=None, detectors="all", noise_levels=(0.004,), device="cpu",
                 logicals="kit"):
        self.device = torch.device(device)
        self.priors = {}
        for p in noise_levels:
            H, L, priors = dem_matrices(memory_experiment(code, basis, rounds, p), basis if detectors == "basis" else None)
            if not self.priors:
                self.H, self.L = H.tocsc(), L.tocsc()
            elif (H != self.H).nnz or (L != self.L).nnz:
                raise ValueError("the error model's mechanisms changed with the noise strength")
            self.priors[p] = torch.tensor(priors, dtype=torch.float32, device=self.device)
        if logicals != "kit":   # label the shots with the model's logical operators instead of the kit's
            self.L = model_label_matrix(build_graph(code, basis, rounds, logicals), self.H, self.L, detectors)
        self.num_detectors, self.num_mechanisms = self.H.shape
        self.num_observables = self.L.shape[0]
        # Row lists of every mechanism, padded with one extra "nowhere" row.
        self._detector_rows = self._padded_rows(self.H)
        self._observable_rows = self._padded_rows(self.L)

    def _padded_rows(self, matrix):
        """[mechanisms, widest column]: the rows each mechanism flips; unused places point at an extra row."""
        counts = np.diff(matrix.indptr)
        table = np.full((matrix.shape[1], max(int(counts.max()), 1)), matrix.shape[0], dtype=np.int64)
        table[np.repeat(np.arange(matrix.shape[1]), counts), _position_in_column(counts)] = matrix.indices
        return torch.from_numpy(table).to(self.device)

    def sample(self, shots, p, generator, return_faults=False):
        """uint8 tensors detectors [shots, detectors] and observables [shots, observables] at noise strength p.

        The generator must live on the sampler's device. With return_faults, the fired
        mechanisms come back too, as (shot index, mechanism index).
        """
        fired = torch.rand((shots, self.num_mechanisms), generator=generator, device=self.device) < self.priors[p]
        shot, mechanism = fired.nonzero(as_tuple=True)
        detectors = self._parity(shot, self._detector_rows[mechanism], shots, self.num_detectors)
        observables = self._parity(shot, self._observable_rows[mechanism], shots, self.num_observables)
        return (detectors, observables, (shot, mechanism)) if return_faults else (detectors, observables)

    @staticmethod
    def _parity(shot, rows, shots, num_rows):
        """Count how often each (shot, row) is flipped and keep the parity. Column num_rows collects the padding."""
        flat = (shot[:, None] * (num_rows + 1) + rows).reshape(-1)
        counts = torch.bincount(flat, minlength=shots * (num_rows + 1)).reshape(shots, num_rows + 1)
        return (counts[:, :num_rows] & 1).to(torch.uint8)


def model_label_matrix(graph, H, L, detectors):
    """L for the model's logical operators (see graph.py), from the kit's H and L.

    model bits = inverse(kit_from_model) x (kit bits + kit_from_checks x check parities), and
    both the kit bits and the check parities are linear in the fired mechanisms. So the change
    of operators is one new matrix L, computed once.
    """
    used = graph.basis_detectors if detectors == "basis" else np.arange(graph.num_detectors)
    as_int = lambda matrix: sp.csr_matrix(matrix).astype(np.int64)
    check_parities = as_int(check_parity_matrix(graph, used)) @ as_int(H)        # [n2, mechanisms]
    kit = as_int(L) + as_int(graph.kit_from_checks) @ check_parities
    kit.data %= 2
    model = as_int(gf2_inverse(graph.kit_from_model)) @ kit
    model.data %= 2
    model.eliminate_zeros()
    return model.astype(np.uint8).tocsc()


def _position_in_column(counts):
    """0, 1, 2, ... within every column of a sparse matrix, for columns of the given lengths."""
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    return np.arange(int(counts.sum())) - starts


# ---------- the check against Stim ----------

class Tally:
    """Running sums over shots: how often each bit fires, and the distribution of the syndrome weight."""

    def __init__(self):
        self.shots, self.fired, self.observed, self.any_observable = 0, 0, 0, 0
        self.weight_sum, self.weight_squares = 0.0, 0.0

    def add(self, detectors, observables):
        detectors, observables = np.asarray(detectors, dtype=np.int64), np.asarray(observables, dtype=np.int64)
        weight = detectors.sum(axis=1)
        self.shots += len(detectors)
        self.fired = self.fired + detectors.sum(axis=0)
        self.observed = self.observed + observables.sum(axis=0)
        self.any_observable += int(observables.any(axis=1).sum())
        self.weight_sum += float(weight.sum())
        self.weight_squares += float((weight ** 2).sum())

    @property
    def weight_mean(self):
        return self.weight_sum / self.shots

    @property
    def weight_variance(self):
        return self.weight_squares / self.shots - self.weight_mean ** 2


def rate_z_scores(count_a, count_b, shots):
    """Two-sample z score of every pair of counts, each out of `shots`. Bits that never fire score 0."""
    pooled = (count_a + count_b) / (2 * shots)
    sigma = np.sqrt(pooled * (1 - pooled) * 2 / shots)
    return np.divide(count_a / shots - count_b / shots, sigma, out=np.zeros(len(sigma)), where=sigma > 0)


def compare_with_stim(code, basis, rounds, detectors, p, shots, seed, chunk=20000):
    """Sample `shots` shots from the error model and from Stim's circuit sampler and compare them.

    Returns a dict of numbers. "passed" is True when no detector or observable rate differs by
    more than 4.5 sigma (a false alarm about once in 100 runs of 1,740 comparisons) and the
    syndrome weight's mean and variance agree within 4.5 sigma.
    """
    circuit = memory_experiment(code, basis, rounds, p)
    sampler = DemSampler(code, basis, rounds, detectors, [p])
    columns = detector_ids(circuit, basis if detectors == "basis" else None)
    generator = torch.Generator().manual_seed(seed)
    ours, stims = Tally(), Tally()
    for start in range(0, shots, chunk):
        size = min(chunk, shots - start)
        ours.add(*(t.numpy() for t in sampler.sample(size, p, generator)))
        det, obs = sample(circuit, size, seed + 1 + start // chunk)
        stims.add(det[:, columns], obs)
    z_detectors = rate_z_scores(ours.fired, stims.fired, shots)
    z_observables = rate_z_scores(ours.observed, stims.observed, shots)
    z_any = rate_z_scores(np.array([ours.any_observable]), np.array([stims.any_observable]), shots)[0]
    variance = (ours.weight_variance + stims.weight_variance) / 2
    z_mean = (ours.weight_mean - stims.weight_mean) / np.sqrt(2 * variance / shots)
    z_variance = (ours.weight_variance - stims.weight_variance) / (variance * np.sqrt(4 / shots))   # normal approximation
    worst = max(np.abs(z_detectors).max(), np.abs(z_observables).max(), abs(z_any), abs(z_mean), abs(z_variance))
    return {
        "detectors": sampler.num_detectors, "mechanisms": sampler.num_mechanisms, "shots": shots,
        "max_z_detector": float(np.abs(z_detectors).max()), "mean_z2_detector": float((z_detectors ** 2).mean()),
        "max_z_observable": float(np.abs(z_observables).max()), "z_any_observable": float(z_any),
        "detector_rate": (float(ours.fired.sum() / shots / sampler.num_detectors),
                          float(stims.fired.sum() / shots / sampler.num_detectors)),
        "observable_rate": (float(ours.observed.mean() / shots), float(stims.observed.mean() / shots)),
        "any_observable_rate": (ours.any_observable / shots, stims.any_observable / shots),
        "weight_mean": (ours.weight_mean, stims.weight_mean), "z_weight_mean": float(z_mean),
        "weight_variance": (ours.weight_variance, stims.weight_variance), "z_weight_variance": float(z_variance),
        "passed": bool(worst < 4.5),
    }


def time_sampler(code, basis, rounds, detectors, p, batch, repeats=20):
    """Seconds per batch on this device, after one warm-up call."""
    sampler = DemSampler(code, basis, rounds, detectors, [p])
    generator = torch.Generator().manual_seed(0)
    sampler.sample(batch, p, generator)
    start = time.perf_counter()
    for _ in range(repeats):
        sampler.sample(batch, p, generator)
    return (time.perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code", default="gross")
    parser.add_argument("--basis", choices=["X", "Z"], default="X")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch", type=int, default=128, help="batch size of the timing")
    parser.add_argument("--detectors", nargs="+", choices=["all", "basis"], default=["all", "basis"])
    parser.add_argument("--threads", type=int, help="CPU threads for torch")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"error-model sampler against Stim's circuit sampler: {args.code}, {args.basis} memory, p={args.p:g}, "
          f"{args.shots} shots each, seed {args.seed}, {torch.get_num_threads()} CPU threads")
    for detectors in args.detectors:
        r = compare_with_stim(args.code, args.basis, args.rounds, detectors, args.p, args.shots, args.seed)
        seconds = time_sampler(args.code, args.basis, args.rounds, detectors, args.p, args.batch)
        print(f"detectors={detectors}: {r['detectors']} detectors, {r['mechanisms']} mechanisms")
        print(f"  detector rates   largest |z| {r['max_z_detector']:.2f}, mean z^2 {r['mean_z2_detector']:.3f} (1 expected), "
              f"mean rate {r['detector_rate'][0]:.5f} against Stim {r['detector_rate'][1]:.5f}")
        print(f"  observable rates largest |z| {r['max_z_observable']:.2f}, "
              f"mean rate {r['observable_rate'][0]:.5f} against Stim {r['observable_rate'][1]:.5f}")
        print(f"  any observable   {r['any_observable_rate'][0]:.5f} against Stim {r['any_observable_rate'][1]:.5f}, "
              f"z {r['z_any_observable']:.2f}")
        print(f"  syndrome weight  mean {r['weight_mean'][0]:.3f} against {r['weight_mean'][1]:.3f} (z {r['z_weight_mean']:.2f}), "
              f"variance {r['weight_variance'][0]:.2f} against {r['weight_variance'][1]:.2f} (z {r['z_weight_variance']:.2f})")
        print("  PASS: every |z| is below 4.5" if r["passed"] else "  FAIL: a |z| of 4.5 or more")
        print(f"  speed on the CPU: {seconds * 1e3:.1f} ms per batch of {args.batch}, {args.batch / seconds:.0f} shots/s")


if __name__ == "__main__":
    main()
