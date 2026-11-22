"""Shared pieces of the gate measurements: datasets, the decoding problem, a chunked runner, statistics.

Run every script from challenge-1-ml-decoder-kit/ as a module, for example
  python -m analysis.g1_relay_failures run ...

How a run is made reproducible whatever the process count:
- a task is a list of shots of one dataset, decoded with one decoder spec;
- "chunk" mode: the task's shots are one block of CHUNK consecutive shots in file order, decoded
  in order on one fresh decoder (relay-BP carries its random state from one decode to the next);
- "fresh" mode: every shot gets its own fresh decoder, once per seed;
- every task's output is written to its own file in the cache folder, and the outputs are put
  back together in file order. A stopped run keeps its finished tasks; a rerun skips them.

Scoring is the kit's: decoders.score, a shot fails when any of the 12 predicted observables is
wrong, and an unconverged shot is scored on the correction the decoder returned.
"""
import hashlib
import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.stats import beta, binomtest

import decoders
from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit

CHUNK = 500
HELD_OUT_PREFIXES = ("test_", "go_")   # the team's held-out shots: never opened here
ANALYSIS = Path(__file__).resolve().parent


# ---------------------------------------------------------------- datasets and the problem

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Dataset(NamedTuple):
    path: str
    sha256: str
    basis: str
    p: float
    seed: int
    det: np.ndarray     # basis detectors only, one row per shot (what the baselines decode)
    obs: np.ndarray     # true observable flips: for scoring only, never passed to a decoder


def refuse_held_out(path):
    if Path(path).name.startswith(HELD_OUT_PREFIXES):
        raise SystemExit(f"refusing to open {path}: test and go files are the team's held-out shots")


def open_dataset(path, full=False):
    """Load a make_dataset.py file. Test and go files are refused.

    det holds the basis detectors (what the baselines decode), or all 1728 with full=True.
    """
    refuse_held_out(path)
    with np.load(path) as z:
        basis, p, seed = str(z["basis"]), float(z["p"]), int(z["seed"])
        columns = z["basis_detectors"]
        det, obs = (z["detectors"] if full else z["detectors"][:, columns]), z["observables"]
    circuit = load_circuit(GROSS_CIRCUITS[basis], p)
    if not np.array_equal(columns, detector_ids(circuit, basis)):
        raise ValueError(f"{path}: basis_detectors differ from the kit's detector_ids")
    return Dataset(str(path), sha256(path), basis, p, seed, np.ascontiguousarray(det), obs)


class Problem(NamedTuple):
    H: object
    L: object
    priors: np.ndarray
    llr: np.ndarray     # log((1 - prior) / prior), the weight relay-BP uses to rank solutions


def problem(basis, p, full=False):
    """The decoding problem of one memory experiment: basis detectors only, or with full=True all
    1728 detectors and every mechanism (dem_matrices(circuit) with no basis), which keeps the
    correlations between X and Z errors."""
    circuit = load_circuit(GROSS_CIRCUITS[basis], p)
    H, L, priors = dem_matrices(circuit, None if full else basis)
    return Problem(H, L, priors, np.log((1 - priors) / priors))


# ---------------------------------------------------------------- decoders

def make_decoder(kind, prob, kwargs, seed):
    """A fresh decoder. kind 'relay' is decoders.relay_decoder, 'bposd' is decoders.bposd_decoder,
    'tesseract' is the Tesseract search decoder on the same H, L and priors."""
    if kind == "relay":
        return decoders.relay_decoder(prob.H, prob.priors, seed=seed, **kwargs)
    if kind == "bposd":
        return decoders.bposd_decoder(prob.H, prob.priors, **kwargs)
    if kind == "tesseract":
        return TesseractOnMatrices(prob, **kwargs)
    raise ValueError(kind)


def dem_from_matrices(prob):
    """A stim DetectorErrorModel with one error per column of H: the same problem the kit's decoders get."""
    import stim
    H, L = prob.H.tocsc(), prob.L.tocsc()
    lines = []
    for k in range(H.shape[1]):
        dets = H.indices[H.indptr[k]:H.indptr[k + 1]]
        obs = L.indices[L.indptr[k]:L.indptr[k + 1]]
        lines.append(f"error({float(prob.priors[k]):.17g}) " + " ".join([f"D{d}" for d in dets] + [f"L{o}" for o in obs]))
    return stim.DetectorErrorModel("\n".join(lines))


class TesseractOnMatrices:
    """Tesseract (pip install tesseract-decoder) on the kit's matrices. decode returns an error vector
    over the columns of H, so decoders.score projects it with L as for the other decoders.
    It searches deterministically and keeps no state between shots."""

    def __init__(self, prob, **config):
        from tesseract_decoder import tesseract
        self.n = prob.H.shape[1]
        self.decoder = tesseract.TesseractDecoder(tesseract.TesseractConfig(dem_from_matrices(prob), **config))
        self.converge, self.iter = False, 0

    def decode(self, syndrome):
        idx = self.decoder.decode_to_errors(np.asarray(syndrome, bool))
        e = np.zeros(self.n, np.uint8)
        np.add.at(e, np.asarray(idx, np.int64), 1)
        e %= 2
        self.converge = not bool(self.decoder.low_confidence_flag)
        return e


class Recorder:
    """Wraps one decoder so that decoders.score can time it while the details are kept aside.

    The wrapper only stores the decoder's outputs; everything derived from them is computed
    after scoring, outside the timer.
    """

    def __init__(self, kind):
        self.kind, self.decoder, self.rows = kind, None, []

    def decode(self, syndrome):
        if self.kind == "relay":
            r = self.decoder.decode_detailed(syndrome)
            error = r.decoding
            self.rows.append((bool(r.success), int(r.iterations), error))
        else:
            error = self.decoder.decode(syndrome)
            self.rows.append((bool(self.decoder.converge), int(self.decoder.iter), np.array(error)))
        return error


def pack_bits(bits):
    """12 observable bits as one uint16 per shot (bit j = observable j)."""
    bits = np.asarray(bits, np.uint16)
    return (bits << np.arange(bits.shape[-1], dtype=np.uint16)).sum(axis=-1).astype(np.uint16)


def unpack_bits(packed, n=12):
    return ((np.asarray(packed, np.uint16)[..., None] >> np.arange(n, dtype=np.uint16)) & 1).astype(np.uint8)


def error_hash(error):
    return np.frombuffer(hashlib.blake2b(np.packbits(error).tobytes(), digest_size=8).digest(), np.uint64)[0]


def details(prob, det, obs, failed, seconds, rows):
    """Per-shot arrays from one scored pass."""
    conv = np.array([r[0] for r in rows], bool)
    its = np.array([r[1] for r in rows], np.int32)
    errors = np.array([np.asarray(r[2], np.uint8) for r in rows], np.uint8)
    pred = ((prob.L @ errors.T) % 2).T.astype(np.uint8)   # the kit's projection, shot by shot
    if not np.array_equal(failed, (pred != obs).any(axis=1)):
        raise AssertionError("failure flags differ from the kit's scoring")
    valid = (((prob.H @ errors.T) % 2).T == det).all(axis=1)
    return dict(failed=failed, seconds=seconds.astype(np.float32), converged=conv, iterations=its,
                pred=pack_bits(pred), wcount=errors.sum(axis=1).astype(np.int32),
                wllr=(errors @ prob.llr).astype(np.float64), valid=valid,
                ehash=np.array([error_hash(e) for e in errors], np.uint64))


def decode_shots(spec, prob, det, obs):
    """Decode shots with one spec. Returns {field: array}; with several seeds each field has a seed axis.

    spec: {"kind": "relay" | "bposd", "kwargs": {...}, "mode": "chunk" | "fresh", "seeds": [...],
           optional "problem": "full" for the joint problem on all 1728 detectors}
    """
    kind, kwargs, seeds = spec["kind"], spec.get("kwargs", {}), spec.get("seeds", [0])
    per_seed = []
    for seed in seeds:
        rec = Recorder(kind)
        if spec["mode"] == "chunk":
            rec.decoder = make_decoder(kind, prob, kwargs, seed)
            before = None
        else:
            def before(k, rec=rec, seed=seed):
                rec.decoder = make_decoder(kind, prob, kwargs, seed)   # not timed: before_shot
        failed, seconds = decoders.score(rec.decode, prob.L, det, obs, before_shot=before)
        per_seed.append(details(prob, det, obs, failed, seconds, rec.rows))
    if len(seeds) == 1 and not spec.get("seed_axis"):
        return per_seed[0]
    return {key: np.stack([d[key] for d in per_seed]) for key in per_seed[0]}


# ---------------------------------------------------------------- the runner

class Task(NamedTuple):
    run: str            # run name, the folder in the cache
    spec: dict
    path: str           # dataset file
    sha256: str
    indices: np.ndarray  # shot indices in file order


def fingerprint(task):
    blob = json.dumps({"spec": task.spec, "sha256": task.sha256,
                       "indices": hashlib.sha256(np.asarray(task.indices, np.int64).tobytes()).hexdigest()},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def task_file(cache, task):
    return Path(cache) / task.run / f"{task.sha256[:12]}_{int(task.indices[0]):06d}_{len(task.indices)}_{fingerprint(task)}.npz"


_DATASETS, _PROBLEMS = {}, {}


def _work(job):
    task, cache = job
    full = task.spec.get("problem") == "full"
    if (task.path, full) not in _DATASETS:
        _DATASETS.clear()   # one dataset at a time per worker keeps memory flat
        _DATASETS[(task.path, full)] = open_dataset(task.path, full)
    ds = _DATASETS[(task.path, full)]
    if ds.sha256 != task.sha256:
        raise ValueError(f"{task.path} changed: sha256 {ds.sha256} is not {task.sha256}")
    key = (ds.basis, ds.p, full)
    if key not in _PROBLEMS:
        if len(_PROBLEMS) > 2:
            _PROBLEMS.clear()
        _PROBLEMS[key] = problem(*key)
    idx = np.asarray(task.indices)
    t0 = time.perf_counter()
    out = decode_shots(task.spec, _PROBLEMS[key], ds.det[idx], ds.obs[idx])
    wall = time.perf_counter() - t0
    path = task_file(cache, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, indices=idx, wall=wall, pid=os.getpid(), fingerprint=fingerprint(task), **out)
    tmp.rename(path)
    return task.run, ds.p, ds.basis, Path(task.path).name, int(idx[0]), len(idx), wall, \
        float(np.mean(out["iterations"])), int(np.sum(out["failed"]))


def run_tasks(tasks, processes, cache, log=print):
    """Run every task whose output is not in the cache yet, in the order given, on a pool."""
    todo = [t for t in tasks if not task_file(cache, t).exists()]
    log(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done, {len(todo)} to run on {processes} processes")
    if not todo:
        return
    start = time.perf_counter()
    jobs = [(t, cache) for t in todo]
    if processes == 1:
        results = map(_work, jobs)
        pool = None
    else:
        pool = Pool(processes)
        results = pool.imap_unordered(_work, jobs)
    try:
        for n, (run, p, basis, name, first, count, wall, mean_its, fails) in enumerate(results, 1):
            log(f"[{time.perf_counter() - start:7.0f} s] {n}/{len(todo)} {run} p={p} {basis} {name} "
                f"shots {first}..{first + count - 1}: {wall:6.1f} s, mean iterations {mean_its:7.1f}, failures {fails}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def chunk_tasks(run, spec, ds_path, ds_sha, n_shots, first=0, last=None):
    """Tasks for consecutive CHUNK-shot blocks of a dataset, in file order."""
    last = n_shots if last is None else last
    return [Task(run, spec, str(ds_path), ds_sha, np.arange(a, min(a + CHUNK, last)))
            for a in range(first, last, CHUNK)]


def collect(tasks, cache):
    """Put the outputs of tasks (all of one run and one dataset) back together in task order."""
    parts = []
    for t in tasks:
        f = task_file(cache, t)
        if not f.exists():
            raise FileNotFoundError(f"missing output {f}: run the tasks first")
        with np.load(f) as z:
            if str(z["fingerprint"]) != fingerprint(t):
                raise ValueError(f"{f}: fingerprint mismatch")
            parts.append({k: z[k] for k in z.files if k not in ("wall", "pid", "fingerprint")})
    axis_of = lambda key, a: a.ndim - 1 if key != "indices" else 0   # the shot axis is the last
    return {k: np.concatenate([p[k] for p in parts], axis=axis_of(k, parts[0][k])) for k in parts[0]}


# ---------------------------------------------------------------- statistics

def wilson(k, n, z=1.959963984540054):
    """95% Wilson interval of k successes in n."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson(k, n, level=0.95):
    """Exact (Clopper-Pearson) interval of k successes in n."""
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def mcnemar_exact(only_a, only_b):
    """Two-sided exact McNemar p-value: a binomial test of the discordant pairs at 1/2."""
    n = only_a + only_b
    return 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)


def paired(wrong_a, wrong_b):
    """2 x 2 counts of two decoders on the same shots, and the exact paired test."""
    a, b = np.asarray(wrong_a, bool), np.asarray(wrong_b, bool)
    both, only_a, only_b, neither = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
    return dict(both_wrong=both, only_a_wrong=only_a, only_b_wrong=only_b, both_right=neither,
                mcnemar_p=mcnemar_exact(only_a, only_b))


def iteration_stats(its):
    its = np.asarray(its, float)
    return dict(mean=float(its.mean()), median=float(np.median(its)), p90=float(np.percentile(its, 90)),
                p99=float(np.percentile(its, 99)), max=float(its.max()))


# ---------------------------------------------------------------- provenance

def machine(processes):
    import ldpc

    import stim
    try:
        from importlib.metadata import version
        relay_version = version("relay-bp")
    except Exception:  # pragma: no cover
        relay_version = "unknown"
    cpu = "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return dict(host=platform.node(), cpu=cpu, logical_cpus=os.cpu_count(),
                affinity=sorted(os.sched_getaffinity(0)), processes=processes,
                python=sys.version.split()[0], numpy=np.__version__, stim=stim.__version__,
                ldpc=getattr(ldpc, "__version__", "unknown"), relay_bp=relay_version,
                nice=os.nice(0))


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
        f.write("\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


class Log:
    """Print and append to a log file, flushing every line."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", buffering=1)

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.f.write(line + "\n")
