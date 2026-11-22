"""Sqale through Superstaq: run circuits, compile statistics, and the QUBO service.

Written against ``qiskit-superstaq`` 0.5.x:

    provider = qss.SuperstaqProvider()                  # reads SUPERSTAQ_API_KEY
    backend = provider.get_backend("cq_sqale_simulator")
    job = backend.run(circuits, shots=4000, method="noise-sim")
    counts = job.result().get_counts()                  # dict for one circuit, list for several
    out = provider.cq_compile(circuits)                 # out.circuit or out.circuits
    out = provider.ibmq_compile(circuits, target="ibmq_kingston_qpu")
    sols = provider.submit_qubo(qubo_dict, target, repetitions=..., method="sim_anneal")

Superstaq already returns counts in Qiskit bit order. Every result is cached on disk
under ``cache_dir`` keyed by the circuit's structure, the shots, the target and the
method, so a rerun (or a clone of the repository with the committed cache) reproduces
the numbers without network access or credits.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from qiskit import QuantumCircuit

from .backends import _clean_counts, circuit_fingerprint, circuit_stats
from .bits import Selection
from .qubo import Qubo


def superstaq_available(api_key: str | None = None) -> bool:
    """True when ``qiskit-superstaq`` is importable and an API key can be found."""
    try:
        import general_superstaq as gss
        import qiskit_superstaq  # noqa: F401
    except Exception:
        return False
    if api_key or os.environ.get("SUPERSTAQ_API_KEY"):
        return True
    try:
        return bool(gss.superstaq_client.find_api_key())
    except Exception:
        return False


def _json_safe(meta: Any) -> dict:
    """Circuit metadata that survives serialization, or an empty dict."""
    if not isinstance(meta, dict):
        return {}
    try:
        return json.loads(json.dumps(meta))
    except (TypeError, ValueError):
        return {}


def _compiled_list(output: Any) -> list[QuantumCircuit]:
    """Circuits from a Superstaq compile result, whatever shape it has."""
    if hasattr(output, "circuits"):
        circuits = output.circuits
    elif hasattr(output, "circuit"):
        circuits = [output.circuit]
    elif hasattr(output, "compiled_circuits"):
        circuits = output.compiled_circuits()
    else:
        raise TypeError(f"unrecognised compile result {type(output).__name__}")
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    return list(circuits)


class SuperstaqBackend:
    """Run sampling circuits on a Superstaq target, with batching and a disk cache."""

    def __init__(
        self,
        target: str = "cq_sqale_simulator",
        method: str | None = "noise-sim",
        provider: Any = None,
        api_key: str | None = None,
        batch_size: int = 20,
        cache_dir: str | os.PathLike | None = None,
        grid_shape: tuple[int, int] | None = None,
        ibm_target: str = "ibmq_kingston_qpu",
        tag: str = "qctx",
        compile_target: str = "cq_sqale_qpu",
        timeout: float | None = None,
    ):
        self.target = target
        self.method = method
        self._provider = provider
        self._api_key = api_key
        self.batch_size = max(1, int(batch_size))
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.grid_shape = tuple(grid_shape) if grid_shape is not None else None
        self.ibm_target = ibm_target
        self.tag = tag
        self.compile_target = compile_target
        self.timeout = timeout
        self.name = f"superstaq[{target}{':' + method if method else ''}]"
        #: one entry per submitted batch: circuits, job id, seconds from submit to result
        self.timings: list[dict[str, Any]] = []
        #: number of jobs actually submitted (cache hits do not count)
        self.submissions = 0

    # ------------------------------------------------------------ connection
    @property
    def provider(self) -> Any:
        if self._provider is None:
            import qiskit_superstaq as qss

            self._provider = qss.SuperstaqProvider(self._api_key)
        return self._provider

    def available(self) -> bool:
        """True when a provider was injected, or the package and an API key are present."""
        return self._provider is not None or superstaq_available(self._api_key)

    # ----------------------------------------------------------------- cache
    def _key(self, circuit: QuantumCircuit, shots: int) -> str:
        payload = f"{circuit_fingerprint(circuit)}|{int(shots)}|{self.target}|{self.method}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / f"{key[:32]}.json" if self.cache_dir is not None else None

    def _load(self, key: str) -> dict[str, int] | None:
        path = self._cache_path(key)
        if path is None or not path.is_file():
            return None
        with path.open(encoding="utf-8") as fh:
            return {str(k): int(v) for k, v in json.load(fh)["counts"].items()}

    def _store(self, key: str, circuit: QuantumCircuit, shots: int, counts: Mapping[str, int],
               job_id: str, wall: float) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "counts": dict(counts),
            "shots": int(shots),
            "target": self.target,
            "method": self.method,
            "job_id": job_id,
            "wall_time_s": wall,
            "circuit": circuit.name,
            "fingerprint": circuit_fingerprint(circuit),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=1)

    def _pending_path(self, keys: Sequence[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256("|".join(keys).encode()).hexdigest()[:32]
        return self.cache_dir / "pending" / f"{digest}.json"

    # ------------------------------------------------------------------- run
    def compile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """The circuit as Superstaq compiles it for Sqale (one network call)."""
        kwargs = {"grid_shape": self.grid_shape} if self.grid_shape else {}
        return _compiled_list(self.provider.cq_compile(self._submittable(circuit),
                                                       target=self.compile_target, **kwargs))[0]

    @staticmethod
    def _submittable(circuit: QuantumCircuit) -> QuantumCircuit:
        copy = circuit.copy()
        copy.metadata = _json_safe(circuit.metadata)
        return copy

    def _submit(self, batch: Sequence[QuantumCircuit], shots: int, keys: Sequence[str]) -> list[dict[str, int]]:
        backend = self.provider.get_backend(self.target)
        pending = self._pending_path(keys)
        t0 = time.time()
        job = None
        if pending is not None and pending.is_file():
            # a previous run submitted this batch and stopped before the result arrived
            with pending.open(encoding="utf-8") as fh:
                job = backend.retrieve_job(json.load(fh)["job_id"])
        if job is None:
            kwargs: dict[str, Any] = {"shots": int(shots), "tag": self.tag}
            if self.method:
                kwargs["method"] = self.method
            job = backend.run([self._submittable(c) for c in batch], **kwargs)
            self.submissions += 1
        job_id = job.job_id() if callable(getattr(job, "job_id", None)) else str(getattr(job, "job_id", ""))
        if pending is not None:
            pending.parent.mkdir(parents=True, exist_ok=True)
            with pending.open("w", encoding="utf-8") as fh:
                json.dump({"job_id": job_id, "keys": list(keys), "target": self.target}, fh)
        t_submit = time.time() - t0
        result = job.result(timeout=self.timeout) if self.timeout else job.result()
        raw = result.get_counts()
        counts_list = [raw] if isinstance(raw, Mapping) else list(raw)
        if len(counts_list) != len(batch):
            raise RuntimeError(f"submitted {len(batch)} circuits, received {len(counts_list)} results")
        wall = time.time() - t0
        self.timings.append({"circuits": len(batch), "shots": int(shots), "job_id": job_id,
                             "submit_s": t_submit, "wall_s": wall})
        if pending is not None and pending.is_file():
            pending.unlink()
        return [_clean_counts(c) for c in counts_list]

    def run(self, circuits: Sequence[QuantumCircuit], shots: int) -> list[dict[str, int]]:
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        circuits = list(circuits)
        keys = [self._key(c, shots) for c in circuits]
        results: list[dict[str, int] | None] = [self._load(k) for k in keys]
        todo = [i for i, r in enumerate(results) if r is None]
        for lo in range(0, len(todo), self.batch_size):
            idx = todo[lo:lo + self.batch_size]
            batch_keys = [keys[i] for i in idx]
            counts_list = self._submit([circuits[i] for i in idx], shots, batch_keys)
            wall = self.timings[-1]["wall_s"] / max(len(idx), 1)
            job_id = self.timings[-1]["job_id"]
            for i, counts in zip(idx, counts_list):
                results[i] = counts
                self._store(keys[i], circuits[i], shots, counts, job_id, wall)
        return [r if r is not None else {} for r in results]

    # --------------------------------------------------------- compile stats
    def compile_stats(self, circuits: Sequence[QuantumCircuit], *, ibm: bool = True) -> list[dict[str, Any]]:
        """Resources before and after ``cq_compile``, plus an IBM compile for comparison.

        Each entry has ``input``, ``sqale`` and ``ibm`` blocks from
        :func:`qsel.backends.circuit_stats`. IBM access is optional: if that compile
        fails, the ``ibm`` block holds the error message instead. ``ibm=False`` skips the
        IBM compile entirely (one fewer billed call).
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        circuits = list(circuits)
        send = [self._submittable(c) for c in circuits]
        kwargs = {"grid_shape": self.grid_shape} if self.grid_shape else {}
        sqale = [circuit_stats(c) for c in
                 _compiled_list(self.provider.cq_compile(send, target=self.compile_target, **kwargs))]
        ibm_rows: list[dict[str, Any]]
        if not ibm:
            ibm_rows = [{"error": "skipped"} for _ in circuits]
        else:
            try:
                ibm_rows = [circuit_stats(c) for c in
                            _compiled_list(self.provider.ibmq_compile(send, target=self.ibm_target))]
            except Exception as exc:  # IBM credentials or target may be missing
                ibm_rows = [{"error": f"{type(exc).__name__}: {exc}"} for _ in circuits]
        out = []
        for i, circuit in enumerate(circuits):
            out.append({
                "name": circuit.name,
                "input": circuit_stats(circuit),
                "sqale": sqale[i] if i < len(sqale) else {"error": "missing result"},
                "sqale_target": self.compile_target,
                "ibm": ibm_rows[i] if i < len(ibm_rows) else {"error": "missing result"},
                "ibm_target": self.ibm_target,
            })
        return out

    # ------------------------------------------------------------------ qubo
    def submit_qubo(
        self,
        qubo: Qubo,
        *,
        method: str = "sim_anneal",
        repetitions: int = 1000,
        target: str = "ss_unconstrained_simulator",
        max_solutions: int = 1000,
        **kwargs: Any,
    ) -> dict[Selection, int]:
        """Solve the QUBO with Superstaq's solver service; returns selection counts.

        ``method`` is ``"sim_anneal"``, ``"bruteforce"``, ``"qaoa"`` or ``"rqaoa"``.
        Variables the service omits (zero weight, no edge) are reported as not selected.
        """
        t0 = time.time()
        solutions = self.provider.submit_qubo(
            qubo.to_superstaq(), target=target, repetitions=int(repetitions), method=method,
            max_solutions=int(max_solutions), **kwargs,
        )
        self.timings.append({"qubo_method": method, "wall_s": time.time() - t0,
                             "solutions": len(solutions)})
        counts: dict[Selection, int] = {}
        for sol in solutions:
            x = tuple(int(bool(sol.get(i, 0))) for i in range(qubo.n))
            counts[x] = counts.get(x, 0) + 1
        return counts
