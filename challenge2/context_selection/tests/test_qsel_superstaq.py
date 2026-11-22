"""Superstaq wrapper, tested against a fake provider (no network, no API key)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile

import iceberg
from qsel import qaoa_circuit, random_qubo, tune_angles
from qsel.backends import IdealBackend, make_backend
from qsel.superstaq_backend import SuperstaqBackend, _compiled_list


class FakeResult:
    def __init__(self, counts):
        self._counts = counts

    def get_counts(self):
        return self._counts[0] if len(self._counts) == 1 else self._counts


class FakeJob:
    def __init__(self, counts, job_id, fail=False):
        self._counts, self._id, self._fail = counts, job_id, fail

    def job_id(self):
        return self._id

    def result(self, **kwargs):
        if self._fail:
            raise TimeoutError("still queued")
        return FakeResult(self._counts)


class FakeOutput:
    def __init__(self, circuits):
        if len(circuits) == 1:
            self.circuit = circuits[0]
        else:
            self.circuits = circuits


class FakeTarget:
    def __init__(self, provider, name):
        self.provider, self.name = provider, name

    def run(self, circuits, shots, method=None, tag=(), **kwargs):
        self.provider.run_calls.append({"n": len(circuits), "shots": shots, "method": method,
                                        "tag": tag, "target": self.name})
        for c in circuits:
            json.dumps(c.metadata)                      # the real client serializes metadata
        counts = IdealBackend(seed=0).run(circuits, shots)
        job = FakeJob(counts, f"job-{len(self.provider.run_calls)}", fail=self.provider.fail_next)
        self.provider.fail_next = False
        self.provider.jobs[job.job_id()] = counts
        return job

    def retrieve_job(self, job_id):
        self.provider.retrieved.append(job_id)
        return FakeJob(self.provider.jobs[job_id], job_id)


class FakeProvider:
    def __init__(self, ibm_ok=True):
        self.run_calls, self.compile_calls, self.qubo_calls = [], [], []
        self.jobs, self.retrieved = {}, []
        self.fail_next = False
        self.ibm_ok = ibm_ok

    def get_backend(self, target):
        return FakeTarget(self, target)

    def cq_compile(self, circuits, target="cq_sqale_qpu", **kwargs):
        circuits = [circuits] if isinstance(circuits, QuantumCircuit) else list(circuits)
        self.compile_calls.append(("cq", target, kwargs, len(circuits)))
        return FakeOutput([transpile(c, basis_gates=["cz", "rz", "sx", "x"], optimization_level=1)
                           for c in circuits])

    def ibmq_compile(self, circuits, target, **kwargs):
        if not self.ibm_ok:
            raise RuntimeError("no IBM credentials")
        self.compile_calls.append(("ibm", target, kwargs, len(circuits)))
        return FakeOutput([transpile(c, basis_gates=["ecr", "rz", "sx", "x"], optimization_level=1)
                           for c in circuits])

    def submit_qubo(self, qubo, target, repetitions, method, max_solutions, **kwargs):
        self.qubo_calls.append({"qubo": qubo, "target": target, "repetitions": repetitions,
                                "method": method})
        return [{0: 1, 2: 1}, {0: 1, 2: 1}, {1: 1}]


@pytest.fixture(scope="module")
def circuits():
    out = []
    for seed in range(5):
        qubo = random_qubo(4, edge_fraction=0.5, seed=seed)
        out.append(qaoa_circuit(qubo, tune_angles(qubo, p=1).angles))
    return out


def test_batching_and_single_vs_list_results(circuits):
    provider = FakeProvider()
    backend = SuperstaqBackend(provider=provider, batch_size=2)
    counts = backend.run(circuits, 300)
    assert [c["n"] for c in provider.run_calls] == [2, 2, 1]        # last batch has one circuit
    assert all(c["method"] == "noise-sim" and c["shots"] == 300 for c in provider.run_calls)
    assert provider.run_calls[0]["target"] == "cq_sqale_simulator"
    assert len(counts) == 5 and all(sum(c.values()) == 300 for c in counts)
    assert counts == IdealBackend(seed=0).run(circuits, 300)        # order preserved
    assert backend.submissions == 3 and len(backend.timings) == 3
    assert backend.available() and backend.name == "superstaq[cq_sqale_simulator:noise-sim]"
    assert backend.run(circuits[0], 300) == [counts[0]]             # a bare circuit is accepted


def test_noiseless_method_is_not_sent(circuits):
    provider = FakeProvider()
    SuperstaqBackend(provider=provider, method=None).run(circuits[:1], 50)
    assert provider.run_calls[0]["method"] is None


def test_cache_hit_avoids_a_second_submission(circuits, tmp_path):
    first = FakeProvider()
    a = SuperstaqBackend(provider=first, batch_size=3, cache_dir=tmp_path)
    counts = a.run(circuits, 200)
    assert a.submissions == 2
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 5
    record = json.loads(files[0].read_text(encoding="utf-8"))
    assert {"counts", "shots", "target", "method", "job_id", "wall_time_s", "fingerprint"} <= set(record)

    second = FakeProvider()
    b = SuperstaqBackend(provider=second, batch_size=3, cache_dir=tmp_path)
    assert b.run(circuits, 200) == counts
    assert second.run_calls == [] and b.submissions == 0
    b.run(circuits[:2], 400)                                        # different shots: new job
    assert len(second.run_calls) == 1
    c = SuperstaqBackend(provider=second, target="cq_sqale_qpu", cache_dir=tmp_path)
    c.run(circuits[:1], 200)                                        # different target: new job
    assert len(second.run_calls) == 2


def test_interrupted_job_is_retrieved_not_resubmitted(circuits, tmp_path):
    provider = FakeProvider()
    provider.fail_next = True
    backend = SuperstaqBackend(provider=provider, cache_dir=tmp_path)
    with pytest.raises(TimeoutError):
        backend.run(circuits[:2], 100)
    assert len(list((tmp_path / "pending").glob("*.json"))) == 1
    again = SuperstaqBackend(provider=provider, cache_dir=tmp_path)
    counts = again.run(circuits[:2], 100)
    assert provider.retrieved == ["job-1"] and len(provider.run_calls) == 1
    assert again.submissions == 0 and len(counts) == 2
    assert list((tmp_path / "pending").glob("*.json")) == []


def test_metadata_that_cannot_be_serialized_is_dropped(circuits):
    provider = FakeProvider()
    circuit = circuits[0].copy()
    circuit.metadata = {"array": np.arange(3)}
    SuperstaqBackend(provider=provider).run([circuit], 50)
    assert isinstance(circuit.metadata["array"], np.ndarray)       # our copy is untouched


def test_iceberg_circuit_roundtrip_through_the_wrapper():
    qubo = random_qubo(4, edge_fraction=0.5, seed=1)
    enc = iceberg.encode(qaoa_circuit(qubo, tune_angles(qubo, p=1).angles))
    counts = SuperstaqBackend(provider=FakeProvider()).run([enc], 500)[0]
    _, stats = iceberg.postselect(counts, enc)
    assert stats.kept_share == 1.0


def test_compile_stats_with_and_without_ibm(circuits):
    provider = FakeProvider()
    backend = SuperstaqBackend(provider=provider, grid_shape=(4, 6))
    rows = backend.compile_stats(circuits[:2])
    assert provider.compile_calls[0] == ("cq", "cq_sqale_qpu", {"grid_shape": (4, 6)}, 2)
    assert provider.compile_calls[1][:2] == ("ibm", "ibmq_kingston_qpu")
    for row in rows:
        assert row["sqale"]["two_qubit_gates"] == row["sqale"]["count_ops"]["cz"]
        assert row["ibm"]["two_qubit_gates"] == row["ibm"]["count_ops"]["ecr"]
        assert row["input"]["two_qubit_gates"] == row["input"]["count_ops"]["rzz"]
    one = backend.compile_stats(circuits[0])                        # single circuit -> .circuit
    assert len(one) == 1 and one[0]["sqale"]["two_qubit_gates"] > 0
    assert backend.compile(circuits[0]).count_ops().get("cz", 0) > 0

    no_ibm = SuperstaqBackend(provider=FakeProvider(ibm_ok=False))
    rows = no_ibm.compile_stats(circuits[:2])
    assert "no IBM credentials" in rows[0]["ibm"]["error"] and "two_qubit_gates" in rows[0]["sqale"]
    with pytest.raises(TypeError):
        _compiled_list(object())


def test_submit_qubo_converts_solutions():
    provider = FakeProvider()
    qubo = random_qubo(4, edge_fraction=0.5, seed=0)
    counts = SuperstaqBackend(provider=provider).submit_qubo(qubo, repetitions=50)
    assert counts == {(1, 0, 1, 0): 2, (0, 1, 0, 0): 1}
    call = provider.qubo_calls[0]
    assert call["qubo"] == qubo.to_superstaq() and call["method"] == "sim_anneal"
    assert call["target"] == "ss_unconstrained_simulator" and call["repetitions"] == 50


def test_make_backend_superstaq(tmp_path):
    backend = make_backend({"kind": "superstaq", "provider": FakeProvider(), "grid_shape": [4, 6],
                            "cache_dir": str(tmp_path), "target": "cq_sqale_qpu", "method": None})
    assert isinstance(backend, SuperstaqBackend) and backend.grid_shape == (4, 6)
    assert backend.name == "superstaq[cq_sqale_qpu]"


def test_first_hour_check_script_runs_end_to_end(monkeypatch, capsys, tmp_path):
    """scripts/check_superstaq.py is run live in the first hour; it must not crash."""
    import importlib.util
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "check_superstaq.py"
    spec = importlib.util.spec_from_file_location("check_superstaq", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "superstaq_available", lambda: False)
    monkeypatch.setattr("sys.argv", ["check_superstaq.py"])
    assert module.main() == 2                                        # no key: explains and stops

    provider = FakeProvider()
    provider.get_balance = lambda: "$123.45"
    provider.backends = lambda: [FakeTarget(provider, "cq_sqale_simulator"), FakeTarget(provider, "ibmq_x")]
    real = module.SuperstaqBackend
    monkeypatch.setattr(module, "superstaq_available", lambda: True)
    monkeypatch.setattr(module, "SuperstaqBackend", lambda **kw: real(provider=provider, **kw))
    log = tmp_path / "check.json"
    monkeypatch.setattr("sys.argv", ["check_superstaq.py", "--shots", "3000", "--n", "4", "--log", str(log)])
    assert module.main() == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "kept 100.0% of shots" in out and "cq_sqale_simulator" in out
    assert "two-qubit gates" in out
    assert "[credits]" in out
    record = json.loads(log.read_text())
    assert record["noiseless"]["pass"] and record["credits"]["start"] == 123.45

    # --budget: no Bell tests, no IBM compile; still the QAOA check, noise-sim and cq_compile
    monkeypatch.setattr("sys.argv", ["check_superstaq.py", "--shots", "3000", "--n", "4", "--budget",
                                     "--log", str(log)])
    assert module.main() == 0
    out = capsys.readouterr().out
    assert "Bell" not in out and "PASS" in out
    steps = [s["step"] for s in json.loads(log.read_text())["credits"]["steps"]]
    assert steps == ["qaoa noiseless", "qaoa noise-sim", "cq_compile"]
