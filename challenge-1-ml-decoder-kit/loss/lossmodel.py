"""The atom-loss model of this folder, the Clifft circuit that simulates it, and the layout of the kit's circuits.

The loss model (our assumption, stated in RESULTS.md):
- after every CX, each of its two atoms is lost with probability s (Clifft's LOSS(s) on both operands);
- a lost atom takes no further part: Clifft skips every gate and every noise channel that touches it;
- a measurement of a lost atom is heralded (perfectly) and records a uniformly random bit;
- a reset restores a lost atom (reset_restores_lost=True). Check atoms are reset every cycle, data atoms never,
  so a lost data atom stays lost and is heralded only at the final readout;
- the kit's Pauli noise at strength p stays as it is.

Optionally the circuit carries status probes: a noiseless, non-destructive HERALD_LOSS_EVENT on both atoms after
every CX. They append ordinary record bits, so every DETECTOR and OBSERVABLE_INCLUDE lookback is remapped. The
probes give the true loss time of every atom. Only the oracle of L3 and the flicker table of L5 read them.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import stim

from circuit import GROSS_CIRCUITS, detector_ids, load_circuit

N = 72                  # checks of each type, and data atoms of each half
CYCLES = 12
X_ANC, LEFT, RIGHT, Z_ANC = (range(k * N, (k + 1) * N) for k in range(4))   # the kit's qubit blocks


class Layout:
    """Where every CX, measurement and detector of a kit circuit sits, read from the circuit itself.

    cx[q]       the CX gates of qubit q in time order: (tick, role 'C' or 'T', partner)
    records     one entry per measurement record: (qubit, tick, basis 'X' or 'Z', occurrence of that qubit)
    det_check   for each detector: (check qubit, round). Round r < 12 compares the check in cycle r with cycle r - 1
                (or stands alone in cycle 0); round 12 compares the last cycle with the final data readout.
    det_records for each detector: the absolute record indices it XORs
    obs_records for each observable: the absolute record indices it XORs
    """

    def __init__(self, circuit):
        self.circuit = circuit
        self.flat = circuit.flattened()
        self.num_qubits = circuit.num_qubits
        self.cx = {q: [] for q in range(self.num_qubits)}
        self.records, self.det_records, self.obs_records = [], [], {}
        seen = {}
        tick = 0
        for inst in self.flat:
            name = inst.name
            if name == "TICK":
                tick += 1
                continue
            targets = inst.targets_copy()
            if name == "CX":
                qs = [t.value for t in targets]
                for c, t in zip(qs[::2], qs[1::2], strict=True):
                    self.cx[c].append((tick, "C", t))
                    self.cx[t].append((tick, "T", c))
            elif name in ("M", "MX"):
                for t in targets:
                    q = t.value
                    self.records.append((q, tick, "Z" if name == "M" else "X", seen.get(q, 0)))
                    seen[q] = seen.get(q, 0) + 1
            elif name == "DETECTOR":
                self.det_records.append([len(self.records) + t.value for t in targets])
            elif name == "OBSERVABLE_INCLUDE":
                k = int(inst.gate_args_copy()[0])
                self.obs_records.setdefault(k, []).extend(len(self.records) + t.value for t in targets)
            elif name not in ("R", "RX", "X_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2", "QUBIT_COORDS",
                              "SHIFT_COORDS"):
                raise ValueError(f"unexpected instruction {name}")
        self.ticks = tick
        self.num_records = len(self.records)
        self._check_blocks()
        self.det_check = [self._detector_check(recs) for recs in self.det_records]

    def _check_blocks(self):
        """The qubit blocks of circuits/README.md: X checks, left data, right data, Z checks."""
        measured = {}
        for q, _, basis, _ in self.records:
            measured.setdefault(q, []).append(basis)
        for q in X_ANC:
            assert measured[q] == ["X"] * CYCLES, f"X check {q} is not measured in X once per cycle"
        for q in Z_ANC:
            assert measured[q] == ["Z"] * CYCLES, f"Z check {q} is not measured in Z once per cycle"
        final = {measured[q][0] for q in (*LEFT, *RIGHT)}
        assert all(len(measured[q]) == 1 for q in (*LEFT, *RIGHT)) and len(final) == 1
        self.memory_basis = final.pop()
        for q in range(self.num_qubits):
            assert len(self.cx[q]) == 6 * CYCLES, f"qubit {q} has {len(self.cx[q])} CX, not 72"

    def _detector_check(self, recs):
        checks = [(self.records[r][0], self.records[r][3]) for r in recs if self.records[r][0] in (*X_ANC, *Z_ANC)]
        data = [r for r in recs if self.records[r][0] not in (*X_ANC, *Z_ANC)]
        qubits = {q for q, _ in checks}
        assert len(qubits) == 1, f"a detector reads more than one check: {qubits}"
        cycle = max(c for _, c in checks)
        return qubits.pop(), CYCLES if data else cycle

    def record_of(self, qubit, occurrence=0):
        """Absolute record index of the occurrence-th measurement of qubit."""
        return self._record_index[(qubit, occurrence)]

    @property
    def _record_index(self):
        if not hasattr(self, "_ri"):
            self._ri = {(q, k): i for i, (q, _, _, k) in enumerate(self.records)}
        return self._ri


def kit_circuit(basis, p):
    """The kit's memory circuit of the given basis at Pauli strength p (p = 0 gives the noiseless circuit)."""
    circuit = load_circuit(GROSS_CIRCUITS[basis], 0.001)
    return circuit.without_noise() if p == 0 else load_circuit(GROSS_CIRCUITS[basis], p)


def clifft_text(circuit, s, probes=False):
    """Clifft circuit text: the flattened kit circuit with LOSS(s) on both atoms after every CX.

    With probes, HERALD_LOSS_EVENT follows each LOSS on the same atoms. Returns (text, probe_layout), where
    probe_layout[i] = (qubit, CX index of that qubit) for the i-th probe record, in record order, or None.
    """
    lines, probe_layout = [], []
    new_index = []            # original record index -> record index in the new circuit
    n_new = 0
    cx_count = {}
    for inst in circuit.flattened():
        name, targets, args = inst.name, inst.targets_copy(), inst.gate_args_copy()
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            n_orig = len(new_index)
            recs = [new_index[n_orig + t.value] - n_new for t in targets]
            arg = "(" + ", ".join(f"{a:g}" for a in args) + ")" if args else ""
            if name == "OBSERVABLE_INCLUDE":
                arg = f"({int(args[0])})"
            lines.append(f"{name}{arg} " + " ".join(f"rec[{r}]" for r in recs))
            continue
        lines.append(str(inst))
        if name in ("M", "MX"):
            for _ in targets:
                new_index.append(n_new)
                n_new += 1
        elif name == "CX":
            qs = " ".join(str(t.value) for t in targets)
            lines.append(f"LOSS({s!r}) {qs}")
            if probes:
                lines.append(f"HERALD_LOSS_EVENT {qs}")
                for t in targets:
                    probe_layout.append((t.value, cx_count.get(t.value, 0)))
                    cx_count[t.value] = cx_count.get(t.value, 0) + 1
                    n_new += 1
    return "\n".join(lines) + "\n", (probe_layout if probes else None), np.array(new_index)


def clifft_model():
    """Perfect three-outcome readout: g and leak_g read 0, e and leak_e read 1, a lost atom is heralded."""
    from clifft import noncomp
    C = np.zeros((3, 5))
    C[0, 0] = C[0, 2] = 1.0
    C[1, 1] = C[1, 3] = 1.0
    C[2, 4] = 1.0
    return noncomp.Model(classifier=noncomp.Classifier(C.tolist()), reset_restores_lost=True)


def sample_chunk(text, shots, seed, threads=1):
    """One Clifft call: shots of the loss circuit with a fixed seed. Returns the raw sample."""
    from clifft import noncomp
    program = noncomp.compile(text, clifft_model(), execution="online")
    return program.sample(shots, seed=seed, threads=threads)


def loss_times(layout, probe_layout, new_index, sample):
    """Per shot: the CX index (1..72) after which each atom was first lost in each cycle, from the probes.

    Returns (data_slot, check_k):
      data_slot[shot, q] for the 144 data atoms (q - 72): 0 if never lost, else j in 1..72 (lost after its j-th CX);
      check_k[shot, a, c] for the 144 check atoms (X checks 0..71, then Z checks 72..143) in cycle c: 0 if not
      lost in that cycle, else k in 1..6 (lost after its k-th CX of the cycle).
    It also asserts what the probes must satisfy: a probe never reads lost before the atom could be lost, a data
    atom once lost stays lost, and the herald of every measurement equals the probe after the atom's last CX.
    """
    meas, heralds = sample.measurements, sample.heralds
    shots = meas.shape[0]
    kept = set(new_index.tolist())
    probe_cols = [i for i in range(meas.shape[1]) if i not in kept]
    assert len(probe_cols) == len(probe_layout)
    status = {}                                   # qubit -> array (shots, 72) of probe bits after each CX
    for col, (q, j) in zip(probe_cols, probe_layout, strict=True):
        status.setdefault(q, np.zeros((shots, 6 * CYCLES), np.uint8))[:, j] = meas[:, col]
        assert not heralds[:, col].any(), "a status probe carried a herald"
    data_slot = np.zeros((shots, 2 * N), np.int16)
    for q in (*LEFT, *RIGHT):
        st = status[q]
        assert (np.diff(st.astype(np.int8), axis=1) >= 0).all(), f"data atom {q} came back after a loss"
        lost = st.any(axis=1)
        data_slot[lost, q - N] = st[lost].argmax(axis=1) + 1
        final = heralds[:, new_index[layout.record_of(q)]]
        assert (final == st[:, -1]).all(), f"final herald of data atom {q} disagrees with its probes"
    check_k = np.zeros((shots, 2 * N, CYCLES), np.int8)
    for a in (*X_ANC, *Z_ANC):
        st = status[a].reshape(shots, CYCLES, 6)
        idx = a if a < N else a - 2 * N      # X checks 0..71, Z checks 72..143
        for c in range(CYCLES):
            s_c = st[:, c]
            assert (np.diff(s_c.astype(np.int8), axis=1) >= 0).all(), f"check atom {a} came back inside a cycle"
            lost = s_c.any(axis=1)
            check_k[lost, idx, c] = s_c[lost].argmax(axis=1) + 1
            h = heralds[:, new_index[layout.record_of(a, c)]]
            assert (h == s_c[:, -1]).all(), f"herald of check atom {a} in cycle {c} disagrees with its probes"
    return data_slot, check_k


def make_dataset(basis, p, s, shots, seed, chunk=2000, threads=1, probes=True):
    """Sample a loss dataset in chunks of fixed size; chunk k uses seed + k. Returns a dict of arrays.

    The chunking is fixed by (shots, chunk), so the same arguments give the same shots for any thread count.
    """
    circuit = kit_circuit(basis, p)
    layout = Layout(circuit)
    text, probe_layout, new_index = clifft_text(circuit, s, probes=probes)
    parts = []
    for k, start in enumerate(range(0, shots, chunk)):
        n = min(chunk, shots - start)
        smp = sample_chunk(text, n, seed + k, threads=threads)
        part = {"detectors": np.asarray(smp.detectors, np.uint8),
                "observables": np.asarray(smp.observables, np.uint8),
                "heralds": np.asarray(smp.heralds, np.uint8)[:, new_index]}
        if probes:
            part["data_slot"], part["check_k"] = loss_times(layout, probe_layout, new_index, smp)
        parts.append(part)
    out = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    out["meta"] = np.array(json.dumps({"basis": basis, "p": p, "s": s, "shots": shots, "seed": seed, "chunk": chunk,
                                       "probes": probes, "clifft": _clifft_version()}))
    return out


def _clifft_version():
    import clifft
    return clifft.__version__


def save_dataset(data, path):
    """Save compressed; return the sha256 of the file."""
    path = Path(path)
    np.savez_compressed(path, **data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(path):
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    data["meta"] = json.loads(str(data["meta"]))
    return data


def basis_rows(circuit, basis):
    """Detector ids of the decoded basis, in the row order of dem_matrices(circuit, basis)."""
    return detector_ids(circuit, basis)
