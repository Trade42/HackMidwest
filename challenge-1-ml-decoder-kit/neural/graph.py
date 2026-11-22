"""The code's structure as index tables for the network, read from the kit.

Nothing here is typed in by hand:
- which data qubits a check touches comes from make_circuit.code_layout;
- which check and which round a detector belongs to comes from the memory circuit itself,
  by following every DETECTOR back to the measurements it compares;
- which data qubits a logical observable reads comes from the circuit's OBSERVABLE_INCLUDE lines.

Nodes. The network keeps numbers on check nodes and on data nodes.
- Check nodes: the n2 X checks, then the n2 Z checks (n2 = 72 for the Gross code).
- Data nodes: the n2 left data qubits, then the n2 right data qubits.
Within each block of n2, node i sits at x = i // m, y = i % m on the ell x m torus.

Time. A memory experiment with R noisy cycles has R + 1 detector rounds for the checks of
the memory basis (round R compares the last check value with the data readout) and R - 1
for the other checks (rounds 1 to R - 1). Check nodes exist in rounds 0 to R, and data
nodes in cycles 0 to R - 1. An error on a data qubit in cycle c shows up on its checks in
round c or c + 1.

Logical operators. The kit's 12 observables of a memory basis are one choice of 12 logical
operators. For the Gross code its X operators touch 12 to 14 data qubits each, its Z operators
12 to 38. With logicals="light" the network predicts 12 other operators of the same basis, as
light as the lighter of the two sets: the other basis' operators, carried over by the code's
X/Z symmetry. The kit's 12 bits follow from those 12 bits exactly:

    kit bits = kit_from_model x model bits + kit_from_checks x check parities   (mod 2)

where the parity of a check is the sum of its detectors over all rounds, which equals the
parity of the data readout on that check. A shot is wrong in one set of operators exactly
when it is wrong in the other.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import stim

from circuit import CHECK_COORDINATE, detector_ids
from make_circuit import CODES, code_layout, gf2_row_reduce, logical_operators, memory_experiment

X_TYPE, Z_TYPE = 0, 1   # the two blocks of check nodes


@dataclass(frozen=True)
class CodeGraph:
    """Index tables of one memory experiment."""
    code: str
    basis: str                       # memory basis, "X" or "Z"
    rounds: int                      # noisy syndrome cycles, R
    ell: int                         # torus size in x
    m: int                           # torus size in y
    check_neighbours: np.ndarray     # [2 n2, 6]: the data nodes of every check node, in the kit's order A1 A2 A3 B1 B2 B3
    data_neighbours: np.ndarray      # [2 n2, 6]: the check nodes of every data node, 3 X checks and then 3 Z checks
    detector_round: np.ndarray       # [detectors]: the round of every detector of the circuit
    detector_check: np.ndarray       # [detectors]: the check node of every detector of the circuit
    basis_detectors: np.ndarray      # ids of the detectors on checks of the memory basis (the baselines' input)
    observable_support: np.ndarray   # [observables, 2 n2]: 1 where the model's logical operator reads the data node
    logicals: str                    # "kit": the kit's operators. "light": the lighter set (see the top of this file)
    kit_from_model: np.ndarray       # [observables, observables] over GF(2); the identity for "kit"
    kit_from_checks: np.ndarray      # [observables, n2] over GF(2); zero for "kit"

    @property
    def n2(self):
        """Checks of one type. Also the data qubits of one block."""
        return self.ell * self.m

    @property
    def num_detectors(self):
        return len(self.detector_round)

    def torus_shift(self, dx, dy):
        """Where every node goes when the torus is shifted by (dx, dy). The same table serves check and data nodes."""
        i = np.arange(self.n2)
        moved = ((i // self.m + dx) % self.ell) * self.m + (i % self.m + dy) % self.m
        return np.concatenate([moved, self.n2 + moved])


@lru_cache(maxsize=None)
def build_graph(code="gross", basis="X", rounds=None, logicals="kit"):
    """The tables of one code and memory basis, from the kit's generator. The result is cached: do not change its arrays."""
    ell, m, _, _, default_rounds = CODES[code]
    rounds = default_rounds if rounds is None else rounds
    layout = code_layout(code)
    circuit = memory_experiment(code, basis, rounds)
    check_neighbours = _check_neighbours(layout)
    detector_round, detector_check = _detector_positions(circuit, layout)
    kit_support = _observable_support(circuit, layout)
    if logicals == "kit":
        support, kit_from_model = kit_support, np.eye(len(kit_support), dtype=np.uint8)
        kit_from_checks = np.zeros((len(kit_support), layout.n2), dtype=np.uint8)
    elif logicals == "light":
        support, kit_from_model, kit_from_checks = _light_logicals(layout, basis, ell, m, kit_support)
    else:
        raise ValueError("logicals is 'kit' or 'light'")
    return CodeGraph(code, basis, rounds, ell, m, check_neighbours, _data_neighbours(check_neighbours),
                     detector_round, detector_check, detector_ids(circuit, basis),
                     support, logicals, kit_from_model, kit_from_checks)


def _light_logicals(layout, basis, ell, m, kit_support):
    """(supports, kit_from_model, kit_from_checks) for the lighter set of logical operators.

    The code looks the same with X and Z exchanged, the two data blocks exchanged and every
    torus position negated. That turns the kit's logical operators of the other basis into
    logical operators of this basis. They are used when their heaviest one is lighter than the
    kit's heaviest; otherwise the kit's own operators are kept.
    """
    n2, other = layout.n2, "Z" if basis == "X" else "X"
    position = np.arange(n2)
    negated = ((-(position // m)) % ell) * m + (-(position % m)) % m
    mirror = np.concatenate([n2 + negated, negated])              # data node (block, position) -> (other block, -position)
    if not np.array_equal(layout.hz, layout.hx[negated][:, mirror]):
        raise ValueError("this code does not have the bivariate bicycle X/Z symmetry")
    checks = layout.checks(basis)
    carried = logical_operators(layout.checks(other), checks)[:, mirror]
    support = carried if carried.sum(axis=1).max() < kit_support.sum(axis=1).max() else kit_support
    # Solve kit operator = sum of some chosen operators + sum of some checks, over GF(2).
    unknowns = np.vstack([support, checks]).astype(np.uint8)
    reduced, pivots = gf2_row_reduce(np.hstack([unknowns.T, kit_support.T]).astype(np.uint8))
    solution = np.zeros((len(kit_support), len(unknowns)), dtype=np.uint8)
    for row, pivot in enumerate(pivots):
        if pivot >= len(unknowns):
            raise ValueError("the chosen operators and the checks do not span the kit's logical operators")
        solution[:, pivot] = reduced[row, len(unknowns):]
    kit_from_model, kit_from_checks = solution[:, :len(support)], solution[:, len(support):]
    if not np.array_equal((kit_from_model @ support + kit_from_checks @ checks) % 2, kit_support):
        raise ValueError("the change of logical operators does not reproduce the kit's")
    return support.astype(np.uint8), kit_from_model, kit_from_checks


def gf2_inverse(matrix):
    """The inverse of a square 0/1 matrix over GF(2)."""
    size = len(matrix)
    reduced, pivots = gf2_row_reduce(np.hstack([matrix, np.eye(size, dtype=np.uint8)]).astype(np.uint8))
    if pivots[:size] != list(range(size)):
        raise ValueError("the matrix has no inverse over GF(2)")
    return reduced[:, size:]


def check_parity_matrix(graph, detector_ids_used):
    """[n2, len(detector_ids_used)] 0/1: which of the given detectors belong to each check of the memory basis.

    Times a shot's detector bits (mod 2) it gives the parity of every check of the memory
    basis over all rounds: the parity of the data readout on that check.
    """
    basis_type = "XZ".index(graph.basis)
    matrix = np.zeros((graph.n2, len(detector_ids_used)), dtype=np.uint8)
    for column, detector in enumerate(detector_ids_used):
        check = graph.detector_check[detector]
        if check // graph.n2 == basis_type:
            matrix[check % graph.n2, column] = 1
    return matrix


def _check_neighbours(layout):
    """[2 n2, 6]: data node of neighbour d of every check node. Data node = qubit index - n2."""
    x = np.stack(layout.x_neighbours, axis=1) - layout.n2
    z = np.stack(layout.z_neighbours, axis=1) - layout.n2
    return np.concatenate([x, z]).astype(np.int64)


def _data_neighbours(check_neighbours):
    """[2 n2, 6]: the checks that touch every data node, found by turning check_neighbours around.

    A left data qubit is neighbour 0, 1 or 2 of exactly one X check each, and the same for the
    Z checks. A right data qubit is neighbour 3, 4 or 5. The slots are ordered X checks first,
    each by that neighbour number, so a slot means the same torus offset for every data qubit.
    """
    n_checks, n_slots = check_neighbours.shape
    n2 = n_checks // 2
    table = np.full((n_checks, n_slots), -1, dtype=np.int64)
    for check in range(n_checks):
        check_type = check // n2
        for d, data in enumerate(check_neighbours[check]):
            slot = 3 * check_type + d % 3
            if d // 3 != data // n2 or table[data, slot] != -1:
                raise ValueError("the code's neighbours do not have the bivariate bicycle pattern")
            table[data, slot] = check
    if (table < 0).any():
        raise ValueError("a data qubit has fewer than six checks")
    return table


def _measured_qubits(circuit):
    """Yield (instruction, qubits measured so far) for every instruction of the flattened circuit."""
    measured = []
    for instruction in circuit.flattened():
        if stim.gate_data(instruction.name).produces_measurements:
            measured.extend(target.value for target in instruction.targets_copy())
        yield instruction, measured


def _detector_positions(circuit, layout):
    """(round, check node) of every detector, in detector order.

    A detector compares a check's newest outcome with its previous one, or (in the last round)
    with the data readout. Its check is the one check qubit among its targets. Its round is the
    number of earlier outcomes of that check, plus one when the data readout takes part.
    """
    data, n2 = set(layout.data), layout.n2
    rounds, checks = [], []
    for instruction, measured in _measured_qubits(circuit):
        if instruction.name != "DETECTOR":
            continue
        qubits = [measured[target.value] for target in instruction.targets_copy()]
        check_qubits = {q for q in qubits if q not in data}
        if len(check_qubits) != 1:
            raise ValueError("a detector must compare outcomes of exactly one check")
        qubit = check_qubits.pop()
        check_type = X_TYPE if qubit in layout.x_checks else Z_TYPE
        if instruction.gate_args_copy() != [CHECK_COORDINATE["XZ"[check_type]]]:
            raise ValueError("a detector's coordinate disagrees with the type of its check")
        uses_readout = any(q in data for q in qubits)
        rounds.append(measured.count(qubit) - 1 + uses_readout)
        checks.append(qubit if check_type == X_TYPE else qubit - 3 * n2 + n2)
    return np.array(rounds, dtype=np.int64), np.array(checks, dtype=np.int64)


def _observable_support(circuit, layout):
    """[observables, data nodes]: 1 where the observable reads that data qubit."""
    support = np.zeros((circuit.num_observables, 2 * layout.n2), dtype=np.uint8)
    for instruction, measured in _measured_qubits(circuit):
        if instruction.name == "OBSERVABLE_INCLUDE":
            index = int(instruction.gate_args_copy()[0])
            for target in instruction.targets_copy():
                support[index, measured[target.value] - layout.n2] ^= 1
    return support
