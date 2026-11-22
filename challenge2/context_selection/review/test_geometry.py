"""Bus layout claims: nearest-neighbour two-qubit gates on a 3 x C grid; no data-(t|b) gate outside init/syndrome."""
import numpy as np
import iceberg, qsel
from qsel import Angles
from vlib import block_labels

rng = np.random.default_rng(0)
n_circ = n_2q = 0
viol_nn, viol_hub, hub_blocks, bad_pos = [], [], {}, []
for k in range(1, 13):
    for ef in (0.0, 0.3, 0.6, 1.0):
        for p in (1, 2):
            for mode in ("direct", "bus"):
                for trial in range(2):
                    q = qsel.random_qubo(k, edge_fraction=ef, seed=100 * k + trial)
                    bare = qsel.qaoa_circuit(q, Angles(tuple(rng.uniform(-1, 1, p)), tuple(rng.uniform(-1, 1, p))))
                    kw = {}
                    if trial == 1:
                        K = k + k % 2
                        kw["placement"] = [int(x) for x in rng.permutation(K)]
                    enc = iceberg.encode_bus(bare, edge_mode=mode, **kw)
                    lay = iceberg.layout_of(enc)
                    pos = lay.positions
                    C = lay.k_code // 2 + 1
                    n_circ += 1
                    if len(set(pos)) != len(pos) or len(pos) != enc.num_qubits or enc.num_qubits != lay.k_code + 2 + C:
                        bad_pos.append((k, mode))
                    if not all(0 <= r <= 2 and 0 <= c < C for r, c in pos):
                        bad_pos.append((k, mode, "range"))
                    # rows: code qubits in rows 0/2, bus in row 1; t,b in last column
                    if not (all(pos[j][0] in (0, 2) for j in lay.code_qubits) and all(pos[j][0] == 1 for j in range(lay.k_code + 2, enc.num_qubits))
                            and pos[lay.top] == (0, C - 1) and pos[lay.bottom] == (2, C - 1)):
                        bad_pos.append((k, mode, "rows"))
                    lab = block_labels(enc)
                    for i, inst in enumerate(enc.data):
                        if inst.operation.num_qubits != 2 or inst.operation.name == "barrier":
                            continue
                        a, b = (enc.find_bit(x).index for x in inst.qubits)
                        n_2q += 1
                        if abs(pos[a][0] - pos[b][0]) + abs(pos[a][1] - pos[b][1]) != 1:
                            viol_nn.append((k, mode, i, a, b))
                        hub = {a, b} & {lay.top, lay.bottom}
                        data = {a, b} & set(range(lay.k_code))
                        if hub and data:
                            hub_blocks[lab[i]] = hub_blocks.get(lab[i], 0) + 1
                            if lab[i] not in ("init", "syndrome"):
                                viol_hub.append((k, mode, i, lab[i]))
print(f"{n_circ} bus circuits (k=1..12), {n_2q} two-qubit gates")
print("position problems:", bad_pos[:5])
print("non-nearest-neighbour gates:", len(viol_nn), viol_nn[:3])
print("data-(t|b) gates by block:", hub_blocks, " outside init/syndrome:", len(viol_hub))
# grid_coupling agrees with from_grid numbering?  (positions -> index r*C+c is NOT the qubit index)
from qiskit.transpiler import CouplingMap
enc = iceberg.encode_bus(qsel.qaoa_circuit(qsel.random_qubo(4, edge_fraction=0.5, seed=1), Angles((0.3,), (0.4,))))
lay = iceberg.layout_of(enc)
C = 3
grid = set(map(tuple, CouplingMap.from_grid(3, C).get_edges()))
gc = iceberg.grid_coupling(lay)
print("qubit index == row*C+col for all qubits:", all(i == r * C + c for i, (r, c) in enumerate(lay.positions)), "; positions:", lay.positions)
