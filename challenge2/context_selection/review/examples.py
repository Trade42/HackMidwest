import iceberg, qsel
from vlib import *
from faultlib import *
q = qsel.random_qubo(4, edge_fraction=0.5, seed=1)
ang = qsel.tune_angles(q, p=1).angles
bare = qsel.qaoa_circuit(q, ang)
print("instance: qsel.random_qubo(4, edge_fraction=0.5, seed=1), tuned p=1 angles", ang)
for title, enc in (("encode_direct default", iceberg.encode_direct(bare)),
                   ("encode_direct init_flag=False", iceberg.encode_direct(bare, init_flag=False)),
                   ("encode_direct sx_flag=False", iceberg.encode_direct(bare, sx_flag=False)),
                   ("encode_bus default", iceberg.encode_bus(bare)),
                   ("encode_bus precopy=False", iceberg.encode_bus(bare, precopy=False)),
                   ("encode_bus init_flag=False", iceberg.encode_bus(bare, init_flag=False))):
    an = Analysis(enc); faults = an.enumerate()
    harmful = [f for f in faults if f.status == "harmful"]
    print("==", title, "| harmful by block:", {b: sum(1 for f in harmful if f.block == b) for b in dict.fromkeys(f.block for f in faults)})
    shown = 0
    for f in harmful:
        big_shift = f.pure_shift >= 0 and popc(f.pure_shift) >= 3
        if f.block in ("init", "syndrome") or big_shift:
            print("   ", describe(enc, f)); shown += 1
        if shown >= 5: break
    if "default" in title:
        tb = [f for f in harmful if an.lay.bottom in f.qubits or an.lay.top in f.qubits]
        print("    harmful faults on gates touching t or b:", len(tb), "max logical weight", max((f.lweight for f in tb), default=0), "| max weight overall", max(f.lweight for f in harmful))
