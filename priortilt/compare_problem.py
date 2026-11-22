"""Is PriorTilt's bundled BB144 circuit the same decoding problem as the kit's?

  python -I compare_problem.py <kit folder> <PriorTiltDecoder folder>

For each basis and p, compares the kit's problem, dem_matrices(load_circuit(GROSS_CIRCUITS[b], p), b), with
PriorTilt's circuit StimCircuit/BB[[144,12,12]],memory_<b>,error_rate=<p>,syndrome_rounds=12.stim converted two
ways: by PriorTilt's own dem_to_matrices (what PriorTilt(dem, code=..., p=...) would decode), and by the kit's
dem_matrices restricted to the same check type. Detector orders may differ, so the comparison is on
permutation-invariant summaries: shapes, sorted priors, column-weight and row-weight histograms, and the
multiset of (column weight, observable weight, prior) per mechanism.
"""
import sys
from collections import Counter

import numpy as np
import stim

KIT, PT = sys.argv[1], sys.argv[2]
sys.path.insert(0, KIT)
sys.path.insert(1, PT)
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit, noise_strengths  # noqa: E402
from priortilt.dem import dem_to_matrices  # noqa: E402


def summary(H, L, pri):
    H = H.tocsc()
    L = L.tocsc()
    colw = np.diff(H.indptr)
    obsw = np.diff(L.indptr)
    roww = np.diff(H.tocsr().indptr)
    mech = Counter(zip(colw.tolist(), obsw.tolist(), np.round(pri, 12).tolist()))
    return dict(shape=H.shape, L_shape=L.shape, priors=np.sort(pri), colw=Counter(colw.tolist()),
                roww=Counter(roww.tolist()), mech=mech, nnz=H.nnz)


def duplicate_detector_columns(H):
    """Columns of H whose detector set repeats (they differ only in observables)."""
    H = H.tocsc()
    keys = Counter(tuple(H.indices[H.indptr[c]:H.indptr[c + 1]].tolist()) for c in range(H.shape[1]))
    return sum(v - 1 for v in keys.values() if v > 1)


def compare(name, a, b):
    same_shape = a["shape"] == b["shape"] and a["L_shape"] == b["L_shape"]
    out = [f"  {name}: H {b['shape']} vs kit {a['shape']}, L {b['L_shape']} vs {a['L_shape']}, nnz {b['nnz']} vs {a['nnz']}"]
    if a["priors"].shape == b["priors"].shape:
        out.append(f"    max |sorted prior difference| = {np.abs(a['priors'] - b['priors']).max():.3e}")
    else:
        out.append("    prior counts differ")
    out.append(f"    column-weight histogram equal: {a['colw'] == b['colw']}; row-weight histogram equal: {a['roww'] == b['roww']}")
    out.append(f"    (column weight, observable weight, prior) multiset equal: {a['mech'] == b['mech']}")
    return same_shape, "\n".join(out)


for basis in "XZ":
    for p in (0.003, 0.004, 0.005):
        kc = load_circuit(GROSS_CIRCUITS[basis], p)
        H, L, pri = dem_matrices(kc, basis)
        kit = summary(H, L, pri)
        path = f"{PT}/StimCircuit/BB[[144,12,12]],memory_{basis},error_rate={p},syndrome_rounds=12.stim"
        pc = stim.Circuit.from_file(path)
        print(f"basis {basis}, p {p}")
        print(f"  kit circuit: {kc.num_qubits} qubits, {kc.num_measurements} measurements, {kc.num_detectors} detectors, "
              f"{kc.num_observables} observables; kit H {H.shape}, mechanisms sharing a detector set: {duplicate_detector_columns(H)}")
        print(f"  PriorTilt circuit: {pc.num_qubits} qubits, {pc.num_measurements} measurements, {pc.num_detectors} detectors, "
              f"{pc.num_observables} observables; noise strengths {sorted(noise_strengths(pc))}")
        m = dem_to_matrices(pc.detector_error_model())
        print(compare("PriorTilt dem_to_matrices", kit, summary(m.check_matrix, m.observables_matrix, m.priors))[1])
        try:
            H2, L2, pri2 = dem_matrices(pc, basis)
            print(compare("kit dem_matrices on PriorTilt circuit", kit, summary(H2, L2, pri2))[1])
        except Exception as exc:  # coordinates may not follow the kit's convention
            print(f"  kit dem_matrices on PriorTilt circuit: not applicable ({exc.__class__.__name__}: {exc})")
