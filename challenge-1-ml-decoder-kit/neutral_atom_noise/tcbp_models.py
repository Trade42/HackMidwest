"""Phase 2 of the TCBP check: three more pictures of the noise, each with a generic, a matched and a head arm.

Exploration, after the main test results, on the branch infleqtion-noise. Declared in tcbp_check2_declared.md
before any number. Every task decodes the kit's X-memory H and L; only the shots and the prior vectors differ.

  wrong-p    the kit's own shots at p = 0.005 (data/p0.005/validation_X.npz, its first 4,000 shots), decoded as if
             the noise were p = 0.004: generic = the kit's priors at 0.004, matched = the kit's priors at 0.005 (the
             truth), head = the network told p = 0.004 (its p input and the mixing base).
  si1000     SI1000 (Gidney, Newman, Fowler, Broughton, Quantum 5, 605 (2021), arXiv:2108.10457, Table 2) at
             p = 0.004 on the kit's circuit: two-qubit depolarizing p after each CX, readout flip 5p, reset flip 2p,
             idle p/10. Its one-qubit-gate noise (p/10) has nothing to act on (the kit's circuit has no one-qubit
             gates), and its resonator idle (2p) is left out: the rewrite gives every idle slot of the kit one rate.
  biased     Z-biased Pauli noise, bias eta = 100, strength p = 0.003: after each CX on each of its two qubits and
             on each idle qubit, one-qubit Pauli noise with total p, pZ = p eta / (eta + 1), pX = pY = p / (2 (eta + 1));
             readout and reset flips p.

The generic arm of si1000 and biased is the kit's uniform model at the model's two-qubit error (the chance that a CX
leaves any Pauli error on its pair; the branch's `generic-2q` rule). Nothing is fitted.

  python -m neutral_atom_noise.tcbp_models prepare --validation <validation_X.npz of p0.005> --out <transfer2 folder>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import stim

from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit, sample
from neutral_atom_noise.model import Noise, with_noise
from neutral_atom_noise.priors import FLOOR, effects, kit_problem, uniform_priors
from neutral_atom_noise.tcbp_check import cap, file_sha256, matrix_hash

BASIS = "X"
SHOTS = 4000
SEEDS = {"si1000": 7100, "biased": 7200}           # fresh seeds, used nowhere else on this branch
WRONG_P, TRUE_P = 0.004, 0.005
SI1000_P = 0.004
BIASED_P, BIASED_ETA = 0.003, 100


def si1000_noise(p):
    """SI1000 on the kit's circuit (see the module's text for what is left out)."""
    return Noise(cx_depolarize2=p, idle_depolarize=p / 10, measure_flip=5 * p, reset_flip=2 * p)


def biased_channel(p, eta):
    """(pX, pY, pZ) of one-qubit Pauli noise with total p and bias eta = pZ / (pX + pY), pX = pY."""
    return p / (2 * (eta + 1)), p / (2 * (eta + 1)), p * eta / (eta + 1)


def biased_circuit(circuit, p, eta):
    """The kit's circuit with each DEPOLARIZE2 and DEPOLARIZE1 replaced by biased one-qubit Pauli noise on the same
    qubits, and each readout or reset flip set to p."""
    out = stim.Circuit()
    channel = biased_channel(p, eta)
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(instruction.repeat_count, biased_circuit(instruction.body_copy(), p, eta)))
        elif instruction.name in ("DEPOLARIZE2", "DEPOLARIZE1"):
            out.append("PAULI_CHANNEL_1", instruction.targets_copy(), channel)
        elif instruction.name in ("X_ERROR", "Z_ERROR"):
            out.append(instruction.name, instruction.targets_copy(), p)
        elif stim.gate_data(instruction.name).is_noisy_gate and instruction.gate_args_copy():
            raise ValueError(f"a noise instruction the kit's circuits do not use: {instruction.name}")
        else:
            out.append(instruction)
    return out


def biased_two_qubit_error(p):
    return 1 - (1 - p) ** 2


def matched_priors_of(circuit, basis=BASIS, floor=FLOOR):
    """The error model of `circuit` on the kit's columns (as priors.matched_priors, for any circuit)."""
    H, L, _ = kit_problem(basis)
    column = {effect: k for k, effect in enumerate(effects(H, L))}
    H_model, L_model, priors_model = dem_matrices(circuit, basis)
    priors, fired = np.full(H.shape[1], floor), np.zeros(H.shape[1], bool)
    for effect, probability in zip(effects(H_model, L_model), priors_model, strict=True):
        if effect not in column:
            raise ValueError(f"the model fires a mechanism that the kit's matrix does not have: {effect}")
        priors[column[effect]], fired[column[effect]] = probability, True
    return priors, fired


def write_task(out, name, detectors, observables, seed, p_generic, generic, matched, fired, what):
    """The kit-format shot file (its p is the generic p: the network's p input and the mixing base), the two
    prior vectors and transfer.json, in out/name/."""
    folder = Path(out) / name
    folder.mkdir(parents=True, exist_ok=True)
    circuit = load_circuit(GROSS_CIRCUITS[BASIS], p_generic)
    shots = folder / f"{name}_X_kitformat_p{p_generic:g}.npz"
    np.savez_compressed(shots, p=p_generic, seed=seed, basis=BASIS, detectors=np.asarray(detectors, np.uint8),
                        observables=np.asarray(observables, np.uint8), basis_detectors=detector_ids(circuit, BASIS))
    H, L, _ = kit_problem(BASIS)
    H_p, L_p, rebuilt = kit_problem(BASIS, p_generic)
    if matrix_hash(H, L) != matrix_hash(H_p, L_p) or not np.array_equal(rebuilt, generic):
        raise SystemExit(f"{name}: the kit's matrices or generic priors do not line up")
    generic, generic_capped = cap(generic)
    matched, matched_capped = cap(matched)
    np.save(folder / "priors_generic.npy", generic)
    np.save(folder / "priors_matched.npy", matched)
    record = {"task": name, "what": what, "shots": int(len(detectors)), "seed": int(seed), "basis": BASIS,
              "converted": shots.name, "converted_sha256": file_sha256(shots), "p": p_generic,
              "columns": int(H.shape[1]), "rows": int(H.shape[0]), "h_l_sha256": matrix_hash(H, L),
              "generic": {"what": f"kit uniform model at p = {p_generic:g}", "max": float(generic.max()),
                          "min": float(generic.min()), "sum": float(generic.sum()), "capped_columns": generic_capped,
                          "sha256": file_sha256(folder / "priors_generic.npy")},
              "matched": {"max": float(matched.max()), "min": float(matched.min()), "sum": float(matched.sum()),
                          "columns_fired": int(fired.sum()), "capped_columns": matched_capped,
                          "sha256": file_sha256(folder / "priors_matched.npy")},
              "zero_syndrome_fraction": float((np.asarray(detectors)[:, detector_ids(circuit, BASIS)].sum(1) == 0).mean())}
    with open(folder / "transfer.json", "w") as file:
        json.dump(record, file, indent=1)
    print(json.dumps({key: record[key] for key in ("task", "shots", "seed", "p", "converted_sha256")}
                     | {"generic_sum": record["generic"]["sum"], "matched_sum": record["matched"]["sum"],
                        "matched_fired": record["matched"]["columns_fired"]}))
    return record


def prepare_command(args):
    out = Path(args.out)
    tasks = []
    # wrong-p: the kit's validation shots at 0.005, labelled with the assumed p 0.004
    with np.load(args.validation, allow_pickle=False) as file:
        if float(file["p"]) != TRUE_P or str(file["basis"]) != BASIS:
            raise SystemExit(f"{args.validation} is not the kit's X file at p = {TRUE_P}")
        detectors, observables, seed = file["detectors"][:SHOTS], file["observables"][:SHOTS], int(file["seed"])
    matched = uniform_priors(BASIS, TRUE_P)
    tasks.append(write_task(out, "wrong-p", detectors, observables, seed, WRONG_P, uniform_priors(BASIS, WRONG_P),
                            matched, np.ones(len(matched), bool),
                            {"source": str(args.validation), "source_sha256": file_sha256(args.validation),
                             "first_shots": SHOTS, "true_p": TRUE_P, "assumed_p": WRONG_P}))
    # si1000
    noise = si1000_noise(SI1000_P)
    circuit = with_noise(stim.Circuit.from_file(GROSS_CIRCUITS[BASIS]), noise)
    detectors, observables = sample(circuit, SHOTS, SEEDS["si1000"])
    matched, fired = matched_priors_of(circuit)
    p_generic = round(noise.two_qubit_error(), 7)
    tasks.append(write_task(out, "si1000", detectors, observables, SEEDS["si1000"], p_generic,
                            uniform_priors(BASIS, p_generic), matched, fired,
                            {"model": "SI1000, Gidney et al., Quantum 5, 605 (2021), Table 2", "p": SI1000_P,
                             "noise": noise.as_dict(), "left_out": "one-qubit-gate noise (no such gates), resonator idle 2p",
                             "stim": stim.__version__}))
    # biased
    circuit = biased_circuit(stim.Circuit.from_file(GROSS_CIRCUITS[BASIS]), BIASED_P, BIASED_ETA)
    detectors, observables = sample(circuit, SHOTS, SEEDS["biased"])
    matched, fired = matched_priors_of(circuit)
    p_generic = round(biased_two_qubit_error(BIASED_P), 7)
    tasks.append(write_task(out, "biased", detectors, observables, SEEDS["biased"], p_generic,
                            uniform_priors(BASIS, p_generic), matched, fired,
                            {"model": "Z-biased one-qubit Pauli noise", "p": BIASED_P, "eta": BIASED_ETA,
                             "pX_pY_pZ": list(biased_channel(BIASED_P, BIASED_ETA)), "stim": stim.__version__}))
    plan = {"tasks": [{"name": t["task"], "data": f"{t['task']}/{t['converted']}", "generic": f"{t['task']}/priors_generic.npy",
                       "matched": f"{t['task']}/priors_matched.npy", "transfer": f"{t['task']}/transfer.json"} for t in tasks]}
    with open(out / "plan.json", "w") as file:
        json.dump(plan, file, indent=1)
    print(f"wrote {out / 'plan.json'}")


def prepare_si1000_command(args):
    """Extra SI1000 strengths (phase 2b): one task per p, each with its own seed, and a plan of them."""
    out = Path(args.out)
    tasks = []
    for p, seed in zip(args.p, args.seed, strict=True):
        noise = si1000_noise(p)
        circuit = with_noise(stim.Circuit.from_file(GROSS_CIRCUITS[BASIS]), noise)
        detectors, observables = sample(circuit, args.shots, seed)
        matched, fired = matched_priors_of(circuit)
        p_generic = round(noise.two_qubit_error(), 7)
        tasks.append(write_task(out, f"si1000-p{p:g}", detectors, observables, seed, p_generic,
                                uniform_priors(BASIS, p_generic), matched, fired,
                                {"model": "SI1000, Gidney et al., Quantum 5, 605 (2021), Table 2", "p": p,
                                 "noise": noise.as_dict(),
                                 "left_out": "one-qubit-gate noise (no such gates), resonator idle 2p",
                                 "stim": stim.__version__}))
    plan = {"tasks": [{"name": t["task"], "data": f"{t['task']}/{t['converted']}", "generic": f"{t['task']}/priors_generic.npy",
                       "matched": f"{t['task']}/priors_matched.npy", "transfer": f"{t['task']}/transfer.json"} for t in tasks]}
    with open(out / "plan.json", "w") as file:
        json.dump(plan, file, indent=1)
    print(f"wrote {out / 'plan.json'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--validation", required=True)
    prepare.add_argument("--out", required=True)
    extra = sub.add_parser("prepare-si1000")
    extra.add_argument("--p", type=float, nargs="+", required=True)
    extra.add_argument("--seed", type=int, nargs="+", required=True)
    extra.add_argument("--shots", type=int, default=3000)
    extra.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"prepare": prepare_command, "prepare-si1000": prepare_si1000_command}[args.command](args)


if __name__ == "__main__":
    main()
