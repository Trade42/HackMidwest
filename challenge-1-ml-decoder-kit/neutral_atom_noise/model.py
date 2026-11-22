"""A noise model with the shape of neutral-atom hardware, built from Infleqtion's published figures.

This is NOT Infleqtion's noise model. It is our Pauli-plus-loss model, built from the figures they published,
with our own choices wherever nothing is published. PUBLISHED lists the figures with their sources, MODEL the
rates we derive from them, and `neutral_atom(factor)` scales all of them by one factor (factor 1 = the
published figures).

The circuit is the kit's, unchanged: the same gates, schedule, detectors and observables. `with_noise` replaces
only the noise channels, by kind of operation:

  after a CX          the kit: two-qubit depolarizing.  The model: a phase error on each atom of the CZ, plus a
                      small one-atom depolarizing error (scattering).
  on an idle qubit    the kit: one-qubit depolarizing.  The model: a phase error.
  before a readout    a flip of the outcome, in both.
  after a reset       a flip of the prepared state, in both.
  one-qubit gates     the kit's circuits have none, so no channel exists to replace (see README.md).

CZ to CX. Their two-qubit gate is CZ; the kit's is CX. A CX is a CZ between two Hadamards on the target atom.
A phase error (Z) on each atom after the CZ therefore becomes, after the closing Hadamard, a Z on the CX
control and an X on the CX target (H Z H = X). So a check atom only ever gets the error that flips its own
readout, and a data atom gets Z errors from its Z-check gates and X errors from its X-check gates.

Only X_ERROR, Z_ERROR, DEPOLARIZE1 and DEPOLARIZE2 are emitted. Each is an exact, independent channel in Stim's
detector error model, and `loss/` (Clifft) reads the same four.
"""
import math
from dataclasses import asdict, dataclass

import stim

from circuit import GROSS_CIRCUITS

HARDWARE_PAPER = "arXiv:2408.08288v3 (Radnaev et al., PRX Quantum 6, 030334)"
MODEL_PAPER = "arXiv:2509.13247v2 (Rines et al.)"
MODEL_TABLE = MODEL_PAPER + ', Appendix "System error model", table "Simulation Noise Model"'


@dataclass(frozen=True)
class Figure:
    """One published number. kind: M measured on hardware, F a parameter of the authors' fitted simulator."""
    value: float
    what: str
    kind: str
    source: str
    checked: bool    # True: we read it in the paper's own LaTeX source (arxiv.org/e-print), 2026-11-22


# The transition probabilities per CZ of the authors' five-level atom, in units of 1e-6. Columns are the state
# before the gate, rows the state after it, in the order 0, 1, 0_leaked, 1_leaked, lost.
CZ_TRANSITIONS = [[17.4, 185.0, 4.9, 165.4, 0],
                  [18.5, 197.5, 4.6, 177.7, 0],
                  [31.1, 420.3, 45.8, 1210.0, 0],
                  [42.1, 590.2, 52.9, 1901.0, 0],
                  [0, 5000, 0, 0, 0]]

PUBLISHED = {
    "cz_phase": Figure(0.0073, "phase error on each atom per CZ: 'Single qubit phase error with 0.73% probability'",
                       "F", MODEL_TABLE + ", row CZ", True),
    "cz_transitions": Figure(float("nan"), "CZ_TRANSITIONS above: 'Spontaneous transitions with probabilities ...'",
                             "F", MODEL_TABLE + ", row CZ (the 4 x 4 block is Eq. 3 of " + HARDWARE_PAPER + ")", True),
    "cz_loss_from_1": Figure(0.005, "loss per CZ of an atom in state 1 (entry 5000e-6 of the matrix; none from state 0)",
                             "F", MODEL_TABLE + ", row CZ", True),
    "cz_postselected_fidelity": Figure(0.987, "CZ fidelity after discarding shots with a lost atom (array median)",
                                       "M", MODEL_PAPER + ', Appendix "Gate Fidelities"; estimated from calibration circuits', True),
    "readout_0_as_1": Figure(0.002, "readout: state 0 read as 1 (epsilon_0)", "F", MODEL_TABLE + ", row Measure", True),
    "readout_1_as_0": Figure(0.023, "readout: state 1 read as 0 (epsilon_1)", "F", MODEL_TABLE + ", row Measure", True),
    "preparation_fidelity": Figure(0.979, "state preparation fidelity (p = 0.007, q = 0.014)",
                                   "F", MODEL_TABLE + ", row State Prep", True),
    "local_gate_fidelity": Figure(0.9980, "local one-qubit gate (RZ) fidelity, 99.80(5)%",
                                  "M", MODEL_PAPER + ', Appendix "Gate Fidelities"', True),
    "global_gate_fidelity": Figure(0.9996, "global one-qubit gate (GR) fidelity, 99.96(1)%",
                                   "M", MODEL_PAPER + ', Appendix "Gate Fidelities"', True),
    "t2_decoupled_s": Figure(2.8, "coherence time T2 with XY8 decoupling: 'our best estimate is 2.8 s, with a 97% "
                             "confidence interval lower bound of 128 ms'", "M",
                             HARDWARE_PAPER + ', Sec. "Quantum processor implementation and characterization"', True),
    "t2_star_s": Figure(0.0127, "coherence time T2* without decoupling, 12.7(5) ms", "M",
                        HARDWARE_PAPER + ', Sec. "Quantum processor implementation and characterization"', True),
    "readout_time_s": Figure(0.006, "camera exposure of one readout, 6 ms", "M",
                             HARDWARE_PAPER + ', Sec. "Non-destructive State Selective Readout"', True),
    "cz_loss_measured_pair": Figure(0.0038, "atom loss per CZ of one benchmarked pair, '~0.38% atom loss error' "
                                    "(CZ fidelity 99.35(4)%, 99.73(3)% post-selected: loss is 0.38 / 0.65 = 58% of the error)",
                                    "M", HARDWARE_PAPER + ', Sec. "Quantum processor implementation and characterization"', True),
}


@dataclass(frozen=True)
class Rate:
    """One rate of our model at factor 1. status: 'published' (used as printed), 'derived' (our arithmetic on
    published figures) or 'our choice' (nothing is published for this operation)."""
    value: float
    status: str
    how: str


def _scatter_inside_qubit():
    """Chance per atom per CZ of a spontaneous transition that ends in state 0 or 1, averaged over the two states."""
    from_0 = CZ_TRANSITIONS[0][0] + CZ_TRANSITIONS[1][0]
    from_1 = CZ_TRANSITIONS[0][1] + CZ_TRANSITIONS[1][1]
    return (from_0 + from_1) / 2 * 1e-6


def _leak_per_atom():
    """Chance per atom per CZ of leaving the qubit for another (unlost) level, averaged over the two states."""
    from_0 = CZ_TRANSITIONS[2][0] + CZ_TRANSITIONS[3][0]
    from_1 = CZ_TRANSITIONS[2][1] + CZ_TRANSITIONS[3][1]
    return (from_0 + from_1) / 2 * 1e-6


IDLE_SECONDS = PUBLISHED["readout_time_s"].value    # our choice: every idle slot of the kit lasts one readout

MODEL = {
    "cx_phase": Rate(PUBLISHED["cz_phase"].value, "published",
                     "the CZ phase error per atom; on the kit's CX it is Z on the control and X on the target"),
    "cx_scatter": Rate(0.75 * _scatter_inside_qubit(), "derived",
                       "one-atom depolarizing after a CX: 3/4 of the chance (2.09e-4, state-averaged from the "
                       "transition matrix) that a scattering event leaves the atom in a random qubit state"),
    "idle_phase": Rate((1 - math.exp(-IDLE_SECONDS / PUBLISHED["t2_decoupled_s"].value)) / 2, "our choice",
                       "Z on an idle atom: (1 - exp(-t / T2)) / 2 with T2 = 2.8 s (decoupled; the paper's lower "
                       "bound of 128 ms would give 2.3e-2) and t = 6 ms. No idle error per step is published. "
                       "Every idle slot of the kit sits beside a readout or reset of check atoms; we charge each "
                       "one readout time"),
    "measure_flip": Rate((PUBLISHED["readout_0_as_1"].value + PUBLISHED["readout_1_as_0"].value) / 2, "derived",
                         "flip of a check readout: the mean of epsilon_0 and epsilon_1 (a Pauli model cannot "
                         "make the flip depend on the state). Published for the end-of-circuit readout; using it "
                         "for a mid-circuit readout is our choice: their machine has none"),
    "reset_flip": Rate(1 - PUBLISHED["preparation_fidelity"].value, "our choice",
                       "flip of a freshly reset check atom: 1 - the start-of-shot preparation fidelity. Their "
                       "machine has no mid-circuit reset"),
    "loss": Rate(PUBLISHED["cz_loss_from_1"].value / 2, "derived",
                 "loss per atom per CX: half the loss of an atom in state 1 (an atom is in state 1 half the "
                 "time: our assumption). Used only by the loss runs (question 2)"),
}

# Published, but NOT in the model. README.md says so.
LEFT_OUT = {
    "leakage": f"{_leak_per_atom():.2e} per atom per CZ (state-averaged from the transition matrix) into levels that "
               "read out as 0 or 1 and ignore later gates. A Pauli model cannot hold it",
    "one-qubit gates": "local 0.2% and global 0.04% error per gate. The kit's circuits contain no one-qubit gate, "
                       "so the Hadamards that turn a CZ into a CX are not charged",
    "readout loss": "0.9(3)% of atoms are lost per readout (" + HARDWARE_PAPER + "). Our loss runs lose atoms at CX only",
    "state-dependent readout": "epsilon_1 is 11 times epsilon_0; we use their mean",
    "atom moves": "the Gross code needs gates between distant atoms; a move costs 2.6% to 4.1% phase error in their "
                  "table. Not charged: the kit's CX is one gate",
}


@dataclass(frozen=True)
class Noise:
    """Error probabilities of one circuit, by kind of operation. Zero switches a channel off."""
    cx_depolarize2: float = 0.0    # two-qubit depolarizing after a CX (the kit's channel)
    cx_phase: float = 0.0          # after a CX: Z on the control and X on the target, each with this probability
    cx_scatter: float = 0.0        # after a CX: one-qubit depolarizing on each of the two atoms
    idle_depolarize: float = 0.0   # one-qubit depolarizing on an idle qubit (the kit's channel)
    idle_phase: float = 0.0        # Z on an idle qubit
    measure_flip: float = 0.0      # flip before a check readout
    reset_flip: float = 0.0        # flip after a reset
    loss: float = 0.0              # atom loss per atom per CX. Not a Stim channel: only the loss runs use it

    def two_qubit_error(self):
        """Chance that a CX leaves any Pauli error on its pair."""
        return 1 - (1 - self.cx_depolarize2) * (1 - self.cx_phase) ** 2 * (1 - self.cx_scatter) ** 2

    def as_dict(self):
        return asdict(self)


def kit_uniform(p):
    """The kit's own model: one strength p on every operation."""
    return Noise(cx_depolarize2=p, idle_depolarize=p, measure_flip=p, reset_flip=p)


def neutral_atom(factor=1.0):
    """Our model with every rate of MODEL multiplied by `factor`. Factor 1 is the published figures."""
    return Noise(**{name: factor * rate.value for name, rate in MODEL.items()})


# ---------- the rewrite ----------

RESETS, READOUTS, FLIPS = ("R", "RX"), ("M", "MX"), ("X_ERROR", "Z_ERROR")
CHANNELS = ("DEPOLARIZE1", "DEPOLARIZE2", *FLIPS)    # the noise channels of the kit's circuits, and ours


def with_noise(circuit, noise):
    """Copy a kit circuit and replace each noise channel by the one `noise` gives for that kind of operation.

    The kind is read from the neighbouring instruction: DEPOLARIZE2 follows its CX, a flip follows its reset or
    precedes its readout, and DEPOLARIZE1 is idle noise. Everything that is not noise is copied unchanged.
    """
    out, instructions = stim.Circuit(), list(circuit)
    for k, instruction in enumerate(instructions):
        if isinstance(instruction, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(instruction.repeat_count, with_noise(instruction.body_copy(), noise)))
        elif instruction.name in CHANNELS:
            before = instructions[k - 1] if k > 0 else None
            after = instructions[k + 1] if k + 1 < len(instructions) else None
            _append_channel(out, instruction, before, after, noise)
        elif stim.gate_data(instruction.name).is_noisy_gate and instruction.gate_args_copy():
            raise ValueError(f"a noise instruction the kit's circuits do not use: {instruction.name}")
        else:
            out.append(instruction)
    return out


def _append_channel(out, channel, before, after, noise):
    """Append our channel for the operation that the kit's `channel` belongs to."""
    targets = channel.targets_copy()
    if channel.name == "DEPOLARIZE2":
        if not _is(before, ("CX",), targets):
            raise ValueError("a DEPOLARIZE2 that does not follow its CX")
        _append(out, "DEPOLARIZE2", targets, noise.cx_depolarize2)
        _append(out, "Z_ERROR", targets[0::2], noise.cx_phase)      # the controls
        _append(out, "X_ERROR", targets[1::2], noise.cx_phase)      # the targets
        _append(out, "DEPOLARIZE1", targets, noise.cx_scatter)
    elif channel.name == "DEPOLARIZE1":
        _append(out, "DEPOLARIZE1", targets, noise.idle_depolarize)
        _append(out, "Z_ERROR", targets, noise.idle_phase)
    elif channel.name in FLIPS and _is(before, RESETS, targets):
        _append(out, channel.name, targets, noise.reset_flip)
    elif channel.name in FLIPS and _is(after, READOUTS, targets):
        _append(out, channel.name, targets, noise.measure_flip)
    else:
        raise ValueError(f"cannot tell which operation this noise belongs to: {channel.name}")


def _is(instruction, names, targets):
    """True if the instruction is one of `names` on exactly these targets."""
    return (instruction is not None and not isinstance(instruction, stim.CircuitRepeatBlock)
            and instruction.name in names and instruction.targets_copy() == targets)


def _append(out, name, targets, probability):
    if probability > 0:
        out.append(name, targets, probability)


def model_circuit(basis, noise):
    """The kit's Gross-code memory circuit of one basis ("X" or "Z") with `noise` in place of the kit's noise."""
    return with_noise(stim.Circuit.from_file(GROSS_CIRCUITS[basis]), noise)


def rates_table():
    """The model's rates at factor 1 as markdown rows."""
    lines = ["| Rate | Value at factor 1 | Status | How |", "|---|---|---|---|"]
    lines += [f"| `{name}` | {rate.value:.3e} | {rate.status} | {rate.how} |" for name, rate in MODEL.items()]
    return "\n".join(lines)


def published_table():
    """The published figures as markdown rows."""
    lines = ["| Figure | Value | Kind | Source | Checked by us |", "|---|---|---|---|---|"]
    for name, f in PUBLISHED.items():
        value = "matrix" if math.isnan(f.value) else f"{f.value:g}"
        lines.append(f"| `{name}`: {f.what} | {value} | {f.kind} | {f.source} | {'yes' if f.checked else 'no'} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(published_table(), rates_table(), sep="\n\n")
    print("\nleft out:", *[f"- {k}: {v}" for k, v in LEFT_OUT.items()], sep="\n")
    print(f"\ntwo-qubit error at factor 1: {neutral_atom(1).two_qubit_error():.4e}")
