"""The registry: which decoders the report shows, in which order, colour and words, and where
each one's result file is expected under report/data/.

Tables and figures both iterate this list, so a decoder has the same name and the same colour
everywhere, and a row whose file has not landed is printed as "not measured".
"""
from dataclasses import dataclass

P_VALUES = ("0.003", "0.004", "0.005")

# The team's saved test shots (data/p<p>/test_X.npz, test_Z.npz; Stim seeds 301 and 302, 10,000
# shots per basis). Every row of a test table must name these files, or it is refused.
TEST_SHA256 = {
    "0.003": {"X": "f7bc181399a0a8fd1324950c98a8419c0082d7c34940c03dfb34c9e2cefc1ff3",
              "Z": "f5ac5b336f4155870fd8a6792ad61e21d5d76f04e05a9c09419879b6512e7f49"},
    "0.004": {"X": "53a7cee301600d36f3dad806a88f59ad6d84b681044b2548268c2ea9b5689e94",
              "Z": "b61c1325de8852385d19ee5ce4eb41ea939b668f83fb3e994ce0f590218b3d56"},
    "0.005": {"X": "1b13521cddb2bf5e5d4c7f1e85a8ecaaa3c72a0b99754a3f76b4a5a4a912fc8b",
              "Z": "b86ae320d13fa555d8358fe399156d28d4e52b01fabf8eafa1b30e60bdcd0cef"},
}


@dataclass(frozen=True)
class Decoder:
    key: str
    label: str      # the name in tables
    short: str      # the name on a figure
    whose: str      # who built it: the gain of this row is theirs
    learned: str    # what in it is learned
    colour: str
    marker: str
    ours: bool = False
    figures: str = "all"   # "all"; "time": only in the time figures (its answers equal another row's); "none": tables only


# Colours: the eight-hue categorical order of the dataviz palette, validated for adjacent
# series; one decoder keeps its hue in every figure. Every series also has its own marker
# shape and a direct label, so colour never carries identity alone.
DECODERS = {decoder.key: decoder for decoder in [
    Decoder("bposd", "BP-OSD (kit baseline)", "BP-OSD", "the kit", "nothing", "#eda100", "s"),
    Decoder("relay-bp", "relay-BP (kit baseline)", "relay-BP", "the kit", "nothing", "#2a78d6", "o"),
    Decoder("router", "relay-BP with a learned router", "router (teammates)", "teammates",
            "a logistic model decides when relay-BP's first answer is enough", "#e87ba4", "D"),
    Decoder("priortilt", "PriorTilt, one run", "PriorTilt (IonQ)", "IonQ (arXiv:2611.13373)",
            "14 parameters, fitted by its authors", "#6250d6", "v"),
    Decoder("hierarchy-weight", "our hierarchy, lowest-weight tie-break", "ours, lowest weight", "ours",
            "nothing of ours (PriorTilt's parameters are its authors')", "#eb6834", "^", ours=True, figures="none"),
    Decoder("hierarchy-converged", "our hierarchy, classical tie-break (lowest weight among the converged candidates)", "ours, no network", "ours",
            "nothing of ours (PriorTilt's parameters are its authors')", "#eb6834", "^", ours=True),
    Decoder("hierarchy", "our hierarchy, neural tie-break", "ours", "ours",
            "a neural network picks among the candidates where the fast decodes disagree", "#eb6834", "^", ours=True),
    Decoder("hierarchy-parallel", "our hierarchy, neural tie-break, the two fast decodes side by side on two cores", "ours, 2 cores", "ours",
            "the same decoder as the row above, with the same answers on every shot (checked); only the wall time differs", "#e34948", ">", ours=True, figures="time"),
    Decoder("ensemble", "our ensemble (combined offline)", "ours, ensemble", "ours",
            "only the rules that use the network have a learned part: see the rule named in the row", "#008300", "X", ours=True),
    Decoder("tcbp", "TCBP-JOBS (NVIDIA, GPU)", "TCBP-JOBS (GPU)", "NVIDIA; run by teammates", "a model it fits at start-up, per memory basis", "#1baf7a", "P"),
]}

REFERENCE = "relay-bp"   # the paired counts are against this row


@dataclass(frozen=True)
class TestRow:
    decoder: str    # a key of DECODERS
    folder: str     # under report/data/test/, with {p}
    kind: str       # "evaluator": a row of evaluation.evaluate; "tcbp": hierarchy.tcbp_member files; "ensemble": hierarchy.ensemble
    name: str = ""  # for "evaluator": the decoder's name inside summary.json, when the folder holds several
    rule: str = ""  # for our plugin: the tie-break the row's own summary.json must record ("network" or "weight")
    member: str = ""  # the row of hierarchy.ensemble's offline output that stands in when there is no evaluator row
    cores: int = 1    # CPU cores the row's one process uses at once


# The rows of the table on the test shots, in table order. Folder names are the server job's.
TEST_ROWS = [
    TestRow("bposd", "eval_p{p}_bposd", "evaluator"),
    TestRow("relay-bp", "eval_p{p}_relay-bp", "evaluator"),
    TestRow("router", "eval_p{p}_router", "evaluator"),
    TestRow("priortilt", "eval_p{p}_priortilt", "evaluator", member="P1"),
    TestRow("hierarchy-weight", "eval_p{p}_hierarchy-weight", "evaluator", rule="weight", member="aw"),
    TestRow("hierarchy-converged", "eval_p{p}_hierarchy-converged", "evaluator", rule="converged", member="ac"),
    TestRow("hierarchy", "eval_p{p}_hierarchy", "evaluator", rule="network"),
    TestRow("hierarchy-parallel", "eval_p{p}_hierarchy-parallel", "evaluator", rule="network", cores=2),
    TestRow("ensemble", "ensemble_p{p}", "ensemble"),
    TestRow("tcbp", "tcbp_p{p}", "tcbp", member="T"),
]

# The ladder "whose gain is whose", on the test shots, from hierarchy.ensemble's offline output: every rung
# is scored on the same per-shot candidate files, so each step is a paired comparison with the rung before.
# (key in the ensemble's files, name in the table, whose work the step is)
LADDER = [
    ("R", "relay-BP (kit baseline)", "the kit"),
    ("P1", "PriorTilt, one run (seed A)", "IonQ"),
    ("aw", "our hierarchy, lowest-weight tie-break (nothing learned by us)", "ours"),
    ("ac", "our hierarchy, lowest weight among the converged candidates (nothing learned by us)", "ours"),
    ("a", "our hierarchy, neural tie-break", "ours"),
    ("floor", "ORACLE: a perfect tie-break on the same candidates (a ceiling, not a decoder)", "nobody: needs the truth"),
]

OPTIONAL_RUNGS = ("ac",)   # left out of the ladder, without a "not measured" line, when the files do not hold them

# Further paired comparisons on the test shots, (row, against): who adds what, and where TCBP-JOBS stands.
# Each is made only where both rows have a per-shot file for the same test files.
COMPARISONS = [
    ("router", "relay-bp"),
    ("priortilt", "relay-bp"),
    ("hierarchy-weight", "priortilt"),
    ("hierarchy-weight", "relay-bp"),
    ("hierarchy", "priortilt"),
    ("hierarchy", "hierarchy-converged"),
    ("hierarchy-converged", "priortilt"),
    ("hierarchy-converged", "relay-bp"),
    ("hierarchy", "hierarchy-weight"),
    ("hierarchy", "relay-bp"),
    ("hierarchy", "tcbp"),
    ("ensemble", "hierarchy"),
    ("ensemble", "tcbp"),
    ("tcbp", "relay-bp"),
]

# The teammates' earlier evaluator runs on the same test files: other machines, other load.
# They are shown in their own table and used as a cross-check, never mixed into the timed table.
EARLIER_ROWS = [
    ("0.003", "ICRN server, before the quiet window", "teammates_earlier/icrn_eval_p0.003", "teammates_earlier/icrn_router_eval_p0.003", "teammates_earlier/icrn_gain_p0.003"),
    ("0.004", "a teammate's Mac (arm64)", "teammates_earlier/mac_eval_p0.004", "teammates_earlier/mac_router_eval_p0.004", "teammates_earlier/mac_gain_p0.004"),
]

# TCBP-JOBS as the teammates first ran it (tcbp_jobs_baseline.py): its own sampled shots.
# usable_time says whether the logged latency may be quoted, and why not.
TCBP_OWN = [
    ("0.003", "tcbp_own_shots/tcbp-jobs_p0.003_shots10000_seed1_20261122_005215_437826+0000.log", True, ""),
    ("0.004", "tcbp_own_shots/tcbp-jobs_p0.004_shots10000_seed1_20261122_043426_147901+0000.log", False,
     "the GPU was shared with a network training of ours"),
    ("0.005", "tcbp_own_shots/tcbp-jobs_p0.005_shots10000_seed1_20261122_050203_539870+0000.log", False,
     "the GPU was shared with a network training of ours"),
]

# The loss study (X memory, p = 0.004): the conditions shown, in order.
@dataclass(frozen=True)
class LossCondition:
    key: str        # the condition's name in the loss tables
    label: str
    colour: str
    marker: str
    line: str
    oracle: bool = False


LOSS_CONDITIONS = [
    LossCondition("blind", "loss-blind relay-BP (ignores the flags)", "#2a78d6", "o", "-"),
    LossCondition("starter", "flag-aware relay-BP (the published method)", "#6250d6", "s", "-"),
    LossCondition("tracker", "ours: flag-aware with the loss-time tracker", "#eb6834", "^", "-"),
    LossCondition("oracle-round", "ORACLE: told the true loss round (a ceiling, not a decoder)", "#52514e", "D", "--", oracle=True),
]

# The learning curves: one tone per noise level (an ordered ramp, light to dark with p).
P_RAMP = {"0.003": "#86b6ef", "0.004": "#3987e5", "0.005": "#184f95"}
