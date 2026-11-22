"""For each of the six frozen router models: where fpga_board's exported verdict differs from the kit's rule, and whether
that shot is a wrong candidate; then the integer threshold that restores agreement.

  cd fpga_board && python ../fpga_router/check_fpga_board/threshold_edge.py <scratch folder> ../challenge-1-ml-decoder-kit

It runs fpga_board's own exporter for every model into <scratch folder>/check/regen_<basis>_p<p> (never into
fpga_board/), except X p = 0.004, whose committed generated/ is read. Reads train/validation files only; the labels
("wrong") are used to say whether a shot that changes is a wrong candidate, never to set the threshold.
"""
import json, subprocess, sys, numpy as np
from scipy.special import expit
S, K = sys.argv[1], sys.argv[2]
sys.path.insert(0, "tools")
import board_demo as bd
for p, b in (("0.003", "X"), ("0.003", "Z"), ("0.004", "Z"), ("0.005", "X"), ("0.005", "Z")):
    subprocess.run([sys.executable, "tools/board_demo.py", "export", "--model", f"{K}/results/router_p{p}/model_{b}.json",
                    "--features", f"{K}/results/router_p{p}/validation_{b}.npz", "--data", f"{K}/data/p{p}/validation_{b}.npz",
                    "--out", f"{S}/check/regen_{b}_p{p}", "--uart-vectors", "32"], check=True, capture_output=True)
for p, b in (("0.003", "X"), ("0.003", "Z"), ("0.004", "X"), ("0.004", "Z"), ("0.005", "X"), ("0.005", "Z")):
    m = json.load(open(f"{K}/results/router_p{p}/model_{b}.json"))
    f = np.load(f"{K}/results/router_p{p}/validation_{b}.npz")
    F, easy, wrong = f["features"], f["first_leg_converged"], f["wrong"].astype(bool)
    w = np.asarray(m["weights"]); z = w[0] + ((F - m["mean"]) / m["scale"]) @ w[1:]
    soft = (~easy) & (expit(z) > m["threshold"])
    gen = "generated" if (p, b) == ("0.004", "X") else f"{S}/check/regen_{b}_p{p}"
    exp = np.frombuffer(open(f"{gen}/expected.bin", "rb").read(), np.uint8).reshape(-1, 8)
    hw = exp[:, 7].astype(bool)
    score = np.array([int.from_bytes(r[:7].tobytes(), "little", signed=True) for r in exp])
    weights, bq, tq, frac, raw, bias = bd.quantize_model(m)
    fl = np.array([bias + sum(a * c for a, c in zip(raw, row)) for row in F.tolist()])
    exporter_float = (~easy) & (1 / (1 + np.exp(-fl)) > m["threshold"])
    hard_wrong = wrong & ~easy
    k = np.flatnonzero(hard_wrong)[np.argmin(z[hard_wrong])]
    print(f"{b} p={p}: board != numpy rule on {np.flatnonzero(hw != soft).tolist()}; board != exporter's float on {np.flatnonzero(hw != exporter_float).tolist()}."
          f" Lowest-risk wrong hard shot {k}: risk - threshold = {expit(z[k]) - m['threshold']:.3g}; fixed score - THRESH = {score[k] - tq}"
          f" -> board {'escalates it' if hw[k] else 'ACCEPTS this wrong candidate'}. Wrong hard shots {int(hard_wrong.sum())}:"
          f" accepted by the board {int((hard_wrong & ~hw).sum())}, by the software {int((hard_wrong & ~soft).sum())}")

print("\nThe proposed fix: THRESH = (lowest fixed score among the hard shots the software rule escalates) - 1,")
print("which uses the frozen rule's verdicts on the validation shots, not the labels. It restores parity if no shot")
print("that the software accepts scores at or above that value.")
for p, b in (("0.003", "X"), ("0.003", "Z"), ("0.004", "X"), ("0.004", "Z"), ("0.005", "X"), ("0.005", "Z")):
    m = json.load(open(f"{K}/results/router_p{p}/model_{b}.json"))
    f = np.load(f"{K}/results/router_p{p}/validation_{b}.npz")
    F, easy, wrong = f["features"], f["first_leg_converged"], f["wrong"].astype(bool)
    w = np.asarray(m["weights"]); z = w[0] + ((F - m["mean"]) / m["scale"]) @ w[1:]
    soft = (~easy) & (expit(z) > m["threshold"])
    gen = "generated" if (p, b) == ("0.004", "X") else f"{S}/check/regen_{b}_p{p}"
    exp = np.frombuffer(open(f"{gen}/expected.bin", "rb").read(), np.uint8).reshape(-1, 8)
    score = np.array([int.from_bytes(r[:7].tobytes(), "little", signed=True) for r in exp])
    _, _, tq, frac, _, _ = bd.quantize_model(m)
    new = int(score[soft].min()) - 1
    accepted_hard = (~easy) & ~soft
    gap = new - int(score[accepted_hard].max())
    fixed = (~easy) & (score > new)
    print(f"  {b} p={p}: THRESH {tq} -> {new} (a change of {new - tq} in units of 2^-{frac});"
          f" highest score among the accepted hard shots is {gap} below it; board != software rule on {int((fixed != soft).sum())} shots")
