"""Train the router that the FPGA runs, and export it with its test vectors.

  python train_router.py KIT_DIR P BASIS OUT_DIR
  python train_router.py ../challenge-1-ml-decoder-kit 0.004 X trained_X_p0.004

What the router is. The teammates' decoder (evaluation/router.py) first asks relay-BP for one solution. If that
took at most 80 iterations the shot is "easy" and the solution is kept. Otherwise a small model looks at a few
numbers about the shot and says "escalate" (decode again with full relay-BP) or "accept". This script fits that
model for the board: same shots, same label, same recipe, but on the 16 whole numbers the board receives.

It reads, and never a test_* or go_* file:
  KIT_DIR/results/router_p<P>/{train,validation}_<BASIS>.npz   the teammates' saved numbers of relay-BP's first solution
                                                               (iterations, converged, mean |posterior| clipped at 100,
                                                               solution weight, 13 syndrome weights) and their label
  KIT_DIR/results/router_p<P>/model_<BASIS>.json               their frozen 17-feature model, for the comparison
  KIT_DIR/data/p<P>/{train,validation}_<BASIS>.npz             the shots; only the 936 detector bits of the memory basis are used

The label ("the first solution's 12 logical bits are wrong") is used to fit, to choose the threshold and to score.
It is never an input: the 16 features come from detector bits and from relay-BP's first solution only.

It writes into OUT_DIR: model.npz (the float fit and the integer threshold), report.json (every number printed),
and through router_fpga.export_arrays: router_params.vh, vectors.mem (all validation shots), tb_vectors.vh.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import router_fpga as rf


def load_split(kit, p, basis, split):
    """One split: the syndromes from the shot file, everything else from the teammates' feature file of the same shots."""
    assert split in ("train", "validation"), "this script opens train and validation files only"
    from evaluation.router import FEATURE_NAMES
    feature_file = kit / "results" / f"router_p{p}" / f"{split}_{basis}.npz"
    data_file = kit / "data" / f"p{p}" / f"{split}_{basis}.npz"
    for path in (feature_file, data_file):
        assert not path.name.startswith(("test_", "go_")), f"{path} is a held-out file"
    with np.load(feature_file, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata"].item()))
        F, wrong, easy = f["features"], f["wrong"].astype(bool), f["first_leg_converged"].astype(bool)
    assert meta["split"] == split and meta["basis"] == basis and meta["feature_names"] == list(FEATURE_NAMES)
    with np.load(data_file, allow_pickle=False) as d:
        assert str(d["basis"]) == basis and int(d["seed"]) == meta["dataset"]["seed"]
        syndrome = d["detectors"][:, d["basis_detectors"]].astype(np.uint8)
    names = list(FEATURE_NAMES)
    it, conv, post, wt = (F[:, names.index(n)] for n in ("first_solution_iterations", "candidate_converged",
                                                         "mean_abs_posterior_ratio", "correction_weight"))
    # The two files describe the same shots in the same order: the 13 syndrome weights agree on every shot.
    assert np.array_equal(syndrome.reshape(len(syndrome), rf.ROUNDS, rf.CHECKS).sum(-1), F[:, 4:]), "rows do not line up"
    assert np.array_equal(it, np.round(it)) and np.array_equal(wt, np.round(wt))
    return {"split": split, "seed": meta["dataset"]["seed"], "dataset_sha256": meta["dataset"]["sha256"], "F17": F,
            "syndrome": syndrome, "iterations": it.astype(np.int64), "converged": conv.astype(bool),
            "mean_abs_post": post, "weight": wt.astype(np.int64), "wrong": wrong, "easy": easy}


def fit_logistic(X, wrong, l2=1.0):
    """The teammates' fit (evaluation/router.py, fit_logistic) for any number of columns: standardise, L2, L-BFGS."""
    X, wrong = np.asarray(X, float), np.asarray(wrong, float)
    mean, scale = X.mean(0), X.std(0)
    scale[scale == 0] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mean) / scale])

    def objective(w):
        z = Z @ w
        gradient = Z.T @ (expit(z) - wrong)
        gradient[1:] += l2 * w[1:]
        return np.logaddexp(0, z).sum() - wrong @ z + .5 * l2 * np.dot(w[1:], w[1:]), gradient

    start = np.zeros(Z.shape[1])
    start[0] = np.log(wrong.mean() / (1 - wrong.mean()))
    result = minimize(objective, start, jac=True, method="L-BFGS-B")
    assert result.success, result.message
    return {"coef": result.x[1:], "intercept": result.x[0], "mean": mean, "scale": scale}


def logit(model, X):
    return model["intercept"] + ((np.asarray(X, float) - model["mean"]) / model["scale"]) @ model["coef"]


def auc(score, wrong):
    """The chance that a wrong shot scores above a right one (ties count half). 0.5 = no information, 1 = perfect."""
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for value in np.unique(score[np.concatenate([[False], np.diff(np.sort(score)) == 0])]):   # average the ranks of ties
        ranks[score == value] = ranks[score == value].mean()
    n1, n0 = int(wrong.sum()), int((~wrong).sum())
    return float((ranks[wrong].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def table(wrong, escalated):
    """Counts of one decision against the label."""
    return {"shots": int(len(wrong)), "wrong": int(wrong.sum()), "escalated": int(escalated.sum()),
            "wrong_escalated": int((wrong & escalated).sum()), "wrong_accepted": int((wrong & ~escalated).sum()),
            "right_escalated": int((~wrong & escalated).sum()), "right_accepted": int((~wrong & ~escalated).sum())}


def main(kit, p, basis, out):
    kit, out = Path(kit).resolve(), Path(out)
    sys.path.insert(0, str(kit))
    from decoders import RELAY_SETTINGS
    from evaluation.router import risk
    easy_iter = RELAY_SETTINGS["pre_iter"]                      # 80: the length of relay-BP's first leg
    train, val = load_split(kit, p, basis, "train"), load_split(kit, p, basis, "validation")
    assert train["dataset_sha256"] != val["dataset_sha256"] and train["seed"] != val["seed"]
    theirs = json.loads((kit / "results" / f"router_p{p}" / f"model_{basis}.json").read_text())
    assert theirs["basis"] == basis and theirs["validation_dataset"]["sha256"] == val["dataset_sha256"]

    for d in (train, val):
        # The board tells an easy shot from its iteration count alone. Check that this equals the teammates' flag.
        assert np.array_equal(d["easy"], (d["iterations"] > 0) & (d["iterations"] <= easy_iter))
        assert not (d["easy"] & ~d["converged"]).any()
        d["hard"] = ~d["easy"]
        d["x_int"] = rf.features(d["syndrome"], d["iterations"], d["weight"], d["mean_abs_post"])   # what the board gets
        d["x_raw"] = np.delete(d["F17"], 1, axis=1)             # the teammates' columns without "candidate_converged"
    th, vh = train["hard"], val["hard"]
    print(f"{basis} memory, p = {p}. Train: {len(th)} shots (Stim seed {train['seed']}), {int(th.sum())} hard, "
          f"{int(train['wrong'][th].sum())} of them wrong. Validation: {len(vh)} shots (seed {val['seed']}), {int(vh.sum())} hard, "
          f"{int(val['wrong'][vh].sum())} of them wrong; {int(val['wrong'][~vh].sum())} easy shots are wrong and no router sees them.")

    # Four models, each scored on the hard validation shots.
    refit = fit_logistic(train["F17"][th], train["wrong"][th])
    drift = float(np.abs(np.r_[refit["intercept"], refit["coef"]] - np.asarray(theirs["weights"])).max())
    print(f"check: the fit of this script on their 17 features gives their frozen weights to within {drift:.1e}")
    assert drift < 1e-3
    raw = fit_logistic(train["x_raw"][th], train["wrong"][th])
    coded = fit_logistic(train["x_int"][th], train["wrong"][th])
    W, BIAS, _, frac, _, _ = rf.quantize(coded | {"threshold": 0.5})
    scores = {
        "A": (risk(theirs, train["F17"]), risk(theirs, val["F17"])),
        "B": (logit(raw, train["x_raw"]), logit(raw, val["x_raw"])),
        "C": (logit(coded, train["x_int"]), logit(coded, val["x_int"])),
        "D": (rf.score_fixed(train["x_int"], W, BIAS), rf.score_fixed(val["x_int"], W, BIAS)),
    }
    names = {"A": "A  the teammates' frozen model: 17 features, floats",
             "B": "B  16 features (theirs without the converged flag), floats",
             "C": "C  the board's 16 whole-number features, float weights",
             "D": f"D  the board's model: 16 whole numbers, 16-bit weights, FRAC = {frac}"}
    val_wrong_hard = val["wrong"] & vh
    thresholds, rows, decisions = {}, {}, {}
    for key, (s_train, s_val) in scores.items():
        lowest = s_val[val_wrong_hard].min()                    # zero wrong hard validation shots accepted
        if key == "A":
            thresholds[key] = theirs["threshold"]               # frozen by them with the same rule, on another machine
            steps = (lowest - theirs["threshold"]) / np.spacing(lowest)
            print(f"check: their frozen threshold lies {steps:.0f} floating-point steps below the lowest wrong shot's risk as computed here")
            recorded, here = theirs["validation"], table(val["wrong"][vh], (s_val > theirs["threshold"])[vh])
            assert steps > 0 and (recorded["escalated"], recorded["wrong_accepted"]) == (here["escalated"], here["wrong_accepted"])
        elif key == "D":
            thresholds[key] = int(lowest) - 1                   # whole numbers: one below the lowest wrong shot's score
        else:
            thresholds[key] = float(np.nextafter(lowest, -np.inf))
        esc_val, esc_train = (s_val > thresholds[key]) & vh, (s_train > thresholds[key]) & th
        decisions[key] = esc_val
        rows[key] = {"model": names[key], "auc_validation_hard": auc(s_val[vh], val["wrong"][vh]),
                     "validation_hard": table(val["wrong"][vh], esc_val[vh]), "validation_all": table(val["wrong"], esc_val),
                     "train_hard": table(train["wrong"][th], esc_train[th])}
    print("\nEach model at its own threshold: the highest one that accepts no wrong hard validation shot.")
    print("The threshold is chosen on the validation shots, so 'wrong accepted = 0' there is by construction;")
    print("the train columns show the same threshold on the shots the weights were fitted on.")
    print(f"{'model':72s} {'AUC':>6s} | validation, {int(vh.sum())} hard: escalated accepted wrong-accepted | train, {int(th.sum())} hard: escalated accepted wrong-accepted")
    for key, row in rows.items():
        v, t = row["validation_hard"], row["train_hard"]
        print(f"{row['model']:72s} {row['auc_validation_hard']:6.3f} | {v['escalated']:28d} {v['shots'] - v['escalated']:8d} {v['wrong_accepted']:14d} |"
              f" {t['escalated']:27d} {t['shots'] - t['escalated']:8d} {t['wrong_accepted']:14d}")

    # The board's model against the teammates', shot by shot.
    a, d = decisions["A"], decisions["D"]
    agreement = {}
    for label, mask in (("validation_hard", vh), ("validation_all", np.ones_like(vh))):
        agreement[label] = {"shots": int(mask.sum()), "same": int((a == d)[mask].sum()), "both_escalate": int((a & d)[mask].sum()),
                            "both_accept": int((~a & ~d)[mask].sum()), "only_theirs_escalates": int((a & ~d)[mask].sum()),
                            "only_board_escalates": int((~a & d)[mask].sum())}
    g = agreement["validation_hard"]
    print(f"\nSame decision, the board's model (D) and the teammates' (A): {g['same']} of {g['shots']} hard validation shots "
          f"(both escalate {g['both_escalate']}, both accept {g['both_accept']}, only theirs escalates {g['only_theirs_escalates']}, "
          f"only the board's escalates {g['only_board_escalates']});")
    g = agreement["validation_all"]
    print(f"  {g['same']} of {g['shots']} validation shots when the easy ones, which both accept by the same rule, are counted.")
    print("Against the label (wrong = the first solution's 12 logical bits are wrong):")
    for key in "AD":
        for label in ("validation_hard", "validation_all"):
            c = rows[key][label]
            print(f"  {key}, {label:15s}: {c['shots']:5d} shots | wrong and escalated {c['wrong_escalated']:3d} | wrong and accepted {c['wrong_accepted']:3d}"
                  f" | right and escalated {c['right_escalated']:4d} | right and accepted {c['right_accepted']:4d}")

    # Export: the weights, and every validation shot as a test vector with the reply the board must give.
    out.mkdir(parents=True, exist_ok=True)
    T = thresholds["D"]
    model = {"coef": coded["coef"], "intercept": coded["intercept"], "mean": coded["mean"], "scale": coded["scale"],
             "threshold": float(expit(T / 2 ** frac)), "thresh_fixed": T, "easy_iter": easy_iter}
    W2, BIAS2, T2, frac2, _, easy2 = rf.quantize(model)
    assert np.array_equal(W, W2) and (BIAS, T, frac, easy_iter) == (BIAS2, T2, frac2, easy2)
    np.savez(out / "model.npz", **model)
    print()
    x, fixed, esc = rf.export_arrays(model, val, out)
    assert np.array_equal(x, val["x_int"]) and np.array_equal(fixed, scores["D"][1]) and np.array_equal(esc, d)
    assert x.max() < 2 ** rf.XW and np.abs(fixed).max() < 2 ** (rf.AW - 1)
    report = {"basis": basis, "p": p, "easy_iter": easy_iter, "frac": frac, "W": W.tolist(), "BIAS": BIAS, "THRESH": T,
              "train": {"seed": train["seed"], "sha256": train["dataset_sha256"], "shots": len(th), "hard": int(th.sum())},
              "validation": {"seed": val["seed"], "sha256": val["dataset_sha256"], "shots": len(vh), "hard": int(vh.sum())},
              "thresholds": thresholds, "models": rows, "board_against_teammates": agreement,
              "largest_feature": int(x.max()), "largest_abs_score": int(np.abs(fixed).max()),
              "posterior_codes_on_hard_train_shots": int(len(np.unique(train["x_int"][th][:, rf.ROUNDS + 2])))}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out}/model.npz and {out}/report.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
