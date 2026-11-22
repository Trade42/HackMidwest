"""Summarise sizing_run.py outputs. usage: python sizing_analyze.py r003.npz r004.npz r005.npz"""
import sys
import numpy as np
def auc(score, label):
    order = np.argsort(score, kind="stable"); r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s = score[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        r[order[i:j + 1]] = (i + j + 2) / 2; i = j + 1
    pos = label == 1; n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")
for f in sys.argv[1:]:
    z = np.load(f); p = float(z["p"]); leg, r1, r5 = z["leg1"], z["relay1"], z["relay5"]
    print(f"=== p={p}  shots: leg1 {len(leg)} relay1 {len(r1)} relay5 {len(r5)}")
    n = min(len(leg), len(r1)); easy = leg[:n, 0] == 1
    print(f" first leg converges on {easy.mean():.3f} of {n}; relay1 failures: easy {int(r1[:n][easy, 2].sum())}/{easy.sum()}  hard {int(r1[:n][~easy, 2].sum())}/{(~easy).sum()}")
    m = min(len(leg), len(r1), len(r5)); e = leg[:m, 0] == 1
    a, b = r1[:m], r5[:m]
    print(f" same {m} shots: iters mean relay1 {a[:,1].mean():.0f} relay5 {b[:,1].mean():.0f} ratio {b[:,1].mean()/a[:,1].mean():.2f} | median {np.median(a[:,1]):.0f} vs {np.median(b[:,1]):.0f} | fails relay1 {int(a[:,2].sum())} relay5 {int(b[:,2].sum())}")
    print(f"   relay5 on easy shots: mean iters {b[e,1].mean():.0f} (leg1 mean {leg[:m][e,1].mean():.0f}), fails {int(b[e,2].sum())}; on hard shots: mean iters {b[~e,1].mean():.0f}, fails {int(b[~e,2].sum())}/{(~e).sum()}")
    tot = b[:, 1].sum(); srt = np.sort(b[:, 1])[::-1]
    print(f"   relay5 iteration share: hard shots {b[~e,1].sum()/tot:.2f}, slowest 1% {srt[:max(1,m//100)].sum()/tot:.2f}, slowest 5% {srt[:max(1,m//20)].sum()/tot:.2f}, unconverged {b[b[:,0]==0,1].sum()/tot:.2f} ({(b[:,0]==0).mean():.3f} of shots)")
    tot1 = a[:, 1].sum()
    print(f"   relay1 iteration share: unconverged {a[a[:,0]==0,1].sum()/tot1:.2f} ({(a[:,0]==0).mean():.4f} of shots) | telescope(accept leg1, else relay5) mean iters {(leg[:m][e,1].sum() + (80 + b[~e,1]).sum())/m:.0f}")
    hard = r1[:n][~easy]; lab = hard[:, 2]
    if lab.sum() >= 5:
        names = ["converged", "iters", "fail", "sec", "llr_weight", "n_flips", "synd_weight", "mean|post|", "p5|post|", "frac|post|<1"]
        print("   AUC for relay1 failure among hard shots (", int(lab.sum()), "fails /", len(hard), "):",
              {names[i]: round(auc(hard[:, i], lab), 3) for i in (1, 4, 5, 6, 7, 8, 9)})
