"""
CALM-Stroke: scanner-level meta-analysis and selection-bias sensitivity.

1. Bayesian random-effects meta-analysis of the AUC across held-out scanner models
   (leave-one-scanner-out), on the logit-AUC scale, with a half-normal(0, 0.5) prior on the
   between-scanner SD (tau) and a flat prior on the mean; posterior by exact marginalization
   over a tau grid. Reports the pooled AUC and the 95% prediction interval for a new scanner.
2. Selection-bias sensitivity: inverse probability weighting of the labeled cohort toward all
   sessions with age and NIHSS recorded (logistic selection model on age, NIHSS, log acute
   lesion volume, mask availability, sex and field strength), then weighted AUC with
   bootstrap intervals.

Usage: python src/scanner_meta_selection.py [repo root] [output dir]
       (defaults: current directory and results/paper)
"""
import sys, glob, os
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
RESD = os.environ.get("CALM_RESULTS", os.path.join(REPO, "results"))
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RESD, "paper")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261117)
coh = pd.read_csv(f"{RESD}/cohort_all.csv").set_index("participant_id")
lab = coh[coh.labelled == True]
y_all = lab.y_poor.astype(int)


def seed_avg(pattern, col="p_gt2"):
    fs = sorted(glob.glob(f"{RESD}/oof/{pattern}"))
    d = pd.concat([pd.read_csv(f) for f in fs])
    return d.groupby("participant_id").agg(p=(col, "mean"), unit=("test_unit", "first"))


def wauc(y, s, w):
    o = np.argsort(s, kind="mergesort"); y, s, w = y[o], s[o], w[o]
    _, st, ct = np.unique(s, return_index=True, return_counts=True)
    neg, pos = w * (y == 0), w * (y == 1)
    gneg = np.add.reduceat(neg, st)
    prev = np.r_[0, np.cumsum(gneg)[:-1]]
    gid = np.repeat(np.arange(len(st)), ct)
    return (pos * (prev[gid] + 0.5 * gneg[gid])).sum() / (pos.sum() * neg.sum())


# ------------------------------------------------------------------ 1. meta-analysis
models = {"CALM-Stroke": "calm25d__loso__s*.csv", "2.5D MIL + anchor": "slice25d__loso__s*.csv",
          "Clinical + lesion (R1b)": "baseline_R1b_*__loso.csv", "Clinical (R0)": "baseline_R0_clinical__loso.csv"}
rows, meta = [], []
taus = np.linspace(1e-4, 2.0, 4000)
prior_tau = 2 * norm.pdf(taus, 0, 0.5)
for name, pat in models.items():
    d = seed_avg(pat).join(y_all, how="inner")
    est = []
    for sc, g in d.groupby("unit"):
        yy, pp = g.y_poor.values, g.p.values
        a = roc_auc_score(yy, pp)
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(g), len(g))
            if yy[i].min() != yy[i].max(): bs.append(roc_auc_score(yy[i], pp[i]))
        lb = np.log(np.array(bs) / (1 - np.array(bs)))
        est.append((sc, len(g), a, np.log(a / (1 - a)), lb.std()))
        rows.append(dict(model=name, scanner=sc, n=len(g), auc=a, lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
    e = pd.DataFrame(est, columns=["scanner", "n", "auc", "th", "se"])
    th, v = e.th.values, e.se.values ** 2
    # marginal posterior of tau (flat prior on mu): p(tau|y) ∝ p(tau) |V|^-1/2 (sum w)^-1/2 exp(-Q/2)
    lp = []
    for t in taus:
        w = 1 / (v + t ** 2); mu = (w * th).sum() / w.sum()
        lp.append(-0.5 * np.log(v + t ** 2).sum() - 0.5 * np.log(w.sum()) - 0.5 * (w * (th - mu) ** 2).sum())
    lp = np.array(lp) + np.log(prior_tau); pt = np.exp(lp - lp.max()); pt /= pt.sum()
    # draws
    ti = rng.choice(len(taus), size=20000, p=pt); t = taus[ti]
    W = 1 / (v[None, :] + t[:, None] ** 2)
    mu_hat = (W * th).sum(1) / W.sum(1); mu = rng.normal(mu_hat, np.sqrt(1 / W.sum(1)))
    new = rng.normal(mu, t)
    ex = lambda z: 1 / (1 + np.exp(-z))
    meta.append(dict(model=name, k=len(e), mu=ex(np.median(mu)), mu_lo=ex(np.quantile(mu, .025)), mu_hi=ex(np.quantile(mu, .975)),
                     pi_lo=ex(np.quantile(new, .025)), pi_hi=ex(np.quantile(new, .975)),
                     tau_med=np.median(t), tau_hi=np.quantile(t, .95), p_new_gt_08=(ex(new) > 0.80).mean()))
PS = pd.DataFrame(rows); MA = pd.DataFrame(meta)
PS.to_csv(f"{OUT}/scanner_auc.csv", index=False, float_format="%.4f")
MA.to_csv(f"{OUT}/scanner_meta.csv", index=False, float_format="%.4f")
print(PS.round(3).to_string()); print(MA.round(3).to_string())

# ------------------------------------------------------------------ 2. IPW selection sensitivity
c = coh[(coh.dwi_ok == True) & coh.age.notna() & coh.nihss.notna()].copy()
c["lab"] = c.labelled.astype(bool).astype(int)
c["logvol"] = np.log1p(c.acute_ml_filled.fillna(0))
c["mask"] = c.has_mask.astype(float); c["male"] = (c.sex == "M").astype(float) if c.sex.dtype == object else c.sex_m.fillna(0)
c["t3"] = (c.field.astype(float) > 2).astype(float)
X = c[["age", "nihss", "logvol", "mask", "male", "t3"]].copy()
X["nihss_log"] = np.log1p(X.nihss)
Xs = (X - X.mean()) / X.std()
sel = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, c.lab)
ps = pd.Series(sel.predict_proba(Xs)[:, 1], index=c.index)
from sklearn.metrics import roc_auc_score as _r
sel_auc = _r(c.lab, ps)
L = c[c.lab == 1]
w = (c.lab.mean() / ps.loc[L.index]).clip(upper=np.quantile(1 / ps.loc[L.index], 0.99) * c.lab.mean())
ess = w.sum() ** 2 / (w ** 2).sum()
preds = {"CALM-Stroke": "calm25d__cv5__s*.csv", "2.5D MIL + anchor": "slice25d__cv5__s*.csv",
         "Clinical + lesion (R1b)": "baseline_R1b_*__cv5.csv", "Clinical (R0)": "baseline_R0_clinical__cv5.csv"}
P = pd.DataFrame({k: seed_avg(v).p for k, v in preds.items()}).loc[L.index]
yy = L.y_poor.values.astype(int); ww = w.values
res = {k: dict(unweighted=wauc(yy, P[k].values, np.ones(len(yy))), weighted=wauc(yy, P[k].values, ww)) for k in P}
B = []
for _ in range(2000):
    i = rng.integers(0, len(yy), len(yy))
    B.append({k: wauc(yy[i], P[k].values[i], ww[i]) for k in P})
B = pd.DataFrame(B)
for k in P:
    res[k]["lo"], res[k]["hi"] = B[k].quantile(.025), B[k].quantile(.975)
dd = B["CALM-Stroke"] - B["Clinical + lesion (R1b)"]
res["diff_calm_R1b"] = dict(weighted=wauc(yy, P["CALM-Stroke"].values, ww) - wauc(yy, P["Clinical + lesion (R1b)"].values, ww),
                            lo=dd.quantile(.025), hi=dd.quantile(.975))
IP = pd.DataFrame(res).T
IP.to_csv(f"{OUT}/ipw.csv", float_format="%.4f")
print(IP.round(3).to_string()); print("selection AUC", round(sel_auc, 3), "ESS", round(ess, 1), "n sessions", len(c),
                                     "weighted median NIHSS", np.round(np.average(L.nihss, weights=ww), 1), "vs", L.nihss.mean().round(1))

with open(f"{OUT}/numbers_extra.tex", "w") as f:
    f.write("% generated by src/scanner_meta_selection.py\n\\providecommand{\\NAX}[2]{\\csname nax@#1@#2\\endcsname}\n")
    def wr(k, fld, v, fmt="%.3f"):
        f.write("\\expandafter\\def\\csname nax@%s@%s\\endcsname{%s}\n" % (k, fld, (fmt % v).replace("-", "$-$")))
    key = {"CALM-Stroke": "calm", "2.5D MIL + anchor": "mil", "Clinical + lesion (R1b)": "R1b", "Clinical (R0)": "R0"}
    for _, r in MA.iterrows():
        for fld in ["mu", "mu_lo", "mu_hi", "pi_lo", "pi_hi"]: wr("ma" + key[r.model], fld.replace("_", ""), r[fld])
        wr("ma" + key[r.model], "taumed", r.tau_med, "%.2f"); wr("ma" + key[r.model], "pnew", r.p_new_gt_08, "%.2f")
    for k in preds:
        for fld in ["weighted", "lo", "hi"]: wr("ipw" + key[k], fld, IP.loc[k, fld])
    for fld in ["weighted", "lo", "hi"]: wr("ipwdiff", fld, IP.loc["diff_calm_R1b", fld])
    wr("ipw", "selauc", sel_auc, "%.2f"); wr("ipw", "ess", ess, "%.0f"); wr("ipw", "nsess", len(c), "%d")
