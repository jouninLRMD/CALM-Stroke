"""
Figures, LaTeX table bodies and number macros of the manuscript (run after evaluate.py).

Reads results/oof/*.csv, results/table_main_<scheme>.csv, results/table_loso_by_scanner.csv, results/cohort_all.csv.
Writes results/paper/:
  fig5_discrimination.pdf/png   ROC, calibration and decision curves (5-fold CV)
  fig6_robustness(_col).pdf/png change in AUC vs the clinical + lesion model across validation schemes; AUC per scanner
  fig7_subgroups.pdf/png        subgroup AUC with bootstrap 95% CI (supplementary Fig. S1)
  fig8_ablation.pdf/png         per-seed and ensemble AUC along the model comparison
  numbers.tex                   macros used in the manuscript text
  tab_main_cv5.tex, tab_external.tex, tab_cohort.tex, tab_perfold.tex   table bodies
Usage: python src/paper_outputs.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from scipy.stats import mannwhitneyu, chi2_contingency
from common import RES, POOR_K, auc, boot_auc_ci, paired_boot_delta

OUT = RES / "paper"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})

# ------------------------------------------------------------------------------------------- names and colours
NAME = {"baseline_R0_clinical": "Clinical (R0)", "baseline_R1_clin_volume": "Clinical + volume (R1)",
        "baseline_R1b_clin_vol_chronic_radc": "Clinical + lesion (R1b)", "baseline_R2a_clin_vol_territory": "Clinical + territory (R2a)",
        "baseline_R2b_clin_vol_topography": "Clinical + topography (R2b)", "baseline_I1_volume_only": "Volume only (I1)",
        "baseline_I2_topography_only": "Topography only (I2)", "resnet3d_noanchor": "3D ResNet-18, no anchor",
        "resnet3d": "3D ResNet-18 + anchor", "calm": "CALM-3D (from scratch)", "slice25d": "2.5D MIL + anchor",
        "slice25d_lesanchor": "2.5D MIL + lesion anchor", "calm25d_noanchor": "CALM-Stroke, no anchor",
        "calm25d_noseg": "CALM-Stroke, no lesion head", "calm25d_noaux": "CALM-Stroke, no auxiliary tasks",
        "calm25d": "CALM-Stroke"}
KEY = {"baseline_R0_clinical": "R0", "baseline_R1_clin_volume": "R1", "baseline_R1b_clin_vol_chronic_radc": "R1b",
       "baseline_R2a_clin_vol_territory": "R2a", "baseline_R2b_clin_vol_topography": "R2b", "baseline_I1_volume_only": "I1",
       "baseline_I2_topography_only": "I2", "resnet3d_noanchor": "resna", "resnet3d": "res", "calm": "calmthreed",
       "slice25d": "mil", "slice25d_lesanchor": "milles", "calm25d_noanchor": "calmna", "calm25d_noseg": "calmns",
       "calm25d_noaux": "calmnx", "calm25d": "calm"}
COL = {"baseline_R0_clinical": "#9e9e9e", "baseline_R1b_clin_vol_chronic_radc": "#3c3c3c",
       "baseline_R2b_clin_vol_topography": "#8c6d31", "resnet3d": "#66a61e", "calm": "#7570b3", "slice25d": "#1f78b4",
       "slice25d_lesanchor": "#a6cee3", "calm25d": "#d95f02"}
R0, R1B = "baseline_R0_clinical", "baseline_R1b_clin_vol_chronic_radc"
PC = f"p_gt{POOR_K}"

coh = pd.read_csv(RES / "cohort_all.csv")
L = coh[coh.labelled].set_index("participant_id")
try:
    FA = pd.read_csv(RES / "features_atlas.csv").set_index("participant_id"); L = L.join(FA[["post_frac", "frac_left"]])
except FileNotFoundError:
    L["post_frac"] = np.nan; L["frac_left"] = np.nan

preds = {}
for f in sorted((RES / "oof").glob("*.csv")):
    parts = f.stem.split("__"); d = pd.read_csv(f).set_index("participant_id")
    preds.setdefault((parts[1], parts[0]), {})[parts[2] if len(parts) > 2 else "s0"] = d


def complete(scheme, m):
    """Ensemble over the seeds whose prediction file covers every labelled patient of the scheme."""
    dd = preds.get((scheme, m), {})
    full = max((len(d) for d in dd.values()), default=0)
    dd = {s: d for s, d in dd.items() if len(d) == full}
    if not dd:
        return None, {}
    ids = sorted(set.intersection(*[set(d.index) for d in dd.values()]) & set(L.index))
    return pd.Series(np.mean([d.loc[ids, PC].values for d in dd.values()], 0), index=ids), dd


ENS = {s: {} for s in ["cv5", "loso", "field"]}
for (s, m) in preds:
    p, _ = complete(s, m)
    if p is not None:
        ENS[s][m] = p
MAIN = "calm25d" if "calm25d" in ENS["cv5"] else "slice25d"
print("Main model:", MAIN, "| cv5 models:", sorted(ENS["cv5"]))
T = {s: pd.read_csv(RES / f"table_main_{s}.csv").set_index("model") for s in ["cv5", "loso", "field"]
     if (RES / f"table_main_{s}.csv").exists()}
ids5 = sorted(set.intersection(*[set(p.index) for p in ENS["cv5"].values()]))
y5 = L.loc[ids5, "y_poor"].values.astype(int)


def lab(m):
    return NAME.get(m, m)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)


def panel(ax, s):
    ax.text(-0.18, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


# ------------------------------------------------------------------------------------------- Fig 5 discrimination
from sklearn.metrics import roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess
show = [m for m in [R0, R1B, "resnet3d", "calm", "slice25d", "calm25d"] if m in ENS["cv5"]]
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw=dict(wspace=0.38))
for m in show:
    p = ENS["cv5"][m].loc[ids5].values; lw = 1.6 if m == MAIN else 1.0
    fpr, tpr, _ = roc_curve(y5, p)
    ax[0].plot(fpr, tpr, color=COL[m], lw=lw, label=f"{lab(m)}  {auc(y5, p):.3f}")
    sm = lowess(y5, p, frac=0.6, it=0, return_sorted=True)
    ax[1].plot(sm[:, 0], sm[:, 1], color=COL[m], lw=lw)
ax[0].plot([0, 1], [0, 1], ls=":", color="#777777", lw=0.7)
ax[0].set(xlabel="1 − specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01))
ax[0].legend(loc="lower right", fontsize=5.6, handlelength=1.4, title="AUC", title_fontsize=5.6, borderaxespad=0.2)
pm = ENS["cv5"][MAIN].loc[ids5].values
q = pd.qcut(pm, 10, labels=False, duplicates="drop")
ax[1].plot([pm[q == i].mean() for i in np.unique(q)], [y5[q == i].mean() for i in np.unique(q)], "o", ms=2.6,
           color=COL[MAIN], mfc="white", mew=0.8, label=f"{lab(MAIN)}, deciles")
ax[1].plot([0, 1], [0, 1], ls=":", color="#777777", lw=0.7)
axh = ax[1].inset_axes([0, 0, 1, 0.14]); axh.hist(pm, bins=40, range=(0, 1), color=COL[MAIN], alpha=0.5); axh.axis("off")
ax[1].set(xlabel="Predicted probability of mRS ≥ 3", ylabel="Observed proportion", xlim=(0, 1), ylim=(0, 1))
ax[1].legend(loc="upper left", fontsize=5.6, handletextpad=0.2)
thr = np.arange(0.05, 0.81, 0.01); prev = y5.mean()
nb = lambda p, t: ((p >= t) & (y5 == 1)).mean() - ((p >= t) & (y5 == 0)).mean() * t / (1 - t)
ax[2].plot(thr, prev - (1 - prev) * thr / (1 - thr), color="#bbbbbb", lw=0.9, label="Treat all")
ax[2].axhline(0, color="#777777", lw=0.7, ls=":", label="Treat none")
for m in dict.fromkeys([R0, R1B, "slice25d", MAIN]):
    if m in ENS["cv5"]:
        p = ENS["cv5"][m].loc[ids5].values
        ax[2].plot(thr, [nb(p, t) for t in thr], color=COL[m], lw=1.6 if m == MAIN else 1.0, label=lab(m))
ax[2].set(xlabel="Threshold probability", ylabel="Net benefit", ylim=(-0.03, 0.5), xlim=(0.05, 0.8))
h, l = ax[2].get_legend_handles_labels(); ax[2].legend(h, l, fontsize=5.6, loc="upper right", handlelength=1.4)
for a, s in zip(ax, "abc"):
    panel(a, s)
save(fig, "fig5_discrimination")

# ------------------------------------------------------------------------------------------- Fig 6 robustness
deep = [m for m in ["resnet3d", "calm", "slice25d", "slice25d_lesanchor", "calm25d"] if m in T.get("cv5", pd.DataFrame()).index]
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.3), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.55))
SCH = [("cv5", "5-fold CV (n = 718)", "o"), ("loso", "Leave one scanner out (n = 718)", "s"),
       ("field", "1.5 T → 3 T (n = 178)", "^")]
col = "dAUC_vs_R1b_clin_vol_chronic_radc"
for j, (s, sl, mk) in enumerate(SCH):
    if s not in T:
        continue
    for i, m in enumerate(deep):
        if m in T[s].index:
            r = T[s].loc[m]; yy = i + (j - 1) * 0.22
            ax[0].errorbar(r[col], yy, xerr=[[r[col] - r[col + "_lo"]], [r[col + "_hi"] - r[col]]], fmt=mk, ms=3,
                           color=COL[m], mfc="white" if j else COL[m], mew=0.8, lw=0.8, capsize=0,
                           label=sl if i == len(deep) - 1 or (m == deep[-1]) else None)
ax[0].axvline(0, color="#777777", lw=0.7, ls=":")
ax[0].set_yticks(range(len(deep))); ax[0].set_yticklabels([lab(m) for m in deep]); ax[0].invert_yaxis()
ax[0].set_xlabel("Change in AUC vs clinical + lesion model (95% CI)")
from matplotlib.lines import Line2D
ax[0].legend([Line2D([], [], marker=mk, ls="", color="#444444", mfc="#444444" if j == 0 else "white", ms=3.5) for j, (_, _, mk) in enumerate(SCH)],
             [sl for _, sl, _ in SCH], fontsize=5.6, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.8)
if (RES / "table_loso_by_scanner.csv").exists():
    S = pd.read_csv(RES / "table_loso_by_scanner.csv").set_index("model")
    cols = list(S.columns)
    n_sc = L.scanner_group.value_counts()
    cats = [f"{c.replace('Ingenia ', '')}\nn = {n_sc.get(c, 0)}" for c in cols] + ["3 T (trained on\n1.5 T), n = 178"]
    mods = [m for m in [R0, R1B, "calm", "slice25d", "calm25d"] if m in S.index]
    for i, m in enumerate(mods):
        vals = list(S.loc[m, cols].values) + [T["field"].loc[m, "auc"] if "field" in T and m in T["field"].index else np.nan]
        xx = np.arange(len(cats)) + (i - (len(mods) - 1) / 2) * 0.13
        ax[1].plot(xx, vals, "o", ms=3.2, color=COL[m], label=lab(m), mfc=COL[m] if m == MAIN else "white", mew=0.9)
    ax[1].axvline(len(cols) - 0.5, color="#cccccc", lw=0.6)
    ax[1].set_xticks(range(len(cats))); ax[1].set_xticklabels(cats, fontsize=5.8)
    ax[1].set_ylabel("AUC on the held-out scanner"); ax[1].set_ylim(0.68, 0.9)
    ax[1].legend(fontsize=5.6, ncol=3, loc="lower left", bbox_to_anchor=(-0.02, 1.0), columnspacing=0.8, handletextpad=0.1)
ax[0].text(-0.62, 1.2, "a", transform=ax[0].transAxes, fontsize=8, fontweight="bold", va="bottom")
ax[1].text(-0.12, 1.2, "b", transform=ax[1].transAxes, fontsize=8, fontweight="bold", va="bottom")
save(fig, "fig6_robustness")

# ------------------------------------------------------------------------------------------- Fig 6, single-column version
fig, ax = plt.subplots(2, 1, figsize=(3.45, 3.9), gridspec_kw=dict(height_ratios=[1, 1.05], hspace=0.62))
for j, (s, sl, mk) in enumerate(SCH):
    if s not in T:
        continue
    for i, m in enumerate(deep):
        if m in T[s].index:
            r = T[s].loc[m]; yy = i + (j - 1) * 0.24
            ax[0].errorbar(r[col], yy, xerr=[[r[col] - r[col + "_lo"]], [r[col + "_hi"] - r[col]]], fmt=mk, ms=3,
                           color=COL[m], mfc="white" if j else COL[m], mew=0.8, lw=0.8, capsize=0)
ax[0].axvline(0, color="#777777", lw=0.7, ls=":"); ax[0].set_xticks([-0.05, 0, 0.05, 0.1])
ax[0].set_yticks(range(len(deep))); ax[0].set_yticklabels([lab(m) for m in deep], fontsize=6.8); ax[0].invert_yaxis()
ax[0].set_xlabel("Change in AUC vs clinical + lesion model (95% CI)")
ax[0].legend([Line2D([], [], marker=mk, ls="", color="#444444", mfc="#444444" if j == 0 else "white", ms=3.5) for j, (_, _, mk) in enumerate(SCH)],
             [sl for _, sl, _ in SCH], fontsize=6.2, loc="lower left", bbox_to_anchor=(-0.62, 1.0), ncol=3, handletextpad=0.1, columnspacing=0.7)
if (RES / "table_loso_by_scanner.csv").exists():
    S = pd.read_csv(RES / "table_loso_by_scanner.csv").set_index("model")
    cols = list(S.columns)
    n_sc = L.scanner_group.value_counts()
    cats = [f"{c.replace('Ingenia ', '')}\nn = {n_sc.get(c, 0)}" for c in cols] + ["3 T test\nn = 178"]
    mods = [m for m in [R0, R1B, "calm", "slice25d", "calm25d"] if m in S.index]
    for i, m in enumerate(mods):
        vals = list(S.loc[m, cols].values) + [T["field"].loc[m, "auc"] if "field" in T and m in T["field"].index else np.nan]
        xx = np.arange(len(cats)) + (i - (len(mods) - 1) / 2) * 0.13
        ax[1].plot(xx, vals, "o", ms=3.2, color=COL[m], label=lab(m), mfc=COL[m] if m == MAIN else "white", mew=0.9)
    ax[1].axvline(len(cols) - 0.5, color="#cccccc", lw=0.6)
    ax[1].set_xticks(range(len(cats))); ax[1].set_xticklabels(cats, fontsize=6.4)
    ax[1].set_ylabel("AUC, held-out scanner"); ax[1].set_ylim(0.68, 0.9)
    ax[1].legend(fontsize=6.2, ncol=3, loc="lower left", bbox_to_anchor=(-0.17, 1.0), columnspacing=0.7, handletextpad=0.1)
ax[0].text(-0.62, 1.22, "a", transform=ax[0].transAxes, fontsize=8, fontweight="bold", va="bottom")
ax[1].text(-0.17, 1.22, "b", transform=ax[1].transAxes, fontsize=8, fontweight="bold", va="bottom")
save(fig, "fig6_robustness_col")

# ------------------------------------------------------------------------------------------- Fig 7 subgroups
G = L.loc[ids5]
SG = [("All patients", np.ones(len(G), bool)), ("1.5 T", G.field < 3), ("3 T", G.field >= 3), ("Female", G.sex == "F"),
      ("Male", G.sex == "M"), ("Black", G.race == "b"), ("White", G.race == "w"), ("Age < 80 y", G.age < 80),
      ("Age ≥ 80 y", G.age >= 80), ("NIHSS ≤ 5", G.nihss <= 5), ("NIHSS > 5", G.nihss > 5),
      ("Anterior lesion", G.post_frac <= 0.5), ("Posterior lesion", G.post_frac > 0.5),
      ("Left hemisphere", G.frac_left > 0.5), ("Right hemisphere", G.frac_left <= 0.5),
      ("No expert mask", ~G.has_mask.astype(bool))]
rows = []
for name, msk in SG:
    msk = np.asarray(msk, bool); ii = np.array(ids5)[msk]; yy = L.loc[ii, "y_poor"].values.astype(int)
    if len(ii) < 20 or yy.min() == yy.max():
        continue
    r = dict(subgroup=name, n=len(ii), n_poor=int(yy.sum()))
    for m in [R1B, MAIN]:
        p = ENS["cv5"][m].loc[ii].values; lo, hi = boot_auc_ci(yy, p, B=1000)
        r[m] = auc(yy, p); r[m + "_lo"] = lo; r[m + "_hi"] = hi
    d, (dlo, dhi), _ = paired_boot_delta(yy, ENS["cv5"][MAIN].loc[ii].values, ENS["cv5"][R1B].loc[ii].values, B=1000)
    r.update(delta=d, delta_lo=dlo, delta_hi=dhi); rows.append(r)
SGT = pd.DataFrame(rows); SGT.round(3).to_csv(OUT / "table_subgroups_ci.csv", index=False)
fig, ax = plt.subplots(figsize=(3.45, 3.5))
for i, r in SGT.iterrows():
    for j, m in enumerate([R1B, MAIN]):
        yy = i + (j - 0.5) * 0.3
        ax.errorbar(r[m], yy, xerr=[[r[m] - r[m + "_lo"]], [r[m + "_hi"] - r[m]]], fmt="o", ms=2.8, lw=0.8,
                    color=COL[m], mfc=COL[m] if m == MAIN else "white", mew=0.8, label=lab(m) if i == 0 else None)
ax.set_yticks(range(len(SGT))); ax.set_yticklabels([f"{r.subgroup} ({r.n})" for r in SGT.itertuples()], fontsize=6)
ax.invert_yaxis(); ax.axvline(SGT.loc[0, MAIN], color=COL[MAIN], lw=0.5, ls=":")
ax.set_xlabel("AUC for discharge mRS ≥ 3 (bootstrap 95% CI)"); ax.set_xlim(0.45, 1.0)
ax.legend(fontsize=5.8, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, handletextpad=0.1)
save(fig, "fig7_subgroups")

# ------------------------------------------------------------------------------------------- Fig 8 model comparison
LAD = [m for m in [R0, "baseline_R1_clin_volume", R1B, "baseline_R2b_clin_vol_topography", "resnet3d_noanchor", "resnet3d",
                   "calm", "slice25d", "slice25d_lesanchor", "calm25d_noanchor", "calm25d_noseg", "calm25d_noaux", "calm25d"]
       if m in ENS["cv5"]]
fig, ax = plt.subplots(figsize=(3.45, 2.9))
seedrows = []
for i, m in enumerate(LAD):
    _, dd = complete("cv5", m)
    sa = [auc(L.loc[ids5, "y_poor"].values, d.reindex(ids5)[PC].values) for d in dd.values()]
    for s, a in zip(dd, sa):
        seedrows.append(dict(model=m, seed=s, auc=a))
    c = COL.get(m, "#555555")
    ax.plot(sa, [i] * len(sa), "|", ms=6, color=c, mew=1.0)
    ax.plot(auc(y5, ENS["cv5"][m].loc[ids5].values), i, "D", ms=3.6, color=c, mfc=c if m == MAIN else "white", mew=0.9)
ax.axvline(auc(y5, ENS["cv5"][R1B].loc[ids5].values), color=COL[R1B], lw=0.6, ls=":")
ax.set_yticks(range(len(LAD))); ax.set_yticklabels([lab(m) for m in LAD], fontsize=6); ax.invert_yaxis()
ax.set_xlabel("AUC, 5-fold CV (| single seed, ◇ seed ensemble)")
save(fig, "fig8_ablation")
pd.DataFrame(seedrows).round(4).to_csv(OUT / "table_seed_auc_cv5.csv", index=False)

# ------------------------------------------------------------------------------------------- number macros
mac = ["% generated by src/paper_outputs.py; do not edit by hand",
       "\\providecommand{\\N}[3]{\\ifcsname n@#1@#2@#3\\endcsname\\csname n@#1@#2@#3\\endcsname\\else\\textbf{\\textcolor{red}{??}}\\fi}"]


def put(k, s, metric, v, fmt="{:.3f}"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return
    mac.append(f"\\expandafter\\def\\csname n@{k}@{s}@{metric}\\endcsname{{{fmt.format(v)}}}")


for s, TT in T.items():
    for m, r in TT.iterrows():
        k = KEY.get(m)
        if k is None:
            continue
        for c in ["auc", "auc_lo", "auc_hi", "auprc", "brier", "cal_intercept", "cal_slope", "ordinal_c", "mae_mrs",
                  "seed_auc_mean", "seed_auc_sd"]:
            if c in r and pd.notna(r[c]):
                put(k, s, c.replace("_", ""), float(r[c]), "{:.2f}" if c in ("cal_intercept", "cal_slope", "mae_mrs") else "{:.3f}")
        if "n_seeds" in r and pd.notna(r["n_seeds"]):
            put(k, s, "nseeds", float(r["n_seeds"]), "{:.0f}")
        for ref, rk in [("R0_clinical", "R0"), ("R1b_clin_vol_chronic_radc", "R1b"), ("R2b_clin_vol_topography", "R2b")]:
            base = f"dAUC_vs_{ref}"
            if base in r and pd.notna(r[base]):
                put(k, s, f"d{rk}", float(r[base])); put(k, s, f"d{rk}lo", float(r[base + "_lo"])); put(k, s, f"d{rk}hi", float(r[base + "_hi"]))
                pv = r.get(f"p_one_sided_vs_{ref}")
                if pd.notna(pv):
                    put(k, s, f"p{rk}", max(float(pv), 0.0005) if pv < 0.001 else float(pv), "$<$0.001" if pv < 0.001 else "{:.3f}")
if (RES / "table_loso_by_scanner.csv").exists():
    S = pd.read_csv(RES / "table_loso_by_scanner.csv").set_index("model")
    for m, r in S.iterrows():
        if m in KEY:
            for c, v in r.items():
                put(KEY[m], "loso", c.replace(" ", "").replace("Ingenia", "").lower(), float(v))
            put(KEY[m], "loso", "scmin", float(r.min())); put(KEY[m], "loso", "scmax", float(r.max()))
for r in SGT.itertuples():
    nm = "".join(ch for ch in r.subgroup.lower() if ch.isalnum())
    put("sg", nm, "n", float(r.n), "{:.0f}"); put("sg", nm, "main", getattr(r, MAIN)); put("sg", nm, "ref", getattr(r, R1B))
    put("sg", nm, "d", r.delta); put("sg", nm, "dlo", r.delta_lo); put("sg", nm, "dhi", r.delta_hi)
put("sg", "all", "nbetter", float((SGT[MAIN] >= SGT[R1B]).sum()), "{:.0f}"); put("sg", "all", "ntotal", float(len(SGT)), "{:.0f}")
for t in [0.2, 0.3, 0.4, 0.5]:
    for m in [R0, R1B, "slice25d", MAIN]:
        if m in ENS["cv5"]:
            put(KEY[m], "cv5", f"nb{int(t * 100)}", nb(ENS["cv5"][m].loc[ids5].values, t))
    put("all", "cv5", f"nb{int(t * 100)}", prev - (1 - prev) * t / (1 - t))
mac.append(f"\\def\\mainmodel{{{lab(MAIN)}}}")
(OUT / "numbers.tex").write_text("\n".join(mac) + "\n")

# ------------------------------------------------------------------------------------------- LaTeX tables
def f3(v):
    return "" if pd.isna(v) else f"{v:.3f}"


def tabrow(m, s):
    r = T[s].loc[m]; d = "dAUC_vs_R1b_clin_vol_chronic_radc"
    dd = "ref." if m == R1B else (f"{r[d]:+.3f} ({r[d + '_lo']:+.3f}, {r[d + '_hi']:+.3f})".replace("-", "−") if d in r else "")
    name = lab(m).replace("−", "-")
    if m == MAIN:
        name = "\\textbf{" + name + "}"
    return (f"{name} & {r.auc:.3f} ({r.auc_lo:.3f}, {r.auc_hi:.3f}) & {dd} & {f3(r.auprc)} & {f3(r.brier)} & "
            f"{r.cal_slope:.2f} & {f3(r.ordinal_c)} & {r.mae_mrs:.2f} \\\\").replace("−", "$-$")


order = [R0, "baseline_R1_clin_volume", R1B, "baseline_R2a_clin_vol_territory", "baseline_R2b_clin_vol_topography",
         "baseline_I1_volume_only", "baseline_I2_topography_only", "resnet3d_noanchor", "resnet3d", "calm", "slice25d",
         "slice25d_lesanchor", "calm25d_noanchor", "calm25d_noseg", "calm25d_noaux", "calm25d"]
if "cv5" in T:
    (OUT / "tab_main_cv5.tex").write_text("\n".join(tabrow(m, "cv5") for m in order if m in T["cv5"].index) + "\n")
ext = []
for m in [R0, R1B, "baseline_R2b_clin_vol_topography", "resnet3d", "calm", "slice25d", "slice25d_lesanchor", "calm25d"]:
    cells = [lab(m) if m != MAIN else "\\textbf{" + lab(m) + "}"]
    for s in ["loso", "field"]:
        if s in T and m in T[s].index:
            r = T[s].loc[m]; d = "dAUC_vs_R1b_clin_vol_chronic_radc"
            cells += [f"{r.auc:.3f} ({r.auc_lo:.3f}, {r.auc_hi:.3f})", "ref." if m == R1B else f"{r[d]:+.3f} ({r[d+'_lo']:+.3f}, {r[d+'_hi']:+.3f})", f"{r.cal_slope:.2f}"]
        else:
            cells += ["", "", ""]
    if any(cells[1:]):
        ext.append((" & ".join(cells) + " \\\\").replace("-0.", "$-$0.").replace("+", "$+$"))
(OUT / "tab_external.tex").write_text("\n".join(ext) + "\n")

# cohort table (Table 1)
Lc = L.copy(); good, poor = Lc[Lc.y_poor == 0], Lc[Lc.y_poor == 1]


def med(c, d):
    v = d[c].dropna(); return f"{v.median():.0f} ({v.quantile(.25):.0f}, {v.quantile(.75):.0f})"


def medf(c, d):
    v = d[c].dropna(); return f"{v.median():.1f} ({v.quantile(.25):.1f}, {v.quantile(.75):.1f})"


def npct(msk, d):
    return f"{int(msk.sum())} ({100 * msk.mean():.1f})"


def pcont(c):
    return mannwhitneyu(good[c].dropna(), poor[c].dropna()).pvalue


def pcat(col):
    return chi2_contingency(pd.crosstab(Lc[col], Lc.y_poor))[1]


def pf(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


Lc["race_c"] = Lc.race.fillna("u"); Lc["toast_c"] = Lc.toast.fillna(0)
Lc["mask_c"] = Lc.has_mask.astype(int); Lc["field_c"] = (Lc.field >= 3).astype(int)
Lc["chronic_c"] = (Lc.chronic_ml_filled.fillna(0) >= 1).astype(int)
good, poor = Lc[Lc.y_poor == 0], Lc[Lc.y_poor == 1]
R = [f"Patients, n & {len(Lc)} & {len(good)} & {len(poor)} & \\\\",
     f"Age, years & {med('age', Lc)} & {med('age', good)} & {med('age', poor)} & {pf(pcont('age'))} \\\\",
     f"Female sex & {npct(Lc.sex == 'F', Lc)} & {npct(good.sex == 'F', good)} & {npct(poor.sex == 'F', poor)} & {pf(chi2_contingency(pd.crosstab(Lc.sex, Lc.y_poor))[1])} \\\\",
     f"Race & & & & {pf(pcat('race_c'))} \\\\",
     f"\\quad Black & {npct(Lc.race == 'b', Lc)} & {npct(good.race == 'b', good)} & {npct(poor.race == 'b', poor)} & \\\\",
     f"\\quad White & {npct(Lc.race == 'w', Lc)} & {npct(good.race == 'w', good)} & {npct(poor.race == 'w', poor)} & \\\\",
     f"\\quad other or unknown & {npct(Lc.race.isna(), Lc)} & {npct(good.race.isna(), good)} & {npct(poor.race.isna(), poor)} & \\\\",
     f"Admission NIHSS & {med('nihss', Lc)} & {med('nihss', good)} & {med('nihss', poor)} & {pf(pcont('nihss'))} \\\\",
     f"BMI, kg/m$^2$ & {med('bmi', Lc)} & {med('bmi', good)} & {med('bmi', poor)} & {pf(pcont('bmi'))} \\\\",
     f"Prior stroke & {npct(Lc.priorstroke == 1, Lc)} & {npct(good.priorstroke == 1, good)} & {npct(poor.priorstroke == 1, poor)} & {pf(pcat('priorstroke'))} \\\\"]
TO = {1: "large artery", 2: "cardioembolism", 3: "small-vessel disease", 4: "other cause", 5: "cryptogenic"}
R.append("TOAST subtype & & & & " + pf(pcat("toast_c")) + " \\\\")
for k, v in TO.items():
    R.append(f"\\quad {v} & {npct(Lc.toast == k, Lc)} & {npct(good.toast == k, good)} & {npct(poor.toast == k, poor)} & \\\\")
R += [f"Scanned at 3 T & {npct(Lc.field >= 3, Lc)} & {npct(good.field >= 3, good)} & {npct(poor.field >= 3, poor)} & {pf(pcat('field_c'))} \\\\",
      f"Expert lesion mask & {npct(Lc.has_mask.astype(bool), Lc)} & {npct(good.has_mask.astype(bool), good)} & {npct(poor.has_mask.astype(bool), poor)} & {pf(pcat('mask_c'))} \\\\"]
mk = Lc[Lc.has_mask.astype(bool)]; mg, mp = mk[mk.y_poor == 0], mk[mk.y_poor == 1]
R += [f"Acute lesion, mL$^{{\\dagger}}$ & {medf('acute_ml', mk)} & {medf('acute_ml', mg)} & {medf('acute_ml', mp)} & {pf(mannwhitneyu(mg.acute_ml.dropna(), mp.acute_ml.dropna()).pvalue)} \\\\",
      f"Chronic infarct$^{{\\dagger}}$ & {npct(mk.chronic_ml_filled.fillna(0) >= 1, mk)} & {npct(mg.chronic_ml_filled.fillna(0) >= 1, mg)} & {npct(mp.chronic_ml_filled.fillna(0) >= 1, mp)} & {pf(chi2_contingency(pd.crosstab(mk.chronic_c, mk.y_poor))[1])} \\\\",
      f"Lesion mean rADC$^{{\\dagger}}$ & {mk.acute_radc_mean.median():.2f} ({mk.acute_radc_mean.quantile(.25):.2f}, {mk.acute_radc_mean.quantile(.75):.2f}) & "
      f"{mg.acute_radc_mean.median():.2f} ({mg.acute_radc_mean.quantile(.25):.2f}, {mg.acute_radc_mean.quantile(.75):.2f}) & "
      f"{mp.acute_radc_mean.median():.2f} ({mp.acute_radc_mean.quantile(.25):.2f}, {mp.acute_radc_mean.quantile(.75):.2f}) & {pf(mannwhitneyu(mg.acute_radc_mean.dropna(), mp.acute_radc_mean.dropna()).pvalue)} \\\\"]
(OUT / "tab_cohort.tex").write_text("\n".join(R).replace("%", "\\%") + "\n")

# per-fold / per-seed table (supplementary)
pf_rows = []
for m in LAD:
    _, dd = complete("cv5", m)
    for s, d in dd.items():
        d = d.loc[d.index.intersection(ids5)]
        r = {"model": lab(m), "seed": s}
        for u, g in d.groupby("test_unit"):
            r[u] = auc(L.loc[g.index, "y_poor"].values, g[PC].values)
        pf_rows.append(r)
PF = pd.DataFrame(pf_rows); PF.round(3).to_csv(OUT / "table_perfold_perseed.csv", index=False)
(OUT / "tab_perfold.tex").write_text("\n".join(
    f"{r['model']} & {r['seed'].replace('s', '')} & " + " & ".join(f"{r[c]:.3f}" if c in r and pd.notna(r[c]) else "" for c in [f"fold{k}" for k in range(5)]) + " \\\\"
    for _, r in PF.iterrows()) + "\n")
print("Written to", OUT)
