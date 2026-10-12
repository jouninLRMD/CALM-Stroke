"""
Evaluates every out-of-fold prediction file in results/oof.

Outputs (results/):
  table_main_<scheme>.csv     AUC [95% CI], paired difference vs the clinical and lesion reference models, AUPRC, Brier,
                              calibration intercept and slope, ordinal C-index, MAE of the expected mRS
  table_loso_by_scanner.csv   AUC per held-out scanner model
  table_subgroups_cv5.csv     AUC by field strength, sex, race, age >= 80, circulation, hemisphere, mask availability
  fig_roc_cv5.png, fig_calibration_cv5.png, fig_delta_auc_cv5.png
  report.md                   all of the above in one readable file
Seeds of the same model are averaged (ensemble); per-seed AUCs are reported too.
"""
import re
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from common import RES, POOR_K, auc, boot_auc_ci, paired_boot_delta, calibration, harrell_c, expected_mrs

coh = pd.read_csv(RES / "cohort_all.csv")
L = coh[coh.labelled].set_index("participant_id")
try:
    FA = pd.read_csv(RES / "features_atlas.csv").set_index("participant_id")
    L = L.join(FA[["post_frac", "frac_left"]])
except FileNotFoundError:
    L["post_frac"] = np.nan; L["frac_left"] = np.nan

PCOLS = [f"p_gt{k}" for k in range(6)]
preds = {}                                             # (scheme, model) -> {seed: df}
for f in sorted((RES / "oof").glob("*.csv")):
    parts = f.stem.split("__")
    model, scheme = parts[0], parts[1]
    seed = parts[2] if len(parts) > 2 else "s0"
    preds.setdefault((scheme, model), {})[seed] = pd.read_csv(f).set_index("participant_id")

REF = ["baseline_R0_clinical", "baseline_R1b_clin_vol_chronic_radc", "baseline_R2b_clin_vol_topography"]
report = ["# CALM-Stroke results (auto-generated)\n"]


def ens(dd):
    ids = sorted(set.intersection(*[set(d.index) for d in dd.values()]))
    P = np.mean([d.loc[ids, PCOLS].values for d in dd.values()], 0)
    unit = list(dd.values())[0].loc[ids, "test_unit"].values
    return pd.DataFrame(P, index=ids, columns=PCOLS).assign(test_unit=unit)


def metrics(y, mrs, P, ref_P):
    p = P[:, POOR_K]; lo, hi = boot_auc_ci(y, p)
    ci, cs = calibration(y, p)
    r = dict(n=len(y), auc=auc(y, p), auc_lo=lo, auc_hi=hi, auprc=average_precision_score(y, p),
             brier=brier_score_loss(y, p), cal_intercept=ci, cal_slope=cs,
             ordinal_c=harrell_c(mrs, expected_mrs(P)), mae_mrs=float(np.mean(np.abs(expected_mrs(P) - mrs))))
    for name, RP in ref_P.items():
        if RP is not None:
            d, (dlo, dhi), pval = paired_boot_delta(y, p, RP[:, POOR_K])
            short = name.replace("baseline_", "")
            r[f"dAUC_vs_{short}"] = d; r[f"dAUC_vs_{short}_lo"] = dlo; r[f"dAUC_vs_{short}_hi"] = dhi
            r[f"p_one_sided_vs_{short}"] = pval
    return r


for scheme in ["cv5", "loso", "field"]:
    models = {m: ens(dd) for (s, m), dd in preds.items() if s == scheme}
    if not models:
        continue
    common_ids = sorted(set.intersection(*[set(d.index) for d in models.values()]) & set(L.index))
    y = L.loc[common_ids, "y_poor"].values.astype(int); mrs = L.loc[common_ids, "mrs"].values
    refP = {r: (models[r].loc[common_ids, PCOLS].values if r in models else None) for r in REF}
    rows = []
    for m, d in models.items():
        r = metrics(y, mrs, d.loc[common_ids, PCOLS].values, refP); r["model"] = m
        seeds = preds[(scheme, m)]
        if len(seeds) > 1:
            sa = [auc(L.loc[common_ids, "y_poor"].values, s.reindex(common_ids)[f"p_gt{POOR_K}"].values) for s in seeds.values()]
            r["seed_auc_mean"], r["seed_auc_sd"], r["n_seeds"] = np.mean(sa), np.std(sa), len(sa)
        rows.append(r)
    T = pd.DataFrame(rows).set_index("model").sort_values("auc", ascending=False)
    T.to_csv(RES / f"table_main_{scheme}.csv")
    show = T[["n", "auc", "auc_lo", "auc_hi"] + [c for c in T if c.startswith("dAUC_vs_R0") or c.startswith("dAUC_vs_R1b")]
             + ["brier", "cal_slope", "ordinal_c"]].round(3)
    report += [f"\n## Scheme: {scheme} (n = {len(common_ids)} patients, models with complete predictions)\n",
               "```", show.to_string(), "```\n"]

    if scheme == "loso":
        rows = []
        for m, d in models.items():
            for g, dd in d.loc[common_ids].groupby("test_unit"):
                yy = L.loc[dd.index, "y_poor"].values
                if len(set(yy)) > 1:
                    rows.append(dict(model=m, held_out_scanner=g, n=len(dd), auc=auc(yy, dd[f"p_gt{POOR_K}"].values)))
        if rows:
            S = pd.DataFrame(rows).pivot(index="model", columns="held_out_scanner", values="auc").round(3)
            S.to_csv(RES / "table_loso_by_scanner.csv")
            report += ["\n### AUC per held-out scanner\n", "```", S.to_string(), "```\n"]

    if scheme == "cv5":
        G = L.loc[common_ids]
        subgroups = {"1.5 T": G.field < 3, "3 T": G.field >= 3, "female": G.sex == "F", "male": G.sex == "M",
                     "Black": G.race == "b", "White": G.race == "w", "age >= 80": G.age >= 80, "age < 80": G.age < 80,
                     "posterior-predominant": G.post_frac > 0.5, "anterior-predominant": G.post_frac <= 0.5,
                     "left-predominant": G.frac_left > 0.5, "right-predominant": G.frac_left <= 0.5,
                     "no expert mask": ~G.has_mask.astype(bool)}
        rows = []
        for sg, msk in subgroups.items():
            ids = np.array(common_ids)[msk.values]
            yy = L.loc[ids, "y_poor"].values
            if len(ids) < 20 or len(set(yy)) < 2:
                continue
            r = dict(subgroup=sg, n=len(ids), n_poor=int(yy.sum()))
            for m, d in models.items():
                r[m] = auc(yy, d.loc[ids, f"p_gt{POOR_K}"].values)
            rows.append(r)
        if rows:
            SG = pd.DataFrame(rows).set_index("subgroup").round(3)
            SG.to_csv(RES / "table_subgroups_cv5.csv")
            report += ["\n### Subgroup AUC (cv5)\n", "```", SG.to_string(), "```\n"]

        # ---------------- figures
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        key = [m for m in T.index if not m.startswith("baseline_") or m in REF]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for m in key:
            fpr, tpr, _ = roc_curve(y, models[m].loc[common_ids, f"p_gt{POOR_K}"].values)
            ax.plot(fpr, tpr, lw=1.5, label=f"{m.replace('baseline_', '')} ({T.loc[m, 'auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
        ax.set_title("Poor outcome (discharge mRS 3-6), 5-fold CV"); ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout(); fig.savefig(RES / "fig_roc_cv5.png", dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        for m in key:
            p = models[m].loc[common_ids, f"p_gt{POOR_K}"].values
            q = pd.qcut(p, 10, labels=False, duplicates="drop")
            ax.plot([p[q == i].mean() for i in np.unique(q)], [y[q == i].mean() for i in np.unique(q)], "o-", ms=3,
                    lw=1, label=m.replace("baseline_", ""))
        ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlabel("Predicted risk"); ax.set_ylabel("Observed rate")
        ax.set_title("Calibration by decile, 5-fold CV"); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(RES / "fig_calibration_cv5.png", dpi=200); plt.close(fig)

        # ---------------- per-fold and per-seed AUC (supplementary S2)
        rows = []
        for m, dd in preds.items():
            if dd and m[0] == "cv5":
                for sd, d in dd.items():
                    d = d.loc[d.index.intersection(common_ids)]
                    for u, g in d.groupby("test_unit"):
                        yy = L.loc[g.index, "y_poor"].values
                        rows.append(dict(model=m[1], seed=sd, fold=u, auc=auc(yy, g[f"p_gt{POOR_K}"].values)))
        if rows:
            PF = pd.DataFrame(rows).pivot_table(index=["model", "seed"], columns="fold", values="auc").round(3)
            PF.to_csv(RES / "table_perfold_perseed_cv5.csv")
            report += ["\n### AUC per fold and seed (cv5)\n", "```", PF.to_string(), "```\n"]
        # ---------------- decision curve analysis (net benefit)
        thr = np.arange(0.05, 0.81, 0.01); prev = y.mean()
        dca = {"treat all": prev - (1 - prev) * thr / (1 - thr), "treat none": np.zeros_like(thr)}
        for m in key:
            p = models[m].loc[common_ids, f"p_gt{POOR_K}"].values
            dca[m.replace("baseline_", "")] = np.array([((p >= t) & (y == 1)).mean() - ((p >= t) & (y == 0)).mean() * t / (1 - t) for t in thr])
        DC = pd.DataFrame(dca, index=np.round(thr, 2)); DC.index.name = "threshold"
        DC.round(4).to_csv(RES / "table_dca_cv5.csv")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for c in DC.columns:
            ax.plot(DC.index, DC[c], lw=1.2 if c.startswith("treat") else 1.6, ls=":" if c.startswith("treat") else "-", label=c)
        ax.set_ylim(-0.05, max(0.5, prev + 0.02)); ax.set_xlabel("Threshold probability of poor outcome"); ax.set_ylabel("Net benefit")
        ax.set_title("Decision curves, 5-fold CV"); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(RES / "fig_dca_cv5.png", dpi=200); plt.close(fig)
        sel = DC.loc[[0.2, 0.3, 0.4, 0.5, 0.6]].round(3)
        report += ["\n### Net benefit at selected thresholds (cv5)\n", "```", sel.to_string(), "```\n"]

        if "dAUC_vs_R0_clinical" in T:
            D = T.drop(index=[r for r in ["baseline_R0_clinical"] if r in T.index]).sort_values("dAUC_vs_R0_clinical")
            fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(D) + 1.2))
            yy = np.arange(len(D))
            ax.errorbar(D.dAUC_vs_R0_clinical, yy, xerr=[D.dAUC_vs_R0_clinical - D.dAUC_vs_R0_clinical_lo,
                        D.dAUC_vs_R0_clinical_hi - D.dAUC_vs_R0_clinical], fmt="o", color="#2a6fdb", capsize=2)
            ax.axvline(0, color="grey", lw=0.8); ax.set_yticks(yy); ax.set_yticklabels([m.replace("baseline_", "") for m in D.index], fontsize=7)
            ax.set_xlabel("Change in AUC vs clinical model (paired bootstrap 95% CI)"); fig.tight_layout()
            fig.savefig(RES / "fig_delta_auc_cv5.png", dpi=200); plt.close(fig)

(RES / "report.md").write_text("\n".join(report))
print("\n".join(report))
print(f"\nWritten: {RES / 'report.md'} and tables/figures in {RES}")
