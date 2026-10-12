"""
What do the lesion-aware parts of CALM-Stroke learn? (run after all CALM-Stroke models are trained)

For the CALM-Stroke models of seed 0 (5-fold CV), on each held-out fold:
  * predicted lesion map vs expert mask on the token grid (soft Dice, lesion load correlation)
  * predicted territory lesion load vs the load computed from the expert mask with the same atlas tokens
  * territory attribution: change of the image residual r_2 (logit of poor outcome) when one territory token is
    replaced by its training-set mean (occlusion), for every territory
  * slice attention weights
  * a check that the reloaded models reproduce the saved out-of-fold predictions
Then, on the CPU, the late-fusion comparator (out-of-fold logit of the image-only 2.5D network stacked with the clinical
variables in an in-fold logistic model) is fitted if results/oof/slice25d_noanchor__cv5__s*.csv exist.
Outputs: results/explain/ (per-patient csv, summary csv, fig9_explain.pdf/png, numbers_explain.tex)
Usage: python src/explain.py [--folds 0 1 2 3 4] [--seed 0]
"""
import argparse, logging
try:
    from PIL import Image  # noqa: F401
except Exception:
    pass
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from common import RES, WORK, RESOURCES, auc, boot_auc_ci, paired_boot_delta, clinical_matrix
from train import Calm25D, SoopDS, build_tab, units

OUT = RES / "explain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = dev.type == "cuda"
    cache = WORK / "cache"
    idx = pd.read_csv(cache / "index.csv"); coh = pd.read_csv(RES / "cohort_all.csv")
    df = idx.merge(coh, on="participant_id", how="left").rename(columns={"has_mask_x": "has_mask_cache"})
    if "has_mask_cache" not in df:
        df["has_mask_cache"] = df["has_mask"]
    df = df.sort_values("row").reset_index(drop=True)
    df["has_mask_f"] = df["has_mask_cache"].astype(float); df["log_chronic_ml"] = np.log1p(df["chronic_ml_filled"].astype(float))
    A = np.load(cache / "atlas.npy"); labs = np.load(cache / "atlas_labels.npy")
    atlas = torch.from_numpy(np.stack([(A == k) for k in labs]).astype(np.float32)).to(dev)
    cent = torch.from_numpy(np.load(cache / "centroids.npy").astype(np.float32))
    nz = np.load(cache / "X.npy", mmap_mode="r").shape[-1]
    z_idx = sorted(set(np.linspace(0.17 * nz, 0.85 * nz, 17).astype(int).tolist()))
    At = atlas[:, :, :, z_idx].permute(3, 0, 1, 2)
    h8, w8 = -(-atlas.shape[1] // 8), -(-atlas.shape[2] // 8)
    At = F.adaptive_avg_pool2d(At, (h8, w8)).permute(1, 2, 3, 0).contiguous()     # (K, h, w, S)
    names = []
    for line in open(RESOURCES / "ArterialAtlas136.txt"):
        if line.strip():
            c = line.strip().split("|"); names.append((int(c[0]), c[1], c[2]))
    nm = {k: n for k, n, _ in names}; full = {k: f for k, _, f in names}
    oof = pd.read_csv(RES / "oof" / f"calm25d__cv5__s{a.seed}.csv").set_index("participant_id")
    rows, attr_rows = [], []
    for unit, test_mask in units(df, "cv5"):
        k = int(unit.replace("fold", ""))
        if k not in a.folds:
            continue
        lab_tr = df.labelled.values & ~test_mask
        rng = np.random.default_rng(a.seed)
        lab_rows = np.where(lab_tr)[0]; ypo = df.y_poor.values[lab_rows]
        val_rows = np.concatenate([rng.choice(lab_rows[ypo == c], int(round(0.15 * (ypo == c).sum())), replace=False) for c in (0, 1)])
        train_lab = np.setdiff1d(lab_rows, val_rows)
        fit_mask = np.zeros(len(df), bool); fit_mask[train_lab] = True
        tab = build_tab(df, fit_mask, use_anchor=True)
        model = Calm25D(z_idx, At.cpu(), cent).to(dev)
        model.load_state_dict(torch.load(WORK / "models" / f"calm25d__cv5__s{a.seed}__{unit}.pt", map_location=dev)); model.eval()
        captured = {}

        def grab(mod, inp, out):
            captured["tok"] = out[0].detach()
        h = model.regions.register_forward_hook(grab)
        # mean territory tokens on (a subsample of) training patients, for occlusion
        tr_dl = DataLoader(SoopDS(cache, train_lab[:160], tab), batch_size=8, num_workers=a.workers)
        toks = []
        with torch.no_grad():
            for b in tr_dl:
                b = {q: v.to(dev) for q, v in b.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    model(b, atlas)
                toks.append(captured["tok"].float().mean(0))
        h.remove(); mean_tok = torch.stack(toks).mean(0)                                   # (K, d)
        te_rows = np.where(test_mask)[0]
        te_dl = DataLoader(SoopDS(cache, te_rows, tab), batch_size=8, num_workers=a.workers)
        for b in te_dl:
            b = {q: v.to(dev) for q, v in b.items()}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(b, atlas)
            res = out["residual"].float()[:, 2]; p = torch.sigmoid(out["ord_logits"].float()[:, 2])
            segp = torch.sigmoid(out["seg_logits"].float()); tgt = out["seg_target"].float()
            inter = (segp * tgt).sum((1, 2, 3, 4)); dice = (2 * inter + 1e-6) / (segp.sum((1, 2, 3, 4)) + tgt.sum((1, 2, 3, 4)) + 1e-6)
            loadp = out["region_load"].float()
            R = At.to(dev).unsqueeze(0)
            loade = (R * tgt).sum((2, 3, 4)) / (R.sum((2, 3, 4)) + 1e-6)                    # expert load, same tokens
            K = loadp.shape[1]; delta = torch.zeros(len(res), K, device=dev)
            for kk in range(K):
                def occl(mod, inp, o, kk=kk):
                    t = o[0].clone(); t[:, kk] = mean_tok[kk].to(t.dtype); return (t, o[1])
                hh = model.regions.register_forward_hook(occl)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    r2 = model(b, atlas)["residual"].float()[:, 2]
                hh.remove(); delta[:, kk] = res - r2                                           # contribution of territory kk
            att = out["slice_attention"].float()
            for i, r in enumerate(b["row"].cpu().numpy()):
                pid = df.participant_id.values[r]
                d = dict(participant_id=pid, fold=k, p_poor=float(p[i]), p_saved=float(oof.loc[pid, "p_gt2"]) if pid in oof.index else np.nan,
                         residual=float(res[i]), has_mask=bool(df.has_mask_cache.values[r]), dice_tokengrid=float(dice[i]),
                         pred_load_total=float(segp[i].mean()), expert_load_total=float(tgt[i].mean()),
                         att_entropy=float(-(att[i] * att[i].clamp_min(1e-9).log()).sum()), att_argmax=int(att[i].argmax()))
                rows.append(d)
                for kk in range(K):
                    lab_k = int(labs[kk])
                    attr_rows.append(dict(participant_id=pid, region=nm.get(lab_k, str(lab_k)), region_id=lab_k,
                                          load_pred=float(loadp[i, kk]), load_expert=float(loade[i, kk]), attribution=float(delta[i, kk])))
        print(f"[explain] {unit}: {len(te_rows)} patients", flush=True)
    P = pd.DataFrame(rows); AT = pd.DataFrame(attr_rows)
    P.to_csv(OUT / "per_patient.csv", index=False); AT.to_csv(OUT / "per_region.csv", index=False)
    summarize(P, AT, coh, full)


def summarize(P=None, AT=None, coh=None, full=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 7, "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    P = pd.read_csv(OUT / "per_patient.csv") if P is None else P
    AT = pd.read_csv(OUT / "per_region.csv") if AT is None else AT
    coh = pd.read_csv(RES / "cohort_all.csv") if coh is None else coh
    C = coh.set_index("participant_id")
    P = P.join(C[["y_poor", "acute_ml", "labelled"]], on="participant_id")
    mac = ["% generated by src/explain.py",
           "\\providecommand{\\NE}[1]{\\ifcsname ne@#1\\endcsname\\csname ne@#1\\endcsname\\else\\textbf{\\textcolor{red}{??}}\\fi}"]
    put = lambda key, v, f="{:.2f}": mac.append(f"\\expandafter\\def\\csname ne@{key}\\endcsname{{{f.format(v)}}}")
    ok = P.p_saved.notna()
    put("maxdiff", float((P.p_poor - P.p_saved)[ok].abs().max()), "{:.4f}")
    M = P[P.has_mask.astype(bool)]
    put("dice", float(M.dice_tokengrid.median())); put("diceiqrlo", float(M.dice_tokengrid.quantile(.25))); put("diceiqrhi", float(M.dice_tokengrid.quantile(.75)))
    put("loadrho", float(spearmanr(M.pred_load_total, M.expert_load_total).correlation))
    Mv = M[M.acute_ml.notna()]
    put("volrho", float(spearmanr(Mv.pred_load_total, Mv.acute_ml).correlation))
    small = Mv.acute_ml < 2
    put("nsmall", float(small.sum()), "{:.0f}"); put("dicezerosmall", 100 * float((Mv.dice_tokengrid[small] < 0.05).mean()), "{:.0f}")
    put("dicezerolarge", 100 * float((Mv.dice_tokengrid[~small] < 0.05).mean()), "{:.0f}"); put("dicelarge", float(Mv.dice_tokengrid[~small].median()))
    put("nmask", float(len(M)), "{:.0f}")
    ATm = AT[AT.participant_id.isin(M.participant_id)]
    rr = ATm.groupby("region").apply(lambda g: spearmanr(g.load_pred, g.load_expert).correlation if g.load_expert.std() > 0 else np.nan, include_groups=False)
    put("regrhomed", float(rr.median()))
    # attribution by region and outcome
    ATj = AT.join(C[["y_poor"]], on="participant_id")
    ATj = ATj[ATj.y_poor.notna()]
    agg = ATj.groupby("region").agg(mean_attr=("attribution", "mean"), mean_abs=("attribution", lambda v: np.abs(v).mean()),
                                    load=("load_expert", "mean")).sort_values("mean_abs", ascending=False)
    agg.to_csv(OUT / "attribution_by_region.csv")
    # attribution vs lesion load within region (does a lesioned territory push toward poor outcome?)
    ATj["lesioned"] = ATj.load_expert > 0.05
    les = ATj.groupby("lesioned").attribution.mean()
    put("attrles", float(les.get(True, np.nan)), "{:.4f}"); put("attrnoles", float(les.get(False, np.nan)), "{:.4f}")
    put("attrabsmax", float(agg.mean_abs.max()), "{:.4f}")
    rho_res = spearmanr(P.residual, P.acute_ml.fillna(0)).correlation; put("resvolrho", float(rho_res))
    put("ressd", float(P.residual.std()))
    # ---- figure: (a) lesion map vs expert, (b) Dice by lesion size, (c) image correction vs lesion volume
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.25), gridspec_kw=dict(wspace=0.42))
    ax[0].scatter(M.expert_load_total, M.pred_load_total, s=3, alpha=0.4, color="#d95f02", lw=0)
    mx = max(M.expert_load_total.max(), M.pred_load_total.max()) * 1.05
    ax[0].plot([0, mx], [0, mx], ls=":", color="#777777", lw=0.7)
    ax[0].set(xlabel="Expert lesion fraction (17 slices)", ylabel="Predicted lesion fraction")
    ax[0].set_xscale("symlog", linthresh=1e-3); ax[0].set_yscale("symlog", linthresh=1e-3)
    bins = np.linspace(0, 0.85, 18)
    ax[1].hist([Mv.dice_tokengrid[~small], Mv.dice_tokengrid[small]], bins=bins, stacked=True, color=["#1f78b4", "#a6cee3"],
               label=[f"lesion 2 mL or more (n = {int((~small).sum())})", f"lesion below 2 mL (n = {int(small.sum())})"])
    ax[1].set(xlabel="Soft Dice on the token grid", ylabel="Patients"); ax[1].legend(fontsize=5.8, loc="upper center")
    Q = P[P.y_poor.notna()]
    vol = Q.acute_ml.fillna(0) + 0.1
    for yv, colr, lab_ in [(0, "#1f78b4", "mRS 0 to 2"), (1, "#c62828", "mRS 3 to 6")]:
        q = Q.y_poor == yv
        ax[2].scatter(vol[q], Q.residual[q], s=3, alpha=0.45, color=colr, lw=0, label=lab_)
    ax[2].set_xscale("log"); ax[2].axhline(0, color="#777777", lw=0.6, ls=":")
    ax[2].set(xlabel="Expert acute lesion volume + 0.1 (mL)", ylabel="Image correction $r_2$ (logit)")
    ax[2].legend(fontsize=5.8, loc="upper left", markerscale=2, handletextpad=0.1)
    for a_, s_ in zip(ax, "abc"):
        a_.text(-0.25, 1.04, s_, transform=a_.transAxes, fontsize=8, fontweight="bold")
    fig.savefig(OUT / "fig9_explain.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT / "fig9_explain.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    # ---- late-fusion stacking baseline (image-only network + clinical variables, in fold)
    fs = sorted((RES / "oof").glob("slice25d_noanchor__cv5__s*.csv"))
    if fs:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        L = coh[coh.labelled].set_index("participant_id")
        img = np.mean([pd.read_csv(f).set_index("participant_id").loc[L.index, "p_gt2"].values for f in fs], 0)
        lg = np.log(np.clip(img, 1e-4, 1 - 1e-4) / np.clip(1 - img, 1e-4, 1))
        y = L.y_poor.values.astype(int); pst = np.zeros(len(L))
        for k in range(5):
            tr, te = (L.fold != k).values, (L.fold == k).values
            Xtr = np.c_[clinical_matrix(L[tr], L[tr]), lg[tr]]; Xte = np.c_[clinical_matrix(L[te], L[tr]), lg[te]]
            sc = StandardScaler().fit(Xtr); m = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), y[tr])
            pst[te] = m.predict_proba(sc.transform(Xte))[:, 1]
        r1b = pd.read_csv(RES / "oof" / "baseline_R1b_clin_vol_chronic_radc__cv5.csv").set_index("participant_id").loc[L.index, "p_gt2"].values
        put("imgonly", auc(y, img), "{:.3f}"); put("stack", auc(y, pst), "{:.3f}")
        lo, hi = boot_auc_ci(y, pst); put("stacklo", lo, "{:.3f}"); put("stackhi", hi, "{:.3f}")
        d, (dl, dh), _ = paired_boot_delta(y, pst, r1b); put("stackdr1b", d, "{:+.3f}"); put("stackdr1blo", dl, "{:+.3f}"); put("stackdr1bhi", dh, "{:+.3f}")
        cs = sorted((RES / "oof").glob("calm25d__cv5__s*.csv"))
        if cs:
            calm = np.mean([pd.read_csv(f).set_index("participant_id").loc[L.index, "p_gt2"].values for f in cs], 0)
            d, (dl, dh), _ = paired_boot_delta(y, calm, pst); put("calmvsstack", d, "{:+.3f}"); put("calmvsstacklo", dl, "{:+.3f}"); put("calmvsstackhi", dh, "{:+.3f}")
        pd.DataFrame(dict(participant_id=L.index, p_stack=pst, p_imgonly=img)).to_csv(OUT / "stacking_oof.csv", index=False)
        # table rows for the image-only network and the late-fusion model (Table II of the paper)
        from sklearn.metrics import average_precision_score, brier_score_loss
        from common import calibration, harrell_c, expected_mrs
        cols = [f"p_gt{k}" for k in range(6)]
        Pimg = np.mean([pd.read_csv(f).set_index("participant_id").loc[L.index, cols].values for f in fs], 0)
        mrs = L.mrs.values
        rows = []
        for name, pr, ordinal in [("2.5D network, image only", img, True), ("Late fusion (image only + clinical)", pst, False)]:
            lo, hi = boot_auc_ci(y, pr); d, (dl, dh), _ = paired_boot_delta(y, pr, r1b)
            sgn = lambda v: f"{v:+.3f}".replace("-", "$-$")
            oc = f"{harrell_c(mrs, expected_mrs(Pimg)):.3f}" if ordinal else "--"
            mae = f"{np.mean(np.abs(expected_mrs(Pimg) - mrs)):.2f}" if ordinal else "--"
            rows.append(f"{name} & {auc(y, pr):.3f} ({lo:.3f}, {hi:.3f}) & {sgn(d)} ({sgn(dl)}, {sgn(dh)}) & "
                        f"{average_precision_score(y, pr):.3f} & {brier_score_loss(y, pr):.3f} & "
                        f"{calibration(y, pr)[1]:.2f} & {oc} & {mae} \\\\")
        (OUT / "tab_latefusion.tex").write_text("\n".join(rows) + "\n")
    (OUT / "numbers_explain.tex").write_text("\n".join(mac) + "\n")
    print((OUT / "numbers_explain.tex").read_text())


if __name__ == "__main__":
    import sys
    if "--summary-only" in sys.argv:
        summarize()
    else:
        main()
