"""
Trains the networks on the GPU and writes out-of-fold predictions.

Models (--model):
  resnet3d  3D ResNet-18 trained from scratch on trace and rADC
  slice25d  2.5D MIL: ImageNet ResNet-18 per axial slice + gated attention pooling over slices
  calm      CALM-3D: lesion-aware 3D encoder trained from scratch + territory tokens + clinical tokens, multitask
  calm25d   CALM-Stroke: pretrained 2.5D slice encoder + lesion head + territory tokens + transformer, multitask
All models output ordinal logits P(mRS > k), k = 0..5. With the clinical anchor (default) the network learns a
residual on top of an in-fold clinical logistic model; --no-anchor gives the image-only version.

Validation (--scheme): cv5 (frozen folds) | loso (leave one scanner model out) | field (train 1.5 T, test 3 T)
Outputs: results/oof/<model>[_variant]__<scheme>__s<seed>.csv, results/logs/..., weights in <work>/models

Examples:
  python src/train.py --model calm25d --scheme cv5 --quick         # two-minute smoke test
  python src/train.py --model calm25d --scheme cv5 --seed 0
  python src/train.py --model slice25d --scheme loso --seed 1
"""
import argparse, math, time, json
try:                      # Windows: load Pillow's DLLs before torch, otherwise torchvision's PIL import can fail
    from PIL import Image  # noqa: F401  ("DLL load failed while importing _imaging")
except Exception:
    pass
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from common import RES, WORK, N_ORD, fit_anchor, auc
from calm_stroke_model import CALMStroke, multitask_loss, ordinal_loss, RegionTokenizer, ClinicalTokenizer

NUM = ["age", "nihss", "bmi"]                    # numeric clinical tokens (z-scored in fold)
CAT = ["sex_m", "race_b", "priorstroke"]         # categorical clinical tokens (TOAST is a target, not an input)


# ----------------------------------------------------------------------------------------------- data
class SoopDS(Dataset):
    def __init__(self, cache, rows, tab):
        self.cache, self.rows, self.tab = cache, np.asarray(rows), tab
        self.X = self.M = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.X is None:                           # open lazily (one memmap per worker process)
            self.X = np.load(self.cache / "X.npy", mmap_mode="r"); self.M = np.load(self.cache / "M.npy", mmap_mode="r")
        r = int(self.rows[i]); t = self.tab
        return dict(x=torch.from_numpy(np.asarray(self.X[r], dtype=np.float32)),
                    mask=torch.from_numpy(np.asarray(self.M[r], dtype=np.float32)),
                    has_mask=torch.tensor(bool(t.has_mask[r])), mrs=torch.tensor(int(t.mrs_i[r])),
                    nihss=torch.tensor(float(t.nihss_z[r])), toast=torch.tensor(int(t.toast_i[r])),
                    anchor=torch.from_numpy(t.anchor[r].astype(np.float32)),
                    clin_num=torch.from_numpy(t.clin_num[r].astype(np.float32)),
                    clin_cat=torch.from_numpy(t.clin_cat[r].astype(np.int64)), row=torch.tensor(r))


class Tab:  # per-split tabular tensors aligned to cache rows
    pass


ANCHOR_SETS = {
    "clin": ["age", "nihss", "priorstroke", "bmi"],
    # clinical + expert-lesion descriptors (volume, chronic lesion, lesion rADC; mask indicator)
    "clin_lesion": ["age", "nihss", "priorstroke", "bmi", "log_acute_ml", "has_mask_f", "log_chronic_ml", "acute_radc_mean"],
}


def build_tab(df, train_mask, use_anchor, anchor_vars="clin"):
    t = Tab(); n = len(df)
    tr = df[train_mask]
    t.has_mask = df.has_mask_cache.values.astype(bool)
    t.mrs_i = np.where(df.mrs.notna() & df.labelled, df.mrs.fillna(-1), -1).astype(int)
    mu, sd = tr.nihss.mean(), tr.nihss.std() + 1e-6
    t.nihss_z = ((df.nihss - mu) / sd).values.astype(np.float32)          # NaN = missing (masked in loss)
    t.toast_i = np.where(df.toast.notna(), df.toast.fillna(1) - 1, -1).astype(int)
    t.clin_num = np.stack([((df[c] - tr[c].mean()) / (tr[c].std() + 1e-6)).values for c in NUM], 1).astype(np.float32)
    t.clin_cat = np.stack([np.where(df[c].notna(), df[c].fillna(0), -1) for c in CAT], 1).astype(np.int64)
    trl = tr[tr.labelled]
    if use_anchor:
        t.anchor = fit_anchor(trl, df, cols=ANCHOR_SETS[anchor_vars])
    else:                                         # constant prior logits per threshold
        prev = np.array([(trl.mrs > k).mean() for k in range(N_ORD)]).clip(1e-3, 1 - 1e-3)
        t.anchor = np.tile(np.log(prev / (1 - prev)), (n, 1)).astype(np.float32)
    return t


# ----------------------------------------------------------------------------------------------- models
class ResNet3DOrd(nn.Module):
    def __init__(self):
        super().__init__()
        from monai.networks.nets import resnet18
        self.net = resnet18(spatial_dims=3, n_input_channels=2, num_classes=1, conv1_t_stride=2)
        nf = self.net.fc.in_features; self.net.fc = nn.Identity()
        self.head = nn.Linear(nf, 1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.bias = nn.Parameter(torch.zeros(N_ORD))

    def forward(self, b, atlas=None):
        r = self.head(self.net(b["x"])) + self.bias
        return dict(ord_logits=b["anchor"] + r, residual=r)


class Slice25D(nn.Module):
    def __init__(self, z_idx):
        super().__init__()
        import torchvision
        try:
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:                      # offline fallback
            print("ImageNet weights unavailable, training from scratch:", e); m = torchvision.models.resnet18()
        m.fc = nn.Identity(); self.enc = m; self.z = z_idx
        self.to3 = nn.Conv2d(2, 3, 1)
        with torch.no_grad():
            self.to3.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [.5, .5]]).view(3, 2, 1, 1)); self.to3.bias.zero_()
        self.att_v = nn.Linear(512, 128); self.att_u = nn.Linear(512, 128); self.att_w = nn.Linear(128, 1)
        self.head = nn.Linear(512, 1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.bias = nn.Parameter(torch.zeros(N_ORD))

    def forward(self, b, atlas=None):
        x = b["x"][:, :, :, :, self.z]                          # (B, 2, D, H, S) axial slices
        B, C, D, H, S = x.shape
        s = x.permute(0, 4, 1, 2, 3).reshape(B * S, C, D, H)
        f = self.enc(self.to3(s)).view(B, S, -1)
        a = self.att_w(torch.tanh(self.att_v(f)) * torch.sigmoid(self.att_u(f)))   # gated attention MIL
        h = (torch.softmax(a, 1) * f).sum(1)
        r = self.head(h) + self.bias
        return dict(ord_logits=b["anchor"] + r, residual=r, slice_attention=a.squeeze(-1))


class Calm25D(nn.Module):
    """CALM-Stroke v2: Clinically Anchored, Lesion-aware, Multi-task network on a pretrained 2.5D backbone.

    Per axial slice, an ImageNet ResNet-18 gives mid-level maps (layer2 + upsampled layer3, 384 ch at 1/8 resolution)
    and a slice vector (layer4). A light head on the mid-level maps predicts the acute lesion (supervised by expert
    masks). Stacking slices gives a feature volume on which arterial-territory tokens are pooled over each territory
    and over its predicted lesion (plus lesion load). A gated-attention summary of the slice vectors gives a global
    token. Territory, clinical and global tokens go through a transformer; the output is a zero-initialised residual
    added to the in-fold clinical anchor. Auxiliary heads: lesion segmentation, NIHSS, TOAST."""
    def __init__(self, z_idx, atlas_tok, centroids, d=192, n_layers=4, heads=6, p_drop_clin=0.2):
        super().__init__()
        import torchvision
        try:
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            print("ImageNet weights unavailable, training from scratch:", e); m = torchvision.models.resnet18()
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.l2, self.l3, self.l4 = m.layer2, m.layer3, m.layer4
        self.z = z_idx
        self.to3 = nn.Conv2d(2, 3, 1)
        with torch.no_grad():
            self.to3.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [.5, .5]]).view(3, 2, 1, 1)); self.to3.bias.zero_()
        self.seg = nn.Sequential(nn.Conv2d(384, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.register_buffer("atlas_tok", atlas_tok)             # (K, h, w, S) territory fractions at token grid
        self.regions = RegionTokenizer(384, d, atlas_tok.shape[0], centroids)
        self.clin = ClinicalTokenizer(len(NUM), (2, 2, 2), d)
        self.att_v = nn.Linear(512, 128); self.att_u = nn.Linear(512, 128); self.att_w = nn.Linear(128, 1)
        self.glob = nn.Linear(512, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d)); self.type_emb = nn.Embedding(4, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.fusion = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(d)
        self.res_head = nn.Linear(d, 1); self.mil_head = nn.Linear(512, 1)
        for h in (self.res_head, self.mil_head):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
        self.res_bias = nn.Parameter(torch.zeros(N_ORD))
        self.nihss_head = nn.Linear(d, 1); self.toast_head = nn.Linear(d, 5)
        self.p_drop_clin = p_drop_clin

    def backbone_params(self):
        return [p for mod in (self.to3, self.stem, self.l2, self.l3, self.l4) for p in mod.parameters()]

    def forward(self, b, atlas=None):
        x = b["x"][:, :, :, :, self.z]; B, C, D, H, S = x.shape
        s = x.permute(0, 4, 1, 2, 3).reshape(B * S, C, D, H)
        f1 = self.stem(self.to3(s)); f2 = self.l2(f1); f3 = self.l3(f2); f4 = self.l4(f3)
        h8, w8 = f2.shape[-2:]
        mid = torch.cat([f2, F.interpolate(f3, size=(h8, w8), mode="bilinear", align_corners=False)], 1)   # (B*S,384,h,w)
        seg = self.seg(mid)                                                                               # (B*S,1,h,w)
        vec = f4.mean((2, 3)).view(B, S, -1)                                                              # slice vectors
        a = torch.softmax(self.att_w(torch.tanh(self.att_v(vec)) * torch.sigmoid(self.att_u(vec))), 1)
        g = (a * vec).sum(1)                                                                              # (B,512)
        vol = mid.view(B, S, 384, h8, w8).permute(0, 2, 3, 4, 1)                                         # (B,384,h,w,S)
        les = torch.sigmoid(seg.float()).view(B, S, 1, h8, w8).permute(0, 2, 3, 4, 1)                    # (B,1,h,w,S)
        rtok, load = self.regions(vol.float(), self.atlas_tok, les)
        ctok = self.clin(b["clin_num"], b["clin_cat"])
        if self.training and self.p_drop_clin > 0:
            ctok = ctok * (torch.rand(B, 1, 1, device=ctok.device) > self.p_drop_clin).float()
        toks = torch.cat([self.cls.expand(B, -1, -1) + self.type_emb.weight[0], ctok + self.type_emb.weight[1],
                          rtok + self.type_emb.weight[2], self.glob(g).unsqueeze(1) + self.type_emb.weight[3]], 1)
        hcls = self.norm(self.fusion(toks))[:, 0]
        residual = self.res_head(hcls) + self.mil_head(g) + self.res_bias
        # segmentation target: expert mask on the same slices, averaged to the token grid (soft target)
        m = b["mask"][:, :, :, :, self.z].permute(0, 4, 1, 2, 3).reshape(B * S, 1, D, H)
        tgt = F.adaptive_avg_pool2d(m, (h8, w8))
        seg_logits = seg.view(B, S, 1, h8, w8).permute(0, 2, 3, 4, 1)
        seg_target = tgt.view(B, S, 1, h8, w8).permute(0, 2, 3, 4, 1)
        return dict(ord_logits=b["anchor"] + residual, residual=residual, seg_logits=seg_logits, seg_target=seg_target,
                    region_load=load, slice_attention=a.squeeze(-1), nihss=self.nihss_head(hcls).squeeze(-1),
                    toast=self.toast_head(hcls))


class CalmWrap(nn.Module):
    def __init__(self, centroids):
        super().__init__()
        self.m = CALMStroke(centroids, n_num=len(NUM), cat_cards=(2, 2, 2), cin=2)

    def forward(self, b, atlas):
        return self.m(b["x"], atlas, b["clin_num"], b["clin_cat"], b["anchor"])


# ----------------------------------------------------------------------------------------------- augmentation
def augment(x, mask, strength):
    B = x.shape[0]; dev = x.device
    ang = (torch.rand(B, 3, device=dev) * 2 - 1) * math.radians(10 * strength)
    sc = 1 + (torch.rand(B, 1, device=dev) * 2 - 1) * 0.08 * strength
    tr = (torch.rand(B, 3, device=dev) * 2 - 1) * 0.05 * strength
    cx, cy, cz = torch.cos(ang).unbind(1); sx, sy, sz = torch.sin(ang).unbind(1)
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], 1).view(B, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], 1).view(B, 3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], 1).view(B, 3, 3)
    A = torch.cat([(Rz @ Ry @ Rx) * sc.view(B, 1, 1), tr.view(B, 3, 1)], 2)
    g = F.affine_grid(A, x.shape, align_corners=False)
    x = F.grid_sample(x, g, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask = (F.grid_sample(mask, g, mode="bilinear", align_corners=False) > 0.5).float()
    x = x * (1 + 0.05 * strength * torch.randn(B, x.shape[1], 1, 1, 1, device=dev)) \
        + 0.05 * strength * torch.randn(B, x.shape[1], 1, 1, 1, device=dev)
    return x + 0.02 * strength * torch.randn_like(x), mask


# ----------------------------------------------------------------------------------------------- loops
def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def predict(model, loader, dev, atlas, amp):
    model.eval(); P, rows, losses = [], [], []
    for b in loader:
        b = to_dev(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(b, atlas)
        lg = out["ord_logits"].float()
        losses.append(ordinal_loss(lg, b["mrs"]).item() * len(lg))
        P.append(torch.sigmoid(lg).cpu().numpy()); rows.append(b["row"].cpu().numpy())
    return np.concatenate(P), np.concatenate(rows), float(np.sum(losses) / max(1, sum(len(r) for r in rows)))


def units(df, scheme):
    L = df[df.labelled]
    if scheme == "cv5":
        return [(f"fold{k}", (df.fold == k).values) for k in range(5)]
    if scheme == "loso":
        vc = L.scanner_group.value_counts()
        return [(str(g), (df.labelled & (df.scanner_group == g)).values) for g in vc[vc >= 30].index]
    if scheme == "field":
        return [("train1.5T_test3T", (df.labelled & (df.field >= 3)).values)]
    raise ValueError(scheme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet3d", "slice25d", "calm", "calm25d"], required=True)
    ap.add_argument("--scheme", choices=["cv5", "loso", "field"], default="cv5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--no-seg", action="store_true", help="CALM ablation: no lesion-segmentation supervision")
    ap.add_argument("--no-aux", action="store_true", help="CALM ablation: no NIHSS/TOAST heads and no unlabelled scans")
    ap.add_argument("--anchor-vars", choices=["clin", "clin_lesion"], default="clin",
                    help="clin_lesion: anchor also uses expert-mask lesion volume, chronic lesion and lesion rADC")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--epoch-size", type=int, default=400)
    ap.add_argument("--batch", type=int, default=0, help="0 = model default")
    ap.add_argument("--lr", type=float, default=0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--units", nargs="*", help="only these test units, e.g. fold0")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 epochs, 16 samples, first unit")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.epoch_size, a.patience = 2, 16, 2
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = dev.type == "cuda"
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device {dev} {torch.cuda.get_device_name(0) if amp else ''}", flush=True)

    cache = WORK / "cache"
    idx = pd.read_csv(cache / "index.csv")
    coh = pd.read_csv(RES / "cohort_all.csv")
    df = idx.merge(coh, on="participant_id", how="left").rename(columns={"has_mask_x": "has_mask_cache"})
    if "has_mask_cache" not in df:
        df["has_mask_cache"] = df["has_mask"]
    df = df.sort_values("row").reset_index(drop=True)
    df["has_mask_f"] = df["has_mask_cache"].astype(float)
    df["log_chronic_ml"] = np.log1p(df["chronic_ml_filled"].astype(float))

    defaults = dict(resnet3d=(4, 3e-4), slice25d=(8, 1e-4), calm=(2, 2e-4), calm25d=(8, 1e-4))
    bs, lr = defaults[a.model]; bs = a.batch or bs; lr = a.lr or lr
    tag = (f"{a.model}{'_noanchor' if a.no_anchor else ''}{'_lesanchor' if a.anchor_vars == 'clin_lesion' else ''}"
           f"{'_noseg' if a.no_seg else ''}{'_noaux' if a.no_aux else ''}__{a.scheme}__s{a.seed}")
    (RES / "oof").mkdir(parents=True, exist_ok=True); (RES / "logs").mkdir(parents=True, exist_ok=True)
    (WORK / "models").mkdir(parents=True, exist_ok=True)

    atlas = None
    if a.model in ("calm", "calm25d"):
        A = np.load(cache / "atlas.npy"); labs = np.load(cache / "atlas_labels.npy")
        atlas = torch.from_numpy(np.stack([(A == k) for k in labs]).astype(np.float32)).to(dev)
        cent = torch.from_numpy(np.load(cache / "centroids.npy").astype(np.float32))

    U = units(df, a.scheme)
    if a.units:
        U = [u for u in U if u[0] in a.units]
    if a.quick:
        U = U[:1]
    all_oof, log = [], []
    for unit, test_mask in U:
        t0 = time.time()
        # training pool: labelled outside the test unit; CALM also uses unlabelled scans (aux tasks),
        # excluding scans from a held-out scanner / field strength (no peeking at the target domain)
        lab_tr = df.labelled.values & ~test_mask
        if a.scheme == "loso":
            dom_ok = (df.scanner_group != unit).values
        elif a.scheme == "field":
            dom_ok = (df.field < 3).values
        else:
            dom_ok = np.ones(len(df), bool)
        unl_tr = ((~df.labelled.values) & dom_ok & ~test_mask) if (a.model in ("calm", "calm25d") and not a.no_aux) \
            else np.zeros(len(df), bool)
        rng = np.random.default_rng(a.seed)
        lab_rows = np.where(lab_tr)[0]
        ypo = df.y_poor.values[lab_rows]
        val_rows = np.concatenate([rng.choice(lab_rows[ypo == c], int(round(0.15 * (ypo == c).sum())), replace=False) for c in (0, 1)])
        train_lab = np.setdiff1d(lab_rows, val_rows)
        train_rows = np.concatenate([train_lab, np.where(unl_tr)[0]])
        fit_mask = np.zeros(len(df), bool); fit_mask[train_lab] = True
        tab = build_tab(df, fit_mask, use_anchor=not a.no_anchor, anchor_vars=a.anchor_vars)
        w = np.where(np.isin(train_rows, train_lab), 1.0, 0.6 * len(train_lab) / max(1, unl_tr.sum()) if unl_tr.sum() else 0)
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), a.epoch_size, replacement=True)
        kw = dict(num_workers=a.workers, pin_memory=amp, persistent_workers=a.workers > 0)
        tr_dl = DataLoader(SoopDS(cache, train_rows, tab), batch_size=bs, sampler=sampler, drop_last=True, **kw)
        va_dl = DataLoader(SoopDS(cache, val_rows, tab), batch_size=bs, shuffle=False, **kw)
        te_rows = np.where(test_mask)[0]
        te_dl = DataLoader(SoopDS(cache, te_rows, tab), batch_size=bs, shuffle=False, **kw)

        if a.model == "resnet3d":
            model = ResNet3DOrd()
        elif a.model == "slice25d":
            nz = np.load(cache / "X.npy", mmap_mode="r").shape[-1]
            model = Slice25D(sorted(set(np.linspace(0.17 * nz, 0.85 * nz, 17).astype(int).tolist())))
        elif a.model == "calm25d":
            nz = np.load(cache / "X.npy", mmap_mode="r").shape[-1]
            z_idx = sorted(set(np.linspace(0.17 * nz, 0.85 * nz, 17).astype(int).tolist()))
            At = atlas[:, :, :, z_idx].permute(3, 0, 1, 2)                       # (S, K, D, H)
            h8, w8 = -(-atlas.shape[1] // 8), -(-atlas.shape[2] // 8)
            At = F.adaptive_avg_pool2d(At, (h8, w8)).permute(1, 2, 3, 0).contiguous()   # (K, h, w, S)
            model = Calm25D(z_idx, At.cpu(), cent)
        else:
            model = CalmWrap(cent)
        model.to(dev)
        if a.model == "calm25d":                     # pretrained backbone at lr, new modules at 3 x lr
            bb = set(id(p) for p in model.backbone_params())
            groups = [{"params": [p for p in model.parameters() if id(p) in bb], "lr": lr},
                      {"params": [p for p in model.parameters() if id(p) not in bb], "lr": 3 * lr}]
            max_lr = [lr, 3 * lr]
        else:
            groups, max_lr = model.parameters(), lr
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=1e-2)
        steps = a.epochs * (a.epoch_size // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=max(steps, 1), pct_start=0.1)
        best, best_state, bad = 1e9, None, 0
        print(f"\n[{tag}] unit {unit}: train {len(train_lab)} labelled + {int(unl_tr.sum())} unlabelled, "
              f"val {len(val_rows)}, test {len(te_rows)}", flush=True)
        for ep in range(a.epochs):
            model.train(); tl, n = 0.0, 0; te0 = time.time()
            for b in tr_dl:
                b = to_dev(b, dev)
                b["x"], b["mask"] = augment(b["x"], b["mask"], 0.5 if a.model in ("calm", "calm25d") else 1.0)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    out = model(b, atlas)
                out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
                if a.model == "calm25d":
                    w = dict(ord=1.0, seg=0.0 if a.no_seg else 1.0, nihss=0.0 if a.no_aux else 0.3,
                             toast=0.0 if a.no_aux else 0.2, res_l2=1e-3)
                    loss = multitask_loss(out, dict(b, mask=out["seg_target"]), w)
                elif a.model == "calm":
                    loss = multitask_loss(out, b)
                else:
                    loss = ordinal_loss(out["ord_logits"], b["mrs"]) + 1e-3 * (out["residual"] ** 2).mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
                tl += loss.item() * len(b["x"]); n += len(b["x"])
            Pv, rv, vloss = predict(model, va_dl, dev, atlas, amp)
            yv = df.y_poor.values[rv]
            vauc = auc(yv, Pv[:, 2]) if len(set(yv)) > 1 else float("nan")
            log.append(dict(unit=unit, epoch=ep, train_loss=tl / max(n, 1), val_ord_loss=vloss, val_auc=vauc,
                            sec=round(time.time() - te0, 1)))
            print(f"  ep {ep:3d} train {tl/max(n,1):.4f} | val ord {vloss:.4f} AUC {vauc:.3f} | {time.time()-te0:.0f}s", flush=True)
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        model.load_state_dict(best_state)
        torch.save(best_state, WORK / "models" / f"{tag}__{unit}.pt")
        P, rows, _ = predict(model, te_dl, dev, atlas, amp)
        o = pd.DataFrame(P, columns=[f"p_gt{k}" for k in range(N_ORD)])
        o.insert(0, "test_unit", unit); o.insert(0, "participant_id", df.participant_id.values[rows])
        all_oof.append(o)
        yt = df.y_poor.values[rows]
        print(f"[{tag}] unit {unit}: test AUC {auc(yt, P[:, 2]):.3f} (n={len(rows)}) | {(time.time()-t0)/60:.1f} min", flush=True)
        pd.concat(all_oof).to_csv(RES / "oof" / f"{tag}.csv", index=False)
        pd.DataFrame(log).to_csv(RES / "logs" / f"{tag}.csv", index=False)
    print(f"\nSaved {RES / 'oof' / (tag + '.csv')}")


if __name__ == "__main__":
    main()
