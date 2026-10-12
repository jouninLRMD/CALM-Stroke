"""
Building blocks shared by the CALM networks.

  Encoder3D, SegDecoder   3D residual encoder and lesion decoder trained from scratch (used by CALM-3D)
  RegionTokenizer         pools image features over each atlas region and over the predicted lesion inside it
  ClinicalTokenizer       one token per clinical variable, with learned tokens for missing values
  CALMStroke              CALM-3D: 3D encoder + territory tokens + clinical tokens + transformer, with the
                          residual added to the in-fold clinical anchor logits
  ordinal_loss, dice_bce, multitask_loss, p_poor

The main 2.5D model, CALM-Stroke, is defined in train.py (class Calm25D) and reuses the tokenizers and losses.

Inputs of CALMStroke (MNI 2 mm space, padded to 96 x 112 x 96):
  img      : (B, C, D, H, W)   image channels (trace and rADC)
  atlas    : (K, D, H, W)      one-hot region atlas
  clin_num : (B, Nn)           numeric clinical variables, z-scored in fold, NaN = missing
  clin_cat : (B, Nc)           categorical clinical variables as integer codes, -1 = missing
  anchor   : (B, 6)            in-fold clinical ordinal logits, logit P(mRS > k), k = 0..5
Outputs:
  ordinal logits (B, 6) = anchor + residual(image, clinical)
  lesion logits (B, 1, D, H, W), NIHSS (B,) and TOAST logits (B, 5) auxiliary heads
Running this file performs a CPU forward and backward check on synthetic tensors.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- 3D encoder-decoder
class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.InstanceNorm3d(cout, affine=True)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.InstanceNorm3d(cout, affine=True)
        self.skip = (nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.InstanceNorm3d(cout, affine=True))
                     if (stride != 1 or cin != cout) else nn.Identity())

    def forward(self, x):
        h = F.leaky_relu(self.n1(self.c1(x)), 0.01)
        h = self.n2(self.c2(h))
        return F.leaky_relu(h + self.skip(x), 0.01)


class Encoder3D(nn.Module):
    """Residual 3D U-Net encoder; returns multi-scale features."""
    def __init__(self, cin=3, widths=(24, 48, 96, 192)):
        super().__init__()
        self.stem = ResBlock3D(cin, widths[0])
        self.stages = nn.ModuleList([ResBlock3D(widths[i], widths[i + 1], stride=2) for i in range(len(widths) - 1)])

    def forward(self, x):
        feats = [self.stem(x)]
        for s in self.stages:
            feats.append(s(feats[-1]))
        return feats  # resolutions: 1, 1/2, 1/4, 1/8


class SegDecoder(nn.Module):
    def __init__(self, widths=(24, 48, 96, 192)):
        super().__init__()
        self.ups = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.ups.append(nn.ModuleDict(dict(
                up=nn.ConvTranspose3d(widths[i], widths[i - 1], 2, 2),
                blk=ResBlock3D(2 * widths[i - 1], widths[i - 1]))))
        self.out = nn.Conv3d(widths[0], 1, 1)

    def forward(self, feats):
        x = feats[-1]
        for j, m in enumerate(self.ups):
            skip = feats[-2 - j]
            x = m["blk"](torch.cat([m["up"](x), skip], 1))
        return self.out(x)


# ----------------------------------------------------------------------------- tokenizers
class RegionTokenizer(nn.Module):
    """Atlas-anchored tokens: for each region k, pool features over (region) and over (region AND lesion).
    Also exposes the interpretable scalar lesion load per region."""
    def __init__(self, cfeat, d_model, n_regions, centroids):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(2 * cfeat + 2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.region_id = nn.Embedding(n_regions, d_model)
        self.register_buffer("centroids", centroids)            # (K, 3) MNI mm, normalised
        self.pos = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, f, atlas, lesion_prob):
        # f: (B,C,d,h,w); atlas: (K,d,h,w) at same res; lesion_prob: (B,1,d,h,w)
        eps = 1e-6
        R = atlas.unsqueeze(0)                                   # (1,K,d,h,w)
        Rl = R * lesion_prob                                      # (B,K,d,h,w)
        rv = R.sum((2, 3, 4)) + eps                               # (1,K)
        lv = Rl.sum((2, 3, 4))                                    # (B,K)
        f_reg = torch.einsum("bcdhw,kdhw->bkc", f, atlas) / rv.unsqueeze(-1)
        f_les = torch.einsum("bcdhw,bkdhw->bkc", f, Rl) / (lv.unsqueeze(-1) + eps)
        load = lv / rv                                            # fraction of region lesioned
        x = torch.cat([f_reg, f_les, load.unsqueeze(-1), torch.log1p(lv).unsqueeze(-1)], -1)
        K = atlas.shape[0]
        tok = self.proj(x) + self.region_id.weight.unsqueeze(0) + self.pos(self.centroids).unsqueeze(0)
        return tok, load                                          # (B,K,d), (B,K)


class ClinicalTokenizer(nn.Module):
    """FT-Transformer style feature tokens with learned 'missing' embeddings."""
    def __init__(self, n_num, cat_cards, d_model):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_num, d_model) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_num, d_model))
        self.miss_num = nn.Parameter(torch.zeros(n_num, d_model))
        self.cat = nn.ModuleList([nn.Embedding(c + 1, d_model) for c in cat_cards])  # last index = missing

    def forward(self, x_num, x_cat):
        miss = torch.isnan(x_num)
        xn = torch.nan_to_num(x_num, 0.0).unsqueeze(-1)
        tok_num = torch.where(miss.unsqueeze(-1), self.miss_num.unsqueeze(0), xn * self.w + self.b)
        tok_cat = [emb(torch.where(x_cat[:, j] < 0, torch.full_like(x_cat[:, j], emb.num_embeddings - 1), x_cat[:, j]))
                   for j, emb in enumerate(self.cat)]
        return torch.cat([tok_num, torch.stack(tok_cat, 1)], 1) if tok_cat else tok_num


# ----------------------------------------------------------------------------- full model
class CALMStroke(nn.Module):
    def __init__(self, atlas_centroids, n_num=3, cat_cards=(2, 2, 2, 5), d_model=192, n_layers=4, n_heads=6,
                 widths=(24, 48, 96, 192), token_scale=2, p_drop_clin=0.2, p_drop_flair=0.3, n_ord=6, n_toast=5, cin=3):
        super().__init__()
        self.cin = cin
        self.enc = Encoder3D(cin, widths)
        self.dec = SegDecoder(widths)
        self.token_scale = token_scale                            # use features at 1/2**token_scale resolution
        cfeat = widths[token_scale]
        self.regions = RegionTokenizer(cfeat, d_model, atlas_centroids.shape[0], atlas_centroids)
        self.clin = ClinicalTokenizer(n_num, cat_cards, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.glob = nn.Linear(widths[-1], d_model)                # whole-brain context token (atrophy, WMH ...)
        self.type_emb = nn.Embedding(4, d_model)                  # 0 cls, 1 clinical, 2 region, 3 global
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.fusion = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        # Residual ordinal head (CORN/CORAL-style shared direction + per-threshold bias), zero-initialised
        self.res_dir = nn.Linear(d_model, 1)
        self.res_bias = nn.Parameter(torch.zeros(n_ord))
        nn.init.zeros_(self.res_dir.weight); nn.init.zeros_(self.res_dir.bias)
        self.nihss_head = nn.Linear(d_model, 1)
        self.toast_head = nn.Linear(d_model, n_toast)
        self.p_drop_clin, self.p_drop_flair = p_drop_clin, p_drop_flair

    def forward(self, img, atlas, clin_num, clin_cat, anchor, flair_available=None):
        B = img.shape[0]
        if self.training and self.p_drop_flair > 0 and self.cin >= 3:   # modality dropout on FLAIR (channel 3)
            keep = (torch.rand(B, device=img.device) > self.p_drop_flair).float().view(B, 1, 1, 1, 1)
            img = torch.cat([img[:, :2], img[:, 2:3] * keep], 1)
        feats = self.enc(img)
        seg_logits = self.dec(feats)
        s = 2 ** self.token_scale
        les = torch.sigmoid(F.avg_pool3d(seg_logits, s))         # lesion prob at token resolution
        atlas_ds = F.avg_pool3d(atlas.unsqueeze(0).float(), s)[0]
        rtok, load = self.regions(feats[self.token_scale], atlas_ds, les)
        ctok = self.clin(clin_num, clin_cat)
        if self.training and self.p_drop_clin > 0:                # forces imaging tokens to carry signal
            m = (torch.rand(B, 1, 1, device=img.device) > self.p_drop_clin).float()
            ctok = ctok * m
        g = self.glob(feats[-1].mean((2, 3, 4))).unsqueeze(1)
        toks = [self.cls.expand(B, -1, -1) + self.type_emb.weight[0],
                ctok + self.type_emb.weight[1], rtok + self.type_emb.weight[2], g + self.type_emb.weight[3]]
        h = self.norm(self.fusion(torch.cat(toks, 1)))[:, 0]
        residual = self.res_dir(h) + self.res_bias                # (B, n_ord)
        ord_logits = anchor + residual                            # clinical anchor + learned residual
        return dict(ord_logits=ord_logits, residual=residual, seg_logits=seg_logits, region_load=load,
                    nihss=self.nihss_head(h).squeeze(-1), toast=self.toast_head(h))


# ----------------------------------------------------------------------------- losses
def ordinal_loss(ord_logits, mrs):
    """Cumulative-link BCE: target_k = 1[mRS > k]. mrs: (B,) long, -1 = missing."""
    valid = mrs >= 0
    if valid.sum() == 0:
        return ord_logits.sum() * 0
    k = torch.arange(ord_logits.shape[1], device=mrs.device)
    tgt = (mrs[valid].unsqueeze(1) > k).float()
    return F.binary_cross_entropy_with_logits(ord_logits[valid], tgt)


def dice_bce(seg_logits, mask, has_mask):
    if has_mask.sum() == 0:
        return seg_logits.sum() * 0
    lg, m = seg_logits[has_mask], mask[has_mask]
    p = torch.sigmoid(lg)
    inter = (p * m).sum((1, 2, 3, 4)); den = p.sum((1, 2, 3, 4)) + m.sum((1, 2, 3, 4))
    return (1 - (2 * inter + 1) / (den + 1)).mean() + F.binary_cross_entropy_with_logits(lg, m)


def multitask_loss(out, batch, w=dict(ord=1.0, seg=1.0, nihss=0.3, toast=0.2, res_l2=1e-3)):
    L = w["ord"] * ordinal_loss(out["ord_logits"], batch["mrs"])
    L = L + w["seg"] * dice_bce(out["seg_logits"], batch["mask"], batch["has_mask"])
    nv = ~torch.isnan(batch["nihss"])
    if nv.any():
        L = L + w["nihss"] * F.huber_loss(out["nihss"][nv], batch["nihss"][nv])
    tv = batch["toast"] >= 0
    if tv.any():
        L = L + w["toast"] * F.cross_entropy(out["toast"][tv], batch["toast"][tv])
    return L + w["res_l2"] * (out["residual"] ** 2).mean()   # keep the imaging residual honest


def p_poor(ord_logits):
    """P(mRS >= 3) = P(mRS > 2) = sigmoid(logit_k=2)."""
    return torch.sigmoid(ord_logits[:, 2])


# ----------------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    torch.manual_seed(0)
    D, H, W, K = 48, 56, 48, 34                               # small grid for the CPU test (real: 96x112x96)
    lab = torch.randint(0, K, (D, H, W))
    atlas = F.one_hot(lab, K).permute(3, 0, 1, 2).float()
    cent = torch.randn(K, 3) * 0.5
    model = CALMStroke(cent)
    nparam = sum(p.numel() for p in model.parameters())
    B = 2
    batch = dict(img=torch.randn(B, 3, D, H, W), clin_num=torch.tensor([[0.3, 1.2, float("nan")], [-0.5, -0.8, 0.1]]),
                 clin_cat=torch.tensor([[1, 0, 0, 2], [0, 1, -1, -1]]), anchor=torch.randn(B, 6).sort(1, descending=True).values,
                 mrs=torch.tensor([4, -1]), mask=(torch.rand(B, 1, D, H, W) > 0.98).float(),
                 has_mask=torch.tensor([True, False]), nihss=torch.tensor([1.1, float("nan")]), toast=torch.tensor([2, -1]))
    model.eval()
    with torch.no_grad():
        out = model(batch["img"], atlas, batch["clin_num"], batch["clin_cat"], batch["anchor"])
    # zero-initialised residual => at initialisation predictions equal the clinical anchor exactly
    assert torch.allclose(out["ord_logits"], batch["anchor"]), "anchor property violated"
    model.train()
    out = model(batch["img"], atlas, batch["clin_num"], batch["clin_cat"], batch["anchor"])
    loss = multitask_loss(out, batch); loss.backward()
    print(f"params {nparam/1e6:.2f} M | loss {loss.item():.4f} | ord {tuple(out['ord_logits'].shape)} "
          f"| seg {tuple(out['seg_logits'].shape)} | region_load {tuple(out['region_load'].shape)} | P(poor) {p_poor(out['ord_logits']).detach().numpy().round(3)}")
    print("OK: forward/backward pass, anchor-at-init property holds, missing labels/inputs handled.")
