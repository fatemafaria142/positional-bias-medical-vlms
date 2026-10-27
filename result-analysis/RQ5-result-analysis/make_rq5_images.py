"""RQ5 figures for the paper.
Figure 1 explains the experiment on one organ: (a) the scan, (b) the ground-truth mask, (c) the box cut into 8 blocks, (d) the block that holds most of the organ replaced by noise.
Figure 2 is a chart without images: the Shapley value of the organ block and of the surrounding blocks for every organ.
Figures 3 to 6 show one organ each, for the upper and the lower layer of blocks: (a) the scan, (b) the ground-truth mask, (c) the Shapley value of every block.
A padded box around the left organ is cut into 2 x 2 x 2 = 8 blocks. The block that holds most of the organ has a thick border.
One subject is used, chosen so that all four organs follow the typical pattern of the 50 subjects."""
import base64, zlib, tarfile, os
import numpy as np, pandas as pd, nibabel as nib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(r"D:\Reasoning\results\neurite-oasis\RQ5")
OUT = Path(__file__).parent
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
TAR = Path(r"D:\Reasoning\data\raw\neurite-oasis.v1.0.tar")
SCRATCH = Path(r"C:\Users\FATEMA~1\AppData\Local\Temp\claude\D--Reasoning\3463f333-3745-4839-9350-42eda44b4d8b\scratchpad")
SUBJECT = "OASIS_OAS1_0001_MR1"
ORGANS = {"cerebellum white matter": "F2_cerebellum_white_matter.png", "cerebellum cortex": "F3_cerebellum_cortex.png", "thalamus": "F4_thalamus.png", "hippocampus": "F5_hippocampus.png"}

b = pd.concat([pd.read_csv(ROOT / "rq5_grid_occlusion_shapley.csv"), pd.read_csv(ROOT / "rq5_grid_occlusion_shapley_part3.csv")], ignore_index=True)
b = b[(b.subject_id == SUBJECT) & b.organ.isin(["left " + o for o in ORGANS])]
t1 = pd.read_csv(OUT / "tables" / "T1_shapley_value_per_organ.csv"); t1.columns = ["organ", "organ_block", "others", "diff", "higher", "total"]
t1 = t1[~t1.organ.str.startswith("Mean")]

def dec(enc, shape):
    runs = np.frombuffer(zlib.decompress(base64.b64decode(enc)), dtype=np.uint32)
    flat = np.zeros(int(np.prod(shape)), dtype=bool); pos, val = 0, False
    for r in runs:
        if val: flat[pos:pos + r] = True
        pos += int(r); val = not val
    return flat.reshape(shape)
sl_of = lambda s: tuple(slice(*map(int, p.split(":"))) for p in s.split("x"))

nii = SCRATCH / SUBJECT / "aligned_norm.nii.gz"
if not nii.exists():
    nii.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TAR) as tf:
        for m in tf:
            if m.name == f"{SUBJECT}/aligned_norm.nii.gz":
                m.name = os.path.basename(m.name); tf.extract(m, nii.parent); break
vol = np.flip(np.transpose(np.asarray(nib.load(nii).dataobj).astype(np.float32), (1, 2, 0)), axis=(0, 2))
vmin, vmax = np.percentile(vol[vol > 0], [1, 99.5])
mu, sd = vol[vol > 0].mean(), vol[vol > 0].std()

avail = {x.name for x in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": next(n for n in ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"] if n in avail)})
INK = "#000000"; GREEN = "#2f9e63"; BLOCK = "#ffc400"             # block borders are amber, because black disappears on the dark parts of the scan
BLUE, RED = "#4f97d6", "#d6453d"                       # blue is the organ block and a positive Shapley value, red is the surrounding blocks and a negative one, in every RQ5 figure (green is only the ground-truth mask)
cmap = LinearSegmentedColormap.from_list("blue_red", [RED, "#f3c4c0", "#ffffff", "#cfe3f6", "#8fbfe8", BLUE])
norm = TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=0.3)
halo = [pe.withStroke(linewidth=2.6, foreground="white")]
mask_cmap = matplotlib.colors.ListedColormap([GREEN])

def prepare(o):
    """Geometry of the box, the blocks and the organ for the left organ o."""
    g = b[b.organ == "left " + o].sort_values("block_id")
    shape = tuple(int(x) for x in g.volume_shape.iloc[0].split("x")); gt = dec(g.gt_mask_rle.iloc[0], shape); assert shape == vol.shape
    bb = [sl_of(s) for s in g.block_bbox]; sh = g.shapley_value.values
    fr = np.array([gt[s].sum() for s in bb]) / gt.sum(); top = int(fr.argmax())
    a1 = (min(s[1].start for s in bb), max(s[1].stop for s in bb)); a2 = (min(s[2].start for s in bb), max(s[2].stop for s in bb))
    HA = int(np.ceil((a1[1] - a1[0]) / 2)) + 8; HB = int(np.ceil((a2[1] - a2[0]) / 2)) + 8
    ca, cb = (a1[0] + a1[1]) / 2, (a2[0] + a2[1]) / 2
    a0 = int(np.clip(round(ca) - HA, 0, shape[1] - 2 * HA)); b0 = int(np.clip(round(cb) - HB, 0, shape[2] - 2 * HB))
    z_edges = sorted({s[0].start for s in bb} | {s[0].stop for s in bb}); zmid = z_edges[1]
    layers = []
    for zlo, zhi in ((zmid, z_edges[2]), (z_edges[0], zmid)):                       # upper layer first, then lower layer
        inside = np.argwhere(gt[zlo:zhi].any(axis=(1, 2)))
        layers.append(int(zlo + (inside.mean() if len(inside) else (zhi - zlo) / 2)))
    return dict(gt=gt, bb=bb, sh=sh, top=top, fr=fr, HA=HA, HB=HB, a0=a0, b0=b0, layers=layers)

def slice_view(G, z, v):
    return np.flipud(v[z, G["a0"]:G["a0"] + 2 * G["HA"], G["b0"]:G["b0"] + 2 * G["HB"]])

def draw_blocks(ax_, G, z, values=None):
    """Draw the blocks that lie on slice z. With values the blocks are coloured by Shapley value, the organ block has a thick border."""
    for k, s_ in enumerate(G["bb"]):
        if not (s_[0].start <= z < s_[0].stop): continue
        r0 = (2 * G["HA"] - 1) - (s_[1].stop - 1 - G["a0"]); r1 = (2 * G["HA"] - 1) - (s_[1].start - G["a0"])       # rows after the vertical flip
        c0, c1 = s_[2].start - G["b0"], s_[2].stop - G["b0"]
        thick = k == G["top"]
        if values is None:
            ax_.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0 + 1, fc="none", ec=BLOCK, lw=3.2 if thick else 1.2))
        else:
            v_ = values[k]
            ax_.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0 + 1, fc=cmap(norm(v_)), ec=BLOCK, lw=3.4 if thick else 1.2, alpha=0.75 if thick else 0.62))
            ax_.text((c0 + c1) / 2 - 0.5, (r0 + r1) / 2, f"{v_:+.3f}".replace("-", "\u2212"), ha="center", va="center", fontsize=10.5, color=INK, fontweight="semibold", path_effects=halo)

def overlay_mask(ax_, G, z, alpha=0.7):
    gtv = slice_view(G, z, G["gt"])
    ax_.imshow(np.ma.masked_where(~gtv, np.ones(gtv.shape)), cmap=mask_cmap, alpha=alpha, interpolation="nearest")

def legend_handles(extra=()):
    halo_l = [pe.Stroke(linewidth=4.0, foreground=INK), pe.Normal()]
    return [Patch(facecolor=GREEN, alpha=0.8, edgecolor="none", label="Ground-truth mask of the organ"),
            Line2D([0], [0], color="white", lw=2, path_effects=halo_l, label="Ground-truth organ outline"),
            Patch(facecolor="none", edgecolor=BLOCK, linewidth=3.2, label="Organ block (holds most of the organ)")] + list(extra)

def panel_label(fig, txt, x, y, letter, title):
    txt(x, y, f"({letter}) {title}", ha="left", va="bottom", fontsize=11, color=INK)

# ---------- Figure 1: Shapley value of the organ block and of the other 7 blocks for every organ, no image ----------
def summary_chart():
    t = t1.copy()
    ok = t[t.total > 0.1].sort_values("diff", ascending=False)
    zero = t[t.total <= 0.1].sort_values("diff", ascending=False)
    y = np.concatenate([np.arange(len(ok)), len(ok) + 0.9 + np.arange(len(zero))])
    d = pd.concat([ok, zero]); n = len(d)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 7.6), facecolor="white", gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.06}, sharey=True)
    fig.subplots_adjust(top=0.87, bottom=0.17, left=0.17, right=0.98)
    ax.set_ylim(y[-1] + 0.8, -1.0); ax.set_xlim(-0.04, 0.29); bx.set_xlim(-0.06, 0.26)
    for yi, ob, ot in zip(y, d.organ_block, d.others):
        ax.plot([ob, ot], [yi, yi], color=INK, lw=1.4, zorder=2)
        ax.scatter([ot], [yi], s=95, color=RED, edgecolor=INK, linewidth=0.9, zorder=3)
        ax.scatter([ob], [yi], s=95, color=BLUE, edgecolor=INK, linewidth=0.9, zorder=4)
        lo, hi = (ob, ot) if ob <= ot else (ot, ob)
        ax.text(lo - 0.006, yi, f"{lo:.3f}".replace("-", "\u2212"), ha="right", va="center", fontsize=9.6, color=INK)
        ax.text(hi + 0.006, yi, f"{hi:.3f}".replace("-", "\u2212"), ha="left", va="center", fontsize=9.6, color=INK)
    for yi, df_, hg, tot in zip(y, d["diff"], d.higher, d.total):
        if tot > 0.1:
            bx.barh(yi, df_, height=0.62, color=BLUE if df_ >= 0 else RED, edgecolor=INK, linewidth=0.9, zorder=3)
            lab = f"{df_:+.3f}".replace("-", "\u2212") + f"   ({hg:.0f}%)"
            bx.text(max(df_, 0) + 0.006, yi, lab, ha="left", va="center", fontsize=9.6, color=INK)
        else:
            bx.text(0.006, yi, "cannot be tested", ha="left", va="center", fontsize=9.6, color=INK)
    bx.tick_params(axis="y", length=0, labelleft=False)
    sep = len(ok) + 0.35
    for a_ in (ax, bx):
        a_.axvline(0, color=INK, lw=1.0, zorder=1); a_.axhline(sep, color=INK, lw=0.8, ls=(0, (4, 3)))
        for sp in ("top", "right", "left"): a_.spines[sp].set_visible(False)
        a_.spines["bottom"].set_color(INK); a_.tick_params(axis="x", colors=INK, length=3, labelsize=10)
    ax.text(0.288, sep + 0.18, "Organs with a Shapley value of about 0", ha="right", va="top", fontsize=9.6, color=INK)
    ax.set_yticks(y, list(d.organ), fontsize=10.5, color=INK); ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xticks(np.arange(0, 0.29, 0.05)); ax.set_xticklabels([f"{v:.2f}" for v in np.arange(0, 0.29, 0.05)])
    bx.set_xticks(np.arange(-0.05, 0.26, 0.05)); bx.set_xticklabels([f"{v:.2f}".replace("-", "\u2212") for v in np.arange(-0.05, 0.26, 0.05)])
    ax.set_xlabel("Shapley value of a block in Dice units, mean over the cases", fontsize=10.5, color=INK, labelpad=8)
    bx.set_xlabel("Difference in Shapley value, organ block minus surrounding blocks", fontsize=10.5, color=INK, labelpad=8)
    ax.set_title("(a) Shapley value of the two groups of blocks", fontsize=11, color=INK, loc="left", pad=8)
    bx.set_title("(b) Difference and share of cases where the organ block is higher", fontsize=11, color=INK, loc="left", pad=8)
    fig.text(0.5, 0.975, "Does the Model Rely More on the Block That Holds the Organ Than on the Surrounding Blocks?", ha="center", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.legend(handles=[Line2D([0], [0], marker="o", ls="none", markersize=9.5, markerfacecolor=BLUE, markeredgecolor=INK, label="Organ block, the block that holds most of the organ (panel a), organ block higher (panel b)"),
                        Line2D([0], [0], marker="o", ls="none", markersize=9.5, markerfacecolor=RED, markeredgecolor=INK, label="Surrounding blocks, mean of the other 7 blocks (panel a), surrounding blocks higher (panel b)")],
               loc="center", bbox_to_anchor=(0.55, 0.045), ncol=1, frameon=False, fontsize=10.5, handletextpad=0.6, labelspacing=0.6)
    fig.savefig(FIG / "F1_shapley_value_per_organ.png", dpi=250, bbox_inches="tight", facecolor="white"); plt.close()

# ---------- Figures 3 to 6: one organ each ----------
def organ_figure(o):
    G = prepare(o)
    PH = 1.9; PW = PH * G["HB"] / G["HA"]                      # the panel has the shape of the image, so there is no empty frame
    X0, PITCH, ROWP = 1.0, PW + 0.1, PH + 0.1
    TOPH = 0.55 + 0.25
    LH = 1.3                                                   # height of the legend box
    figw, figh = X0 + 5 * PITCH + 0.05, TOPH + 2 * ROWP + LH - 0.1
    extra = 0.0
    fig = plt.figure(figsize=(figw, figh), facecolor="white")
    txt = lambda x, y, s, **kw: fig.text(x / figw, (figh - y) / figh, s, color=INK, **kw)
    txt(figw / 2, 0.08, f"Which Image Region Does the Model Rely On to Segment the {o.title()}?", ha="center", va="top", fontsize=13.5, fontweight="bold")
    for ci, title in enumerate(("(a) Scan", "(b) Ground-truth\nmask", "(c) Box cut into\n8 blocks", "(d) One block\nreplaced by noise", "(e) Shapley value of\neach block, Dice units")):
        txt(X0 + ci * PITCH + PW / 2, TOPH - 0.08, title, ha="center", va="bottom", fontsize=10.8, linespacing=1.25)
    for ri, (z, lab) in enumerate(zip(G["layers"], ("Upper layer\nof blocks", "Lower layer\nof blocks"))):
        y0 = TOPH + ri * ROWP
        txt(0.05, y0 + PH / 2, lab, ha="left", va="center", fontsize=10.8, linespacing=1.3)
        ks = [k for k, s_ in enumerate(G["bb"]) if s_[0].start <= z < s_[0].stop]                    # the 4 blocks of this layer
        hid = G["top"] if G["top"] in ks else max(ks, key=lambda k: G["fr"][k])                       # block shown as hidden in panel (d)
        vol2 = vol.copy(); s_ = G["bb"][hid]; vol2[s_] = np.random.default_rng(0).normal(mu, sd, vol2[s_].shape)
        for ci in range(5):
            ax_ = fig.add_axes([(X0 + ci * PITCH) / figw, (figh - y0 - PH) / figh, PW / figw, PH / figh]); ax_.axis("off")
            ax_.imshow(slice_view(G, z, vol2 if ci == 3 else vol), cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")
            if ci == 1: overlay_mask(ax_, G, z)
            if ci in (2, 3): draw_blocks(ax_, G, z)
            if ci == 3: ax_.contour(slice_view(G, z, G["gt"]).astype(float), levels=[0.5], colors="white", linewidths=1.3)
            if ci == 4:
                draw_blocks(ax_, G, z, values=G["sh"]); ax_.contour(slice_view(G, z, G["gt"]).astype(float), levels=[0.5], colors="white", linewidths=1.3)
    # vertical colour bar for panel (e)
    cx = X0 + 5 * PITCH - 0.05
    # legend, one compact framed box as wide as the images (same format as the RQ6 figures)
    LX0, LX1 = X0, X0 + 4 * PITCH + PW                                            # left edge of panel (a) to right edge of panel (e)
    lg = fig.add_axes([0, 0, 1, LH / figh]); lg.set_xlim(0, figw); lg.set_ylim(0, LH); lg.axis("off")
    lg.add_patch(Rectangle((LX0, 0.08), LX1 - LX0, LH - 0.16, fc="white", ec=INK, lw=0.5))
    x1 = LX0 + 0.54 * (LX1 - LX0)
    lg.plot([x1, x1], [0.2, LH - 0.2], color=INK, lw=0.4)
    HY = LH - 0.28                                                                  # y of the column headings
    R1, R2, R3 = 0.78, 0.52, 0.26                                                  # y of the three rows of every column
    # column 1, Shapley value colour scale
    lg.text(LX0 + 0.25, HY, "Shapley value of a block, in Dice units", ha="left", va="center", fontsize=9.5, fontweight="bold", color=INK)
    gx0, gx1, ly = LX0 + 0.25, x1 - 0.3, 0.62
    lg.imshow(np.linspace(0, 1, 256)[None, :], extent=(gx0, gx1, ly, ly + 0.2), cmap=cmap, vmin=0, vmax=1, aspect="auto", zorder=3)
    lg.set_xlim(0, figw); lg.set_ylim(0, LH)
    lg.add_patch(Rectangle((gx0, ly), gx1 - gx0, 0.2, fc="none", ec=INK, lw=0.5, zorder=4))
    for v_ in (-0.1, 0, 0.1, 0.2, 0.3):
        xx = gx0 + (gx1 - gx0) * float(norm(v_)); lg.plot([xx] * 2, [ly - 0.05, ly], color=INK, lw=0.5, zorder=4)
        lg.text(xx, ly - 0.18, f"{v_:g}".replace("-", "−"), ha="center", va="center", fontsize=8.5, color=INK)
    lg.text(gx0, 0.2, "Red lowers the Dice score", ha="left", va="center", fontsize=8.5, color=INK)
    lg.text(gx1, 0.2, "Blue raises the Dice score", ha="right", va="center", fontsize=8.5, color=INK)
    # column 2, masks and blocks
    lg.text(x1 + 0.25, HY, "Masks and blocks", ha="left", va="center", fontsize=9.5, fontweight="bold", color=INK)
    lg.add_patch(Rectangle((x1 + 0.25, R1 - 0.09), 0.36, 0.18, fc=GREEN, alpha=0.8, ec="none"))
    lg.text(x1 + 0.75, R1, "True organ mask", ha="left", va="center", fontsize=8.5, color=INK)
    lg.plot([x1 + 0.25, x1 + 0.61], [R2, R2], color="white", lw=2, path_effects=[pe.Stroke(linewidth=4.0, foreground=INK), pe.Normal()])
    lg.text(x1 + 0.75, R2, "True organ outline", ha="left", va="center", fontsize=8.5, color=INK)
    lg.add_patch(Rectangle((x1 + 0.25, R3 - 0.09), 0.36, 0.18, fc="none", ec=BLOCK, lw=2.4))
    lg.text(x1 + 0.75, R3, "Organ block, holds most of the organ", ha="left", va="center", fontsize=8.5, color=INK)
    fig.savefig(FIG / ORGANS[o], dpi=240, bbox_inches="tight", facecolor="white"); plt.close()

summary_chart()
for o in ORGANS: organ_figure(o)
print("done")
