"""RQ4 image figures: predicted masks on the MRI for one example subject, two organs per figure, patched layer by layer.
Needs aligned_norm.nii.gz of the subject (extracted from neurite-oasis.v1.0.tar into IMG_DIR)."""
import base64, zlib, tarfile, os
import numpy as np, pandas as pd, nibabel as nib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle
import matplotlib.patheffects as pe

ROOT = Path(r"D:\Reasoning\results\neurite-oasis\RQ4")
TAR = Path(r"D:\Reasoning\data\raw\neurite-oasis.v1.0.tar")
SUBJECT = "OASIS_OAS1_0004_MR1"
IMG_DIR = Path(r"C:\Users\FATEMA~1\AppData\Local\Temp\claude\D--Reasoning\cda2fa20-8216-49b7-9dc5-f417374d9481\scratchpad") / SUBJECT
FIG = Path(__file__).parent / "figures"; FIG.mkdir(exist_ok=True)

# ---------- image, brought into the same frame as the saved masks ----------
nii = IMG_DIR / "aligned_norm.nii.gz"
if not nii.exists():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TAR) as t:
        for m in t:
            if m.name == f"{SUBJECT}/aligned_norm.nii.gz":
                m.name = os.path.basename(m.name); t.extract(m, IMG_DIR); break
arr = np.asarray(nib.load(nii).dataobj).astype(np.float32)
vol = np.flip(np.transpose(arr, (1, 2, 0)), axis=(0, 2))       # axes = superior, anterior, right

def dec(enc, shape):
    runs = np.frombuffer(zlib.decompress(base64.b64decode(enc)), dtype=np.uint32)
    flat = np.zeros(int(np.prod(shape)), dtype=bool); pos, val = 0, False
    for r in runs:
        if val: flat[pos:pos + r] = True
        pos += int(r); val = not val
    return flat.reshape(shape)

# ---------- masks of this subject ----------
rows = []
for ch in pd.read_csv(ROOT / "rq4_activation_patching.csv", chunksize=3000):
    rows.append(ch[ch.subject_id == SUBJECT])
P = pd.concat(rows)
G = pd.read_csv(ROOT / "rq4_ground_truth_masks.csv"); G = G[G.subject_id == SUBJECT]
SHAPE = tuple(int(x) for x in P.volume_shape.iloc[0].split("x"))
assert SHAPE == vol.shape, (SHAPE, vol.shape)

avail = {x.name for x in font_manager.fontManager.ttflist}
FONT = next(n for n in ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"] if n in avail)
plt.rcParams.update({"font.family": FONT})
INK, MUTED = "#000000", "#000000"            # black everywhere, no gray text
GREEN, RED = "#2f9e63", "#d6453d"                             # left = green, right = red, as in Figure 1
COLS = ["Ground truth", "Unpatched"] + [f"Layer {l}" for l in range(6)] + ["Layers 0\u20135", "Whole state"]
HALF = {"cerebral white matter": (62, 80)}                    # larger window for the big organ; others use the default below
HALF_DEFAULT = (48, 62)                                        # window half-size (anterior-posterior, left-right) in voxels

def panels_for(pair, dst):
    q = P[(P.pair == pair) & (P.dst == dst)]
    def pick(variant, layer=None):
        r = q[q.variant == variant]
        if layer is not None: r = r[r.layer == layer]
        r = r.iloc[0]; return dec(r.pred_mask_rle, SHAPE), r
    return ([pick("baseline")] + [pick("single_layer_update", l) for l in range(6)]
            + [pick("updates_from_layer", 0), pick("whole_state_control", 0)])

def rich_prompt(fig, x, y, figh, side, organ):
    """Prompt line: a slim accent bar, a small PROMPT label, then the sentence with the side word coloured like its masks."""
    figw = fig.get_figwidth()
    sidecol = {"left": "#1f7a49", "right": "#b3342d"}[side]
    h = 0.26 / figh
    fig.add_artist(Rectangle((x, y - 0.02 / figh), 0.0032, h, transform=fig.transFigure, fc=sidecol, ec="none"))
    x += 0.0032 + 0.006
    r = fig.canvas.get_renderer()
    parts = [("PROMPT", MUTED, "bold", 9.5, 0.0), ("Segment the", INK, "normal", 13, 0.55), (side, sidecol, "bold", 13, 0.0),
             (organ, INK, "bold", 13, 0.0), ("in this image.", INK, "normal", 13, 0.0)]
    gap = lambda pts: pts / 72 / figw
    for i, (txt, col, wt, fs, extra) in enumerate(parts):
        if i == 1: x += gap(10)
        t = fig.text(x, y, txt, ha="left", va="bottom", fontsize=fs, fontweight=wt, color=col,
                     **({"fontstyle": "normal"}))
        x += t.get_window_extent(r).width / fig.bbox.width + gap(0.55 * fs)

def make(dst, organs, fname, heading):
    src = "right" if dst == "left" else "left"
    n = len(organs)
    ph = 1.62                                                   # panel height in inches
    figh = ph * n + 2.95                                      # extra room for the column titles of each row and the legend
    fig, axes = plt.subplots(n, 10, figsize=(16.4, figh), facecolor="white")
    fig.subplots_adjust(top=1 - 1.3 / figh, bottom=0.95 / figh, left=0.004, right=0.996, hspace=0.98, wspace=0.04)
    vmin, vmax = np.percentile(vol[vol > 0], [1, 99.5])
    for ri, pair in enumerate(organs):
        gt = {s_: dec(G[(G.pair == pair) & (G.side == s_)].gt_mask_rle.iloc[0], SHAPE) for s_ in ("left", "right")}
        pans = panels_for(pair, dst)
        z = int(round(np.argwhere(gt[dst])[:, 0].mean()))
        both = gt["left"] | gt["right"]
        cen = np.argwhere(both[z - 6:z + 7].any(axis=0)).mean(0)
        HALF_A, HALF_B = HALF.get(pair, HALF_DEFAULT)
        a0 = int(np.clip(round(cen[0]) - HALF_A, 0, SHAPE[1] - 2 * HALF_A)); b0 = int(np.clip(round(cen[1]) - HALF_B, 0, SHAPE[2] - 2 * HALF_B))
        sl = (z, slice(a0, a0 + 2 * HALF_A), slice(b0, b0 + 2 * HALF_B))
        cl = np.argwhere(gt["left"])[:, 2].mean(); cr = np.argwhere(gt["right"])[:, 2].mean()
        assert cl < cr                                         # patient left is on the left of the picture
        zone_left = np.zeros(SHAPE, bool); zone_left[:, :, :int(round((cl + cr) / 2))] = True
        view = lambda a: np.flipud(a[sl])
        img = view(vol)
        for ci, (m, r) in enumerate([(None, None)] + list(pans)):
            ax = axes[ri, ci]; ax.axis("off")
            arts = [ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")]
            parts = ((gt["left"], GREEN), (gt["right"], RED)) if m is None else ((m & zone_left, GREEN), (m & ~zone_left, RED))
            for part, colr in parts:
                arts.append(ax.imshow(np.ma.masked_where(~view(part), np.ones_like(img)), cmap=matplotlib.colors.ListedColormap([colr]),
                                      alpha=0.7, interpolation="nearest"))
            for side, ls in ((dst, "solid"), (src, "dashed")):
                arts.append(ax.contour(view(gt[side]).astype(float), levels=[0.5], colors="white", linewidths=1.1, linestyles=ls))
            ax.set_title(COLS[ci], fontsize=12.5, fontweight="normal", color=INK, pad=5)
            if m is None:
                ax.text(0.5, -0.035, "True organs", transform=ax.transAxes, ha="center", va="top", fontsize=11.2,
                        fontweight="semibold", color=INK)
                continue
            dprom = r.dice_left_organ if dst == "left" else r.dice_right_organ
            side = {"left": "left", "right": "right", "empty": "empty", "neither": "neither", "both": "both"}[r.call]
            tcol = {"left": "#1f7a49", "right": "#b3342d"}.get(r.call, INK)
            ax.text(0.5, -0.035, f"Dice {dprom:.2f}  ·  {side}", transform=ax.transAxes, ha="center", va="top", fontsize=11.2,
                    fontweight="semibold", color=tcol)
        pos = axes[ri, 0].get_position()
        rich_prompt(fig, pos.x0 + 0.002, pos.y1 + 0.46 / figh, figh, dst, pair)
    fig.text(0.5, 1 - 0.3 / figh, heading, ha="center", va="bottom", fontsize=14, fontweight="bold", color=INK)
    halo = [pe.Stroke(linewidth=4.2, foreground=INK), pe.Normal()]          # keeps the white outline visible on the white page
    fig.legend(handles=[Patch(color=GREEN, alpha=0.85, label="Mask on the left side"),
                        Patch(color=RED, alpha=0.85, label="Mask on the right side"),
                        Line2D([0], [0], color="white", lw=2, path_effects=halo, label="True organ on the prompted side"),
                        Line2D([0], [0], color="white", lw=2, ls=(0, (2.2, 2.2)), path_effects=[pe.Stroke(linewidth=3.6, foreground=INK), pe.Normal()],
                               label="True organ on the other side")],
               loc="center", bbox_to_anchor=(0.5, 0.28 / figh), ncol=4, frameon=False,
               fontsize=10.5, labelcolor=MUTED, handlelength=2.4, columnspacing=2.4)
    fig.savefig(FIG / fname, dpi=230, bbox_inches="tight", facecolor="white"); plt.close()
    print(fname, organs)

make("left", ["hippocampus", "cerebral white matter"], "F2_images_left_prompt.png", "Does Replacing the Prompt Decoder Layer Updates of a Left Prompt With Those of the Right Prompt Move the Mask to the Right Side?")
make("right", ["hippocampus", "cerebral white matter"], "F3_images_right_prompt.png", "Does Replacing the Prompt Decoder Layer Updates of a Right Prompt With Those of the Left Prompt Move the Mask to the Left Side?")
