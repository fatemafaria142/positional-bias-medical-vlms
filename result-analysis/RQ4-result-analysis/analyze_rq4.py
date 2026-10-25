"""RQ4 result analysis (activation patching, layer-wise): two tables + Figure 1 from results/neurite-oasis/RQ4.
The image overlays are made by make_rq4_images.py."""
import base64, zlib
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle

ROOT = Path(r"D:\Reasoning\results\neurite-oasis\RQ4")
OUT = Path(__file__).parent
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

d = pd.read_csv(ROOT / "rq4_activation_patching.csv", usecols=lambda c: c != "pred_mask_rle")   # 50 subjects
# size of each true organ, to turn "share of the organ covered" into "voxels of the mask inside the organ"
G = pd.read_csv(ROOT / "rq4_ground_truth_masks.csv")
def nvox(enc):
    runs = np.frombuffer(zlib.decompress(base64.b64decode(enc)), dtype=np.uint32); return int(runs[1::2].sum())
G["n"] = G.gt_mask_rle.map(nvox)
sizes = G.pivot_table(index=["subject_id", "pair"], columns="side", values="n").rename(columns={"left": "gl", "right": "gr"})
d = d.join(sizes, on=["subject_id", "pair"])
key = ["subject_id", "pair", "dst"]
base = d[d.variant == "baseline"].set_index(key)
base["dice_dst"] = np.where(base.index.get_level_values("dst") == "left", base.dice_left_organ, base.dice_right_organ)
base["ok"] = base.call == base.index.get_level_values("dst")          # unpatched prediction lands on the prompted side

def prep(variant):
    v = d[d.variant == variant].copy()
    v = v.join(base[["ok", "dice_dst"]].add_prefix("b_"), on=key)
    v["dice_dst"] = np.where(v.dst == "left", v.dice_left_organ, v.dice_right_organ)
    v["dice_src"] = np.where(v.dst == "left", v.dice_right_organ, v.dice_left_organ)
    v["flip"] = v.moved_to_source_side.astype(str).str.lower().eq("true")
    return v[v.b_ok]                                                   # only cases where a flip is meaningful

single, whole, upfrom, bl = prep("single_layer_update"), prep("whole_state_control"), prep("updates_from_layer"), prep("baseline")
n_ok = int(base.ok.sum())
LAYERS = list(range(6))
print("baseline-correct cases:", n_ok, "of", len(base))

pm = lambda x, k=3: f"{x.mean():.{k}f} \u00b1 {x.std():.{k}f}"
SETS = [("None (unpatched)", bl)] + [(f"Layer {l} only", single[single.layer == l]) for l in LAYERS] + [
    ("Layers 3\u20135 together", upfrom[upfrom.layer == 3]), ("Layers 0\u20135 together", upfrom[upfrom.layer == 0]),
    ("Whole state at Layer 0", whole[whole.layer == 0]), ("Whole state at Layer 5", whole[whole.layer == 5])]

# ---------- Table 1, effect on the side and on the Dice ----------
rows = []
for name, x in SETS:
    stay = (x.call == x.dst).mean(); mv = x.flip.mean()
    rows.append({"Layers whose update is replaced": name, "Mask moves to the other side": f"{mv:.1%}",
                 "Mask stays on the prompted side": f"{stay:.1%}",
                 "Mask on both sides, on neither side or empty": f"{1 - mv - stay:.1%}",
                 "Dice of the prompted side": pm(x.dice_dst), "Dice of the other side": pm(x.dice_src)})
t1 = pd.DataFrame(rows); t1.to_csv(TAB / "T1_layerwise_patching.csv", index=False, encoding="utf-8-sig")

# ---------- Table 2, where the predicted mask lies (shares that add up to 100%) ----------
def share(x):
    """Mean share of the predicted mask lying on the left organ, on the right organ and outside both (non-empty masks)."""
    y = x[x.n_pred_voxels > 0]
    L = (y.cov_left_organ * y.gl / y.n_pred_voxels).clip(upper=1).mean()
    R = (y.cov_right_organ * y.gr / y.n_pred_voxels).clip(upper=1).mean()
    return L, R, 1 - L - R, len(x) - len(y)
rows = []; SH = {}
for name, x in SETS:
    r = {"Layers whose update is replaced": name}
    for dst, lab in (("left", "Left prompts"), ("right", "Right prompts")):
        y = x[x.dst == dst]; L, R, O, ne = share(y); SH[(name, dst)] = (L, R, O, ne)
        r[f"{lab}, mask on the left organ"] = f"{L:.1%}"; r[f"{lab}, mask on the right organ"] = f"{R:.1%}"
        r[f"{lab}, mask outside both organs"] = f"{O:.1%}"; r[f"{lab}, mask moves to the other side"] = f"{y.flip.mean():.1%}"
    rows.append(r)
t2 = pd.DataFrame(rows); t2.to_csv(TAB / "T2_where_the_mask_lies.csv", index=False, encoding="utf-8-sig")
print(t1.to_string(index=False)); print(t2.to_string(index=False))
print("empty masks excluded from the shares:", {k[0] + " / " + k[1]: v[3] for k, v in SH.items() if v[3]})

# ---------- shared style ----------
avail = {x.name for x in font_manager.fontManager.ttflist}
FONT = next(n for n in ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"] if n in avail)
plt.rcParams.update({"font.family": FONT})
INK, MUTED, LINE = "#000000", "#000000", "#000000"   # black everywhere, no gray text or lines
GREEN, RED, GREEN_T, RED_T = "#2f9e63", "#d6453d", "#1f7a49", "#b3342d"   # left = green, right = red
SWITCH, OUTSIDE = "#5f6b7a", "#b9c1cc"                                      # gray fills (modern look) for the replaced-layer squares and the outside-both-organs segment

# ---------- Figure 1, one example (the hippocampus of one subject, the same as in the image figures) ----------
SUBJ, ORG = "OASIS_OAS1_0004_MR1", "hippocampus"
def ex_share(variant, layer, dst):
    x = d[(d.subject_id == SUBJ) & (d.pair == ORG) & (d.dst == dst) & (d.variant == variant)]
    if layer is not None: x = x[x.layer == layer]
    assert len(x) == 1
    return share(x)
ROWS = [("Unpatched", "None (unpatched)", ())] + [(f"Layer {l}", f"Layer {l} only", (l,)) for l in LAYERS] + [
    ("Layers 3\u20135", "Layers 3\u20135 together", (3, 4, 5)), ("Layers 0\u20135", "Layers 0\u20135 together", tuple(LAYERS))]
SPEC = {"Unpatched": ("baseline", None), "Layers 3–5": ("updates_from_layer", 3), "Layers 0–5": ("updates_from_layer", 0),
        **{f"Layer {l}": ("single_layer_update", l) for l in LAYERS}}
EX = {(nm, dst): ex_share(*SPEC[nm], dst) for nm, _, _ in ROWS for dst in ("left", "right")}
print({k: tuple(round(v, 3) for v in vals) for k, vals in EX.items()})
ypos = [0.0] + [1.75 + 1.3 * i for i in range(6)] + [1.75 + 1.3 * 5 + 1.75, 1.75 + 1.3 * 5 + 3.05]
FW, FH = 12.4, 8.8
fig = plt.figure(figsize=(FW, FH), facecolor="white")
gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 2.2, 2.2], wspace=0.05, left=0.015, right=0.985, top=0.83, bottom=0.135)
axl, axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
for ax in (axl, axa, axb):
    ax.set_ylim(max(ypos) + 0.9, -1.5)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
# the layer boxes are exact squares and as tall as the bars, so every row lines up
pos = axl.get_position(); w_in, h_in = pos.width * FW, pos.height * FH      # size of the left column in inches
span = max(ypos) + 0.9 + 1.5                                              # y range of the axes in data units
gap_in, x0 = 0.075, 0.46                                                   # space between boxes and where the first box starts
side_in = min(0.84 * h_in / span, ((1 - x0) * w_in - 5 * gap_in) / 6)     # side of one box in inches
sqw, sqg = side_in / w_in, (side_in + gap_in) / w_in                       # box width and box step in axes units
sqh = side_in * span / h_in                                                # box height in data units (equal to its width on the page)
# left column, condition names and which layers are replaced
axl.set_xlim(0, 1); axl.axis("off")
axl.text(x0 + (6 * sqg - gap_in / w_in) / 2, -1.05, "Layers replaced", ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
for y, (nm, _, on) in zip(ypos, ROWS):
    axl.text(x0 - 0.04, y, nm, ha="right", va="center", fontsize=12, color=INK)
    for l in LAYERS:
        is_on = l in on
        axl.add_patch(Rectangle((x0 + l * sqg, y - sqh / 2), sqw, sqh, fc=SWITCH if is_on else "white", ec=SWITCH, lw=0.9))
        axl.text(x0 + l * sqg + sqw / 2, y, str(l), ha="center", va="center", fontsize=10, color="white" if is_on else INK)
# two panels of stacked bars, each row adds up to 100%, with the same height as the boxes
H = sqh
for ax, dst, head in ((axa, "left", "Left prompt"), (axb, "right", "Right prompt")):  # one example prompt per panel
    ax.set_xlim(0, 1.3); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"]); ax.set_yticks([])
    ax.spines["bottom"].set_color(LINE); ax.tick_params(axis="x", labelsize=10.5, colors=INK, length=3)
    ax.set_title(head, fontsize=13, fontweight="bold", color=INK, pad=26)
    ax.text(0.385, 1.012, f'"Segment the {dst} hippocampus in this image."', transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5, color=INK)
    for y, (nm, key_, on) in zip(ypos, ROWS):
        L, R, O, ne = EX[(nm, dst)]; left = 0.0
        for val, col, tcol in ((L, GREEN, "white"), (R, RED, "white"), (O, OUTSIDE, INK)):
            ax.barh(y, val, left=left, height=H, color=col, edgecolor="white", linewidth=0.6, zorder=3)
            if col == OUTSIDE:                                   # the share outside both organs is written to the right of the bar
                ax.text(1.03, y, f"{val:.0%}", ha="left", va="center", fontsize=10.5, color=INK, zorder=4)
            elif val >= 0.09:
                ax.text(left + val / 2, y, f"{val:.0%}", ha="center", va="center", fontsize=10.5, fontweight="bold", color=tcol, zorder=4)
            left += val
axa.set_xlabel("Share of the predicted mask (one subject)", fontsize=11, color=INK, labelpad=6)
axb.set_xlabel("Share of the predicted mask (one subject)", fontsize=11, color=INK, labelpad=6)
fig.text(0.5, 0.955, "Where the Predicted Mask Lies in One Example When One Prompt Decoder Layer's Update Is Replaced With the Other Prompt's Update",
         ha="center", va="bottom", fontsize=13.5, fontweight="bold", color=INK)
fig.legend(handles=[Patch(facecolor=GREEN, edgecolor="white", label="On the left organ"), Patch(facecolor=RED, edgecolor="white", label="On the right organ"),
                    Patch(facecolor=OUTSIDE, edgecolor="white", label="Outside both organs")],
           loc="center", bbox_to_anchor=(0.5, 0.03), ncol=3, frameon=False, fontsize=11, handlelength=1.4, columnspacing=2.4)
fig.savefig(FIG / "F1_layer_on_left_right.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
