"""RQ1 result analysis: one table (T2) + one figure (F1) from results/neurite-oasis/RQ1."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SRC = Path(r"D:\Reasoning\results\neurite-oasis\RQ1")
OUT = Path(__file__).parent
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
STAGES = ["intermediate_0", "intermediate_1", "intermediate_2", "intermediate_3", "finest_logit"]
SL = ["Stage 0\n(coarsest)", "Stage 1", "Stage 2", "Stage 3", "Finest logit\n(final output)"]

m = pd.read_csv(SRC / "rq1_masks.csv", usecols=range(15))
f = pd.read_csv(SRC / "rq1_fusion_channel_similarity.csv")
# first 50 subjects (sorted by ID), the same convention as the other RQs
SUBJECTS = sorted(m.subject_id.unique())[:50]
m = m[m.subject_id.isin(SUBJECTS)].copy()
f = f[f.subject_id.isin(SUBJECTS)].copy()
m["empty_pred"] = m.n_pred_voxels == 0
m["dice_ok"] = m.dice_named_organ >= 0.5
m["wrong_side"] = m.dice_other_side_organ > 0

# ---------- T2: per-organ segmentation ----------
g = m.groupby("organ_pair")
t2 = pd.DataFrame({
    "n": g.size(), "dice_mean": g.dice_named_organ.mean(), "dice_median": g.dice_named_organ.median(),
    "dice_std": g.dice_named_organ.std(), "iou_mean": g.iou_named_organ.mean(),
    "empty_pred_rate": g.empty_pred.mean(), "dice_ge_0.5_rate": g.dice_ok.mean(),
    "wrong_side_rows": g.wrong_side.sum(), "mean_dice_other_side": g.dice_other_side_organ.mean(),
    "mean_gt_voxels": g.n_gt_voxels.mean(), "mean_pred_voxels": g.n_pred_voxels.mean()
}).sort_values("dice_mean", ascending=False)
t2.round(4).to_csv(TAB / "T2_per_organ_segmentation.csv", index=True)

# ---------- F1: simple heatmap, left/right fusion-channel similarity ----------
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import font_manager

avail = {x.name for x in font_manager.fontManager.ttflist}
FONT = next(n for n in ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"] if n in avail)
plt.rcParams.update({"font.family": FONT, "pdf.fonttype": 42, "svg.fonttype": "none"})

INK, MUTED = "#000000", "#000000"            # black everywhere, no gray text
cmap = LinearSegmentedColormap.from_list(                      # red = the prompts differ, green = the prompts are identical
    "red_green", [(0.0, "#d6453d"), (0.2, "#eba49e"), (0.4, "#f3efe9"), (0.7, "#8fcba5"), (1.0, "#1f7a49")])
norm = Normalize(0.5, 1.0)

cos = f.pivot_table(index="organ_pair", columns="stage", values="cosine_mean")[STAGES]
order = t2.index.tolist()                      # best Dice first
n = len(order)
pretty = lambda o: "Ventral DC" if o == "ventral DC" else o.capitalize()

def lum(rgb):
    r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

fig, ax = plt.subplots(figsize=(7.4, 7.2), facecolor="white")
ax.set_xlim(-2.6, 5.0); ax.set_ylim(n + 3.1, -2.5); ax.axis("off")

ax.text(1.2, -2.4, "Left vs. Right Fusion-Channel Similarity Across U-Net Decoder Stages",
        fontsize=12.5, fontweight="bold", color=INK, va="top", ha="center")
heads = ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Finest logit"]
for k, h in enumerate(heads):
    ax.text(k + 0.5, -0.05, h, ha="center", va="bottom", fontsize=10, fontweight="semibold", color=INK)
for i, o in enumerate(order):
    ax.text(-0.12, i + 0.5, pretty(o), ha="right", va="center", fontsize=10, color=INK)
    for k, st in enumerate(STAGES):
        v = cos.loc[o, st]; col = cmap(norm(v))
        ax.add_patch(FancyBboxPatch((k + 0.04, i + 0.1), 0.92, 0.8, boxstyle="square,pad=0",
                                    fc=col, ec="none"))
        ax.text(k + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=9,
                color="white" if lum(col) < 0.28 else INK)

# legend: slim gradient bar centred under the tiles
lw_, lh = 3.6, 0.26
lx, ly = 1.2 - lw_ / 2, n + 1.5      # 1.2 is the horizontal centre of the whole figure
ax.imshow(np.linspace(0.5, 1.0, 256)[None, :], extent=(lx, lx + lw_, ly + lh, ly), cmap=cmap, norm=norm,
          aspect="auto")
ax.set_xlim(-2.6, 5.0); ax.set_ylim(n + 3.2, -2.5)
ax.text(lx + lw_ / 2, ly - 0.2, "Mean cosine similarity between left and right prompts", ha="center", va="bottom",
        fontsize=9.5, fontweight="semibold", color=INK)
for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    x = lx + (v - 0.5) / 0.5 * lw_
    ax.plot([x, x], [ly + lh, ly + lh + 0.1], color=MUTED, lw=0.8)
    ax.text(x, ly + lh + 0.16, f"{v:.1f}", ha="center", va="top", fontsize=8.5, color=MUTED)
ax.text(lx, ly + lh + 0.75, "0.5 = different", ha="left", va="top", fontsize=8.5, color=INK)
ax.text(lx + lw_, ly + lh + 0.75, "1.0 = identical", ha="right", va="top", fontsize=8.5, color=INK)
ax.text(lx + lw_ / 2, ly + lh + 1.4, "Each tile is the mean over the 32 channels and over subjects", ha="center", va="top", fontsize=9, color=INK)

fig.savefig(FIG / "F1_heatmap_cosine_organ_stage.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

# ---------- T1: metric-wise summary ----------
COLS = ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Finest logit"]
pm = lambda x: f"{x.mean():.3f} ± {x.std():.3f}"
rows = []
for col, nm in [("cosine_mean", "Cosine similarity, mean of 32 channels"), ("cosine_min", "Cosine similarity, min of 32 channels"),
                ("corr_mean", "Pearson correlation"), ("norm_l2_mean", "Normalized L2 difference")]:
    r = {"group": "Fusion channels: left vs right prompt", "metric": nm, "Overall": ""}
    r.update({c: pm(f.loc[f.stage == st, col]) for c, st in zip(COLS, STAGES)})
    rows.append(r)
seg = [("Dice, named side", m.dice_named_organ), ("IoU, named side", m.iou_named_organ),
       ("Dice, wrong side", m.dice_other_side_organ), ("IoU, wrong side", m.iou_other_side_organ)]
for nm, x in seg:
    rows.append({"group": "Segmentation: predicted mask", "metric": nm, **{c: "" for c in COLS}, "Overall": pm(x)})
rows.append({"group": "Segmentation: predicted mask", "metric": "Empty-prediction rate", **{c: "" for c in COLS},
             "Overall": f"{m.empty_pred.mean():.1%}"})
rows.append({"group": "Segmentation: predicted mask", "metric": "Predictions with Dice ≥ 0.5", **{c: "" for c in COLS},
             "Overall": f"{m.dice_ok.mean():.1%}"})
rows.append({"group": "Segmentation: predicted mask", "metric": "Rows with any wrong-side overlap", **{c: "" for c in COLS},
             "Overall": f"{int(m.wrong_side.sum())} / {len(m)}"})
tm = pd.DataFrame(rows)[["group", "metric"] + COLS + ["Overall"]]
tm.to_csv(TAB / "T1_metrics_summary.csv", index=False, encoding="utf-8-sig")
print(tm.to_string(index=False))
