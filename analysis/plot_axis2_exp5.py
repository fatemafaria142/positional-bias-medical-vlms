"""
Figures for Axis II Experiment V (`exp5_decoder_ablation`): image decoder vs. text (prompt) decoder ablation.

Settings (no retraining - components are bypassed at inference time):
  both_on    = normal VoxTell
  image_only = U-Net image decoder ON,  6-layer prompt (text) decoder bypassed
  text_only  = prompt decoder ON,       U-Net image decoder bypassed (stem features x mask embedding)
  both_off   = both bypassed

Run:  .venv/bin/python analysis/plot_axis2_exp5.py
Writes PNGs to results/results_axis2_exp5/plots/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import axis2_plot_style as S

EXP_ROOT = S.RESULTS_DIR / "results_axis2_exp5"
EXP_DIR = EXP_ROOT / "exp5_decoder_ablation"
OUT = EXP_ROOT / "plots"

SETTINGS = ["both_on", "image_only", "text_only", "both_off"]
SLABEL = {"both_on": "Both ON\n(normal VoxTell)", "image_only": "Image decoder only\n(prompt decoder OFF)",
          "text_only": "Prompt decoder only\n(image decoder OFF)", "both_off": "Both OFF"}
SSHORT = {"both_on": "Both ON\n(normal)", "image_only": "Prompt dec.\nOFF", "text_only": "Image dec.\nOFF", "both_off": "Both\nOFF"}


def load():
    m = pd.read_csv(EXP_DIR / "metrics_no_masks.csv")
    s = pd.read_csv(EXP_DIR / "summary.csv")
    return m, s


def fig_grid(m: pd.DataFrame):
    lat = m[m.variant_type == "lateralized"]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.axis("off")
    cells = {("on", "on"): "both_on", ("on", "off"): "image_only", ("off", "on"): "text_only", ("off", "off"): "both_off"}
    for (img, txt), setting in cells.items():
        x0 = 0 if txt == "on" else 1
        y0 = 1 if img == "on" else 0
        d = lat[lat.ablation_setting == setting]
        dice = d.dice_vs_matched_gt.mean()
        wrong = d.dice_vs_opposite_gt.mean()
        empty = (d.n_pred_voxels == 0).mean()
        col = plt.cm.RdYlGn(dice)
        ax.add_patch(plt.Rectangle((x0 + 0.03, y0 + 0.03), 0.94, 0.94, color=col, alpha=0.85, ec="white", lw=2))
        ax.text(x0 + 0.5, y0 + 0.78, SLABEL[setting].replace("\n", " "), ha="center", fontsize=9.5, fontweight="bold", color=S.INK)
        ax.text(x0 + 0.5, y0 + 0.52, f"Dice {dice:.2f}", ha="center", fontsize=20, fontweight="bold", color=S.INK)
        ax.text(x0 + 0.5, y0 + 0.30, f"opposite-side Dice {wrong:.2f}\nempty masks {empty * 100:.0f}%",
                ha="center", fontsize=9, color=S.INK)
    ax.text(0.5, 2.03, "Prompt (text) decoder ON", ha="center", fontsize=10.5, color=S.POSITION, fontweight="bold")
    ax.text(1.5, 2.03, "Prompt (text) decoder OFF", ha="center", fontsize=10.5, color=S.POSITION, fontweight="bold")
    ax.text(-0.04, 1.5, "Image decoder ON", rotation=90, va="center", ha="right", fontsize=10.5, color=S.ORGAN, fontweight="bold")
    ax.text(-0.04, 0.5, "Image decoder OFF", rotation=90, va="center", ha="right", fontsize=10.5, color=S.ORGAN, fontweight="bold")
    S.suptitle(fig, "Experiment V - 2x2 decoder ablation (left/right prompts)",
               "Mean over 7 structures x 100 subjects x 4 lateral prompts. Only the full model segments; each decoder alone fails.")
    S.save(fig, OUT, "exp5_01_ablation_2x2_overview.png")


def fig_heatmap(m: pd.DataFrame):
    lat = m[m.variant_type == "lateralized"]
    mat = np.array([[lat[(lat.ablation_setting == st) & (lat.target == t)].dice_vs_matched_gt.mean() for t in S.CORE7] for st in SETTINGS])
    fig, ax = plt.subplots(figsize=(13, 3.9))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color="white" if mat[i, j] < 0.3 else S.INK)
    ax.set_xticks(range(len(S.CORE7)), [S.short(t) for t in S.CORE7])
    ax.set_yticks(range(len(SETTINGS)), [SLABEL[s].replace("\n", " ") for s in SETTINGS], fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("Dice vs. requested side GT")
    S.suptitle(fig, "Segmentation quality per structure under each ablation",
               "Left/right prompts (canonical + short). Cerebral WM keeps a little Dice in the broken settings only because it is huge and the masks are diffuse.")
    S.save(fig, OUT, "exp5_02_dice_heatmap_by_structure.png")


def fig_laterality(m: pd.DataFrame):
    lat = m[m.variant_type == "lateralized"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    x = np.arange(len(SETTINGS))
    w = 0.38
    for ax, side in zip(axes, ["left", "right"]):
        d = lat[lat.side == side]
        gl = [d[d.ablation_setting == st].dice_vs_gt_left.mean() for st in SETTINGS]
        gr = [d[d.ablation_setting == st].dice_vs_gt_right.mean() for st in SETTINGS]
        el = [S.ci95(d[d.ablation_setting == st].dice_vs_gt_left) for st in SETTINGS]
        er = [S.ci95(d[d.ablation_setting == st].dice_vs_gt_right) for st in SETTINGS]
        b1 = ax.bar(x - w / 2, gl, w, yerr=el, color=S.LEFT, capsize=2, label="Dice with LEFT GT")
        b2 = ax.bar(x + w / 2, gr, w, yerr=er, color=S.RIGHT, capsize=2, label="Dice with RIGHT GT")
        S.bar_labels(ax, b1, errs=el)
        S.bar_labels(ax, b2, errs=er)
        ax.set_xticks(x, [SSHORT[s] for s in SETTINGS], fontsize=9)
        ax.set_title(f"'{side} X' prompts", color=S.LEFT if side == "left" else S.RIGHT)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Dice")
    axes[0].legend()
    S.suptitle(fig, "Which pathway carries the side word?",
               "With the prompt decoder bypassed, masks still land on the requested side (just weaker): side information is already in the projected text embedding.\n"
               "With the image decoder bypassed, left and right prompts give the same diffuse mask - laterality needs the U-Net fusion to be expressed.")
    S.save(fig, OUT, "exp5_03_laterality_by_setting.png")


def fig_overlap_volume(m: pd.DataFrame):
    lat = m[(m.variant_type == "lateralized") & (m.variant_subtype == "canonical_sentence")]
    p = lat.pivot_table(index=["ablation_setting", "subject_id", "target"], columns="side",
                        values=["n_pred_voxels", "setting_dice_left_vs_right"], aggfunc="first")
    nl, nr = p[("n_pred_voxels", "left")], p[("n_pred_voxels", "right")]
    lr = p[("setting_dice_left_vs_right", "left")]
    status = pd.DataFrame({"both_empty": (nl == 0) & (nr == 0), "one_empty": (nl == 0) ^ (nr == 0)})
    status["both_pred"] = ~(status.both_empty | status.one_empty)
    status["lr"] = lr
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    x = np.arange(len(SETTINGS))
    ax = axes[0]
    real = [status.loc[st][status.loc[st].both_pred].lr.mean() for st in SETTINGS]
    b = ax.bar(x, real, color=[S.SETTING_COLORS[s] for s in SETTINGS])
    S.bar_labels(ax, b)
    ax.set_xticks(x, [SSHORT[s] for s in SETTINGS], fontsize=9)
    ax.set_ylabel("Dice(left-prompt mask, right-prompt mask)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Are the left and right masks the same?\n(only scans where both masks are non-empty)", fontsize=10)

    ax = axes[1]
    be = [status.loc[st].both_empty.mean() * 100 for st in SETTINGS]
    oe = [status.loc[st].one_empty.mean() * 100 for st in SETTINGS]
    ax.bar(x, be, color=S.BAD, label="both left and right empty")
    ax.bar(x, oe, bottom=be, color="#F4A3AE", label="exactly one empty")
    for xi, a, c in zip(x, be, oe):
        if a + c > 0:
            ax.text(xi, a + c + 1, f"{a + c:.0f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x, [SSHORT[s] for s in SETTINGS], fontsize=9)
    ax.set_ylabel("% of (subject, structure) pairs")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8.5)
    ax.set_title("Empty predictions", fontsize=10)

    ax = axes[2]
    allp = m[m.variant_type == "lateralized"]
    for st in SETTINGS:
        d = allp[allp.ablation_setting == st]
        ratio = [d[d.target == t].n_pred_voxels.mean() for t in S.CORE7]
        ax.plot(range(len(S.CORE7)), ratio, "o-", color=S.SETTING_COLORS[st], lw=2, label=SLABEL[st].replace("\n", " "))
    ax.set_yscale("log")
    ax.set_xticks(range(len(S.CORE7)), [S.short(t) for t in S.CORE7], rotation=30, ha="right")
    ax.set_ylabel("Mean predicted voxels (log)")
    ax.legend(fontsize=7.5)
    ax.set_title("Mask size", fontsize=10)
    S.suptitle(fig, "How the broken settings fail",
               "Prompt decoder OFF -> masks shrink or vanish. Image decoder OFF -> large diffuse masks that are almost identical for 'left' and 'right'.")
    S.save(fig, OUT, "exp5_04_failure_modes_overlap_empty_volume.png")


def fig_synonyms(m: pd.DataFrame):
    types = ["base", "lateralized", "synonym", "paraphrase", "location_based"]
    lab = {"base": "Base", "lateralized": "Lateral", "synonym": "Synonym", "paraphrase": "Paraphrase", "location_based": "Location"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    x = np.arange(len(types))
    w = 0.2
    for k, st in enumerate(SETTINGS):
        d = m[m.ablation_setting == st]
        axes[0].bar(x + (k - 1.5) * w, [d[d.variant_type == v].dice_vs_matched_gt.mean() for v in types], w,
                    color=S.SETTING_COLORS[st], label=SLABEL[st].replace("\n", " "))
        nonref = d[(d.variant_type != "base") & ~((d.variant_type == "lateralized") & (d.variant_subtype == "canonical_sentence"))]
        axes[1].bar(k, nonref.pred_change_1_minus_dice_ref.mean(), 0.6, color=S.SETTING_COLORS[st])
        axes[1].text(k, nonref.pred_change_1_minus_dice_ref.mean() + 0.01, f"{nonref.pred_change_1_minus_dice_ref.mean():.2f}", ha="center", fontsize=8.5)
    axes[0].set_xticks(x, [lab[v] for v in types])
    axes[0].set_ylabel("Dice vs. GT")
    axes[0].set_ylim(0, 1.2)
    axes[0].legend(fontsize=8, ncol=2, loc="upper center")
    axes[0].set_title("Quality per prompt type", fontsize=10.5)
    axes[1].set_xticks(range(len(SETTINGS)), [SSHORT[s] for s in SETTINGS], fontsize=9)
    axes[1].set_ylabel("Prediction change vs. canonical prompt\n(1 - Dice), re-wordings only")
    axes[1].set_title("Wording sensitivity", fontsize=10.5)
    S.suptitle(fig, "Re-wording sensitivity under ablation",
               "With the prompt decoder bypassed, synonyms and location wordings collapse (Dice ~0.05) and the mask changes most with wording - the 6 cross-attention layers make VoxTell robust to phrasing.")
    S.save(fig, OUT, "exp5_05_wording_sensitivity_by_setting.png")


def fig_examples(subject_id: str = "OASIS_OAS1_0001_MR1", label: str = ""):
    npz_p = EXP_DIR / "masks" / f"{subject_id}_exp5_ablation_masks_packed.npz"
    subj = S.load_subject(subject_id)
    if subj is None or not npz_p.exists():
        print("  (skipping exp5 example overlays: volume or packed masks not available locally)")
        return
    img, lab = subj
    z = np.load(npz_p)
    shape = z["shape"]
    img_n = S.norm_img(img)
    for t in ["hippocampus", "caudate"]:
        lid, rid = S.LABEL_PAIRS[t]
        both = (lab == lid) | (lab == rid)
        a_idx = int(np.round(np.argwhere(both)[:, 1].mean()))
        crop = S.square_crop(S.coronal(both, a_idx), margin=28)
        fig, axes = plt.subplots(3, 4, figsize=(13, 10.2), gridspec_kw=dict(wspace=0.04, hspace=0.08))
        for i, (side, col, rl) in enumerate([("none", S.ORGAN, f"'Segment the {t} ...'"),
                                             ("left", S.LEFT, f"'... left {t} ...'"),
                                             ("right", S.RIGHT, f"'... right {t} ...'")]):
            for j, st in enumerate(SETTINGS):
                ax = axes[i, j]
                mk = S.unpack(z[f"{st}_{t}_{side}"], shape)
                S.overlay(ax, S.coronal(img_n, a_idx), [(S.coronal(mk, a_idx), col), (S.coronal(both, a_idx), "white")],
                          crop=crop, contour_only=[False, True])
                S.lr_tags(ax)
                ax.text(0.98, 0.95, f"{int(mk.sum()):,} vox", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, ec="none"))
                if i == 0:
                    ax.set_title(SLABEL[st], fontsize=9.5)
                if j == 0:
                    ax.set_ylabel(rl, fontsize=9.5, fontweight="bold")
        S.suptitle(fig, f"Example subject {subject_id} ({label}) - {t} under the 4 decoder settings",
                   "Filled = VoxTell mask (3D voxel count top-right), white outline = bilateral GT; coronal slice.")
        S.save(fig, OUT, f"exp5_06_example_masks_{t}_{S.subj_tag(subject_id)}.png")


def fig_examples_combined(subject_id: str, label: str, out_dir, fname: str):
    """One figure per subject: hippocampus (rows 1-3) and caudate (rows 4-6) under the 4 decoder settings."""
    npz_p = EXP_DIR / "masks" / f"{subject_id}_exp5_ablation_masks_packed.npz"
    subj = S.load_subject(subject_id)
    if subj is None or not npz_p.exists():
        print("  (skipping exp5 combined overlay: volume or packed masks not available locally)")
        return
    img, lab = subj
    z = np.load(npz_p)
    shape = z["shape"]
    img_n = S.norm_img(img)
    fig, axes = plt.subplots(6, 4, figsize=(12, 16.2), gridspec_kw=dict(wspace=0.04, hspace=0.08, top=0.975, bottom=0.01, left=0.06, right=0.99))
    for k, t in enumerate(["hippocampus", "caudate"]):
        lid, rid = S.LABEL_PAIRS[t]
        both = (lab == lid) | (lab == rid)
        a_idx = int(np.round(np.argwhere(both)[:, 1].mean()))
        crop = S.square_crop(S.coronal(both, a_idx), margin=28)
        for i, (side, col, rl) in enumerate([("none", S.ORGAN, f"'Segment the {t} ...'"),
                                             ("left", S.LEFT, f"'... left {t} ...'"),
                                             ("right", S.RIGHT, f"'... right {t} ...'")]):
            for j, st in enumerate(SETTINGS):
                ax = axes[3 * k + i, j]
                mk = S.unpack(z[f"{st}_{t}_{side}"], shape)
                S.overlay(ax, S.coronal(img_n, a_idx), [(S.coronal(mk, a_idx), col), (S.coronal(both, a_idx), "white")],
                          crop=crop, contour_only=[False, True])
                S.lr_tags(ax)
                ax.text(0.98, 0.95, f"{int(mk.sum()):,} vox", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, ec="none"))
                if 3 * k + i == 0:
                    ax.set_title(SLABEL[st], fontsize=9.5)
                if j == 0:
                    ax.set_ylabel(rl, fontsize=9, fontweight="bold")
    S.suptitle(fig, f"Subject {subject_id} ({label}) - hippocampus and caudate under the 4 decoder settings",
               "Filled = VoxTell mask (3D voxel count top-right), white outline = bilateral GT; coronal slice.")
    S.save(fig, out_dir, fname)


def main():
    S.apply_style()
    m, s = load()
    fig_grid(m)
    fig_heatmap(m)
    fig_laterality(m)
    fig_overlap_volume(m)
    fig_synonyms(m)
    for sid, label in S.EXAMPLE_SUBJECTS:
        fig_examples(sid, label)


if __name__ == "__main__":
    main()
