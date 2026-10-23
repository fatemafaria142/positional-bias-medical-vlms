"""
Figures for Axis II Experiment I (`exp1_side_word_comparison`): does the side word change the output?

Run:  .venv/bin/python analysis/plot_axis2_exp1.py
Writes PNGs to results/results_axis2_exp1/plots/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import axis2_plot_style as S

EXP_ROOT = S.RESULTS_DIR / "results_axis2_exp1"
EXP_DIR = EXP_ROOT / "exp1_side_word_comparison"
OUT = EXP_ROOT / "plots"

METRIC_COLS = [
    "subject_id", "target", "template_style", "condition", "side", "prompt_text", "n_pred_voxels",
    "n_gt_left_voxels", "n_gt_right_voxels", "dice_vs_gt_left", "dice_vs_gt_right", "dice_vs_gt_bilateral",
    "dice_vs_matched_gt", "dice_vs_contradictory_opposite_gt", "dice_left_vs_right",
    "frac_voxels_right_of_midline", "wrong_side_error",
]

COND_ORDER = [
    ("canonical_sentence", "base", "Base\n(no side)"),
    ("canonical_sentence", "left", "Left"),
    ("canonical_sentence", "right", "Right"),
    ("canonical_sentence", "contradictory_left_on_right", "left X on\nRIGHT hemi."),
    ("canonical_sentence", "contradictory_right_on_left", "right X on\nLEFT hemi."),
    ("short_term", "base", "Short base\n'X'"),
    ("short_term", "left", "Short\n'left X'"),
    ("short_term", "right", "Short\n'right X'"),
]


def load():
    m = pd.read_csv(EXP_DIR / "metrics.csv", usecols=lambda c: c in METRIC_COLS)
    s = pd.read_csv(EXP_DIR / "summary.csv")
    return m, s


def fig_heatmap(m: pd.DataFrame):
    mat = np.full((len(S.CORE7), len(COND_ORDER)), np.nan)
    for i, t in enumerate(S.CORE7):
        for j, (tmpl, cond, _) in enumerate(COND_ORDER):
            v = m[(m.target == t) & (m.template_style == tmpl) & (m.condition == cond)].dice_vs_matched_gt
            mat[i, j] = v.mean()
    fig, ax = plt.subplots(figsize=(12, 5.2))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(COND_ORDER)), [c[2] for c in COND_ORDER], fontsize=9)
    ax.set_yticks(range(len(S.CORE7)), [S.short(t) for t in S.CORE7])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] < 0.35 else S.INK, fontweight="bold")
    ax.axvline(4.5, color=S.INK, lw=1.5)
    ax.text(2, -0.85, "Sentence template: 'Segment the [side] X in this image.'", ha="center", fontsize=9, color="#555")
    ax.text(6, -0.85, "Short noun phrase", ha="center", fontsize=9, color="#555")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Dice vs. the ground truth the prompt asks for")
    S.suptitle(fig, "Experiment I - Segmentation quality for every prompt condition",
               "Base vs. bilateral GT, Left/Right vs. that side's GT, contradictory vs. the side named before the organ. 100 OASIS subjects.")
    S.save(fig, OUT, "01_dice_heatmap_by_condition.png")


def fig_side_bars(m: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    x = np.arange(len(S.CORE7))
    w = 0.38
    for ax, side in zip(axes, ["left", "right"]):
        d = m[(m.template_style == "canonical_sentence") & (m.condition == side)]
        gl = [d[d.target == t].dice_vs_gt_left.mean() for t in S.CORE7]
        gr = [d[d.target == t].dice_vs_gt_right.mean() for t in S.CORE7]
        el = [S.ci95(d[d.target == t].dice_vs_gt_left) for t in S.CORE7]
        er = [S.ci95(d[d.target == t].dice_vs_gt_right) for t in S.CORE7]
        b1 = ax.bar(x - w / 2, gl, w, yerr=el, color=S.LEFT, label="overlap with LEFT GT", capsize=2)
        b2 = ax.bar(x + w / 2, gr, w, yerr=er, color=S.RIGHT, label="overlap with RIGHT GT", capsize=2)
        S.bar_labels(ax, b1, errs=el)
        S.bar_labels(ax, b2, errs=er)
        ax.set_xticks(x, [S.short(t) for t in S.CORE7], rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Prompt: 'Segment the {side} X in this image.'", color=S.LEFT if side == "left" else S.RIGHT)
    axes[0].set_ylabel("Dice")
    axes[0].legend(loc="upper left", ncol=2)
    S.suptitle(fig, "The side word sends the mask to the correct hemisphere - and only there",
               "Mean Dice (95% CI over 100 subjects). The bar for the opposite hemisphere is ~0 for every structure.")
    S.save(fig, OUT, "02_left_right_prompts_hit_correct_side.png")


def fig_base_preference(m: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    x = np.arange(len(S.CORE7))
    w = 0.38
    for ax, tmpl, title in zip(axes, ["canonical_sentence", "short_term"],
                               ["'Segment the X in this image.'", "'X' (short noun phrase)"]):
        d = m[(m.template_style == tmpl) & (m.condition == "base")]
        fr = [d[d.target == t].frac_voxels_right_of_midline.mean() for t in S.CORE7]
        gl = [d[d.target == t].dice_vs_gt_left.mean() for t in S.CORE7]
        gr = [d[d.target == t].dice_vs_gt_right.mean() for t in S.CORE7]
        b1 = ax.bar(x - w / 2, gl, w, color=S.LEFT, label="Dice with LEFT GT")
        b2 = ax.bar(x + w / 2, gr, w, color=S.RIGHT, label="Dice with RIGHT GT")
        S.bar_labels(ax, b1)
        S.bar_labels(ax, b2)
        ax2 = ax.twinx()
        ax2.plot(x, fr, "D-", color=S.INK, ms=5, lw=1.2, label="fraction of mask right of midline")
        ax2.axhline(0.5, color=S.INK, ls=":", lw=1)
        ax2.set_ylim(0, 1.05)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        if tmpl == "short_term":
            ax2.set_ylabel("Fraction of mask voxels in right hemisphere")
        else:
            ax2.set_yticklabels([])
        ax.set_xticks(x, [S.short(t) for t in S.CORE7], rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Base prompt {title}")
    axes[0].set_ylabel("Dice")
    h1, l1 = axes[0].get_legend_handles_labels()
    axes[0].legend(h1 + [plt.Line2D([], [], color=S.INK, marker="D")], l1 + ["fraction right of midline (right axis)"],
                   loc="upper left", fontsize=8.5)
    S.suptitle(fig, "Without a side word, the sentence template leans towards the RIGHT hemisphere",
               "A balanced bilateral mask gives equal left/right Dice and 0.5 of voxels on each side. The short noun phrase is balanced; the sentence is not.")
    S.save(fig, OUT, "03_base_prompt_right_hemisphere_preference.png")


def fig_contradictory(m: pd.DataFrame):
    rows = []
    for t in S.CORE7:
        for cond, named, hemi, lat in [("contradictory_left_on_right", "left", "right", "left"),
                                       ("contradictory_right_on_left", "right", "left", "right")]:
            d = m[(m.target == t) & (m.template_style == "canonical_sentence") & (m.condition == cond)]
            lat_vox = m[(m.target == t) & (m.template_style == "canonical_sentence") & (m.condition == lat)].n_pred_voxels.mean()
            rows.append(dict(target=t, cond=cond,
                             named=d[f"dice_vs_gt_{named}"].mean(), hemi=d[f"dice_vs_gt_{hemi}"].mean(),
                             vol_ratio=d.n_pred_voxels.mean() / max(lat_vox, 1),
                             empty=(d.n_pred_voxels < 50).mean()))
    r = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(S.CORE7))
    w = 0.38
    for ax, cond, title, ncol, hcol in [
        (axes[0], "contradictory_left_on_right", "'Segment the LEFT X located on the RIGHT hemisphere'", S.LEFT, S.RIGHT),
        (axes[1], "contradictory_right_on_left", "'Segment the RIGHT X located on the LEFT hemisphere'", S.RIGHT, S.LEFT),
    ]:
        d = r[r.cond == cond].set_index("target").loc[S.CORE7]
        b1 = ax.bar(x - w / 2, d.named, w, color=ncol, label="follows side word before organ")
        b2 = ax.bar(x + w / 2, d.hemi, w, color=hcol, hatch="//", edgecolor="white", label="follows 'hemisphere' phrase")
        S.bar_labels(ax, b1, fontsize=7.5)
        S.bar_labels(ax, b2, fontsize=7.5)
        ax.set_xticks(x, [S.short(t) for t in S.CORE7], rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Dice")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    ax = axes[2]
    for cond, col, lab in [("contradictory_left_on_right", S.LEFT, "left X on right hemi."),
                           ("contradictory_right_on_left", S.RIGHT, "right X on left hemi.")]:
        d = r[r.cond == cond].set_index("target").loc[S.CORE7]
        ax.plot(x, d.vol_ratio, "o-", color=col, lw=2, label=lab)
    ax.axhline(1, color=S.INK, ls="--", lw=1)
    ax.text(len(S.CORE7) - 1, 1.03, "same size as plain lateral prompt", ha="right", fontsize=8)
    ax.set_xticks(x, [S.short(t) for t in S.CORE7], rotation=30, ha="right")
    ax.set_ylabel("Mask volume / volume with plain lateral prompt")
    ax.set_ylim(0, 1.25)
    ax.set_title("Mask size under contradiction", fontsize=10)
    ax.legend(fontsize=8)
    S.suptitle(fig, "Contradictory prompts: an asymmetric failure",
               "'right X on left hemisphere' is resolved to the right X; 'left X on right hemisphere' collapses (small, wrong or empty masks). Neither ever follows the 'hemisphere' phrase.")
    S.save(fig, OUT, "04_contradictory_prompt_asymmetry.png")
    return r


def fig_volume(m: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    conds = [("base", "Base (no side)", S.BASE), ("left", "Left", S.LEFT), ("right", "Right", S.RIGHT)]
    x = np.arange(len(S.CORE7))
    w = 0.26
    for k, (cond, lab, col) in enumerate(conds):
        vals, errs = [], []
        for t in S.CORE7:
            d = m[(m.target == t) & (m.template_style == "short_term") & (m.condition == cond)]
            ratio = d.n_pred_voxels / (d.n_gt_left_voxels + d.n_gt_right_voxels)
            vals.append(ratio.mean())
            errs.append(S.ci95(ratio))
        b = ax.bar(x + (k - 1) * w, vals, w, yerr=errs, color=col, label=lab, capsize=2)
        S.bar_labels(ax, b, fontsize=7.5, errs=errs)
    ax.axhline(1.0, color=S.INK, ls="--", lw=1)
    ax.axhline(0.5, color=S.INK, ls=":", lw=1)
    ax.text(-0.5, 1.02, "both hemispheres", fontsize=8)
    ax.text(-0.5, 0.52, "one hemisphere", fontsize=8)
    ax.set_xticks(x, [S.short(t) for t in S.CORE7])
    ax.set_ylabel("Predicted volume / bilateral GT volume")
    ax.legend(ncol=3, loc="upper right")
    ax.set_ylim(0, 1.3)
    S.suptitle(fig, "Mask volume: base prompt covers both sides, a side word halves it",
               "Short noun-phrase prompts ('X', 'left X', 'right X'); mean with 95% CI over 100 subjects.")
    S.save(fig, OUT, "05_mask_volume_base_vs_side.png")


def fig_absent(s: pd.DataFrame, m_all: pd.DataFrame):
    d = s[s.condition == "absent_organ_negative"].copy()
    prompts = {
        ("kidney", "base_absent"): "Segment the kidney in this image.",
        ("kidney", "lateralized_absent"): "Segment the left/right kidney ...",
        ("adrenal tumor", "lesion_base_absent"): "adrenal tumor",
        ("adrenal tumor", "lesion_left_absent"): "left adrenal tumor",
        ("adrenal tumor", "lesion_right_absent"): "right adrenal tumor",
        ("lung nodule", "lung_lobe_right_upper_absent"): "nodule in right upper lobe",
        ("lung nodule", "lung_lobe_left_lower_absent"): "nodule in left lower lobe",
    }
    neg = m_all[m_all.condition == "absent_organ_negative"]
    d["label"] = [prompts.get((a, b), f"{a} ({b})") for a, b in zip(d.target, d.template_style)]
    d["any_fp"] = [((neg.target == a) & (neg.template_style == b) & (neg.n_pred_voxels > 0)).sum()
                   / max(((neg.target == a) & (neg.template_style == b)).sum(), 1) for a, b in zip(d.target, d.template_style)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    y = np.arange(len(d))
    axes[0].barh(y, d.mean_pred_voxels, color="#6D597A")
    for yi, v in zip(y, d.mean_pred_voxels):
        axes[0].text(v + 0.5, yi, f"{v:.1f}", va="center", fontsize=8)
    axes[0].set_yticks(y, d.label)
    axes[0].set_xlabel("Mean predicted voxels per brain MRI (brain structures are 4k-600k voxels)")
    axes[0].set_title("How much does VoxTell segment?")
    axes[1].barh(y, d.any_fp * 100, color=S.BAD)
    for yi, v in zip(y, d.any_fp * 100):
        axes[1].text(v + 0.3, yi, f"{v:.0f}%", va="center", fontsize=8)
    axes[1].set_xlabel("% of scans with any false-positive voxel")
    axes[1].set_xlim(0, max(10, d.any_fp.max() * 130))
    axes[1].set_title("How often does it hallucinate?")
    axes[0].invert_yaxis()
    S.suptitle(fig, "Absent-organ prompts on brain MRI: VoxTell correctly returns (almost) nothing",
               "Organs that are not in the image are not hallucinated, with or without a side word (kidney: a few tiny blobs).")
    S.save(fig, OUT, "06_absent_organ_negative_prompts.png")


def fig_distribution(m: pd.DataFrame):
    order = [c for c in COND_ORDER if c[0] == "canonical_sentence"]
    data = [m[(m.template_style == tmpl) & (m.condition == cond) & m.target.isin(S.CORE7)].dice_vs_matched_gt.dropna().values
            for tmpl, cond, _ in order]
    cols = [S.BASE, S.LEFT, S.RIGHT, S.CONTRA, "#F4A3AE"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    vp = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.8)
    for body, c in zip(vp["bodies"], cols):
        body.set_facecolor(c)
        body.set_alpha(0.55)
    for k in ("cmedians", "cbars", "cmins", "cmaxes"):
        vp[k].set_color(S.INK)
    rng = np.random.default_rng(0)
    for i, (dvals, c) in enumerate(zip(data, cols)):
        ax.scatter(i + 1 + rng.uniform(-0.18, 0.18, len(dvals)), dvals, s=3, color=c, alpha=0.35)
        ax.text(i + 1, 1.04, f"failures (<0.1): {np.mean(dvals < 0.1) * 100:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(range(1, len(order) + 1), [o[2] for o in order])
    ax.set_ylim(-0.02, 1.1)
    ax.set_ylabel("Dice vs. requested GT")
    S.suptitle(fig, "Per-scan distribution (7 structures x 100 subjects per violin)",
               "Plain left/right prompts are tightly clustered near 0.85; 'left X on right hemisphere' is spread out and fails completely (<0.1) on 41% of scans.")
    S.save(fig, OUT, "07_per_scan_dice_distribution.png")


def fig_examples(subject_id: str = "OASIS_OAS1_0001_MR1", label: str = "", out_dir=None, fname=None):
    npz_p = EXP_DIR / "masks" / f"{subject_id}_exp1_masks_packed.npz"
    subj = S.load_subject(subject_id)
    if subj is None or not npz_p.exists():
        print("  (skipping example overlays: volume or packed masks not available locally)")
        return
    img, lab = subj
    z = np.load(npz_p)
    shape = z["shape"]
    img_n = S.norm_img(img)
    targets = ["hippocampus", "thalamus", "caudate", "putamen"]
    cols = [("gt", "Ground truth\n(L purple / R orange)"), ("base", "'Segment the X\nin this image.'"),
            ("left", "'... the left X ...'"), ("right", "'... the right X ...'"),
            ("contradictory_left_on_right", "'left X located on\nthe right hemisphere'"),
            ("contradictory_right_on_left", "'right X located on\nthe left hemisphere'")]
    fig, axes = plt.subplots(len(targets), len(cols), figsize=(15, 2.5 * len(targets)),
                             gridspec_kw=dict(wspace=0.04, hspace=0.06))
    for i, t in enumerate(targets):
        lid, rid = S.LABEL_PAIRS[t]
        gl, gr = lab == lid, lab == rid
        both = gl | gr
        a_idx = int(np.round(np.argwhere(both)[:, 1].mean()))
        crop = S.square_crop(S.coronal(both, a_idx), margin=10)
        for j, (key, title) in enumerate(cols):
            ax = axes[i, j]
            base = S.coronal(img_n, a_idx)
            if key == "gt":
                S.overlay(ax, base, [(S.coronal(gl, a_idx), S.LEFT), (S.coronal(gr, a_idx), S.RIGHT)], crop=crop)
            else:
                m = S.unpack(z[f"{t}_canonical_sentence_{key}"], shape)
                col = {"base": S.ORGAN, "left": S.LEFT, "right": S.RIGHT}.get(key, S.CONTRA)
                S.overlay(ax, base, [(S.coronal(m, a_idx), col), (S.coronal(both, a_idx), "white")],
                          crop=crop, contour_only=[False, True])
            S.lr_tags(ax)
            if i == 0:
                ax.set_title(title, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(S.short(t), fontsize=11, fontweight="bold")
    S.suptitle(fig, f"Example subject {subject_id} ({label}) - coronal slice through each structure",
               "Filled color = VoxTell mask; white outline = bilateral ground truth. Image left = patient left.")
    S.save(fig, out_dir or OUT, fname or f"08_example_masks_side_and_contradictory_{S.subj_tag(subject_id)}.png")


def main():
    S.apply_style()
    m, s = load()
    m_core = m[m.target.isin(S.CORE7)]
    fig_heatmap(m_core)
    fig_side_bars(m_core)
    fig_base_preference(m_core)
    r = fig_contradictory(m_core)
    fig_volume(m_core)
    fig_absent(s, m)
    fig_distribution(m_core)
    for sid, label in S.EXAMPLE_SUBJECTS:
        fig_examples(sid, label)
    r.round(4).to_csv(OUT / "contradictory_prompt_breakdown.csv", index=False)


if __name__ == "__main__":
    main()
