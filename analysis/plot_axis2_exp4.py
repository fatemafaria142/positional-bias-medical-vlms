"""
Figures for Axis II Experiment IV (`exp4_mc_dropout_uncertainty`): MC-dropout uncertainty per prompt wording.

Run:  .venv/bin/python analysis/plot_axis2_exp4.py
Writes PNGs to results/results_axis2_exp4/plots/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

import axis2_plot_style as S

EXP_ROOT = S.RESULTS_DIR / "results_axis2_exp4"
EXP_DIR = EXP_ROOT / "exp4_mc_dropout_uncertainty"
OUT = EXP_ROOT / "plots"

TARGETS = ["hippocampus", "thalamus", "caudate", "putamen"]
VTYPES = ["base", "lateralized", "synonym", "paraphrase", "location_based", "contradictory", "absent_organ_negative"]
VLABEL = {"base": "Base\n(no side)", "lateralized": "Lateral", "synonym": "Synonym", "paraphrase": "Paraphrase",
          "location_based": "Location\ndescription", "contradictory": "Contra-\ndictory", "absent_organ_negative": "Absent\norgan"}


def load():
    m = pd.read_csv(EXP_DIR / "metrics.csv", usecols=lambda c: c != "mean_mask_rle_b64")
    st = pd.read_csv(EXP_DIR / "synonym_uncertainty_stability.csv")
    return m, st


def fig_by_type(m: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    x = np.arange(len(VTYPES))
    for ax, col, ylabel in [(axes[0], "roi_epistemic_mi_bits", "Epistemic uncertainty (mutual information, bits)"),
                            (axes[1], "inter_pass_mean_dice", "Agreement between dropout passes (mean pairwise Dice)"),
                            (axes[2], "dice_mean_mask_vs_gt", "Dice of the averaged mask vs. GT")]:
        vals = [m[m.variant_type == v][col].mean() for v in VTYPES]
        errs = [S.ci95(m[m.variant_type == v][col]) for v in VTYPES]
        b = ax.bar(x, vals, yerr=errs, color=[S.VARIANT_COLORS[v] for v in VTYPES], capsize=2)
        S.bar_labels(ax, b, fontsize=8, errs=errs)
        ax.set_xticks(x, [VLABEL[v].replace("-\n", "").replace("\n", " ") for v in VTYPES], fontsize=8.5, rotation=30, ha="right")
        ax.set_ylabel(ylabel, fontsize=9.5)
    axes[1].set_ylim(0.5, 1.02)
    axes[2].set_ylim(0, 1.05)
    S.suptitle(fig, "Experiment IV - Which wordings make VoxTell unsure of itself?",
               "5 MC-dropout passes per prompt, 4 structures x 100 subjects (absent organ: kidney / adrenal / lung prompts on brain MRI). 95% CI.")
    S.save(fig, OUT, "exp4_01_uncertainty_by_prompt_type.png")


def fig_left_right(m: pd.DataFrame):
    types = ["lateralized", "synonym", "paraphrase", "location_based", "contradictory"]
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(17, 4.4), sharey=True)
    for ax, t in zip(axes, TARGETS):
        x = np.arange(len(types))
        for k, (side, c) in enumerate([("left", S.LEFT), ("right", S.RIGHT)]):
            d = m[(m.target == t) & (m.side == side)]
            vals = [d[d.variant_type == v].roi_epistemic_mi_bits.mean() for v in types]
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=c, label=f"{side} prompt")
        base = m[(m.target == t) & (m.variant_type == "base")].roi_epistemic_mi_bits.mean()
        ax.axhline(base, color=S.BASE, ls="--", lw=1.2)
        ax.text(len(types) - 0.5, base, "base", va="bottom", ha="right", fontsize=8, color=S.BASE)
        ax.set_xticks(x, ["Lateral", "Synonym", "Paraphr.", "Location", "Contra."], rotation=30, ha="right", fontsize=8.5)
        ax.set_title(S.short(t))
    axes[0].set_ylabel("Epistemic MI (bits)")
    axes[0].legend(fontsize=8.5)
    S.suptitle(fig, "Left-side wordings carry more uncertainty than the same wording on the right",
               "Mean ROI mutual information per structure. Contradictory LEFT prompts ('left X on the right hemisphere') are the most uncertain - except thalamus, where VoxTell is confidently wrong.")
    S.save(fig, OUT, "exp4_02_left_vs_right_uncertainty_by_structure.png")


def fig_uncertainty_vs_error(m: pd.DataFrame):
    d = m[m.variant_type != "absent_organ_negative"].dropna(subset=["dice_mean_mask_vs_gt"]).copy()
    fail = (d.dice_mean_mask_vs_gt < 0.5).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    ax = axes[0]
    for v in ["base", "lateralized", "synonym", "paraphrase", "location_based", "contradictory"]:
        dv = d[d.variant_type == v]
        ax.scatter(dv.roi_epistemic_mi_bits, dv.dice_mean_mask_vs_gt, s=6, alpha=0.35, color=S.VARIANT_COLORS[v], label=VLABEL[v].replace("-\n", "").replace("\n", " "))
    rho, p = spearmanr(d.roi_epistemic_mi_bits, d.dice_mean_mask_vs_gt)
    ax.set_xlabel("Epistemic MI (bits)")
    ax.set_ylabel("Dice vs. GT")
    ax.set_title(f"Each dot = one prompt on one scan (n = {len(d)})\nSpearman rho = {rho:.2f}", fontsize=10)
    ax.legend(fontsize=7.5, markerscale=2.5, loc="upper right")
    ax.annotate("confidently empty /\nconfidently wrong", xy=(0.01, 0.02), xytext=(0.12, 0.2), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=S.INK))

    ax = axes[1]
    for col, lab, c in [("roi_epistemic_mi_bits", "Epistemic MI", S.POSITION), ("roi_epistemic_variance", "Epistemic variance", S.ORGAN),
                        ("pass_volume_cv", "Pass volume CV", "#FF9F1C"), ("roi_aleatoric_entropy_bits", "Aleatoric entropy", S.BASE)]:
        fpr, tpr, _ = roc_curve(fail, d[col].fillna(0))
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{lab}: AUROC {roc_auc_score(fail, d[col].fillna(0)):.2f}")
    ax.plot([0, 1], [0, 1], ":", color=S.INK)
    ax.set_xlabel("False-alarm rate")
    ax.set_ylabel("Failures caught")
    ax.set_title(f"Can uncertainty flag failures (Dice < 0.5, {fail.mean() * 100:.0f}% of cases)?", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    bins = np.quantile(d.roi_epistemic_mi_bits, np.linspace(0, 1, 11))
    d["bin"] = np.clip(np.digitize(d.roi_epistemic_mi_bits, bins[1:-1]), 0, 9)
    g = d.groupby("bin").agg(mi=("roi_epistemic_mi_bits", "mean"), dice=("dice_mean_mask_vs_gt", "mean"),
                             failrate=("dice_mean_mask_vs_gt", lambda s: (s < 0.5).mean()))
    ax.bar(np.arange(len(g)), g.failrate * 100, color=S.BAD, alpha=0.85)
    ax.set_xticks(np.arange(len(g)), [f"{v:.2f}" for v in g.mi], rotation=45, fontsize=8)
    ax.set_xlabel("Epistemic MI decile (mean bits)")
    ax.set_ylabel("% of cases that failed (Dice < 0.5)")
    ax.set_title("Failure rate by uncertainty decile", fontsize=10)
    S.suptitle(fig, "Does high uncertainty mean a bad mask?",
               "Mostly yes (label-free signals, AUROC ~0.89). But even the lowest-uncertainty decile fails ~12% of the time - mostly 'left thalamus on the right hemisphere',\n"
               "where VoxTell confidently predicts (almost) nothing, so there is nothing to be uncertain about.")
    S.save(fig, OUT, "exp4_03_uncertainty_predicts_failure.png")


def fig_contradictory(st: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    labels = [f"{S.short(t)}\n{s}" for t, s in zip(st.target, st.side)]
    x = np.arange(len(st))
    w = 0.27
    ax.bar(x - w, st.canonical_epistemic_var, w, color=S.ORGAN, label="canonical 'Segment the side X ...'")
    ax.bar(x, st.synonym_mean_epistemic_var, w, yerr=st.synonym_std_epistemic_var, color=S.POSITION, capsize=2,
           label="synonyms / paraphrases (mean ± SD)")
    ax.bar(x + w, st.contradictory_epistemic_var, w, color=S.CONTRA, label="contradictory")
    for xi, flag in zip(x, st.contradictory_gt_canonical_uncertainty):
        ax.text(xi + w, st.contradictory_epistemic_var.iloc[xi] + 0.004, "↑" if flag else "↓", ha="center", fontsize=11,
                color=S.CONTRA if flag else S.INK)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean ROI epistemic variance")
    ax.legend(fontsize=8.5)
    S.suptitle(fig, "Is a contradictory prompt more uncertain than the canonical prompt?",
               "↑ = yes. True in 5 of 8 structure/side cases; strongest for LEFT caudate / hippocampus / putamen. Synonyms vary up to 20-50x between scans (max/min ratio).")
    S.save(fig, OUT, "exp4_04_contradictory_vs_canonical_uncertainty.png")


def fig_decomposition(m: pd.DataFrame):
    types = VTYPES
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    x = np.arange(len(types))
    epi = [m[m.variant_type == v].roi_epistemic_mi_bits.mean() for v in types]
    ale = [m[m.variant_type == v].roi_aleatoric_entropy_bits.mean() for v in types]
    axes[0].bar(x, ale, color="#C9C2B8", label="aleatoric (data / boundary ambiguity)")
    axes[0].bar(x, epi, bottom=ale, color=S.POSITION, label="epistemic (model unsure)")
    for xi, a, e in zip(x, ale, epi):
        axes[0].text(xi, a + e + 0.01, f"{e / max(a + e, 1e-9) * 100:.0f}%\nepistemic", ha="center", fontsize=7.5)
    axes[0].set_xticks(x, [VLABEL[v] for v in types], fontsize=8.5)
    axes[0].set_ylabel("Predictive entropy (bits)")
    axes[0].set_ylim(0, 0.75)
    axes[0].legend(fontsize=8.5, loc="upper left")
    axes[0].set_title("What kind of uncertainty?", fontsize=10.5)

    ece = [m[m.variant_type == v].ece_roi.mean() for v in types]
    cv = [m[m.variant_type == v].pass_volume_cv.mean() for v in types]
    b1 = axes[1].bar(x - 0.2, ece, 0.4, color=S.BASE, label="ECE (calibration error, lower = better)")
    b2 = axes[1].bar(x + 0.2, cv, 0.4, color="#FF9F1C", label="volume CV across passes")
    S.bar_labels(axes[1], b1, fontsize=7.5)
    S.bar_labels(axes[1], b2, fontsize=7.5)
    axes[1].set_xticks(x, [VLABEL[v] for v in types], fontsize=8.5)
    axes[1].legend(fontsize=8.5)
    axes[1].set_title("Calibration and volume stability", fontsize=10.5)
    S.suptitle(fig, "Unusual wordings mainly add MODEL (epistemic) uncertainty, and the probabilities become poorly calibrated",
               "Predictive entropy = aleatoric + epistemic (mutual information). Location descriptions and contradictions have the worst calibration.")
    S.save(fig, OUT, "exp4_05_uncertainty_decomposition_and_calibration.png")


def fig_maps(subject_id: str = "OASIS_OAS1_0001_MR1", label: str = "", out_dir=None, fname=None):
    npz_p = EXP_DIR / "masks" / f"{subject_id}_exp4_uncertainty_maps.npz"
    subj = S.load_subject(subject_id)
    if subj is None or not npz_p.exists():
        print("  (skipping uncertainty-map examples: volume or maps not available locally)")
        return
    img, lab = subj
    z = np.load(npz_p)
    shape = tuple(int(v) for v in z["orig_shape"])
    img_n = S.norm_img(img)

    def full(pid: str, key: str) -> np.ndarray:
        out = np.zeros(shape, dtype=np.float32)
        box = z[f"{pid}_box"]
        out[tuple(slice(int(a), int(b)) for a, b in box)] = z[f"{pid}_{key}"].astype(np.float32)
        return out

    rows = [("hippocampus", [("OASIS_P0001", "'Segment the hippocampus ...'"), ("OASIS_P0003", "'... left hippocampus ...'"),
                             ("OASIS_P0015", "'... right hippocampus ...'"), ("OASIS_P0013", "'structure in the left\nmedial temporal lobe'"),
                             ("OASIS_P0014", "'left hippocampus on the\nright hemisphere'")]),
            ("caudate", [("OASIS_P0053", "'Segment the caudate ...'"), ("OASIS_P0055", "'... left caudate ...'"),
                         ("OASIS_P0067", "'... right caudate ...'"), ("OASIS_P0065", "'C-shaped ... bordering the\nleft lateral ventricle'"),
                         ("OASIS_P0066", "'left caudate on the\nright hemisphere'")])]
    fig, axes = plt.subplots(2 * len(rows), 5, figsize=(15, 5.6 * len(rows)), gridspec_kw=dict(wspace=0.04, hspace=0.22))
    for r, (t, items) in enumerate(rows):
        lid, rid = S.LABEL_PAIRS[t]
        both = (lab == lid) | (lab == rid)
        a_idx = int(np.round(np.argwhere(both)[:, 1].mean()))
        crop = S.square_crop(S.coronal(both, a_idx), margin=10)
        base = S.coronal(img_n, a_idx)[crop]
        gt_c = S.coronal(both, a_idx)[crop]
        for j, (pid, title) in enumerate(items):
            for k, (key, cmap, vmax, lab_txt) in enumerate([("mean_prob", "viridis", 1.0, "mean probability"),
                                                            ("epistemic_var", "magma", 0.25, "epistemic variance")]):
                ax = axes[2 * r + k, j]
                mp = S.coronal(full(pid, key), a_idx)[crop]
                ax.imshow(base, cmap="gray")
                ax.imshow(np.ma.masked_less(mp, 0.01 if key == "mean_prob" else 0.002), cmap=cmap, vmin=0, vmax=vmax, alpha=0.85)
                ax.contour(gt_c.astype(float), levels=[0.5], colors=["white"], linewidths=0.9)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                S.lr_tags(ax)
                if k == 0:
                    ax.set_title(title, fontsize=8.5)
                if j == 0:
                    ax.set_ylabel(f"{S.short(t)}\n{lab_txt}", fontsize=9.5, fontweight="bold")
    S.suptitle(fig, f"Example subject {subject_id} ({label}) - MC-dropout mean probability and epistemic variance",
               "White outline = bilateral GT. Uncertainty concentrates on boundaries for clear prompts and spreads over the whole mask for confusing ones.")
    S.save(fig, out_dir or OUT, fname or f"exp4_06_example_uncertainty_maps_{S.subj_tag(subject_id)}.png")


def main():
    S.apply_style()
    m, st = load()
    fig_by_type(m)
    fig_left_right(m)
    fig_uncertainty_vs_error(m)
    fig_contradictory(st)
    fig_decomposition(m)
    for sid, label in S.EXAMPLE_SUBJECTS:
        fig_maps(sid, label)


if __name__ == "__main__":
    main()
