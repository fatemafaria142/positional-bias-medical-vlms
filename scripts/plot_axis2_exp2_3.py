"""
Figures for Axis II Experiments II + III (`results_axis2_exp2_3`):
  - Experiment II  (`exp2_synonym_shift`): how much do synonyms / paraphrases / re-wordings change the mask,
    and does embedding distance predict that change?
  - Experiment III (`exp3_layer_localization`): where in VoxTell (14 stages) do side and wording information live?

Run:  .venv/bin/python scripts/plot_axis2_exp2_3.py
Writes PNGs to results/results_axis2_exp2_3/plots/.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

import axis2_plot_style as S

EXP_ROOT = S.RESULTS_DIR / "results_axis2_exp2_3"
EXP2 = EXP_ROOT / "exp2_synonym_shift"
EXP3 = EXP_ROOT / "exp3_layer_localization"
EXP1_MASKS = S.RESULTS_DIR / "results_axis2_exp1" / "exp1_side_word_comparison" / "masks"
OUT = EXP_ROOT / "plots"

SUBTYPE_ORDER = [
    ("short_term", "'left X'"),
    ("clinical_synonym_1", "Synonym 1"),
    ("clinical_synonym_2", "Synonym 2"),
    ("clinical_synonym_3", "Synonym 3"),
    ("clinical_synonym_4", "'l.' / 'r.'"),
    ("clinical_synonym_5", "Synonym 5"),
    ("paraphrase_1", "Paraphrase 1"),
    ("paraphrase_2", "Paraphrase 2"),
    ("paraphrase_3", "Paraphrase 3"),
    ("anatomical_location_wording", "Location\ndescription"),
    ("conflicting_left_on_right", "Contradictory"),
]
VTYPES = ["lateralized", "synonym", "paraphrase", "location_based", "contradictory"]
VLABEL = {"lateralized": "Short term", "synonym": "Synonym", "paraphrase": "Paraphrase",
          "location_based": "Location description", "contradictory": "Contradictory"}

STAGE_SHORT = ["Qwen3", "Proj", "Dec L1", "Dec L2", "Dec L3", "Dec L4", "Dec L5", "Dec L6", "Norm",
               "Fus 12³", "Fus 24³", "Fus 48³", "Fus 96³", "Logits 192³"]


def subtype_key(row_subtype: str, side: str) -> str:
    if row_subtype == "conflicting_right_on_left":
        return "conflicting_left_on_right"
    return row_subtype


# ----------------------------------------------------------------------------- Experiment II
def exp2_heatmap(m: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.4), sharey=True)
    cols = ["reference"] + [k for k, _ in SUBTYPE_ORDER]
    labels = ["Reference\n'Segment the\nside X ...'"] + [l for _, l in SUBTYPE_ORDER]
    for ax, side in zip(axes, ["left", "right"]):
        d = m[m.side == side].copy()
        d["key"] = [subtype_key(s, side) for s in d.variant_subtype]
        mat = np.full((len(S.CORE7), len(cols)), np.nan)
        for i, t in enumerate(S.CORE7):
            dt = d[d.target == t]
            mat[i, 0] = dt.ref_dice_vs_gt.mean()
            for j, k in enumerate(cols[1:], start=1):
                mat[i, j] = dt[dt.key == k].dice_vs_gt.mean()
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] < 0.35 else S.INK)
        ax.set_xticks(range(len(cols)), labels, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(S.CORE7)), [S.short(t) for t in S.CORE7])
        ax.axvline(0.5, color=S.INK, lw=1.5)
        ax.set_title(f"{side.upper()}-side prompts", color=S.LEFT if side == "left" else S.RIGHT)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("Dice vs. GT of the requested side")
    S.suptitle(fig, "Experiment II - Segmentation quality for every re-wording of the same request",
               "Mean over 100 subjects. Green = as good as the reference wording, red = the re-wording broke the segmentation.")
    S.save(fig, OUT, "exp2_01_dice_by_wording_heatmap.png")


def exp2_left_vs_right(m: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(VTYPES))
    w = 0.38
    for ax, col, ylabel, title in [
        (axes[0], "pred_change_1_minus_dice_ref", "Prediction change (1 - Dice vs. reference mask)",
         "How much does the mask move away from the reference wording?"),
        (axes[1], "dice_vs_gt", "Dice vs. GT", "How good is the mask?"),
    ]:
        for k, (side, c) in enumerate([("left", S.LEFT), ("right", S.RIGHT)]):
            vals = [m[(m.side == side) & (m.variant_type == v)][col].mean() for v in VTYPES]
            errs = [S.ci95(m[(m.side == side) & (m.variant_type == v)][col]) for v in VTYPES]
            b = ax.bar(x + (k - 0.5) * w, vals, w, yerr=errs, color=c, capsize=2, label=f"{side} prompts")
            S.bar_labels(ax, b, fontsize=7.5, errs=errs)
        ax.set_xticks(x, [VLABEL[v].replace(" ", "\n", 1) if v == "location_based" else VLABEL[v] for v in VTYPES])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(0, 1)
    axes[0].legend()
    S.suptitle(fig, "Re-wording hurts LEFT-side prompts more than RIGHT-side prompts",
               "Same wording templates on both sides; the left-side versions drift further from the reference and lose more Dice (95% CI, 7 structures x 100 subjects).")
    S.save(fig, OUT, "exp2_02_left_vs_right_wording_sensitivity.png")


def exp2_scatter(s: pd.DataFrame, corr: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, col, title in [(axes[0], "qwen3_cosine_dist_vs_ref", "Frozen Qwen3 text embedding"),
                           (axes[1], "prompt_decoder_cosine_dist_vs_ref", "After VoxTell's prompt decoder")]:
        for v in VTYPES:
            for side, mk in [("left", "o"), ("right", "^")]:
                d = s[(s.variant_type == v) & (s.side == side)]
                ax.scatter(d[col], d.mean_pred_change_1_minus_dice, s=38, marker=mk,
                           color=S.VARIANT_COLORS[v], alpha=0.8, edgecolor="white", lw=0.5)
        rho, p = spearmanr(s[col], s.mean_pred_change_1_minus_dice)
        ax.set_title(f"{title}\nprompt-level Spearman rho = {rho:.2f} (p = {p:.1e})", fontsize=10.5)
        ax.set_xlabel("Cosine distance to the reference prompt (1 - cos)")
    axes[0].set_ylabel("Prediction change (1 - Dice vs. reference mask)")
    handles = [Line2D([], [], marker="o", ls="", color=S.VARIANT_COLORS[v], label=VLABEL[v]) for v in VTYPES]
    handles += [Line2D([], [], marker="o", ls="", color="gray", label="left prompt"),
                Line2D([], [], marker="^", ls="", color="gray", label="right prompt")]
    axes[1].legend(handles=handles, loc="center right", fontsize=8.5)
    notes = [("thalamus", "left", "contradictory", (6, -2)), ("cerebellum white matter", "left", "contradictory", (6, -6)),
             ("caudate", "left", "location_based", (-8, -14))]
    for t, side, v, off in notes:
        r = s[(s.target == t) & (s.side == side) & (s.variant_type == v)].iloc[0]
        label = f"{S.short(t)} ({VLABEL[v].lower()}{', L & R' if v == 'location_based' else ', left'})"
        axes[1].annotate(label, (r.prompt_decoder_cosine_dist_vs_ref, r.mean_pred_change_1_minus_dice),
                         xytext=off, textcoords="offset points", fontsize=7.5, ha="right" if off[0] < 0 else "left")
    S.suptitle(fig, "Does 'how different the words look to the model' predict 'how different the mask is'?",
               "Each point = one prompt, averaged over 100 subjects. Moderate positive relation; contradictory prompts are the outliers - small embedding change, huge mask change.")
    S.save(fig, OUT, "exp2_03_embedding_distance_vs_prediction_change.png")

    fig, ax = plt.subplots(figsize=(11, 4.4))
    c = corr.set_index("target").loc[S.CORE7]
    x = np.arange(len(S.CORE7))
    w = 0.38
    b1 = ax.bar(x - w / 2, c.qwen3_vs_pred_shift_spearman_rho, w, color="#B8B2A7", label="Frozen Qwen3 distance")
    b2 = ax.bar(x + w / 2, c.prompt_dec_vs_pred_shift_spearman_rho, w, color=S.ORGAN, label="Prompt-decoder distance")
    S.bar_labels(ax, b1)
    S.bar_labels(ax, b2)
    ax.set_xticks(x, [S.short(t) for t in S.CORE7])
    ax.set_ylabel("Spearman rho with prediction change")
    ax.set_ylim(0, 1)
    ax.legend()
    S.suptitle(fig, "Embedding distance vs. prediction change, per structure (2,000 prompt x subject pairs each)",
               "The prompt decoder's embedding predicts the mask change at least as well as the raw text embedding for every structure; all p < 1e-40.")
    S.save(fig, OUT, "exp2_04_correlation_by_structure.png")


def exp2_worst_prompts(s: pd.DataFrame, n: int = 16):
    d = s.sort_values("mean_pred_change_1_minus_dice", ascending=False).head(n).iloc[::-1]
    fig = plt.figure(figsize=(16, 0.42 * n + 1.4))
    ax_txt = fig.add_axes([0.0, 0.05, 0.62, 0.88])
    ax_bar = fig.add_axes([0.66, 0.05, 0.32, 0.88])
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(-0.5, n - 0.5)
    ax_txt.axis("off")
    fig.canvas.draw()
    for i, (_, r) in enumerate(d.iterrows()):
        y = (i + 0.5) / n
        ax_txt.text(0.0, y, f"{S.short(r.target)}", transform=ax_txt.transAxes, fontsize=8.5, va="center",
                    fontweight="bold", color=S.INK)
        text = r.prompt_text.replace("Segment the ", "").replace(" in this image.", "")
        S.word_chips(ax_txt, 0.14, y, S.tokenize_prompt(text, r.target), fontsize=8)
    y = np.arange(n)
    ax_bar.barh(y - 0.2, d.mean_dice_vs_gt, 0.4, color=[S.VARIANT_COLORS[v] for v in d.variant_type], label="Dice vs. GT")
    ax_bar.barh(y + 0.2, d.mean_pred_change_1_minus_dice, 0.4, color="#D9D3CA", label="prediction change")
    for yi, a, b in zip(y, d.mean_dice_vs_gt, d.mean_pred_change_1_minus_dice):
        ax_bar.text(a + 0.01, yi - 0.2, f"{a:.2f}", va="center", fontsize=7.5)
        ax_bar.text(b + 0.01, yi + 0.2, f"{b:.2f}", va="center", fontsize=7.5, color="#777")
    ax_bar.set_yticks([])
    ax_bar.set_xlim(0, 1.12)
    ax_bar.set_ylim(-0.5, n - 0.5)
    ax_bar.set_xlabel("colored = Dice vs. GT (color = prompt type), gray = 1 - Dice vs. reference")
    handles = [plt.Rectangle((0, 0), 1, 1, color=S.VARIANT_COLORS[v]) for v in VTYPES]
    ax_bar.legend(handles, [VLABEL[v] for v in VTYPES], loc="lower right", fontsize=7.5)
    n_left = int((d.side == "left").sum())
    S.suptitle(fig, f"The {n} wordings that change VoxTell's mask the most",
               f"Chips: purple = side word, teal = organ word, gray = other words ('Segment the ... in this image.' omitted). {n_left} of {n} are LEFT-side prompts.")
    S.save(fig, OUT, "exp2_05_most_disruptive_wordings.png")


def exp2_examples(subject_id: str = "OASIS_OAS1_0001_MR1", label: str = "", out_dir=None, fname=None):
    npz2 = EXP2 / "masks" / f"{subject_id}_exp2_synonym_masks_packed.npz"
    npz1 = EXP1_MASKS / f"{subject_id}_exp1_masks_packed.npz"
    subj = S.load_subject(subject_id)
    if subj is None or not npz2.exists() or not npz1.exists():
        print("  (skipping exp2 example overlays: volume or packed masks not available locally)")
        return
    img, lab = subj
    z2, z1 = np.load(npz2), np.load(npz1)
    shape = z2["shape"]
    img_n = S.norm_img(img)
    rows = [
        ("caudate", [("ref", "Segment the left caudate ..."), ("OASIS_P0057", "left caudate nucleus"),
                     ("OASIS_P0060", "l. caudate"), ("OASIS_P0062", "Find the caudate that is on the left."),
                     ("OASIS_P0065", "C-shaped basal ganglia nucleus\nbordering the left lateral ventricle"),
                     ("OASIS_P0066", "left caudate located on\nthe right hemisphere")]),
        ("thalamus", [("ref", "Segment the left thalamus ..."), ("OASIS_P0031", "left dorsal thalamus"),
                      ("OASIS_P0033", "left thalamic nucleus"), ("OASIS_P0036", "Find the thalamus that is on the left."),
                      ("OASIS_P0039", "deep relay nucleus in the\nleft dorsal diencephalon"),
                      ("OASIS_P0040", "left thalamus located on\nthe right hemisphere")]),
    ]
    fig, axes = plt.subplots(len(rows), 6, figsize=(16, 3.3 * len(rows)), gridspec_kw=dict(wspace=0.04, hspace=0.38))
    for i, (t, items) in enumerate(rows):
        lid, rid = S.LABEL_PAIRS[t]
        gl = lab == lid
        both = gl | (lab == rid)
        a_idx = int(np.round(np.argwhere(gl)[:, 1].mean()))
        crop = S.square_crop(S.coronal(both, a_idx), margin=10)
        for j, (pid, title) in enumerate(items):
            ax = axes[i, j]
            if pid == "ref":
                mk = S.unpack(z1[f"{t}_canonical_sentence_left"], z1["shape"])
                col = S.ORGAN
            else:
                mk = S.unpack(z2[f"{pid}_mask"], shape)
                col = S.LEFT if j < 4 else (S.VARIANT_COLORS["location_based"] if j == 4 else S.CONTRA)
            dsc = 2 * (mk & gl).sum() / max(mk.sum() + gl.sum(), 1)
            S.overlay(ax, S.coronal(img_n, a_idx), [(S.coronal(mk, a_idx), col), (S.coronal(gl, a_idx), "white")],
                      crop=crop, contour_only=[False, True])
            S.lr_tags(ax)
            ax.set_title(textwrap.fill(f"'{title}'".replace("\n", " "), 30) + f"\nDice {dsc:.2f}", fontsize=8.2)
            if j == 0:
                ax.set_ylabel(S.short(t), fontsize=11, fontweight="bold")
    S.suptitle(fig, f"Example subject {subject_id} ({label}) - the same LEFT request, worded six ways",
               "Filled = VoxTell mask, white outline = left GT (image left = patient left). Reference mask from the identical Exp I run.")
    S.save(fig, out_dir or OUT, fname or f"exp2_06_example_masks_rewordings_{S.subj_tag(subject_id)}.png")


# ----------------------------------------------------------------------------- Experiment III
LAYER_SERIES = [
    ("left_vs_right", "Left vs. Right", S.INK, "-", 2.6),
    ("base_vs_left", "Base vs. Left", S.LEFT, "-", 1.8),
    ("base_vs_right", "Base vs. Right", S.RIGHT, "-", 1.8),
    ("synonym_vs_ref (lateralized)", "Short term vs. ref", S.VARIANT_COLORS["lateralized"], "--", 1.3),
    ("synonym_vs_ref (synonym)", "Synonym vs. ref", S.VARIANT_COLORS["synonym"], "--", 1.3),
    ("synonym_vs_ref (paraphrase)", "Paraphrase vs. ref", S.VARIANT_COLORS["paraphrase"], "--", 1.3),
    ("synonym_vs_ref (location_based)", "Location vs. ref", S.VARIANT_COLORS["location_based"], "--", 1.3),
    ("synonym_vs_ref (contradictory)", "Contradictory vs. ref", S.VARIANT_COLORS["contradictory"], "--", 1.3),
]


def _stage_bands(ax):
    for lo, hi, lab, c in [(-0.5, 1.5, "text encoder\n+ projection", "#EFE9FF"),
                           (1.5, 8.5, "prompt decoder (6 cross-attn layers)", "#E6F6F6"),
                           (8.5, 13.5, "image-decoder fusion (U-Net)", "#FFF1E3")]:
        ax.axvspan(lo, hi, color=c, zorder=0)
        ax.text((lo + hi) / 2, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, color="#555")
    ax.axvspan(11.5, 12.5, color="#DDDDDD", alpha=0.6, hatch="xx", zorder=0, lw=0)
    ax.set_xticks(range(14), STAGE_SHORT, rotation=45, ha="right", fontsize=8.5)
    ax.set_xlim(-0.5, 13.5)


def exp3_trajectory(s3: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4))
    for ax, col, ylabel in [(axes[0], "mean_cosine_distance", "Cosine distance (1 - cos)"),
                            (axes[1], "mean_relative_l2_distance", "Relative L2 distance")]:
        _stage_bands(ax)
        for ct, lab, c, ls, lw in LAYER_SERIES:
            d = s3[s3.comparison_type == ct].sort_values("stage_index")
            ax.plot(d.stage_index, d[col], ls, color=c, lw=lw, marker="o", ms=3.5, label=lab)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
    axes[0].legend(fontsize=8, ncol=2, loc="lower left")
    S.suptitle(fig, "Experiment III - How far apart are two prompts' representations at each of VoxTell's 14 stages?",
               "Mean over 7 structures x 100 subjects (log scale). Solid = side comparisons, dashed = re-wording vs. reference. "
               "Gray hatched 96³ stage: head-averaged fusion map is near zero, so its distance is unstable.")
    S.save(fig, OUT, "exp3_01_layerwise_distance_trajectory.png")


def exp3_signal_to_noise(s3: pd.DataFrame):
    p = s3.pivot(index="stage_index", columns="comparison_type", values="mean_cosine_distance")
    fig, ax = plt.subplots(figsize=(13, 4.8))
    _stage_bands(ax)
    for ct, lab, c in [("synonym_vs_ref (synonym)", "÷ synonym re-wording", S.VARIANT_COLORS["synonym"]),
                       ("synonym_vs_ref (paraphrase)", "÷ paraphrase re-wording", S.VARIANT_COLORS["paraphrase"]),
                       ("synonym_vs_ref (lateralized)", "÷ short-term re-wording", S.VARIANT_COLORS["lateralized"])]:
        r = p["left_vs_right"] / p[ct]
        ax.plot(r.index, r.values, "o-", color=c, lw=2, label=lab)
        for xi, yi in zip(r.index, r.values):
            if ct == "synonym_vs_ref (synonym)":
                ax.text(xi, yi * 1.12, f"{yi:.1f}x", ha="center", fontsize=7.5, color=c)
    ax.axhline(1, color=S.INK, ls="--", lw=1)
    ax.text(0, 1.08, "side gap = wording gap", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Left-vs-Right distance / re-wording distance")
    ax.legend(loc="upper left", fontsize=9)
    S.suptitle(fig, "The side signal is amplified once text meets the image",
               "Ratio of the Left-vs-Right gap to the gap caused by re-wording the same request. In the text stages, left<->right moves the embedding ~3x a synonym swap\n"
               "but no more than a paraphrase; in the U-Net fusion stages it becomes 6-21x a synonym swap (96³ stage unstable, see exp3_01).")
    S.save(fig, OUT, "exp3_02_side_signal_vs_wording_noise.png")


def exp3_base_bias(m3: pd.DataFrame):
    d = m3[m3.comparison_type.isin(["base_vs_left", "base_vs_right"])].copy()
    d["target"] = d.pair_label.str.split(":").str[0]
    g = d.groupby(["target", "stage_index", "comparison_type"]).cosine_distance.mean().unstack()
    bias = (g["base_vs_left"] - g["base_vs_right"]) / (g["base_vs_left"] + g["base_vs_right"])
    mat = np.array([[bias.get((t, k), np.nan) for k in range(14)] for t in S.CORE7])
    fig, ax = plt.subplots(figsize=(14, 4.6))
    im = ax.imshow(mat, cmap="PuOr_r", vmin=-1, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(mat[i, j]) > 0.6 else S.INK)
    ax.set_xticks(range(14), STAGE_SHORT, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(S.CORE7)), [S.short(t) for t in S.CORE7])
    ax.axvline(8.5, color=S.INK, lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("<- base closer to LEFT  |  base closer to RIGHT ->")
    S.suptitle(fig, "Where does the no-side prompt pick a hemisphere?",
               "(d(base,left) - d(base,right)) / sum, cosine distance. Orange = base resembles the RIGHT prompt, purple = the LEFT prompt. The lean is decided at the first image-fusion stage (12³) for 5 of 7 structures.")
    S.save(fig, OUT, "exp3_03_base_prompt_side_lean_by_stage.png")


def exp3_cross_attention(s3: pd.DataFrame):
    d = s3[s3.stage_index.between(2, 7)]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    x = np.arange(6)
    series = [s for s in LAYER_SERIES if s[0] != "synonym_vs_ref (lateralized)"]
    w = 0.8 / len(series)
    for k, (ct, lab, c, _, _) in enumerate(series):
        v = d[d.comparison_type == ct].sort_values("stage_index").mean_cross_attn_cosine_distance.values
        ax.bar(x + (k - len(series) / 2 + 0.5) * w, v, w, color=c, label=lab)
    ax.set_xticks(x, [f"Decoder layer {i}" for i in range(1, 7)])
    ax.set_ylabel("Cosine distance between\ncross-attention maps")
    ax.legend(ncol=4, fontsize=8)
    S.suptitle(fig, "Do two prompts look at different image regions inside the prompt decoder?",
               "Cosine distance between the text query's cross-attention over image tokens. Left vs. Right differs most in layer 1; location descriptions drift late (layers 5-6). All values < 0.04: attention maps stay > 96% similar.")
    S.save(fig, OUT, "exp3_04_cross_attention_difference_by_layer.png")


def exp3_lr_by_structure(m3: pd.DataFrame):
    d = m3[m3.comparison_type == "left_vs_right"].copy()
    d["target"] = d.pair_label.str.split(":").str[0]
    g = d.groupby(["target", "stage_index"]).cosine_distance.mean().unstack()
    mat = g.loc[S.CORE7].values
    norm = mat / mat.max(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(14, 4.6))
    im = ax.imshow(norm, cmap="magma_r", vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.6 else S.INK)
    ax.set_xticks(range(14), STAGE_SHORT, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(S.CORE7)), [S.short(t) for t in S.CORE7])
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("fraction of the structure's peak")
    S.suptitle(fig, "Left-vs-Right separation per structure and stage",
               "Numbers = mean cosine distance between the left and right prompt; color is normalised per row. The four deep nuclei are most separated inside the prompt decoder;\n"
               "the large cerebellar structures separate most at the image-fusion stages. In absolute terms all stages after 12³ shrink, but re-wording noise shrinks much faster (exp3_02).")
    S.save(fig, OUT, "exp3_05_left_right_gap_by_structure.png")


def main():
    S.apply_style()
    m2 = pd.read_csv(EXP2 / "metrics_no_masks.csv")
    s2 = pd.read_csv(EXP2 / "summary.csv")
    corr = pd.read_csv(EXP2 / "embedding_vs_prediction_correlations.csv")
    s3 = pd.read_csv(EXP3 / "summary.csv")
    m3 = pd.read_csv(EXP3 / "metrics.csv", usecols=["target", "comparison_type", "pair_label", "stage_index", "cosine_distance"])
    exp2_heatmap(m2)
    exp2_left_vs_right(m2)
    exp2_scatter(s2, corr)
    exp2_worst_prompts(s2)
    for sid, label in S.EXAMPLE_SUBJECTS:
        exp2_examples(sid, label)
    exp3_trajectory(s3)
    exp3_signal_to_noise(s3)
    exp3_base_bias(m3)
    exp3_cross_attention(s3)
    exp3_lr_by_structure(m3)


if __name__ == "__main__":
    main()
