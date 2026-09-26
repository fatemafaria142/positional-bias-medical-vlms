"""
Plan-question figures for ONE example subject, organ by organ (rows 1-3 and 5-11; row 4 is left out).

Output: local_plots_groundedness/example_subject/<AMOS|OASIS>/
    pair_<organ>/row01_..., row02_..., row03_...        one folder per left/right pair
    <left|right>_<organ>/row05_... to row11_...          one folder per organ
The subject's identifier is used only for reading the data; it is not written into the figures.

Usage:
    python scripts/make_example_subject_figures.py                       # the default example subjects
    python scripts/make_example_subject_figures.py amos amos_0508        # another subject: <amos|oasis> <subject_id>
"""

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_plan_question_figures import (DARK, DATASETS, INK, INK2, LIGHT_GREY, MUTED, SURFACE, VERSIONS, frame,  # noqa
                                        load, patch)
from plot_groundedness_results import canonical_affine, decode_mask_rle  # noqa

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "local_plots_groundedness" / "example_subject"
DEFAULT_SUBJECT = {"AMOS": "amos_0575", "OASIS": "OASIS_OAS1_0419_MR1"}
TRUE_C = "#22c55e"


def new_fig_dir(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------------------------------------
class Subject:
    def __init__(self, cfg, sid):
        self.cfg, self.sid = cfg, sid
        self.bp = load(cfg, "baseline_predictions")
        self.bp = self.bp[self.bp.subject_id == sid].set_index("organ")
        self._vol = {}

    def volume(self, organ):
        row = self.bp.loc[organ]
        key = row.image_path
        if key not in self._vol:
            img = nib.load(str(REPO / row.image_path))
            image = img.get_fdata().astype(np.float32)
            axc = nib.aff2axcodes(img.affine)
            axis = next((a for a, c in enumerate(axc) if c in ("S", "I")), int(np.argmin(image.shape)))
            label = nib.load(str(REPO / row.label_path)).get_fdata().astype(int)
            self._vol[key] = (img, image, label, axis, canonical_affine(REPO / row.image_path))
        return self._vol[key]

    def true_mask(self, organ):
        _, _, label, _, _ = self.volume(organ)
        return label == int(self.bp.loc[organ].label_id)

    def predicted_mask(self, organ):
        """Baseline prediction, resampled onto the raw image grid so it lines up with the scan."""
        row = self.bp.loc[organ]
        img, _, _, _, affine = self.volume(organ)
        shape = tuple(int(s) for s in row.volume_shape_ras.split("x"))
        m = decode_mask_rle(row.pred_mask_rle_b64, shape)
        nii = nib.Nifti1Image(m.transpose(2, 1, 0).astype(np.uint8), affine)
        return nibabel.processing.resample_from_to(nii, img, order=0).get_fdata().astype(bool)


def crop_box(mask2d, pad=28):
    ys, xs = np.nonzero(mask2d)
    h, w = mask2d.shape
    return (max(ys.min() - pad, 0), min(ys.max() + pad, h), max(xs.min() - pad, 0), min(xs.max() + pad, w))


def show(ax, img2d, box, masks):
    """masks: [(mask2d, colour, filled)]"""
    y0, y1, x0, x1 = box
    ax.imshow(img2d[y0:y1, x0:x1], cmap="gray", origin="lower")
    for m, col, filled in masks:
        m = m[y0:y1, x0:x1]
        if filled:
            rgba = np.zeros((*m.shape, 4))
            rgba[..., :3] = matplotlib.colors.to_rgb(col)
            rgba[..., 3] = np.where(m, 0.55, 0.0)
            ax.imshow(rgba, origin="lower")
        if m.any():
            ax.contour(m, levels=[0.5], colors=col if not filled else "white", linewidths=1.6 if not filled else 1.0)
    ax.axis("off")


def slices(sub, organs):
    """The 2D slice through the middle of the given organs, and a crop box around them."""
    _, image, _, axis, _ = sub.volume(organs[0])
    union = np.zeros(image.shape, bool)
    for o in organs:
        union |= sub.true_mask(o)
    coords = np.nonzero(union)
    k = int(round(coords[axis].mean()))
    idx = [slice(None)] * 3
    idx[axis] = k
    idx = tuple(idx)
    img2 = image[idx].T
    box = crop_box(union[idx].T)
    return idx, img2, box


# ------------------------------------------------------------------------------------------------
def row1(sub, core, out):
    cfg = sub.cfg
    left, right = f"left {core}", f"right {core}"
    idx, img2, box = slices(sub, [left, right])
    true_l = sub.true_mask(left)[idx].T
    pred_l, pred_r = sub.predicted_mask(left)[idx].T, sub.predicted_mask(right)[idx].T
    r1 = load(cfg, "row1_image_text_alignment")
    r1 = r1[(r1.subject_id == sub.sid) & (r1.organ_pair == f"{left}/{right}")].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    show(axes[0], img2, box, [(true_l, TRUE_C, True)])
    show(axes[1], img2, box, [(pred_l, cfg["org"], True), (true_l, TRUE_C, False)])
    show(axes[2], img2, box, [(pred_r, cfg["pos"], True), (true_l, TRUE_C, False)])
    caps = [(f"The true {left}", None),
            (f'Prompt: "{left}"', f"overlap with the true organ: {r1.iou_matched:.2f}"),
            (f'Prompt: "{right}"', f"overlap with the true organ: {r1.iou_mismatched:.2f}")]
    for ax, (t, s) in zip(axes, caps):
        ax.set_title(t, fontsize=12.5, fontweight="bold", pad=8)
        if s:
            ax.text(0.5, -0.05, s, transform=ax.transAxes, ha="center", va="top", fontsize=11, color=INK2)
    frame(fig, cfg, "Does the prompt find the right organ?",
          [patch(TRUE_C), patch(cfg["org"]), patch(cfg["pos"])],
          ["True organ (outline)", "Mask for the correct prompt", "Mask for the prompt of the other side"])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.2, wspace=0.05)
    fig.savefig(new_fig_dir(out), dpi=160)
    plt.close(fig)


def row3(sub, core, out):
    cfg = sub.cfg
    left, right = f"left {core}", f"right {core}"
    idx, img2, box = slices(sub, [left, right])
    true_l, true_r = sub.true_mask(left)[idx].T, sub.true_mask(right)[idx].T
    pred_l, pred_r = sub.predicted_mask(left)[idx].T, sub.predicted_mask(right)[idx].T
    r3 = load(cfg, "row3_causal_embedding_swap")
    r3 = r3[(r3.subject_id == sub.sid) & (r3.organ_pair == f"{left}/{right}")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    show(axes[0], img2, box, [(pred_l, cfg["org"], True), (true_l, TRUE_C, False), (true_r, LIGHT_GREY, False)])
    show(axes[1], img2, box, [(pred_r, cfg["pos"], True), (true_r, TRUE_C, False), (true_l, LIGHT_GREY, False)])
    ok = lambda b: "correct side" if b else "wrong side"
    for ax, t, good in [(axes[0], left, r3.left_prompt_predicts_left_side), (axes[1], right, r3.right_prompt_predicts_right_side)]:
        ax.set_title(f'Prompt: "{t}"', fontsize=12.5, fontweight="bold", pad=8)
        ax.text(0.5, -0.05, f"the mask lands on the {ok(bool(good))}", transform=ax.transAxes, ha="center", va="top",
                fontsize=11, color=INK2)
    frame(fig, cfg, "Does each prompt give a mask on the side that it names?",
          [patch(TRUE_C), patch(LIGHT_GREY), patch(cfg["org"]), patch(cfg["pos"])],
          ["True organ named by the prompt", "True organ on the other side", "Mask for the left prompt",
           "Mask for the right prompt"], ncol=2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.24, wspace=0.05)
    fig.savefig(new_fig_dir(out), dpi=160)
    plt.close(fig)


def row2(sub, core, out):
    cfg = sub.cfg
    left, right = f"left {core}", f"right {core}"
    r2 = load(cfg, "row2_embedding_output_regression")
    r2 = r2[(r2.subject_id == sub.sid) & (r2.organ_pair == f"{left}/{right}")].iloc[0]
    true_gap = abs(sub.bp.loc[left].gt_centroid_vox_2 - sub.bp.loc[right].gt_centroid_vox_2)
    vals = [true_gap, r2.output_centroid_dist_vox]
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    bars = ax.bar([0, 1], vals, 0.55, color=[DARK, cfg["org"]], edgecolor=SURFACE, linewidth=1.5)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=11,
                    xytext=(0, 3), textcoords="offset points")
    ax.set_xticks([])
    ax.set_ylabel("Distance between the left and right side (voxels)", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.18)
    frame(fig, cfg, 'When "left" is swapped for "right", do the masks end up as far apart as the real organs?',
          [patch(DARK), patch(cfg["org"])], ["Between the two true organs", "Between the two masks the model gave"])
    fig.text(0.5, 0.875, f"{core.capitalize()}, difference between the two prompts (text): {r2.text_embed_cos_dist:.3f}",
             ha="center", va="top", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 0.86))
    fig.savefig(new_fig_dir(out), dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
def bars(sub, organ, question, sub_line, labels, values, colors, ylabel, fmt, out, marks=None, marks_label=None,
         log_from=None, ylim=None):
    from matplotlib.lines import Line2D
    cfg = sub.cfg
    fig, ax = plt.subplots(figsize=(8.6, 5.9))
    x = np.arange(len(values))
    values = np.array(values, dtype=float)
    if log_from:
        ax.set_yscale("log")
        b = ax.bar(x, values - log_from, 0.6, bottom=log_from, color=colors, edgecolor=SURFACE, linewidth=1.5)
    else:
        b = ax.bar(x, values, 0.6, color=colors, edgecolor=SURFACE, linewidth=1.5)
    for bar, v in zip(b, values):
        up = v >= (log_from or 0)
        ax.annotate(fmt(v), (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom" if up else "top",
                    fontsize=11, xytext=(0, 3 if up else -3), textcoords="offset points")
    extra = []
    if marks is not None:
        ax.hlines(marks, x[0] - 0.4, x[-1] + 0.4, color=DARK, linewidth=1.8, linestyle=(0, (3, 2)), zorder=4)
        extra.append((Line2D([0], [0], color=DARK, linewidth=1.8, linestyle=(0, (3, 2))), marks_label))
    if log_from:
        ax.axhline(log_from, color=DARK, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
        extra.append((Line2D([0], [0], color=DARK, linewidth=1.1, linestyle=(0, (4, 3))), "Size with the full prompt"))
        ax.set_ylim(min(values) / 2.5, max(values) * 2.5)
        ticks = [t for t in (0.001, 0.01, 0.1, 1, 10, 100, 1000) if min(values) / 2.5 <= t <= max(values) * 2.5]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"×{t:g}" for t in ticks])
        ax.minorticks_off()
    else:
        ax.axhline(0, color=INK2, linewidth=0.8, zorder=2)
        lo, hi = min(0, values.min(), marks if marks is not None else 0), max(0, values.max(), marks if marks is not None else 0)
        span = (hi - lo) or 1.0
        ax.set_ylim(ylim if ylim else (lo - 0.15 * span if lo < 0 else 0, hi + 0.15 * span))
    ax.set_xticks([])
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.grid(axis="x", visible=False)
    handles = [patch(c) for c in colors] + [h for h, _ in extra]
    names = list(labels) + [l for _, l in extra]
    frame(fig, cfg, question, handles, names, ncol=2 if len(handles) > 3 else None)
    fig.text(0.5, 0.875, sub_line, ha="center", va="top", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.12 if len(handles) > 3 else 0.07, 1, 0.86))
    fig.savefig(new_fig_dir(out), dpi=160)
    plt.close(fig)


def organ_rows(sub, organ, out_dir):
    cfg = sub.cfg
    pos, core = organ.split(None, 1)
    words = [f"Organ name ({core})", f"Side word ({pos})"]
    wcol = [cfg["org"], cfg["pos"]]
    wa, sh, li = load(cfg, "word_attribution"), load(cfg, "shap_attribution"), load(cfg, "lime_attribution")
    sel = lambda d: d[(d.subject_id == sub.sid) & (d.organ == organ)]
    for row, name, label, d, col, ylab in [
            (5, "exact_shapley", "Exact Shapley", sel(wa)[sel(wa).metric == "f_dice_gt"], "shapley_value", "Contribution to the Dice score"),
            (6, "lime", "LIME", sel(li), "lime_weight", "LIME weight (its own scale)"),
            (7, "shap", "SHAP", sel(sh)[sel(sh).metric == "f_dice_gt"], "shap_value", "Contribution to the Dice score")]:
        v = d.set_index("role")[col]
        bars(sub, organ, "Which word helps the model more, the organ name or the side?",
             f"{organ.capitalize()}  |  Method: {label}", words, [v["organ"], v["position"]], wcol, ylab,
             lambda x: f"{x:.3g}", out_dir / f"row{row:02d}_{name}.png")

    ab = load(cfg, "ablation_masks")
    ab = ab[(ab.subject_id == sub.sid) & (ab.organ == organ)].copy()
    ab["roles_dropped"] = ab.roles_dropped.fillna("none")
    ab = ab.set_index("roles_dropped")
    vcol = {"none": DARK, "position": cfg["pos"], "organ": cfg["org"], "organ,position": LIGHT_GREY}
    vname = dict(VERSIONS)
    keys = [k for k, _ in VERSIONS]
    bars(sub, organ, "What happens to the accuracy when a word is removed?", organ.capitalize(),
         [vname[k] for k in keys], [ab.loc[k, "f_dice_gt"] for k in keys], [vcol[k] for k in keys],
         "Dice score (1 = perfect match with the true organ)", lambda x: f"{x:.2f}", out_dir / "row08_accuracy.png",
         ylim=(0, 1.15))
    keys = keys[1:]
    bars(sub, organ, "How much does the answer change when a word is removed?", organ.capitalize(),
         [vname[k] for k in keys], [ab.loc[k, "f_dice_self"] for k in keys], [vcol[k] for k in keys],
         "Overlap with the answer for the full prompt (1 = unchanged)", lambda x: f"{x:.2f}",
         out_dir / "row09_change_from_own_answer.png", ylim=(0, 1.15))

    other = ("right " if pos == "left" else "left ") + core
    half = abs(sub.bp.loc[organ].gt_centroid_vox_2 - sub.bp.loc[other].gt_centroid_vox_2) / 2
    sign = 1 if pos == "left" else -1
    bars(sub, organ, "Does removing the side word push the mask to the other side?", organ.capitalize(),
         ["Mask centre without the side word"], [ab.loc["position", "f_centroid_shift"]], [cfg["pos"]],
         "Shift of the mask centre (voxels, + = towards the midline)", lambda x: f"{x:+.0f}",
         out_dir / "row10_left_right_shift.png", marks=sign * half, marks_label="Midline between the two organs")
    full = ab.loc["none", "n_pred_voxels"]
    bars(sub, organ, "Does the mask get bigger or smaller when a word is removed?", organ.capitalize(),
         [vname[k] for k in keys], [max(ab.loc[k, "n_pred_voxels"], 1) / full for k in keys], [vcol[k] for k in keys],
         "Mask size compared with the full prompt (log scale)",
         lambda v: f"×{v:.2f}" if v < 10 else f"×{v:.0f}", out_dir / "row11_mask_size.png", log_from=1.0)


def build(folder, sid=None):
    cfg = DATASETS[folder]
    sid = sid or DEFAULT_SUBJECT[folder]
    sub = Subject(cfg, sid)
    root = OUT / folder
    cores = [c for c in cfg["pair_order"] if c not in cfg["not_segmented"]]
    for core in cores:
        d = root / f"pair_{core.replace(' ', '_')}"
        row1(sub, core, d / "row01_finds_the_right_organ.png")
        row2(sub, core, d / "row02_masks_as_far_apart_as_the_organs.png")
        row3(sub, core, d / "row03_mask_on_the_named_side.png")
    for organ in cfg["organs"]:
        organ_rows(sub, organ, root / organ.replace(" ", "_"))
    print("wrote", folder, "for the example subject (", len(cores) * 3 + len(cfg["organs"]) * 7, "figures )")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        build({"amos": "AMOS", "oasis": "OASIS"}[sys.argv[1]], sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        for f in ("AMOS", "OASIS"):
            build(f)
