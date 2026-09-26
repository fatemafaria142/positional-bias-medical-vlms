"""
For every organ, the figures that answer the plan questions 1-3 and 5-11 (row 4 is left out), with the ten subjects
shown separately (nothing is averaged). They are written into the organ's own folder:

    local_plots_groundedness/word_highlight/<AMOS|OASIS>/<organ>/row01_... to row11_...

Subjects are numbered 1..10 on the x axis; their identifiers are not written into the figures.

Usage:  python scripts/make_organ_question_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_plan_question_figures import DARK, DATASETS, LIGHT_GREY, VERSIONS, grouped_bars, load  # noqa
from plot_groundedness_results import decode_mask_rle  # noqa

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "local_plots_groundedness" / "word_highlight"
FOLDER = {"AMOS": "AMOS", "OASIS": "OASIS"}


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def subjects_of(cfg):
    return sorted(load(cfg, "baseline_predictions").subject_id.unique())


def figures_for_organ(dname, cfg, organ, data):
    subs = data["subjects"]
    cats = [str(i + 1) for i in range(len(subs))]
    pos, core = organ.split(None, 1)
    other = ("right " if pos == "left" else "left ") + core
    out = ROOT / FOLDER[dname] / organ.replace(" ", "_")
    kw = dict(xlabel="Subject", width=13.5)
    bp = data["bp"]

    # ---- row 1: does the prompt find the right organ? (overlap with the true organ) ----
    matched, mismatched = [], []
    for s in subs:
        a, b = bp.loc[(s, organ)], bp.loc[(s, other)]
        shape = tuple(int(x) for x in a.volume_shape_ras.split("x"))
        gt = decode_mask_rle(a.gt_mask_rle_b64, shape)
        matched.append(iou(decode_mask_rle(a.pred_mask_rle_b64, shape), gt))
        mismatched.append(iou(decode_mask_rle(b.pred_mask_rle_b64, shape), gt))
    grouped_bars(cfg, cats, [(f'Prompt "{organ}" (correct)', matched, cfg["org"]),
                             (f'Prompt "{other}" (other side)', mismatched, LIGHT_GREY)],
                 "Does the prompt find the right organ?", f"Overlap with the true {organ} (1 = perfect)",
                 lambda v: f"{v:.2f}", out / "row01_finds_the_right_organ.png", ylim=(0, 1.15), label_fs=8.5, **kw)

    # ---- row 2: masks as far apart as the real organs? ----
    r2 = data["r2"]
    pair = f"left {core}/right {core}"
    true_gap, mask_gap = [], []
    for s in subs:
        true_gap.append(abs(bp.loc[(s, f"left {core}")].gt_centroid_vox_2 - bp.loc[(s, f"right {core}")].gt_centroid_vox_2))
        mask_gap.append(float(r2[(r2.subject_id == s) & (r2.organ_pair == pair)].output_centroid_dist_vox.iloc[0]))
    grouped_bars(cfg, cats, [("Between the two true organs", true_gap, DARK),
                             ("Between the two masks the model gave", mask_gap, cfg["org"])],
                 'When "left" is swapped for "right", do the masks end up as far apart as the real organs?',
                 "Distance between the left and right side (voxels)", lambda v: f"{v:.0f}",
                 out / "row02_masks_as_far_apart_as_the_organs.png", label_fs=8, **kw)

    # ---- row 3: mask on the named side ----
    r3 = data["r3"]
    col = "left_prompt_predicts_left_side" if pos == "left" else "right_prompt_predicts_right_side"
    ok = [float(r3[(r3.subject_id == s) & (r3.organ_pair == pair)][col].iloc[0]) for s in subs]
    grouped_bars(cfg, cats, [(f'Prompt "{organ}": the mask is on the named side', ok, cfg["org"])],
                 "Does each prompt give a mask on the side that it names?", "1 = correct side, 0 = wrong side",
                 lambda v: "yes" if v > 0.5 else "no", out / "row03_mask_on_the_named_side.png", ylim=(0, 1.25),
                 **kw)

    # ---- rows 5-7: which word helps more ----
    q = "Which word helps the model more, the organ name or the side?"
    for row, name, label, d, colname, ylab in [
            (5, "exact_shapley", "Exact Shapley", data["wa"], "shapley_value", "Contribution to the Dice score"),
            (6, "lime", "LIME", data["li"], "lime_weight", "LIME weight (its own scale)"),
            (7, "shap", "SHAP", data["sh"], "shap_value", "Contribution to the Dice score")]:
        d = d[d.organ == organ]
        piv = d.pivot_table(index="subject_id", columns="role", values=colname).reindex(subs)
        grouped_bars(cfg, cats, [(f"Organ name ({core})", list(piv["organ"]), cfg["org"]),
                                 (f"Side word ({pos})", list(piv["position"]), cfg["pos"])],
                     q + f"  ({label})", ylab, lambda v: f"{v:.2g}", out / f"row{row:02d}_{name}.png", label_fs=7.5,
                     **kw)

    # ---- rows 8-11: removing a word ----
    ab = data["ab"]
    ab = ab[ab.organ == organ]
    vcol = {"none": DARK, "position": cfg["pos"], "organ": cfg["org"], "organ,position": LIGHT_GREY}
    vname = dict(VERSIONS)

    def per_subject(colname, key):
        piv = ab.pivot_table(index="subject_id", columns="roles_dropped", values=colname).reindex(subs)
        return list(piv[key])

    keys = [k for k, _ in VERSIONS]
    grouped_bars(cfg, cats, [(vname[k], per_subject("f_dice_gt", k), vcol[k]) for k in keys],
                 "What happens to the accuracy when a word is removed?",
                 "Dice score (1 = perfect match with the true organ)", lambda v: f"{v:.2f}",
                 out / "row08_accuracy.png", ylim=(0, 1.15), show_labels=False, **kw)
    keys = keys[1:]
    grouped_bars(cfg, cats, [(vname[k], per_subject("f_dice_self", k), vcol[k]) for k in keys],
                 "How much does the answer change when a word is removed?",
                 "Overlap with the answer for the full prompt (1 = unchanged)", lambda v: f"{v:.2f}",
                 out / "row09_change_from_own_answer.png", ylim=(0, 1.15), show_labels=False, **kw)
    sign = 1 if pos == "left" else -1
    half = [sign * abs(bp.loc[(s, organ)].gt_centroid_vox_2 - bp.loc[(s, other)].gt_centroid_vox_2) / 2 for s in subs]
    grouped_bars(cfg, cats, [("Mask centre without the side word", per_subject("f_centroid_shift", "position"), cfg["pos"])],
                 "Does removing the side word push the mask to the other side?",
                 "Shift of the mask centre (voxels, + = towards the midline)", lambda v: f"{v:+.0f}",
                 out / "row10_left_right_shift.png", marks=half, marks_label="Midline between the two organs",
                 label_fs=8, **kw)
    full = np.array(per_subject("n_pred_voxels", "none"), dtype=float)
    ratio = lambda k: list(np.maximum(np.array(per_subject("n_pred_voxels", k), dtype=float), 1) / full)
    grouped_bars(cfg, cats, [(vname[k], ratio(k), vcol[k]) for k in keys],
                 "Does the mask get bigger or smaller when a word is removed?",
                 "Mask size compared with the full prompt (log scale)", lambda v: f"×{v:.2g}",
                 out / "row11_mask_size.png", log_from=1.0, ref_label="Size with the full prompt",
                 show_labels=False, **kw)


def main():
    for dname, cfg in DATASETS.items():
        bp = load(cfg, "baseline_predictions").set_index(["subject_id", "organ"])
        ab = load(cfg, "ablation_masks")
        ab["roles_dropped"] = ab.roles_dropped.fillna("none")
        wa = load(cfg, "word_attribution")
        sh = load(cfg, "shap_attribution")
        data = dict(subjects=subjects_of(cfg), bp=bp, ab=ab,
                    wa=wa[wa.metric == "f_dice_gt"], sh=sh[sh.metric == "f_dice_gt"], li=load(cfg, "lime_attribution"),
                    r2=load(cfg, "row2_embedding_output_regression"), r3=load(cfg, "row3_causal_embedding_swap"))
        for organ in cfg["organs"]:
            figures_for_organ(dname, cfg, organ, data)
            print("wrote 10 figures for", dname, organ)


if __name__ == "__main__":
    main()
