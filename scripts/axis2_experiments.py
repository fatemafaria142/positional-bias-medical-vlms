"""
Axis II: Sensitivity to Prompt Variation — Complete Experimental Suite
for VoxTell on Neurite-OASIS (100 subjects).

Implements all experiments and deliverables from `Axis_II_Prompt_Sensitivity_Nipu.docx`:
  - Experiment I (`run_experiment_1_side_words`):
      Base vs. Left vs. Right vs. Contradictory & Absent-Organ prompts.
      Computes pairwise Dice/IoU (base-vs-left, base-vs-right, left-vs-right),
      predicted volume differences, centroid shifts, midline side classification,
      Dice/IoU against left/right/bilateral GT, and failure-criterion flags.
  - Experiment II (`run_experiment_2_synonym_shift`):
      Synonym & Paraphrase shift analysis across 9 variants per side per target
      (5 clinical synonyms + 3 paraphrases + 1 anatomical-location wording + short term).
      Measures embedding distance (cosine, L2, relative L2, Pearson r) at BOTH
      the frozen Qwen3-Embedding-4B output and after the Prompt Decoder (`transformer_decoder`),
      prediction change (Dice/IoU/volume/centroid shift vs. reference prompt and vs. GT),
      Spearman & Pearson correlation between embedding shift and prediction shift,
      rank-order agreement, and spatial similarity maps.
  - Experiment III (`run_experiment_3_layer_localization`):
      Layer-wise localization of side information (Left vs. Right) and wording
      information (Synonym vs. Reference) across all 14 stages of VoxTell:
        Stage 0: `qwen3_output` (Frozen Qwen3-Embedding-4B, 2560-d)
        Stage 1: `project_text_embed` (Initial query state `tgt0`, 2048-d)
        Stages 2..7: `transformer_decoder.layers.0..5` (Query states + Cross-attention maps)
        Stage 8: `transformer_decoder.norm` (Final `mask_embedding`, 2048-d)
        Stages 9..13: `decoder_fusion_stage_0..4` (Multi-scale U-Net einsum fusion maps)
  - Experiment IV (`run_experiment_4_mc_dropout_uncertainty`):
      Monte Carlo (MC) Dropout Uncertainty Estimation (N stochastic forward passes
      with dropout layers unfrozen in `transformer_decoder` + optional spatial dropout).
      Computes per-voxel and per-mask mean probability, predictive variance (epistemic),
      mutual information (epistemic), average per-pass binary entropy (aleatoric),
      predictive entropy (total), coefficient of variation across passes, and ECE.
      Compares uncertainty across Base, Lateralized, Synonyms, Contradictory, and
      Absent-Organ prompts.
  - Experiment V (`run_experiment_5_decoder_ablation`):
      4-way Image vs. Text Decoder Ablation:
        1. `both_on`    (Image Decoder = ON,  Text Decoder = ON)
        2. `image_only` (Image Decoder = ON,  Text Decoder = OFF)
        3. `text_only`  (Image Decoder = OFF, Text Decoder = ON)
        4. `both_off`   (Image Decoder = OFF, Text Decoder = OFF)
      Repeats Experiments I and II under all 4 settings to isolate which pathway
      carries laterality and synonym sensitivity.

Every experiment saves its outputs into its own dedicated folder under `results_root`
with CSV (`metrics.csv`, `summary.csv`), JSON (`results.json`, `summary.json`),
Markdown (`REPORT.md`), and Masks (`masks/*.npz`, RLE in CSV/JSON, and `.nii.gz`).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy import stats

from axis2_prompts import (
    ALL_16_STRUCTURES,
    CORE_7_STRUCTURES,
    EXTENDED_11_STRUCTURES,
    OASIS_BILATERAL_PAIRS,
    PromptRecord,
    build_oasis_prompt_catalog,
)
from axis2_voxtell_engine import (
    RL_AXIS,
    ExperimentOutputManager,
    centroid_distance,
    decode_mask_rle,
    dice_score,
    encode_mask_rle,
    iou_score,
    load_seg_ras,
    load_volume_ras,
    logits_to_full_masks,
    mask_geometry,
    preprocess_volume,
    rows_to_markdown_table,
    run_sliding_window_custom,
    shape_to_str,
    vector_distances,
)


# ---------------------------------------------------------------------------
# Dataset Discovery Helper (Reads Volumes from HF Dir + Synonym Manifest from Git)
# ---------------------------------------------------------------------------
def discover_oasis_subjects(
    neurite_oasis_dir: Union[str, Path],
    max_subjects: Optional[int] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    subject_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Locate Neurite-OASIS subjects downloaded from Hugging Face (`neurite_oasis_dir/volumes`).
    Returns ordered list of dicts: `{'subject_id', 'image_path', 'label_path'}`.
    """
    root = Path(neurite_oasis_dir)
    vol_dir = root / "volumes"
    if not vol_dir.exists():
        raise FileNotFoundError(f"Volumes directory not found: {vol_dir}")

    img_files = sorted(vol_dir.glob("*_image.nii.gz"))
    items: List[Dict[str, Any]] = []
    wanted = set(subject_ids) if subject_ids else None

    for img_p in img_files:
        sid = img_p.name[: -len("_image.nii.gz")]
        if wanted is not None and sid not in wanted:
            continue
        lbl_p = vol_dir / f"{sid}_label.nii.gz"
        if not lbl_p.exists():
            continue
        items.append({"subject_id": sid, "image_path": img_p, "label_path": lbl_p})

    if end_index is not None:
        items = items[start_index:end_index]
    elif start_index > 0:
        items = items[start_index:]

    if max_subjects is not None:
        items = items[:max_subjects]

    return items


def resolve_structures(scope: str = "core_7") -> List[str]:
    if scope == "core_7":
        return list(CORE_7_STRUCTURES)
    if scope == "extended_11":
        return list(EXTENDED_11_STRUCTURES)
    if scope == "all_16":
        return list(ALL_16_STRUCTURES)
    # Comma-separated custom list
    custom = [s.strip() for s in scope.split(",") if s.strip() in OASIS_BILATERAL_PAIRS]
    if not custom:
        raise ValueError(f"Unknown structure scope {scope!r}. Choose 'core_7', 'extended_11', or 'all_16'.")
    return custom


# ---------------------------------------------------------------------------
# Unified Single-Pass Subject Evaluator for Experiments I, II, and III
# ---------------------------------------------------------------------------
def evaluate_subject_prompts_and_internals(
    predictor,
    subject_info: Dict[str, Any],
    prompt_records: List[PromptRecord],
    image_decoder_on: bool = True,
    text_decoder_on: bool = True,
    capture_internals: bool = True,
    return_probs: bool = False,
) -> Dict[str, Any]:
    """
    Run all `prompt_records` on a single subject in ONE encoder pass per sliding-window patch.
    Returns masks, probabilities (optional), ground-truth masks, and internal layer activations.
    """
    img_arr, img_props = load_volume_ras(subject_info["image_path"])
    lbl_arr, _ = load_seg_ras(subject_info["label_path"])
    data, bbox, orig_shape = preprocess_volume(predictor, img_arr)

    prompt_texts = [pr.prompt_text for pr in prompt_records]
    text_embeddings = predictor.embed_text_prompts(prompt_texts)  # (1, N, 2560)

    logits, internals = run_sliding_window_custom(
        predictor,
        data,
        text_embeddings,
        image_decoder_on=image_decoder_on,
        text_decoder_on=text_decoder_on,
        enable_mc_dropout=False,
        capture_internals=capture_internals,
    )

    if return_probs:
        masks, probs = logits_to_full_masks(logits, bbox, orig_shape, return_probs=True)
    else:
        masks = logits_to_full_masks(logits, bbox, orig_shape, return_probs=False)
        probs = None

    return {
        "subject_id": subject_info["subject_id"],
        "image_path": subject_info["image_path"],
        "label_path": subject_info["label_path"],
        "img_arr": img_arr,
        "lbl_arr": lbl_arr,
        "orig_shape": orig_shape,
        "bbox": bbox,
        "masks": masks,
        "probs": probs,
        "logits_cropped": logits,
        "internals": internals,
    }


# ===========================================================================
# EXPERIMENT I: Does the Side Word Change the Output? (Base / Left / Right / Contradictory)
# ===========================================================================
def run_experiment_1_side_words(
    predictor,
    neurite_oasis_dir: Union[str, Path],
    results_root: Union[str, Path],
    structure_scope: str = "core_7",
    max_subjects: Optional[int] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    save_nifti_subjects: int = 3,
    save_rle_in_csv: bool = True,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Experiment I (Section 2 of Axis_II_Prompt_Sensitivity_Nipu.docx):
      - Runs Base (no side), Left, Right, Contradictory, and Absent-Organ prompts on each scan.
      - Computes:
          * Dice and IoU between each pair of masks (`base_vs_left`, `base_vs_right`, `left_vs_right`)
          * Predicted volume (`n_voxels`) and volume differences
          * Centroid shift (voxels and normalized by volume diagonal) & midline side (`L` vs `R`)
          * Dice and IoU against Left GT, Right GT, and Combined Bilateral GT
          * Contradictory / Wrong-Side behavior (Dice of `"left X"` on Right GT and `"right X"` on Left GT,
            plus self-contradictory prompt `"left X on right hemisphere"` and absent-organ hallucination rate)
          * Failure criterion flag (`outputs_nearly_identical`: pairwise Dice > 0.90 across base/left/right)
    """
    mgr = ExperimentOutputManager(results_root, "exp1_side_word_comparison")
    structures = resolve_structures(structure_scope)
    all_catalog = build_oasis_prompt_catalog(structures=structures, include_absent_negatives=True)

    # Filter to prompts used in Experiment I:
    # base (canonical_sentence & short_term), lateralized (canonical_sentence & short_term),
    # contradictory, and absent_organ_negative
    exp1_prompts = [
        pr for pr in all_catalog
        if pr.variant_type in ("base", "lateralized", "contradictory", "absent_organ_negative")
    ]
    subjects = discover_oasis_subjects(
        neurite_oasis_dir, max_subjects=max_subjects, start_index=start_index, end_index=end_index
    )

    all_rows: List[Dict[str, Any]] = []
    for s_idx, s_info in enumerate(subjects):
        sid = s_info["subject_id"]
        if resume and mgr.has_subject_checkpoint(sid):
            ckpt = mgr.load_subject_checkpoint(sid)
            all_rows.extend(ckpt["rows"])
            print(f"[Exp1][resume {s_idx + 1}/{len(subjects)}] {sid} ({len(ckpt['rows'])} rows)")
            continue

        t0 = time.time()
        bundle = evaluate_subject_prompts_and_internals(
            predictor, s_info, exp1_prompts, capture_internals=False, return_probs=False
        )
        lbl_arr = bundle["lbl_arr"]
        masks = bundle["masks"]
        orig_shape = bundle["orig_shape"]
        shape_s = shape_to_str(orig_shape)

        # Index masks by (target, side, variant_type, variant_subtype)
        mask_by_key: Dict[Tuple[str, str, str, str], np.ndarray] = {}
        pr_by_key: Dict[Tuple[str, str, str, str], PromptRecord] = {}
        for i, pr in enumerate(exp1_prompts):
            key = (pr.target, pr.side, pr.variant_type, pr.variant_subtype)
            mask_by_key[key] = masks[i]
            pr_by_key[key] = pr

        subj_rows: List[Dict[str, Any]] = []
        npz_dict: Dict[str, np.ndarray] = {}

        # 1. Evaluate each bilateral structure across Base / Left / Right / Contradictory
        for target in structures:
            left_id, right_id = OASIS_BILATERAL_PAIRS[target]
            gt_left = (lbl_arr == left_id)
            gt_right = (lbl_arr == right_id)
            gt_both = np.logical_or(gt_left, gt_right)

            geom_gt_l = mask_geometry(gt_left)
            geom_gt_r = mask_geometry(gt_right)
            # Empirical anatomical midline between Left and Right GT centroids along RL_AXIS
            if not geom_gt_l["is_empty"] and not geom_gt_r["is_empty"]:
                rl_midline = 0.5 * (geom_gt_l["centroid_2"] + geom_gt_r["centroid_2"])
            else:
                rl_midline = float(orig_shape[RL_AXIS]) / 2.0

            for tmpl_style in ("canonical_sentence", "short_term"):
                m_base = mask_by_key[(target, "none", "base", tmpl_style)]
                m_left = mask_by_key[(target, "left", "lateralized", tmpl_style)]
                m_right = mask_by_key[(target, "right", "lateralized", tmpl_style)]
                pr_base = pr_by_key[(target, "none", "base", tmpl_style)]
                pr_left = pr_by_key[(target, "left", "lateralized", tmpl_style)]
                pr_right = pr_by_key[(target, "right", "lateralized", tmpl_style)]

                g_base = mask_geometry(m_base, rl_midline=rl_midline)
                g_left = mask_geometry(m_left, rl_midline=rl_midline)
                g_right = mask_geometry(m_right, rl_midline=rl_midline)

                # Pairwise mask comparisons (base vs left, base vs right, left vs right)
                dice_bl = dice_score(m_base, m_left)
                dice_br = dice_score(m_base, m_right)
                dice_lr = dice_score(m_left, m_right)
                iou_bl = iou_score(m_base, m_left)
                iou_br = iou_score(m_base, m_right)
                iou_lr = iou_score(m_left, m_right)

                shift_lr_vox, shift_lr_norm = centroid_distance(g_left, g_right, orig_shape)
                shift_bl_vox, shift_bl_norm = centroid_distance(g_base, g_left, orig_shape)
                shift_br_vox, shift_br_norm = centroid_distance(g_base, m_right_geom := g_right, orig_shape)

                # Failure criterion from Axis II doc Section 2:
                # "If the three outputs are nearly identical, the location words change nothing."
                nearly_identical = bool(dice_bl > 0.90 and dice_br > 0.90 and dice_lr > 0.90)
                lr_disjoint = bool(dice_lr < 0.05)

                # Also check contradictory prompts (only defined once per side, compare with canonical)
                m_contra_l = mask_by_key[(target, "left", "contradictory", "conflicting_left_on_right")]
                m_contra_r = mask_by_key[(target, "right", "contradictory", "conflicting_right_on_left")]
                pr_contra_l = pr_by_key[(target, "left", "contradictory", "conflicting_left_on_right")]
                pr_contra_r = pr_by_key[(target, "right", "contradictory", "conflicting_right_on_left")]

                variants_to_record = [
                    ("base", "none", pr_base, m_base, g_base),
                    ("left", "left", pr_left, m_left, g_left),
                    ("right", "right", pr_right, m_right, g_right),
                ]
                if tmpl_style == "canonical_sentence":
                    variants_to_record.extend([
                        ("contradictory_left_on_right", "left", pr_contra_l, m_contra_l, mask_geometry(m_contra_l, rl_midline=rl_midline)),
                        ("contradictory_right_on_left", "right", pr_contra_r, m_contra_r, mask_geometry(m_contra_r, rl_midline=rl_midline)),
                    ])

                for cond_label, side_tag, pr_obj, m_arr, g_info in variants_to_record:
                    d_gt_l = dice_score(m_arr, gt_left)
                    d_gt_r = dice_score(m_arr, gt_right)
                    d_gt_both = dice_score(m_arr, gt_both)
                    i_gt_l = iou_score(m_arr, gt_left)
                    i_gt_r = iou_score(m_arr, gt_right)
                    i_gt_both = iou_score(m_arr, gt_both)

                    if side_tag == "left":
                        dice_matched_gt = d_gt_l
                        dice_contradictory_gt = d_gt_r
                        iou_matched_gt = i_gt_l
                        iou_contradictory_gt = i_gt_r
                        gt_geom_matched = geom_gt_l
                        wrong_side_error = bool(d_gt_r > d_gt_l and d_gt_r > 0.1)
                    elif side_tag == "right":
                        dice_matched_gt = d_gt_r
                        dice_contradictory_gt = d_gt_l
                        iou_matched_gt = i_gt_r
                        iou_contradictory_gt = i_gt_l
                        gt_geom_matched = geom_gt_r
                        wrong_side_error = bool(d_gt_l > d_gt_r and d_gt_l > 0.1)
                    else:
                        dice_matched_gt = d_gt_both
                        dice_contradictory_gt = np.nan
                        iou_matched_gt = i_gt_both
                        iou_contradictory_gt = np.nan
                        gt_geom_matched = mask_geometry(gt_both, rl_midline=rl_midline)
                        wrong_side_error = False

                    err_vox, err_norm = centroid_distance(g_info, gt_geom_matched, orig_shape)

                    row = {
                        "subject_id": sid,
                        "target": target,
                        "template_style": tmpl_style,
                        "condition": cond_label,
                        "side": side_tag,
                        "prompt_id": pr_obj.prompt_id,
                        "prompt_text": pr_obj.prompt_text,
                        "volume_shape_ras": shape_s,
                        "n_pred_voxels": g_info["n_voxels"],
                        "n_gt_left_voxels": geom_gt_l["n_voxels"],
                        "n_gt_right_voxels": geom_gt_r["n_voxels"],
                        "vol_diff_left_minus_right": g_left["n_voxels"] - g_right["n_voxels"],
                        "vol_diff_base_minus_left": g_base["n_voxels"] - g_left["n_voxels"],
                        "vol_diff_base_minus_right": g_base["n_voxels"] - g_right["n_voxels"],
                        "dice_base_vs_left": round(dice_bl, 4),
                        "dice_base_vs_right": round(dice_br, 4),
                        "dice_left_vs_right": round(dice_lr, 4),
                        "iou_base_vs_left": round(iou_bl, 4),
                        "iou_base_vs_right": round(iou_br, 4),
                        "iou_left_vs_right": round(iou_lr, 4),
                        "centroid_shift_left_vs_right_vox": shift_lr_vox,
                        "centroid_shift_left_vs_right_norm": shift_lr_norm,
                        "centroid_shift_base_vs_left_vox": shift_bl_vox,
                        "centroid_shift_base_vs_right_vox": shift_br_vox,
                        "dice_vs_gt_left": round(d_gt_l, 4),
                        "dice_vs_gt_right": round(d_gt_r, 4),
                        "dice_vs_gt_bilateral": round(d_gt_both, 4),
                        "dice_vs_matched_gt": round(dice_matched_gt, 4),
                        "dice_vs_contradictory_opposite_gt": round(dice_contradictory_gt, 4) if not math.isnan(dice_contradictory_gt) else np.nan,
                        "iou_vs_matched_gt": round(iou_matched_gt, 4),
                        "iou_vs_contradictory_opposite_gt": round(iou_contradictory_gt, 4) if not math.isnan(iou_contradictory_gt) else np.nan,
                        "centroid_error_vs_matched_gt_vox": err_vox,
                        "centroid_error_vs_matched_gt_norm": err_norm,
                        "pred_centroid_0": g_info["centroid_0"],
                        "pred_centroid_1": g_info["centroid_1"],
                        "pred_centroid_2": g_info["centroid_2"],
                        "pred_side_midline": g_info["side_rl"],
                        "frac_voxels_right_of_midline": g_info["frac_right_of_midline"],
                        "frac_voxels_left_of_midline": g_info["frac_left_of_midline"],
                        "wrong_side_error": wrong_side_error,
                        "failure_nearly_identical_flag": nearly_identical,
                        "left_right_disjoint_flag": lr_disjoint,
                    }
                    if save_rle_in_csv:
                        row["pred_mask_rle_b64"] = encode_mask_rle(m_arr)
                    subj_rows.append(row)

                    slug = f"{target.replace(' ', '_')}_{tmpl_style}_{cond_label}"
                    npz_dict[slug] = np.packbits(m_arr.ravel())

                    if s_idx < save_nifti_subjects and tmpl_style == "canonical_sentence" and cond_label in ("base", "left", "right"):
                        mgr.save_nifti(
                            f"{sid}_{target.replace(' ', '_')}_{cond_label}.nii.gz",
                            m_arr,
                            reference_nifti_path=s_info["image_path"],
                        )

        # 2. Absent-organ negative prompts on this brain scan
        for pr in exp1_prompts:
            if pr.variant_type != "absent_organ_negative":
                continue
            key = (pr.target, pr.side, pr.variant_type, pr.variant_subtype)
            m_neg = mask_by_key[key]
            g_neg = mask_geometry(m_neg)
            hallucinated = bool(g_neg["n_voxels"] > 0)
            row = {
                "subject_id": sid,
                "target": pr.target,
                "template_style": pr.variant_subtype,
                "condition": "absent_organ_negative",
                "side": pr.side,
                "prompt_id": pr.prompt_id,
                "prompt_text": pr.prompt_text,
                "volume_shape_ras": shape_s,
                "n_pred_voxels": g_neg["n_voxels"],
                "n_gt_left_voxels": 0,
                "n_gt_right_voxels": 0,
                "vol_diff_left_minus_right": 0,
                "vol_diff_base_minus_left": 0,
                "vol_diff_base_minus_right": 0,
                "dice_base_vs_left": np.nan,
                "dice_base_vs_right": np.nan,
                "dice_left_vs_right": np.nan,
                "iou_base_vs_left": np.nan,
                "iou_base_vs_right": np.nan,
                "iou_left_vs_right": np.nan,
                "centroid_shift_left_vs_right_vox": np.nan,
                "centroid_shift_left_vs_right_norm": np.nan,
                "centroid_shift_base_vs_left_vox": np.nan,
                "centroid_shift_base_vs_right_vox": np.nan,
                "dice_vs_gt_left": np.nan,
                "dice_vs_gt_right": np.nan,
                "dice_vs_gt_bilateral": np.nan,
                "dice_vs_matched_gt": 1.0 if not hallucinated else 0.0,
                "dice_vs_contradictory_opposite_gt": np.nan,
                "iou_vs_matched_gt": 1.0 if not hallucinated else 0.0,
                "iou_vs_contradictory_opposite_gt": np.nan,
                "centroid_error_vs_matched_gt_vox": np.nan,
                "centroid_error_vs_matched_gt_norm": np.nan,
                "pred_centroid_0": g_neg["centroid_0"],
                "pred_centroid_1": g_neg["centroid_1"],
                "pred_centroid_2": g_neg["centroid_2"],
                "pred_side_midline": g_neg["side_rl"],
                "frac_voxels_right_of_midline": g_neg["frac_right_of_midline"],
                "frac_voxels_left_of_midline": g_neg["frac_left_of_midline"],
                "wrong_side_error": hallucinated,
                "failure_nearly_identical_flag": False,
                "left_right_disjoint_flag": not hallucinated,
            }
            if save_rle_in_csv:
                row["pred_mask_rle_b64"] = encode_mask_rle(m_neg)
            subj_rows.append(row)

        mgr.save_npz_bundle(f"{sid}_exp1_masks_packed.npz", shape=np.array(orig_shape), **npz_dict)
        mgr.save_subject_checkpoint(sid, subj_rows)
        all_rows.extend(subj_rows)
        print(f"[Exp1][{s_idx + 1}/{len(subjects)}] {sid} done in {time.time() - t0:.1f}s ({len(subj_rows)} rows)")

    summary_rows = _summarize_experiment_1(all_rows)
    sections = [
        (
            "Experiment I Overview",
            "Tests whether location words (`left`, `right`, vs. `base` no-side) and contradictory/absent-organ prompts "
            "change VoxTell's 3D segmentation masks on Neurite-OASIS.",
        ),
    ]
    paths = mgr.write_all_formats(
        metrics_rows=all_rows,
        summary_rows=summary_rows,
        report_title="Axis II — Experiment I: Does the Side Word Change the Output?",
        report_sections=sections,
        metadata={"n_subjects": len(subjects), "structures": structures},
    )
    return {"paths": paths, "summary": summary_rows, "n_rows": len(all_rows)}


def _summarize_experiment_1(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_group: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["target"], r["template_style"], r["condition"])
        by_group.setdefault(key, []).append(r)

    summary: List[Dict[str, Any]] = []
    for (target, tmpl, cond), grp in sorted(by_group.items()):
        def _mean(col):
            vals = [float(x[col]) for x in grp if x.get(col) is not None and not (isinstance(x[col], float) and math.isnan(x[col]))]
            return round(float(np.mean(vals)), 4) if vals else np.nan

        summary.append({
            "target": target,
            "template_style": tmpl,
            "condition": cond,
            "n_subjects": len(grp),
            "mean_pred_voxels": _mean("n_pred_voxels"),
            "mean_dice_vs_gt_left": _mean("dice_vs_gt_left"),
            "mean_dice_vs_gt_right": _mean("dice_vs_gt_right"),
            "mean_dice_vs_gt_bilateral": _mean("dice_vs_gt_bilateral"),
            "mean_dice_vs_matched_gt": _mean("dice_vs_matched_gt"),
            "mean_dice_vs_contradictory_opposite_gt": _mean("dice_vs_contradictory_opposite_gt"),
            "mean_dice_base_vs_left": _mean("dice_base_vs_left"),
            "mean_dice_base_vs_right": _mean("dice_base_vs_right"),
            "mean_dice_left_vs_right": _mean("dice_left_vs_right"),
            "mean_centroid_shift_lr_vox": _mean("centroid_shift_left_vs_right_vox"),
            "wrong_side_rate": _mean("wrong_side_error"),
            "failure_nearly_identical_rate": _mean("failure_nearly_identical_flag"),
            "left_right_disjoint_rate": _mean("left_right_disjoint_flag"),
        })
    return summary


# ===========================================================================
# EXPERIMENT II & III: Synonym Shift Analysis + Layer-wise Localization
# ===========================================================================
def run_experiment_2_and_3_synonyms_and_layers(
    predictor,
    neurite_oasis_dir: Union[str, Path],
    results_root: Union[str, Path],
    structure_scope: str = "core_7",
    max_subjects: Optional[int] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    save_nifti_subjects: int = 2,
    save_rle_in_csv: bool = True,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Runs Experiment II (Synonym Shift Analysis) AND Experiment III (Layer Localization)
    jointly in a SINGLE forward pass per subject so the encoder and transformer states
    are computed once and shared across both deliverables.

    Experiment II tracks:
      1. Embedding change (cosine distance, L2 distance, relative L2, Pearson r) between
         each synonym/paraphrase/location prompt and the reference prompt, measured at:
           - Frozen Qwen3-Embedding-4B output (`qwen3_embed`)
           - Prompt Decoder output (`mask_embedding` after the 6 cross-attention layers)
      2. Prediction change: Dice, IoU, volume shift, and centroid shift between the mask
         of each synonym and the mask of the reference prompt.
      3. Link between the two: Spearman ρ and Pearson r correlating embedding distance
         with prediction change (`1 - Dice(synonym, ref)`), plus rank-order comparison.
      4. Effect on performance: Dice & IoU against ground truth for each prompt variant.
      5. Similarity map comparison per synonym.

    Experiment III tracks:
      - Stage-by-stage distance (cosine distance, relative L2 distance, Pearson correlation,
        and cross-attention spatial map cosine/JS distance) across all 14 stages:
          0: `00_qwen3_frozen`
          1: `01_project_text_embed`
          2..7: `02_transformer_layer_1` .. `07_transformer_layer_6`
          8: `08_transformer_norm_mask_emb`
          9..13: `09_unet_fusion_stage_0_12x12x12` .. `13_unet_fusion_stage_4_192x192x192`
      - Evaluated for:
          (a) `left_vs_right` (canonical left vs. canonical right prompt for each structure)
          (b) `base_vs_left` and `base_vs_right`
          (c) `synonym_vs_reference` (every synonym/paraphrase/location variant vs. its canonical reference)
    """
    mgr_exp2 = ExperimentOutputManager(results_root, "exp2_synonym_shift")
    mgr_exp3 = ExperimentOutputManager(results_root, "exp3_layer_localization")

    structures = resolve_structures(structure_scope)
    catalog = [
        pr for pr in build_oasis_prompt_catalog(structures=structures, include_absent_negatives=False)
    ]
    subjects = discover_oasis_subjects(
        neurite_oasis_dir, max_subjects=max_subjects, start_index=start_index, end_index=end_index
    )

    exp2_all_rows: List[Dict[str, Any]] = []
    exp3_all_rows: List[Dict[str, Any]] = []

    for s_idx, s_info in enumerate(subjects):
        sid = s_info["subject_id"]
        if resume and mgr_exp2.has_subject_checkpoint(sid) and mgr_exp3.has_subject_checkpoint(sid):
            c2 = mgr_exp2.load_subject_checkpoint(sid)
            c3 = mgr_exp3.load_subject_checkpoint(sid)
            exp2_all_rows.extend(c2["rows"])
            exp3_all_rows.extend(c3["rows"])
            print(f"[Exp2+3][resume {s_idx + 1}/{len(subjects)}] {sid}")
            continue

        t0 = time.time()
        bundle = evaluate_subject_prompts_and_internals(
            predictor, s_info, catalog, capture_internals=True, return_probs=False
        )
        lbl_arr = bundle["lbl_arr"]
        masks = bundle["masks"]
        orig_shape = bundle["orig_shape"]
        shape_s = shape_to_str(orig_shape)
        internals = bundle["internals"]

        # Map prompt_text -> index in catalog
        idx_by_prompt: Dict[str, int] = {pr.prompt_text: i for i, pr in enumerate(catalog)}

        subj_exp2_rows: List[Dict[str, Any]] = []
        subj_exp3_rows: List[Dict[str, Any]] = []
        npz_exp2: Dict[str, np.ndarray] = {}

        # Extract internal stage tensors (each of shape (N_prompts, dim))
        stage_defs: List[Tuple[int, str, torch.Tensor]] = [
            (0, "00_qwen3_frozen", internals["qwen3_embed"]),
            (1, "01_project_text_embed", internals["tgt0"]),
        ]
        for l_i, ls in enumerate(internals["layer_states"]):
            stage_defs.append((2 + l_i, f"{2 + l_i:02d}_transformer_layer_{l_i + 1}", ls))
        stage_defs.append((8, "08_transformer_norm_mask_emb", internals["mask_embedding"]))
        for f_i, fs in enumerate(internals["fusion_summaries"]):
            res = 12 * (2 ** f_i)
            stage_defs.append((9 + f_i, f"{9 + f_i:02d}_unet_fusion_stage_{f_i}_{res}cubed", fs))

        # --- Process Experiment II (Synonym Shift) per target & side ---
        for target in structures:
            left_id, right_id = OASIS_BILATERAL_PAIRS[target]
            gt_left = (lbl_arr == left_id)
            gt_right = (lbl_arr == right_id)

            for side, gt_matched, gt_opp in [("left", gt_left, gt_right), ("right", gt_right, gt_left)]:
                ref_pr = next(
                    pr for pr in catalog
                    if pr.target == target and pr.side == side
                    and pr.variant_type == "lateralized" and pr.variant_subtype == "canonical_sentence"
                )
                ref_i = idx_by_prompt[ref_pr.prompt_text]
                m_ref = masks[ref_i]
                g_ref = mask_geometry(m_ref)
                d_ref_gt = dice_score(m_ref, gt_matched)

                # Base prompt for comparison
                base_pr = next(
                    pr for pr in catalog
                    if pr.target == target and pr.side == "none"
                    and pr.variant_type == "base" and pr.variant_subtype == "canonical_sentence"
                )
                base_i = idx_by_prompt[base_pr.prompt_text]
                m_base_target = masks[base_i]

                # Variants for this (target, side): short_term, 5 clinical synonyms, 3 paraphrases, 1 location_based, 1 contradictory
                variants = [
                    pr for pr in catalog
                    if pr.target == target and pr.side == side
                    and not (pr.variant_type == "lateralized" and pr.variant_subtype == "canonical_sentence")
                ]

                subject_side_rows: List[Dict[str, Any]] = []
                for var_pr in variants:
                    v_i = idx_by_prompt[var_pr.prompt_text]
                    m_var = masks[v_i]
                    g_var = mask_geometry(m_var)

                    # 1. Embedding distance at frozen Qwen3 output vs. reference and vs. base
                    qwen_vs_ref = vector_distances(internals["qwen3_embed"][v_i], internals["qwen3_embed"][ref_i])
                    qwen_vs_base = vector_distances(internals["qwen3_embed"][v_i], internals["qwen3_embed"][base_i])

                    # 2. Embedding distance after Prompt Decoder (`mask_embedding`) vs. reference and vs. base
                    pdec_vs_ref = vector_distances(internals["mask_embedding"][v_i], internals["mask_embedding"][ref_i])
                    pdec_vs_base = vector_distances(internals["mask_embedding"][v_i], internals["mask_embedding"][base_i])

                    # 3. Similarity map comparison (vs. reference AND vs. base)
                    if internals["fusion_summaries"]:
                        sim_map_metrics = vector_distances(
                            internals["fusion_summaries"][-1][v_i],
                            internals["fusion_summaries"][-1][ref_i],
                        )
                        sim_map_corr = sim_map_metrics["pearson_correlation"]
                        sim_map_cos_dist = sim_map_metrics["cosine_distance"]
                        sim_map_base = vector_distances(
                            internals["fusion_summaries"][-1][v_i],
                            internals["fusion_summaries"][-1][base_i],
                        )
                        sim_map_corr_base = sim_map_base["pearson_correlation"]
                    else:
                        sim_map_corr = np.nan
                        sim_map_cos_dist = np.nan
                        sim_map_corr_base = np.nan

                    # 4. Prediction change vs. reference mask AND vs. base mask
                    dice_vs_ref = dice_score(m_var, m_ref)
                    pred_change_1_minus_dice = 1.0 - dice_vs_ref
                    iou_vs_ref = iou_score(m_var, m_ref)
                    dice_vs_base = dice_score(m_var, m_base_target)
                    pred_change_vs_base = 1.0 - dice_vs_base
                    vol_shift = g_var["n_voxels"] - g_ref["n_voxels"]
                    rel_vol_shift = abs(vol_shift) / max(g_ref["n_voxels"], 1)
                    c_shift_vox, c_shift_norm = centroid_distance(g_var, g_ref, orig_shape)

                    # 5. Performance vs. Ground Truth
                    d_gt = dice_score(m_var, gt_matched)
                    d_opp = dice_score(m_var, gt_opp)
                    i_gt = iou_score(m_var, gt_matched)
                    dice_drop_vs_ref = d_ref_gt - d_gt

                    row2 = {
                        "subject_id": sid,
                        "target": target,
                        "side": side,
                        "prompt_id": var_pr.prompt_id,
                        "variant_type": var_pr.variant_type,
                        "variant_subtype": var_pr.variant_subtype,
                        "prompt_text": var_pr.prompt_text,
                        "reference_prompt_text": ref_pr.prompt_text,
                        "volume_shape_ras": shape_s,
                        # Embedding change at Frozen Qwen3
                        "qwen3_cosine_dist_vs_ref": qwen_vs_ref["cosine_distance"],
                        "qwen3_l2_dist_vs_ref": qwen_vs_ref["l2_distance"],
                        "qwen3_rel_l2_dist_vs_ref": qwen_vs_ref["relative_l2_distance"],
                        "qwen3_pearson_r_vs_ref": qwen_vs_ref["pearson_correlation"],
                        "qwen3_cosine_dist_vs_base": qwen_vs_base["cosine_distance"],
                        "qwen3_l2_dist_vs_base": qwen_vs_base["l2_distance"],
                        "qwen3_pearson_r_vs_base": qwen_vs_base["pearson_correlation"],
                        # Embedding change after Prompt Decoder
                        "prompt_decoder_cosine_dist_vs_ref": pdec_vs_ref["cosine_distance"],
                        "prompt_decoder_l2_dist_vs_ref": pdec_vs_ref["l2_distance"],
                        "prompt_decoder_rel_l2_dist_vs_ref": pdec_vs_ref["relative_l2_distance"],
                        "prompt_decoder_pearson_r_vs_ref": pdec_vs_ref["pearson_correlation"],
                        "prompt_decoder_cosine_dist_vs_base": pdec_vs_base["cosine_distance"],
                        "prompt_decoder_l2_dist_vs_base": pdec_vs_base["l2_distance"],
                        "prompt_decoder_pearson_r_vs_base": pdec_vs_base["pearson_correlation"],
                        # Similarity map agreement
                        "similarity_map_pearson_r_vs_ref": sim_map_corr,
                        "similarity_map_cosine_dist_vs_ref": sim_map_cos_dist,
                        "similarity_map_pearson_r_vs_base": sim_map_corr_base,
                        # Prediction change vs Reference prompt & Base prompt
                        "dice_vs_ref_prompt": round(dice_vs_ref, 4),
                        "pred_change_1_minus_dice_ref": round(pred_change_1_minus_dice, 4),
                        "iou_vs_ref_prompt": round(iou_vs_ref, 4),
                        "dice_vs_base_prompt": round(dice_vs_base, 4),
                        "pred_change_1_minus_dice_base": round(pred_change_vs_base, 4),
                        "n_pred_voxels": g_var["n_voxels"],
                        "n_ref_voxels": g_ref["n_voxels"],
                        "volume_shift_voxels": vol_shift,
                        "rel_volume_shift": round(rel_vol_shift, 4),
                        "centroid_shift_vs_ref_vox": c_shift_vox,
                        "centroid_shift_vs_ref_norm": c_shift_norm,
                        # Ground-truth performance
                        "dice_vs_gt": round(d_gt, 4),
                        "ref_dice_vs_gt": round(d_ref_gt, 4),
                        "dice_drop_vs_ref_gt": round(dice_drop_vs_ref, 4),
                        "dice_vs_opposite_gt": round(d_opp, 4),
                        "iou_vs_gt": round(i_gt, 4),
                        "wrong_side_error": bool(d_opp > d_gt and d_opp > 0.1),
                        # Rank columns populated below across synonyms for this (subject, target, side)
                        "rank_by_qwen3_dist": 0,
                        "rank_by_prompt_decoder_dist": 0,
                        "rank_by_pred_change": 0,
                    }
                    if save_rle_in_csv:
                        row2["pred_mask_rle_b64"] = encode_mask_rle(m_var)
                    subject_side_rows.append(row2)
                    npz_exp2[f"{var_pr.prompt_id}_mask"] = np.packbits(m_var.ravel())
                    if internals["fusion_summaries"]:
                        npz_exp2[f"{var_pr.prompt_id}_sim_map_6cubed"] = (
                            internals["fusion_summaries"][-1][v_i].numpy().reshape(6, 6, 6).astype(np.float16)
                        )

                    if s_idx < save_nifti_subjects and var_pr.variant_subtype in ("clinical_synonym_1", "paraphrase_1", "anatomical_location_wording"):
                        mgr_exp2.save_nifti(
                            f"{sid}_{target.replace(' ', '_')}_{side}_{var_pr.variant_subtype}.nii.gz",
                            m_var,
                            reference_nifti_path=s_info["image_path"],
                        )

                    # Experiment III: Record stage-by-stage trajectory for Synonym vs Reference
                    for st_idx, st_name, st_tensor in stage_defs:
                        vd = vector_distances(st_tensor[v_i], st_tensor[ref_i])
                        attn_cos_dist = np.nan
                        if 2 <= st_idx <= 7:
                            att_layer = internals["attns"][st_idx - 2]
                            attn_vd = vector_distances(att_layer[v_i], att_layer[ref_i])
                            attn_cos_dist = attn_vd["cosine_distance"]
                        subj_exp3_rows.append({
                            "subject_id": sid,
                            "target": target,
                            "comparison_type": f"synonym_vs_ref ({var_pr.variant_type})",
                            "pair_label": f"{side}:{var_pr.variant_subtype}_vs_canonical",
                            "stage_index": st_idx,
                            "stage_name": st_name,
                            "cosine_distance": vd["cosine_distance"],
                            "l2_distance": vd["l2_distance"],
                            "relative_l2_distance": vd["relative_l2_distance"],
                            "pearson_correlation": vd["pearson_correlation"],
                            "cross_attn_cosine_distance": attn_cos_dist,
                        })

                # Populate synonym distance ranks vs. output prediction-change ranks (Section 3 "Other angles to try")
                q_order = np.argsort([r["qwen3_cosine_dist_vs_ref"] for r in subject_side_rows])
                p_order = np.argsort([r["prompt_decoder_cosine_dist_vs_ref"] for r in subject_side_rows])
                o_order = np.argsort([r["pred_change_1_minus_dice_ref"] for r in subject_side_rows])
                for rk, idx_row in enumerate(q_order, 1):
                    subject_side_rows[idx_row]["rank_by_qwen3_dist"] = int(rk)
                for rk, idx_row in enumerate(p_order, 1):
                    subject_side_rows[idx_row]["rank_by_prompt_decoder_dist"] = int(rk)
                for rk, idx_row in enumerate(o_order, 1):
                    subject_side_rows[idx_row]["rank_by_pred_change"] = int(rk)

                subj_exp2_rows.extend(subject_side_rows)

            # Experiment III: Record Left vs Right and Base vs Left/Right across all 14 stages
            left_pr = next(
                pr for pr in catalog
                if pr.target == target and pr.side == "left"
                and pr.variant_type == "lateralized" and pr.variant_subtype == "canonical_sentence"
            )
            right_pr = next(
                pr for pr in catalog
                if pr.target == target and pr.side == "right"
                and pr.variant_type == "lateralized" and pr.variant_subtype == "canonical_sentence"
            )
            base_pr = next(
                pr for pr in catalog
                if pr.target == target and pr.side == "none"
                and pr.variant_type == "base" and pr.variant_subtype == "canonical_sentence"
            )
            li, ri, bi = idx_by_prompt[left_pr.prompt_text], idx_by_prompt[right_pr.prompt_text], idx_by_prompt[base_pr.prompt_text]

            for comp_type, p_label, idx_a, idx_b in [
                ("left_vs_right", f"{target}:left_vs_right", li, ri),
                ("base_vs_left", f"{target}:base_vs_left", bi, li),
                ("base_vs_right", f"{target}:base_vs_right", bi, ri),
            ]:
                for st_idx, st_name, st_tensor in stage_defs:
                    vd = vector_distances(st_tensor[idx_a], st_tensor[idx_b])
                    # Also measure cross-attention map distance if this stage is one of the 6 transformer layers
                    attn_cos_dist = np.nan
                    if 2 <= st_idx <= 7:
                        att_layer = internals["attns"][st_idx - 2]
                        attn_vd = vector_distances(att_layer[idx_a], att_layer[idx_b])
                        attn_cos_dist = attn_vd["cosine_distance"]

                    subj_exp3_rows.append({
                        "subject_id": sid,
                        "target": target,
                        "comparison_type": comp_type,
                        "pair_label": p_label,
                        "stage_index": st_idx,
                        "stage_name": st_name,
                        "cosine_distance": vd["cosine_distance"],
                        "l2_distance": vd["l2_distance"],
                        "relative_l2_distance": vd["relative_l2_distance"],
                        "pearson_correlation": vd["pearson_correlation"],
                        "cross_attn_cosine_distance": attn_cos_dist,
                    })

        mgr_exp2.save_npz_bundle(f"{sid}_exp2_synonym_masks_packed.npz", shape=np.array(orig_shape), **npz_exp2)
        mgr_exp2.save_subject_checkpoint(sid, subj_exp2_rows)
        mgr_exp3.save_subject_checkpoint(sid, subj_exp3_rows)

        exp2_all_rows.extend(subj_exp2_rows)
        exp3_all_rows.extend(subj_exp3_rows)
        print(
            f"[Exp2+3][{s_idx + 1}/{len(subjects)}] {sid} done in {time.time() - t0:.1f}s "
            f"(Exp2: {len(subj_exp2_rows)} rows, Exp3: {len(subj_exp3_rows)} rows)"
        )

    # Summarize & write Experiment II
    exp2_summary, exp2_corr_rows = _summarize_experiment_2(exp2_all_rows)
    # Save correlation table as an additional CSV in exp2 folder
    corr_csv = mgr_exp2.exp_dir / "embedding_vs_prediction_correlations.csv"
    if exp2_corr_rows:
        import csv as _csv
        with corr_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(exp2_corr_rows[0].keys()))
            w.writeheader()
            w.writerows(exp2_corr_rows)

    exp2_paths = mgr_exp2.write_all_formats(
        metrics_rows=exp2_all_rows,
        summary_rows=exp2_summary,
        report_title="Axis II — Experiment II: Synonym & Paraphrase Shift Analysis",
        report_sections=[
            (
                "Experiment II Overview",
                "Tracks (1) embedding change at the frozen Qwen3 output and after the Prompt Decoder, "
                "(2) prediction change (Dice/IoU/volume/centroid shift vs. canonical reference), "
                "(3) correlation between embedding distance and prediction shift, and "
                "(4) Dice performance against ground truth across synonyms, paraphrases, and location wordings.",
            ),
            (
                "Embedding Distance vs. Prediction Shift Correlations",
                rows_to_markdown_table(exp2_corr_rows),
            ),
        ],
        metadata={"n_subjects": len(subjects), "structures": structures, "correlations": exp2_corr_rows},
    )

    # Summarize & write Experiment III + generate Layer Localization Plot (Deliverable 4)
    exp3_summary = _summarize_experiment_3(exp3_all_rows)
    plot_path = _plot_layer_localization(exp3_summary, mgr_exp3.exp_dir / "layer_localization_plot.png")
    exp3_paths = mgr_exp3.write_all_formats(
        metrics_rows=exp3_all_rows,
        summary_rows=exp3_summary,
        report_title="Axis II — Experiment III: Layer-wise Localization of Side & Synonym Information",
        report_sections=[
            (
                "Experiment III Overview",
                "Measures feature distance (cosine distance, relative L2, Pearson r, and cross-attention divergence) "
                "between Left vs. Right prompts and Synonym vs. Reference prompts across all 14 stages of VoxTell: "
                "Frozen Qwen3 -> `project_text_embed` -> Transformer Decoder Layers 1..6 -> `LayerNorm` -> "
                "U-Net Multi-Scale Fusion Stages 0..4.\n\n"
                f"- **Saved Layer-wise Change Plot:** `{plot_path.name}`",
            ),
        ],
        metadata={"n_subjects": len(subjects), "structures": structures, "plot_path": str(plot_path)},
    )
    exp3_paths["layer_plot_png"] = plot_path

    return {
        "exp2": {"paths": exp2_paths, "summary": exp2_summary, "correlations": exp2_corr_rows},
        "exp3": {"paths": exp3_paths, "summary": exp3_summary},
    }


def _summarize_experiment_2(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_var: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    by_target: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_var.setdefault((r["target"], r["side"], r["variant_type"], r["variant_subtype"]), []).append(r)
        if r["variant_type"] in ("synonym", "paraphrase", "location_based", "lateralized"):
            by_target.setdefault(r["target"], []).append(r)

    summary: List[Dict[str, Any]] = []
    for (target, side, vtype, vsub), grp in sorted(by_var.items()):
        def _m(col):
            vals = [float(x[col]) for x in grp if x.get(col) is not None and not (isinstance(x[col], float) and math.isnan(x[col]))]
            return round(float(np.mean(vals)), 4) if vals else np.nan

        summary.append({
            "target": target,
            "side": side,
            "variant_type": vtype,
            "variant_subtype": vsub,
            "prompt_text": grp[0]["prompt_text"],
            "n_subjects": len(grp),
            "qwen3_cosine_dist_vs_ref": _m("qwen3_cosine_dist_vs_ref"),
            "prompt_decoder_cosine_dist_vs_ref": _m("prompt_decoder_cosine_dist_vs_ref"),
            "similarity_map_pearson_r_vs_ref": _m("similarity_map_pearson_r_vs_ref"),
            "mean_dice_vs_ref_prompt": _m("dice_vs_ref_prompt"),
            "mean_pred_change_1_minus_dice": _m("pred_change_1_minus_dice_ref"),
            "mean_rel_volume_shift": _m("rel_volume_shift"),
            "mean_dice_vs_gt": _m("dice_vs_gt"),
            "mean_dice_drop_vs_ref_gt": _m("dice_drop_vs_ref_gt"),
            "wrong_side_rate": _m("wrong_side_error"),
            "mean_rank_by_qwen3_dist": _m("rank_by_qwen3_dist"),
            "mean_rank_by_prompt_decoder_dist": _m("rank_by_prompt_decoder_dist"),
            "mean_rank_by_pred_change": _m("rank_by_pred_change"),
        })

    # Compute correlation between embedding distance and prediction change (1 - Dice_vs_ref)
    corr_rows: List[Dict[str, Any]] = []
    for target, grp in sorted(by_target.items()):
        q_dist = np.array([float(x["qwen3_cosine_dist_vs_ref"]) for x in grp])
        p_dist = np.array([float(x["prompt_decoder_cosine_dist_vs_ref"]) for x in grp])
        pred_shift = np.array([float(x["pred_change_1_minus_dice_ref"]) for x in grp])
        vol_shift = np.array([float(x["rel_volume_shift"]) for x in grp])

        def _safe_corr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float, float]:
            if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
                return (np.nan, np.nan, np.nan, np.nan)
            pr, pp = stats.pearsonr(a, b)
            sr, sp = stats.spearmanr(a, b)
            return (round(float(pr), 4), round(float(pp), 4), round(float(sr), 4), round(float(sp), 4))

        q_pr, q_pp, q_sr, q_sp = _safe_corr(q_dist, pred_shift)
        p_pr, p_pp, p_sr, p_sp = _safe_corr(p_dist, pred_shift)
        qv_pr, _, qv_sr, _ = _safe_corr(q_dist, vol_shift)
        pv_pr, _, pv_sr, _ = _safe_corr(p_dist, vol_shift)

        corr_rows.append({
            "target": target,
            "n_observations": len(grp),
            "qwen3_vs_pred_shift_pearson_r": q_pr,
            "qwen3_vs_pred_shift_pearson_p": q_pp,
            "qwen3_vs_pred_shift_spearman_rho": q_sr,
            "qwen3_vs_pred_shift_spearman_p": q_sp,
            "prompt_dec_vs_pred_shift_pearson_r": p_pr,
            "prompt_dec_vs_pred_shift_pearson_p": p_pp,
            "prompt_dec_vs_pred_shift_spearman_rho": p_sr,
            "prompt_dec_vs_pred_shift_spearman_p": p_sp,
            "qwen3_vs_vol_shift_spearman_rho": qv_sr,
            "prompt_dec_vs_vol_shift_spearman_rho": pv_sr,
        })

    return summary, corr_rows


def _summarize_experiment_3(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_stage: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["comparison_type"], int(r["stage_index"]), r["stage_name"])
        by_stage.setdefault(key, []).append(r)

    summary: List[Dict[str, Any]] = []
    for (comp_type, st_idx, st_name), grp in sorted(by_stage.items()):
        def _m(col):
            vals = [float(x[col]) for x in grp if x.get(col) is not None and not (isinstance(x[col], float) and math.isnan(x[col]))]
            return round(float(np.mean(vals)), 6) if vals else np.nan

        summary.append({
            "comparison_type": comp_type,
            "stage_index": st_idx,
            "stage_name": st_name,
            "n_pairs": len(grp),
            "mean_cosine_distance": _m("cosine_distance"),
            "mean_relative_l2_distance": _m("relative_l2_distance"),
            "mean_pearson_correlation": _m("pearson_correlation"),
            "mean_cross_attn_cosine_distance": _m("cross_attn_cosine_distance"),
        })
    return summary


def _plot_layer_localization(summary_rows: List[Dict[str, Any]], out_png: Path) -> Path:
    """
    Generate Deliverable 4 ('Layer-wise change plot') plotting Cosine Distance and
    Relative L2 Distance against network stage depth (Stages 0..13) for Left vs. Right,
    Base vs. Left/Right, and Synonym vs. Reference.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        by_comp: Dict[str, List[Dict[str, Any]]] = {}
        for r in summary_rows:
            by_comp.setdefault(r["comparison_type"], []).append(r)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
        for comp_type, rows in sorted(by_comp.items()):
            rows_sorted = sorted(rows, key=lambda x: int(x["stage_index"]))
            xs = [int(x["stage_index"]) for x in rows_sorted]
            cos_y = [float(x["mean_cosine_distance"]) for x in rows_sorted]
            l2_y = [float(x["mean_relative_l2_distance"]) for x in rows_sorted]
            axes[0].plot(xs, cos_y, marker="o", linewidth=2, label=comp_type)
            axes[1].plot(xs, l2_y, marker="s", linewidth=2, label=comp_type)

        stage_labels = [
            "0:Qwen3", "1:Proj", "2:DecL1", "3:DecL2", "4:DecL3",
            "5:DecL4", "6:DecL5", "7:DecL6", "8:Norm",
            "9:Fus0", "10:Fus1", "11:Fus2", "12:Fus3", "13:Fus4",
        ]
        for ax, title, ylabel in [
            (axes[0], "Layer-wise Cosine Distance vs. Stage Depth", "Mean Cosine Distance (1 - cos)"),
            (axes[1], "Layer-wise Relative L2 Distance vs. Stage Depth", "Mean Relative L2 Distance"),
        ]:
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("VoxTell Network Stage")
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(len(stage_labels)))
            ax.set_xticklabels(stage_labels, rotation=45, ha="right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(str(out_png), bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not render layer_localization_plot.png: {exc}")
    return out_png


# ===========================================================================
# EXPERIMENT IV: Uncertainty Estimation via MC Dropout
# ===========================================================================
def compute_mc_uncertainty_metrics(
    pass_probs: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    n_ece_bins: int = 10,
) -> Dict[str, Any]:
    """
    Given `pass_probs` of shape `(K, X, Y, Z)` (probabilities in `[0, 1]` from `K`
    stochastic MC Dropout forward passes for one prompt) and optional `gt_mask` `(X, Y, Z)`:

    Computes the exact quantities specified in Section 5 of `Axis_II_Prompt_Sensitivity_Nipu.docx`:
      - `mean_prob_map`: `p_bar = (1/K) sum_k p_k`
      - `var_prob_map` (Epistemic Variance): `(1/K) sum_k (p_k - p_bar)^2`
      - `predictive_entropy_map` (Total Uncertainty): `H(p_bar) = -p_bar log(p_bar) - (1-p_bar) log(1-p_bar)`
      - `aleatoric_entropy_map` (Aleatoric Uncertainty): `(1/K) sum_k H(p_k)`
      - `mutual_information_map` (Epistemic MI): `H(p_bar) - (1/K) sum_k H(p_k)`
      - Voxel-wise and mask-region (union of predicted foregrounds across passes or top-prob region) averages
      - Pass-to-pass mask stability: mean pairwise Dice across the `K` passes, volume std/CV
      - Expected Calibration Error (ECE) if `gt_mask` is provided.
    """
    eps = 1e-7
    K = pass_probs.shape[0]
    p = np.clip(pass_probs.astype(np.float32), eps, 1.0 - eps)

    p_bar = p.mean(axis=0)                         # (X, Y, Z)
    var_map = p.var(axis=0)                        # Epistemic variance

    # Binary entropy in nats (normalized by ln(2) so max entropy = 1.0 bit)
    inv_ln2 = 1.0 / math.log(2.0)
    pred_ent_map = -(p_bar * np.log(p_bar) + (1.0 - p_bar) * np.log(1.0 - p_bar)) * inv_ln2
    pass_ent = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) * inv_ln2
    alea_ent_map = pass_ent.mean(axis=0)
    mi_map = np.maximum(0.0, pred_ent_map - alea_ent_map)

    mean_mask = p_bar > 0.5
    pass_masks = p > 0.5                           # (K, X, Y, Z)
    any_fg_mask = pass_masks.any(axis=0)

    # Region of interest for localized uncertainty: union of predicted foreground across passes
    # (or if completely empty, voxels with p_bar > 0.05; or whole volume)
    if int(any_fg_mask.sum()) > 0:
        roi = any_fg_mask
    elif int((p_bar > 0.05).sum()) > 0:
        roi = p_bar > 0.05
    else:
        roi = np.ones_like(mean_mask, dtype=bool)

    # Pass-to-pass volume and Dice consistency
    pass_vols = pass_masks.reshape(K, -1).sum(axis=1).astype(np.float64)
    vol_mean = float(pass_vols.mean())
    vol_std = float(pass_vols.std())
    vol_cv = float(vol_std / max(vol_mean, 1.0))

    pairwise_dices = []
    for i in range(K):
        for j in range(i + 1, K):
            pairwise_dices.append(dice_score(pass_masks[i], pass_masks[j]))
    inter_pass_dice = float(np.mean(pairwise_dices)) if pairwise_dices else 1.0

    # Expected Calibration Error (ECE) on ROI + GT union
    ece = np.nan
    dice_mean_mask_gt = np.nan
    if gt_mask is not None:
        gt_b = np.asarray(gt_mask, dtype=bool)
        dice_mean_mask_gt = dice_score(mean_mask, gt_b)
        eval_region = np.logical_or(roi, gt_b)
        if int(eval_region.sum()) > 0:
            probs_eval = p_bar[eval_region]
            labels_eval = gt_b[eval_region].astype(np.float32)
            # Confidence of predicted class (foreground if p >= 0.5 else background)
            conf = np.where(probs_eval >= 0.5, probs_eval, 1.0 - probs_eval)
            preds = (probs_eval >= 0.5).astype(np.float32)
            acc = (preds == labels_eval).astype(np.float32)
            bins = np.linspace(0.5, 1.0, n_ece_bins + 1)
            ece_val = 0.0
            n_tot = float(probs_eval.size)
            for b_i in range(n_ece_bins):
                in_bin = (conf > bins[b_i]) & (conf <= bins[b_i + 1])
                if np.any(in_bin):
                    ece_val += (float(in_bin.sum()) / n_tot) * abs(float(acc[in_bin].mean()) - float(conf[in_bin].mean()))
            ece = round(float(ece_val), 6)

    return {
        "n_passes": K,
        "mean_mask_voxels": int(mean_mask.sum()),
        "union_fg_voxels": int(any_fg_mask.sum()),
        "pass_volume_mean": round(vol_mean, 2),
        "pass_volume_std": round(vol_std, 2),
        "pass_volume_cv": round(vol_cv, 4),
        "inter_pass_mean_dice": round(inter_pass_dice, 4),
        "dice_mean_mask_vs_gt": round(float(dice_mean_mask_gt), 4) if not math.isnan(dice_mean_mask_gt) else np.nan,
        # Whole-volume uncertainty averages
        "voxel_mean_prob": round(float(p_bar.mean()), 6),
        "voxel_epistemic_variance": round(float(var_map.mean()), 6),
        "voxel_epistemic_mi_bits": round(float(mi_map.mean()), 6),
        "voxel_aleatoric_entropy_bits": round(float(alea_ent_map.mean()), 6),
        "voxel_predictive_entropy_bits": round(float(pred_ent_map.mean()), 6),
        # ROI / Mask-level uncertainty averages
        "roi_mean_prob": round(float(p_bar[roi].mean()), 6),
        "roi_epistemic_variance": round(float(var_map[roi].mean()), 6),
        "roi_max_epistemic_variance": round(float(var_map[roi].max()), 6),
        "roi_epistemic_mi_bits": round(float(mi_map[roi].mean()), 6),
        "roi_aleatoric_entropy_bits": round(float(alea_ent_map[roi].mean()), 6),
        "roi_predictive_entropy_bits": round(float(pred_ent_map[roi].mean()), 6),
        "ece_roi": ece,
        # Maps for saving
        "_maps": {
            "mean_prob": p_bar.astype(np.float16),
            "epistemic_var": var_map.astype(np.float16),
            "epistemic_mi": mi_map.astype(np.float16),
            "aleatoric_entropy": alea_ent_map.astype(np.float16),
            "predictive_entropy": pred_ent_map.astype(np.float16),
            "mean_mask": mean_mask,
        },
    }


def run_experiment_4_mc_dropout_uncertainty(
    predictor,
    neurite_oasis_dir: Union[str, Path],
    results_root: Union[str, Path],
    structure_scope: str = "core_7",
    n_passes: int = 10,
    spatial_dropout_p: float = 0.0,
    max_subjects: Optional[int] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    save_nifti_subjects: int = 2,
    save_rle_in_csv: bool = True,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Experiment IV (Section 5 of `Axis_II_Prompt_Sensitivity_Nipu.docx`):
      - Unfreezes the dropout layers in VoxTell (`transformer_decoder` `nn.Dropout` + `MultiheadAttention`
        dropout, plus optional `spatial_dropout_p`), keeps everything else fixed, and runs
        `n_passes` stochastic forward passes per prompt.
      - Evaluates across:
          * Base prompt (no side)
          * Canonical Left & Right prompts
          * Clinical Synonyms & Paraphrases & Location descriptions
          * Contradictory prompts (e.g., `"left X located on the right hemisphere"`, AND
            evaluating `"left X"` uncertainty when only Right GT is considered vs. Matched GT)
          * Absent-Organ negative prompts (`"left kidney"`, `"adrenal tumor"`, etc.)
      - Saves voxel-wise uncertainty maps (`mean_prob`, `epistemic_var`, `aleatoric_entropy`,
        `epistemic_mi`) and full CSV/JSON/Markdown deliverables.
    """
    mgr = ExperimentOutputManager(results_root, "exp4_mc_dropout_uncertainty")
    structures = resolve_structures(structure_scope)
    all_catalog = build_oasis_prompt_catalog(structures=structures, include_absent_negatives=True)

    # Select representative set per structure so 10 passes run rapidly per subject:
    # base, canonical left/right, short left/right, 2 clinical synonyms per side,
    # 1 paraphrase per side, 1 location_based per side, contradictory per side, plus absent negatives
    wanted_subtypes = {
        "canonical_sentence",
        "short_term",
        "clinical_synonym_1",
        "clinical_synonym_2",
        "paraphrase_1",
        "anatomical_location_wording",
        "conflicting_left_on_right",
        "conflicting_right_on_left",
        "base_absent",
        "lateralized_absent",
        "lesion_left_absent",
        "lung_lobe_right_upper_absent",
    }
    mc_prompts = [pr for pr in all_catalog if pr.variant_subtype in wanted_subtypes]

    subjects = discover_oasis_subjects(
        neurite_oasis_dir, max_subjects=max_subjects, start_index=start_index, end_index=end_index
    )

    prompt_texts = [pr.prompt_text for pr in mc_prompts]
    text_embeddings = predictor.embed_text_prompts(prompt_texts)  # (1, M, 2560)

    all_rows: List[Dict[str, Any]] = []

    for s_idx, s_info in enumerate(subjects):
        sid = s_info["subject_id"]
        if resume and mgr.has_subject_checkpoint(sid):
            ckpt = mgr.load_subject_checkpoint(sid)
            all_rows.extend(ckpt["rows"])
            print(f"[Exp4-MCDropout][resume {s_idx + 1}/{len(subjects)}] {sid}")
            continue

        t0 = time.time()
        img_arr, _ = load_volume_ras(s_info["image_path"])
        lbl_arr, _ = load_seg_ras(s_info["label_path"])
        data, bbox, orig_shape = preprocess_volume(predictor, img_arr)
        shape_s = shape_to_str(orig_shape)

        # Run K stochastic forward passes in cropped space to save RAM, then compute metrics
        # Shape of cropped logits per pass: (M, Xc, Yc, Zc)
        M = len(mc_prompts)
        cropped_shape = data.shape[1:]
        pass_probs_cropped = np.zeros((n_passes, M, *cropped_shape), dtype=np.float16)

        for k in range(n_passes):
            torch.manual_seed(1000 + k)
            logits_k, _ = run_sliding_window_custom(
                predictor,
                data,
                text_embeddings,
                image_decoder_on=True,
                text_decoder_on=True,
                enable_mc_dropout=True,
                spatial_dropout_p=spatial_dropout_p,
                capture_internals=False,
            )
            with torch.no_grad():
                probs_k = torch.sigmoid(logits_k.float()).cpu().numpy().astype(np.float16)
            pass_probs_cropped[k] = probs_k

        # Crop ground-truth label map to `bbox` for exact alignment with `pass_probs_cropped`
        crop_slicer = tuple(slice(int(b[0]), int(b[1])) for b in bbox)
        lbl_cropped = lbl_arr[crop_slicer]

        subj_rows: List[Dict[str, Any]] = []
        npz_maps: Dict[str, np.ndarray] = {}

        for p_i, pr in enumerate(mc_prompts):
            p_passes = pass_probs_cropped[:, p_i]  # (K, Xc, Yc, Zc)

            if pr.expected_label_id is not None:
                gt_crop = (lbl_cropped == pr.expected_label_id)
            elif pr.gt_label_id_left is not None and pr.gt_label_id_right is not None:
                gt_crop = np.logical_or(lbl_cropped == pr.gt_label_id_left, lbl_cropped == pr.gt_label_id_right)
            else:
                gt_crop = np.zeros_like(lbl_cropped, dtype=bool)

            uq = compute_mc_uncertainty_metrics(p_passes, gt_mask=gt_crop)
            maps = uq.pop("_maps")

            # Reconstruct full-volume mean mask for RLE & optional NIfTI saving
            full_mean_mask = np.zeros(orig_shape, dtype=bool)
            full_mean_mask[crop_slicer] = maps["mean_mask"]

            row = {
                "subject_id": sid,
                "target": pr.target,
                "side": pr.side,
                "prompt_id": pr.prompt_id,
                "variant_type": pr.variant_type,
                "variant_subtype": pr.variant_subtype,
                "matches_image_side": pr.matches_image_side,
                "prompt_text": pr.prompt_text,
                "volume_shape_ras": shape_s,
                **uq,
            }
            if save_rle_in_csv:
                row["mean_mask_rle_b64"] = encode_mask_rle(full_mean_mask)
            subj_rows.append(row)

            # Save compressed ROI uncertainty maps (mean & variance & entropy) for all prompts in this subject
            npz_maps[f"{pr.prompt_id}_mean_prob_crop"] = maps["mean_prob"]
            npz_maps[f"{pr.prompt_id}_epistemic_var_crop"] = maps["epistemic_var"]
            npz_maps[f"{pr.prompt_id}_aleatoric_ent_crop"] = maps["aleatoric_entropy"]

            if s_idx < save_nifti_subjects and pr.target == "hippocampus" and pr.variant_subtype in ("canonical_sentence", "conflicting_left_on_right", "lateralized_absent"):
                full_var = np.zeros(orig_shape, dtype=np.float32)
                full_var[crop_slicer] = maps["epistemic_var"].astype(np.float32)
                mgr.save_nifti(
                    f"{sid}_{pr.prompt_id}_{pr.target}_{pr.side}_epistemic_var.nii.gz",
                    full_var,
                    reference_nifti_path=s_info["image_path"],
                )
                mgr.save_nifti(
                    f"{sid}_{pr.prompt_id}_{pr.target}_{pr.side}_mean_mask.nii.gz",
                    full_mean_mask,
                    reference_nifti_path=s_info["image_path"],
                )

        mgr.save_npz_bundle(
            f"{sid}_exp4_uncertainty_maps.npz",
            orig_shape=np.array(orig_shape),
            bbox=np.array(bbox),
            **npz_maps,
        )
        mgr.save_subject_checkpoint(sid, subj_rows)
        all_rows.extend(subj_rows)
        print(f"[Exp4-MCDropout][{s_idx + 1}/{len(subjects)}] {sid} ({n_passes} passes) done in {time.time() - t0:.1f}s")

    summary_rows, stability_rows = _summarize_experiment_4(all_rows)
    stab_csv = mgr.exp_dir / "synonym_uncertainty_stability.csv"
    if stability_rows:
        import csv as _csv
        with stab_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(stability_rows[0].keys()))
            w.writeheader()
            w.writerows(stability_rows)

    paths = mgr.write_all_formats(
        metrics_rows=all_rows,
        summary_rows=summary_rows,
        report_title="Axis II — Experiment IV: MC Dropout Uncertainty Estimation",
        report_sections=[
            (
                "Experiment IV Overview",
                f"Runs N={n_passes} stochastic forward passes per prompt with VoxTell's dropout layers unfrozen. "
                "Measures Epistemic Uncertainty (predictive variance & mutual information across passes) and "
                "Aleatoric Uncertainty (average per-pass binary entropy), plus pass-to-pass Dice stability and ECE.",
            ),
            (
                "Wording Stability & Contradictory Prompt Uncertainty",
                rows_to_markdown_table(stability_rows),
            ),
        ],
        metadata={
            "n_subjects": len(subjects),
            "n_passes": n_passes,
            "spatial_dropout_p": spatial_dropout_p,
            "structures": structures,
            "stability_analysis": stability_rows,
        },
    )
    return {"paths": paths, "summary": summary_rows, "stability": stability_rows}


def _summarize_experiment_4(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_group: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    by_target_side: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for r in rows:
        by_group.setdefault((r["target"], r["side"], r["variant_type"], r["variant_subtype"]), []).append(r)
        if r["variant_type"] in ("lateralized", "synonym", "paraphrase", "location_based"):
            by_target_side.setdefault((r["target"], r["side"]), []).append(r)

    summary: List[Dict[str, Any]] = []
    for (target, side, vtype, vsub), grp in sorted(by_group.items()):
        def _m(col):
            vals = [float(x[col]) for x in grp if x.get(col) is not None and not (isinstance(x[col], float) and math.isnan(x[col]))]
            return round(float(np.mean(vals)), 6) if vals else np.nan

        summary.append({
            "target": target,
            "side": side,
            "variant_type": vtype,
            "variant_subtype": vsub,
            "n_subjects": len(grp),
            "mean_dice_vs_gt": _m("dice_mean_mask_vs_gt"),
            "inter_pass_mean_dice": _m("inter_pass_mean_dice"),
            "pass_volume_cv": _m("pass_volume_cv"),
            "roi_epistemic_variance": _m("roi_epistemic_variance"),
            "roi_epistemic_mi_bits": _m("roi_epistemic_mi_bits"),
            "roi_aleatoric_entropy_bits": _m("roi_aleatoric_entropy_bits"),
            "roi_predictive_entropy_bits": _m("roi_predictive_entropy_bits"),
            "ece_roi": _m("ece_roi"),
        })

    # Wording stability & contradictory vs canonical comparison
    stability: List[Dict[str, Any]] = []
    for (target, side), grp in sorted(by_target_side.items()):
        epi_vars = [float(x["roi_epistemic_variance"]) for x in grp]
        alea_ents = [float(x["roi_aleatoric_entropy_bits"]) for x in grp]

        # Compare with contradictory prompt for same (target, side)
        contra_rows = [
            x for x in rows
            if x["target"] == target and x["side"] == side and x["variant_type"] == "contradictory"
        ]
        canon_rows = [
            x for x in grp
            if x["variant_type"] == "lateralized" and x["variant_subtype"] == "canonical_sentence"
        ]
        canon_epi = float(np.mean([x["roi_epistemic_variance"] for x in canon_rows])) if canon_rows else np.nan
        contra_epi = float(np.mean([x["roi_epistemic_variance"] for x in contra_rows])) if contra_rows else np.nan

        stability.append({
            "target": target,
            "side": side,
            "n_synonym_evals": len(grp),
            "canonical_epistemic_var": round(canon_epi, 6),
            "synonym_mean_epistemic_var": round(float(np.mean(epi_vars)), 6),
            "synonym_std_epistemic_var": round(float(np.std(epi_vars)), 6),
            "synonym_max_min_ratio_epistemic_var": round(float(np.max(epi_vars) / max(float(np.min(epi_vars)), 1e-8)), 3),
            "contradictory_epistemic_var": round(contra_epi, 6) if not math.isnan(contra_epi) else np.nan,
            "contradictory_gt_canonical_uncertainty": bool(contra_epi > canon_epi) if not (math.isnan(contra_epi) or math.isnan(canon_epi)) else False,
            "synonym_mean_aleatoric_entropy": round(float(np.mean(alea_ents)), 6),
        })

    return summary, stability


# ===========================================================================
# EXPERIMENT V: 4-Way Image vs. Text Decoder Ablation
# ===========================================================================
ABLATION_SETTINGS: List[Tuple[str, bool, bool]] = [
    ("both_on", True, True),
    ("image_only", True, False),
    ("text_only", False, True),
    ("both_off", False, False),
]


def run_experiment_5_decoder_ablation(
    predictor,
    neurite_oasis_dir: Union[str, Path],
    results_root: Union[str, Path],
    structure_scope: str = "core_7",
    max_subjects: Optional[int] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    save_nifti_subjects: int = 2,
    save_rle_in_csv: bool = True,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Experiment V (Section 6 of `Axis_II_Prompt_Sensitivity_Nipu.docx`):
      Runs the 4 combinations of Image Decoder (`VoxTellDecoder`) and Text Prompt Decoder
      (`TransformerDecoder`):
        1. `both_on`    (Image Decoder = ON,  Text Decoder = ON)
        2. `image_only` (Image Decoder = ON,  Text Decoder = OFF)
        3. `text_only`  (Image Decoder = OFF, Text Decoder = ON)
        4. `both_off`   (Image Decoder = OFF, Text Decoder = OFF)
      For each setting, repeats Experiments I (Base / Left / Right comparison) and
      II (Synonym / Paraphrase sensitivity) to determine which pathway carries
      laterality and wording sensitivity.
    """
    mgr = ExperimentOutputManager(results_root, "exp5_decoder_ablation")
    structures = resolve_structures(structure_scope)
    all_catalog = build_oasis_prompt_catalog(structures=structures, include_absent_negatives=False)

    # Include Base, Canonical Left/Right, Short Left/Right, Clinical Synonym 1 & 2, Paraphrase 1, Location wording
    ablation_subtypes = {
        "canonical_sentence",
        "short_term",
        "clinical_synonym_1",
        "clinical_synonym_2",
        "paraphrase_1",
        "anatomical_location_wording",
    }
    ablation_prompts = [
        pr for pr in all_catalog
        if pr.variant_subtype in ablation_subtypes
    ]
    prompt_texts = [pr.prompt_text for pr in ablation_prompts]
    text_embeddings = predictor.embed_text_prompts(prompt_texts)

    subjects = discover_oasis_subjects(
        neurite_oasis_dir, max_subjects=max_subjects, start_index=start_index, end_index=end_index
    )

    all_rows: List[Dict[str, Any]] = []

    for s_idx, s_info in enumerate(subjects):
        sid = s_info["subject_id"]
        if resume and mgr.has_subject_checkpoint(sid):
            ckpt = mgr.load_subject_checkpoint(sid)
            all_rows.extend(ckpt["rows"])
            print(f"[Exp5-Ablation][resume {s_idx + 1}/{len(subjects)}] {sid}")
            continue

        t0 = time.time()
        img_arr, _ = load_volume_ras(s_info["image_path"])
        lbl_arr, _ = load_seg_ras(s_info["label_path"])
        data, bbox, orig_shape = preprocess_volume(predictor, img_arr)
        shape_s = shape_to_str(orig_shape)

        subj_rows: List[Dict[str, Any]] = []
        npz_dict: Dict[str, np.ndarray] = {}

        for setting_name, img_dec_on, txt_dec_on in ABLATION_SETTINGS:
            logits, _ = run_sliding_window_custom(
                predictor,
                data,
                text_embeddings,
                image_decoder_on=img_dec_on,
                text_decoder_on=txt_dec_on,
                enable_mc_dropout=False,
                capture_internals=False,
            )
            masks = logits_to_full_masks(logits, bbox, orig_shape, return_probs=False)

            mask_by_key: Dict[Tuple[str, str, str, str], np.ndarray] = {}
            pr_by_key: Dict[Tuple[str, str, str, str], PromptRecord] = {}
            for i, pr in enumerate(ablation_prompts):
                key = (pr.target, pr.side, pr.variant_type, pr.variant_subtype)
                mask_by_key[key] = masks[i]
                pr_by_key[key] = pr

            for target in structures:
                left_id, right_id = OASIS_BILATERAL_PAIRS[target]
                gt_left = (lbl_arr == left_id)
                gt_right = (lbl_arr == right_id)
                gt_both = np.logical_or(gt_left, gt_right)

                m_base = mask_by_key[(target, "none", "base", "canonical_sentence")]
                m_left = mask_by_key[(target, "left", "lateralized", "canonical_sentence")]
                m_right = mask_by_key[(target, "right", "lateralized", "canonical_sentence")]

                dice_lr = dice_score(m_left, m_right)
                dice_bl = dice_score(m_base, m_left)
                dice_br = dice_score(m_base, m_right)
                g_left = mask_geometry(m_left)
                g_right = mask_geometry(m_right)
                shift_lr_vox, shift_lr_norm = centroid_distance(g_left, g_right, orig_shape)

                # Evaluate every prompt for this target under this ablation setting
                target_prs = [pr for pr in ablation_prompts if pr.target == target]
                for pr in target_prs:
                    key = (pr.target, pr.side, pr.variant_type, pr.variant_subtype)
                    m_cur = mask_by_key[key]
                    g_cur = mask_geometry(m_cur)

                    d_gt_l = dice_score(m_cur, gt_left)
                    d_gt_r = dice_score(m_cur, gt_right)
                    d_gt_both = dice_score(m_cur, gt_both)

                    if pr.side == "left":
                        d_matched = d_gt_l
                        d_opp = d_gt_r
                        d_vs_ref = dice_score(m_cur, m_left)
                    elif pr.side == "right":
                        d_matched = d_gt_r
                        d_opp = d_gt_l
                        d_vs_ref = dice_score(m_cur, m_right)
                    else:
                        d_matched = d_gt_both
                        d_opp = np.nan
                        d_vs_ref = dice_score(m_cur, m_base)

                    row = {
                        "subject_id": sid,
                        "ablation_setting": setting_name,
                        "image_decoder_on": img_dec_on,
                        "text_decoder_on": txt_dec_on,
                        "target": target,
                        "side": pr.side,
                        "prompt_id": pr.prompt_id,
                        "variant_type": pr.variant_type,
                        "variant_subtype": pr.variant_subtype,
                        "prompt_text": pr.prompt_text,
                        "volume_shape_ras": shape_s,
                        "n_pred_voxels": g_cur["n_voxels"],
                        "dice_vs_matched_gt": round(d_matched, 4),
                        "dice_vs_opposite_gt": round(d_opp, 4) if not math.isnan(d_opp) else np.nan,
                        "dice_vs_gt_left": round(d_gt_l, 4),
                        "dice_vs_gt_right": round(d_gt_r, 4),
                        "dice_vs_canonical_ref": round(d_vs_ref, 4),
                        "pred_change_1_minus_dice_ref": round(1.0 - d_vs_ref, 4),
                        "setting_dice_left_vs_right": round(dice_lr, 4),
                        "setting_dice_base_vs_left": round(dice_bl, 4),
                        "setting_dice_base_vs_right": round(dice_br, 4),
                        "setting_centroid_shift_lr_vox": shift_lr_vox,
                        "setting_centroid_shift_lr_norm": shift_lr_norm,
                        "wrong_side_error": bool(not math.isnan(d_opp) and d_opp > d_matched and d_opp > 0.1),
                    }
                    if save_rle_in_csv:
                        row["pred_mask_rle_b64"] = encode_mask_rle(m_cur)
                    subj_rows.append(row)

                    if pr.variant_subtype == "canonical_sentence":
                        npz_dict[f"{setting_name}_{target.replace(' ', '_')}_{pr.side}"] = np.packbits(m_cur.ravel())
                        if s_idx < save_nifti_subjects and target == "hippocampus":
                            mgr.save_nifti(
                                f"{sid}_{setting_name}_{target}_{pr.side}.nii.gz",
                                m_cur,
                                reference_nifti_path=s_info["image_path"],
                            )

        mgr.save_npz_bundle(f"{sid}_exp5_ablation_masks_packed.npz", shape=np.array(orig_shape), **npz_dict)
        mgr.save_subject_checkpoint(sid, subj_rows)
        all_rows.extend(subj_rows)
        print(f"[Exp5-Ablation][{s_idx + 1}/{len(subjects)}] {sid} (4 settings) done in {time.time() - t0:.1f}s")

    summary_rows = _summarize_experiment_5(all_rows)
    paths = mgr.write_all_formats(
        metrics_rows=all_rows,
        summary_rows=summary_rows,
        report_title="Axis II — Experiment V: 4-Way Image vs. Text Decoder Ablation",
        report_sections=[
            (
                "Experiment V Overview",
                "Compares the 4 combinations of Image Decoder (`VoxTellDecoder`) and Text Prompt Decoder "
                "(`TransformerDecoder`): `both_on`, `image_only`, `text_only`, and `both_off` across "
                "Experiment I (Base/Left/Right separation) and Experiment II (Synonym sensitivity).",
            ),
        ],
        metadata={"n_subjects": len(subjects), "structures": structures, "settings": [s[0] for s in ABLATION_SETTINGS]},
    )
    return {"paths": paths, "summary": summary_rows}


def _summarize_experiment_5(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_setting_target: Dict[Tuple[str, bool, bool, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["ablation_setting"], bool(r["image_decoder_on"]), bool(r["text_decoder_on"]), r["target"])
        by_setting_target.setdefault(key, []).append(r)

    summary: List[Dict[str, Any]] = []
    for (setting, img_on, txt_on, target), grp in sorted(by_setting_target.items()):
        canon_lat = [x for x in grp if x["variant_type"] == "lateralized" and x["variant_subtype"] == "canonical_sentence"]
        syn_grp = [x for x in grp if x["variant_type"] in ("synonym", "paraphrase", "location_based")]

        def _m(sub, col):
            vals = [float(x[col]) for x in sub if x.get(col) is not None and not (isinstance(x[col], float) and math.isnan(x[col]))]
            return round(float(np.mean(vals)), 4) if vals else np.nan

        summary.append({
            "ablation_setting": setting,
            "image_decoder": "On" if img_on else "Off",
            "text_decoder": "On" if txt_on else "Off",
            "target": target,
            "n_evals": len(grp),
            "canonical_dice_vs_matched_gt": _m(canon_lat, "dice_vs_matched_gt"),
            "canonical_dice_vs_opposite_gt": _m(canon_lat, "dice_vs_opposite_gt"),
            "canonical_wrong_side_rate": _m(canon_lat, "wrong_side_error"),
            "dice_left_vs_right_overlap": _m(canon_lat, "setting_dice_left_vs_right"),
            "dice_base_vs_left_overlap": _m(canon_lat, "setting_dice_base_vs_left"),
            "centroid_shift_lr_vox": _m(canon_lat, "setting_centroid_shift_lr_vox"),
            "synonym_mean_dice_vs_matched_gt": _m(syn_grp, "dice_vs_matched_gt"),
            "synonym_mean_pred_change_vs_ref": _m(syn_grp, "pred_change_1_minus_dice_ref"),
        })
    return summary
