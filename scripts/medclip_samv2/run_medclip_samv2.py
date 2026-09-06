"""Run MedCLIP-SAMv2's zero-shot pipeline (BiomedCLIP saliency -> kmeans -> SAM) over
one dataset's manifest CSV, in small resumable batches, on CPU.

Usage:
    models/MedCLIP-SAMv2/.venv/Scripts/python.exe scripts/run_medclip_samv2.py acdc --batch-size 5
    models/MedCLIP-SAMv2/.venv/Scripts/python.exe scripts/run_medclip_samv2.py isic_task2 --batch-size 5
    models/MedCLIP-SAMv2/.venv/Scripts/python.exe scripts/run_medclip_samv2.py chaos --batch-size 5

Must be run with the MedCLIP-SAMv2 venv's python (models/MedCLIP-SAMv2/.venv), not the
project's own environment, since it needs transformers==4.49.0 (pinned for compatibility
with the community BiomedCLIP wrapper) plus SAM/open_clip/etc.

Resumability: results are appended to Results/medclip_samv2/<dataset>_results.csv after
every batch. Re-running the same command skips (dataset, sample_id, region, prompt_id)
rows already present in that file, so you can stop anytime (Ctrl+C) and pick up later.

Model choices (see conversation record for why):
- Base (non-finetuned) BiomedCLIP -- MedCLIP-SAMv2's shipped fine-tuned checkpoint was
  only tuned on brain/breast tumors, which doesn't match ACDC/ISIC/CHAOS's domains, so
  using it would bias saliency quality unevenly across datasets. Base BiomedCLIP keeps
  this a true zero-shot comparison across all 3.
- SAM ViT-B (not ViT-H) -- ~9s/image forward pass on CPU vs. minutes for ViT-H.

P_ALL scoring note: a combined prompt naming several regions ("segment the liver, the
right kidney, ...") still only gets ONE saliency call, because BiomedCLIP pools the
whole sentence into one embedding -- there's no per-region signal inside it to split
out. Tried forcing multiple output blobs via --num-contours>1 + SAM's --multicontour;
verified visually that this only added corner-noise artifacts, not real per-organ
regions, so it was reverted. P_ALL is scored as one merged prediction against the union
of all named regions' masks -- an honest "did it find the general combined area"
completeness check, not a per-region localization result. Genuine per-region
coordinates come from the isolated P0-P4 rows (--isolated flag), which each get their
own dedicated saliency call for that one region's own text.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEDCLIP_DIR = os.path.join(REPO_ROOT, "models", "MedCLIP-SAMv2")
VENV_PY = os.path.join(MEDCLIP_DIR, ".venv", "Scripts", "python.exe")
SAM_CHECKPOINT = os.path.join(MEDCLIP_DIR, "segment-anything", "sam_checkpoints", "sam_vit_b_01ec64.pth")

STAGING_DIR = os.path.join(REPO_ROOT, "Results", "medclip_samv2", "_staging")
RESULTS_DIR = os.path.join(REPO_ROOT, "Results", "medclip_samv2")


def mask_ground_truth(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if len(xs) == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return cx, cy, bbox


def dice_iou(pred_bool, gt_bool):
    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    pred_sum = pred_bool.sum()
    gt_sum = gt_bool.sum()
    dice = (2.0 * inter / (pred_sum + gt_sum)) if (pred_sum + gt_sum) > 0 else 0.0
    iou = (inter / union) if union > 0 else 0.0
    return float(dice), float(iou)


def load_manifest(dataset_name):
    manifest_path = os.path.join(REPO_ROOT, "dataset", dataset_name, f"manifest_{dataset_name}.csv")
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def load_done_keys(results_path):
    if not os.path.exists(results_path):
        return set()
    with open(results_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {(r["sample_id"], r["region"], r["prompt_id"]) for r in rows}


RESULT_FIELDS = [
    "dataset", "sample_id", "region", "prompt_id", "prompt_text",
    "image_path", "gt_mask_path", "pred_mask_path",
    "dice", "iou",
    "gt_centroid_x", "gt_centroid_y", "gt_side_lr", "gt_side_ud",
    "pred_centroid_x", "pred_centroid_y", "pred_side_lr", "pred_side_ud",
    "pred_blob_count",
    "runtime_seconds",
]


def append_results(results_path, rows):
    file_exists = os.path.exists(results_path)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def predicted_blobs(pred_mask, min_area=20):
    """Return [(cx, cy, area), ...] for each separate connected blob in pred_mask,
    largest first, dropping tiny noise blobs below min_area pixels."""
    n, _, stats, centroids = cv2.connectedComponentsWithStats(pred_mask.astype(np.uint8))
    blobs = []
    for i in range(1, n):  # 0 = background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cx, cy = centroids[i]
            blobs.append((float(cx), float(cy), int(area)))
    blobs.sort(key=lambda b: -b[2])
    return blobs


def union_mask(dataset_dir, mask_paths):
    masks = [np.array(Image.open(os.path.join(dataset_dir, p)).convert("L")) > 0 for p in mask_paths]
    out = masks[0]
    for m in masks[1:]:
        out = np.logical_or(out, m)
    return out


def run_batch(dataset_name, batch_rows, results_path):
    """batch_rows: list of manifest row dicts. Supports both isolated P0-P4 rows (one
    mask each, via mask_path) and P_ALL rows (multiple masks, via mask_paths, scored
    against their union)."""
    batch_dir = os.path.join(STAGING_DIR, dataset_name)
    images_dir = os.path.join(batch_dir, "images")
    sal_dir = os.path.join(batch_dir, "saliency")
    coarse_dir = os.path.join(batch_dir, "coarse")
    sam_dir = os.path.join(batch_dir, "sam")
    for d in (images_dir, sal_dir, coarse_dir, sam_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    dataset_dir = os.path.join(REPO_ROOT, "dataset", dataset_name)
    prompt_map = {}
    row_by_stage_name = {}
    for row in batch_rows:
        stage_name = f"{row['sample_id']}__{row['region']}__{row['prompt_id']}.png"
        src_image = os.path.join(dataset_dir, row["image_path"])
        img = Image.open(src_image).convert("RGB")
        img.save(os.path.join(images_dir, stage_name))
        prompt_map[stage_name] = row["prompt_text"]
        row_by_stage_name[stage_name] = row

    json_path = os.path.join(batch_dir, "prompts.json")
    with open(json_path, "w") as f:
        json.dump(prompt_map, f)

    t0 = time.time()
    subprocess.run(
        [
            VENV_PY, os.path.join(MEDCLIP_DIR, "saliency_maps", "generate_saliency_maps.py"),
            "--input-path", images_dir,
            "--output-path", sal_dir,
            "--model-name", "BiomedCLIP",
            "--json-path", json_path,
            "--reproduce",
            "--device", "cpu",
        ],
        cwd=os.path.join(MEDCLIP_DIR, "saliency_maps"),
        check=True,
    )

    # num-contours=1 (the default) everywhere: BiomedCLIP pools a whole sentence into
    # one embedding, so even a P_ALL prompt naming several regions only ever produces
    # ONE blended saliency blob -- there is no real per-region signal in it to split
    # out. Tried num-contours>1 + --multicontour to get separate per-region blobs; it
    # only added corner-noise artifacts (verified visually), not real extra regions.
    # Genuine per-region coordinates come from the isolated P0-P4 rows instead, which
    # each get their own dedicated saliency call for that one region's own text.
    subprocess.run(
        [
            VENV_PY, os.path.join(MEDCLIP_DIR, "postprocessing", "postprocess_saliency_maps.py"),
            "--input-path", images_dir,
            "--output-path", coarse_dir,
            "--sal-path", sal_dir,
            "--postprocess", "kmeans",
            "--filter",
        ],
        cwd=MEDCLIP_DIR,
        check=True,
    )

    subprocess.run(
        [
            VENV_PY, os.path.join(MEDCLIP_DIR, "segment-anything", "prompt_sam.py"),
            "--input", images_dir,
            "--mask-input", coarse_dir,
            "--output", sam_dir,
            "--model-type", "vit_b",
            "--checkpoint", SAM_CHECKPOINT,
            "--prompts", "boxes",
            "--device", "cpu",
        ],
        cwd=MEDCLIP_DIR,
        check=True,
    )
    batch_runtime = time.time() - t0
    per_row_runtime = batch_runtime / max(len(batch_rows), 1)

    result_rows = []
    for stage_name, row in row_by_stage_name.items():
        pred_path = os.path.join(sam_dir, stage_name)
        is_combined = row["prompt_id"] == "P_ALL"
        if is_combined:
            mask_list = row["mask_paths"].split(";")
            gt_mask = union_mask(dataset_dir, mask_list)
            gt_mask_path_field = row["mask_paths"]
            gt_result = mask_ground_truth(gt_mask)
            gt_cx, gt_cy, _ = gt_result
            height, width = gt_mask.shape
            gt_side_lr = "right" if gt_cx > width / 2 else "left"
            gt_side_ud = "upper" if gt_cy < height / 2 else "lower"
        else:
            gt_path = os.path.join(dataset_dir, row["mask_path"])
            gt_mask = np.array(Image.open(gt_path).convert("L")) > 0
            gt_mask_path_field = row["mask_path"]
            gt_cx, gt_cy = row["gt_centroid_x"], row["gt_centroid_y"]
            gt_side_lr, gt_side_ud = row["gt_side_lr"], row["gt_side_ud"]

        if not os.path.exists(pred_path):
            result_rows.append({
                "dataset": dataset_name, "sample_id": row["sample_id"], "region": row["region"],
                "prompt_id": row["prompt_id"], "prompt_text": row["prompt_text"],
                "image_path": row["image_path"], "gt_mask_path": gt_mask_path_field,
                "pred_mask_path": "", "dice": 0.0, "iou": 0.0,
                "gt_centroid_x": gt_cx, "gt_centroid_y": gt_cy,
                "gt_side_lr": gt_side_lr, "gt_side_ud": gt_side_ud,
                "pred_centroid_x": "", "pred_centroid_y": "", "pred_side_lr": "", "pred_side_ud": "",
                "pred_blob_count": 0,
                "runtime_seconds": round(per_row_runtime, 1),
            })
            continue

        pred_mask = np.array(Image.open(pred_path).convert("L")) > 127
        if pred_mask.shape != gt_mask.shape:
            pred_mask = np.array(
                Image.fromarray(pred_mask.astype(np.uint8) * 255).resize(
                    (gt_mask.shape[1], gt_mask.shape[0])
                )
            ) > 127

        dice, iou = dice_iou(pred_mask, gt_mask)
        pred_result = mask_ground_truth(pred_mask)
        height, width = gt_mask.shape
        if pred_result is not None:
            cx, cy, _ = pred_result
            pred_side_lr = "right" if cx > width / 2 else "left"
            pred_side_ud = "upper" if cy < height / 2 else "lower"
        else:
            cx, cy, pred_side_lr, pred_side_ud = "", "", "", ""

        # keep a permanent copy of the predicted mask before staging is wiped
        keep_dir = os.path.join(RESULTS_DIR, dataset_name, "pred_masks")
        os.makedirs(keep_dir, exist_ok=True)
        kept_path = os.path.join(keep_dir, stage_name)
        shutil.copyfile(pred_path, kept_path)

        blob_count = len(predicted_blobs(pred_mask))

        result_rows.append({
            "dataset": dataset_name, "sample_id": row["sample_id"], "region": row["region"],
            "prompt_id": row["prompt_id"], "prompt_text": row["prompt_text"],
            "image_path": row["image_path"], "gt_mask_path": gt_mask_path_field,
            "pred_mask_path": os.path.relpath(kept_path, REPO_ROOT),
            "dice": round(dice, 4), "iou": round(iou, 4),
            "gt_centroid_x": gt_cx, "gt_centroid_y": gt_cy,
            "gt_side_lr": gt_side_lr, "gt_side_ud": gt_side_ud,
            "pred_centroid_x": cx, "pred_centroid_y": cy,
            "pred_side_lr": pred_side_lr, "pred_side_ud": pred_side_ud,
            "pred_blob_count": blob_count,
            "runtime_seconds": round(per_row_runtime, 1),
        })

    append_results(results_path, result_rows)
    shutil.rmtree(batch_dir)
    return len(result_rows), batch_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["acdc", "isic_task2", "chaos"])
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None, help="Stop after this many rows this run (for a quick session)")
    parser.add_argument(
        "--isolated", action="store_true",
        help="Process the per-region P0-P4 isolated-prompt rows instead of the default P_ALL combined-prompt rows.",
    )
    args = parser.parse_args()

    all_rows = load_manifest(args.dataset)
    if args.isolated:
        rows = [r for r in all_rows if r["prompt_id"] != "P_ALL"]
    else:
        rows = [r for r in all_rows if r["prompt_id"] == "P_ALL"]

    results_path = os.path.join(RESULTS_DIR, f"{args.dataset}_results.csv")
    done = load_done_keys(results_path)
    todo = [r for r in rows if (r["sample_id"], r["region"], r["prompt_id"]) not in done]

    print(f"{args.dataset}: {len(rows)} total rows, {len(done)} already done, {len(todo)} remaining.")

    if args.max_rows:
        todo = todo[: args.max_rows]
        print(f"Limiting this run to {len(todo)} rows (--max-rows).")

    processed = 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        n, runtime = run_batch(args.dataset, batch, results_path)
        processed += n
        print(
            f"[{processed}/{len(todo)}] batch of {n} done in {runtime:.1f}s "
            f"({runtime / max(n, 1):.1f}s/row) -> {results_path}"
        )

    print("Done for this run. Re-run the same command later to continue where you left off.")


if __name__ == "__main__":
    main()
