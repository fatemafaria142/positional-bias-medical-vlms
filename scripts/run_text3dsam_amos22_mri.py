"""
Run the pretrained Text3DSAM checkpoint (MagicXin/Text3DSAM) against every
(subject, organ) row in samples/amos22_mri/manifest_amos22_mri.csv, and
record the model's *predicted* voxel centroid/bbox/anatomical-side next to
the existing ground truth -- for the cross-organ positional-bias comparison
described in samples/amos22_mri/AMOS22_DATASET.md.

For each subject the full image volume is loaded and preprocessed once
(resized to the model's expected input size, e.g. 128x256x256); each of that
subject's 13 organ prompts is then run through the model independently
(one `model.generate()` call per prompt, not the combined multi-class
argmax pred.py uses for its saved segmentation) so that each organ's
predicted mask reflects only that organ's own prompt -- this is what makes
drift toward a different organ's location visible.

Requires the Text3DSAM checkout + CPU venv set up under Text3DSAM/ (see
Text3DSAM/checkpoints/Text3DSAM for the downloaded pretrained weights).

Usage (from repo root, using the Text3DSAM venv):
    Text3DSAM/.venv/Scripts/python scripts/run_text3dsam_amos22_mri.py
"""

import csv
import sys
import time
from pathlib import Path

import monai.transforms as mtf
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
TEXT3DSAM_ROOT = REPO_ROOT / "Text3DSAM"
sys.path.insert(0, str(TEXT3DSAM_ROOT))

from src.model.modeling import Med3DSeg  # noqa: E402
from transformers import BertTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common3d import axcodes_for, mask_ground_truth_3d  # noqa: E402

MODEL_PATH = TEXT3DSAM_ROOT / "checkpoints" / "Text3DSAM"
MANIFEST_PATH = REPO_ROOT / "samples" / "amos22_mri" / "manifest_amos22_mri.csv"
DATA_DIR = REPO_ROOT / "samples" / "amos22_mri"
OUT_PATH = REPO_ROOT / "samples" / "amos22_mri" / "predictions_text3dsam.csv"
THRESHOLD = 0.5


def load_manifest_by_subject(manifest_path):
    subjects = {}
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            subjects.setdefault(row["subject_id"], []).append(row)
    return subjects


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def main():
    print(f"[info] loading model from {MODEL_PATH}")
    model = Med3DSeg.from_pretrained(str(MODEL_PATH))
    tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH))
    model.eval()

    pre_transforms = mtf.Compose(
        [
            mtf.EnsureChannelFirst(channel_dim="no_channel"),
            mtf.Resize(spatial_size=model.config.image_size, mode="trilinear"),
            mtf.ToTensor(dtype=torch.float32),
        ]
    )

    subjects = load_manifest_by_subject(MANIFEST_PATH)
    print(f"[info] {len(subjects)} subjects, "
          f"{sum(len(v) for v in subjects.values())} organ prompts total")

    out_rows = []
    t_start = time.time()

    for si, (subject_id, rows) in enumerate(subjects.items(), 1):
        image_path = DATA_DIR / rows[0]["image_path"]
        label_path = DATA_DIR / rows[0]["label_path"]

        image_nii = nib.load(str(image_path))
        image_np = image_nii.get_fdata().astype(np.float32)
        label_vol = nib.load(str(label_path)).get_fdata().astype(int)
        axcodes = axcodes_for(image_path)
        origin_shape = image_np.shape

        preprocessed = pre_transforms(image_np).unsqueeze(0)

        print(f"[{si}/{len(subjects)}] {subject_id} shape={origin_shape} "
              f"organs={len(rows)}")

        for row in rows:
            organ = row["organ"]
            label_id = int(row["label_id"])
            prompt_text = row["prompt_text"]

            prompt_tokens = tokenizer(
                prompt_text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            t0 = time.time()
            with torch.no_grad():
                logits = model.generate(
                    image=preprocessed,
                    input_ids=prompt_tokens["input_ids"],
                    attention_mask=prompt_tokens["attention_mask"],
                )
                logits = F.interpolate(logits, size=origin_shape, mode="trilinear")
                pred_prob = torch.sigmoid(logits).squeeze(0).squeeze(0).numpy()
            elapsed = time.time() - t0

            pred_mask = pred_prob > THRESHOLD
            gt_mask = label_vol == label_id
            dice = dice_score(pred_mask, gt_mask)
            n_voxels = int(pred_mask.sum())

            gt = mask_ground_truth_3d(pred_mask, axcodes) if n_voxels > 0 else None

            out_row = {
                "subject_id": subject_id,
                "split": row["split"],
                "organ": organ,
                "label_id": label_id,
                "prompt_text": prompt_text,
                "volume_shape": row["volume_shape"],
                "n_pred_voxels": n_voxels,
                "dice": round(dice, 4),
                "inference_seconds": round(elapsed, 2),
                "pred_centroid_vox_0": None,
                "pred_centroid_vox_1": None,
                "pred_centroid_vox_2": None,
                "pred_bbox_vox_0min": None,
                "pred_bbox_vox_1min": None,
                "pred_bbox_vox_2min": None,
                "pred_bbox_vox_0max": None,
                "pred_bbox_vox_1max": None,
                "pred_bbox_vox_2max": None,
                "pred_side_axis0": None,
                "pred_side_axis1": None,
                "pred_side_axis2": None,
                "gt_centroid_vox_0": row["gt_centroid_vox_0"],
                "gt_centroid_vox_1": row["gt_centroid_vox_1"],
                "gt_centroid_vox_2": row["gt_centroid_vox_2"],
                "gt_side_axis0": row["gt_side_axis0"],
                "gt_side_axis1": row["gt_side_axis1"],
                "gt_side_axis2": row["gt_side_axis2"],
            }

            if gt is not None:
                (cx, cy, cz), bbox, (s0, s1, s2) = gt
                out_row.update({
                    "pred_centroid_vox_0": cx,
                    "pred_centroid_vox_1": cy,
                    "pred_centroid_vox_2": cz,
                    "pred_bbox_vox_0min": bbox[0],
                    "pred_bbox_vox_1min": bbox[1],
                    "pred_bbox_vox_2min": bbox[2],
                    "pred_bbox_vox_0max": bbox[3],
                    "pred_bbox_vox_1max": bbox[4],
                    "pred_bbox_vox_2max": bbox[5],
                    "pred_side_axis0": s0,
                    "pred_side_axis1": s1,
                    "pred_side_axis2": s2,
                })

            out_rows.append(out_row)
            print(f"    {organ:22s} dice={dice:.3f} voxels={n_voxels:8d} "
                  f"({elapsed:.1f}s)")

    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[done] wrote {len(out_rows)} predictions -> {OUT_PATH} "
          f"in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
