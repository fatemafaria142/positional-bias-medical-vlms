"""
Scores a model's raw predictions (e.g. results/paligemma/results_paligemma.csv)
against ground truth, producing the wrong-side / no-shift / distance metrics
from position_bias_research_plan.md's Comparison Specification.

Expects each row to already have: dataset, sample_id, prompt_id,
pred_centroid_x/y, pred_side_lr/ud, gt_centroid_x/y, gt_side_lr/ud
(this is exactly what models/paligemma/run_paligemma.py writes; any other
model's runner should write the same columns so this scorer works unchanged).

Adds, per row:
  - side_correct: predicted side matches the prompted side (P1-P4 only)
  - distance_norm: Euclidean distance between predicted and ground-truth
    centroid, normalized by the image diagonal
  - shifted_from_p0: predicted centroid moved more than 5% of image
    width/height relative to this model's own P0 prediction on the same image

Usage:
    python scripts/score_predictions.py results/paligemma/results_paligemma.csv
"""

import csv
import math
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIFT_TOLERANCE_FRAC = 0.05  # 5% of image width/height, per the research plan


def image_size(dataset: str, sample_id: str, rows_by_key) -> tuple[int, int]:
    # image_path is only in the manifest, not the results file, so re-read it once per sample.
    sample_dir = REPO_ROOT / "samples" / dataset
    manifest_path = sample_dir / f"manifest_{dataset}.csv"
    if sample_id not in rows_by_key:
        with manifest_path.open() as f:
            for r in csv.DictReader(f):
                rows_by_key.setdefault(r["sample_id"], sample_dir / r["image_path"])
    img_path = rows_by_key[sample_id]
    with Image.open(img_path) as im:
        return im.size


def score(results_path: Path):
    with results_path.open() as f:
        rows = list(csv.DictReader(f))

    img_path_cache: dict[str, dict] = {}
    size_cache: dict[tuple, tuple] = {}

    def get_size(dataset, sample_id):
        key = (dataset, sample_id)
        if key not in size_cache:
            cache = img_path_cache.setdefault(dataset, {})
            size_cache[key] = image_size(dataset, sample_id, cache)
        return size_cache[key]

    # index P0 predictions per (dataset, sample_id) for the shift check
    p0_pred = {}
    for r in rows:
        if r["prompt_id"] == "P0" and r["pred_centroid_x"]:
            p0_pred[(r["dataset"], r["sample_id"])] = (
                float(r["pred_centroid_x"]), float(r["pred_centroid_y"])
            )

    scored = []
    for r in rows:
        out = dict(r)
        has_pred = bool(r["pred_centroid_x"])
        w, h = get_size(r["dataset"], r["sample_id"])

        if not has_pred:
            out.update(side_correct="", distance_norm="", shifted_from_p0="")
            scored.append(out)
            continue

        px, py = float(r["pred_centroid_x"]), float(r["pred_centroid_y"])
        gx, gy = float(r["gt_centroid_x"]), float(r["gt_centroid_y"])

        # Wrong-side check: only meaningful for positional prompts (P1-P4),
        # comparing predicted side to the PROMPTED side, not the gt side --
        # per RQ3, we want to know if the model followed the text.
        if r["prompt_id"] == "P0":
            side_correct = ""
        else:
            prompted_lr = "left" if r["prompt_id"] == "P1" else "right" if r["prompt_id"] == "P2" else None
            prompted_ud = "upper" if r["prompt_id"] == "P3" else "lower" if r["prompt_id"] == "P4" else None
            if prompted_lr:
                side_correct = r["pred_side_lr"] == prompted_lr
            else:
                side_correct = r["pred_side_ud"] == prompted_ud

        diagonal = math.hypot(w, h)
        distance_norm = math.hypot(px - gx, py - gy) / diagonal

        p0 = p0_pred.get((r["dataset"], r["sample_id"]))
        if p0 is None or r["prompt_id"] == "P0":
            shifted_from_p0 = ""
        else:
            shift_px = math.hypot(px - p0[0], py - p0[1])
            tolerance = SHIFT_TOLERANCE_FRAC * max(w, h)
            shifted_from_p0 = shift_px > tolerance

        out.update(side_correct=side_correct, distance_norm=distance_norm, shifted_from_p0=shifted_from_p0)
        scored.append(out)

    out_path = results_path.with_name(results_path.stem + "_scored.csv")
    fieldnames = list(scored[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored)

    n_pred = sum(1 for r in scored if r["pred_centroid_x"])
    n_correct = sum(1 for r in scored if r["side_correct"] is True)
    n_positional = sum(1 for r in scored if r["prompt_id"] != "P0" and r["pred_centroid_x"])
    n_noshift = sum(1 for r in scored if r["shifted_from_p0"] is False)

    print(f"[ok] scored {len(scored)} rows -> {out_path}")
    print(f"[summary] predictions with a detection: {n_pred}/{len(scored)}")
    if n_positional:
        print(f"[summary] wrong-side rate (P1-P4): {1 - n_correct / n_positional:.1%}")
        print(f"[summary] no-shift (prompt-following failure) rate (P1-P4): {n_noshift / n_positional:.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} <results_csv_path>")
    score(Path(sys.argv[1]))
