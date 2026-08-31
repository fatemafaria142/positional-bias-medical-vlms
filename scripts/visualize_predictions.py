"""
Draws BOTH the ground-truth box/centroid AND a model's predicted box/centroid
onto each sampled image, so you can visually compare where the model pointed
vs. where the object actually is -- the prediction-side counterpart to
scripts/visualize_ground_truth.py (which only draws ground truth).

Reads a model's raw results CSV (e.g. results/paligemma/results_paligemma.csv,
as written by models/paligemma/run_paligemma.py) -- works for any model's
results file as long as it has the same columns: dataset, sample_id,
prompt_id, prompt_text, pred_bbox_xmin/ymin/xmax/ymax, pred_centroid_x/y,
pred_side_lr/ud, gt_centroid_x/y, gt_side_lr/ud.

One output image per (image, prompt) row -- unlike visualize_ground_truth.py,
which only needs one image per sample_id (ground truth doesn't change across
prompts), here the prediction CAN change per prompt, so all 5 P0-P4 variants
are drawn separately.

Output: <results_csv_dir>/visual_check/<sample_id>_<prompt_id>.png
    - yellow lines: image midline / midheight (reference)
    - red box + dot: ground truth
    - blue box + dot: model's prediction (skipped if the model had no detection)
    - top-left label: prompt_id, ground-truth side, predicted side

Usage:
    python scripts/visualize_predictions.py results/paligemma/results_paligemma.csv
    python scripts/visualize_predictions.py results/paligemma/results_paligemma.csv --limit 10
"""

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent


def visualize(results_path: Path, limit: int | None = None):
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")

    out_dir = results_path.parent / "visual_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    with results_path.open() as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]

    # image_path isn't in the results CSV, only in each dataset's manifest --
    # look it up once per (dataset, sample_id) instead of re-reading per row.
    image_path_cache: dict[str, dict[str, str]] = {}

    def get_image_path(dataset: str, sample_id: str) -> Path:
        cache = image_path_cache.setdefault(dataset, {})
        if not cache:
            manifest_path = REPO_ROOT / "samples" / dataset / f"manifest_{dataset}.csv"
            with manifest_path.open() as f:
                for r in csv.DictReader(f):
                    cache.setdefault(r["sample_id"], r["image_path"])
        return REPO_ROOT / "samples" / dataset / cache[sample_id]

    written = 0
    for row in rows:
        img_path = get_image_path(row["dataset"], row["sample_id"])
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        draw = ImageDraw.Draw(image)

        # Reference lines: true midline (vertical) and midheight (horizontal).
        draw.line([(w / 2, 0), (w / 2, h)], fill="yellow", width=1)
        draw.line([(0, h / 2), (w, h / 2)], fill="yellow", width=1)

        # Ground truth: red.
        gcx, gcy = float(row["gt_centroid_x"]), float(row["gt_centroid_y"])
        if row.get("gt_bbox_xmin"):
            gt_bbox = (float(row["gt_bbox_xmin"]), float(row["gt_bbox_ymin"]),
                       float(row["gt_bbox_xmax"]), float(row["gt_bbox_ymax"]))
            draw.rectangle(gt_bbox, outline="red", width=3)
        draw.ellipse((gcx - 5, gcy - 5, gcx + 5, gcy + 5), fill="red")

        # Prediction: blue -- only if the model actually returned a detection.
        has_pred = bool(row.get("pred_centroid_x"))
        if has_pred:
            pcx, pcy = float(row["pred_centroid_x"]), float(row["pred_centroid_y"])
            if row.get("pred_bbox_xmin"):
                pred_bbox = (float(row["pred_bbox_xmin"]), float(row["pred_bbox_ymin"]),
                             float(row["pred_bbox_xmax"]), float(row["pred_bbox_ymax"]))
                draw.rectangle(pred_bbox, outline="blue", width=3)
            draw.ellipse((pcx - 5, pcy - 5, pcx + 5, pcy + 5), fill="blue")
            pred_label = f"{row['pred_side_lr']}/{row['pred_side_ud']}"
        else:
            pred_label = "no-detection"

        label = (f"{row['sample_id']} | {row['prompt_id']} "
                  f"| GT(red)={row['gt_side_lr']}/{row['gt_side_ud']} "
                  f"| Pred(blue)={pred_label}")
        draw.rectangle((0, 0, draw.textlength(label) + 8, 16), fill="black")
        draw.text((4, 2), label, fill="white")

        out_file = out_dir / f"{row['sample_id']}_{row['prompt_id']}.png"
        image.save(out_file)
        written += 1

    print(f"[ok] wrote {written} annotated images -> {out_dir}")
    print("Red = ground truth. Blue = model prediction. "
          "Compare the blue box's side to the prompted direction, and to the red box's actual side.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit(f"Usage: python {sys.argv[0]} <results_csv_path> [--limit N]")

    limit = None
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(args[i + 1])
        del args[i:i + 2]

    visualize(Path(args[0]), limit=limit)
