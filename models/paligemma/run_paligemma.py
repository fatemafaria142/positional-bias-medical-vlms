"""
End-to-end runner for PaliGemma 2 (3B) against all 3 sampled datasets
(samples/lgg, samples/kvasir, samples/isic).

What this does, per row of each dataset's manifest_<dataset>.csv:
  1. Load the image and reformat that row's prompt_text into PaliGemma's
     required "segment {X}" syntax (see note below).
  2. Run the model, get its raw text output.
  3. Parse the 4 <locXXXX> tokens PaliGemma always emits (the bounding box
     around the segmented object) into pixel coordinates.
  4. Reduce that bbox to a center point, and classify its left/right and
     upper/lower side, exactly like mask_ground_truth() does for ground truth.
  5. Write one row per (image, prompt) into results/paligemma/results_paligemma.csv.

Note on prompt syntax: PaliGemma 2's segmentation task is normally invoked
as the fixed "segment {noun}" instruction (see position_bias_research_plan.md
and HOW_TO_RUN.md). To keep the P1-P4 positional wording meaningful for this
model (rather than silently dropping it), this script keeps the noun phrase
descriptive instead of a single word, e.g.:
    "Please locate the polyp on the right side of this image."
    -> "segment the polyp on the right side of this image"
This is a judgment call, not an official PaliGemma spec -- verify the model
actually responds to the positional wording (vs. ignoring everything after
the first noun) before trusting P1-P4 results for this model.

Note on mask decoding: PaliGemma's segmentation output also emits 16
<segXXX> tokens that encode a pixel mask. This script does NOT decode those
into a full mask -- it scores PaliGemma at the bbox/centroid level only
(derived from its <loc> tokens, which are always present). This is enough
for the core positional-bias RQ1-RQ3 (side/shift/distance), but means no
Dice/IoU shape-quality score for this model.

Setup:
    1. pip install transformers accelerate torch pillow
    2. Accept the PaliGemma 2 license at
       https://huggingface.co/google/paligemma2-3b-pt-224
    3. Put your HF token in a `.env` file at the repo root (already done,
       see .env -- never commit this file, it's in .gitignore):
           HF_TOKEN=hf_...

Usage:
    python models/paligemma/run_paligemma.py lgg
    python models/paligemma/run_paligemma.py kvasir
    python models/paligemma/run_paligemma.py isic
    python models/paligemma/run_paligemma.py all
"""

import csv
import os
import re
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_ID = "google/paligemma2-3b-pt-224"
DATASETS = ("lgg", "kvasir", "isic")
LOC_RE = re.compile(r"<loc(\d{4})>")


def load_env():
    """Minimal .env loader (no extra dependency) -- reads REPO_ROOT/.env
    and sets HF_TOKEN in os.environ if not already set."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def to_paligemma_prompt(prompt_text: str) -> str:
    """'Please locate the polyp on the right side of this image.'
    -> 'segment the polyp on the right side of this image'"""
    s = prompt_text.strip()
    if s.lower().startswith("please locate "):
        s = s[len("please locate "):]
    s = s.rstrip(". ").strip()
    return f"segment {s}"


def parse_bbox_from_loc_tokens(generated_text: str, img_w: int, img_h: int):
    """PaliGemma emits 4 <locNNNN> tokens per detected/segmented object, in
    (y_min, x_min, y_max, x_max) order, each normalized to a 0-1023 grid.
    Returns (xmin, ymin, xmax, ymax) in pixel coordinates, or None if the
    model didn't emit 4 loc tokens (e.g. it claims not to find the object)."""
    matches = LOC_RE.findall(generated_text)
    if len(matches) < 4:
        return None
    y0, x0, y1, x1 = (int(v) / 1023.0 for v in matches[:4])
    return (x0 * img_w, y0 * img_h, x1 * img_w, y1 * img_h)


def bbox_center_and_side(bbox, img_w, img_h):
    xmin, ymin, xmax, ymax = bbox
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    side_lr = "left" if cx < img_w / 2 else "right"
    side_ud = "upper" if cy < img_h / 2 else "lower"
    return cx, cy, side_lr, side_ud


def load_model():
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    token = os.environ.get("HF_TOKEN")
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return processor, model


def run_one(processor, model, image: Image.Image, prompt: str) -> str:
    import torch

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    return processor.decode(output[0][input_len:], skip_special_tokens=False)


def run_dataset(dataset_name: str, processor, model, writer: csv.DictWriter):
    sample_dir = REPO_ROOT / "samples" / dataset_name
    manifest_path = sample_dir / f"manifest_{dataset_name}.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}\nRun scripts/sample_{dataset_name}.py first.")

    with manifest_path.open() as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows, 1):
        img_path = sample_dir / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        pg_prompt = to_paligemma_prompt(row["prompt_text"])
        raw_output = run_one(processor, model, image, pg_prompt)
        bbox = parse_bbox_from_loc_tokens(raw_output, img_w, img_h)

        if bbox is None:
            pred_cx = pred_cy = pred_side_lr = pred_side_ud = ""
        else:
            pred_cx, pred_cy, pred_side_lr, pred_side_ud = bbox_center_and_side(bbox, img_w, img_h)

        writer.writerow({
            "model": "paligemma2-3b",
            "dataset": dataset_name,
            "sample_id": row["sample_id"],
            "prompt_id": row["prompt_id"],
            "prompt_text": row["prompt_text"],
            "model_prompt": pg_prompt,
            "raw_output": raw_output,
            "pred_bbox_xmin": bbox[0] if bbox else "",
            "pred_bbox_ymin": bbox[1] if bbox else "",
            "pred_bbox_xmax": bbox[2] if bbox else "",
            "pred_bbox_ymax": bbox[3] if bbox else "",
            "pred_centroid_x": pred_cx,
            "pred_centroid_y": pred_cy,
            "pred_side_lr": pred_side_lr,
            "pred_side_ud": pred_side_ud,
            "gt_centroid_x": row["gt_centroid_x"],
            "gt_centroid_y": row["gt_centroid_y"],
            "gt_side_lr": row["gt_side_lr"],
            "gt_side_ud": row["gt_side_ud"],
        })
        print(f"[{dataset_name}] {i}/{len(rows)} {row['sample_id']} {row['prompt_id']} -> "
              f"{'no-detection' if bbox is None else f'{pred_side_lr}/{pred_side_ud}'}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in (*DATASETS, "all"):
        raise SystemExit(f"Usage: python {sys.argv[0]} [{'|'.join(DATASETS)}|all]")

    load_env()
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN not set. Add it to .env at the repo root.")

    targets = DATASETS if sys.argv[1] == "all" else (sys.argv[1],)

    processor, model = load_model()

    out_dir = REPO_ROOT / "results" / "paligemma"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_paligemma.csv"

    fieldnames = [
        "model", "dataset", "sample_id", "prompt_id", "prompt_text", "model_prompt",
        "raw_output", "pred_bbox_xmin", "pred_bbox_ymin", "pred_bbox_xmax", "pred_bbox_ymax",
        "pred_centroid_x", "pred_centroid_y", "pred_side_lr", "pred_side_ud",
        "gt_centroid_x", "gt_centroid_y", "gt_side_lr", "gt_side_ud",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dataset_name in targets:
            run_dataset(dataset_name, processor, model, writer)

    print(f"[done] wrote predictions -> {out_path}")


if __name__ == "__main__":
    main()
