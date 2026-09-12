"""
One-off generator for the two VoxTell rotation-sweep Kaggle notebooks
(voxtell-amos-rotation-multiangle-kaggle.ipynb, voxtell-oasis-rotation-multiangle-kaggle.ipynb).

Starts from the existing voxtell-{amos,oasis}-mirror-kaggle.ipynb (same
install/model-loading/helper cells -- those don't change), and rewrites only
the dataset-specific cells so the run:
  - is driven by ONE `PERSON` variable set in a single place -- every
    annotator (faria/nipu/hridoy) runs the exact same notebook/code, just
    changes that one line to their own name, and gets their own 2 subjects
    x 7 angles x organs
  - groups manifest rows by (subject_id, rotation_angle_deg) instead of just
    subject_id, since each angle is a physically different rotated volume
  - writes the predictions CSV with 'person', 'rotation_angle_deg',
    'image_path', 'label_path' columns added (repo-relative paths), PLUS the
    actual segmentation masks themselves -- 'gt_mask_rle_b64' and
    'pred_mask_rle_b64' (run-length-encoded, zlib-compressed, base64 text)
    and 'volume_shape_ras' (the shape needed to decode them). This makes
    each CSV fully self-contained for plotting real segmentation masks
    (not just a bbox/centroid) -- no need to also hand over the results zip
    or clone the repo. Predicted masks are still additionally saved as
    .npz (keyed by person_subject_rotANGLE) as a redundant backup.
  - adds a Dice-vs-rotation-angle analysis cell in place of the old
    cross-subject organ grid (which doesn't generalize to the angle axis)

Usage:
    python scripts/build_rotation_kaggle_notebooks.py
"""

import copy
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BRANCH = "faria/voxtell3d-rotation-medical-segmentation"


def cell(nb, i):
    return nb["cells"][i]


def set_source(c, text):
    lines = text.splitlines(keepends=True)
    c["source"] = lines
    c["outputs"] = c.get("outputs", [])
    if c["cell_type"] == "code":
        c["execution_count"] = None
        c["outputs"] = []


def build(template_name, dataset_name, organ, title, out_name):
    nb = json.loads((REPO_ROOT / template_name).read_text(encoding="utf-8"))

    # Cell 15: append RLE mask encode/decode helpers to the shared helpers cell
    # (dice_score/mask_ground_truth_3d/etc. are untouched -- this just adds two
    # new functions so the predictions CSV can carry the actual masks).
    helpers_cell = cell(nb, 15)
    helpers_src = "".join(helpers_cell["source"])
    helpers_src += '''

import base64
import zlib


def encode_mask_rle(mask: np.ndarray) -> str:
    """Run-length-encode a boolean mask (C-order flatten), zlib-compress the
    run lengths, and base64-encode the result as ASCII text -- compact
    (organ masks are small/contiguous blobs relative to the whole volume)
    and safe to store directly as one CSV cell. Decode with
    decode_mask_rle(s, shape) using the paired volume_shape_ras column."""
    flat = np.asarray(mask, dtype=bool).ravel(order='C')
    change_idx = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], change_idx, [flat.size]))
    runs = np.diff(boundaries).astype(np.uint32)
    if flat.size and flat[0]:
        runs = np.concatenate(([0], runs)).astype(np.uint32)  # runs always alternate starting with a (possibly zero) False-run
    raw = zlib.compress(runs.tobytes(), level=9)
    return base64.b64encode(raw).decode('ascii')


def decode_mask_rle(encoded: str, shape) -> np.ndarray:
    """Inverse of encode_mask_rle(); shape = volume_shape_ras split on 'x'."""
    runs = np.frombuffer(zlib.decompress(base64.b64decode(encoded)), dtype=np.uint32)
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    idx = 0
    value = False
    for run_len in runs:
        if value and run_len:
            flat[idx:idx + run_len] = True
        idx += int(run_len)
        value = not value
    return flat.reshape(shape, order='C')
'''
    set_source(helpers_cell, helpers_src)

    # Cell 0: title / description
    set_source(cell(nb, 0), f"""# VoxTell on {title} -- ROTATION SWEEP (Kaggle GPU)

Runs VoxTell (CVPR 2026) over the full **0/30/60/90/120/150/180 degree rotation sweep**
(`scripts/rotate_multiangle_samples.py`) for {dataset_name}, for ONE team member's subjects
at a time -- reads `<PERSON>/samples/{dataset_name}/` (every organ, every angle).

**Everyone on the team runs the exact same notebook.** The only thing that changes per
person is the single `PERSON = ...` line in the "Dataset config" section below -- set it to
your own name (matching your folder at the repo root) and the rest of the notebook picks up
your data automatically. Each team member runs it once for their own subjects.

Positional-bias design (see the per-dataset manifests): the prompt text and the TRUE
anatomical side (`gt_side_axis0/1/2`) are identical at every angle for a given
(subject, organ) -- rotating the image doesn't change what was asked or the real
anatomy. Only the organ's on-screen position (`gt_centroid_vox_*`, `gt_bbox_vox_*`)
changes with angle. So any drop in Dice as angle increases is a genuine positional bias,
not a change in the question being asked.

**The CSV is self-contained -- you only need to download this one file.** In addition to
`person`, `rotation_angle_deg`, `image_path`, `label_path`, every row also carries
`gt_mask_rle_b64` and `pred_mask_rle_b64`: the actual ground-truth and predicted
segmentation masks, run-length-encoded and zlib-compressed as base64 text (plus
`volume_shape_ras`, the shape needed to decode them back into a 3D array). So the full
mask -- not just a bounding box -- can be plotted later from this CSV alone, no need to
also share the results zip, the raw `.npz` masks, or clone the repo.
""")

    # Cell 9: clone -- point at THIS branch (where the per-annotator rotation
    # data actually lives), not main/voxtell3d-medical-segmentation.
    clone_cell = cell(nb, 9)
    clone_src = "".join(clone_cell["source"])
    clone_src = clone_src.replace(
        "'--branch', 'faria/voxtell3d-medical-segmentation'",
        f"'--branch', '{BRANCH}'",
    )
    set_source(clone_cell, clone_src)

    # Cell 16: dataset config markdown
    set_source(cell(nb, 16), f"""## 7. Dataset config -- {title}

**Set your name here -- this is the only line that differs between annotators.** Reads
`manifest_{dataset_name}.csv` from `<PERSON>/samples/{dataset_name}/` and groups rows by
`(subject_id, rotation_angle_deg)` -- each (subject, angle) pair is a distinct rotated
volume, so every organ at that angle is predicted in ONE `predict_single_image` call.
""")

    # Cell 17: dataset config code
    set_source(cell(nb, 17), f"""# vvv SET YOUR NAME HERE -- the only line that changes per team member vvv
PERSON = 'faria'  # <- put your own name here, matching your folder at the repo root
# ^^^ same notebook/code for everyone -- just change this one line and re-run ^^^

DATASET_NAME = '{dataset_name}'
ANGLES_DEG = [0, 30, 60, 90, 120, 150, 180]

# None = run every subject present in PERSON's manifest. Set an int to cap it (e.g. for a quick test).
N_SUBJECTS = None

def load_manifest_grouped(manifest_path):
    \"\"\"Group rows by (subject_id, rotation_angle_deg) -- each pair is a
    physically distinct rotated volume file.\"\"\"
    groups = {{}}
    with open(manifest_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['subject_id'], int(row['rotation_angle_deg']))
            groups.setdefault(key, []).append(row)
    return groups

DATA_DIR = RESEARCH_REPO / PERSON / 'samples' / DATASET_NAME
MANIFEST_PATH = DATA_DIR / f'manifest_{{DATASET_NAME}}.csv'

grouped = load_manifest_grouped(MANIFEST_PATH)
subject_ids = sorted({{sid for sid, _ in grouped}})
if N_SUBJECTS is not None:
    subject_ids = subject_ids[:N_SUBJECTS]

all_groups = {{  # (subject_id, angle) -> rows
    (sid, angle): rows for (sid, angle), rows in grouped.items() if sid in subject_ids
}}

n_organs_per_group = len(next(iter(all_groups.values())))
print(f'{{PERSON}} / {{DATASET_NAME}}: {{len(subject_ids)}} subject(s) {{subject_ids}}, '
      f'{{len(all_groups)}} (subject, angle) groups, {{n_organs_per_group}} organs/group '
      f'-> {{len(all_groups) * n_organs_per_group}} total rows')
""")

    # Cell 18: smoke-test markdown (angle-aware wording)
    set_source(cell(nb, 18), """## 8. Smoke test -- ONE (subject, angle) group, all its organs in ONE call

Time this before committing to the full run -- the next cell uses this number to
extrapolate the full sweep's runtime.
""")

    # Cell 19: smoke-test code
    set_source(cell(nb, 19), """(smoke_subject_id, smoke_angle), smoke_rows = next(iter(all_groups.items()))
smoke_organs = [r['organ'] for r in smoke_rows]
smoke_image_path = DATA_DIR / smoke_rows[0]['image_path']
smoke_label_path = DATA_DIR / smoke_rows[0]['label_path']
print(f'smoke test: {PERSON}/{smoke_subject_id} @ {smoke_angle} deg -- organs:', smoke_organs)

t0 = time.time()
smoke_img, smoke_props = load_volume_ras(smoke_image_path)  # raw (1, X, Y, Z), as VoxTell expects
smoke_pred = predictor.predict_single_image(smoke_img, smoke_organs)  # -> (num_prompts, X, Y, Z), already binary 0/1
smoke_elapsed = time.time() - t0

print(f'smoke test elapsed: {smoke_elapsed:.1f}s for {len(smoke_organs)} organs '
      f'({smoke_elapsed / len(smoke_organs):.2f}s/organ if it scaled linearly, '
      'though in practice most of this cost is the ONE shared image encoding pass)')
print('input image shape (fed to model):', smoke_img.shape)
print('prediction array shape:', smoke_pred.shape, 'dtype:', smoke_pred.dtype,
      '-- already binary (0/1), no thresholding needed')
""")

    # Cell 20: runtime markdown (unchanged wording, angle-aware)
    set_source(cell(nb, 20), """## 9. Runtime estimate for this sweep

Measured from the smoke test above (not guessed): VoxTell encodes each (subject, angle)
volume once and predicts all of that group's organs from that one encoding, so runtime
scales with the number of (person, subject, angle) groups, not (groups x organs).
""")

    # Cell 21: runtime code
    set_source(cell(nb, 21), """per_group_seconds = smoke_elapsed
n_groups = len(all_groups)

print(f'Measured: {per_group_seconds:.1f}s for 1 group ({n_organs_per_group} organs) on this GPU.')
print()
print(f'This sweep ({n_groups} groups, {n_groups * n_organs_per_group} rows): '
      f'~{per_group_seconds * n_groups / 60:.1f} min total, linearly extrapolated from this one group.')
print()
print('Caveat: volume shapes vary across subjects, so per-group time will vary -- run the')
print('full loop below and check the actual per-group timing it prints rather than trusting')
print('this single-group estimate.')
""")

    # Cell 22: full-run markdown
    set_source(cell(nb, 22), f"""## 10. Full run -- one VoxTell call per (subject, angle) group

**How results are saved:**

- **CSV** -- `results/predictions_voxtell_{dataset_name}_rotation_{{PERSON}}.csv`, one row
  per (subject, angle, organ). Columns:
  - `person`, `dataset`, `subject_id`, `rotation_angle_deg` -- which annotator/subject/angle
  - `split`, `organ`, `label_id`, `prompt_text` -- which case this row is (prompt is
    identical across all 7 angles for a given subject/organ, by design)
  - `image_path`, `label_path` -- repo-relative paths to the exact rotated volume/label
    used for this row (e.g. `faria/samples/{dataset_name}/volumes/<subject>_rot030_image.nii.gz`)
  - `volume_shape_ras`, `n_pred_voxels`, `dice`, `inference_seconds`
  - `gt_mask_rle_b64`, `pred_mask_rle_b64` -- **the actual segmentation masks**, run-length
    encoded + zlib-compressed + base64 text, decodable back to a 3D boolean array of shape
    `volume_shape_ras` (see the `decode_mask_rle()` helper above). This is the fix for "the
    CSV only gives a bbox" -- both the true and predicted mask shapes are fully recoverable
    from this CSV alone, no need to also share the results zip or the repo.
  - `pred_centroid_vox_0/1/2`, `pred_bbox_vox_{{0,1,2}}{{min,max}}`, `pred_side_axis0/1/2` --
    predicted mask's centroid/bbox/side (a convenience summary; the full mask is in
    `pred_mask_rle_b64` above), in RAS-reoriented voxel space
  - `gt_centroid_vox_0/1/2`, `gt_side_axis0/1/2` -- ground truth, copied straight from the
    manifest (`gt_side_axis*` is the TRUE anatomical side, invariant across angles by design)
  - Row is flushed to disk immediately after each organ finishes -- safe to stop anytime
- **Visualizations** -- `results/results_voxtell_{dataset_name}_rotation/<subject_id>_rot<angle>/<organ>.png`,
  one 3-panel figure (Original / Ground Truth / Prediction, Dice in the title) per row
- **Raw predicted masks** -- `results/masks_voxtell_{dataset_name}_rotation/<organ>/<subject_id>_rot<angle>.npz`
  (key `'segs'`) -- a redundant backup of the same mask already embedded in the CSV
""")

    # Cell 23: full run code
    set_source(cell(nb, 23), f"""def run_{dataset_name}_rotation():
    fig_root = RESULTS_ROOT / 'results_voxtell_{dataset_name}_rotation'
    fig_root.mkdir(parents=True, exist_ok=True)
    masks_root = RESULTS_ROOT / 'masks_voxtell_{dataset_name}_rotation'
    masks_root.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_ROOT / f'predictions_voxtell_{dataset_name}_rotation_{{PERSON}}.csv'

    fieldnames = ['person', 'dataset', 'subject_id', 'rotation_angle_deg', 'split', 'organ',
                  'label_id', 'prompt_text', 'image_path', 'label_path', 'volume_shape_ras',
                  'n_pred_voxels', 'dice', 'inference_seconds', 'gt_mask_rle_b64', 'pred_mask_rle_b64',
                  'pred_centroid_vox_0', 'pred_centroid_vox_1', 'pred_centroid_vox_2',
                  'pred_bbox_vox_0min', 'pred_bbox_vox_1min', 'pred_bbox_vox_2min',
                  'pred_bbox_vox_0max', 'pred_bbox_vox_1max', 'pred_bbox_vox_2max',
                  'pred_side_axis0', 'pred_side_axis1', 'pred_side_axis2',
                  'gt_centroid_vox_0', 'gt_centroid_vox_1', 'gt_centroid_vox_2',
                  'gt_side_axis0', 'gt_side_axis1', 'gt_side_axis2']

    already_done = set()
    write_header = not csv_path.exists()
    if csv_path.exists():
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                already_done.add((row['subject_id'], int(row['rotation_angle_deg']), row['organ']))
        print(f'Resuming: found {{len(already_done)}} already completed cases in {{csv_path.name}}')

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        for gi, ((subject_id, angle), rows) in enumerate(all_groups.items(), 1):
            organs = [r['organ'] for r in rows]
            pending = [r for r in rows if (subject_id, angle, r['organ']) not in already_done]
            if not pending:
                print(f'[{{gi}}/{{len(all_groups)}}] {{PERSON}}/{{subject_id}} @ {{angle}} deg: already completed, skipping')
                continue

            image_path = DATA_DIR / rows[0]['image_path']
            label_path = DATA_DIR / rows[0]['label_path']

            print(f'[{{gi}}/{{len(all_groups)}}] {{PERSON}}/{{subject_id}} @ {{angle}} deg: predicting {{len(organs)}} organs in one call...')
            t0 = time.time()
            img_arr, props = load_volume_ras(image_path)  # raw (1, X, Y, Z), what the model expects
            img_spatial = spatial_view(img_arr)  # (X, Y, Z) -- for slicing/plotting only, never fed to the model
            try:
                pred_all = predictor.predict_single_image(img_arr, organs)  # -> (n_organs, X, Y, Z), already binary 0/1
            except Exception as e:
                print(f'  !! prediction failed for {{PERSON}}/{{subject_id}} @ {{angle}}: {{e}}')
                continue
            elapsed_total = time.time() - t0
            print(f'  done in {{elapsed_total:.1f}}s ({{elapsed_total / len(organs):.2f}}s/organ average)')

            label_vol, _ = load_seg_ras(label_path)

            for oi, row0 in enumerate(rows):
                organ = row0['organ']
                if (subject_id, angle, organ) in already_done:
                    continue
                label_id = int(row0['label_id'])

                pred_mask = pred_all[oi].astype(bool)  # already binary 0/1 from predict_single_image, no thresholding needed
                gt_mask = label_vol == label_id
                dice = dice_score(pred_mask, gt_mask)
                n_voxels = int(pred_mask.sum())
                gt = mask_ground_truth_3d(pred_mask) if n_voxels > 0 else None

                out_row = {{
                    'person': PERSON, 'dataset': DATASET_NAME, 'subject_id': subject_id,
                    'rotation_angle_deg': angle, 'split': row0['split'], 'organ': organ,
                    'label_id': label_id, 'prompt_text': row0['prompt_text'],
                    'image_path': f'{{PERSON}}/samples/{{DATASET_NAME}}/{{row0["image_path"]}}',
                    'label_path': f'{{PERSON}}/samples/{{DATASET_NAME}}/{{row0["label_path"]}}',
                    'volume_shape_ras': 'x'.join(str(s) for s in gt_mask.shape),
                    'n_pred_voxels': n_voxels,
                    'dice': round(dice, 4), 'inference_seconds': round(elapsed_total / len(organs), 2),
                    'gt_mask_rle_b64': encode_mask_rle(gt_mask),
                    'pred_mask_rle_b64': encode_mask_rle(pred_mask),
                    'pred_centroid_vox_0': None, 'pred_centroid_vox_1': None, 'pred_centroid_vox_2': None,
                    'pred_bbox_vox_0min': None, 'pred_bbox_vox_1min': None, 'pred_bbox_vox_2min': None,
                    'pred_bbox_vox_0max': None, 'pred_bbox_vox_1max': None, 'pred_bbox_vox_2max': None,
                    'pred_side_axis0': None, 'pred_side_axis1': None, 'pred_side_axis2': None,
                    'gt_centroid_vox_0': row0['gt_centroid_vox_0'], 'gt_centroid_vox_1': row0['gt_centroid_vox_1'],
                    'gt_centroid_vox_2': row0['gt_centroid_vox_2'], 'gt_side_axis0': row0['gt_side_axis0'],
                    'gt_side_axis1': row0['gt_side_axis1'], 'gt_side_axis2': row0['gt_side_axis2'],
                }}
                if gt is not None:
                    (cx, cy, cz), bbox, (s0, s1, s2) = gt
                    out_row.update({{
                        'pred_centroid_vox_0': cx, 'pred_centroid_vox_1': cy, 'pred_centroid_vox_2': cz,
                        'pred_bbox_vox_0min': bbox[0], 'pred_bbox_vox_1min': bbox[1], 'pred_bbox_vox_2min': bbox[2],
                        'pred_bbox_vox_0max': bbox[3], 'pred_bbox_vox_1max': bbox[4], 'pred_bbox_vox_2max': bbox[5],
                        'pred_side_axis0': s0, 'pred_side_axis1': s1, 'pred_side_axis2': s2,
                    }})
                writer.writerow(out_row)
                f.flush()
                already_done.add((subject_id, angle, organ))
                print(f'    {{organ}}: dice={{dice:.3f}}, pred_voxels={{n_voxels}}')

                organ_mask_dir = masks_root / struct_slug(organ)
                organ_mask_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(organ_mask_dir / f'{{subject_id}}_rot{{angle:03d}}.npz', segs=pred_mask)

                # NOTE: slice_axis/take_slice use img_spatial (3D), NOT the raw 4D img_arr fed to
                # the model above -- np.argmin(img_arr.shape) would wrongly pick the channel axis (size 1).
                slice_axis = int(np.argmin(img_spatial.shape))
                gt_coords = np.nonzero(gt_mask)
                slice_idx = int(round(gt_coords[slice_axis].mean())) if len(gt_coords[0]) else img_spatial.shape[slice_axis] // 2
                def take_slice(vol, ax=slice_axis, idx=slice_idx):
                    sl = [slice(None)] * 3; sl[ax] = idx
                    return vol[tuple(sl)].T
                subj_fig_dir = fig_root / f'{{subject_id}}_rot{{angle:03d}}'
                subj_fig_dir.mkdir(parents=True, exist_ok=True)
                save_panel_figure(
                    subj_fig_dir / f'{{struct_slug(organ)}}.png',
                    row0['prompt_text'], dice,
                    window(take_slice(img_spatial)), take_slice(gt_mask), take_slice(pred_mask),
                )

    print(f'all done -> {{csv_path}}, {{fig_root}}')

run_{dataset_name}_rotation()
""")

    # Cell 24: Dice-vs-angle markdown (replaces the old cross-subject organ grid)
    set_source(cell(nb, 24), f"""## 11. Positional-bias analysis -- Dice vs. rotation angle

This is the actual question this sweep exists to answer: does Dice drop as the organ's
on-screen position changes (angle increases), even though the prompt and the true
anatomical side never change? A flat line means the model is following image content; a
falling line as angle moves away from 0/180 (or degrades on some organs but not others)
is positional bias.
""")

    # Cell 25: Dice-vs-angle code
    set_source(cell(nb, 25), f"""pred_df = pd.read_csv(RESULTS_ROOT / f'predictions_voxtell_{dataset_name}_rotation_{{PERSON}}.csv')

angle_summary = pred_df.groupby('rotation_angle_deg')['dice'].agg(['mean', 'std', 'count']).reset_index()
print(angle_summary)

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(angle_summary['rotation_angle_deg'], angle_summary['mean'], yerr=angle_summary['std'],
            marker='o', capsize=4, color='#2563eb')
ax.set_xlabel('Rotation angle (degrees)')
ax.set_ylabel('Mean Dice (all organs, all subjects/annotators)')
ax.set_title('VoxTell Dice vs. rotation angle -- {title}')
ax.set_xticks(ANGLES_DEG)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
fig.tight_layout()
dice_vs_angle_path = RESULTS_ROOT / 'dice_vs_angle_{dataset_name}.png'
fig.savefig(dice_vs_angle_path, dpi=150)
plt.close(fig)
print('wrote', dice_vs_angle_path)

# Per-organ breakdown: does the drift concentrate in specific (e.g. laterally-named) organs?
organ_angle = pred_df.groupby(['organ', 'rotation_angle_deg'])['dice'].mean().unstack('rotation_angle_deg')
organ_angle_path = RESULTS_ROOT / 'dice_by_organ_and_angle_{dataset_name}.csv'
organ_angle.to_csv(organ_angle_path)
print('wrote', organ_angle_path)
organ_angle.round(3)
""")

    # Cell 26: summary markdown -- unchanged text is fine, leave.

    # Cell 27: summary code
    set_source(cell(nb, 27), f"""import shutil
import pandas as pd

csv_path = RESULTS_ROOT / f'predictions_voxtell_{dataset_name}_rotation_{{PERSON}}.csv'
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f'=== {title} ROTATION SWEEP (VoxTell) Summary -- {{PERSON}} ===')
    print(f'Total evaluated cases: {{len(df)}}')
    print(f"Mean Dice: {{df['dice'].mean():.4f}}")
    print()
    print('Mean Dice by rotation angle:')
    print(df.groupby('rotation_angle_deg')['dice'].mean().round(4))
    print()
    print('Mean Dice by organ:')
    print(df.groupby('organ')['dice'].mean().sort_values(ascending=False).round(4))

zip_path = WORKING_DIR / f'voxtell_{dataset_name}_rotation_results_{{PERSON}}'
shutil.make_archive(str(zip_path), 'zip', str(RESULTS_ROOT))
print(f"\\nArchive created at {{zip_path}}.zip ({{os.path.getsize(str(zip_path) + '.zip') / 1e6:.1f}} MB)")
print("This zip file will appear directly in Kaggle's Output panel.")
""")

    # Cell 28: download instructions markdown
    set_source(cell(nb, 28), f"""## 13. Downloading the results

**You only need one file: `predictions_voxtell_{dataset_name}_rotation_<PERSON>.csv`.**
Every row already carries `gt_mask_rle_b64` and `pred_mask_rle_b64` (the actual
segmentation masks, compressed) plus `volume_shape_ras` to decode them -- that CSV alone is
enough to plot both the true and predicted masks later. Find it in the Kaggle **Output**
tab (after a **Save Version** run finishes) and download just that CSV.

The cell above also wrote `voxtell_{dataset_name}_rotation_results_<PERSON>.zip` (the CSV
again, plus `results_voxtell_{dataset_name}_rotation/` PNGs and
`masks_voxtell_{dataset_name}_rotation/` raw `.npz` masks) -- only grab this if you want the
pre-rendered figures too; it isn't needed just to hand back the CSV for plotting.

Note: Output only appears once a **Save Version** (Save & Run All) finishes, not from just
an interactive "Run All" in the editor session.
""")

    out_path = REPO_ROOT / out_name
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    build(
        "voxtell-amos-mirror-kaggle.ipynb", "amos22_mri", "right kidney",
        "AMOS22 MRI", "voxtell-amos-rotation-multiangle-kaggle.ipynb",
    )
    build(
        "voxtell-oasis-mirror-kaggle.ipynb", "neurite_oasis", "left hippocampus",
        "Neurite OASIS", "voxtell-oasis-rotation-multiangle-kaggle.ipynb",
    )
