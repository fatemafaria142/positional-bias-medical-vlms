"""
Generator for the two VoxTell model-internals investigation Kaggle notebooks
(voxtell-amos-groundedness-investigation-kaggle.ipynb,
voxtell-oasis-groundedness-investigation-kaggle.ipynb).

Implements every row of voxtell-model-internals-investigation-plan.md (rows 1-11:
image-text alignment, fusion, word-attribution) for a FIXED set of 3 subjects per
dataset, no rotation (rot000 only), in ONE notebook per dataset.

Starts from the existing voxtell-{amos,oasis}-rotation-multiangle-kaggle.ipynb
(cells 0-15: GPU check, working dirs, install voxtell, clone repo, load model,
fp16 text-encoder patch, helpers -- these don't change), then:
  - fixes cell 9 (clone step) back to the Kaggle-Secrets-only version -- the
    checked-in rotation notebooks currently have a hardcoded GitHub PAT
    (commit 9790045 re-added one after commit ee869d3 had removed it); this
    generator never re-introduces a literal token.
  - replaces the rotation-specific dataset-config/run-loop cells (16 onward)
    with the investigation pipeline.

Key design decisions, verified against the actual installed `voxtell` v0.1.0
source (voxtell/inference/predictor.py, voxtell/model/voxtell_model.py):
  - Rows 1, 3, 4 (the plan's "needs hook" rows) turn out to need NO
    forward_hook at all. `predict_sliding_window_return_logits(image,
    text_embeddings)` already returns the raw, continuous, spatial
    pre-sigmoid logits -- exactly "the internal representation right before
    the final mask" that rows 1/4 ask for -- and `text_embeddings` is a
    plain tensor argument from `embed_text_prompts()`, so the row-3
    embedding-swap causal test is just calling the API with a different
    (already-computed) embedding. No monkeypatching of voxtell's own code.
  - The per-subject cost is dominated by ONE sliding-window pass over the
    volume (shared across however many text prompts are batched together)
    plus a per-prompt decoder step -- so batching the word-ablation combos
    together with the baseline organ prompts into a SINGLE
    predict_sliding_window_return_logits call per subject is both correct
    and efficient (measured from the existing rotation CSVs: ~0.2s/prompt
    for AMOS-sized volumes, ~0.014s/prompt for OASIS-sized volumes).
  - Rows 1, 2, 3, 4 are computed as pure post-hoc analysis of the SAME
    per-subject baseline predictions + embeddings already produced for rows
    5-11 (the "all roles present" entry of the word-ablation batch, for each
    bilateral organ pair) -- no extra model calls needed beyond that one
    pass per subject.
  - Word attribution (rows 5-11) uses exact Shapley (row 5's "recommended
    default") over role-aware word slots (position/organ/the/in/this/image),
    not raw token deletion -- matching the plan's own supporting note that
    raw per-token ablation breaks grammar. The "organ" role's two states are
    {real organ name, generic placeholder "structure"} rather than
    {present, absent}, since the organ slot is grammatically required.

Usage:
    python scripts/build_groundedness_investigation_notebooks.py
"""

import copy
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BRANCH = "faria/voxtell3d-rotation-medical-segmentation"

# Fixed 3-subject scope per dataset -- reuses subjects that already have
# rot000 volumes committed to the repo (no new data staging needed).
AMOS_SUBJECTS = [("nipu", "amos_0508"), ("faria", "amos_0530"), ("hridoy", "amos_0555")]
OASIS_SUBJECTS = [
    ("faria", "OASIS_OAS1_0014_MR1"),
    ("hridoy", "OASIS_OAS1_0123_MR1"),
    ("nipu", "OASIS_OAS1_0062_MR1"),
]


def cell(nb, i):
    return nb["cells"][i]


def set_source(c, text):
    c["source"] = text.splitlines(keepends=True)
    c["outputs"] = c.get("outputs", [])
    if c["cell_type"] == "code":
        c["execution_count"] = None
        c["outputs"] = []


def new_cell(cell_type):
    c = {"cell_type": cell_type, "metadata": {}, "source": []}
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def add_cell(nb, cell_type, text):
    c = new_cell(cell_type)
    set_source(c, text)
    nb["cells"].append(c)


def build(template_name, dataset_name, title, subjects, out_name):
    nb = json.loads((REPO_ROOT / template_name).read_text(encoding="utf-8"))
    nb["cells"] = nb["cells"][:16]  # keep cells 0-15 only (install/model-load/helpers)

    # ---- Cell 0: title/description ----------------------------------------------------
    subj_list_str = ", ".join(f"{p}/{sid}" for p, sid in subjects)
    set_source(cell(nb, 0), f"""# VoxTell on {title} -- MODEL-INTERNALS / GROUNDING INVESTIGATION (Kaggle GPU)

Implements the full 11-row investigation from `voxtell-model-internals-investigation-plan.md`:
**is VoxTell's segmentation output actually driven by the position word ("left"/"right") in
the prompt, or by a shortcut/memorized prior (organ identity, image content, fixed pose)?**

- Image-Text Alignment (row 1), Fusion (rows 2-4), Word Attribution (rows 5-11).
- Fixed scope: **3 subjects, no rotation** (rot000 only) -- {subj_list_str}.
- Word attribution uses **exact Shapley** (row 5's recommended default) over role-aware word
  slots (position / organ / "the" / "in" / "this" / "image"), not raw token deletion.
- Rows 1, 3, 4 need **no `forward_hook`** -- verified directly against the installed
  `voxtell` v0.1.0 source. `predictor.predict_sliding_window_return_logits(image,
  text_embeddings)` already returns the raw spatial pre-sigmoid logits (exactly "the internal
  representation right before the final mask"), and `text_embeddings` is a plain tensor
  argument from `predictor.embed_text_prompts()` -- so the row-3 causal embedding-swap test is
  just calling the public API with a different embedding, no monkeypatching of voxtell's code.
- Rows 1, 2, 3, 4 are derived as post-hoc analysis of the SAME per-subject baseline
  predictions + embeddings already produced for rows 5-11 -- no extra model calls needed.

This notebook is self-contained (unlike the rotation-sweep notebooks, it doesn't loop over a
`PERSON` variable) -- the 3 subjects above span different annotators' folders directly.
""")

    # ---- Cell 9: clone step -- restore Kaggle-Secrets-only version (no hardcoded token) --
    clone_cell = cell(nb, 9)
    set_source(clone_cell, f"""import os
import shutil
import subprocess

from kaggle_secrets import UserSecretsClient

_gh_token = UserSecretsClient().get_secret("GITHUB_PAT")
assert _gh_token, "Add a Kaggle Secret named GITHUB_PAT (Add-ons -> Secrets) before running this cell."

_clone_url = f'https://{{_gh_token}}@github.com/fatemafaria142/positional-bias-medical-vlms.git'
_dest = '/kaggle/working/research_repo'
shutil.rmtree(_dest, ignore_errors=True)

_result = subprocess.run(
    ['git', 'clone', '--branch', '{BRANCH}', '--depth', '1', _clone_url, _dest],
    capture_output=True, text=True,
)
_stderr_scrubbed = _result.stderr.replace(_gh_token, '***')
del _gh_token, _clone_url

if _result.returncode != 0:
    raise RuntimeError(f'git clone failed (exit {{_result.returncode}}):\\n{{_stderr_scrubbed}}')

RESEARCH_REPO = Path(_dest)
print('cloned into', RESEARCH_REPO)
print('repo root:', os.listdir(RESEARCH_REPO))
print('annotator folders present:', [p for p in ('faria', 'nipu', 'hridoy') if (RESEARCH_REPO / p).is_dir()])
""")

    # ---- Cell 15: trim the rotation-notebook's panel-figure helpers (not used here), ----
    # ---- keep only what this notebook needs, and add the RLE helpers. -------------------
    set_source(cell(nb, 15), """import csv, time
from pathlib import Path

import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

RESEARCH_REPO = Path(RESEARCH_REPO)
reader = NibabelIOWithReorient()

_OPPOSITE = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}

def load_volume_ras(path):
    \"\"\"Load a NIfTI image via VoxTell's own reader so predicted masks and ground
    truth end up in the exact same RAS-reoriented array space. Returns (1, X, Y, Z),
    the channel-first shape voxtell.inference.predictor expects.\"\"\"
    data, props = reader.read_images([str(path)])
    return np.asarray(data), props

def spatial_view(arr_4d):
    a = np.asarray(arr_4d)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    return a

def load_seg_ras(path):
    data, props = reader.read_seg(str(path))
    return spatial_view(data).astype(np.int32), props

def mask_centroid_side(mask: np.ndarray):
    \"\"\"Centroid + bbox + anatomical side in RAS-reoriented voxel space. axis0/1/2 are
    always R/A/S respectively once reoriented (axis0 increases toward Right). Returns
    (centroid, bbox, sides) where bbox = (min0,min1,min2,max0,max1,max2).\"\"\"
    coords = np.nonzero(mask)
    if len(coords[0]) == 0:
        return None
    centroid = tuple(float(c.mean()) for c in coords)
    bbox = tuple(int(c.min()) for c in coords) + tuple(int(c.max()) for c in coords)
    axcodes = ("R", "A", "S")
    sides = tuple(code if centroid_val > mask.shape[axis] / 2 else _OPPOSITE[code]
                  for axis, (code, centroid_val) in enumerate(zip(axcodes, centroid)))
    return centroid, bbox, sides

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2.0 * inter / denom)

def iou_score(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(inter / union)

def struct_slug(name: str) -> str:
    return name.strip().lower().replace(' ', '_').replace('/', '-')

def load_manifest_rot000(person, dataset_name):
    \"\"\"Only rot000 rows -- this investigation doesn't use the rotation sweep.\"\"\"
    path = RESEARCH_REPO / person / 'samples' / dataset_name / f'manifest_{dataset_name}.csv'
    with open(path, newline='') as f:
        return [row for row in csv.DictReader(f) if int(row['rotation_angle_deg']) == 0]

print('helpers ready')


import base64
import zlib


def encode_mask_rle(mask: np.ndarray) -> str:
    \"\"\"Run-length-encode a boolean mask (C-order flatten), zlib-compress, base64-encode.\"\"\"
    flat = np.asarray(mask, dtype=bool).ravel(order='C')
    change_idx = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], change_idx, [flat.size]))
    runs = np.diff(boundaries).astype(np.uint32)
    if flat.size and flat[0]:
        runs = np.concatenate(([0], runs)).astype(np.uint32)
    raw = zlib.compress(runs.tobytes(), level=9)
    return base64.b64encode(raw).decode('ascii')


def decode_mask_rle(encoded: str, shape) -> np.ndarray:
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
""")

    # ---- Install LIME + SHAP (row 6/7 of the plan) -- pure-Python, no torch/CUDA pins ---
    # involved, so this is safe to install after the careful torch+voxtell reinstall above
    # without risking the ABI-matched torch/torchvision pair that cell fixed.
    add_cell(nb, "markdown", """## 6b. Install LIME + SHAP (rows 6-7 -- required for this study's paper comparison)

Both are pure-Python (numpy/scikit-learn/scikit-image for LIME; numpy/pandas/numba for SHAP)
-- no torch/CUDA version pins involved, safe to install after the careful torch+voxtell
reinstall above without disturbing it.

`shap` pulls in `numba`, which can lag behind whatever NumPy is already installed in the
Kaggle image (observed failure: `"Numba needs NumPy 2.0 or less. Got NumPy 2.x"` at `import
shap` time) -- fixed by upgrading `numba` specifically (NOT `numpy`, which is already loaded
and used by torch/voxtell elsewhere in this kernel; swapping it now would hit the exact same
"can't safely reload a loaded C extension" problem the very first install cell's comment block
describes for torch). If `shap` still can't be imported after that, this cell degrades
gracefully -- `SHAP_AVAILABLE=False` -- so the rest of the notebook (rows 1-6, 8-11) still
runs; only row 7 depends on it.
""")
    add_cell(nb, "code", """import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lime', 'shap'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'numba'], check=True)

try:
    import lime
    import lime.lime_text
    LIME_AVAILABLE = True
    print('lime', lime.__version__, 'installed')
except ImportError as e:
    LIME_AVAILABLE = False
    print(f'!! lime failed to import: {e}')
    print('!! continuing WITHOUT row 6 (LIME) -- rows 1-5, 7-11 are unaffected.')

try:
    import shap
    SHAP_AVAILABLE = True
    print('shap', shap.__version__, 'installed')
except ImportError as e:
    SHAP_AVAILABLE = False
    print(f'!! shap failed to import even after upgrading numba: {e}')
    print('!! continuing WITHOUT row 7 (SHAP) -- rows 1-6, 8-11 (including exact Shapley, row 5) are unaffected.')
    print('!! to retry: check the exact numba/numpy versions with `pip show numba numpy`, or try a Kaggle')
    print('!!   session restart (Run All from a fresh kernel) in case a stale numba wheel got cached.')
""")

    # ---- Cell 16/17: dataset config (fixed subjects, no PERSON toggle) ------------------
    subjects_py = ",\n    ".join(f'("{p}", "{sid}")' for p, sid in subjects)
    add_cell(nb, "markdown", f"""## 7. Dataset config -- {title}, fixed 3-subject ablation study (no rotation)

No `PERSON` toggle here (unlike the rotation notebooks) -- this investigation is a fixed
scope of 3 subjects spanning different annotators' folders, read directly.
""")
    add_cell(nb, "code", f"""DATASET_NAME = '{dataset_name}'

# vvv fixed scope for this investigation -- (person, subject_id) vvv
SUBJECTS = [
    {subjects_py},
]
# ^^^ change here to swap subjects; no rotation is used (rot000 only) ^^^

subject_organs = {{}}  # subject_id -> {{'person':..., 'rows': [manifest rows for this subject]}}
for person, sid in SUBJECTS:
    rows = [r for r in load_manifest_rot000(person, DATASET_NAME) if r['subject_id'] == sid]
    assert rows, f'no rot000 rows found for {{person}}/{{sid}} in {{DATASET_NAME}}'
    subject_organs[sid] = {{'person': person, 'rows': rows}}

n_organs = len(next(iter(subject_organs.values()))['rows'])
print(f'{{DATASET_NAME}}: {{len(subject_organs)}} subjects {{list(subject_organs)}}, {{n_organs}} organs/subject')
for sid, info in subject_organs.items():
    print(f"  {{info['person']}}/{{sid}}: organs =", [r['organ'] for r in info['rows']])
""")

    # ---- Cell 18/19: role-aware ablation + exact Shapley helpers ------------------------
    add_cell(nb, "markdown", """## 8. Word-ablation roles + exact Shapley (rows 5-11 method)

Role-aware templated ablation (not raw token deletion -- see the plan's supporting notes:
dropping "kidney" from "Segment the left kidney in this image." is ungrammatical). Each
prompt's words collapse into role SLOTS: `position` (only for bilateral organs, e.g.
"left"/"right"), `organ` (the real name, or a generic placeholder "structure" when
"removed"), and the function words `the`/`in`/`this`/`image`. Exact Shapley (row 5's
recommended default) is computed over EVERY 2^n present/absent combination of these role
slots -- 64 combos for a bilateral organ (6 roles), 32 for a non-bilateral organ (5 roles,
no position slot).
""")
    add_cell(nb, "code", """import itertools
from math import factorial

POSITION_WORDS = {'left', 'right'}
GENERIC_ORGAN_WORD = 'structure'

def parse_organ_roles(organ_name):
    \"\"\"'left kidney' -> ('left', 'kidney'); 'spleen' -> (None, 'spleen').\"\"\"
    parts = organ_name.split()
    if parts[0].lower() in POSITION_WORDS:
        return parts[0].lower(), ' '.join(parts[1:])
    return None, organ_name

def role_names_for(organ_name):
    position, _ = parse_organ_roles(organ_name)
    return (['position'] if position else []) + ['organ', 'the', 'in', 'this', 'image']

def build_prompt(organ_name, present):
    \"\"\"present: dict[role] -> bool. role 'organ': True = real name, False = generic
    placeholder (the organ slot is grammatically required, so it's never truly absent --
    see the plan's supporting note on role-aware ablation).\"\"\"
    position, organ_core = parse_organ_roles(organ_name)
    words = ['Segment']
    if present.get('the', True):
        words.append('the')
    if position is not None and present.get('position', True):
        words.append(position)
    words.append(organ_core if present.get('organ', True) else GENERIC_ORGAN_WORD)
    if present.get('in', True):
        words.append('in')
    if present.get('this', True):
        words.append('this')
    if present.get('image', True):
        words.append('image')
    sentence = ' '.join(words)
    return sentence[0].upper() + sentence[1:] + '.'

def all_role_combos(roles):
    \"\"\"All 2^n present/absent combinations. Returns list of (frozenset(present roles), present_dict).\"\"\"
    combos = []
    for bits in itertools.product([True, False], repeat=len(roles)):
        present = dict(zip(roles, bits))
        key = frozenset(r for r in roles if present[r])
        combos.append((key, present))
    return combos

def shapley_values(v, roles):
    \"\"\"Exact Shapley values. v: dict[frozenset(present roles)] -> payoff (float).
    roles: the full player set N (all role names for this organ).\"\"\"
    n = len(roles)
    phi = {r: 0.0 for r in roles}
    for i in roles:
        others = [r for r in roles if r != i]
        for bits in itertools.product([True, False], repeat=len(others)):
            S = frozenset(r for r, b in zip(others, bits) if b)
            s = len(S)
            weight = factorial(s) * factorial(n - s - 1) / factorial(n)
            phi[i] += weight * (v[frozenset(S | {i})] - v[S])
    return phi

# Sanity check: the "all roles present" prompt must exactly match the manifest's real prompt_text.
_check_row = next(iter(subject_organs.values()))['rows'][0]
_roles = role_names_for(_check_row['organ'])
_built = build_prompt(_check_row['organ'], {r: True for r in _roles})
assert _built == _check_row['prompt_text'], f'role-template mismatch: {_built!r} != {_check_row["prompt_text"]!r}'
print('role-template sanity check passed:', _built)
print(f'bilateral organ example: {role_names_for("left kidney" if DATASET_NAME == "amos22_mri" else "left hippocampus")} '
      f'({2**len(role_names_for("left kidney" if DATASET_NAME == "amos22_mri" else "left hippocampus"))} combos)')
""")

    # ---- Cell 20/21: smoke test on one subject -------------------------------------------
    add_cell(nb, "markdown", """## 9. Smoke test -- one subject, full combined prompt list (all organs x all ablation combos)

One `embed_text_prompts` + one `predict_sliding_window_return_logits` call per subject
covers every organ's baseline prompt AND every one of its Shapley ablation combos in a
single sliding-window pass over the volume (confirmed from `voxtell_model.py`: the image
encoder runs once regardless of prompt count). Time this before committing to the full run.
""")
    add_cell(nb, "code", """def build_subject_prompt_records(rows):
    \"\"\"rows: this subject's manifest rows (one per organ). Returns a flat list of
    dicts, one per (organ, role-combo) -- the baseline ("all roles present") entry is
    included as just another combo, keyed by the full role frozenset.\"\"\"
    records = []
    for row in rows:
        organ = row['organ']
        roles = role_names_for(organ)
        for key, present in all_role_combos(roles):
            records.append({
                'organ': organ, 'label_id': int(row['label_id']), 'roles': roles,
                'subset': key, 'prompt': build_prompt(organ, present),
                'gt_side_axis0': row['gt_side_axis0'],
            })
    return records

smoke_sid = next(iter(subject_organs))
smoke_rows = subject_organs[smoke_sid]['rows']
smoke_records = build_subject_prompt_records(smoke_rows)
smoke_prompts = [r['prompt'] for r in smoke_records]
print(f'smoke test: {subject_organs[smoke_sid]["person"]}/{smoke_sid} -- '
      f'{len(smoke_rows)} organs -> {len(smoke_prompts)} total prompts (baseline + Shapley combos)')

smoke_image_path = RESEARCH_REPO / subject_organs[smoke_sid]['person'] / 'samples' / DATASET_NAME / smoke_rows[0]['image_path']

t0 = time.time()
smoke_img, _ = load_volume_ras(smoke_image_path)
smoke_data, smoke_bbox, smoke_orig_shape = predictor.preprocess(smoke_img)
smoke_embeddings = predictor.embed_text_prompts(smoke_prompts)
smoke_logits = predictor.predict_sliding_window_return_logits(smoke_data, smoke_embeddings).to('cpu')
smoke_elapsed = time.time() - t0

print(f'smoke test elapsed: {smoke_elapsed:.1f}s for {len(smoke_prompts)} prompts '
      f'({smoke_elapsed / len(smoke_prompts):.3f}s/prompt if it scaled linearly)')
print('logits shape:', tuple(smoke_logits.shape))

n_subjects = len(subject_organs)
print(f'\\nExtrapolated: ~{smoke_elapsed * n_subjects / 60:.1f} min for all {n_subjects} subjects '
      '(rows 5-11 word-attribution pass). Rows 1-4 reuse this same data, no extra model calls.')
""")

    # ---- Cell 22/23: full run -- word attribution (rows 5-11) ---------------------------
    add_cell(nb, "markdown", f"""## 10. Full run -- rows 5-11 (word attribution) for every subject x organ

For each subject: one combined `embed_text_prompts` + `predict_sliding_window_return_logits`
call over every organ's full 2^n role-combo list, thresholded (sigmoid > 0.5) and
crop-reverted exactly like `predict_single_image` does internally. Per organ, per combo,
computes all four target metrics from the plan:
  - **row 8** `f_dice_gt` -- Dice vs. ground truth
  - **row 9** `f_dice_self` -- Dice vs. this organ's own baseline (all-roles-present) prediction
  - **row 10** `f_centroid_shift` -- signed shift along the R/L axis (axis 0) vs. baseline centroid
  - **row 11** `f_volume` -- predicted voxel count vs. baseline voxel count

then computes exact Shapley values (row 5) for every role, for each of the four metrics.
Also keeps each organ's baseline (all-roles-present) mask + embedding in memory --
`subject_results` -- for rows 1-4 in the next section (no extra model calls needed there).

Results: `word_attribution_{dataset_name}.csv` (one row per subject x organ x role x metric).
""")
    add_cell(nb, "code", f"""subject_results = {{}}  # subject_id -> {{'person', organ -> {{'baseline_mask','baseline_embed','gt_mask','label_id','centroid','side'}}}}

wa_fieldnames = ['dataset', 'person', 'subject_id', 'organ', 'is_bilateral', 'n_roles', 'n_combos',
                  'role', 'metric', 'shapley_value']
wa_csv_path = RESULTS_ROOT / f'word_attribution_{dataset_name}.csv'
wa_rows_out = []

# Row 7: real `shap` library output (KernelExplainer over the exact v(S) lookup table) --
# same shape as word_attribution.csv, plus a per-row check against the hand-rolled exact value.
shap_fieldnames = ['dataset', 'person', 'subject_id', 'organ', 'is_bilateral', 'n_roles', 'n_combos',
                    'role', 'metric', 'shap_value', 'matches_exact_shapley']
shap_csv_path = RESULTS_ROOT / f'shap_attribution_{dataset_name}.csv'
shap_rows_out = []

# Baseline (all-roles-present, i.e. the REAL prompt) predictions -- the actual segmentation
# masks + bbox/centroid/side, downloadable and plottable entirely locally afterwards (no
# GPU/torch needed -- see scripts/plot_groundedness_results.py, which only needs numpy/
# pandas/matplotlib and the decode_mask_rle() helper also defined in this notebook).
bp_fieldnames = ['dataset', 'person', 'subject_id', 'organ', 'label_id', 'is_bilateral', 'split',
                  'prompt_text', 'image_path', 'label_path', 'volume_shape_ras',
                  'n_pred_voxels', 'dice_gt', 'gt_mask_rle_b64', 'pred_mask_rle_b64',
                  'pred_centroid_vox_0', 'pred_centroid_vox_1', 'pred_centroid_vox_2',
                  'pred_bbox_vox_0min', 'pred_bbox_vox_1min', 'pred_bbox_vox_2min',
                  'pred_bbox_vox_0max', 'pred_bbox_vox_1max', 'pred_bbox_vox_2max',
                  'pred_side_axis0', 'pred_side_axis1', 'pred_side_axis2',
                  'gt_centroid_vox_0', 'gt_centroid_vox_1', 'gt_centroid_vox_2',
                  'gt_bbox_vox_0min', 'gt_bbox_vox_1min', 'gt_bbox_vox_2min',
                  'gt_bbox_vox_0max', 'gt_bbox_vox_1max', 'gt_bbox_vox_2max',
                  'gt_side_axis0', 'gt_side_axis1', 'gt_side_axis2']
bp_csv_path = RESULTS_ROOT / f'baseline_predictions_{dataset_name}.csv'
bp_rows_out = []

# EVERY ablation combo's mask -- no experiment left out. For a bilateral organ this is
# 2^6=64 rows (combo), for a non-bilateral organ 2^5=32 rows -- written incrementally
# (flushed per row) since this file gets large (thousands of masks across 3 subjects x
# every organ x every combo). 'is_baseline' marks the all-roles-present (real prompt) row,
# which is the same mask as in baseline_predictions_{dataset_name}.csv.
am_fieldnames = ['dataset', 'person', 'subject_id', 'organ', 'label_id', 'image_path', 'label_path',
                  'is_bilateral', 'is_baseline', 'roles_present', 'roles_dropped', 'prompt_text',
                  'volume_shape_ras', 'n_pred_voxels', 'f_dice_gt', 'f_dice_self', 'f_centroid_shift',
                  'f_volume', 'mask_rle_b64']
am_csv_path = RESULTS_ROOT / f'ablation_masks_{dataset_name}.csv'
am_f = open(am_csv_path, 'w', newline='')
am_writer = csv.DictWriter(am_f, fieldnames=am_fieldnames)
am_writer.writeheader()
am_n_written = 0

for si, (sid, info) in enumerate(subject_organs.items(), 1):
    person, rows = info['person'], info['rows']
    manifest_by_organ = {{r['organ']: r for r in rows}}
    print(f'[{{si}}/{{len(subject_organs)}}] {{person}}/{{sid}}: building prompt list...')
    records = build_subject_prompt_records(rows)
    prompts = [r['prompt'] for r in records]

    image_path = RESEARCH_REPO / person / 'samples' / DATASET_NAME / rows[0]['image_path']
    label_path = RESEARCH_REPO / person / 'samples' / DATASET_NAME / rows[0]['label_path']
    img_arr, _ = load_volume_ras(image_path)
    label_vol, _ = load_seg_ras(label_path)

    t0 = time.time()
    data, bbox, orig_shape = predictor.preprocess(img_arr)
    embeddings = predictor.embed_text_prompts(prompts)  # (1, n_prompts, dim)
    logits = predictor.predict_sliding_window_return_logits(data, embeddings).to('cpu')
    with torch.no_grad():
        masks_cropped = (torch.sigmoid(logits.float()) > 0.5).numpy().astype(np.uint8)
    masks = np.zeros((masks_cropped.shape[0], *orig_shape), dtype=np.uint8)
    masks = insert_crop_into_image(masks, masks_cropped, bbox)
    elapsed = time.time() - t0
    print(f'  {{len(prompts)}} prompts in {{elapsed:.1f}}s ({{elapsed / len(prompts):.3f}}s/prompt)')

    # index records by organ -> subset -> (record, mask index)
    by_organ = {{}}
    for idx, rec in enumerate(records):
        by_organ.setdefault(rec['organ'], []).append((rec, idx))

    subject_results[sid] = {{'person': person, 'organs': {{}}}}
    volume_shape_str = None

    for organ, entries in by_organ.items():
        label_id = entries[0][0]['label_id']
        roles = entries[0][0]['roles']
        full_key = frozenset(roles)
        gt_mask = (label_vol == label_id)
        manifest_row = manifest_by_organ[organ]
        volume_shape_str = 'x'.join(str(s) for s in gt_mask.shape)

        mask_by_subset = {{rec['subset']: masks[idx].astype(bool) for rec, idx in entries}}
        prompt_by_subset = {{rec['subset']: rec['prompt'] for rec, idx in entries}}
        baseline_mask = mask_by_subset[full_key]
        baseline_n_voxels = int(baseline_mask.sum())
        baseline_cs = mask_centroid_side(baseline_mask)
        baseline_centroid = baseline_cs[0] if baseline_cs else None

        baseline_idx = next(idx for rec, idx in entries if rec['subset'] == full_key)
        baseline_embed = embeddings[0, baseline_idx].detach().cpu().numpy()

        subject_results[sid]['organs'][organ] = {{
            'label_id': label_id, 'baseline_mask': baseline_mask, 'baseline_embed': baseline_embed,
            'gt_mask': gt_mask, 'gt_side_axis0': entries[0][0]['gt_side_axis0'],
            'baseline_centroid': baseline_centroid,
        }}

        # v(S) per metric, for every 2^n subset -- ALSO write out that subset's actual mask
        # (ablation_masks CSV) so no ablation-combo experiment is summary-only.
        v_dice_gt, v_dice_self, v_centroid_shift, v_volume = {{}}, {{}}, {{}}, {{}}
        for subset, mask in mask_by_subset.items():
            dice_gt_val = dice_score(mask, gt_mask)
            dice_self_val = dice_score(mask, baseline_mask)
            n_vox = int(mask.sum())
            volume_val = float(n_vox - baseline_n_voxels)
            cs = mask_centroid_side(mask)
            centroid_shift_val = float(cs[0][0] - baseline_centroid[0]) if (cs and baseline_centroid is not None) else 0.0

            v_dice_gt[subset] = dice_gt_val
            v_dice_self[subset] = dice_self_val
            v_volume[subset] = volume_val
            v_centroid_shift[subset] = centroid_shift_val

            am_writer.writerow({{
                'dataset': DATASET_NAME, 'person': person, 'subject_id': sid, 'organ': organ,
                'label_id': label_id,
                'image_path': f'{{person}}/samples/{{DATASET_NAME}}/{{manifest_row["image_path"]}}',
                'label_path': f'{{person}}/samples/{{DATASET_NAME}}/{{manifest_row["label_path"]}}',
                'is_bilateral': 'position' in roles, 'is_baseline': subset == full_key,
                'roles_present': ','.join(sorted(subset)), 'roles_dropped': ','.join(sorted(full_key - subset)),
                'prompt_text': prompt_by_subset[subset], 'volume_shape_ras': volume_shape_str,
                'n_pred_voxels': n_vox, 'f_dice_gt': round(dice_gt_val, 4), 'f_dice_self': round(dice_self_val, 4),
                'f_centroid_shift': round(centroid_shift_val, 2), 'f_volume': round(volume_val, 1),
                'mask_rle_b64': encode_mask_rle(mask),
            }})
            am_n_written += 1
        am_f.flush()

        roles_sorted = sorted(roles)  # fixed order for the SHAP feature matrix
        for metric_name, v in [('f_dice_gt', v_dice_gt), ('f_dice_self', v_dice_self),
                                 ('f_centroid_shift', v_centroid_shift), ('f_volume', v_volume)]:
            exact_shap_vals = shapley_values(v, roles)
            for role, val in exact_shap_vals.items():
                wa_rows_out.append({{
                    'dataset': DATASET_NAME, 'person': person, 'subject_id': sid, 'organ': organ,
                    'is_bilateral': 'position' in roles, 'n_roles': len(roles), 'n_combos': 2 ** len(roles),
                    'role': role, 'metric': metric_name, 'shapley_value': round(val, 6),
                }})

            # Row 7: real `shap` library (KernelExplainer) over the SAME v(S) lookup table --
            # with this few features (n<=6) shap internally enumerates the full 2^n space, so
            # its values should match exact_shap_vals above (printed as a sanity check in the
            # summary section) -- this is what satisfies "must use SHAP" for the paper comparison.
            # Skipped gracefully (SHAP_AVAILABLE=False) if the shap import failed earlier --
            # rows 1-6, 8-11 (including exact Shapley, row 5) are unaffected either way.
            if SHAP_AVAILABLE:
                def _shap_value_fn(X, _v=v, _roles=roles_sorted):
                    return np.array([_v[frozenset(r for r, b in zip(_roles, row) if b)] for row in X])
                _background = np.zeros((1, len(roles_sorted)))
                _explainer = shap.KernelExplainer(_shap_value_fn, _background)
                _shap_out = _explainer.shap_values(np.ones((1, len(roles_sorted))), nsamples=2 ** len(roles_sorted), silent=True)
                _shap_out = np.asarray(_shap_out).reshape(-1)
                for role, val in zip(roles_sorted, _shap_out):
                    shap_rows_out.append({{
                        'dataset': DATASET_NAME, 'person': person, 'subject_id': sid, 'organ': organ,
                        'is_bilateral': 'position' in roles, 'n_roles': len(roles), 'n_combos': 2 ** len(roles),
                        'role': role, 'metric': metric_name, 'shap_value': round(float(val), 6),
                        'matches_exact_shapley': bool(abs(float(val) - exact_shap_vals[role]) < 1e-4),
                    }})

        bp_row = {{
            'dataset': DATASET_NAME, 'person': person, 'subject_id': sid, 'organ': organ,
            'label_id': label_id, 'is_bilateral': 'position' in roles, 'split': manifest_row['split'],
            'prompt_text': manifest_row['prompt_text'],
            'image_path': f'{{person}}/samples/{{DATASET_NAME}}/{{manifest_row["image_path"]}}',
            'label_path': f'{{person}}/samples/{{DATASET_NAME}}/{{manifest_row["label_path"]}}',
            'volume_shape_ras': 'x'.join(str(s) for s in gt_mask.shape),
            'n_pred_voxels': baseline_n_voxels, 'dice_gt': round(v_dice_gt[full_key], 4),
            'gt_mask_rle_b64': encode_mask_rle(gt_mask), 'pred_mask_rle_b64': encode_mask_rle(baseline_mask),
            'pred_centroid_vox_0': None, 'pred_centroid_vox_1': None, 'pred_centroid_vox_2': None,
            'pred_bbox_vox_0min': None, 'pred_bbox_vox_1min': None, 'pred_bbox_vox_2min': None,
            'pred_bbox_vox_0max': None, 'pred_bbox_vox_1max': None, 'pred_bbox_vox_2max': None,
            'pred_side_axis0': None, 'pred_side_axis1': None, 'pred_side_axis2': None,
            'gt_centroid_vox_0': manifest_row['gt_centroid_vox_0'], 'gt_centroid_vox_1': manifest_row['gt_centroid_vox_1'],
            'gt_centroid_vox_2': manifest_row['gt_centroid_vox_2'],
            'gt_bbox_vox_0min': manifest_row['gt_bbox_vox_0min'], 'gt_bbox_vox_1min': manifest_row['gt_bbox_vox_1min'],
            'gt_bbox_vox_2min': manifest_row['gt_bbox_vox_2min'], 'gt_bbox_vox_0max': manifest_row['gt_bbox_vox_0max'],
            'gt_bbox_vox_1max': manifest_row['gt_bbox_vox_1max'], 'gt_bbox_vox_2max': manifest_row['gt_bbox_vox_2max'],
            'gt_side_axis0': manifest_row['gt_side_axis0'], 'gt_side_axis1': manifest_row['gt_side_axis1'],
            'gt_side_axis2': manifest_row['gt_side_axis2'],
        }}
        if baseline_cs is not None:
            (cx, cy, cz), (bx0, by0, bz0, bx1, by1, bz1), (s0, s1, s2) = baseline_cs
            bp_row.update({{
                'pred_centroid_vox_0': cx, 'pred_centroid_vox_1': cy, 'pred_centroid_vox_2': cz,
                'pred_bbox_vox_0min': bx0, 'pred_bbox_vox_1min': by0, 'pred_bbox_vox_2min': bz0,
                'pred_bbox_vox_0max': bx1, 'pred_bbox_vox_1max': by1, 'pred_bbox_vox_2max': bz1,
                'pred_side_axis0': s0, 'pred_side_axis1': s1, 'pred_side_axis2': s2,
            }})
        bp_rows_out.append(bp_row)

with open(wa_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=wa_fieldnames)
    writer.writeheader()
    writer.writerows(wa_rows_out)
print(f'wrote {{wa_csv_path}} ({{len(wa_rows_out)}} rows)')

with open(shap_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=shap_fieldnames)
    writer.writeheader()
    writer.writerows(shap_rows_out)
n_mismatch = sum(1 for r in shap_rows_out if not r['matches_exact_shapley'])
print(f'wrote {{shap_csv_path}} ({{len(shap_rows_out)}} rows) -- {{n_mismatch}} rows differ from the exact '
      'Shapley value by > 1e-4 (expect 0 -- SHAP internally enumerates the full space at this feature count)')

with open(bp_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=bp_fieldnames)
    writer.writeheader()
    writer.writerows(bp_rows_out)
print(f'wrote {{bp_csv_path}} ({{len(bp_rows_out)}} rows) -- includes gt_mask_rle_b64/pred_mask_rle_b64, '
      'bbox, centroid, side for every subject x organ -- this is the file to download for local plotting.')

am_f.close()
print(f'wrote {{am_csv_path}} ({{am_n_written}} rows) -- EVERY ablation combo\\'s mask (mask_rle_b64), '
      'not just the baseline -- no experiment left out.')
""")

    # ---- Cell 24/25: rows 1-4, hook-free, reusing subject_results -----------------------
    add_cell(nb, "markdown", """## 11. Rows 1-4 -- image-text alignment & fusion (hook-free, no extra model calls)

Derived entirely from `subject_results` (the baseline masks + embeddings already produced
above) for every bilateral organ pair (left/right) in each subject -- no `forward_hook`, no
extra `predict_*` calls.

- **Row 1** (matched vs. mismatched spatial overlap): IoU of the LEFT-prompt's predicted
  mask against the true left-side location (matched) vs. IoU of the RIGHT-prompt's predicted
  mask (run on the same image) against that SAME true left-side location (mismatched). High
  overlap in both cases would mean the image alone drives the activation regardless of the
  word asked.
- **Row 2** (embedding-distance vs. output-distance regression): cosine distance between the
  left/right text embeddings vs. Dice/centroid distance between their predicted masks, fit
  across every bilateral pair x subject.
- **Row 3** (causal embedding test): since `text_embeddings` is a plain tensor argument to
  `predict_sliding_window_return_logits`, "does the left-prompt embedding causally produce
  the left-side mask" is already answered by comparing the LEFT-prompt baseline prediction's
  side against ground truth, and the RIGHT-prompt baseline prediction's side against ground
  truth, on the same image -- no separate substitution call needed, since a "substituted"
  call is computationally identical to just calling with that embedding directly.
- **Row 4** (spatial diff map): XOR between the left-prompt and right-prompt predicted masks
  -- where (if anywhere) they diverge.
""")
    add_cell(nb, "code", f"""def bilateral_pairs(organs):
    \"\"\"organ names -> list of (left_name, right_name) pairs sharing the same core noun.\"\"\"
    by_core = {{}}
    for o in organs:
        position, core = parse_organ_roles(o)
        if position:
            by_core.setdefault(core, {{}})[position] = o
    return [(v['left'], v['right']) for v in by_core.values() if 'left' in v and 'right' in v]

row1_rows, row2_rows, row3_rows, row4_rows = [], [], [], []

for sid, res in subject_results.items():
    person = res['person']
    organs = list(res['organs'])
    for left_name, right_name in bilateral_pairs(organs):
        L, R = res['organs'][left_name], res['organs'][right_name]

        # Row 1: matched vs. mismatched IoU against the TRUE left-side location
        gt_left = L['gt_mask']
        iou_matched = iou_score(L['baseline_mask'], gt_left)      # left prompt vs. left GT
        iou_mismatched = iou_score(R['baseline_mask'], gt_left)   # right prompt vs. left GT
        row1_rows.append({{'dataset': DATASET_NAME, 'person': person, 'subject_id': sid,
                            'organ_pair': f'{{left_name}}/{{right_name}}',
                            'iou_matched': round(iou_matched, 4), 'iou_mismatched': round(iou_mismatched, 4),
                            'genuine_grounding': iou_matched > iou_mismatched}})

        # Row 2: embedding cosine distance vs. output distance
        e_l, e_r = L['baseline_embed'], R['baseline_embed']
        cos_dist = 1.0 - float(np.dot(e_l, e_r) / (np.linalg.norm(e_l) * np.linalg.norm(e_r) + 1e-8))
        dice_diff = 1.0 - dice_score(L['baseline_mask'], R['baseline_mask'])
        centroid_dist = (abs(L['baseline_centroid'][0] - R['baseline_centroid'][0])
                          if L['baseline_centroid'] and R['baseline_centroid'] else None)
        row2_rows.append({{'dataset': DATASET_NAME, 'person': person, 'subject_id': sid,
                            'organ_pair': f'{{left_name}}/{{right_name}}',
                            'text_embed_cos_dist': round(cos_dist, 4), 'output_dice_dist': round(dice_diff, 4),
                            'output_centroid_dist_vox': round(centroid_dist, 2) if centroid_dist is not None else None}})

        # Row 3: causal side-agreement (left-prompt -> left side, right-prompt -> right side)
        L_cs, R_cs = mask_centroid_side(L['baseline_mask']), mask_centroid_side(R['baseline_mask'])
        left_pred_matches_left_gt = bool(L['gt_side_axis0'] and L_cs and L_cs[2][0] == L['gt_side_axis0'])
        right_pred_matches_right_gt = bool(R['gt_side_axis0'] and R_cs and R_cs[2][0] == R['gt_side_axis0'])
        row3_rows.append({{'dataset': DATASET_NAME, 'person': person, 'subject_id': sid,
                            'organ_pair': f'{{left_name}}/{{right_name}}',
                            'left_prompt_predicts_left_side': bool(left_pred_matches_left_gt),
                            'right_prompt_predicts_right_side': bool(right_pred_matches_right_gt),
                            'causally_grounded': bool(left_pred_matches_left_gt and right_pred_matches_right_gt)}})

        # Row 4: spatial diff map between left-prompt and right-prompt masks
        diff = np.logical_xor(L['baseline_mask'], R['baseline_mask'])
        n_diff = int(diff.sum())
        n_total = int(L['baseline_mask'].sum() + R['baseline_mask'].sum())
        diff_coords = np.nonzero(diff)
        diff_centroid_axis0 = float(diff_coords[0].mean()) if len(diff_coords[0]) else None
        row4_rows.append({{'dataset': DATASET_NAME, 'person': person, 'subject_id': sid,
                            'organ_pair': f'{{left_name}}/{{right_name}}',
                            'diff_voxels': n_diff, 'diff_fraction_of_union': round(n_diff / max(n_total, 1), 4),
                            'diff_centroid_axis0': round(diff_centroid_axis0, 2) if diff_centroid_axis0 is not None else None}})

for name, rows_out in [('row1_image_text_alignment', row1_rows), ('row2_embedding_output_regression', row2_rows),
                         ('row3_causal_embedding_swap', row3_rows), ('row4_fused_feature_diff', row4_rows)]:
    path = RESULTS_ROOT / f'{{name}}_{dataset_name}.csv'
    if rows_out:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
        print(f'wrote {{path}} ({{len(rows_out)}} rows)')

# Row 2's regression slope (embedding distance -> output distance)
if row2_rows:
    xs = np.array([r['text_embed_cos_dist'] for r in row2_rows])
    ys = np.array([r['output_dice_dist'] for r in row2_rows])
    if len(xs) >= 2 and xs.std() > 0:
        slope, intercept = np.polyfit(xs, ys, 1)
        r2 = np.corrcoef(xs, ys)[0, 1] ** 2
        print(f'\\nRow 2 regression: output_dice_dist = {{slope:.3f}} * cos_dist + {{intercept:.3f}} (R^2={{r2:.3f}})')
        print('  positive slope -> differing text understanding translates into differing output (genuine grounding)')
        print('  flat/near-zero slope -> text difference does not predict output difference (shortcut/memorized)')
""")

    # ---- Cell: Row 6 -- LIME (genuinely new inference, unlike SHAP above) ---------------
    add_cell(nb, "markdown", f"""## 12. Row 6 -- LIME (`lime.lime_text.LimeTextExplainer`)

Unlike SHAP above (which reused the already-computed exact combos at zero extra cost), LIME
generates its OWN perturbed prompts -- literal random word deletion on `prompt_text`, no
grammar repair -- so this requires genuinely new model inference. Each organ's
`explain_instance` call is batched internally (LIME calls the classifier function once with
all `LIME_NUM_SAMPLES` perturbations together, not one-by-one), reusing the SAME
image-preprocessing trick as the word-attribution pass (one `predictor.preprocess()` per
subject, one `predict_sliding_window_return_logits` call per organ covering all of that
organ's LIME samples at once).

Target metric: Dice vs. ground truth for each perturbed prompt (LIME's `classifier_fn` must
return a 2-column pseudo-probability array `[1 - dice, dice]`, since `LimeTextExplainer` is
built for classification -- this is the standard workaround for using it in a regression-like
setting). Output: `lime_attribution_{dataset_name}.csv`, one row per (subject, organ, word)
with LIME's regression-coefficient-style weight for that word.
""")
    add_cell(nb, "code", f"""LIME_NUM_SAMPLES = 100  # perturbed prompts per organ -- keeps runtime bounded (~13-35 organs x 3 subjects)

def make_lime_classifier_fn(data, orig_shape, bbox, gt_mask):
    def classifier_fn(text_list):
        texts = list(text_list)
        embeddings = predictor.embed_text_prompts(texts)
        logits = predictor.predict_sliding_window_return_logits(data, embeddings).to('cpu')
        with torch.no_grad():
            masks_cropped = (torch.sigmoid(logits.float()) > 0.5).numpy().astype(np.uint8)
        masks_full = np.zeros((masks_cropped.shape[0], *orig_shape), dtype=np.uint8)
        masks_full = insert_crop_into_image(masks_full, masks_cropped, bbox)
        scores = np.array([dice_score(masks_full[i].astype(bool), gt_mask) for i in range(masks_full.shape[0])])
        return np.stack([1.0 - scores, scores], axis=1)
    return classifier_fn

lime_explainer = lime.lime_text.LimeTextExplainer(class_names=['not_organ', 'dice_gt']) if LIME_AVAILABLE else None

lime_fieldnames = ['dataset', 'person', 'subject_id', 'organ', 'is_bilateral', 'prompt_text',
                    'word', 'lime_weight', 'num_samples']
lime_csv_path = RESULTS_ROOT / f'lime_attribution_{dataset_name}.csv'
lime_f = open(lime_csv_path, 'w', newline='')
lime_writer = csv.DictWriter(lime_f, fieldnames=lime_fieldnames)
lime_writer.writeheader()
lime_n_written = 0

if not LIME_AVAILABLE:
    print('LIME_AVAILABLE=False (import failed earlier) -- skipping row 6, '
          f'writing an empty {{lime_csv_path.name}}. Rows 1-5, 7-11 are unaffected.')
else:
    for si, (sid, info) in enumerate(subject_organs.items(), 1):
        person, rows = info['person'], info['rows']
        print(f'[{{si}}/{{len(subject_organs)}}] {{person}}/{{sid}}: LIME (row 6)...')
        image_path = RESEARCH_REPO / person / 'samples' / DATASET_NAME / rows[0]['image_path']
        label_path = RESEARCH_REPO / person / 'samples' / DATASET_NAME / rows[0]['label_path']
        img_arr, _ = load_volume_ras(image_path)
        label_vol, _ = load_seg_ras(label_path)
        data, bbox, orig_shape = predictor.preprocess(img_arr)

        for row0 in rows:
            organ = row0['organ']
            label_id = int(row0['label_id'])
            prompt_text = row0['prompt_text']
            gt_mask = (label_vol == label_id)
            roles = role_names_for(organ)

            classifier_fn = make_lime_classifier_fn(data, orig_shape, bbox, gt_mask)
            t0 = time.time()
            exp = lime_explainer.explain_instance(
                prompt_text, classifier_fn, labels=[1],
                num_features=len(prompt_text.split()), num_samples=LIME_NUM_SAMPLES)
            elapsed = time.time() - t0
            for word, weight in exp.as_list(label=1):
                lime_writer.writerow({{
                    'dataset': DATASET_NAME, 'person': person, 'subject_id': sid, 'organ': organ,
                    'is_bilateral': 'position' in roles, 'prompt_text': prompt_text,
                    'word': word, 'lime_weight': round(float(weight), 6), 'num_samples': LIME_NUM_SAMPLES,
                }})
                lime_n_written += 1
            lime_f.flush()
            print(f'    {{organ}}: {{elapsed:.1f}}s for {{LIME_NUM_SAMPLES}} samples')

lime_f.close()
print(f'\\nwrote {{lime_csv_path}} ({{lime_n_written}} rows)')
""")

    # ---- Cell 26/27: summary ------------------------------------------------------------
    add_cell(nb, "markdown", "## 13. Summary\n")
    add_cell(nb, "code", f"""import shutil
import pandas as pd

wa_df = pd.read_csv(RESULTS_ROOT / f'word_attribution_{dataset_name}.csv')
shap_df = pd.read_csv(RESULTS_ROOT / f'shap_attribution_{dataset_name}.csv')
lime_df = pd.read_csv(RESULTS_ROOT / f'lime_attribution_{dataset_name}.csv')
bp_df = pd.read_csv(RESULTS_ROOT / f'baseline_predictions_{dataset_name}.csv')
am_df_len = sum(1 for _ in open(RESULTS_ROOT / f'ablation_masks_{dataset_name}.csv')) - 1  # -1 header, avoids loading the whole (large) file into memory
print(f'=== {title} MODEL-INTERNALS INVESTIGATION Summary ===')
print(f'Subjects: {{list(subject_organs)}}')
print()
print(f'Baseline predictions: {{len(bp_df)}} (subject x organ) rows, mean Dice vs. GT: {{bp_df["dice_gt"].mean():.4f}}')
print(f'Ablation masks (every combo, no experiment left out): {{am_df_len}} rows -> ablation_masks_{dataset_name}.csv')
print()
print('Row 5 (exact Shapley) -- mean |value| by role, f_dice_gt metric:')
print(wa_df[wa_df['metric'] == 'f_dice_gt'].groupby('role')['shapley_value'].apply(lambda s: s.abs().mean()).sort_values(ascending=False).round(4))
print()
print(f'Row 7 (SHAP KernelExplainer) -- {{(~shap_df["matches_exact_shapley"]).sum()}}/{{len(shap_df)}} rows '
      'differ from exact Shapley by > 1e-4 (expect 0). Mean |shap_value| by role, f_dice_gt metric:')
print(shap_df[shap_df['metric'] == 'f_dice_gt'].groupby('role')['shap_value'].apply(lambda s: s.abs().mean()).sort_values(ascending=False).round(4))
print()
print(f'Row 6 (LIME) -- {{len(lime_df)}} (subject, organ, word) rows. Mean |lime_weight| by word '
      '(NOTE: LIME tokenizes the raw prompt, so words here are literal tokens, not the role-collapsed '
      'names used above -- e.g. "left"/"right" appear directly instead of as a "position" role):')
print(lime_df.groupby('word')['lime_weight'].apply(lambda s: s.abs().mean()).sort_values(ascending=False).round(4).head(10))
print()
print('  -- is "left"/"right" ranked above the organ-irrelevant function words (the/in/this/image), across')
print('     both exact Shapley/SHAP (role-based) and LIME (word-based)? Agreement across all three methods')
print('     is the strongest version of this evidence for the paper comparison.')

for name in ['row1_image_text_alignment', 'row2_embedding_output_regression',
             'row3_causal_embedding_swap', 'row4_fused_feature_diff']:
    p = RESULTS_ROOT / f'{{name}}_{dataset_name}.csv'
    if p.exists():
        df = pd.read_csv(p)
        print(f'\\n{{name}}: {{len(df)}} rows -> {{p.name}}')

zip_path = WORKING_DIR / f'voxtell_{dataset_name}_groundedness_investigation_results'
shutil.make_archive(str(zip_path), 'zip', str(RESULTS_ROOT))
print(f"\\nArchive created at {{zip_path}}.zip ({{os.path.getsize(str(zip_path) + '.zip') / 1e6:.1f}} MB)")
print("This zip file will appear directly in Kaggle's Output panel.")
""")

    # ---- Cell 28: download instructions --------------------------------------------------
    add_cell(nb, "markdown", f"""## 14. Downloading the results -- everything you need for LOCAL analysis/plotting

Nine CSVs in the Kaggle **Output** tab (after a **Save Version** run finishes), all under
`results/`. None of the local analysis below needs a GPU or `voxtell`/`torch` installed --
just `numpy`/`pandas`/`matplotlib` (`decode_mask_rle()` is pure Python/numpy/zlib; plotting
LIME/SHAP values doesn't need those libraries installed locally either, just their CSV output).

- **`baseline_predictions_{dataset_name}.csv`** -- **the one to download for plotting the real
  (unablated) prediction.** One row per (subject, organ): `prompt_text`, `dice_gt`,
  `n_pred_voxels`, `volume_shape_ras`, the actual **`gt_mask_rle_b64` / `pred_mask_rle_b64`**
  segmentation masks (run-length-encoded + zlib-compressed + base64, decodable with
  `decode_mask_rle(s, shape)` -- same helper as your rotation-sweep CSVs), plus
  `pred_centroid_vox_0/1/2`, `pred_bbox_vox_{{0,1,2}}{{min,max}}`, `pred_side_axis0/1/2`, and the
  matching `gt_centroid_vox_*` / `gt_bbox_vox_*` / `gt_side_axis*` columns. Use with
  `scripts/plot_groundedness_results.py`.
- **`ablation_masks_{dataset_name}.csv`** -- **every one of the 2^n word-ablation combos'
  masks, not just the baseline -- no experiment left out.** One row per (subject, organ,
  combo): `roles_present`/`roles_dropped` (which word slots were kept/dropped for this combo),
  `prompt_text` (the actual ablated prompt sent to the model), `f_dice_gt`/`f_dice_self`/
  `f_centroid_shift`/`f_volume` (rows 8-11's metrics for this combo), `n_pred_voxels`,
  `volume_shape_ras`, and **`mask_rle_b64`** -- that combo's actual predicted mask, decodable
  the same way. `is_baseline=True` marks the all-roles-present (real prompt) row, identical to
  that organ's row in `baseline_predictions_{dataset_name}.csv`. **This file is large** (up to
  {2**6} rows per bilateral organ, {2**5} per non-bilateral organ, x every organ x subject --
  the word_attribution summary below is the exact-Shapley numbers already computed FROM these
  rows, so you only need this file if you want to visually inspect a *specific* ablated
  prediction, e.g. "what did dropping 'left' actually look like on this kidney").
- `word_attribution_{dataset_name}.csv` -- row 5 (exact Shapley values per role x organ x
  subject x metric) -- the aggregated numbers, one row per role instead of per combo.
- **`shap_attribution_{dataset_name}.csv`** -- row 7, real `shap.KernelExplainer` output over
  the same role/metric space, plus a `matches_exact_shapley` boolean per row (should be all
  `True` at this feature count -- SHAP internally enumerates the full 2^n space here, so it's
  mathematically expected to agree with row 5; the agreement itself is evidence worth citing).
- **`lime_attribution_{dataset_name}.csv`** -- row 6, real `lime.lime_text.LimeTextExplainer`
  output. One row per (subject, organ, word) -- note these are LIME's own literal word tokens
  from the raw prompt (e.g. "left", "kidney" as separate entries), not the role-collapsed names
  used in the Shapley/SHAP files, since LIME does raw text perturbation rather than
  grammar-aware role ablation.
- `row1_image_text_alignment_{dataset_name}.csv` -- row 1 (matched vs. mismatched IoU)
- `row2_embedding_output_regression_{dataset_name}.csv` -- row 2 (embedding-distance vs.
  output-distance points; fit the regression slope from these, or use the number printed in
  section 11)
- `row3_causal_embedding_swap_{dataset_name}.csv` -- row 3 (causal side-agreement)
- `row4_fused_feature_diff_{dataset_name}.csv` -- row 4 (spatial diff map summary)

Note: Output only appears once a **Save Version** (Save & Run All) finishes, not from just an
interactive "Run All" in the editor session.
""")

    out_path = REPO_ROOT / out_name
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    build(
        "voxtell-amos-rotation-multiangle-kaggle.ipynb", "amos22_mri", "AMOS22 MRI",
        AMOS_SUBJECTS, "voxtell-amos-groundedness-investigation-kaggle.ipynb",
    )
    build(
        "voxtell-oasis-rotation-multiangle-kaggle.ipynb", "neurite_oasis", "Neurite OASIS",
        OASIS_SUBJECTS, "voxtell-oasis-groundedness-investigation-kaggle.ipynb",
    )
