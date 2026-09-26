"""
Generator for the two Kaggle notebooks that run plan rows 12-16 on VoxTell:

    voxtell-amos-mechanistic-experiments-kaggle.ipynb
    voxtell-oasis-mechanistic-experiments-kaggle.ipynb

  row 12  image left-right flip, prompt unchanged
  row 13  organ filled with noise / both organs filled with noise / noise image
  row 14  embedding blend and per-layer patching between "left X" and "right X"
  row 15  cross-attention of each prompt over every organ region, per decoder layer
  row 16  distance between the left and right query states at every decoder layer

The first 16 cells (GPU check, install, clone, model load, helpers) come from the existing groundedness
notebooks; the experiment code is embedded from scripts/voxtell_mechanistic.py and scripts/voxtell_experiments.py
(both tested locally on a tiny stand-in model, including that patching a whole layer state reproduces the source
prompt exactly).

Usage:  python scripts/build_mechanistic_notebooks.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent

AMOS_SUBJECTS = [("nipu", "amos_0508"), ("faria", "amos_0530"), ("hridoy", "amos_0555"), ("faria", "amos_0544"),
                 ("nipu", "amos_0553"), ("hridoy", "amos_0558"), ("nipu", "amos_0522"), ("faria", "amos_0532"),
                 ("hridoy", "amos_0554"), ("nipu", "amos_0575")]
OASIS_SUBJECTS = [("faria", "OASIS_OAS1_0014_MR1"), ("faria", "OASIS_OAS1_0057_MR1"), ("hridoy", "OASIS_OAS1_0123_MR1"),
                  ("hridoy", "OASIS_OAS1_0135_MR1"), ("nipu", "OASIS_OAS1_0062_MR1"), ("faria", "OASIS_OAS1_0363_MR1"),
                  ("hridoy", "OASIS_OAS1_0419_MR1"), ("nipu", "OASIS_OAS1_0076_MR1"), ("nipu", "OASIS_OAS1_0151_MR1"),
                  ("nipu", "OASIS_OAS1_0421_MR1")]
# OASIS: only the structures VoxTell actually segments with the full prompt (Dice ~ 0.8-0.9); the other three pairs of
# the earlier runs have Dice ~ 0 even with the full prompt, so an intervention on them cannot show anything.
# Rows 13-16 OASIS notebook: the 5 subjects with the largest masks (sum of the 4 organs below, rotation 0).
OASIS_TOP5 = [("hridoy", "OASIS_OAS1_0123_MR1"), ("hridoy", "OASIS_OAS1_0135_MR1"), ("faria", "OASIS_OAS1_0057_MR1"),
              ("nipu", "OASIS_OAS1_0062_MR1"), ("nipu", "OASIS_OAS1_0076_MR1")]
OASIS_CORES = ["cerebral white matter", "cerebellum cortex"]

TITLE = """# VoxTell on {title} -- PLAN ROWS 12-16: image flip, organ noise, embedding intervention, attention, layer distance (Kaggle GPU)

The five experiments the earlier notebooks could not run (they only used the public API and final masks):

| Row | Experiment | Question |
|---|---|---|
| 12 | Flip the image left-right, keep the prompt | Does "left" follow the patient's anatomy, or the left side of the image? |
| 13 | Fill the organ / both organs with noise, or use a noise image | Does the model need image evidence, or does it draw a memorized organ? |
| 14 | Blend or patch the "left" and "right" representations inside the model | Where is the side decided, and does the mask change smoothly? |
| 15 | Cross-attention of each prompt over the organ regions | Does "left kidney" look at the left kidney inside the model? |
| 16 | Distance between the left and right query states at every decoder layer | At which layer do left and right first differ? |

- Subjects: **{n_subjects}** ({subjects}); organs: **{organs}**; no rotation.
- Rows 14-16 use one custom sliding-window pass per subject (`voxtell_mechanistic.run_outputs`): a single image-encoder
  pass per patch serves every variant. The custom loop is **checked against the predictor's own output** before anything
  else runs (section 3), and the "whole state" patch of a layer is a built-in positive control (it must reproduce the
  source prompt's mask, up to rounding).
- Output: CSVs under `results/` (+ `attention_maps/`), zipped at the end.
- To run only part of the notebook, skip the sections you do not need; each section is independent once sections 1-3 have run.
"""

SRC_CELL = """import json, sys, types

# The experiment code is embedded here so the notebook is self-contained. (Sources: scripts/voxtell_mechanistic.py and
# scripts/voxtell_experiments.py in the repo -- edit them there and rebuild with scripts/build_mechanistic_notebooks.py.)
_MECH_SRC = json.loads(@@MECH@@)
_EXP_SRC = json.loads(@@EXP@@)
_m = types.ModuleType('voxtell_mechanistic')
exec(compile(_MECH_SRC, 'voxtell_mechanistic', 'exec'), _m.__dict__)
sys.modules['voxtell_mechanistic'] = _m
_e = types.ModuleType('voxtell_experiments')
exec(compile(_EXP_SRC, 'voxtell_experiments', 'exec'), _e.__dict__)
sys.modules['voxtell_experiments'] = _e
import voxtell_mechanistic as vm
import voxtell_experiments as ve
print('experiment modules loaded')
"""

CONFIG_CELL = """import gc

DATASET_NAME = '@@DATASET@@'
SUBJECTS = [
    @@SUBJECTS@@
]
ORGAN_CORES = @@CORES@@   # None = every bilateral organ

subject_organs = {}
for person, sid in SUBJECTS:
    rows = [r for r in load_manifest_rot000(person, DATASET_NAME) if r['subject_id'] == sid]
    rows = [r for r in rows if r['organ'].split()[0].lower() in ('left', 'right')]
    if ORGAN_CORES:
        rows = [r for r in rows if r['organ'].split(None, 1)[1] in ORGAN_CORES]
    have = {r['organ'] for r in rows}
    rows = [r for r in rows if ve.mirror(r['organ']) in have]          # keep complete left/right pairs only
    assert rows, f'no bilateral organs for {person}/{sid}'
    subject_organs[sid] = {'person': person, 'rows': rows}
print(len(subject_organs), 'subjects;', 'organs per subject:', [len(v['rows']) for v in subject_organs.values()][:3], '...')


def load_subject(sid):
    info = subject_organs[sid]
    rows = info['rows']
    base = RESEARCH_REPO / info['person'] / 'samples' / DATASET_NAME
    img_arr, _ = load_volume_ras(base / rows[0]['image_path'])
    label_vol, _ = load_seg_ras(base / rows[0]['label_path'])
    organs = [r['organ'] for r in rows]
    ids = {r['organ']: int(r['label_id']) for r in rows}
    prompts = {r['organ']: r['prompt_text'] for r in rows}
    return info['person'], img_arr, label_vol, organs, ids, prompts


def append_csv(path, rows, first):
    if not rows:
        return
    write_header = first or not path.exists()      # also right if an earlier subject produced no rows for this file
    with open(path, 'w' if first else 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)


def tag(rows, sid, person):
    return [{'dataset': DATASET_NAME, 'person': person, 'subject_id': sid, **r} for r in rows]
"""

VALIDATE_CELL = """# The custom loop must reproduce the predictor's own masks, otherwise rows 14-16 would not be trustworthy.
_sid = next(iter(subject_organs))
_, _img, _lab, _organs, _ids, _prompts = load_subject(_sid)
_agree = ve.validate_custom_loop(predictor, _img, [_prompts[o] for o in _organs])
print(f'custom sliding-window loop vs. the predictor: {_agree * 100:.3f}% of voxels have the same sign')
assert _agree > 0.995, 'the custom loop does not match the predictor -- do not trust rows 14-16'
del _img, _lab
gc.collect(); torch.cuda.empty_cache()
"""

ROWS_12_13_CELL = """import traceback

r12_path = RESULTS_ROOT / f'row12_image_flip_@@SUFFIX@@.csv'
r13_path = RESULTS_ROOT / f'row13_organ_noise_@@SUFFIX@@.csv'
first = True
for si, sid in enumerate(subject_organs, 1):
    t0 = time.time()
    try:
        person, img_arr, label_vol, organs, ids, prompts = load_subject(sid)
        rows12 = ve.run_flip(predictor, img_arr, label_vol, organs, ids, prompts, rl_axis=RL_AXIS)
        rows13 = ve.run_erasure(predictor, img_arr, label_vol, organs, ids, prompts, seed=si)
        append_csv(r12_path, tag(rows12, sid, person), first)
        append_csv(r13_path, tag(rows13, sid, person), first)
        first = False
        print(f'[{si}/{len(subject_organs)}] {sid}: rows 12-13 done in {time.time() - t0:.0f}s')
    except Exception:
        print(f'[{si}/{len(subject_organs)}] {sid}: FAILED'); traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()
print('wrote', r12_path.name, r13_path.name)
"""

ROWS_14_16_CELL = """import numpy as np, traceback

(RESULTS_ROOT / 'attention_maps').mkdir(exist_ok=True)
paths = {k: RESULTS_ROOT / f'{k}_@@SUFFIX@@.csv' for k in
         ('row14_embedding_intervention', 'row15_attention_enrichment', 'row15_attention_centre', 'row16_layer_distance')}
first = True
for si, sid in enumerate(subject_organs, 1):
    t0 = time.time()
    try:
        person, img_arr, label_vol, organs, ids, prompts = load_subject(sid)
        res = ve.run_interventions(predictor, img_arr, label_vol, organs, ids, prompts, rl_axis=RL_AXIS)
        append_csv(paths['row14_embedding_intervention'], tag(res['row14'], sid, person), first)
        append_csv(paths['row15_attention_enrichment'], tag(res['row15'], sid, person), first)
        append_csv(paths['row15_attention_centre'], tag(res['row15_centre'], sid, person), first)
        append_csv(paths['row16_layer_distance'], tag(res['row16'], sid, person), first)
        np.savez_compressed(RESULTS_ROOT / 'attention_maps' / f'{sid}.npz', **res['attention_maps'])
        first = False
        print(f'[{si}/{len(subject_organs)}] {sid}: rows 14-16 done in {time.time() - t0:.0f}s')
    except Exception:
        print(f'[{si}/{len(subject_organs)}] {sid}: FAILED'); traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()
print('wrote', [p.name for p in paths.values()])
"""

SUMMARY_CELL = """import shutil
import pandas as pd

pd.set_option('display.width', 200, 'display.max_columns', 30)
R = lambda n: pd.read_csv(RESULTS_ROOT / f'{n}_@@SUFFIX@@.csv')

print('=== Row 12: image flip -- what does the mask cover? (share of organs) ===')
d = R('row12_image_flip')
print(d.groupby(['condition', 'call']).size().unstack(fill_value=0))

print('\\n=== Row 13: organ noise -- mask size vs. the original prediction, share of empty masks ===')
d = R('row13_organ_noise')
print(d.groupby('condition').agg(size_vs_original=('size_vs_original', 'median'), empty=('empty', 'mean'),
                                 in_noised_region=('frac_of_mask_in_region', 'mean')).round(3))

print('\\n=== Row 14: which side does the mask land on? ===')
d = R('row14_embedding_intervention')
b = d[d.variant == 'blend']
print('blend (alpha 0 = "left", 1 = "right"):'); print(b.groupby(['alpha', 'call']).size().unstack(fill_value=0))
u = d[d.variant == 'single_layer_update']
print('single-layer update from the other prompt -- share of masks that moved to the source side, by layer:')
print(u.groupby(['dst', 'layer']).moved_to_source_side.mean().unstack().round(2))
c = d[d.variant == 'whole_state_control']
print('positive control (a whole-state patch must reproduce the source mask): voxel agreement with the source prompt, mean / min:',
      c.agreement_with_source_baseline.mean().round(5), '/', c.agreement_with_source_baseline.min().round(5), '(expected >= 0.999)')

print('\\n=== Row 15: attention enrichment (1 = as much as elsewhere), by layer ===')
a = R('row15_attention_enrichment')
a['kind'] = np.where(a.is_matched, 'named organ', np.where(a.is_wrong_side, 'other side, same organ', 'different organ'))
print(a.groupby(['layer', 'kind']).enrichment.mean().unstack().round(2))
cc = R('row15_attention_centre')
print('attention centre closer to the named organ (share):'); print(cc.groupby('layer').closer_to_named_organ.mean().round(2).to_dict())

print('\\n=== Row 16: cosine distance between the query states, by layer (0 = input) ===')
l = R('row16_layer_distance')
print(l[l.scope == 'organ_patches'].pivot_table(index='layer', columns='pair', values='cosine_distance').round(3))

zip_path = WORKING_DIR / 'voxtell_@@SUFFIX@@_mechanistic_results'
shutil.make_archive(str(zip_path), 'zip', str(RESULTS_ROOT))
print('\\nArchive:', str(zip_path) + '.zip')
"""

END_MD = """## Outputs (all under `results/`)

| File | Row | Content |
|---|---|---|
| `row12_image_flip_@@SUFFIX@@.csv` | 12 | per organ, original vs. flipped: coverage of the named organ / the organ on the other side, `call` = `follows_named_organ` or `follows_image_side` |
| `row13_organ_noise_@@SUFFIX@@.csv` | 13 | per organ and condition (`original`, `noise_target`, `noise_pair`, `noise_image`): mask size vs. the original prediction, coverage of the noised region, offset of the mask centre |
| `row14_embedding_intervention_@@SUFFIX@@.csv` | 14 | per pair: baselines, embedding blends (alpha), single-layer update patches, updates from a layer onward, whole-state control; side of the resulting mask |
| `row15_attention_enrichment_@@SUFFIX@@.csv` | 15 | per prompt x organ region x decoder layer: attention enrichment and mass fraction |
| `row15_attention_centre_@@SUFFIX@@.csv` | 15 | per prompt x layer: where the attention centre lies vs. the named and the other-side organ |
| `row16_layer_distance_@@SUFFIX@@.csv` | 16 | cosine / relative L2 distance between the left and right query states per layer (layer 0 = input), with an organ-vs-organ reference |
| `attention_maps/<subject>.npz` | 15 | down-sampled attention maps (last layer and mean of layers) for plotting |

Reading guide: in row 14, `moved_to_source_side` is True when patching with the *other* prompt's contribution turns the mask
into the other side. `agreement_with_source_baseline` is the machinery check: for every `whole_state_control` row it should be
about 1 (>= 0.999; patching a whole layer state must reproduce the source prompt's mask, up to floating-point rounding).
"""


ROW_13_CELL = """import traceback

r13_path = RESULTS_ROOT / f'row13_organ_noise_@@SUFFIX@@.csv'
first = True
for si, sid in enumerate(subject_organs, 1):
    t0 = time.time()
    try:
        person, img_arr, label_vol, organs, ids, prompts = load_subject(sid)
        rows13 = ve.run_erasure(predictor, img_arr, label_vol, organs, ids, prompts, seed=si)
        append_csv(r13_path, tag(rows13, sid, person), first)
        first = False
        print(f'[{si}/{len(subject_organs)}] {sid}: row 13 done in {time.time() - t0:.0f}s')
    except Exception:
        print(f'[{si}/{len(subject_organs)}] {sid}: FAILED'); traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()
print('wrote', r13_path.name)
"""


def drop_row12(text):
    """Remove the row-12 pieces from the shared title / summary / outputs text (rows-13-16-only notebook)."""
    text = text.replace("PLAN ROWS 12-16: image flip, organ noise", "PLAN ROWS 13-16: organ noise")
    text = text.replace("The five experiments", "The four experiments")
    lines = [l for l in text.splitlines(keepends=True) if not (l.startswith("| 12 |") or l.startswith("| `row12_"))]
    text = "".join(lines)
    a = text.find("print('=== Row 12")
    if a >= 0:
        b = text.rfind("print(", 0, text.index("=== Row 13"))
        text = text[:a] + text[b:]
    return text


def cell(kind, text):
    src = text.splitlines(keepends=True)
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def build(base_nb, title, dataset, subjects, cores, suffix, out_name, rows13_16=False):
    nb = json.loads((REPO / base_nb).read_text(encoding="utf-8"))
    nb["cells"] = nb["cells"][:16]
    subj_str = ", ".join(sid for _, sid in subjects)
    organs = ", ".join(cores) if cores else "kidney and adrenal gland (left/right)"
    title_md = TITLE.format(title=title, n_subjects=len(subjects), subjects=subj_str, organs=organs)
    nb["cells"][0] = cell("markdown", drop_row12(title_md) if rows13_16 else title_md)
    sub = lambda t: (t.replace("@@SUFFIX@@", suffix).replace("@@DATASET@@", dataset))
    mech = json.dumps((SCRIPTS / "voxtell_mechanistic.py").read_text(encoding="utf-8"))
    exp = json.dumps((SCRIPTS / "voxtell_experiments.py").read_text(encoding="utf-8"))
    sec4 = [
        cell("markdown", "## 4. Row 13 -- organ filled with noise, both organs filled with noise, noise image\n\n"
                         "Public-API predictions only (no changes inside the model). Replaces the voxels of the named organ (and, separately, of both organs "
                         "of the pair) with Gaussian noise (mean and spread of the image), or replaces the whole image by such noise."),
        cell("code", sub(ROW_13_CELL)),
    ] if rows13_16 else [
        cell("markdown", "## 4. Rows 12 and 13 -- image flip, organ noise, noise image\n\n"
                         "Public-API predictions only (no changes inside the model). Row 12 flips the image and its labels along the "
                         "right-left axis and keeps every prompt. Row 13 replaces the named organ (and, separately, both organs of the "
                         "pair) with Gaussian noise, or replaces the whole image by noise with the image's mean and spread."),
        cell("code", sub(ROWS_12_13_CELL)),
    ]
    nb["cells"] += [
        cell("markdown", "## 1. Load the experiment code"),
        cell("code", SRC_CELL.replace("@@MECH@@", repr(mech)).replace("@@EXP@@", repr(exp))),
        cell("markdown", "## 2. Subjects, organs and helpers"),
        cell("code", sub(CONFIG_CELL).replace("@@SUBJECTS@@", ",\n    ".join(f'("{p}", "{s}")' for p, s in subjects))
             .replace("@@CORES@@", repr(cores))),
        cell("markdown", "## 3. Check: the custom sliding-window loop must match the predictor"),
        cell("code", VALIDATE_CELL),
        *sec4,
        cell("markdown", "## 5. Rows 14, 15 and 16 -- one custom pass per subject\n\n"
                         "Per subject and left/right pair: baselines, five embedding blends (alpha = -0.5, 0.25, 0.5, 0.75, 1.5 between "
                         "\"left\" and \"right\"), single-layer patches of the query update (all six layers, both directions), patches from a "
                         "layer onward, and whole-state controls -- while recording the cross-attention (row 15) and the left/right distance per layer (row 16)."),
        cell("code", sub(ROWS_14_16_CELL)),
        cell("markdown", "## 6. Summary and archive"),
        cell("code", sub(drop_row12(SUMMARY_CELL) if rows13_16 else SUMMARY_CELL)),
        cell("markdown", sub(drop_row12(END_MD) if rows13_16 else END_MD)),
    ]
    out = REPO / out_name
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[ok] wrote", out)


if __name__ == "__main__":
    build("voxtell-amos-groundedness-investigation-kaggle.ipynb", "AMOS22 MRI", "amos22_mri", AMOS_SUBJECTS, None,
          "amos22_mri", "voxtell-amos-mechanistic-experiments-kaggle.ipynb")
    build("voxtell-oasis-groundedness-investigation-batch1-kaggle.ipynb", "Neurite-OASIS", "neurite_oasis", OASIS_SUBJECTS,
          OASIS_CORES, "neurite_oasis", "voxtell-oasis-mechanistic-experiments-kaggle.ipynb")
    build("voxtell-amos-groundedness-investigation-kaggle.ipynb", "AMOS22 MRI", "amos22_mri", AMOS_SUBJECTS, None,
          "amos22_mri", "voxtell-amos-mechanistic-rows13-16-kaggle.ipynb", rows13_16=True)
    build("voxtell-oasis-groundedness-investigation-batch1-kaggle.ipynb", "Neurite-OASIS", "neurite_oasis", OASIS_TOP5,
          OASIS_CORES, "neurite_oasis", "voxtell-oasis-mechanistic-rows13-16-kaggle.ipynb", rows13_16=True)
