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
# Rows 17-24 AMOS notebook: 3 subjects, one per annotator (the first 3 of AMOS_SUBJECTS above).
# cores=None -> every bilateral organ AMOS has, i.e. left/right kidney and left/right adrenal gland
# (the only two bilateral organ pairs in this dataset's manifest).
AMOS_THREE_SUBJECTS = AMOS_SUBJECTS[:3]
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
# One OASIS subject (Faria's data only), restricted to the 9 largest-mask bilateral organ cores
# (by mean GT voxel count across subjects, rotation 0; decoded from gt_mask_rle_b64 in
# predictions_voxtell_neurite_oasis_rotation_*.csv -- brain stem excluded, it has no left/right pair):
# cerebral cortex > cerebral white matter > cerebellum cortex > cerebellum white matter >
# lateral ventricle > thalamus > putamen > hippocampus > caudate.
# OASIS_OAS1_0057_MR1 is the largest-mask of Faria's 3 subjects over these 9 cores (1,443,837 vs.
# 1,408,755 voxels for OASIS_OAS1_0014_MR1; OASIS_OAS1_0363_MR1 has no rotation CSV to verify).
OASIS_ONE_SUBJECT = [("faria", "OASIS_OAS1_0057_MR1")]
OASIS_RICH_CORES = ["cerebral cortex", "cerebral white matter", "cerebellum cortex", "cerebellum white matter",
                     "lateral ventricle", "thalamus", "putamen", "hippocampus", "caudate"]

TITLE = """# VoxTell on {title} -- PLAN ROWS 12-24: image flip, organ noise, embedding intervention, attention, layer distance, image attribution, probing, ablation (Kaggle GPU)

The experiments the earlier (public-API-only) notebooks could not run, plus the image-side attribution suite (rows 17-21) and the model-internals suite (rows 22-24: is laterality linearly decodable, is there one shared "left->right" direction, which specific attention head is responsible):

| Row | Experiment | Question |
|---|---|---|
| 12 | Flip the image left-right, keep the prompt | Does "left" follow the patient's anatomy, or the left side of the image? |
| 13 | Fill the organ / both organs with noise, or use a noise image | Does the model need image evidence, or does it draw a memorized organ? |
| 14 | Blend or patch the "left" and "right" representations inside the model | Where is the side decided, and does the mask change smoothly? |
| 15 | Cross-attention of each prompt over the organ regions | Does "left kidney" look at the left kidney inside the model? |
| 16 | Distance between the left and right query states at every decoder layer | At which layer do left and right first differ? |
| 17 | Voxel-grid occlusion Shapley map around one organ | Which *region* of the image, if removed, changes the mask most -- the image analogue of the word-Shapley table? |
| 18 | Per-layer KL divergence between cross-attention and the GT mask | Which decoder layer's attention best matches the clinically relevant region? |
| 19 | Vanilla-gradient + Grad-CAM saliency (one patch per organ) | A cheap, approximate counterpart to row 17's exact map (the LIME-to-Shapley relationship, for images) |
| 20 | Fused spatial feature map, matched vs. mismatched prompt | Does the model's internal image+text fusion actually move with the prompt, or stay fixed regardless of what's asked? |
| 21 | Joint Shapley over image / organ word / position word | Puts image contribution and word contribution on one common scale |
| 22 | Linear probe for left/right at every decoder layer | Is laterality linearly DECODABLE in the hidden state, independent of what mask comes out -- stricter than row 16's distance |
| 23 | "right - left" embedding geometry across organs | Is there one reusable "left->right" direction, or a different, unrelated one per organ (is "left" a composable concept)? |
| 24 | Zero-ablate one cross-attention head at a time | Which SPECIFIC head, not just which layer, is causally responsible for laterality |

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

ROW_0_BASELINE_CELL = """import traceback

# Shared plotting export, run once per subject before any of rows 12-21: the full-prompt predicted
# mask and the GT mask for EVERY organ, RLE-encoded so you can decode and overlay them on the
# source image (image_path/label_path) slice by slice -- not just the summary metrics each row
# below computes. See scripts/voxtell_experiments.py's decode_mask_rle to reload these.
r0_path = RESULTS_ROOT / f'row0_baseline_masks_@@SUFFIX@@.csv'
first = True
for si, sid in enumerate(subject_organs, 1):
    t0 = time.time()
    try:
        person, img_arr, label_vol, organs, ids, prompts = load_subject(sid)
        info = subject_organs[sid]
        img_path = info['rows'][0]['image_path']
        lab_path = info['rows'][0]['label_path']
        rows0 = ve.export_baseline_masks(predictor, img_arr, label_vol, organs, ids, prompts,
                                          image_path=img_path, label_path=lab_path)
        append_csv(r0_path, tag(rows0, sid, person), first)
        first = False
        print(f'[{si}/{len(subject_organs)}] {sid}: row 0 (baseline export) done in {time.time() - t0:.0f}s, {len(rows0)} organs')
    except Exception:
        print(f'[{si}/{len(subject_organs)}] {sid}: FAILED'); traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()
print('wrote', r0_path.name)
"""

ROW_17_CELL = """import traceback

# Row 17 is expensive (2**n_blocks model calls per organ -- default 2**8=256, same affordability
# argument as row 5's exact word-Shapley), so it runs on only the first bilateral organ per subject
# by default. Widen ROW17_ORGANS if you want more coverage. The per-block Shapley values are also
# reassembled into a voxel-shaped map (one per organ) and saved to row17_shapley_maps/<subject>.npz
# for plotting against the image/GT from row 0.
ROW17_GRID = 2          # 2**3 = 8 blocks -> 256 coalitions
ROW17_N_ORGANS = 1       # organs per subject (the first N of subject_organs[sid]['rows'])
(RESULTS_ROOT / 'row17_shapley_maps').mkdir(exist_ok=True)
r17_path = RESULTS_ROOT / f'row17_grid_shapley_@@SUFFIX@@.csv'
first = True
for si, sid in enumerate(subject_organs, 1):
    t0 = time.time()
    try:
        person, img_arr, label_vol, organs, ids, prompts = load_subject(sid)
        rows17, maps17 = [], {}
        for organ in organs[:ROW17_N_ORGANS]:
            r, m = ve.run_grid_occlusion(predictor, img_arr, label_vol, organ, ids, prompts[organ],
                                          grid=ROW17_GRID, seed=si, return_map=True)
            rows17 += r
            maps17[organ] = m.astype(np.float16)
        append_csv(r17_path, tag(rows17, sid, person), first)
        if maps17:
            np.savez_compressed(RESULTS_ROOT / 'row17_shapley_maps' / f'{sid}.npz', **maps17)
        first = False
        print(f'[{si}/{len(subject_organs)}] {sid}: row 17 done in {time.time() - t0:.0f}s')
    except Exception:
        print(f'[{si}/{len(subject_organs)}] {sid}: FAILED'); traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()
print('wrote', r17_path.name, 'and row17_shapley_maps/*.npz')
"""

ROW_18_CELL = """import traceback

# Attention maps (last layer and mean of layers, per organ) are saved to row18_attention_maps/<subject>.npz
# for plotting -- the CSV alone only has the scalar KL per layer, no spatial map.
(RESULTS_ROOT / 'row18_attention_maps').mkdir(exist_ok=True)
r18_path = RESULTS_ROOT / f'row18_attention_kl_@@SUFFIX@@.csv'
first = True
for si, sid in enumerate(subject_organs, 1):
    t0 = time.time()
    try:
        person, img_arr, label_vol, organs, ids, prompts = load_subject(sid)
        rows18, maps18 = ve.run_attention_kl(predictor, img_arr, label_vol, organs, ids, prompts, return_maps=True)
        append_csv(r18_path, tag(rows18, sid, person), first)
        if maps18:
            np.savez_compressed(RESULTS_ROOT / 'row18_attention_maps' / f'{sid}.npz', **maps18)
        first = False
        print(f'[{si}/{len(subject_organs)}] {sid}: row 18 done in {time.time() - t0:.0f}s')
    except Exception:
        print(f'[{si}/{len(subject_organs)}] {sid}: FAILED'); traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()
print('wrote', r18_path.name, 'and row18_attention_maps/*.npz')
"""

ROW_19_CELL = """import traceback

# Row 19 needs real gradients -- MUST run outside inference_mode/no_grad (it is not decorated with
# either, unlike every other row in this notebook, which only ever call the model for inference).
# Saliency and Grad-CAM maps (in the crop's coordinate frame, same as row 0's masks) are saved to
# row19_saliency_maps/<subject>.npz for plotting.
(RESULTS_ROOT / 'row19_saliency_maps').mkdir(exist_ok=True)
r19_path = RESULTS_ROOT / f'row19_gradient_saliency_@@SUFFIX@@.csv'
first = True
for si, sid in enumerate(subject_organs, 1):
    t0 = time.time()
    try:
        person, img_arr, label_vol, organs, ids, prompts = load_subject(sid)
        rows19, maps19 = [], {}
        for organ in organs:
            out = ve.run_gradient_saliency(predictor, img_arr, label_vol, organ, ids, prompts[organ], return_maps=True)
            if out is None:
                continue
            r, sal, cam = out
            rows19.append(r)
            maps19[f'{organ}__saliency'] = sal.astype(np.float16)
            maps19[f'{organ}__cam'] = cam.astype(np.float16)
        append_csv(r19_path, tag(rows19, sid, person), first)
        if maps19:
            np.savez_compressed(RESULTS_ROOT / 'row19_saliency_maps' / f'{sid}.npz', **maps19)
        first = False
        print(f'[{si}/{len(subject_organs)}] {sid}: row 19 done in {time.time() - t0:.0f}s')
    except Exception:
        print(f'[{si}/{len(subject_organs)}] {sid}: FAILED'); traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()
print('wrote', r19_path.name, 'and row19_saliency_maps/*.npz')
"""

ROW_20_CELL = """import traceback

# Row 20's hook point (VoxTellDecoder's per-stage `fusion_features`, via seg_layers' forward_pre_hook)
# is UNVERIFIED against a live model -- the first subject's printed stage shapes below should look
# like the four documented resolutions (24, 48, 96, 192 cubed) before trusting the IoU numbers.
# The fused-feature map itself (per organ x condition) is saved to row20_fused_maps/<subject>.npz for
# plotting -- the CSV alone only has the scalar IoU, no spatial map.
(RESULTS_ROOT / 'row20_fused_maps').mkdir(exist_ok=True)
r20_path = RESULTS_ROOT / f'row20_fused_activation_overlap_@@SUFFIX@@.csv'
first = True
_printed_shapes = False
for si, sid in enumerate(subject_organs, 1):
    t0 = time.time()
    try:
        person, img_arr, label_vol, organs, ids, prompts = load_subject(sid)
        rows20, maps20 = [], {}
        for organ in organs:
            r, m = ve.run_activation_overlap(predictor, img_arr, label_vol, organ, ids, prompts, return_maps=True)
            rows20 += r
            maps20.update(m)
        if not _printed_shapes and rows20:
            print('row 20 stages seen:', sorted({r["stage"] for r in rows20}))
            _printed_shapes = True
        append_csv(r20_path, tag(rows20, sid, person), first)
        if maps20:
            np.savez_compressed(RESULTS_ROOT / 'row20_fused_maps' / f'{sid}.npz', **maps20)
        first = False
        print(f'[{si}/{len(subject_organs)}] {sid}: row 20 done in {time.time() - t0:.0f}s')
    except Exception:
        print(f'[{si}/{len(subject_organs)}] {sid}: FAILED'); traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()
print('wrote', r20_path.name, 'and row20_fused_maps/*.npz')
"""

ROW_21_CELL = """import traceback

# row21_coalition_dice_@@SUFFIX@@.csv carries the literal numbers the Shapley credit above is
# computed FROM: the actual Dice score for each of the 8 image/organ-word/position-word present
# combinations (e.g. "image only, no words at all" -> its own real Dice score) -- more intuitive to
# read directly than the derived Shapley credit, at no extra model-call cost (same 8 predictions).
ROW21_N_ORGANS = 1       # organs per subject (cheap: 8 model calls each, but keep default modest)
r21_path = RESULTS_ROOT / f'row21_joint_shapley_@@SUFFIX@@.csv'
r21_coal_path = RESULTS_ROOT / f'row21_coalition_dice_@@SUFFIX@@.csv'
first = True
for si, sid in enumerate(subject_organs, 1):
    t0 = time.time()
    try:
        person, img_arr, label_vol, organs, ids, prompts = load_subject(sid)
        rows21, coal21 = [], []
        for organ in organs[:ROW21_N_ORGANS]:
            r, c = ve.run_joint_image_word_shapley(predictor, img_arr, label_vol, organ, ids, seed=si, return_coalitions=True)
            rows21 += r
            coal21 += c
        append_csv(r21_path, tag(rows21, sid, person), first)
        append_csv(r21_coal_path, tag(coal21, sid, person), first)
        first = False
        print(f'[{si}/{len(subject_organs)}] {sid}: row 21 done in {time.time() - t0:.0f}s')
    except Exception:
        print(f'[{si}/{len(subject_organs)}] {sid}: FAILED'); traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()
print('wrote', r21_path.name, 'and', r21_coal_path.name)
"""

ROW_22_CELL = """import traceback

# Row 22 reuses the SAME validated custom sliding-window loop as rows 14-16 (section 3's check
# above covers it too -- no separate validation needed here). The hidden state for each organ is
# read from its own "home patch" (the one sliding-window patch containing that organ's own GT
# centroid), not averaged over the whole, mostly-irrelevant, volume.
r22_path = RESULTS_ROOT / f'row22_laterality_probing_@@SUFFIX@@.csv'
first = True
for si, sid in enumerate(subject_organs, 1):
    t0 = time.time()
    try:
        person, img_arr, label_vol, organs, ids, prompts = load_subject(sid)
        rows22 = ve.run_laterality_probing(predictor, img_arr, label_vol, organs, ids, prompts)
        append_csv(r22_path, tag(rows22, sid, person), first)
        first = False
        print(f'[{si}/{len(subject_organs)}] {sid}: row 22 done in {time.time() - t0:.0f}s, {len(rows22)} layer rows')
    except Exception:
        print(f'[{si}/{len(subject_organs)}] {sid}: FAILED'); traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()
print('wrote', r22_path.name)
"""

ROW_23_CELL = """import traceback

# Pure text-embedding arithmetic -- no image, no sliding window: the cheapest row in this notebook.
r23_path = RESULTS_ROOT / f'row23_embedding_geometry_@@SUFFIX@@.csv'
first = True
for si, sid in enumerate(subject_organs, 1):
    t0 = time.time()
    try:
        person, img_arr, label_vol, organs, ids, prompts = load_subject(sid)
        rows23 = ve.run_embedding_geometry(predictor, organs, prompts)
        append_csv(r23_path, tag(rows23, sid, person), first)
        first = False
        print(f'[{si}/{len(subject_organs)}] {sid}: row 23 done in {time.time() - t0:.0f}s, {len(rows23)} rows')
    except Exception:
        print(f'[{si}/{len(subject_organs)}] {sid}: FAILED'); traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()
print('wrote', r23_path.name)
"""

ROW_24_CELL = """import traceback

# Row 24 is expensive (n_organs x n_layers x n_heads full sliding-window predictions -- default
# layers=last-layer-only, the same affordability trade-off as row 17's block count; pass
# layers=range(6) to ve.run_attention_head_ablation for every layer, at 6x the cost). Validated
# ONCE, on the first subject's first organ, before trusting any result: ablating ZERO heads at a
# real layer MUST reproduce the unablated prediction exactly -- see
# voxtell_mechanistic.HeadAblation's docstring for why this is mathematically guaranteed if the
# hook is wired correctly (and a loud AttributeError, not a silent wrong answer, if it is not).
r24_path = RESULTS_ROOT / f'row24_attention_head_ablation_@@SUFFIX@@.csv'
first = True
_row24_validated = False
for si, sid in enumerate(subject_organs, 1):
    t0 = time.time()
    try:
        person, img_arr, label_vol, organs, ids, prompts = load_subject(sid)
        if not _row24_validated:
            _last_layer = len(predictor.network.transformer_decoder.layers) - 1
            _agree = vm.validate_head_ablation_noop(
                lambda: ve.predict_masks(predictor, img_arr, [prompts[organs[0]]])[0],
                _last_layer, predictor.network.transformer_decoder.layers)
            print(f'row 24 validation: ablating zero heads agrees with the unablated baseline on {_agree * 100:.3f}% of voxels')
            assert _agree > 0.999, 'HeadAblation is not a true no-op with zero heads -- do not trust row 24'
            _row24_validated = True
        rows24 = ve.run_attention_head_ablation(predictor, img_arr, label_vol, organs, ids, prompts)
        append_csv(r24_path, tag(rows24, sid, person), first)
        first = False
        print(f'[{si}/{len(subject_organs)}] {sid}: row 24 done in {time.time() - t0:.0f}s, {len(rows24)} rows')
    except Exception:
        print(f'[{si}/{len(subject_organs)}] {sid}: FAILED'); traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()
print('wrote', r24_path.name)
"""

SUMMARY_CELL = """import shutil
import pandas as pd

pd.set_option('display.width', 200, 'display.max_columns', 30)
R = lambda n: pd.read_csv(RESULTS_ROOT / f'{n}_@@SUFFIX@@.csv')

print('=== Row 0: baseline Dice per organ (sanity check before trusting any row below) ===')
d = R('row0_baseline_masks')
print(d[['organ', 'dice', 'n_pred_voxels', 'n_gt_voxels']].to_string(index=False))

print('\\n=== Row 12: image flip -- what does the mask cover? (share of organs) ===')
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

print('\\n=== Row 17: voxel-grid occlusion Shapley -- which block matters most, by organ ===')
d = R('row17_grid_shapley')
print(d.groupby(['organ', 'block_id']).shapley_value.mean().unstack().round(3))

print('\\n=== Row 18: attention-GT KL divergence, by layer (lower = better match) ===')
d = R('row18_attention_kl')
print(d.groupby('layer')[['kl_divergence', 'kl_divergence_vs_generic_normalized']].mean().round(3))

print('\\n=== Row 19: gradient saliency enrichment inside vs. outside the organ (1 = no enrichment) ===')
d = R('row19_gradient_saliency')
print(d.groupby('organ')[['saliency_enrichment', 'cam_enrichment']].mean().round(3))

print('\\n=== Row 20: fused-feature activation IoU with GT, matched vs. mismatched prompt ===')
d = R('row20_fused_activation_overlap')
print(d.groupby('condition').iou_with_gt.mean().round(4))

print('\\n=== Row 21: joint image / organ-word / position-word Shapley (f_dice_gt) ===')
d = R('row21_joint_shapley')
print(d[d.metric == 'f_dice_gt'].groupby('role').shapley_value.mean().round(3))

print('\\n=== Row 21: raw Dice score per role-combination (the numbers the Shapley credit above comes from) ===')
dc = R('row21_coalition_dice')
print(dc[['roles_present', 'dice_gt', 'dice_self']].to_string(index=False))

print('\\n=== Row 22: is left/right linearly decodable at each layer? (leave-one-pair-out CV accuracy) ===')
d = R('row22_laterality_probing')
print(d.groupby('layer')[['cv_accuracy', 'train_accuracy']].mean().round(3))

print('\\n=== Row 23: is there one consistent "left -> right" direction across organs? ===')
d = R('row23_embedding_geometry')
s = d[d.kind == 'summary']
print(s[['mean_cosine_similarity', 'coherence_ratio', 'n_organs']].to_string(index=False))

print('\\n=== Row 24: cross-attention head ablation -- dice vs. GT after zeroing each head, by layer ===')
d = R('row24_attention_head_ablation')
print(d.groupby(['layer', 'head'])[['dice_vs_gt', 'dice_vs_baseline']].mean().round(3))

zip_path = WORKING_DIR / 'voxtell_@@SUFFIX@@_mechanistic_results'
shutil.make_archive(str(zip_path), 'zip', str(RESULTS_ROOT))
print('\\nArchive:', str(zip_path) + '.zip')
"""

END_MD = """## Outputs (all under `results/`)

| File | Row | Content |
|---|---|---|
| `row0_baseline_masks_@@SUFFIX@@.csv` | - | per organ: full-prompt predicted mask and GT mask, RLE-encoded (`decode_mask_rle`), plus centroids and Dice -- the shared source for plotting image + GT + prediction overlays |
| `row12_image_flip_@@SUFFIX@@.csv` | 12 | per organ, original vs. flipped: coverage of the named organ / the organ on the other side, `call` = `follows_named_organ` or `follows_image_side` |
| `row13_organ_noise_@@SUFFIX@@.csv` | 13 | per organ and condition (`original`, `noise_target`, `noise_pair`, `noise_image`): mask size vs. the original prediction, coverage of the noised region, offset of the mask centre |
| `row14_embedding_intervention_@@SUFFIX@@.csv` | 14 | per pair: baselines, embedding blends (alpha), single-layer update patches, updates from a layer onward, whole-state control; side of the resulting mask |
| `row15_attention_enrichment_@@SUFFIX@@.csv` | 15 | per prompt x organ region x decoder layer: attention enrichment and mass fraction |
| `row15_attention_centre_@@SUFFIX@@.csv` | 15 | per prompt x layer: where the attention centre lies vs. the named and the other-side organ |
| `row16_layer_distance_@@SUFFIX@@.csv` | 16 | cosine / relative L2 distance between the left and right query states per layer (layer 0 = input), with an organ-vs-organ reference |
| `attention_maps/<subject>.npz` | 15 | down-sampled attention maps (last layer and mean of layers) for plotting |
| `row17_grid_shapley_@@SUFFIX@@.csv` | 17 | per organ and voxel-grid block: exact Shapley credit for that block's image content (the image analogue of the word-Shapley table) |
| `row17_shapley_maps/<subject>.npz` | 17 | per organ: the same Shapley credit reassembled into a full voxel-shaped map, for plotting against row 0's image/GT |
| `row18_attention_kl_@@SUFFIX@@.csv` | 18 | per organ and layer: KL divergence between the GT mask and the cross-attention map, raw and generic-prompt-normalized |
| `row18_attention_maps/<subject>.npz` | 18 | per organ: last-layer and mean-of-layers attention density, downsampled, for plotting against row 0's image/GT |
| `row19_gradient_saliency_@@SUFFIX@@.csv` | 19 | per organ: vanilla-gradient and Grad-CAM saliency, mean inside vs. outside the organ, and their ratio |
| `row19_saliency_maps/<subject>.npz` | 19 | per organ: the vanilla-gradient saliency map and the upsampled Grad-CAM map, voxel-shaped, for plotting |
| `row20_fused_activation_overlap_@@SUFFIX@@.csv` | 20 | per organ and condition (matched / mismatched_side / mismatched_organ): IoU of the fused spatial feature map's hot region with the GT mask |
| `row20_fused_maps/<subject>.npz` | 20 | per organ x condition: the fused-feature map itself, upsampled to the crop frame and downsampled for storage, for plotting |
| `row21_joint_shapley_@@SUFFIX@@.csv` | 21 | per organ, role (image / organ_word / position_word) and metric: Shapley credit on one common scale |
| `row21_coalition_dice_@@SUFFIX@@.csv` | 21 | per organ and role-combination (`roles_present`): the literal Dice score for that combination (e.g. image only, no words at all), against ground truth (`dice_gt`) and against the full-coalition baseline (`dice_self`) -- the raw numbers the Shapley credit above is computed from |
| `row22_laterality_probing_@@SUFFIX@@.csv` | 22 | per decoder layer: leave-one-pair-out cross-validated and train accuracy of a linear probe predicting left/right from the hidden state alone, `n_pairs` |
| `row23_embedding_geometry_@@SUFFIX@@.csv` | 23 | pairwise cosine similarity between every two organs' "right - left" text-embedding difference vectors (`kind='pair'`), plus one `kind='summary'` row: mean cosine similarity and `coherence_ratio` (1 = one shared left->right direction, 0 = a different, unrelated direction per organ) |
| `row24_attention_head_ablation_@@SUFFIX@@.csv` | 24 | per organ, layer and head: Dice vs. GT and vs. the unablated baseline after zero-ablating that one cross-attention head, plus `call` (does the mask still land on this organ, its mirror, both, or neither) |

Reading guide: in row 14, `moved_to_source_side` is True when patching with the *other* prompt's contribution turns the mask
into the other side. `agreement_with_source_baseline` is the machinery check: for every `whole_state_control` row it should be
about 1 (>= 0.999; patching a whole layer state must reproduce the source prompt's mask, up to floating-point rounding).
"""


ROW_13_CELL = """import traceback

# Row 13 also saves, for every organ and condition, one 2-D plane of the image the model received and of its mask
# (results/row13_slices/<subject>.npz), so that the experiment can be drawn on real scan images.
ROW13_ONLY = None            # e.g. ['amos_0508'] to run a single subject; None = all subjects
(RESULTS_ROOT / 'row13_slices').mkdir(exist_ok=True)
r13_path = RESULTS_ROOT / (f'row13_organ_noise_@@SUFFIX@@.csv' if ROW13_ONLY is None else f'row13_organ_noise_subset_@@SUFFIX@@.csv')
first = True
for si, sid in enumerate(subject_organs, 1):
    if ROW13_ONLY is not None and sid not in ROW13_ONLY:
        continue
    t0 = time.time()
    try:
        person, img_arr, label_vol, organs, ids, prompts = load_subject(sid)
        rows13, slices = ve.run_erasure(predictor, img_arr, label_vol, organs, ids, prompts, seed=si, return_slices=True)
        append_csv(r13_path, tag(rows13, sid, person), first)
        np.savez_compressed(RESULTS_ROOT / 'row13_slices' / f'{sid}.npz', **slices)
        first = False
        print(f'[{si}/{len(subject_organs)}] {sid}: row 13 done in {time.time() - t0:.0f}s')
    except Exception:
        print(f'[{si}/{len(subject_organs)}] {sid}: FAILED'); traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()
print('wrote', r13_path.name, 'and row13_slices/*.npz')
"""


def drop_row12(text):
    """Remove the row-12 pieces from the shared title / summary / outputs text (rows-13-24 notebook)."""
    text = text.replace("PLAN ROWS 12-24: image flip, organ noise", "PLAN ROWS 13-24: organ noise")
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
        cell("markdown", "## 3b. Row 0 -- baseline masks + ground truth, for plotting\n\n"
                         "Not a plan row -- a shared export run once per subject, before any experiment: the "
                         "full-prompt predicted mask and the GT mask for every organ, RLE-encoded so they can be "
                         "decoded and overlaid on the source image later (image + GT + prediction, side by side)."),
        cell("code", sub(ROW_0_BASELINE_CELL)),
        *sec4,
        cell("markdown", "## 5. Rows 14, 15 and 16 -- one custom pass per subject\n\n"
                         "Per subject and left/right pair: baselines, five embedding blends (alpha = -0.5, 0.25, 0.5, 0.75, 1.5 between "
                         "\"left\" and \"right\"), single-layer patches of the query update (all six layers, both directions), patches from a "
                         "layer onward, and whole-state controls -- while recording the cross-attention (row 15) and the left/right distance per layer (row 16)."),
        cell("code", sub(ROWS_14_16_CELL)),
        cell("markdown", "## 6. Row 17 -- voxel-grid occlusion Shapley map\n\n"
                         "Image-region analogue of row 5's exact word-Shapley: tile a padded box around one organ's GT "
                         "region into 8 blocks, noise each coalition's absent blocks, score f_dice_gt, and compute the "
                         "exact Shapley credit per block."),
        cell("code", sub(ROW_17_CELL)),
        cell("markdown", "## 7. Row 18 -- per-layer attention/KL divergence\n\n"
                         "Extends row 15: KL(GT mask || cross-attention density) per decoder layer, plus the same "
                         "quantity after dividing out a generic prompt's attention map (isolates prompt-specific "
                         "attention from wherever VoxTell looks by default)."),
        cell("code", sub(ROW_18_CELL)),
        cell("markdown", "## 8. Row 19 -- gradient-based saliency\n\n"
                         "Cheap, approximate counterpart to row 17 (one backward pass instead of 256 forward passes): "
                         "vanilla-gradient saliency on the input voxels and Grad-CAM on the bottleneck feature map, for "
                         "one patch centred on each organ."),
        cell("code", sub(ROW_19_CELL)),
        cell("markdown", "## 9. Row 20 -- fused spatial feature map (resolves plan row 1)\n\n"
                         "Hooks VoxTellDecoder's per-stage spatial `fusion_features` and checks whether its "
                         "highest-activation region moves with the prompt (matched vs. mismatched side/organ)."),
        cell("code", sub(ROW_20_CELL)),
        cell("markdown", "## 10. Row 21 -- joint image / organ-word / position-word Shapley\n\n"
                         "Three roles (image intact/noised, organ word, position word), 8 coalitions, one Shapley "
                         "table that puts image contribution and word contribution on the same scale."),
        cell("code", sub(ROW_21_CELL)),
        cell("markdown", "## 11. Row 22 -- linear probing: is left/right linearly decodable at each layer?\n\n"
                         "A stricter question than row 16's cosine distance: distance can stay small while a "
                         "linear probe still finds a perfectly usable separating direction. If a layer's probe is "
                         "near-perfect but the final mask still gets the side wrong, that is direct evidence the "
                         "DECODER discards information that was already there."),
        cell("code", sub(ROW_22_CELL)),
        cell("markdown", "## 12. Row 23 -- embedding geometry: one shared \"left -> right\" direction, or one per organ?\n\n"
                         "The classic word-vector-arithmetic test (\"king - man + woman\"), applied to "
                         "\"right X - left X\" across every organ. Pure text-embedding arithmetic -- no image, "
                         "no sliding window, the cheapest row in this notebook."),
        cell("code", sub(ROW_23_CELL)),
        cell("markdown", "## 13. Row 24 -- cross-attention head ablation: which specific head?\n\n"
                         "Zero-ablates one cross-attention head at a time (`voxtell_mechanistic.HeadAblation`) and "
                         "re-predicts the full mask -- row 15/18's whole-layer attention cannot tell \"one head does "
                         "this\" from \"every head does this a little\"; this can. Validated once per notebook run: "
                         "ablating zero heads must reproduce the unablated prediction exactly."),
        cell("code", sub(ROW_24_CELL)),
        cell("markdown", "## 14. Summary and archive"),
        cell("code", sub(drop_row12(SUMMARY_CELL) if rows13_16 else SUMMARY_CELL)),
        cell("markdown", sub(drop_row12(END_MD) if rows13_16 else END_MD)),
    ]
    out = REPO / out_name
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[ok] wrote", out)


TITLE_17_21 = """# VoxTell on {title} -- PLAN ROWS 17-21: image-region attribution suite (Kaggle GPU)

Only the image-region attribution rows -- rows 1-16 are assumed already run separately. Row 0 (not a
numbered plan row) is a shared baseline/GT export, run first, that every row below plots against.

| Row | Experiment | Question |
|---|---|---|
| 17 | Voxel-grid occlusion Shapley map around one organ | Which *region* of the image, if removed, changes the mask most -- the image analogue of the word-Shapley table? |
| 18 | Per-layer KL divergence between cross-attention and the GT mask | Which decoder layer's attention best matches the clinically relevant region? |
| 19 | Vanilla-gradient + Grad-CAM saliency (one patch per organ) | A cheap, approximate counterpart to row 17's exact map |
| 20 | Fused spatial feature map, matched vs. mismatched prompt | Does the model's internal image+text fusion actually move with the prompt, or stay fixed regardless of what's asked? |
| 21 | Joint Shapley over image / organ word / position word | Puts image contribution and word contribution on one common scale |

- Subjects: **{n_subjects}** ({subjects}); organs: **{organs}**; no rotation.
- Section 3 still verifies the custom sliding-window loop against the predictor's own output before
  anything else runs -- rows 17-21 all depend on it (row 19 is the one exception: it needs gradients
  enabled, so it bypasses the loop's `inference_mode` entirely -- see its own cell).
- Output: CSVs + `row17_shapley_maps/`, `row19_saliency_maps/` under `results/`, zipped at the end.
"""

SUMMARY_17_21_CELL = """import shutil
import pandas as pd

pd.set_option('display.width', 200, 'display.max_columns', 30)
R = lambda n: pd.read_csv(RESULTS_ROOT / f'{n}_@@SUFFIX@@.csv')

print('=== Row 0: baseline Dice per organ (sanity check before trusting any row below) ===')
d = R('row0_baseline_masks')
print(d[['organ', 'dice', 'n_pred_voxels', 'n_gt_voxels']].to_string(index=False))

print('\\n=== Row 17: voxel-grid occlusion Shapley -- which block matters most, by organ ===')
d = R('row17_grid_shapley')
print(d.groupby(['organ', 'block_id']).shapley_value.mean().unstack().round(3))

print('\\n=== Row 18: attention-GT KL divergence, by layer (lower = better match) ===')
d = R('row18_attention_kl')
print(d.groupby('layer')[['kl_divergence', 'kl_divergence_vs_generic_normalized']].mean().round(3))

print('\\n=== Row 19: gradient saliency enrichment inside vs. outside the organ (1 = no enrichment) ===')
d = R('row19_gradient_saliency')
print(d.groupby('organ')[['saliency_enrichment', 'cam_enrichment']].mean().round(3))

print('\\n=== Row 20: fused-feature activation IoU with GT, matched vs. mismatched prompt ===')
d = R('row20_fused_activation_overlap')
print(d.groupby('condition').iou_with_gt.mean().round(4))

print('\\n=== Row 21: joint image / organ-word / position-word Shapley (f_dice_gt) ===')
d = R('row21_joint_shapley')
print(d[d.metric == 'f_dice_gt'].groupby('role').shapley_value.mean().round(3))

print('\\n=== Row 21: raw Dice score per role-combination (the numbers the Shapley credit above comes from) ===')
dc = R('row21_coalition_dice')
print(dc[['roles_present', 'dice_gt', 'dice_self']].to_string(index=False))

zip_path = WORKING_DIR / 'voxtell_@@SUFFIX@@_rows17_21_results'
shutil.make_archive(str(zip_path), 'zip', str(RESULTS_ROOT))
print('\\nArchive:', str(zip_path) + '.zip')
"""

END_MD_17_21 = """## Outputs (all under `results/`)

| File | Row | Content |
|---|---|---|
| `row0_baseline_masks_@@SUFFIX@@.csv` | - | per organ: full-prompt predicted mask and GT mask, RLE-encoded (`decode_mask_rle`), plus centroids and Dice -- the shared source for plotting image + GT + prediction overlays |
| `row17_grid_shapley_@@SUFFIX@@.csv` | 17 | per organ and voxel-grid block: exact Shapley credit for that block's image content (the image analogue of the word-Shapley table) |
| `row17_shapley_maps/<subject>.npz` | 17 | per organ: the same Shapley credit reassembled into a full voxel-shaped map, for plotting against row 0's image/GT |
| `row18_attention_kl_@@SUFFIX@@.csv` | 18 | per organ and layer: KL divergence between the GT mask and the cross-attention map, raw and generic-prompt-normalized |
| `row18_attention_maps/<subject>.npz` | 18 | per organ: last-layer and mean-of-layers attention density, downsampled, for plotting against row 0's image/GT |
| `row19_gradient_saliency_@@SUFFIX@@.csv` | 19 | per organ: vanilla-gradient and Grad-CAM saliency, mean inside vs. outside the organ, and their ratio |
| `row19_saliency_maps/<subject>.npz` | 19 | per organ: the vanilla-gradient saliency map and the upsampled Grad-CAM map, voxel-shaped, for plotting |
| `row20_fused_activation_overlap_@@SUFFIX@@.csv` | 20 | per organ and condition (matched / mismatched_side / mismatched_organ): IoU of the fused spatial feature map's hot region with the GT mask |
| `row20_fused_maps/<subject>.npz` | 20 | per organ x condition: the fused-feature map itself, upsampled to the crop frame and downsampled for storage, for plotting |
| `row21_joint_shapley_@@SUFFIX@@.csv` | 21 | per organ, role (image / organ_word / position_word) and metric: Shapley credit on one common scale |
| `row21_coalition_dice_@@SUFFIX@@.csv` | 21 | per organ and role-combination (`roles_present`): the literal Dice score for that combination (e.g. image only, no words at all), against ground truth (`dice_gt`) and against the full-coalition baseline (`dice_self`) -- the raw numbers the Shapley credit above is computed from |
"""


ROWS_22_24_TITLE_ROWS = (
    "| 22 | Linear probe for left/right at every decoder layer | Is laterality linearly DECODABLE in the hidden "
    "state, independent of what mask comes out -- stricter than row 16's distance |\n"
    "| 23 | \"right - left\" embedding geometry across organs | Is there one reusable \"left->right\" direction, "
    "or a different, unrelated one per organ (is \"left\" a composable concept)? |\n"
    "| 24 | Zero-ablate one cross-attention head at a time | Which SPECIFIC head, not just which layer, is "
    "causally responsible for laterality |\n"
)

ROWS_22_24_SUMMARY_APPEND = """
print('\\n=== Row 22: is left/right linearly decodable at each layer? (leave-one-pair-out CV accuracy) ===')
d = R('row22_laterality_probing')
print(d.groupby('layer')[['cv_accuracy', 'train_accuracy']].mean().round(3))

print('\\n=== Row 23: is there one consistent "left -> right" direction across organs? ===')
d = R('row23_embedding_geometry')
s = d[d.kind == 'summary']
print(s[['mean_cosine_similarity', 'coherence_ratio', 'n_organs']].to_string(index=False))

print('\\n=== Row 24: cross-attention head ablation -- dice vs. GT after zeroing each head, by layer ===')
d = R('row24_attention_head_ablation')
print(d.groupby(['layer', 'head'])[['dice_vs_gt', 'dice_vs_baseline']].mean().round(3))
"""

ROWS_22_24_END_MD_ROWS = (
    "| `row22_laterality_probing_@@SUFFIX@@.csv` | 22 | per decoder layer: leave-one-pair-out cross-validated and "
    "train accuracy of a linear probe predicting left/right from the hidden state alone, `n_pairs` |\n"
    "| `row23_embedding_geometry_@@SUFFIX@@.csv` | 23 | pairwise cosine similarity between every two organs' "
    "\"right - left\" text-embedding difference vectors (`kind='pair'`), plus one `kind='summary'` row: mean "
    "cosine similarity and `coherence_ratio` |\n"
    "| `row24_attention_head_ablation_@@SUFFIX@@.csv` | 24 | per organ, layer and head: Dice vs. GT and vs. the "
    "unablated baseline after zero-ablating that one cross-attention head, plus `call` |\n"
)


def build_17_21(base_nb, title, dataset, subjects, cores, suffix, out_name, include_22_24=False):
    """Rows 17-21 only (plus the shared row-0 baseline/GT export) -- for when rows 1-16 are
    already covered by other notebooks and only the image-region attribution suite is needed.
    include_22_24=True extends this to rows 17-24, adding the model-internals suite (probing,
    embedding geometry, head ablation) -- those three need no spatial maps/row0-registration of
    their own, so they bolt on cleanly after row 21 without changing anything above it."""
    nb = json.loads((REPO / base_nb).read_text(encoding="utf-8"))
    nb["cells"] = nb["cells"][:16]
    subj_str = ", ".join(sid for _, sid in subjects)
    organs = ", ".join(cores) if cores else "every bilateral organ"
    title_md = TITLE_17_21.format(title=title, n_subjects=len(subjects), subjects=subj_str, organs=organs)
    if include_22_24:
        title_md = title_md.replace("PLAN ROWS 17-21: image-region attribution suite",
                                     "PLAN ROWS 17-24: image-region attribution + model-internals suite")
        title_md = title_md.replace(
            "| 21 | Joint Shapley over image / organ word / position word | Puts image contribution and word contribution on one common scale |\n",
            "| 21 | Joint Shapley over image / organ word / position word | Puts image contribution and word contribution on one common scale |\n"
            + ROWS_22_24_TITLE_ROWS)
    nb["cells"][0] = cell("markdown", title_md)
    sub = lambda t: (t.replace("@@SUFFIX@@", suffix).replace("@@DATASET@@", dataset))
    mech = json.dumps((SCRIPTS / "voxtell_mechanistic.py").read_text(encoding="utf-8"))
    exp = json.dumps((SCRIPTS / "voxtell_experiments.py").read_text(encoding="utf-8"))
    rows_22_24_cells = [
        cell("markdown", "## 9. Row 22 -- linear probing: is left/right linearly decodable at each layer?\n\n"
                         "A stricter question than row 16's cosine distance: distance can stay small while a "
                         "linear probe still finds a perfectly usable separating direction. If a layer's probe is "
                         "near-perfect but the final mask still gets the side wrong, that is direct evidence the "
                         "DECODER discards information that was already there."),
        cell("code", sub(ROW_22_CELL)),
        cell("markdown", "## 10. Row 23 -- embedding geometry: one shared \"left -> right\" direction, or one per organ?\n\n"
                         "The classic word-vector-arithmetic test (\"king - man + woman\"), applied to "
                         "\"right X - left X\" across every organ. Pure text-embedding arithmetic -- no image, "
                         "no sliding window, the cheapest row in this notebook."),
        cell("code", sub(ROW_23_CELL)),
        cell("markdown", "## 11. Row 24 -- cross-attention head ablation: which specific head?\n\n"
                         "Zero-ablates one cross-attention head at a time (`voxtell_mechanistic.HeadAblation`) and "
                         "re-predicts the full mask -- row 15/18's whole-layer attention cannot tell \"one head does "
                         "this\" from \"every head does this a little\"; this can. Validated once per notebook run: "
                         "ablating zero heads must reproduce the unablated prediction exactly."),
        cell("code", sub(ROW_24_CELL)),
    ] if include_22_24 else []
    summary_cell_text = sub(SUMMARY_17_21_CELL)
    end_md_text = sub(END_MD_17_21)
    if include_22_24:
        summary_cell_text = summary_cell_text.replace("\nzip_path =", ROWS_22_24_SUMMARY_APPEND + "\nzip_path =")
        summary_cell_text = summary_cell_text.replace("rows17_21_results", "rows17_24_results")
        anchor = ("| `row21_coalition_dice_@@SUFFIX@@.csv` | 21 | per organ and role-combination (`roles_present`): "
                  "the literal Dice score for that combination (e.g. image only, no words at all), against ground "
                  "truth (`dice_gt`) and against the full-coalition baseline (`dice_self`) -- the raw numbers the "
                  "Shapley credit above is computed from |\n").replace("@@SUFFIX@@", suffix)
        end_md_text = end_md_text.replace(anchor, anchor + sub(ROWS_22_24_END_MD_ROWS))
    summary_heading = "## 12. Summary and archive" if include_22_24 else "## 9. Summary and archive"
    nb["cells"] += [
        cell("markdown", "## 1. Load the experiment code"),
        cell("code", SRC_CELL.replace("@@MECH@@", repr(mech)).replace("@@EXP@@", repr(exp))),
        cell("markdown", "## 2. Subjects, organs and helpers"),
        cell("code", sub(CONFIG_CELL).replace("@@SUBJECTS@@", ",\n    ".join(f'("{p}", "{s}")' for p, s in subjects))
             .replace("@@CORES@@", repr(cores))),
        cell("markdown", "## 3. Check: the custom sliding-window loop must match the predictor"),
        cell("code", VALIDATE_CELL),
        cell("markdown", "## 3b. Row 0 -- baseline masks + ground truth, for plotting\n\n"
                         "Not a plan row -- a shared export run once per subject, before any experiment: the "
                         "full-prompt predicted mask and the GT mask for every organ, RLE-encoded so they can be "
                         "decoded and overlaid on the source image later (image + GT + prediction, side by side)."),
        cell("code", sub(ROW_0_BASELINE_CELL)),
        cell("markdown", "## 4. Row 17 -- voxel-grid occlusion Shapley map\n\n"
                         "Image-region analogue of row 5's exact word-Shapley: tile a padded box around one organ's GT "
                         "region into 8 blocks, noise each coalition's absent blocks, score f_dice_gt, and compute the "
                         "exact Shapley credit per block."),
        cell("code", sub(ROW_17_CELL)),
        cell("markdown", "## 5. Row 18 -- per-layer attention/KL divergence\n\n"
                         "Reuses the same cross-attention pipeline as row 15: KL(GT mask || cross-attention density) per "
                         "decoder layer, plus the same quantity after dividing out a generic prompt's attention map "
                         "(isolates prompt-specific attention from wherever VoxTell looks by default)."),
        cell("code", sub(ROW_18_CELL)),
        cell("markdown", "## 6. Row 19 -- gradient-based saliency\n\n"
                         "Cheap, approximate counterpart to row 17 (one backward pass instead of 256 forward passes): "
                         "vanilla-gradient saliency on the input voxels and Grad-CAM on the bottleneck feature map, for "
                         "one patch centred on each organ."),
        cell("code", sub(ROW_19_CELL)),
        cell("markdown", "## 7. Row 20 -- fused spatial feature map\n\n"
                         "Hooks VoxTellDecoder's per-stage spatial `fusion_features` and checks whether its "
                         "highest-activation region moves with the prompt (matched vs. mismatched side/organ)."),
        cell("code", sub(ROW_20_CELL)),
        cell("markdown", "## 8. Row 21 -- joint image / organ-word / position-word Shapley\n\n"
                         "Three roles (image intact/noised, organ word, position word), 8 coalitions, one Shapley "
                         "table that puts image contribution and word contribution on the same scale."),
        cell("code", sub(ROW_21_CELL)),
        *rows_22_24_cells,
        cell("markdown", summary_heading),
        cell("code", summary_cell_text),
        cell("markdown", end_md_text),
    ]
    out = REPO / out_name
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[ok] wrote", out)


if __name__ == "__main__":
    build("voxtell-amos-groundedness-investigation-kaggle.ipynb", "AMOS22 MRI", "amos22_mri", AMOS_SUBJECTS, None,
          "amos22_mri", "voxtell-amos-mechanistic-experiments-kaggle.ipynb")
    build("voxtell-oasis-groundedness-investigation-batch1-kaggle.ipynb", "Neurite-OASIS", "neurite_oasis", OASIS_ONE_SUBJECT,
          OASIS_RICH_CORES, "neurite_oasis", "voxtell-oasis-mechanistic-experiments-kaggle.ipynb")
    build("voxtell-amos-groundedness-investigation-kaggle.ipynb", "AMOS22 MRI", "amos22_mri", AMOS_SUBJECTS, None,
          "amos22_mri", "voxtell-amos-mechanistic-rows13-16-kaggle.ipynb", rows13_16=True)
    build("voxtell-oasis-groundedness-investigation-batch1-kaggle.ipynb", "Neurite-OASIS", "neurite_oasis", OASIS_ONE_SUBJECT,
          OASIS_RICH_CORES, "neurite_oasis", "voxtell-oasis-mechanistic-rows13-24-kaggle.ipynb", rows13_16=True)
    # Standalone rows 17-21 notebook: same subject/organs, so it can be run independently of (or in
    # parallel with) the rows13-24 notebook above -- its row17-21 cells are identical either way
    # (rows 22-24 are model-internals/circuit experiments, not image-region attribution, so they are
    # deliberately NOT included in this image-attribution-only standalone notebook).
    build_17_21("voxtell-oasis-groundedness-investigation-batch1-kaggle.ipynb", "Neurite-OASIS", "neurite_oasis",
                OASIS_ONE_SUBJECT, OASIS_RICH_CORES, "neurite_oasis", "voxtell-oasis-rows17-21-kaggle.ipynb")
    # Standalone AMOS rows 17-24 notebook: 3 subjects (one per annotator), every bilateral organ
    # (kidney + adrenal gland, left/right) -- includes the model-internals suite (22-24) too, unlike
    # the OASIS rows17-21 notebook above, since this is the first and only AMOS rows17+ notebook.
    build_17_21("voxtell-amos-groundedness-investigation-kaggle.ipynb", "AMOS22 MRI", "amos22_mri",
                AMOS_THREE_SUBJECTS, None, "amos22_mri", "voxtell-amos-rows17-24-kaggle.ipynb", include_22_24=True)
