"""
Experiments for plan rows 12-16 and 17-21 (all take a VoxTellPredictor-like object and plain numpy
arrays, so they can be tested without a GPU).

  row 12  run_flip                image left-right flip, prompt unchanged
  row 13  run_erasure             organ filled with noise / both organs filled with noise / noise image
  row 14  run_interventions       embedding blend and per-layer patching between "left X" and "right X"
  row 15  (same run)              cross-attention of every prompt over every organ region, per decoder layer
  row 16  (same run)              distance between the left and right query states at every decoder layer
  row 17  run_grid_occlusion      voxel-grid occlusion Shapley/SHAP map around one organ
  row 18  run_attention_kl        per-layer KL divergence between cross-attention and the GT organ mask
  row 19  run_gradient_saliency   vanilla-gradient + Grad-CAM saliency, one patch around the organ
  row 20  run_activation_overlap  fused spatial feature map overlap, matched vs. mismatched prompt
  row 21  run_joint_image_word_shapley   image (intact/noised) x organ-word x position-word, 3-role Shapley
  row 22  run_laterality_probing  linear probe per decoder layer: is left/right linearly decodable, not just different?
  row 23  run_embedding_geometry  is there one consistent "left -> right" direction across organs, or a different one per organ?
  row 24  run_attention_head_ablation   zero-ablate one cross-attention head at a time -- which head, not just which layer

Arrays: `img_arr` is (1, X, Y, Z) in the RAS-reoriented grid used by the notebooks, `label_vol` is (X, Y, Z).
`rl_axis` is the right-left axis of the spatial arrays (2 in the notebooks).
"""

import base64
import itertools
import zlib

import numpy as np
import torch

import voxtell_mechanistic as vm


# ------------------------------------------------------------------------------------------------
# RLE mask encode/decode (duplicated, not imported, from build_groundedness_investigation_notebooks.py
# -- same self-contained-module convention as the Shapley helpers above). Used so every row's CSV can
# carry the full predicted/GT mask, not just summary numbers, for later plotting (image + GT + mask
# overlays) exactly like the rotation-sweep CSVs already do.
# ------------------------------------------------------------------------------------------------
def encode_mask_rle(mask):
    """Run-length-encode a boolean mask (C-order flatten), zlib-compress, base64-encode."""
    flat = np.asarray(mask, dtype=bool).ravel(order='C')
    change_idx = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], change_idx, [flat.size]))
    runs = np.diff(boundaries).astype(np.uint32)
    if flat.size and flat[0]:
        runs = np.concatenate(([0], runs)).astype(np.uint32)
    raw = zlib.compress(runs.tobytes(), level=9)
    return base64.b64encode(raw).decode('ascii')


def decode_mask_rle(encoded, shape):
    runs = np.frombuffer(zlib.decompress(base64.b64decode(encoded)), dtype=np.uint32)
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    pos, val = 0, False
    for r in runs:
        if val:
            flat[pos:pos + r] = True
        pos += r
        val = not val
    return flat.reshape(shape)


# ------------------------------------------------------------------------------------------------
# shared Shapley machinery (duplicated, not imported, from build_groundedness_investigation_notebooks.py's
# rows 5-7 -- kept self-contained here since this module, like voxtell_mechanistic.py, is embedded
# verbatim into the mechanistic notebooks; see that file's `shapley_values`/`all_role_combos` for the
# word-attribution use of the exact same math)
# ------------------------------------------------------------------------------------------------
def all_role_combos(roles):
    """All 2^n present/absent combinations. Returns list of (frozenset(present roles), present_dict)."""
    combos = []
    for bits in itertools.product([True, False], repeat=len(roles)):
        present = dict(zip(roles, bits))
        key = frozenset(r for r in roles if present[r])
        combos.append((key, present))
    return combos


def shapley_values(v, roles):
    """Exact Shapley values. v: dict[frozenset(present roles)] -> payoff (float). roles: full player set N."""
    import math
    n = len(roles)
    phi = {r: 0.0 for r in roles}
    for i in roles:
        others = [r for r in roles if r != i]
        for bits in itertools.product([True, False], repeat=len(others)):
            s = frozenset(r for r, b in zip(others, bits) if b)
            weight = math.factorial(len(s)) * math.factorial(n - len(s) - 1) / math.factorial(n)
            phi[i] += weight * (v[s | {i}] - v[s])
    return phi


# ------------------------------------------------------------------------------------------------
def mirror(organ):
    """'left kidney' <-> 'right kidney'."""
    side, core = organ.split(None, 1)
    return ("right " if side == "left" else "left ") + core


def centroid(mask):
    idx = np.nonzero(mask)
    return np.array([c.mean() for c in idx]) if len(idx[0]) else None


def _to_full_masks(logits, bbox, orig_shape):
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
    with torch.no_grad():
        cropped = torch.cat([(torch.sigmoid(logits[i:i + 8].float()) > 0.5).to("cpu", torch.uint8)
                             for i in range(0, logits.shape[0], 8)]).numpy()
    full = np.zeros((cropped.shape[0], *orig_shape), dtype=np.uint8)
    return insert_crop_into_image(full, cropped, bbox).astype(bool)


def predict_masks(predictor, img_arr, prompts, chunk=32):
    """Same steps as the notebooks: preprocess, embed, sliding window, threshold, put back into the full volume."""
    data, bbox, orig_shape = predictor.preprocess(img_arr)
    emb = predictor.embed_text_prompts(list(prompts))
    parts = []
    for s in range(0, emb.shape[1], chunk):
        parts.append(predictor.predict_sliding_window_return_logits(data, emb[:, s:s + chunk]).to("cpu"))
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return _to_full_masks(torch.cat(parts, dim=0), bbox, orig_shape)


# ------------------------------------------------------------------------------------------------
# row 12 -- image left-right flip
# ------------------------------------------------------------------------------------------------
def run_flip(predictor, img_arr, label_vol, organs, ids, prompts, rl_axis=2):
    """Flip the image and its labels along the right-left axis and keep every prompt unchanged.
    call 'a' = the mask covers the organ that the prompt names (follows the anatomy / the label);
    call 'b' = the mask covers the organ on the other side (follows the side of the image)."""
    flipped_img = np.ascontiguousarray(np.flip(img_arr, axis=rl_axis + 1))
    flipped_lab = np.ascontiguousarray(np.flip(label_vol, axis=rl_axis))
    rows = []
    for cond, img, lab in (("original", img_arr, label_vol), ("flipped", flipped_img, flipped_lab)):
        masks = predict_masks(predictor, img, [prompts[o] for o in organs])
        for k, o in enumerate(organs):
            sc = vm.score_masks(masks[k], lab == ids[o], lab == ids[mirror(o)])
            rows.append({"organ": o, "condition": cond, "prompt": prompts[o], "n_pred_voxels": sc["n_pred_voxels"],
                         "cov_named_organ": sc["cov_a"], "cov_other_side_organ": sc["cov_b"],
                         "dice_named_organ": sc["dice_a"], "dice_other_side_organ": sc["dice_b"],
                         "call": {"a": "follows_named_organ", "b": "follows_image_side"}.get(sc["call"], sc["call"])})
    return rows


# ------------------------------------------------------------------------------------------------
# row 13 -- organ replaced by noise / noise image
# ------------------------------------------------------------------------------------------------
def noise_region(img_arr, region, rng):
    """Replace the voxels of `region` by Gaussian noise with the mean and spread of the image's non-zero voxels."""
    out = img_arr.copy()
    vals = img_arr[img_arr != 0]
    out[0][region] = rng.normal(float(vals.mean()), float(vals.std()), size=int(region.sum()))
    return out


def noise_image(img_arr, rng):
    vals = img_arr[img_arr != 0]
    return rng.normal(float(vals.mean()), float(vals.std()), size=img_arr.shape).astype(img_arr.dtype)


def _region_metrics(mask, region, orig_voxels):
    n = int(mask.sum())
    inter = int(np.logical_and(mask, region).sum())
    cm, cr = centroid(mask), centroid(region)
    return {"n_pred_voxels": n, "size_vs_original": round(n / orig_voxels, 4) if orig_voxels else float("nan"),
            "cov_of_noised_region": round(inter / max(int(region.sum()), 1), 4),
            "frac_of_mask_in_region": round(inter / n, 4) if n else 0.0,
            "centroid_offset_vox": round(float(np.linalg.norm(cm - cr)), 1) if cm is not None else float("nan"),
            "empty": n == 0}


def run_erasure(predictor, img_arr, label_vol, organs, ids, prompts, seed=0, return_slices=False):
    """Row 13. With return_slices=True it returns (rows, slices): for every prompt organ and condition one 2-D plane of the image
    the model received (float16) and of the predicted mask (uint8), plus the true organs on the same plane, so that the
    experiment can be drawn on real scan images. The plane cuts the volume across its thinnest axis at the named organ's centroid."""
    rng = np.random.default_rng(seed)
    regions = {o: label_vol == ids[o] for o in organs}
    orig = predict_masks(predictor, img_arr, [prompts[o] for o in organs])
    orig_n = {o: int(orig[k].sum()) for k, o in enumerate(organs)}
    rows, slices = [], {}
    axis = int(np.argmin(label_vol.shape))                  # through-plane axis of the scan
    plane = {o: int(round(centroid(regions[o])[axis])) if regions[o].any() else label_vol.shape[axis] // 2 for o in organs}

    def add(cond, o, mask, img):
        rows.append({"organ": o, "condition": cond, "prompt": prompts[o],
                     **_region_metrics(mask, regions[o], orig_n[o])})
        if return_slices:
            i = plane[o]
            slices[f"{o}|{cond}|image"] = np.take(img[0], i, axis=axis).astype(np.float16)
            slices[f"{o}|{cond}|mask"] = np.take(mask, i, axis=axis).astype(np.uint8)

    for k, o in enumerate(organs):
        add("original", o, orig[k], img_arr)
    for o in organs:                                        # only the named organ replaced by noise
        img = noise_region(img_arr, regions[o], rng)
        add("noise_target", o, predict_masks(predictor, img, [prompts[o]])[0], img)
    for o in [x for x in organs if x.startswith("left") and mirror(x) in organs]:   # both organs of the pair replaced by noise
        img = noise_region(noise_region(img_arr, regions[o], rng), regions[mirror(o)], rng)
        m = predict_masks(predictor, img, [prompts[o], prompts[mirror(o)]])
        add("noise_pair", o, m[0], img)
        add("noise_pair", mirror(o), m[1], img)
    nimg = noise_image(img_arr, rng)
    noisy = predict_masks(predictor, nimg, [prompts[o] for o in organs])
    for k, o in enumerate(organs):
        add("noise_image", o, noisy[k], nimg)
    if return_slices:
        for o in organs:
            slices[f"{o}|axis"] = np.array(axis)
            slices[f"{o}|index"] = np.array(plane[o])
            for p_ in organs:
                slices[f"{o}|true|{p_}"] = np.take(regions[p_], plane[o], axis=axis).astype(np.uint8)
        return rows, slices
    return rows


# ------------------------------------------------------------------------------------------------
# rows 14-16 -- embedding intervention, cross-attention, layer-by-layer distance
# ------------------------------------------------------------------------------------------------
class LayerDistanceRecorder:
    """Row 16: cosine distance and relative L2 distance between two prompts' query states, per decoder layer
    (index 0 = the input state, 1..6 = after each layer), averaged over the patches that contain the organs
    and over all patches."""

    def __init__(self, pair_defs, union_crop):
        self.pair_defs, self.union_crop = pair_defs, union_crop      # [(name, row_i, row_j)]
        self.union_padded = None
        self.cos, self.l2, self.flags = [], [], []

    def reset(self):
        self.cos, self.l2, self.flags = [], [], []

    def on_patch(self, slicer, states, attns, ctx):
        if self.union_padded is None:
            self.union_padded = np.zeros(ctx["padded_shape"], bool)
            self.union_padded[ctx["revert"][1:]] = self.union_crop
        self.flags.append(bool(self.union_padded[slicer[1:]].any()))
        c = np.zeros((len(self.pair_defs), len(states)))
        d = np.zeros_like(c)
        for pi, (_, i, j) in enumerate(self.pair_defs):
            for l, st in enumerate(states):
                a, b = st[i, 0].float(), st[j, 0].float()
                c[pi, l] = 1 - float(torch.nn.functional.cosine_similarity(a, b, dim=0))
                d[pi, l] = float(torch.linalg.norm(a - b) / (0.5 * (torch.linalg.norm(a) + torch.linalg.norm(b)) + 1e-8))
        self.cos.append(c)
        self.l2.append(d)

    def rows(self):
        cos, l2, flags = np.array(self.cos), np.array(self.l2), np.array(self.flags)
        out = []
        for scope, sel in (("organ_patches", flags), ("all_patches", np.ones_like(flags))):
            if not sel.any():
                continue
            for pi, (name, _, _) in enumerate(self.pair_defs):
                for l in range(cos.shape[2]):
                    out.append({"pair": name, "scope": scope, "layer": l, "n_patches": int(sel.sum()),
                                "cosine_distance": round(float(cos[sel, pi, l].mean()), 6),
                                "relative_l2_distance": round(float(l2[sel, pi, l].mean()), 6)})
        return out


def _attention_rows(attn, organs, regions_crop, rl_axis):
    """Row 15. attn: (L, n_prompts, X, Y, Z) attention density (mean 1) in crop space; regions_crop: {organ: bool}."""
    dens_shape = attn.shape[2:]
    coords = torch.arange(dens_shape[rl_axis], dtype=torch.float32, device=attn.device)
    view = [1, 1, 1]
    view[rl_axis] = -1
    coords = coords.view(*view)
    rows, com_rows = [], []
    for l in range(attn.shape[0]):
        for p, prompt_organ in enumerate(organs):
            d = attn[l, p].float()
            total, mean_all = d.sum(), d.mean()
            for region_organ in organs:
                r = torch.from_numpy(regions_crop[region_organ]).to(d.device)
                if not bool(r.any()):
                    continue
                rows.append({"layer": l + 1, "prompt_organ": prompt_organ, "region_organ": region_organ,
                             "is_matched": region_organ == prompt_organ, "is_wrong_side": region_organ == mirror(prompt_organ),
                             "enrichment": round(float(d[r].mean() / mean_all), 4),
                             "mass_fraction": round(float(d[r].sum() / total), 6)})
            com = float((d * coords).sum() / total)
            named = centroid(regions_crop[prompt_organ])
            other = centroid(regions_crop[mirror(prompt_organ)]) if mirror(prompt_organ) in regions_crop else None
            if named is not None and other is not None:
                com_rows.append({"layer": l + 1, "prompt_organ": prompt_organ, "attention_centre_rl": round(com, 2),
                                 "named_organ_centre_rl": round(float(named[rl_axis]), 2),
                                 "other_side_organ_centre_rl": round(float(other[rl_axis]), 2),
                                 "closer_to_named_organ": abs(com - named[rl_axis]) < abs(com - other[rl_axis])})
    return rows, com_rows


_OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)      # older torch only has the cuda one


def _run_chunked(predictor, data, emb_all, outputs, bbox, orig_shape, attn_factory, attn_rows, on_patch, reset, chunk=12):
    """Run the (many) outputs in chunks so the logits / decoder activations fit on the GPU. The attention and the layer
    recorder only need one pass, so they ride on the first chunk. A chunk that runs out of memory is retried with half the
    size. Returns (bool masks (n_out, X, Y, Z) in the original grid, attention or None)."""
    import gc
    masks, attn, i = [], None, 0
    while i < len(outputs):
        part, first = outputs[i:i + chunk], i == 0
        oom = False
        try:
            logits, a = vm.run_outputs(predictor, data, emb_all, part,
                                       attn_acc_factory=attn_factory if first else None,
                                       attn_rows=attn_rows if first else None, on_patch=on_patch if first else None)
        except _OOM:
            oom = True
        if oom:                                   # clean up outside the except block so the failed frames are released
            gc.collect(); torch.cuda.empty_cache()
            if chunk == 1:
                raise RuntimeError("out of GPU memory even with one output per pass")
            chunk = max(1, chunk // 2)
            if first:
                reset()
            continue
        masks.append(_to_full_masks(logits, bbox, orig_shape))
        del logits
        if first:
            attn = a.cpu() if a is not None else None
        i += len(part)
        gc.collect(); torch.cuda.empty_cache()
    return np.concatenate(masks), attn


def run_interventions(predictor, img_arr, label_vol, organs, ids, prompts, rl_axis=2,
                      alphas=(-0.5, 0.25, 0.5, 0.75, 1.5), single_layers=range(6), from_layers=(0, 3),
                      state_layers=(0, 5), map_downsample=4, out_chunk=12):
    """Returns dict(row14=[...], row15=[...], row15_centre=[...], row16=[...], attention_maps={name: float16 array})."""
    pairs = [(o, mirror(o)) for o in organs if o.startswith("left") and mirror(o) in organs]
    data, bbox, orig_shape = predictor.preprocess(img_arr)
    emb_base = predictor.embed_text_prompts([prompts[o] for o in organs])          # (1, n, D)
    row_of = {o: i for i, o in enumerate(organs)}
    extra = []
    blend_row = {}
    for L, R in pairs:
        for a in alphas:
            blend_row[(L, a)] = len(organs) + len(extra)
            extra.append((1 - a) * emb_base[0, row_of[L]] + a * emb_base[0, row_of[R]])
    emb_all = torch.cat([emb_base, torch.stack(extra)[None]], dim=1) if extra else emb_base

    outputs, meta = [], []

    def add(name, row, ops, m):
        outputs.append({"name": name, "row": row, "ops": ops})
        meta.append(m)

    for L, R in pairs:
        core = L.split(None, 1)[1]
        for o in (L, R):
            add(f"{o}|baseline", row_of[o], [], dict(pair=core, variant="baseline", dst=o.split()[0], src="", alpha=np.nan, layer=-1))
        for a in alphas:
            add(f"{L}|blend|{a}", blend_row[(L, a)], [], dict(pair=core, variant="blend", dst="left", src="right", alpha=a, layer=-1))
        for dst, src in ((L, R), (R, L)):
            d, s = dst.split()[0], src.split()[0]
            for k in single_layers:
                add(f"{dst}|update|{k}", row_of[dst], [{"layer": k, "mode": "update", "src": row_of[src]}],
                    dict(pair=core, variant="single_layer_update", dst=d, src=s, alpha=np.nan, layer=k))
            for k in from_layers:
                add(f"{dst}|from|{k}", row_of[dst], [{"layer": j, "mode": "update", "src": row_of[src]} for j in range(k, 6)],
                    dict(pair=core, variant="updates_from_layer", dst=d, src=s, alpha=np.nan, layer=k))
            for k in state_layers:
                add(f"{dst}|state|{k}", row_of[dst], [{"layer": k, "mode": "state", "src": row_of[src]}],
                    dict(pair=core, variant="whole_state_control", dst=d, src=s, alpha=np.nan, layer=k))

    # ground truth in crop space (for the attention statistics and the recorder)
    crop_sl = tuple(slice(int(b[0]), int(b[1])) for b in bbox)
    label_crop = label_vol[crop_sl]
    regions_crop = {o: label_crop == ids[o] for o in organs}
    union_crop = np.zeros(label_crop.shape, bool)
    for r in regions_crop.values():
        union_crop |= r
    ref_pairs = []
    for L, R in pairs:
        others = [o for o in organs if o.split(None, 1)[1] != L.split(None, 1)[1] and o.startswith("left")]
        ref_pairs.append((f"{L.split(None, 1)[1]}: left vs right", row_of[L], row_of[R]))
        if others:
            ref_pairs.append((f"{L.split(None, 1)[1]}: {L} vs {others[0]} (reference)", row_of[L], row_of[others[0]]))
    recorder = LayerDistanceRecorder(ref_pairs, union_crop)

    masks, attn = _run_chunked(
        predictor, data, emb_all, outputs, bbox, orig_shape,
        attn_factory=lambda shape, n_layers, n_rows: vm.AttnAccumulator(shape, n_layers, n_rows, predictor.patch_size, predictor.device),
        attn_rows=len(organs), on_patch=recorder.on_patch, reset=recorder.reset, chunk=out_chunk)
    base_idx = {(m["pair"], m["dst"]): k for k, m in enumerate(meta) if m["variant"] == "baseline"}
    rows14 = []
    for k, m in enumerate(meta):
        L = f"left {m['pair']}"
        R = f"right {m['pair']}"
        sc = vm.score_masks(masks[k], label_vol == ids[L], label_vol == ids[R])
        call = {"a": "left", "b": "right"}.get(sc["call"], sc["call"])
        moved = bool((m["dst"] == "left" and call == "right") or (m["dst"] == "right" and call == "left")) if m["variant"] != "blend" else np.nan
        agree = float((masks[k] == masks[base_idx[(m["pair"], m["src"])]]).mean()) if m["variant"] == "whole_state_control" else np.nan
        rows14.append({**m, "agreement_with_source_baseline": round(agree, 6) if agree == agree else agree,
                       "n_pred_voxels": sc["n_pred_voxels"], "cov_left_organ": sc["cov_a"], "cov_right_organ": sc["cov_b"],
                       "dice_left_organ": sc["dice_a"], "dice_right_organ": sc["dice_b"], "call": call, "moved_to_source_side": moved})
    rows15, com15 = _attention_rows(attn, organs, regions_crop, rl_axis)
    maps = {}
    if map_downsample:
        pool = torch.nn.AvgPool3d(map_downsample, ceil_mode=True)
        for p, o in enumerate(organs):
            last = pool(attn[-1, p][None, None].float())[0, 0]
            mean = pool(attn[:, p].float().mean(0)[None, None])[0, 0]
            maps[f"{o}__last_layer"] = last.half().cpu().numpy()
            maps[f"{o}__mean_of_layers"] = mean.half().cpu().numpy()
    return {"row14": rows14, "row15": rows15, "row15_centre": com15, "row16": recorder.rows(), "attention_maps": maps}


def validate_custom_loop(predictor, img_arr, prompts):
    """The custom sliding-window loop must reproduce the predictor's own masks; returns the voxel agreement."""
    data, bbox, orig_shape = predictor.preprocess(img_arr)
    emb = predictor.embed_text_prompts(list(prompts))
    ref = predictor.predict_sliding_window_return_logits(data, emb)
    mine, _ = vm.run_outputs(predictor, data, emb, [{"name": str(i), "row": i, "ops": []} for i in range(emb.shape[1])])
    return float(((ref.float() > 0) == (mine.float() > 0)).float().mean())


# ------------------------------------------------------------------------------------------------
# row 17 -- voxel-grid occlusion Shapley map
# ------------------------------------------------------------------------------------------------
def _block_mask(shape, block_slicer):
    m = np.zeros(shape, dtype=bool)
    m[block_slicer] = True
    return m


def _bbox_str(block_slicer):
    return "x".join(f"{s.start}:{s.stop}" for s in block_slicer)


def run_grid_occlusion(predictor, img_arr, label_vol, organ, ids, prompt, grid=2, pad=10, seed=0, return_map=False):
    """Row 17. Tile a padded box around `organ`'s ground-truth region into `grid`**3 blocks (default
    8 blocks -> 2**8=256 coalitions, the same affordability argument already used for row 5's exact
    word-Shapley: 2**8 is cheap to enumerate exactly). Each block is independently noised/kept
    (reusing `noise_region`), `f_dice_gt` is scored per coalition, and `shapley_values` (same math as
    the word-attribution rows) turns that into one Shapley credit per block -- the image-region
    analogue of row 5's word table. Returns rows with columns: organ, block_id, block_bbox, shapley_value."""
    rng = np.random.default_rng(seed)
    region = label_vol == ids[organ]
    if not region.any():
        return []
    idx = np.array(np.nonzero(region))
    mins = np.clip(idx.min(axis=1) - pad, 0, None)
    maxs = np.minimum(idx.max(axis=1) + pad + 1, np.array(label_vol.shape))
    edges = [np.linspace(mins[a], maxs[a], grid + 1).round().astype(int) for a in range(3)]
    blocks = [(slice(edges[0][i], edges[0][i + 1]), slice(edges[1][j], edges[1][j + 1]), slice(edges[2][k], edges[2][k + 1]))
              for i in range(grid) for j in range(grid) for k in range(grid)]
    block_ids = list(range(len(blocks)))

    v = {}
    for key, present in all_role_combos(block_ids):
        img = img_arr
        for b in block_ids:
            if not present[b]:
                img = noise_region(img, _block_mask(label_vol.shape, blocks[b]), rng)
        mask = predict_masks(predictor, img, [prompt])[0]
        v[key] = vm.score_masks(mask, region, region)["dice_a"]
    phi = shapley_values(v, block_ids)
    rows = [{"organ": organ, "block_id": b, "block_bbox": _bbox_str(blocks[b]), "shapley_value": round(phi[b], 6)}
            for b in block_ids]
    if not return_map:
        return rows
    voxel_map = np.full(label_vol.shape, np.nan, dtype=np.float32)
    for b in block_ids:
        voxel_map[blocks[b]] = phi[b]
    return rows, voxel_map


# ------------------------------------------------------------------------------------------------
# row 18 -- per-layer attention/KL divergence against the ground-truth mask
# ------------------------------------------------------------------------------------------------
def run_attention_kl(predictor, img_arr, label_vol, organs, ids, prompts,
                      generic_prompt="Segment the anatomy in this image.", out_chunk=12,
                      return_maps=False, map_downsample=4):
    """Row 18 (extends row 15's attention pipeline). For every organ and decoder layer, computes
    KL(GT mask distribution || cross-attention density distribution) over voxels -- the diagnostic
    used by the VGMED/VGRefine paper (arXiv 2603.14323) to find which layer's attention best matches
    the clinically relevant region. Also reports the same KL after dividing out a generic, non-spatial
    prompt's attention map (their reference-prompt normalization), which isolates prompt-specific
    attention from wherever VoxTell looks by default. Returns rows: organ, layer, kl_divergence,
    kl_divergence_vs_generic_normalized (lower = attention better matches the GT region).

    return_maps=True also returns a dict (same convention as `run_interventions`'s `attention_maps`):
    per organ, the last-layer and mean-of-layers attention density, downsampled by `map_downsample`
    -- an actual spatial map to overlay on the image/GT (row 0), not just the scalar KL above."""
    data, bbox, orig_shape = predictor.preprocess(img_arr)
    emb_real = predictor.embed_text_prompts([prompts[o] for o in organs])
    emb_generic = predictor.embed_text_prompts([generic_prompt])
    emb_all = torch.cat([emb_real, emb_generic], dim=1)
    outputs = [{"name": o, "row": i, "ops": []} for i, o in enumerate(organs)]
    outputs.append({"name": "__generic__", "row": len(organs), "ops": []})

    crop_sl = tuple(slice(int(b[0]), int(b[1])) for b in bbox)
    label_crop = label_vol[crop_sl]
    regions_crop = {o: label_crop == ids[o] for o in organs}

    _, attn = _run_chunked(
        predictor, data, emb_all, outputs, bbox, orig_shape,
        attn_factory=lambda shape, n_layers, n_rows: vm.AttnAccumulator(shape, n_layers, n_rows, predictor.patch_size, predictor.device),
        attn_rows=len(organs) + 1, on_patch=None, reset=lambda: None, chunk=out_chunk)

    eps = 1e-8

    def kl(p_norm, q_norm):
        return float((p_norm * (torch.log(p_norm.clamp_min(eps)) - torch.log(q_norm.clamp_min(eps)))).sum())

    rows = []
    for p, organ in enumerate(organs):
        gt = torch.from_numpy(regions_crop[organ].astype(np.float32))
        if float(gt.sum()) == 0:
            continue
        gt_norm = gt / gt.sum()
        for l in range(attn.shape[0]):
            d = attn[l, p].float()
            d_norm = d / d.sum().clamp_min(eps)
            g = attn[l, len(organs)].float()
            g_norm = g / g.sum().clamp_min(eps)
            ratio = d_norm / g_norm.clamp_min(eps)
            ratio_norm = ratio / ratio.sum().clamp_min(eps)
            rows.append({"organ": organ, "layer": l + 1,
                         "kl_divergence": round(kl(gt_norm, d_norm), 6),
                         "kl_divergence_vs_generic_normalized": round(kl(gt_norm, ratio_norm), 6)})
    if not return_maps:
        return rows
    maps = {}
    pool = torch.nn.AvgPool3d(map_downsample, ceil_mode=True)
    for p, organ in enumerate(organs):
        last = pool(attn[-1, p][None, None].float())[0, 0]
        mean = pool(attn[:, p].float().mean(0)[None, None])[0, 0]
        maps[f"{organ}__last_layer"] = last.half().cpu().numpy()
        maps[f"{organ}__mean_of_layers"] = mean.half().cpu().numpy()
    return rows, maps


# ------------------------------------------------------------------------------------------------
# row 21 -- joint image-vs-word Shapley (3 roles: image, organ word, position word)
# ------------------------------------------------------------------------------------------------
def _build_role_prompt(organ_name, organ_word_present, position_word_present):
    position, core = None, organ_name
    parts = organ_name.split(None, 1)
    if len(parts) == 2 and parts[0].lower() in ("left", "right"):
        position, core = parts[0].lower(), parts[1]
    words = ["Segment", "the"]
    if position_word_present and position:
        words.append(position)
    words.append(core if organ_word_present else "structure")
    words += ["in", "this", "image"]
    s = " ".join(words)
    return s[0].upper() + s[1:] + "."


def _crop_patch_around(data, centre_vox, patch_size):
    """Crop a `patch_size`-shaped window of `data` (C, X, Y, Z) centred on `centre_vox`, clamped to
    stay inside the array. `data` is `predictor.preprocess(img_arr)`'s own output -- a torch tensor,
    not a numpy array (confirmed by every other row's usage, e.g. it's passed straight into
    `predictor.predict_sliding_window_return_logits`/`vm.run_outputs` elsewhere in this file) -- so
    this works natively in torch, not `np.zeros`/`torch.from_numpy`, which both raise on a tensor
    input (`np.zeros(..., dtype=<torch dtype>)` cannot interpret a torch dtype; `torch.from_numpy`
    requires an actual numpy array). Still accepts a plain numpy array too, for any future caller.
    Returns (patch (1, C, *patch_size), offset (3,) -- the crop's start index, so results can be
    mapped back into `data`'s coordinate frame if needed)."""
    is_tensor = torch.is_tensor(data)
    shape = np.array(data.shape[1:])
    ps = np.array(patch_size)
    start = np.clip(np.round(np.asarray(centre_vox) - ps / 2).astype(int), 0, np.maximum(shape - ps, 0))
    end = np.minimum(start + ps, shape)
    start = np.maximum(end - ps, 0)  # re-clamp if the patch is larger than the volume near an edge
    sl = tuple(slice(int(s), int(s) + int(p)) for s, p in zip(start, ps))
    patch = data[(slice(None), *sl)]
    if tuple(patch.shape[1:]) != tuple(int(p) for p in ps):  # volume smaller than patch_size on some axis -- pad with zeros
        if is_tensor:
            padded = torch.zeros((data.shape[0], *(int(p) for p in ps)), dtype=data.dtype, device=data.device)
        else:
            padded = np.zeros((data.shape[0], *(int(p) for p in ps)), dtype=data.dtype)
        padded[(slice(None), *tuple(slice(0, s) for s in patch.shape[1:]))] = patch
        patch = padded
    return patch[None], start


# ------------------------------------------------------------------------------------------------
# row 19 -- gradient-based saliency (vanilla gradient + Grad-CAM), one patch around the organ
# ------------------------------------------------------------------------------------------------
def run_gradient_saliency(predictor, img_arr, label_vol, organ, ids, prompt, return_maps=False):
    """Row 19. Crops one patch_size window of the preprocessed image centred on the organ's GT
    centroid (cheap: one encoder + one backward pass, vs. row 17's full sliding-window Shapley
    enumeration -- the same LIME-to-Shapley relationship, just for images), computes vanilla-gradient
    and Grad-CAM saliency (`vm.gradient_saliency`), and summarizes it against the GT mask inside that
    same patch. MUST be called outside any `torch.no_grad()`/`inference_mode()` context.
    Returns a single-row dict: organ, mean_saliency_in_organ, mean_saliency_outside_organ,
    saliency_enrichment, cam_mean_in_organ, cam_mean_outside_organ."""
    data, bbox, orig_shape = predictor.preprocess(img_arr)
    crop_sl = tuple(slice(int(b[0]), int(b[1])) for b in bbox)
    label_crop = label_vol[crop_sl]
    region = label_crop == ids[organ]
    c = centroid(region)
    if c is None:
        return None
    patch, offset = _crop_patch_around(data, c, predictor.patch_size)
    patch_t = (patch if torch.is_tensor(patch) else torch.from_numpy(patch)).to(predictor.device)
    emb = predictor.embed_text_prompts([prompt])[0].to(predictor.device)  # (1, D)
    saliency, cam = vm.gradient_saliency(predictor.network, patch_t, row=0, prompts_embed=emb)
    end = offset + np.array(predictor.patch_size)
    region_in_patch = np.zeros_like(saliency, dtype=bool)
    rs = tuple(slice(max(int(o), 0), min(int(e), int(l))) for o, e, l in zip(offset, end, label_crop.shape))
    ps = tuple(slice(0, s.stop - s.start) for s in rs)
    region_in_patch[ps] = region[rs]
    if not region_in_patch.any():
        return None
    cam_up = vm.grad_cam_upsample(cam, predictor.patch_size)
    out_sal_in, out_sal_out = float(saliency[region_in_patch].mean()), float(saliency[~region_in_patch].mean())
    cam_in, cam_out = float(cam_up[region_in_patch].mean()), float(cam_up[~region_in_patch].mean())
    row = {"organ": organ, "mean_saliency_in_organ": round(out_sal_in, 6),
           "mean_saliency_outside_organ": round(out_sal_out, 6),
           "saliency_enrichment": round(out_sal_in / max(out_sal_out, 1e-8), 4),
           "cam_mean_in_organ": round(cam_in, 6), "cam_mean_outside_organ": round(cam_out, 6),
           "cam_enrichment": round(cam_in / max(cam_out, 1e-8), 4)}
    if not return_maps:
        return row
    # place the patch's saliency/CAM back into the crop's own coordinate frame, for plotting
    # against the same image/GT the rest of this module works in
    full_saliency = np.zeros(label_crop.shape, dtype=np.float32)
    full_cam = np.zeros(label_crop.shape, dtype=np.float32)
    full_saliency[rs] = saliency[ps]
    full_cam[rs] = cam_up[ps]
    return row, full_saliency, full_cam


# ------------------------------------------------------------------------------------------------
# row 20 -- fused spatial feature map, matched vs. mismatched prompt (resolves plan row 1)
# ------------------------------------------------------------------------------------------------
def run_activation_overlap(predictor, img_arr, label_vol, organ, ids, prompts, stage=-1,
                            return_maps=False, map_downsample=4):
    """Row 20 -- a deeper repeat of plan row 1 (already run using the final pre-sigmoid logits as
    its "internal representation") at a genuine intermediate layer. For `organ`'s matched prompt
    and two mismatches (wrong side via `mirror`, wrong organ via a different organ in `ids`),
    captures VoxTellDecoder's spatial `fusion_features` (`vm.FusedSpatialCapture`) and measures IoU
    between its highest-activation region and the GT bbox mask (`vm.activation_overlap_iou`).
    `stage`: which captured stage to read (-1 = highest resolution / last stage). Returns up to 3
    rows: organ, condition, stage, iou_with_gt.

    return_maps=True also returns a dict {condition: array}, the fused-feature map itself, upsampled
    to the crop's coordinate frame (same frame as row 0's masks) and then downsampled by
    `map_downsample` for storage -- an actual spatial map to overlay, not just the scalar IoU."""
    gt = label_vol == ids[organ]
    if not gt.any():
        return ([], {}) if return_maps else []
    other_organ = next((o for o in ids if o != organ and o != mirror(organ)), None)
    conditions = [("matched", prompts[organ])]
    if mirror(organ) in prompts:
        conditions.append(("mismatched_side", prompts[mirror(organ)]))
    if other_organ is not None:
        conditions.append(("mismatched_organ", prompts[other_organ]))

    rows, maps = [], {}
    pool = torch.nn.AvgPool3d(map_downsample, ceil_mode=True)
    for cond, prompt in conditions:
        cap = vm.FusedSpatialCapture(predictor.network.decoder)
        cap.attach()
        try:
            data, bbox, orig_shape = predictor.preprocess(img_arr)
            emb = predictor.embed_text_prompts([prompt])
            with torch.no_grad():
                predictor.predict_sliding_window_return_logits(data, emb)
        finally:
            cap.detach()
        if not cap.maps:
            continue
        stage_idx = sorted(cap.maps)[stage]
        crop_sl = tuple(slice(int(b[0]), int(b[1])) for b in bbox)
        gt_crop = gt[crop_sl]
        iou = vm.activation_overlap_iou(cap.maps[stage_idx][0], gt_crop)
        rows.append({"organ": organ, "condition": cond, "stage": stage_idx, "iou_with_gt": round(iou, 4)})
        if return_maps:
            m = cap.maps[stage_idx][0].float()
            if m.ndim == 4:
                m = m.abs().mean(dim=0)
            up = torch.nn.functional.interpolate(m[None, None], size=tuple(gt_crop.shape), mode='trilinear', align_corners=False)
            maps[f"{organ}__{cond}"] = pool(up)[0, 0].half().cpu().numpy()
    return (rows, maps) if return_maps else rows


# ------------------------------------------------------------------------------------------------
# row 0 -- baseline prediction + GT export, for plotting (image + GT + predicted mask overlays)
# ------------------------------------------------------------------------------------------------
def export_baseline_masks(predictor, img_arr, label_vol, organs, ids, prompts, image_path=None, label_path=None):
    """Not a plan row of its own -- a shared plotting export run once per subject, before any of
    rows 12-21. For every organ: the full-prompt predicted mask and the GT mask, both RLE-encoded
    (`encode_mask_rle`) so the complete 3D mask -- not just a summary metric -- can be reloaded and
    overlaid on the source image slice-by-slice later (`decode_mask_rle` + `image_path`/`label_path`
    from the manifest). Returns one row per organ: organ, prompt, image_path, label_path,
    volume_shape, dice, n_pred_voxels, pred_centroid_vox_{0,1,2}, gt_centroid_vox_{0,1,2},
    pred_mask_rle_b64, gt_mask_rle_b64."""
    masks = predict_masks(predictor, img_arr, [prompts[o] for o in organs])
    rows = []
    for k, o in enumerate(organs):
        gt = label_vol == ids[o]
        pred = masks[k]
        pc, gc = centroid(pred), centroid(gt)
        rows.append({
            "organ": o, "prompt": prompts[o],
            "image_path": image_path, "label_path": label_path,
            "volume_shape": "x".join(str(s) for s in label_vol.shape),
            "dice": round(vm.score_masks(pred, gt, gt)["dice_a"], 4),
            "n_pred_voxels": int(pred.sum()), "n_gt_voxels": int(gt.sum()),
            "pred_centroid_vox_0": round(float(pc[0]), 2) if pc is not None else None,
            "pred_centroid_vox_1": round(float(pc[1]), 2) if pc is not None else None,
            "pred_centroid_vox_2": round(float(pc[2]), 2) if pc is not None else None,
            "gt_centroid_vox_0": round(float(gc[0]), 2) if gc is not None else None,
            "gt_centroid_vox_1": round(float(gc[1]), 2) if gc is not None else None,
            "gt_centroid_vox_2": round(float(gc[2]), 2) if gc is not None else None,
            "pred_mask_rle_b64": encode_mask_rle(pred),
            "gt_mask_rle_b64": encode_mask_rle(gt),
        })
    return rows


def run_joint_image_word_shapley(predictor, img_arr, label_vol, organ, ids, seed=0, return_coalitions=False):
    """Row 21. Three roles -- image (intact / noised over the organ's own GT region, reusing row 13's
    `noise_region`), organ_word, position_word (reusing the same prompt-building logic as rows 5-11) --
    give 2**3=8 coalitions, scored by both f_dice_gt and f_dice_self and fed into the same
    `shapley_values` used for row 17 and the word-attribution rows. This is the single number that
    puts image contribution and word contribution on one common scale. Returns rows: organ, role,
    metric, shapley_value (already in Dice-score units: shapley_value is literally "how many Dice
    points this role is worth, averaged fairly over every order the roles could be added in").

    return_coalitions=True additionally returns the literal, un-derived numbers the Shapley credit
    above is computed FROM: the actual Dice score for each of the 8 present/absent combinations
    (e.g. "if the model only gets the image, no words at all, what Dice does it get?"). These are
    more intuitive to read directly but don't have the Shapley math's fairness guarantees (e.g. they
    don't sum to the total the way the credits do) -- returns (summary_rows, coalition_rows), where
    coalition_rows has columns: organ, roles_present, image, organ_word, position_word, dice_gt,
    dice_self."""
    rng = np.random.default_rng(seed)
    gt = label_vol == ids[organ]
    roles = ["image", "organ_word", "position_word"]
    masks = {}
    for key, present in all_role_combos(roles):
        img = img_arr if present["image"] else noise_region(img_arr, gt, rng)
        prompt = _build_role_prompt(organ, present["organ_word"], present["position_word"])
        masks[key] = predict_masks(predictor, img, [prompt])[0]
    base_mask = masks[frozenset(roles)]
    v_gt = {k: vm.score_masks(m, gt, gt)["dice_a"] for k, m in masks.items()}
    v_self = {k: vm.score_masks(m, base_mask, base_mask)["dice_a"] for k, m in masks.items()}
    phi_gt, phi_self = shapley_values(v_gt, roles), shapley_values(v_self, roles)
    rows = []
    for r in roles:
        rows.append({"organ": organ, "role": r, "metric": "f_dice_gt", "shapley_value": round(phi_gt[r], 6)})
        rows.append({"organ": organ, "role": r, "metric": "f_dice_self", "shapley_value": round(phi_self[r], 6)})
    if not return_coalitions:
        return rows
    coalition_rows = []
    for key in v_gt:
        coalition_rows.append({
            "organ": organ,
            "roles_present": "+".join(sorted(key)) if key else "none",
            "image": "image" in key, "organ_word": "organ_word" in key, "position_word": "position_word" in key,
            "dice_gt": round(v_gt[key], 4), "dice_self": round(v_self[key], 4),
        })
    return rows, coalition_rows


# ------------------------------------------------------------------------------------------------
# row 22 -- linear probing: is left/right linearly DECODABLE at each decoder layer, not just different?
# ------------------------------------------------------------------------------------------------
def _home_patch_indices(predictor, data, bbox, label_vol, organs, ids):
    """For every organ, the sliding-window patch index (into the SAME slicer list `run_outputs`
    builds internally, in the same order -- both are pure functions of `data.shape` and
    `predictor.patch_size`, so the index computed here lines up with `on_patch`'s call order)
    whose block contains that organ's own ground-truth centroid, or is nearest to it if no single
    block contains it. This is "the patch that actually produces the final mask there" -- the
    representative sliding-window pass for that organ, as opposed to averaging over the whole,
    mostly-irrelevant, volume."""
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    data_p, revert = pad_nd_image(data, predictor.patch_size, 'constant', {'value': 0}, True, None)
    slicers = predictor._internal_get_sliding_window_slicers(data_p.shape[1:])
    crop_sl = tuple(slice(int(b[0]), int(b[1])) for b in bbox)
    label_crop = label_vol[crop_sl]
    pad_before = [s.start for s in revert[1:]]
    home = {}
    for organ in organs:
        c = centroid(label_crop == ids[organ])
        if c is None:
            continue
        pt = [float(c[a]) + pad_before[a] for a in range(3)]
        best_i, best_score = None, None
        for i, slicer in enumerate(slicers):
            sl = slicer[1:]
            inside = all(sl[a].start <= pt[a] < sl[a].stop for a in range(3))
            centre = [(sl[a].start + sl[a].stop) / 2 for a in range(3)]
            dist2 = sum((pt[a] - centre[a]) ** 2 for a in range(3))
            score = (0 if inside else 1, dist2)
            if best_score is None or score < best_score:
                best_i, best_score = i, score
        home[organ] = best_i
    return home


def run_laterality_probing(predictor, img_arr, label_vol, organs, ids, prompts):
    """Row 22. Trains a linear probe (logistic regression) at every decoder layer -- state 0 is the
    projected text embedding itself (`tgt0`, before any cross-attention), states 1-6 are after each
    decoder layer -- to predict left vs. right from the hidden state ALONE, independent of what mask
    the model actually outputs. This is a stricter question than row 16's cosine distance: distance
    can stay small while a linear probe still finds a perfectly usable separating direction (distance
    measures "how different two specific states are", the probe measures "is side actually recoverable
    from this representation by a simple linear read-out"). If a layer's probe is near-perfect but the
    final mask still gets the side wrong, that is direct evidence the DECODER discards information
    that was already there, not that the information was never computed (an alignment failure would
    instead show a probe that stays near-chance at every layer).

    The hidden state for each organ is read from its own "home patch" (`_home_patch_indices` above) --
    the one sliding-window patch that actually contains that organ's ground-truth centroid -- not
    averaged across the whole volume, most of which is irrelevant to that organ's prediction.

    Small-n caveat (stated explicitly in the returned rows via `n_pairs`): with a single subject and
    typically single-digit bilateral organ pairs, cross-validation here is leave-one-ORGAN-pair-out
    (both the left and right instance of a pair are held out together, so the probe is never trained
    and tested on the same organ identity) -- indicative, not a robust estimate. Returns rows: layer,
    cv_accuracy, train_accuracy, n_pairs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut

    data, bbox, orig_shape = predictor.preprocess(img_arr)
    home = _home_patch_indices(predictor, data, bbox, label_vol, organs, ids)
    cores = sorted({o.split(None, 1)[1] for o in organs if o.split(None, 1)[0] in ("left", "right")})
    pairs = [c for c in cores if f"left {c}" in home and f"right {c}" in home]
    if len(pairs) < 2:
        return []

    wanted = {}
    for i, c in enumerate(pairs):
        wanted[f"left {c}"] = 2 * i
        wanted[f"right {c}"] = 2 * i + 1
    ordered_organs = sorted(wanted, key=wanted.get)
    emb = predictor.embed_text_prompts([prompts[o] for o in ordered_organs])
    target_patches = {home[o]: o for o in wanted}

    captured = {}
    counter = {"i": 0}

    def on_patch(slicer, states, attns, ctx):
        organ = target_patches.get(counter["i"])
        counter["i"] += 1
        if organ is not None:
            r = wanted[organ]
            captured[organ] = [s[r, 0].float().cpu().numpy() for s in states]

    vm.run_outputs(predictor, data, emb, [{"name": o, "row": wanted[o], "ops": []} for o in ordered_organs],
                    on_patch=on_patch)

    organs_order = [o for o in ordered_organs if o in captured]
    if len(organs_order) < 4:  # need at least 2 complete pairs to do leave-one-pair-out at all
        return []
    n_layers = len(captured[organs_order[0]])
    y = np.array([1 if o.startswith("right") else 0 for o in organs_order])
    groups = np.array([o.split(None, 1)[1] for o in organs_order])

    rows = []
    for layer in range(n_layers):
        X = np.stack([captured[o][layer] for o in organs_order])
        logo = LeaveOneGroupOut()
        correct, total = 0, 0
        for tr, te in logo.split(X, y, groups=groups):
            if len(set(y[tr])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
            correct += int((clf.predict(X[te]) == y[te]).sum())
            total += len(te)
        cv_acc = correct / total if total else float("nan")
        clf_full = LogisticRegression(max_iter=2000).fit(X, y)
        train_acc = float((clf_full.predict(X) == y).mean())
        rows.append({"layer": layer, "cv_accuracy": round(cv_acc, 4) if total else None,
                     "train_accuracy": round(train_acc, 4), "n_pairs": len(pairs)})
    return rows


# ------------------------------------------------------------------------------------------------
# row 23 -- embedding geometry: one consistent "left -> right" direction, or a different one per organ?
# ------------------------------------------------------------------------------------------------
def run_embedding_geometry(predictor, organs, prompts):
    """Row 23. Does the model use ONE consistent "left -> right" direction across every organ, the
    way a genuinely spatial/relational representation would (the classic word-vector-arithmetic
    test, e.g. "king - man + woman"), or does each organ get its own, unrelated left/right
    direction (meaning "left" is re-derived per organ, not a shared, composable concept)? For
    every bilateral pair, computes `embed("right X") - embed("left X")` in the raw text-encoder
    embedding space (the same embedding rows 2-4 and row 22's layer-0 state are built from, before
    `project_text_embed`). Pure text-embedding arithmetic -- no image, no sliding window, no model
    forward pass beyond the text encoder: the cheapest row in this whole module.

    Returns one schema for every row (fields not applicable to a row's `kind` are None), two kinds:
      kind='pair'    -- organ_a, organ_b, cosine_similarity between their two diff vectors
      kind='summary' -- mean_cosine_similarity (mean of all pairwise similarities), coherence_ratio
                        = ||mean(diff vectors)|| / mean(||diff vector||): 1.0 means every organ's
                        "left->right" step points the same way (one reusable direction), 0.0 means
                        the directions are unrelated/random per organ, n_organs."""
    cores = sorted({o.split(None, 1)[1] for o in organs if o.split(None, 1)[0] in ("left", "right")})
    cores = [c for c in cores if f"left {c}" in prompts and f"right {c}" in prompts]
    if len(cores) < 2:
        return []
    left_emb = predictor.embed_text_prompts([prompts[f"left {c}"] for c in cores])[0]
    right_emb = predictor.embed_text_prompts([prompts[f"right {c}"] for c in cores])[0]
    diffs = (right_emb - left_emb).float().cpu().numpy()
    norms = np.linalg.norm(diffs, axis=1)
    unit = diffs / np.clip(norms, 1e-8, None)[:, None]
    sim = unit @ unit.T

    rows = []
    for i in range(len(cores)):
        for j in range(i + 1, len(cores)):
            rows.append({"kind": "pair", "organ_a": cores[i], "organ_b": cores[j],
                         "cosine_similarity": round(float(sim[i, j]), 4),
                         "mean_cosine_similarity": None, "coherence_ratio": None, "n_organs": None})
    mean_cos = float(sim[np.triu_indices(len(cores), k=1)].mean())
    coherence = float(np.linalg.norm(diffs.mean(axis=0)) / max(float(norms.mean()), 1e-8))
    rows.append({"kind": "summary", "organ_a": None, "organ_b": None, "cosine_similarity": None,
                 "mean_cosine_similarity": round(mean_cos, 4), "coherence_ratio": round(coherence, 4),
                 "n_organs": len(cores)})
    return rows


# ------------------------------------------------------------------------------------------------
# row 24 -- cross-attention head ablation: which specific head, not just which layer
# ------------------------------------------------------------------------------------------------
def run_attention_head_ablation(predictor, img_arr, label_vol, organs, ids, prompts, layers=None):
    """Row 24. Zero-ablates each cross-attention HEAD, one at a time (`vm.HeadAblation`), at each
    layer in `layers` (default: the LAST layer only -- cheapest, and closest to the final decision;
    pass e.g. `layers=range(6)` for every layer, at 6x the cost: n_organs x n_layers x n_heads full
    sliding-window predictions, the same affordability trade-off as row 17's block count), and
    re-predicts the FULL mask through the predictor's own public API. The ablation hook operates at
    the `nn.Module` level, so it affects the public `predict_sliding_window_return_logits` path
    identically to the custom loop -- no custom inference loop needed here, unlike rows 14-16.

    MUST call `vm.validate_head_ablation_noop` first and confirm it returns 1.0 (up to floating-point
    rounding) -- see this module's own validation cell in the notebook.

    For every organ, layer and head: re-predicts the mask with that head zeroed, scores Dice against
    GT and against the organ's own UNABLATED baseline mask (f_dice_self -- did ablating this head
    change the answer AT ALL), and the same `call` classification rows 12/13 use (does the mask still
    land on this organ, on its mirror, on both, or neither) against {this organ, its mirror}. A head
    whose ablation collapses dice_vs_gt or flips `call` toward the mirror organ is doing real
    laterality work; a head whose ablation changes nothing (dice_vs_baseline ~= 1) is not involved in
    this decision at all. Returns rows: organ, layer, head, n_heads, dice_vs_gt, dice_vs_baseline, call."""
    num_layers = len(predictor.network.transformer_decoder.layers)
    layers = list(layers) if layers is not None else [num_layers - 1]
    n_heads = vm.num_cross_attn_heads(predictor.network.transformer_decoder.layers)

    baselines = {o: predict_masks(predictor, img_arr, [prompts[o]])[0] for o in organs}
    rows = []
    for organ in organs:
        gt = label_vol == ids[organ]
        gt_mirror = label_vol == ids[mirror(organ)] if mirror(organ) in ids else np.zeros_like(gt)
        base_mask = baselines[organ]
        for layer in layers:
            for head in range(n_heads):
                ablation = vm.HeadAblation(predictor.network.transformer_decoder.layers, [layer], [head])
                ablation.attach()
                try:
                    mask = predict_masks(predictor, img_arr, [prompts[organ]])[0]
                finally:
                    ablation.detach()
                scored = vm.score_masks(mask, gt, gt_mirror)
                rows.append({
                    "organ": organ, "layer": layer, "head": head, "n_heads": n_heads,
                    "dice_vs_gt": scored["dice_a"],
                    "dice_vs_baseline": vm.score_masks(mask, base_mask, base_mask)["dice_a"],
                    "call": scored["call"],
                })
    return rows
