"""
Experiments for plan rows 12-16 (all take a VoxTellPredictor-like object and plain numpy arrays, so they can be
tested without a GPU).

  row 12  run_flip                image left-right flip, prompt unchanged
  row 13  run_erasure             organ filled with noise / both organs filled with noise / noise image
  row 14  run_interventions       embedding blend and per-layer patching between "left X" and "right X"
  row 15  (same run)              cross-attention of every prompt over every organ region, per decoder layer
  row 16  (same run)              distance between the left and right query states at every decoder layer

Arrays: `img_arr` is (1, X, Y, Z) in the RAS-reoriented grid used by the notebooks, `label_vol` is (X, Y, Z).
`rl_axis` is the right-left axis of the spatial arrays (2 in the notebooks).
"""

import numpy as np
import torch

import voxtell_mechanistic as vm


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
