"""
Mechanistic helpers for the VoxTell groundedness investigation (plan rows 12-15).

Everything here is written against the installed `voxtell` v0.1.0 source
(voxtell/model/voxtell_model.py, voxtell/model/transformer.py,
voxtell/inference/predictor.py) and reuses the model's own modules -- nothing in
voxtell is monkeypatched.

Facts about VoxTell this code depends on (all verified from the source):
  - The image encoder never sees the text; its layer-4 features (12x12x12 tokens
    for a 192^3 patch) become the cross-attention *memory*.
  - The text embedding is projected (`project_text_embed`) and becomes the initial
    query state of a 6-layer pre-norm transformer decoder (`normalize_before=True`
    => `forward_pre`: cross-attention + FFN only, NO self-attention between the
    queries, so prompts in a batch never interact).
  - Each layer returns (state, cross-attention weights averaged over heads,
    shape (1, n_queries, n_tokens)).
  - After the last layer a LayerNorm gives the "mask embedding", which the
    convolutional decoder turns into a mask (one full decoder pass per prompt).
  - Because the decoder is a residual stream, the original text only reaches later
    layers through the running state. Replacing the *whole state* at layer k with
    another prompt's state therefore reproduces that prompt exactly (used as a
    positive control); the informative intervention is to replace a single layer's
    *update* (state_out - state_in).

Public API
  tokens_to_patch_grid, encode, base_trajectory, patched_state, decode,
  AttnAccumulator, run_outputs, score_masks
"""

import base64
import contextlib
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------------
# CSV-only mask/array persistence (Kaggle notebooks can't easily download image files,
# so every mask and image patch these experiments produce is encoded straight into a CSV
# cell instead of saved as a separate file)
# ---------------------------------------------------------------------------------
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
    idx = 0
    value = False
    for run_len in runs:
        if value and run_len:
            flat[idx:idx + run_len] = True
        idx += int(run_len)
        value = not value
    return flat.reshape(shape, order='C')


def encode_array_b64(arr, dtype=np.float16):
    """zlib+base64-encode an arbitrary numpy array (image patches, logit/fusion maps -- anything
    that isn't a sparse boolean mask, where RLE above is far more compact). Cast to `dtype`
    first to keep CSV cell size down; decode with the SAME dtype and shape."""
    raw = zlib.compress(np.asarray(arr).astype(dtype).tobytes(order='C'), level=9)
    return base64.b64encode(raw).decode('ascii')


def decode_array_b64(encoded, shape, dtype=np.float16):
    raw = zlib.decompress(base64.b64decode(encoded))
    return np.frombuffer(raw, dtype=dtype).reshape(shape, order='C')


# ---------------------------------------------------------------------------------
# token <-> voxel bookkeeping
# ---------------------------------------------------------------------------------
def tokens_to_patch_grid(w, g):
    """`w`: (..., g**3) values over VoxTell's image tokens -> (..., g, g, g) indexed in
    the patch's own axis order (axis0, axis1, axis2).

    VoxTell flattens tokens as `rearrange(feat, 'b c d h w -> (h w d) b c')`, i.e.
    token index = (h*g + w)*g + d, where (d, h, w) are the patch axes (0, 1, 2).
    """
    lead = w.shape[:-1]
    x = w.reshape(*lead, g, g, g)  # [h, w, d] == [axis1, axis2, axis0]
    nd = x.ndim
    return x.permute(*range(nd - 3), nd - 1, nd - 3, nd - 2)  # -> [d, h, w]


# ---------------------------------------------------------------------------------
# model pieces (mirror VoxTellModel.forward, split so we can read / patch the decoder)
# ---------------------------------------------------------------------------------
def encode(model, patch):
    """patch: (1, C, X, Y, Z) -> (skips, memory (S, 1, Q))."""
    skips = model.encoder(patch)
    feat = skips[model.selected_decoder_layer]
    mem = rearrange(feat, 'b c d h w -> b h w d c')
    mem = model.project_bottleneck_embed(mem)
    mem = rearrange(mem, 'b h w d c -> (h w d) b c')
    return skips, mem


def base_trajectory(model, tgt0, memory, pos):
    """Run the decoder layers on `tgt0` (M, 1, Q). Returns (states, attns):
    states[k] is the state entering layer k (states[-1] = after the last layer, before
    the final norm); attns[k] = (1, M, S) cross-attention weights of layer k."""
    layers = model.transformer_decoder.layers
    assert all(l.normalize_before for l in layers), 'expected the pre-norm (forward_pre) decoder'
    states, attns = [tgt0], []
    t = tgt0
    for layer in layers:
        t, w = layer(t, memory, pos=pos)
        states.append(t)
        attns.append(w)
    return states, attns


def patched_state(model, tgt0_row, memory, pos, base_states, ops):
    """Re-run the decoder for ONE query (tgt0_row: (1, 1, Q)) with interventions.

    ops: list of dicts {'layer': k, 'mode': 'update' | 'state', 'src': row}
      - 'update': the layer-k output is `state_in + (src's layer-k update)`, i.e. only that
                  layer's contribution is taken from the source prompt
      - 'state' : the layer-k output is the source prompt's whole state (positive control)
    `base_states` is the list returned by `base_trajectory` for all prompts (M rows)."""
    by_layer = {}
    for op in ops:
        by_layer.setdefault(op['layer'], []).append(op)
    t = tgt0_row
    for k, layer in enumerate(model.transformer_decoder.layers):
        new, _ = layer(t, memory, pos=pos)
        for op in by_layer.get(k, []):
            s = op['src']
            if op['mode'] == 'update':
                new = t + (base_states[k + 1][s:s + 1] - base_states[k][s:s + 1])
            elif op['mode'] == 'state':
                new = base_states[k + 1][s:s + 1].clone()
            else:
                raise ValueError(op['mode'])
        t = new
    return t


def decode(model, skips, mask_emb):
    """mask_emb: (n, Q) -> logits (n, X, Y, Z). Same per-prompt loop as VoxTellModel.forward."""
    me = mask_emb.unsqueeze(0)  # (1, n, Q)
    mes = [proj(me) for proj in model.project_to_decoder_channels]
    outs = []
    for i in range(me.shape[1]):
        prompt_embeds = [m[:, i:i + 1] for m in mes]
        outs.append(model.decoder(skips, prompt_embeds)[0])
    return torch.cat(outs, dim=1)[0]


# ---------------------------------------------------------------------------------
# attention accumulation over the sliding window
# ---------------------------------------------------------------------------------
class AttnAccumulator:
    """Accumulates cross-attention *density* (attention weight x n_tokens, so the mean is 1)
    into voxel space, weighted by the same Gaussian as the logits."""

    def __init__(self, spatial_shape, n_layers, n_rows, patch_size, device):
        self.buf = torch.zeros((n_layers, n_rows, *spatial_shape), dtype=torch.float32, device=device)
        self.cnt = torch.zeros(tuple(spatial_shape), dtype=torch.float32, device=device)
        self.patch_size = tuple(int(p) for p in patch_size)

    def add(self, slicer, attn, gaussian):
        """attn: (L, M, S) float; gaussian: patch-shaped weights."""
        L, M, S = attn.shape
        g = int(round(S ** (1.0 / 3.0)))
        assert g ** 3 == S, f'{S} tokens is not a cube'
        dens = tokens_to_patch_grid(attn.float(), g) * S  # (L, M, g, g, g)
        gz = gaussian.float()
        for l in range(L):
            up = F.interpolate(dens[l][None], size=self.patch_size, mode='nearest')[0]  # (M, X, Y, Z)
            self.buf[(l, slice(None), *slicer[1:])] += up * gz
        self.cnt[slicer[1:]] += gz

    def finalize(self, revert_slicer):
        out = self.buf / self.cnt.clamp_min(1e-6)
        return out[(slice(None), slice(None), *revert_slicer[1:])]


# ---------------------------------------------------------------------------------
# sliding-window runner (same padding / step / Gaussian / accumulation as the predictor)
# ---------------------------------------------------------------------------------
@torch.inference_mode()
def run_outputs(predictor, data, embeddings, outputs, attn_acc_factory=None, attn_rows=None, on_patch=None):
    """Sliding-window inference for several *outputs* that share one encoder pass per patch.

    data        : preprocessed image tensor (C, X, Y, Z) (from predictor.preprocess)
    embeddings  : (1, M, D) text embeddings (rows may be blends built by the caller)
    outputs     : list of {'name': str, 'row': int, 'ops': [...]}; `ops` as in `patched_state`
    attn_acc_factory : optional callable(spatial_shape, n_layers, n_rows) -> AttnAccumulator
    attn_rows        : accumulate attention only for the first `attn_rows` embedding rows (default: all rows)
    on_patch         : optional callable(slicer, states, attns, ctx) called for every patch; `states` = list of
                       7 tensors (M, 1, Q) (input state + output of each layer), `attns` = 6 tensors (1, M, S),
                       patch_ctx = {'revert': padding-revert slicer, 'padded_shape': spatial shape after padding}
    Returns (logits (n_out, X, Y, Z) half, attention (L, M, X, Y, Z) or None), padding reverted."""
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    device = predictor.device
    model = predictor.network.to(device)
    data_p, revert = pad_nd_image(data, predictor.patch_size, 'constant', {'value': 0}, True, None)
    slicers = predictor._internal_get_sliding_window_slicers(data_p.shape[1:])
    data_p = data_p.to(device)
    emb = embeddings[0].to(device)  # (M, D)

    n_out = len(outputs)
    logits = torch.zeros((n_out, *data_p.shape[1:]), dtype=torch.half, device=device)
    npred = torch.zeros(data_p.shape[1:], dtype=torch.half, device=device)
    gaussian = compute_gaussian(tuple(predictor.patch_size), sigma_scale=1. / 8,
                                value_scaling_factor=10, device=device)
    attn_acc = None
    if attn_acc_factory is not None:
        n_attn = attn_rows if attn_rows is not None else emb.shape[0]
        attn_acc = attn_acc_factory(tuple(data_p.shape[1:]), len(model.transformer_decoder.layers), n_attn)
    patch_ctx = {"revert": revert, "padded_shape": tuple(data_p.shape[1:])}
    ctx = torch.autocast(device.type) if device.type == 'cuda' else contextlib.nullcontext()

    for slicer in slicers:
        patch = data_p[slicer][None]
        with ctx:
            skips, memory = encode(model, patch)
            tgt0 = model.project_text_embed(emb).unsqueeze(1)  # (M, 1, Q)
            states, attns = base_trajectory(model, tgt0, memory, model.pos_embed)
            embs = []
            for o in outputs:
                r = o['row']
                if o.get('ops'):
                    t = patched_state(model, tgt0[r:r + 1], memory, model.pos_embed, states, o['ops'])
                else:
                    t = states[-1][r:r + 1]
                embs.append(model.transformer_decoder.norm(t))
            mask_emb = torch.cat(embs, dim=0)[:, 0]  # (n_out, Q)
            pred = decode(model, skips, mask_emb)
        pred = (pred.float() * gaussian.float()).half()
        logits[(slice(None), *slicer[1:])] += pred
        npred[slicer[1:]] += gaussian.half()
        if attn_acc is not None:
            stack = torch.stack([a[0] for a in attns]).float()   # (L, M, S)
            attn_acc.add(slicer, stack[:, :n_attn], gaussian)
        if on_patch is not None:
            on_patch(slicer, states, attns, patch_ctx)

    torch.div(logits, npred, out=logits)
    if torch.any(torch.isinf(logits)):
        raise RuntimeError('inf in logits')
    logits = logits[(slice(None), *revert[1:])]
    attn = attn_acc.finalize(revert) if attn_acc is not None else None
    return logits, attn


# ---------------------------------------------------------------------------------
# scoring (numpy, side-agnostic: uses overlap with the two candidate ground-truth organs)
# ---------------------------------------------------------------------------------
def _dice(a, b):
    d = a.sum() + b.sum()
    return 1.0 if d == 0 else float(2.0 * np.logical_and(a, b).sum() / d)


def score_masks(mask, gt_a, gt_b, cov_thr=0.3, other_thr=0.2):
    """Overlap of a predicted mask with two candidate ground-truth organs.
    cov_x = |mask & gt_x| / |gt_x|. `call` is 'a' / 'b' when the mask covers that organ
    (>= cov_thr) and the other one less than other_thr, 'both' when both are covered,
    'empty' for an empty mask and 'neither' otherwise."""
    mask, gt_a, gt_b = (np.asarray(x, dtype=bool) for x in (mask, gt_a, gt_b))
    n = int(mask.sum())
    cov_a = float(np.logical_and(mask, gt_a).sum() / max(gt_a.sum(), 1))
    cov_b = float(np.logical_and(mask, gt_b).sum() / max(gt_b.sum(), 1))
    if n == 0:
        call = 'empty'
    elif cov_a >= cov_thr and cov_b >= cov_thr:
        call = 'both'
    elif cov_a >= cov_thr and cov_b < other_thr:
        call = 'a'
    elif cov_b >= cov_thr and cov_a < other_thr:
        call = 'b'
    else:
        call = 'neither'
    return {'n_pred_voxels': n, 'cov_a': round(cov_a, 4), 'cov_b': round(cov_b, 4),
            'dice_a': round(_dice(mask, gt_a), 4), 'dice_b': round(_dice(mask, gt_b), 4), 'call': call}


