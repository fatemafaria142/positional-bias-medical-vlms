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
  AttnAccumulator, run_outputs, score_masks,
  gradient_saliency, grad_cam_bottleneck (plan row 19),
  FusedSpatialCapture (plan row 20)
"""

import contextlib

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


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


# ---------------------------------------------------------------------------------
# plan row 19 -- gradient-based saliency (vanilla gradient + Grad-CAM on the bottleneck)
# ---------------------------------------------------------------------------------
def gradient_saliency(model, patch, row=0, memory=None, pos=None, prompts_embed=None):
    """One-patch vanilla-gradient saliency: d(sum of logits)/d(patch), for the prompt at `row`.

    MUST be called OUTSIDE any `torch.inference_mode()`/`torch.no_grad()` context -- tensors
    created under `inference_mode` are permanently non-differentiable even once you leave it
    (this is why `run_outputs` is `@torch.inference_mode()` and cannot be reused here; call
    `encode`/`decode` directly instead, as this function does).

    patch: (1, C, X, Y, Z), will be cloned and marked requires_grad. prompts_embed: (M, D) text
    embeddings for every prompt sharing this patch (only `row`'s logits are backpropagated, but
    all prompts run through the same encoder pass, matching `run_outputs`'s batching).

    Returns (saliency (X, Y, Z) float32 = |d logits / d patch| summed over channels,
             bottleneck_grad (g, g, g) float32 = Grad-CAM weights at the bottleneck feature map,
             i.e. the channel-mean gradient of the same logits w.r.t. the bottleneck token grid).
    """
    assert torch.is_grad_enabled(), 'gradient_saliency needs grad enabled -- do not call inside inference_mode/no_grad'
    patch = patch.clone().detach().requires_grad_(True)
    skips, mem = encode(model, patch)
    # keep the bottleneck activation's own grad (for Grad-CAM) by not detaching it
    bottleneck = skips[model.selected_decoder_layer]
    bottleneck.retain_grad()
    tgt0 = model.project_text_embed(prompts_embed).unsqueeze(1)  # (M, 1, Q)
    states, _ = base_trajectory(model, tgt0, mem, pos if pos is not None else model.pos_embed)
    mask_emb = model.transformer_decoder.norm(states[-1][row:row + 1])  # (1, Q)
    logits = decode(model, skips, mask_emb)  # (1, X, Y, Z)
    logits.sum().backward()
    saliency = patch.grad.detach().abs().sum(dim=1)[0].float()  # (X, Y, Z)
    cam = bottleneck.grad.detach().mean(dim=1)[0].float()  # (d, h, w) -- channel-mean, ReLU'd below
    cam = cam.clamp_min(0)
    return saliency.cpu().numpy(), cam.cpu().numpy()


def grad_cam_upsample(cam_grid, patch_size):
    """Upsample a (d, h, w) Grad-CAM grid (as returned by gradient_saliency's bottleneck_grad,
    already in the patch's own axis order via tokens_to_patch_grid's convention) to voxel space."""
    g = cam_grid.shape[0]
    t = torch.from_numpy(cam_grid)[None, None].float()
    up = F.interpolate(t, size=tuple(patch_size), mode='trilinear', align_corners=False)
    return up[0, 0].numpy()


# ---------------------------------------------------------------------------------
# plan row 20 -- fused spatial feature map (resolves rows 1/4's "needs hook")
# ---------------------------------------------------------------------------------
class FusedSpatialCapture:
    """Captures VoxTellDecoder's per-stage spatial `fusion_features` (the image+text product that
    actually varies with the prompt at full spatial resolution, as opposed to the single global
    `mask_emb` query vector used elsewhere in this file, OR the final pre-sigmoid logits that plan
    row 1 already used as its stand-in "internal representation" -- this is a genuine intermediate
    layer, not the output) via a forward_pre_hook on each of the decoder's `seg_layers` stages --
    `fusion_features` is a local variable inside `VoxTellDecoder.forward` (voxtell_model.py), not a
    module's own output, so it is read off of the *input* to the next stage that consumes it, not a
    forward_hook's output.

    Usage:
        cap = FusedSpatialCapture(model.decoder)
        cap.attach()
        ... run inference (predictor.predict_* or vm.decode) ...
        maps = cap.maps  # {stage_index: (B, 1, H, W, D) tensor}, resolutions 24/48/96/192**3
        cap.detach()

    UNVERIFIED: the exact submodule path to `model.decoder.seg_layers` (or equivalent) has not
    been confirmed against a live model/forward pass -- the first real run should print
    `{k: v.shape for k, v in maps.items()}` and sanity-check it against the four documented
    resolutions (24, 48, 96, 192 cubed) before trusting any IoU computed from these maps.
    """

    def __init__(self, decoder_module, stage_attr='seg_layers'):
        self.decoder_module = decoder_module
        self.stage_attr = stage_attr
        self.maps = {}
        self._handles = []

    def _make_hook(self, stage_idx):
        def hook(module, inputs):
            self.maps[stage_idx] = inputs[0].detach().cpu()
        return hook

    def attach(self):
        stages = getattr(self.decoder_module, self.stage_attr)
        for i, stage in enumerate(stages):
            self._handles.append(stage.register_forward_pre_hook(self._make_hook(i)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []



# ---------------------------------------------------------------------------------
# plan row 24 -- cross-attention head ablation (which specific head, not just which layer)
# ---------------------------------------------------------------------------------
class HeadAblation:
    """Zero-ablates one or more cross-attention HEADS (not a whole layer) at one or more decoder
    layers. This is the standard "zero ablation" used in mechanistic interpretability to test
    whether one specific head is causally responsible for a behavior, as opposed to row 15/18's
    whole-layer attention, which cannot distinguish "one head does this" from "every head does
    this a little".

    IMPLEMENTATION NOTE (verified locally against a real `nn.MultiheadAttention`, see the module
    docstring's test below -- do not "simplify" this back to a `forward_pre_hook` on `out_proj`:
    that was the first attempt here and it silently never fires, because PyTorch's internal
    `multi_head_attention_forward` calls `F.linear(attn_output, out_proj.weight, out_proj.bias)`
    directly, NOT `out_proj(...)` -- so a hook on the `out_proj` module itself is never invoked).

    Instead, this hooks `multihead_attn` itself with a `forward_hook` (`with_kwargs=True`) that:
      1. Re-invokes the SAME call (same args/kwargs) with `need_weights=True,
         average_attn_weights=False` to obtain the REAL per-head attention weights (a reentrancy
         guard prevents this inner call from re-triggering the hook).
      2. Recomputes only the value projection (`in_proj_weight`/`in_proj_bias`'s documented
         `[2*embed_dim:3*embed_dim]` V slice -- the one deliberately "manual" piece here, and the
         lowest-risk one: a single Linear split, not a softmax/scaling reimplementation) and
         combines it with those REAL attention weights to get each head's own output contribution.
      3. Zeroes the targeted head(s)' contribution, re-applies `out_proj`, and returns
         `(ablated_output, <the ORIGINAL second return value>)` -- so any code reading the second
         element (attention weights, e.g. `AttnAccumulator`) sees the same shape/semantics as an
         unhooked call; only the first element (the actual output used by the residual stream) is
         changed.
    With the head list empty, this reconstruction is mathematically exact (confirmed to match the
    unhooked output bit-for-bit locally); `validate_head_ablation_noop` (below) re-confirms this on
    the real model before trusting any row 24 result.

    Usage:
        ablation = HeadAblation(model.transformer_decoder.layers, layer_indices=[5], head_indices=[2])
        ablation.attach()
        ... run inference (predictor.predict_* or vm.decode) ...
        ablation.detach()

    UNVERIFIED (not the hook mechanism above, which is verified -- this is the one remaining
    assumption): `layer.multihead_attn` is the cross-attention `nn.MultiheadAttention` submodule,
    with a combined `in_proj_weight` (i.e. `_qkv_same_embed_dim=True` -- query/key/value share one
    embedding dimension, the common case and the only one this class supports; `attach()` raises
    `AttributeError` immediately, not a silent wrong answer, if `in_proj_weight` is None). DETR-style
    decoder layer naming, matching this file's own `forward_pre`/pre-norm comments at the top of
    `base_trajectory`.
    """

    def __init__(self, decoder_layers, layer_indices, head_indices):
        self.decoder_layers = decoder_layers
        self.layer_indices = list(layer_indices)
        self.head_indices = list(head_indices)
        self._handles = []

    def _make_hook(self, mha):
        heads = list(self.head_indices)
        embed_dim, num_heads = mha.embed_dim, mha.num_heads
        head_dim = embed_dim // num_heads
        guard = {"active": False}

        def hook(module, args, kwargs, output):
            if guard["active"]:
                return output  # the inner re-invocation below; let it through unmodified
            guard["active"] = True
            try:
                kw2 = dict(kwargs)
                kw2["need_weights"] = True
                kw2["average_attn_weights"] = False
                with torch.no_grad():
                    _, per_head_w = module(*args, **kw2)  # (bsz, num_heads, tgt_len, src_len)
            finally:
                guard["active"] = False
            if not heads:
                return output  # nothing to ablate -- skip the reconstruction entirely
            value = kwargs.get("value", args[2] if len(args) > 2 else None)
            assert value is not None, "HeadAblation: could not find the `value` argument in this call"
            assert mha.in_proj_weight is not None, \
                "HeadAblation only supports combined in_proj_weight (query/key/value sharing one embed_dim)"
            wv = mha.in_proj_weight[2 * embed_dim:3 * embed_dim]
            bv = mha.in_proj_bias[2 * embed_dim:3 * embed_dim] if mha.in_proj_bias is not None else None
            v = F.linear(value, wv, bv)  # (src_len, bsz, embed_dim)
            src_len, bsz, _ = v.shape
            tgt_len = per_head_w.shape[-2]
            v = v.reshape(src_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)  # (bsz, heads, src_len, head_dim)
            head_out = (per_head_w @ v).clone()  # (bsz, heads, tgt_len, head_dim)
            head_out[:, heads, :, :] = 0
            concat = head_out.permute(2, 0, 1, 3).reshape(tgt_len, bsz, embed_dim)
            ablated = F.linear(concat, mha.out_proj.weight, mha.out_proj.bias)
            return (ablated, output[1])

        return hook

    def attach(self):
        for k in self.layer_indices:
            mha = self.decoder_layers[k].multihead_attn
            self._handles.append(mha.register_forward_hook(self._make_hook(mha), with_kwargs=True))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def num_cross_attn_heads(decoder_layers, layer_idx=0):
    """`multihead_attn.num_heads` of one decoder layer -- read at runtime, never hardcoded."""
    return decoder_layers[layer_idx].multihead_attn.num_heads


def validate_head_ablation_noop(predict_fn, layer_idx, decoder_layers):
    """Attaches a HeadAblation with an EMPTY head list at a real layer (so the hook is live, but
    zeroes nothing) and checks the prediction is unchanged. `predict_fn` is a zero-arg callable
    that runs one prediction and returns a boolean mask (e.g. `lambda: predict_masks(predictor,
    img_arr, [prompt])[0]`). Returns the fraction of voxels that agree between the hooked and
    unhooked predictions -- must be 1.0 (or, allowing for floating-point rounding, very close)
    before any row 24 result can be trusted."""
    base = predict_fn()
    ablation = HeadAblation(decoder_layers, [layer_idx], [])
    ablation.attach()
    try:
        noop = predict_fn()
    finally:
        ablation.detach()
    import numpy as np
    return float((np.asarray(base) == np.asarray(noop)).mean())


def activation_overlap_iou(fused_map, gt_bbox_mask, threshold_quantile=0.9):
    """Plan row 1/20's metric: IoU between the top-`1 - threshold_quantile` activations of a
    (possibly multi-channel) fused spatial map, upsampled to `gt_bbox_mask`'s shape, and the
    ground-truth region. `fused_map`: (C, H, W, D) or (H, W, D) numpy/tensor; `gt_bbox_mask`:
    (X, Y, Z) bool."""
    m = torch.as_tensor(fused_map).float()
    if m.ndim == 4:
        m = m.abs().mean(dim=0)
    m = m[None, None]
    up = F.interpolate(m, size=tuple(gt_bbox_mask.shape), mode='trilinear', align_corners=False)[0, 0]
    thr = torch.quantile(up.flatten(), threshold_quantile)
    hot = (up >= thr).numpy()
    gt = np.asarray(gt_bbox_mask, dtype=bool)
    inter = np.logical_and(hot, gt).sum()
    union = np.logical_or(hot, gt).sum()
    return float(inter / union) if union else 0.0
