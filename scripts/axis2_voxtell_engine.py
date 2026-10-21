"""
Axis II: Sensitivity to Prompt Variation — Core VoxTell Engine & Reproducibility Manager.

Built against VoxTell v0.1.0 (`voxtell/model/voxtell_model.py`, `voxtell/model/transformer.py`,
`voxtell/inference/predictor.py`).

Capabilities:
  1. Volume & Segmentation Loading in VoxTell's exact RAS-reoriented voxel space:
     - Uses `NibabelIOWithReorient` when available, with a pure-nibabel fallback
       that produces the identical `(S, A, R)` array layout.
  2. Single-Encoder-Pass Sliding-Window Inference (`run_sliding_window_multi`):
     - Encodes each 192^3 patch ONCE through `model.encoder` (the expensive 3D ResNet),
       then runs arbitrary prompt embeddings, transformer interventions, or decoder
       ablation settings through the lightweight prompt/mask decoders.
     - Supports 4-way Decoder Ablation (`both_on`, `image_only`, `text_only`, `both_off`).
     - Supports MC Dropout (`enable_mc_dropout=True`) by activating all `nn.Dropout`
       modules in `transformer_decoder` (plus optional spatial feature dropout on
       `project_bottleneck_embed` / decoder stages) while keeping BatchNorm/InstanceNorm
       and encoder weights frozen.
     - Captures intermediate representations at every network stage:
       * Stage 0: Frozen Qwen3-Embedding-4B output (`qwen3_embed`, dim 2560)
       * Stage 1: `project_text_embed` output (`tgt0`, dim 2048)
       * Stages 2..7: Output of each of the 6 `TransformerDecoderLayer`s (`layer_1`..`layer_6`, dim 2048) + cross-attention weights
       * Stage 8: Post-LayerNorm mask embedding (`mask_embedding`, dim 2048)
       * Stages 9..13: Projected per-stage U-Net fusion vectors (`proj_stage_0`..`proj_stage_4`)
         and fused 3D spatial feature maps (`einsum('b c h w d, b n c -> b n h w d', x, mask_emb)`)
  3. Reproducible Multi-Format Artifact Saving (`ExperimentOutputManager`):
     - Each experiment gets its own dedicated folder (`exp1_side_word_comparison/`,
       `exp2_synonym_shift/`, `exp3_layer_localization/`, `exp4_mc_dropout_uncertainty/`,
       `exp5_decoder_ablation/`) with:
         * `metrics.csv` & `summary.csv`
         * `results.json` & `summary.json`
         * `REPORT.md` (human-readable Markdown tables & findings)
         * `masks/` (compressed RLE in CSV/JSON + `.npz` masks/probability/uncertainty maps + optional `.nii.gz`)
         * `checkpoints/` (per-subject JSON/CSV checkpoints so interrupted runs resume automatically)
"""

from __future__ import annotations

import base64
import contextlib
import csv
import json
import math
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from einops import rearrange, repeat
except ImportError:
    def rearrange(tensor: torch.Tensor, pattern: str) -> torch.Tensor:
        if pattern == "b c d h w -> b h w d c":
            return tensor.permute(0, 3, 4, 2, 1).contiguous()
        if pattern == "b h w d c -> (h w d) b c":
            b, h, w, d, c = tensor.shape
            return tensor.reshape(b, h * w * d, c).permute(1, 0, 2).contiguous()
        raise ValueError(f"Unsupported rearrange pattern: {pattern}")

    def repeat(tensor: torch.Tensor, pattern: str) -> torch.Tensor:
        if pattern == "b n dim -> n b dim":
            return tensor.permute(1, 0, 2).contiguous()
        if pattern == "n b dim -> b n dim":
            return tensor.permute(1, 0, 2).contiguous()
        raise ValueError(f"Unsupported repeat pattern: {pattern}")


# ---------------------------------------------------------------------------
# Orientation & Anatomical Axis Constants
# ---------------------------------------------------------------------------
# NibabelIOWithReorient (nnU-Net's reader) returns arrays in (S, A, R) axis order:
#   axis 0: Superior-Inferior (S)
#   axis 1: Anterior-Posterior (A)
#   axis 2: Right-Left (R)  <-- increasing index moves toward anatomical Right (R)
AXCODES = ("S", "A", "R")
RL_AXIS = 2
_OPPOSITE = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}


# ---------------------------------------------------------------------------
# RLE Mask Encoding / Decoding (Lossless, Compact 3D Mask Serialization)
# ---------------------------------------------------------------------------
def encode_mask_rle(mask: np.ndarray) -> str:
    """Run-length-encode a 3D boolean mask (C-order flatten), zlib-compress, base64-encode."""
    flat = np.asarray(mask, dtype=bool).ravel(order="C")
    change_idx = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], change_idx, [flat.size]))
    runs = np.diff(boundaries).astype(np.uint32)
    if flat.size and flat[0]:
        runs = np.concatenate(([0], runs)).astype(np.uint32)
    raw = zlib.compress(runs.tobytes(), level=9)
    return base64.b64encode(raw).decode("ascii")


def decode_mask_rle(encoded: str, shape: Tuple[int, ...]) -> np.ndarray:
    """Reconstruct an exact 3D boolean mask from its base64 RLE string and shape."""
    runs = np.frombuffer(zlib.decompress(base64.b64decode(encoded)), dtype=np.uint32)
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    pos, val = 0, False
    for r in runs:
        if val:
            flat[pos:pos + int(r)] = True
        pos += int(r)
        val = not val
    return flat.reshape(shape)


def shape_to_str(shape: Sequence[int]) -> str:
    return "x".join(str(int(s)) for s in shape)


def str_to_shape(shape_str: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in shape_str.split("x"))


# ---------------------------------------------------------------------------
# Volume & Ground-Truth Loading in VoxTell RAS Space
# ---------------------------------------------------------------------------
def load_volume_ras(path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a NIfTI image in VoxTell's exact RAS-reoriented array space `(1, X, Y, Z)`.
    Uses `NibabelIOWithReorient` when installed, or equivalent `nib.as_closest_canonical`
    + axis transpose `[2, 1, 0]` (which is what SimpleITK/NibabelIOWithReorient does).
    """
    try:
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        reader = NibabelIOWithReorient()
        data, props = reader.read_images([str(path)])
        return np.asarray(data, dtype=np.float32), props
    except ImportError:
        nii = nib.load(str(path))
        ras_nii = nib.as_closest_canonical(nii)
        arr = np.asarray(ras_nii.dataobj, dtype=np.float32)
        # NibabelIOWithReorient transposes canonical (R, A, S) -> (S, A, R) via .T ([2, 1, 0])
        arr_sar = np.ascontiguousarray(arr.transpose(2, 1, 0))[None]
        props = {"nibabel_stuff": {"reoriented_affine": ras_nii.affine}, "spacing": ras_nii.header.get_zooms()[:3][::-1]}
        return arr_sar, props


def load_seg_ras(path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a NIfTI segmentation label map in VoxTell's RAS-reoriented space `(X, Y, Z)`."""
    try:
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        reader = NibabelIOWithReorient()
        data, props = reader.read_seg(str(path))
        arr = np.asarray(data)
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        return arr.astype(np.int32), props
    except ImportError:
        nii = nib.load(str(path))
        ras_nii = nib.as_closest_canonical(nii)
        arr = np.asarray(ras_nii.dataobj)
        arr_sar = np.ascontiguousarray(arr.transpose(2, 1, 0)).astype(np.int32)
        props = {"nibabel_stuff": {"reoriented_affine": ras_nii.affine}, "spacing": ras_nii.header.get_zooms()[:3][::-1]}
        return arr_sar, props


def save_mask_nifti(mask: np.ndarray, output_path: Union[str, Path], reference_nifti_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Save a 3D mask (or float map) in `(S, A, R)` space back to a NIfTI `.nii.gz` file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(mask)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)
    # Undo (S, A, R) -> canonical (R, A, S)
    arr_ras = np.ascontiguousarray(arr.transpose(2, 1, 0))
    affine = np.eye(4)
    if reference_nifti_path is not None and Path(reference_nifti_path).exists():
        ref_ras = nib.as_closest_canonical(nib.load(str(reference_nifti_path)))
        affine = ref_ras.affine
    nii = nib.Nifti1Image(arr_ras, affine=affine)
    nib.save(nii, str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Segmentation & Spatial Geometry Metrics
# ---------------------------------------------------------------------------
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = np.asarray(pred, dtype=bool)
    gt_b = np.asarray(gt, dtype=bool)
    denom = int(pred_b.sum()) + int(gt_b.sum())
    if denom == 0:
        return 1.0
    inter = int(np.logical_and(pred_b, gt_b).sum())
    return float(2.0 * inter / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = np.asarray(pred, dtype=bool)
    gt_b = np.asarray(gt, dtype=bool)
    union = int(np.logical_or(pred_b, gt_b).sum())
    if union == 0:
        return 1.0
    inter = int(np.logical_and(pred_b, gt_b).sum())
    return float(inter / union)


def mask_geometry(mask: np.ndarray, rl_midline: Optional[float] = None) -> Dict[str, Any]:
    """
    Compute volume, voxel centroid, bounding box, and anatomical side (L/R)
    for a 3D binary mask in `(S, A, R)` voxel space.
    """
    mask_b = np.asarray(mask, dtype=bool)
    n_vox = int(mask_b.sum())
    shape = mask_b.shape
    mid_rl = float(shape[RL_AXIS]) / 2.0 if rl_midline is None else float(rl_midline)

    if n_vox == 0:
        return {
            "n_voxels": 0,
            "is_empty": True,
            "centroid_0": np.nan,
            "centroid_1": np.nan,
            "centroid_2": np.nan,
            "bbox_0min": -1,
            "bbox_1min": -1,
            "bbox_2min": -1,
            "bbox_0max": -1,
            "bbox_1max": -1,
            "bbox_2max": -1,
            "side_rl": "empty",
            "frac_right_of_midline": np.nan,
            "frac_left_of_midline": np.nan,
        }

    coords = np.nonzero(mask_b)
    c0, c1, c2 = (float(c.mean()) for c in coords)
    b0min, b1min, b2min = (int(c.min()) for c in coords)
    b0max, b1max, b2max = (int(c.max()) for c in coords)

    # In (S, A, R), axis 2 increases toward Right (R).
    # However, let's check both midpoint and fraction of voxels on each half of RL_AXIS.
    rl_coords = coords[RL_AXIS]
    frac_high = float((rl_coords > mid_rl).mean())
    frac_low = float((rl_coords <= mid_rl).mean())
    side_rl = "R" if c2 > mid_rl else "L"

    return {
        "n_voxels": n_vox,
        "is_empty": False,
        "centroid_0": round(c0, 3),
        "centroid_1": round(c1, 3),
        "centroid_2": round(c2, 3),
        "bbox_0min": b0min,
        "bbox_1min": b1min,
        "bbox_2min": b2min,
        "bbox_0max": b0max,
        "bbox_1max": b1max,
        "bbox_2max": b2max,
        "side_rl": side_rl,
        "frac_right_of_midline": round(frac_high, 4),
        "frac_left_of_midline": round(frac_low, 4),
    }


def centroid_distance(geom_a: Dict[str, Any], geom_b: Dict[str, Any], vol_shape: Optional[Sequence[int]] = None) -> Tuple[float, float]:
    """
    Return `(euclidean_dist_vox, normalized_dist_by_diag)` between two mask geometries.
    Returns `(np.nan, np.nan)` if either mask is empty.
    """
    if geom_a.get("is_empty", True) or geom_b.get("is_empty", True):
        return (np.nan, np.nan)
    ca = np.array([geom_a["centroid_0"], geom_a["centroid_1"], geom_a["centroid_2"]], dtype=np.float64)
    cb = np.array([geom_b["centroid_0"], geom_b["centroid_1"], geom_b["centroid_2"]], dtype=np.float64)
    dist = float(np.linalg.norm(ca - cb))
    if vol_shape is not None:
        diag = float(np.linalg.norm(np.asarray(vol_shape, dtype=np.float64)))
        norm_dist = dist / max(diag, 1e-8)
    else:
        norm_dist = np.nan
    return (round(dist, 4), round(norm_dist, 6) if not np.isnan(norm_dist) else np.nan)


# ---------------------------------------------------------------------------
# Embedding Distance Metrics (Cosine Distance, L2 Distance, Pearson Correlation)
# ---------------------------------------------------------------------------
def vector_distances(u: Union[np.ndarray, torch.Tensor], v: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """
    Compute cosine similarity, cosine distance, L2 distance, relative L2 distance,
    and Pearson correlation between two 1D feature vectors `u` and `v`.
    """
    ua = u.detach().cpu().float().numpy().ravel() if isinstance(u, torch.Tensor) else np.asarray(u, dtype=np.float32).ravel()
    va = v.detach().cpu().float().numpy().ravel() if isinstance(v, torch.Tensor) else np.asarray(v, dtype=np.float32).ravel()

    norm_u = float(np.linalg.norm(ua))
    norm_v = float(np.linalg.norm(va))
    if norm_u < 1e-12 or norm_v < 1e-12:
        cos_sim = 0.0
    else:
        cos_sim = float(np.dot(ua, va) / (norm_u * norm_v))
    cos_dist = 1.0 - cos_sim
    l2_dist = float(np.linalg.norm(ua - va))
    rel_l2 = float(l2_dist / (0.5 * (norm_u + norm_v) + 1e-8))

    ua_c = ua - ua.mean()
    va_c = va - va.mean()
    denom_corr = float(np.linalg.norm(ua_c) * np.linalg.norm(va_c))
    pearson_r = float(np.dot(ua_c, va_c) / denom_corr) if denom_corr > 1e-12 else 0.0

    return {
        "cosine_similarity": round(cos_sim, 6),
        "cosine_distance": round(cos_dist, 6),
        "l2_distance": round(l2_dist, 6),
        "relative_l2_distance": round(rel_l2, 6),
        "pearson_correlation": round(pearson_r, 6),
    }


# ---------------------------------------------------------------------------
# VoxTell Internal Forward & Ablation / Uncertainty Hooks
# ---------------------------------------------------------------------------
def preprocess_volume(predictor, img_arr: np.ndarray) -> Tuple[torch.Tensor, List[List[int]], Tuple[int, ...]]:
    """
    Preprocess `img_arr` `(1, X, Y, Z)` using VoxTell's `predictor.preprocess`, or
    equivalent z-score + nonzero crop if running in standalone test mode.
    """
    if hasattr(predictor, "preprocess"):
        return predictor.preprocess(img_arr)
    data = img_arr.astype(np.float32)
    orig_shape = data.shape[1:]
    nz = np.nonzero(data[0] != 0)
    if len(nz[0]) == 0:
        bbox = [[0, s] for s in orig_shape]
        cropped = data
    else:
        bbox = [[int(c.min()), int(c.max()) + 1] for c in nz]
        slicer = (slice(None), *[slice(b[0], b[1]) for b in bbox])
        cropped = data[slicer]
    mean, std = cropped.mean(), cropped.std()
    cropped = (cropped - mean) / max(float(std), 1e-8)
    return torch.from_numpy(cropped), bbox, orig_shape


def encode_patch(model: nn.Module, patch: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Run VoxTell's 3D ResidualEncoder and bottleneck projection on a single 3D patch `(1, 1, D, H, W)`.
    Returns `(skips, memory)` where `memory` has shape `(S, 1, query_dim)` (`S = 12*12*12 = 1728`).
    """
    skips = model.encoder(patch)
    selected_feature = skips[model.selected_decoder_layer]
    bottleneck_embed = rearrange(selected_feature, "b c d h w -> b h w d c")
    bottleneck_embed = model.project_bottleneck_embed(bottleneck_embed)
    bottleneck_embed = rearrange(bottleneck_embed, "b h w d c -> (h w d) b c")
    return skips, bottleneck_embed


def run_prompt_decoder_trajectory(
    model: nn.Module,
    text_embeddings: torch.Tensor,
    memory: torch.Tensor,
    text_decoder_on: bool = True,
) -> Dict[str, Any]:
    """
    Pass precomputed Qwen3 embeddings `text_embeddings` `(1, N, 2560)` and image `memory`
    `(S, 1, 2048)` through VoxTell's text projection and 6-layer `transformer_decoder`.

    If `text_decoder_on=False` (for Experiment V's `image_only` or `both_off` setting):
      - Bypasses the 6 cross-attention transformer layers (`transformer_decoder.layers`)
        and projects `tgt0` directly through `transformer_decoder.norm`, testing what happens
        when the text prompt decoder's cross-attention pathway is turned off.

    Returns dict with:
      - `qwen3_embed`: `(N, 2560)`
      - `tgt0`: `(N, 2048)` (after `project_text_embed`)
      - `layer_states`: list of 6 tensors `(N, 2048)` after layers 0..5
      - `attns`: list of 6 tensors `(N, S)` cross-attention weights
      - `mask_embedding`: `(1, N, 2048)` (after `transformer_decoder.norm`)
      - `stage_mask_embeddings`: list of 5 tensors `(1, N, C_stage)` from `project_to_decoder_channels`
    """
    te = text_embeddings.squeeze(2) if text_embeddings.ndim == 4 else text_embeddings  # (1, N, D)
    qwen3_embed = te[0]  # (N, 2560)

    text_embed = repeat(te, "b n dim -> n b dim")  # (N, 1, 2560)
    tgt0 = model.project_text_embed(text_embed)    # (N, 1, 2048)

    layer_states: List[torch.Tensor] = []
    attns: List[torch.Tensor] = []

    t = tgt0
    if text_decoder_on:
        for layer in model.transformer_decoder.layers:
            t, w = layer(
                t,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=model.pos_embed,
                query_pos=None,
            )
            layer_states.append(t[:, 0, :])  # (N, 2048)
            attns.append(w[0])               # (N, S)
    else:
        # Text prompt decoder off: keep tgt0 unchanged across all 6 layers
        S = memory.shape[0]
        N = tgt0.shape[0]
        uniform_attn = torch.full((N, S), 1.0 / float(S), device=tgt0.device, dtype=tgt0.dtype)
        for _ in model.transformer_decoder.layers:
            layer_states.append(t[:, 0, :])
            attns.append(uniform_attn)

    normed = model.transformer_decoder.norm(t) if model.transformer_decoder.norm is not None else t
    mask_embedding = repeat(normed, "n b dim -> b n dim")  # (1, N, 2048)

    stage_mask_embeddings = [
        proj(mask_embedding) for proj in model.project_to_decoder_channels
    ]

    return {
        "qwen3_embed": qwen3_embed,
        "tgt0": tgt0[:, 0, :],
        "layer_states": layer_states,
        "attns": attns,
        "mask_embedding": mask_embedding,
        "stage_mask_embeddings": stage_mask_embeddings,
    }


def decode_with_ablation(
    model: nn.Module,
    skips: List[torch.Tensor],
    stage_mask_embeddings: List[torch.Tensor],
    image_decoder_on: bool = True,
    capture_fusion_norms: bool = False,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    """
    Run VoxTell's `VoxTellDecoder` across all prompts in `stage_mask_embeddings`.

    Supports the 4 settings of Experiment V (Section 6 of Axis II doc):
      - `image_decoder_on=True`: standard U-Net convolutional upsampling + skip connections
        fused with `stage_mask_embeddings` at all 5 maskformer stages.
      - `image_decoder_on=False`: bypasses the U-Net decoder's learned convolutional
        refinement stages (`decoder.stages` / `decoder.transpconvs`) by trilinearly
        upsampling the raw encoder features to match each stage's channel dimension
        via parameter-free channel pooling/tiling and taking the direct dot-product
        similarity map with `stage_mask_embeddings[0]`. This isolates the text-query
        pathway without the learned U-Net image decoder.

    If `capture_fusion_norms=True`, also returns per-stage fused spatial feature summaries
    `(N, C_or_H, ...)` for Experiment III (Layer Localization across U-Net fusion stages).
    """
    decoder = model.decoder
    num_prompts = stage_mask_embeddings[0].shape[1]

    if not image_decoder_on:
        # Image decoder OFF: compute direct token-space similarity between encoder
        # bottleneck memory and prompt mask_embedding, then upsample to patch size.
        # Specifically, use skips[0] (full-res stem features, 32 channels) directly
        # without any U-Net decoder stages, dotted with stage_mask_embeddings[0] (32 channels).
        stem_feat = skips[0]  # (1, 32, D, H, W)
        final_mask_emb = stage_mask_embeddings[0]  # (1, N, 32)
        seg_pred = torch.einsum("b c h w d, b n c -> b n h w d", stem_feat, final_mask_emb)
        fusion_summaries = None
        if capture_fusion_norms:
            # Summarize across the 5 stages using interpolated encoder skips
            fusion_summaries = []
            rev_embs = stage_mask_embeddings[::-1]
            for s_idx in range(len(decoder.stages)):
                enc_feat = skips[-(s_idx + 2)]
                me = rev_embs[s_idx]  # (1, N, C_stage)
                b_sz, n_p, c_dim = me.shape
                if s_idx == len(decoder.stages) - 1:
                    f_map = torch.einsum("b c h w d, b n c -> b n h w d", enc_feat, me)
                else:
                    me_r = me.view(b_sz, n_p, decoder.num_heads, -1).mean(dim=2)
                    f_map = torch.einsum("b c h w d, b n c -> b n h w d", enc_feat, me_r)
                # Spatial profile flattened to 12x12x12 grid for fast comparison
                pooled = F.adaptive_avg_pool3d(f_map[0].float(), (6, 6, 6)).flatten(1)  # (N, 216)
                fusion_summaries.append(pooled)
        return seg_pred[0], fusion_summaries

    # Standard U-Net image decoder ON
    outs = []
    per_prompt_fusions: List[List[torch.Tensor]] = [[] for _ in range(len(decoder.stages))]

    for prompt_idx in range(num_prompts):
        prompt_embeds = [m[:, prompt_idx:prompt_idx + 1] for m in stage_mask_embeddings]
        lres_input = skips[-1]
        mask_embs_rev = prompt_embeds[::-1]
        seg_outputs = []

        for stage_idx in range(len(decoder.stages)):
            x = decoder.transpconvs[stage_idx](lres_input)
            x = torch.cat((x, skips[-(stage_idx + 2)]), dim=1)
            x = decoder.stages[stage_idx](x)

            if stage_idx == (len(decoder.stages) - 1):
                seg_pred = torch.einsum("b c h w d, b n c -> b n h w d", x, mask_embs_rev[-1])
                seg_outputs.append(seg_pred)
                if capture_fusion_norms:
                    pooled = F.adaptive_avg_pool3d(seg_pred[0].float(), (6, 6, 6)).flatten(1)
                    per_prompt_fusions[stage_idx].append(pooled)
            elif stage_idx >= len(decoder.stages) - len(mask_embs_rev):
                mask_embedding = mask_embs_rev.pop(0)
                batch_size, _, _ = mask_embedding.shape
                mask_embedding_reshaped = mask_embedding.view(batch_size, decoder.num_heads, -1)
                fusion_features = torch.einsum("b c h w d, b n c -> b n h w d", x, mask_embedding_reshaped)
                if capture_fusion_norms:
                    # Mean across the 32 fusion heads -> (1, D, H, W), then pool to 6x6x6
                    pooled = F.adaptive_avg_pool3d(fusion_features.float().mean(dim=1), (6, 6, 6)).flatten(1)
                    per_prompt_fusions[stage_idx].append(pooled)
                x = torch.cat((x, fusion_features), dim=1)
                seg_outputs.append(decoder.seg_layers[stage_idx](x))

            lres_input = x

        seg_outputs = seg_outputs[::-1]
        outs.append(seg_outputs[0])

    logits = torch.cat(outs, dim=1)[0]  # (N, D, H, W)
    fusion_summaries = None
    if capture_fusion_norms:
        fusion_summaries = [torch.cat(stage_list, dim=0) for stage_list in per_prompt_fusions]
    return logits, fusion_summaries


@contextlib.contextmanager
def mc_dropout_context(model: nn.Module, enable: bool = True, spatial_dropout_p: float = 0.0):
    """
    Context manager for Experiment IV (MC Dropout Uncertainty Estimation):
      - Keeps the entire network in `.eval()` mode (so InstanceNorm/BatchNorm stay frozen),
      - Explicitly switches all `nn.Dropout` / `_DropoutNd` modules to `.train(True)` AND
        Temporarily patches `nn.MultiheadAttention.forward` on `transformer_decoder` layers
        if needed because PyTorch's `MultiheadAttention` passes `training=self.training`
        into `F.multi_head_attention_forward`. Setting `layer.multihead_attn.train(True)`
        and `layer.self_attn.train(True)` alongside `nn.Dropout.train(True)` activates both
        attention-weight dropout and residual/FFN dropout (`dropout`, `dropout1`, `dropout2`, `dropout3`).
      - Optionally injects feature dropout (`spatial_dropout_p > 0`) on `project_bottleneck_embed`
        if additional stochasticity is requested.
    """
    if not enable:
        yield
        return

    toggled_modules: List[Tuple[nn.Module, bool]] = []
    hooks = []
    try:
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.MultiheadAttention)):
                toggled_modules.append((m, m.training))
                m.train(True)

        if spatial_dropout_p > 0.0 and hasattr(model, "project_bottleneck_embed"):
            def _drop_hook(_mod, _inp, out):
                return F.dropout(out, p=spatial_dropout_p, training=True)
            hooks.append(model.project_bottleneck_embed.register_forward_hook(_drop_hook))

        yield
    finally:
        for h in hooks:
            h.remove()
        for m, was_training in toggled_modules:
            m.train(was_training)


def logits_to_full_masks(
    logits: torch.Tensor,
    bbox: Sequence[Sequence[int]],
    orig_shape: Tuple[int, ...],
    return_probs: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert cropped sliding-window `logits` `(N, Xc, Yc, Zc)` into full-volume
    boolean masks `(N, X, Y, Z)` (and optionally float16 probabilities `(N, X, Y, Z)`).
    """
    with torch.no_grad():
        N = logits.shape[0]
        cropped_masks = np.zeros(logits.shape, dtype=np.uint8)
        cropped_probs = np.zeros(logits.shape, dtype=np.float16) if return_probs else None
        for i in range(0, N, 8):
            chunk = logits[i:i + 8].float()
            probs = torch.sigmoid(chunk)
            cropped_masks[i:i + 8] = (probs > 0.5).cpu().numpy().astype(np.uint8)
            if return_probs:
                cropped_probs[i:i + 8] = probs.cpu().numpy().astype(np.float16)

    slicer = (slice(None), *[slice(int(b[0]), int(b[1])) for b in bbox])
    full_masks = np.zeros((N, *orig_shape), dtype=bool)
    full_masks[slicer] = cropped_masks.astype(bool)

    if return_probs:
        full_probs = np.zeros((N, *orig_shape), dtype=np.float16)
        full_probs[slicer] = cropped_probs
        return full_masks, full_probs
    return full_masks


@torch.inference_mode()
def run_sliding_window_custom(
    predictor,
    data: torch.Tensor,
    text_embeddings: torch.Tensor,
    image_decoder_on: bool = True,
    text_decoder_on: bool = True,
    enable_mc_dropout: bool = False,
    spatial_dropout_p: float = 0.0,
    capture_internals: bool = False,
    prompt_chunk_size: int = 16,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """
    Run VoxTell sliding-window inference on preprocessed `data` `(1, Xc, Yc, Zc)`
    for `text_embeddings` `(1, N, 2560)`.

    Key optimization:
      - Encodes each 3D patch through `model.encoder` ONCE per patch,
      - Processes all `N` prompts in chunks of `prompt_chunk_size` through the
        transformer decoder and U-Net decoder.
      - Supports decoder ablation (`image_decoder_on`, `text_decoder_on`),
      - Supports MC Dropout (`enable_mc_dropout=True`),
      - Optionally captures and averages internal stage representations across patches
        (`capture_internals=True`).
    """
    device = predictor.device
    model = predictor.network.to(device)
    patch_size = tuple(int(p) for p in predictor.patch_size)

    try:
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        data_p, revert = pad_nd_image(data, patch_size, "constant", {"value": 0}, True, None)
        slicers = predictor._internal_get_sliding_window_slicers(data_p.shape[1:])
        # Note: nnunetv2.compute_gaussian uses float16 on non-CPU devices unless computed on CPU first,
        # and some MPS builds do not support float16 in certain ops. Compute on CPU then cast to float32 on device.
        g_dev = device if device.type == "cuda" else torch.device("cpu")
        gaussian = compute_gaussian(
            patch_size, sigma_scale=1.0 / 8.0, value_scaling_factor=10.0, device=g_dev
        ).float().to(device)
    except ImportError:
        # Standalone fallback when acvl_utils / nnunetv2 are not installed locally
        c, x, y, z = data.shape
        px, py, pz = patch_size
        pad_x, pad_y, pad_z = max(0, px - x), max(0, py - y), max(0, pz - z)
        data_p = F.pad(data, (0, pad_z, 0, pad_y, 0, pad_x), mode="constant", value=0)
        revert = (slice(None), slice(0, x), slice(0, y), slice(0, z))
        slicers = [(slice(None), slice(0, px), slice(0, py), slice(0, pz))]
        gaussian = torch.ones(patch_size, dtype=torch.float32, device=device)

    data_p = data_p.to(device)
    N = text_embeddings.shape[1]
    logits = torch.zeros((N, *data_p.shape[1:]), dtype=torch.float32, device=device)
    n_pred = torch.zeros(data_p.shape[1:], dtype=torch.float32, device=device)

    internals_accum: Optional[Dict[str, Any]] = None
    n_patches = len(slicers)

    autocast_ctx = torch.autocast(device.type, enabled=True) if device.type == "cuda" else contextlib.nullcontext()

    with mc_dropout_context(model, enable=enable_mc_dropout, spatial_dropout_p=spatial_dropout_p):
        for patch_idx, slicer in enumerate(slicers):
            patch = torch.clone(data_p[slicer][None], memory_format=torch.contiguous_format).to(device)
            with autocast_ctx:
                skips, memory = encode_patch(model, patch)

                patch_logits_chunks = []
                qwen3_list, tgt0_list, mask_emb_list = [], [], []
                layer_lists: List[List[torch.Tensor]] = [[] for _ in range(len(model.transformer_decoder.layers))]
                attn_lists: List[List[torch.Tensor]] = [[] for _ in range(len(model.transformer_decoder.layers))]
                fusion_lists: List[List[torch.Tensor]] = [[] for _ in range(len(model.decoder.stages))]

                for start in range(0, N, prompt_chunk_size):
                    emb_chunk = text_embeddings[:, start:start + prompt_chunk_size].to(device)
                    traj = run_prompt_decoder_trajectory(
                        model, emb_chunk, memory, text_decoder_on=text_decoder_on
                    )
                    pred_chunk, f_summaries = decode_with_ablation(
                        model,
                        skips,
                        traj["stage_mask_embeddings"],
                        image_decoder_on=image_decoder_on,
                        capture_fusion_norms=capture_internals,
                    )
                    patch_logits_chunks.append(pred_chunk.float())

                    if capture_internals:
                        qwen3_list.append(traj["qwen3_embed"].detach().float().cpu())
                        tgt0_list.append(traj["tgt0"].detach().float().cpu())
                        mask_emb_list.append(traj["mask_embedding"][0].detach().float().cpu())
                        for l_i, ls in enumerate(traj["layer_states"]):
                            layer_lists[l_i].append(ls.detach().float().cpu())
                        for l_i, att in enumerate(traj["attns"]):
                            attn_lists[l_i].append(att.detach().float().cpu())
                        if f_summaries is not None:
                            for s_i, fs in enumerate(f_summaries):
                                fusion_lists[s_i].append(fs.detach().float().cpu())

                patch_logits = torch.cat(patch_logits_chunks, dim=0)  # (N, px, py, pz)

            gz = gaussian.float()
            logits[(slice(None), *slicer[1:])] += patch_logits * gz
            n_pred[slicer[1:]] += gz

            if capture_internals:
                cur_qwen3 = torch.cat(qwen3_list, dim=0)
                cur_tgt0 = torch.cat(tgt0_list, dim=0)
                cur_mask_emb = torch.cat(mask_emb_list, dim=0)
                cur_layers = [torch.cat(ll, dim=0) for ll in layer_lists]
                cur_attns = [torch.cat(al, dim=0) for al in attn_lists]
                cur_fusions = [torch.cat(fl, dim=0) for fl in fusion_lists] if any(fusion_lists) else []

                if internals_accum is None:
                    internals_accum = {
                        "qwen3_embed": cur_qwen3,
                        "tgt0": cur_tgt0,
                        "layer_states": cur_layers,
                        "attns": cur_attns,
                        "mask_embedding": cur_mask_emb,
                        "fusion_summaries": cur_fusions,
                    }
                else:
                    internals_accum["layer_states"] = [
                        a + b for a, b in zip(internals_accum["layer_states"], cur_layers)
                    ]
                    internals_accum["attns"] = [
                        a + b for a, b in zip(internals_accum["attns"], cur_attns)
                    ]
                    internals_accum["mask_embedding"] = internals_accum["mask_embedding"] + cur_mask_emb
                    if cur_fusions:
                        internals_accum["fusion_summaries"] = [
                            a + b for a, b in zip(internals_accum["fusion_summaries"], cur_fusions)
                        ]

    logits = logits / n_pred.clamp_min(1e-6)
    logits = logits[(slice(None), *revert[1:])]

    if internals_accum is not None and n_patches > 1:
        internals_accum["layer_states"] = [x / float(n_patches) for x in internals_accum["layer_states"]]
        internals_accum["attns"] = [x / float(n_patches) for x in internals_accum["attns"]]
        internals_accum["mask_embedding"] = internals_accum["mask_embedding"] / float(n_patches)
        internals_accum["fusion_summaries"] = [x / float(n_patches) for x in internals_accum["fusion_summaries"]]

    return logits.cpu(), internals_accum


# ---------------------------------------------------------------------------
# Reproducibility & Multi-Format Output Manager (CSV, Masks, JSON, Markdown)
# ---------------------------------------------------------------------------
class ExperimentOutputManager:
    """
    Manages a dedicated output directory for a single Axis II experiment:
      - `metrics.csv`: row-level experimental results (with RLE-encoded 3D masks if applicable)
      - `summary.csv`: aggregated statistics across subjects/structures
      - `results.json`: complete structured JSON (metadata + config + summary + row records)
      - `REPORT.md`: formatted Markdown report with tables and interpretation
      - `masks/`: `.npz` compressed 3D mask/probability/uncertainty arrays and optional `.nii.gz` files
      - `checkpoints/`: per-subject checkpoint files so runs can stop and resume seamlessly
    """

    def __init__(self, base_results_dir: Union[str, Path], experiment_name: str) -> None:
        self.base_dir = Path(base_results_dir)
        self.exp_name = experiment_name
        self.exp_dir = self.base_dir / experiment_name
        self.masks_dir = self.exp_dir / "masks"
        self.ckpt_dir = self.exp_dir / "checkpoints"

        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def subject_checkpoint_path(self, subject_id: str) -> Path:
        return self.ckpt_dir / f"{subject_id}.json"

    def has_subject_checkpoint(self, subject_id: str) -> bool:
        return self.subject_checkpoint_path(subject_id).exists()

    def save_subject_checkpoint(self, subject_id: str, rows: List[Dict[str, Any]], extra: Optional[Dict[str, Any]] = None) -> Path:
        p = self.subject_checkpoint_path(subject_id)
        payload = {"subject_id": subject_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows}
        if extra:
            payload["extra"] = extra
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return p

    def load_subject_checkpoint(self, subject_id: str) -> Dict[str, Any]:
        return json.loads(self.subject_checkpoint_path(subject_id).read_text(encoding="utf-8"))

    def save_npz_bundle(self, filename: str, **arrays: np.ndarray) -> Path:
        if not filename.endswith(".npz"):
            filename = f"{filename}.npz"
        out_path = self.masks_dir / filename
        np.savez_compressed(str(out_path), **arrays)
        return out_path

    def save_nifti(self, filename: str, array_3d: np.ndarray, reference_nifti_path: Optional[Union[str, Path]] = None) -> Path:
        if not (filename.endswith(".nii.gz") or filename.endswith(".nii")):
            filename = f"{filename}.nii.gz"
        out_path = self.masks_dir / filename
        return save_mask_nifti(array_3d, out_path, reference_nifti_path=reference_nifti_path)

    def write_all_formats(
        self,
        metrics_rows: List[Dict[str, Any]],
        summary_rows: List[Dict[str, Any]],
        report_title: str,
        report_sections: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        metrics_filename: str = "metrics.csv",
        summary_filename: str = "summary.csv",
        json_filename: str = "results.json",
        md_filename: str = "REPORT.md",
    ) -> Dict[str, Path]:
        """
        Write CSV (`metrics.csv`, `summary.csv`), JSON (`results.json`), and Markdown (`REPORT.md`)
        into `self.exp_dir`.
        """
        metrics_csv_path = self.exp_dir / metrics_filename
        if metrics_rows:
            fieldnames = list(metrics_rows[0].keys())
            with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metrics_rows)

        summary_csv_path = self.exp_dir / summary_filename
        if summary_rows:
            s_fields = list(summary_rows[0].keys())
            with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=s_fields)
                writer.writeheader()
                writer.writerows(summary_rows)

        # Strip huge base64 RLE strings from the top-level JSON only if needed? No — keep a lightweight
        # summary.json AND full results.json so both human inspection and full mask reconstruction work!
        json_path = self.exp_dir / json_filename
        full_payload = {
            "experiment": self.exp_name,
            "title": report_title,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": metadata or {},
            "n_metrics_rows": len(metrics_rows),
            "n_summary_rows": len(summary_rows),
            "summary": summary_rows,
            "metrics": metrics_rows,
        }
        json_path.write_text(json.dumps(full_payload, indent=2, default=_json_default), encoding="utf-8")

        summary_json_path = self.exp_dir / "summary.json"
        summary_json_path.write_text(
            json.dumps(
                {
                    "experiment": self.exp_name,
                    "title": report_title,
                    "generated_at_utc": full_payload["generated_at_utc"],
                    "metadata": metadata or {},
                    "summary": summary_rows,
                },
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )

        md_path = self.exp_dir / md_filename
        md_lines = [
            f"# {report_title}",
            "",
            f"- **Experiment Folder:** `{self.exp_name}`",
            f"- **Generated (UTC):** `{full_payload['generated_at_utc']}`",
            f"- **Total Detail Rows (`{metrics_filename}`):** {len(metrics_rows)}",
            f"- **Total Summary Rows (`{summary_filename}`):** {len(summary_rows)}",
            f"- **Saved Formats:** `{metrics_filename}`, `{summary_filename}`, `{json_filename}`, `summary.json`, `{md_filename}`, `masks/`",
            "",
        ]
        for sec_heading, sec_body in report_sections:
            md_lines.append(f"## {sec_heading}")
            md_lines.append("")
            md_lines.append(sec_body)
            md_lines.append("")

        if summary_rows:
            md_lines.append("## Summary Table")
            md_lines.append("")
            md_lines.append(rows_to_markdown_table(summary_rows))
            md_lines.append("")

        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        return {
            "metrics_csv": metrics_csv_path,
            "summary_csv": summary_csv_path,
            "results_json": json_path,
            "summary_json": summary_json_path,
            "report_md": md_path,
            "masks_dir": self.masks_dir,
        }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def rows_to_markdown_table(rows: Sequence[Dict[str, Any]], max_rows: int = 60) -> str:
    """Render a list of dicts as a clean GitHub-flavored Markdown table."""
    if not rows:
        return "_No rows available._"
    # Exclude RLE mask columns from Markdown display
    cols = [k for k in rows[0].keys() if not k.endswith("_rle_b64")]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for r in rows[:max_rows]:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append("NaN" if math.isnan(v) else f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(rows)} rows. Full table in `summary.csv` / `metrics.csv`._")
    return "\n".join(lines)
