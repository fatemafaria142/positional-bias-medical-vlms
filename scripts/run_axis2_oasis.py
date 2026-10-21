"""
CLI & Notebook Entrypoint for Axis II: Sensitivity to Prompt Variation
on Neurite-OASIS (100 subjects) with VoxTell.

Can be invoked from a Kaggle notebook cell or terminal:
    python scripts/run_axis2_oasis.py \
        --oasis-dir /kaggle/working/neurite_oasis_100 \
        --results-dir /kaggle/working/results_axis2 \
        --model-dir /kaggle/working/checkpoints/voxtell_v1.1 \
        --experiments all \
        --structure-scope core_7 \
        --max-subjects 100

Or imported directly in the Kaggle notebook:
    from run_axis2_oasis import run_axis2_pipeline
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# Ensure `scripts/` is on sys.path
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from axis2_experiments import (
    run_experiment_1_side_words,
    run_experiment_2_and_3_synonyms_and_layers,
    run_experiment_4_mc_dropout_uncertainty,
    run_experiment_5_decoder_ablation,
)
from axis2_prompts import export_frozen_prompt_table


def load_voxtell_predictor_safe(
    model_dir: Union[str, Path],
    device: Optional[str] = None,
    fp16_text_backbone: bool = True,
):
    """
    Instantiate `VoxTellPredictor` and apply the fp16 Qwen3-Embedding-4B safeguard
    so it fits comfortably on a single Kaggle T4 / P100 GPU (15-16 GB VRAM) alongside
    the 3D segmentation network.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    from voxtell.inference.predictor import VoxTellPredictor

    if device:
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    predictor = VoxTellPredictor(model_dir=str(model_dir), device=dev)

    if fp16_text_backbone:
        if not hasattr(predictor, "_orig_embed_text_prompts"):
            predictor._orig_embed_text_prompts = predictor.embed_text_prompts

        dtype = torch.float16 if dev.type in ("cuda", "mps") else torch.float32
        predictor.text_backbone = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding-4B", dtype=dtype
        ).eval()
        if getattr(predictor, "tokenizer", None) is None:
            predictor.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-Embedding-4B", padding_side="left"
            )

        @torch.inference_mode()
        def _embed_text_prompts_fp16_safe(text_prompts):
            embeddings = predictor._orig_embed_text_prompts(text_prompts)
            if dev.type == "mps":
                torch.mps.empty_cache()
            return embeddings.float()

        predictor.embed_text_prompts = _embed_text_prompts_fp16_safe

    return predictor


def run_axis2_pipeline(
    predictor,
    neurite_oasis_dir: Union[str, Path],
    results_root: Union[str, Path],
    experiments: Sequence[str] = ("exp0", "exp1", "exp2_3", "exp4", "exp5"),
    structure_scope: str = "core_7",
    max_subjects: Optional[int] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    mc_passes: int = 10,
    spatial_dropout_p: float = 0.0,
    save_nifti_subjects: int = 2,
    save_rle_in_csv: bool = True,
    resume: bool = True,
    zip_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Run selected Axis II experiments and write a master reproducibility index (`MASTER_REPORT.md`
    and `master_manifest.json`) plus a `.zip` archive of `results_root`.
    """
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    neurite_oasis_dir = Path(neurite_oasis_dir)

    exp_set = set(experiments)
    if "all" in exp_set:
        exp_set = {"exp0", "exp1", "exp2_3", "exp4", "exp5"}
    if "exp2" in exp_set or "exp3" in exp_set:
        exp_set.add("exp2_3")

    outputs: Dict[str, Any] = {}
    t_start = time.time()

    # Deliverable 1: Frozen Prompt Table (OASIS + AMOS + Adrenal Lesion + Lung Lobe + 100-subject Synonym Manifest)
    if "exp0" in exp_set:
        print("\n=== [Exp 0] Generating Frozen Prompt Catalog & 100-Subject Synonym Manifest ===")
        oasis_manifest = neurite_oasis_dir / "manifest_neurite_oasis_100.csv"
        p0 = export_frozen_prompt_table(
            results_root / "exp0_frozen_prompts",
            oasis_manifest_csv=oasis_manifest if oasis_manifest.exists() else None,
        )
        outputs["exp0_frozen_prompts"] = {k: str(v) for k, v in p0.items()}

    # Experiment I: Base vs. Left vs. Right & Contradictory / Absent Organ
    if "exp1" in exp_set:
        print("\n=== [Exp I] Running Base / Left / Right / Contradictory Mask Comparison ===")
        res1 = run_experiment_1_side_words(
            predictor=predictor,
            neurite_oasis_dir=neurite_oasis_dir,
            results_root=results_root,
            structure_scope=structure_scope,
            max_subjects=max_subjects,
            start_index=start_index,
            end_index=end_index,
            save_nifti_subjects=save_nifti_subjects,
            save_rle_in_csv=save_rle_in_csv,
            resume=resume,
        )
        outputs["exp1_side_word_comparison"] = {
            "paths": {k: str(v) for k, v in res1["paths"].items()},
            "n_rows": res1["n_rows"],
        }

    # Experiments II & III: Synonym Shift Analysis + Layer Localization
    if "exp2_3" in exp_set:
        print("\n=== [Exp II & III] Running Synonym Shift Analysis & 14-Stage Layer Localization ===")
        res23 = run_experiment_2_and_3_synonyms_and_layers(
            predictor=predictor,
            neurite_oasis_dir=neurite_oasis_dir,
            results_root=results_root,
            structure_scope=structure_scope,
            max_subjects=max_subjects,
            start_index=start_index,
            end_index=end_index,
            save_nifti_subjects=save_nifti_subjects,
            save_rle_in_csv=save_rle_in_csv,
            resume=resume,
        )
        outputs["exp2_synonym_shift"] = {
            "paths": {k: str(v) for k, v in res23["exp2"]["paths"].items()},
        }
        outputs["exp3_layer_localization"] = {
            "paths": {k: str(v) for k, v in res23["exp3"]["paths"].items()},
        }

    # Experiment IV: MC Dropout Uncertainty Estimation
    if "exp4" in exp_set:
        print(f"\n=== [Exp IV] Running MC Dropout Uncertainty Estimation (N={mc_passes} passes) ===")
        res4 = run_experiment_4_mc_dropout_uncertainty(
            predictor=predictor,
            neurite_oasis_dir=neurite_oasis_dir,
            results_root=results_root,
            structure_scope=structure_scope,
            n_passes=mc_passes,
            spatial_dropout_p=spatial_dropout_p,
            max_subjects=max_subjects,
            start_index=start_index,
            end_index=end_index,
            save_nifti_subjects=save_nifti_subjects,
            save_rle_in_csv=save_rle_in_csv,
            resume=resume,
        )
        outputs["exp4_mc_dropout_uncertainty"] = {
            "paths": {k: str(v) for k, v in res4["paths"].items()},
        }

    # Experiment V: 4-Way Decoder Ablation (Image vs. Text Decoder)
    if "exp5" in exp_set:
        print("\n=== [Exp V] Running 4-Way Image vs. Text Decoder Ablation ===")
        res5 = run_experiment_5_decoder_ablation(
            predictor=predictor,
            neurite_oasis_dir=neurite_oasis_dir,
            results_root=results_root,
            structure_scope=structure_scope,
            max_subjects=max_subjects,
            start_index=start_index,
            end_index=end_index,
            save_nifti_subjects=save_nifti_subjects,
            save_rle_in_csv=save_rle_in_csv,
            resume=resume,
        )
        outputs["exp5_decoder_ablation"] = {
            "paths": {k: str(v) for k, v in res5["paths"].items()},
        }

    elapsed = round(time.time() - t_start, 2)
    master_json = results_root / "master_manifest.json"
    master_payload = {
        "axis": "Axis II: Sensitivity to Prompt Variation",
        "model": "VoxTell v1.1 (Qwen3-Embedding-4B)",
        "dataset": str(neurite_oasis_dir),
        "structure_scope": structure_scope,
        "max_subjects": max_subjects,
        "start_index": start_index,
        "end_index": end_index,
        "mc_passes": mc_passes,
        "elapsed_seconds": elapsed,
        "experiments_completed": outputs,
    }
    master_json.write_text(json.dumps(master_payload, indent=2), encoding="utf-8")

    master_md = results_root / "MASTER_REPORT.md"
    md_lines = [
        "# Axis II: Sensitivity to Prompt Variation — Master Reproducibility Index",
        "",
        f"- **Elapsed Time:** `{elapsed}s`",
        f"- **Structure Scope:** `{structure_scope}`",
        f"- **MC Dropout Passes (N):** `{mc_passes}`",
        "",
        "## Dedicated Experiment Output Folders",
        "| Experiment Folder | Deliverable | Saved Formats |",
        "|---|---|---|",
        "| `exp0_frozen_prompts/` | Frozen Prompt Table (OASIS + AMOS + Lesion + Lung Lobe + 100-Subject Synonym Manifest) | `.csv`, `.json`, `.md` |",
        "| `exp1_side_word_comparison/` | Experiment I: Base / Left / Right / Contradictory Mask Comparison | `metrics.csv`, `summary.csv`, `results.json`, `summary.json`, `REPORT.md`, `masks/*.npz`, `masks/*.nii.gz` |",
        "| `exp2_synonym_shift/` | Experiment II: Synonym Shift (Qwen3 & Prompt Decoder Embeddings vs. Mask Change) | `metrics.csv`, `summary.csv`, `embedding_vs_prediction_correlations.csv`, `results.json`, `REPORT.md`, `masks/*.npz`, `masks/*.nii.gz` |",
        "| `exp3_layer_localization/` | Experiment III: 14-Stage Layer-wise Localization of Side & Synonym Information | `metrics.csv`, `summary.csv`, `results.json`, `summary.json`, `REPORT.md` |",
        "| `exp4_mc_dropout_uncertainty/` | Experiment IV: MC Dropout Uncertainty (Epistemic Variance/MI & Aleatoric Entropy) | `metrics.csv`, `summary.csv`, `synonym_uncertainty_stability.csv`, `results.json`, `REPORT.md`, `masks/*.npz`, `masks/*.nii.gz` |",
        "| `exp5_decoder_ablation/` | Experiment V: 4-Way Image vs. Text Decoder On/Off Ablation | `metrics.csv`, `summary.csv`, `results.json`, `summary.json`, `REPORT.md`, `masks/*.npz`, `masks/*.nii.gz` |",
        "",
        "## Reconstructing Any 3D Mask Without Re-running VoxTell",
        "```python",
        "import pandas as pd",
        "from axis2_voxtell_engine import decode_mask_rle, str_to_shape, save_mask_nifti",
        "",
        "df = pd.read_csv('results_axis2/exp1_side_word_comparison/metrics.csv')",
        "row = df.iloc[0]",
        "mask_3d = decode_mask_rle(row['pred_mask_rle_b64'], str_to_shape(row['volume_shape_ras']))",
        "save_mask_nifti(mask_3d, 'reconstructed_mask.nii.gz')",
        "```",
    ]
    master_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    if zip_outputs:
        zip_base = results_root.parent / f"{results_root.name}_archive"
        zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(results_root))
        outputs["zip_archive"] = zip_path
        print(f"\n[Archive] Created full reproducible zip archive: {zip_path}")

    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Axis II Prompt Sensitivity & Uncertainty Experiments on Neurite-OASIS")
    p.add_argument("--oasis-dir", type=Path, required=True, help="Path to downloaded neurite_oasis_100 directory")
    p.add_argument("--results-dir", type=Path, default=Path("results_axis2"), help="Output directory for all experiment folders")
    p.add_argument("--model-dir", type=Path, default=None, help="Local VoxTell v1.1 checkpoint folder")
    p.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        choices=["all", "exp0", "exp1", "exp2", "exp3", "exp2_3", "exp4", "exp5"],
        help="Which experiments to run",
    )
    p.add_argument("--structure-scope", type=str, default="core_7", help="core_7 | extended_11 | all_16")
    p.add_argument("--max-subjects", type=int, default=None, help="Max subjects to process (default: all 100)")
    p.add_argument("--start-index", type=int, default=0, help="Start subject index (0-based)")
    p.add_argument("--end-index", type=int, default=None, help="End subject index (exclusive)")
    p.add_argument("--mc-passes", type=int, default=10, help="Number of stochastic MC dropout passes per prompt")
    p.add_argument("--spatial-dropout-p", type=float, default=0.0, help="Optional extra spatial dropout on bottleneck features")
    p.add_argument("--save-nifti-subjects", type=int, default=2, help="Number of subjects to save full .nii.gz volumes for")
    p.add_argument("--no-rle", action="store_true", help="Omit RLE mask strings from CSV/JSON")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing per-subject checkpoints and recompute")
    p.add_argument("--no-zip", action="store_true", help="Skip creating final zip archive")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = None
    if set(args.experiments) != {"exp0"}:
        if args.model_dir is None:
            raise ValueError("--model-dir is required when running model experiments (exp1..exp5).")
        predictor = load_voxtell_predictor_safe(args.model_dir)

    run_axis2_pipeline(
        predictor=predictor,
        neurite_oasis_dir=args.oasis_dir,
        results_root=args.results_dir,
        experiments=args.experiments,
        structure_scope=args.structure_scope,
        max_subjects=args.max_subjects,
        start_index=args.start_index,
        end_index=args.end_index,
        mc_passes=args.mc_passes,
        spatial_dropout_p=args.spatial_dropout_p,
        save_nifti_subjects=args.save_nifti_subjects,
        save_rle_in_csv=not args.no_rle,
        resume=not args.no_resume,
        zip_outputs=not args.no_zip,
    )
