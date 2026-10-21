"""
Merge Axis II results produced in several Kaggle sessions (e.g. subjects 0-34, 34-67, 67-100)
into one combined results folder, WITHOUT a GPU and WITHOUT re-running VoxTell.

Every experiment writes one checkpoint per subject (`<exp>/checkpoints/<subject_id>.json`) plus
per-subject mask/map files (`<exp>/masks/`). This script copies those from every input folder
into `--output`, then re-runs each experiment function with `resume=True` restricted to the
subjects that have checkpoints, which rebuilds metrics.csv / summary.csv / results.json /
summary.json / REPORT.md (and the Exp III plot) from the checkpoints alone.

Usage (inputs can be unzipped `results_axis2_exp4.zip` folders or whole results roots):
    python scripts/merge_axis2_results.py \
        --inputs part1/ part2/ part3/ \
        --output results_axis2_merged \
        --oasis-dir data/neurite_oasis_100
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from axis2_experiments import (  # noqa: E402
    run_experiment_1_side_words,
    run_experiment_2_and_3_synonyms_and_layers,
    run_experiment_4_mc_dropout_uncertainty,
    run_experiment_5_decoder_ablation,
)

EXPERIMENT_FOLDERS = [
    "exp0_frozen_prompts",
    "exp1_side_word_comparison",
    "exp2_synonym_shift",
    "exp3_layer_localization",
    "exp4_mc_dropout_uncertainty",
    "exp5_decoder_ablation",
]


def _find_exp_dirs(root: Path, name: str):
    """An input may be a results root (`root/<name>`) or contain it one level deeper."""
    return [p for p in [root / name, *root.glob(f"*/{name}")] if p.is_dir()]


def _copy_tree_files(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            n += 1
    return n


def merge(inputs, output: Path, oasis_dir: Path, structure_scope: str, mc_passes: int):
    output.mkdir(parents=True, exist_ok=True)

    for name in EXPERIMENT_FOLDERS:
        for inp in inputs:
            for exp_dir in _find_exp_dirs(Path(inp), name):
                if name == "exp0_frozen_prompts":
                    _copy_tree_files(exp_dir, output / name)
                    continue
                n_ckpt = _copy_tree_files(exp_dir / "checkpoints", output / name / "checkpoints") if (exp_dir / "checkpoints").is_dir() else 0
                n_mask = _copy_tree_files(exp_dir / "masks", output / name / "masks") if (exp_dir / "masks").is_dir() else 0
                for extra in exp_dir.glob("*.png"):
                    shutil.copy2(extra, output / name / extra.name)
                print(f"[copy] {exp_dir} -> {name}: {n_ckpt} checkpoints, {n_mask} mask files")

    def _ids(name):
        d = output / name / "checkpoints"
        return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []

    common = dict(predictor=None, neurite_oasis_dir=oasis_dir, results_root=output,
                  structure_scope=structure_scope, resume=True)

    ids = _ids("exp1_side_word_comparison")
    if ids:
        print(f"\n[merge] Exp I: {len(ids)} subjects")
        run_experiment_1_side_words(subject_ids=ids, **common)

    ids2, ids3 = set(_ids("exp2_synonym_shift")), set(_ids("exp3_layer_localization"))
    if ids2 and ids3:
        both = sorted(ids2 & ids3)
        if len(both) != len(ids2 | ids3):
            print(f"[warn] Exp II/III: {len(ids2 ^ ids3)} subjects have only one of the two checkpoints; skipping them")
        print(f"\n[merge] Exp II & III: {len(both)} subjects")
        run_experiment_2_and_3_synonyms_and_layers(subject_ids=both, **common)

    ids = _ids("exp4_mc_dropout_uncertainty")
    if ids:
        print(f"\n[merge] Exp IV: {len(ids)} subjects")
        run_experiment_4_mc_dropout_uncertainty(subject_ids=ids, n_passes=mc_passes, **common)

    ids = _ids("exp5_decoder_ablation")
    if ids:
        print(f"\n[merge] Exp V: {len(ids)} subjects")
        run_experiment_5_decoder_ablation(subject_ids=ids, **common)

    print(f"\n[done] merged results in {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--oasis-dir", type=Path, required=True, help="Neurite-OASIS 100-subject folder (with volumes/)")
    ap.add_argument("--structure-scope", default="core_7", help="Must match the scope used in the Kaggle runs")
    ap.add_argument("--mc-passes", type=int, default=10, help="Must match MC_DROPOUT_PASSES used in the Kaggle runs")
    args = ap.parse_args()
    merge(args.inputs, args.output, args.oasis_dir, args.structure_scope, args.mc_passes)


if __name__ == "__main__":
    main()
