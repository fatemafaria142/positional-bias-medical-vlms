"""
Run any Axis II experiment on all available GPUs (e.g. Kaggle "GPU T4 x2").

How it works
  1. The notebook process embeds every prompt ONCE with Qwen3-Embedding-4B and saves them as a
     VoxTell embedding bank (`prompt_embedding_bank.npz`). Workers load VoxTell with that bank, so
     the 4B text encoder (~8 GB RAM) is never loaded in the workers.
  2. Subjects still missing a checkpoint are split round-robin across GPUs. One worker process per
     GPU (`CUDA_VISIBLE_DEVICES=<i>`) runs the normal experiment function on its subjects and writes
     the usual per-subject checkpoints / masks into the shared results folder.
  3. When all workers finish, the experiment function is called once more in the notebook process
     with `resume=True` and no model: it reads every checkpoint and writes the combined
     metrics.csv / summary.csv / results.json / summary.json / REPORT.md (and plots).

With a single GPU (or on CPU / Apple MPS) the experiment simply runs in-process.

Notebook usage:
    from axis2_multigpu import run_experiment_parallel
    res = run_experiment_parallel('exp4', predictor, VOXTELL_MODEL_DIR, NEURITE_OASIS_DIR, RESULTS_ROOT,
                                  structure_scope='hippocampus,thalamus,caudate,putamen', n_passes=5)

Worker usage (internal):
    python axis2_multigpu.py --job <job.json>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

EXPERIMENTS = {
    "exp1": ("run_experiment_1_side_words", ["exp1_side_word_comparison"]),
    "exp2_3": ("run_experiment_2_and_3_synonyms_and_layers", ["exp2_synonym_shift", "exp3_layer_localization"]),
    "exp4": ("run_experiment_4_mc_dropout_uncertainty", ["exp4_mc_dropout_uncertainty"]),
    "exp5": ("run_experiment_5_decoder_ablation", ["exp5_decoder_ablation"]),
}


def _experiment_fn(name: str):
    import axis2_experiments
    return getattr(axis2_experiments, EXPERIMENTS[name][0])


def build_prompt_embedding_bank(predictor, structure_scopes: Sequence[str], bank_path: Union[str, Path]) -> Path:
    """Embed every catalog prompt for the given structure scopes (plus absent-organ negatives) and
    save them in VoxTell's embedding-bank format (`labels`, `embeddings` float16). Reuses an existing
    bank file if it already contains all needed prompts."""
    import numpy as np
    from axis2_experiments import resolve_structures
    from axis2_prompts import build_oasis_prompt_catalog

    structures: List[str] = []
    for scope in structure_scopes:
        for s in resolve_structures(scope):
            if s not in structures:
                structures.append(s)
    prompts = sorted({pr.prompt_text.lower() for pr in build_oasis_prompt_catalog(structures=structures, include_absent_negatives=True)})

    bank_path = Path(bank_path)
    existing: Dict[str, np.ndarray] = {}
    if bank_path.exists():
        with np.load(bank_path, allow_pickle=False) as d:
            existing = {str(lbl): d["embeddings"][i] for i, lbl in enumerate(d["labels"])}
    missing = [p for p in prompts if p not in existing]
    if missing:
        print(f"[bank] embedding {len(missing)} prompts with the text encoder ...")
        emb = predictor.embed_text_prompts(missing)[0].detach().float().cpu().numpy()
        for p, v in zip(missing, emb):
            existing[p] = v.astype(np.float16)
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        labels = sorted(existing)
        np.savez(bank_path, labels=np.array(labels), embeddings=np.stack([existing[l] for l in labels]))
    print(f"[bank] {len(existing)} prompt embeddings in {bank_path}")
    return bank_path


def _free_notebook_gpu(predictor) -> None:
    """Move the notebook's copy of the model off GPU 0 and drop the text encoder, so the workers get
    the full GPUs and enough RAM."""
    import gc
    import torch
    from axis2_experiments import release_text_backbone

    if predictor is not None:
        release_text_backbone(predictor)
        if getattr(predictor, "network", None) is not None:
            predictor.network.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _has_checkpoints(results_root: Path, name: str, sid: str) -> bool:
    return all((results_root / folder / "checkpoints" / f"{sid}.json").exists() for folder in EXPERIMENTS[name][1])


def _stream(proc: subprocess.Popen, prefix: str, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(f"{prefix} {line}", end="", flush=True)


def run_experiment_parallel(
    name: str,
    predictor,
    model_dir: Union[str, Path],
    neurite_oasis_dir: Union[str, Path],
    results_root: Union[str, Path],
    n_gpus: Optional[int] = None,
    structure_scope: str = "core_7",
    max_subjects: Optional[int] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    save_nifti_subjects: int = 2,
    save_rle_in_csv: bool = True,
    resume: bool = True,
    bank_scopes: Optional[Sequence[str]] = None,
    **exp_kwargs: Any,
) -> Dict[str, Any]:
    """
    Run experiment `name` ('exp1', 'exp2_3', 'exp4', 'exp5') across `n_gpus` GPUs (default: all
    visible CUDA GPUs) and return the same dict as the underlying `run_experiment_*` function.
    Extra keyword arguments (e.g. `n_passes`, `spatial_dropout_p`, `prompt_group_size` for exp4) are
    forwarded to the experiment function.
    """
    import torch
    from axis2_experiments import discover_oasis_subjects

    if name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {name!r}; choose from {list(EXPERIMENTS)}")
    fn = _experiment_fn(name)
    results_root = Path(results_root).resolve()
    neurite_oasis_dir = Path(neurite_oasis_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    if n_gpus is None:
        n_gpus = torch.cuda.device_count()

    subjects = discover_oasis_subjects(neurite_oasis_dir, max_subjects=max_subjects,
                                       start_index=start_index, end_index=end_index)
    ids = [s["subject_id"] for s in subjects]
    common = dict(neurite_oasis_dir=str(neurite_oasis_dir), results_root=str(results_root),
                  structure_scope=structure_scope, save_rle_in_csv=save_rle_in_csv, **exp_kwargs)

    if n_gpus <= 1:
        print(f"[{name}] single device -> running in-process on {len(ids)} subjects")
        return fn(predictor=predictor, subject_ids=ids, resume=resume,
                  save_nifti_subjects=save_nifti_subjects, **common)

    pending = [sid for sid in ids if not (resume and _has_checkpoints(results_root, name, sid))]
    print(f"[{name}] {len(ids)} subjects, {len(ids) - len(pending)} already done, "
          f"{len(pending)} to run on {n_gpus} GPUs")

    if pending:
        bank = build_prompt_embedding_bank(
            predictor, list(bank_scopes or [structure_scope]), results_root / "prompt_embedding_bank.npz"
        )
        _free_notebook_gpu(predictor)

        jobs_dir = results_root / "_parallel_jobs"
        jobs_dir.mkdir(exist_ok=True)
        procs, threads = [], []
        t0 = time.time()
        for g in range(n_gpus):
            chunk = pending[g::n_gpus]
            if not chunk:
                continue
            job = {
                "experiment": name,
                "model_dir": str(model_dir),
                "embedding_bank": str(bank),
                "kwargs": dict(common, subject_ids=chunk, resume=resume,
                               save_nifti_subjects=save_nifti_subjects if g == 0 else 0),
            }
            job_path = jobs_dir / f"{name}_gpu{g}.json"
            job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            proc = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--job", str(job_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(_SCRIPTS_DIR),
            )
            th = threading.Thread(target=_stream, args=(proc, f"[GPU{g}]", jobs_dir / f"{name}_gpu{g}.log"), daemon=True)
            th.start()
            procs.append((g, proc))
            threads.append(th)
            print(f"[{name}] GPU{g}: {len(chunk)} subjects (log: {jobs_dir / f'{name}_gpu{g}.log'})")

        failed = []
        for g, proc in procs:
            if proc.wait() != 0:
                failed.append(g)
        for th in threads:
            th.join(timeout=10)
        print(f"[{name}] workers finished in {(time.time() - t0) / 60:.1f} min")
        if failed:
            raise RuntimeError(
                f"[{name}] worker(s) on GPU {failed} failed -- see {jobs_dir}/*.log. Completed subjects are "
                f"checkpointed; re-running this cell resumes from them."
            )

    print(f"[{name}] building combined outputs from checkpoints ...")
    return fn(predictor=None, subject_ids=ids, resume=True, save_nifti_subjects=0, **common)


def _worker(job_path: Path) -> None:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    import torch

    factory = os.environ.get("AXIS2_PREDICTOR_FACTORY")  # "module:function", only used for offline testing
    if factory:
        mod_name, fn_name = factory.split(":")
        import importlib
        predictor = getattr(importlib.import_module(mod_name), fn_name)()
        _experiment_fn(job["experiment"])(predictor=predictor, **job["kwargs"])
        return

    from voxtell.inference.predictor import VoxTellPredictor

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    predictor = VoxTellPredictor(
        model_dir=job["model_dir"],
        device=device,
        embedding_bank=job["embedding_bank"],
        use_precomputed_embeddings=False,
    )
    print(f"worker ready on {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}: "
          f"{len(job['kwargs']['subject_ids'])} subjects", flush=True)
    _experiment_fn(job["experiment"])(predictor=predictor, **job["kwargs"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    _worker(ap.parse_args().job)
