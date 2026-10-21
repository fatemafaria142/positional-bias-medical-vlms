"""
Build per-experiment Kaggle notebooks from the main notebook
`voxtell-oasis-mechanistic-experiments-kaggle.ipynb`.

Each generated notebook keeps the shared setup (sections 0-7: install, GPU check, clone repo,
download dataset, load VoxTell, frozen prompts, configuration) unchanged, then runs ONLY its own
experiment section and zips only that experiment's output folder(s).

Re-run this script whenever the main notebook changes:
    python scripts/build_split_kaggle_notebooks.py
"""

import copy
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_NB = REPO_ROOT / "voxtell-oasis-mechanistic-experiments-kaggle.ipynb"

# Last shared cell is the experiment configuration (section 7)
SHARED_UNTIL_ID = "axis2-config-code"

SPLITS = [
    {
        "file": "voxtell-oasis-exp1-side-words-kaggle.ipynb",
        "title": "Experiment I — Does the Side Word Change the Output?",
        "cell_ids": ["axis2-exp1-md", "axis2-exp1-code"],
        "folders": ["exp0_frozen_prompts", "exp1_side_word_comparison"],
        "zip_name": "results_axis2_exp1",
    },
    {
        "file": "voxtell-oasis-exp2-3-synonyms-layers-kaggle.ipynb",
        "title": "Experiments II & III — Synonym Shift + 14-Stage Layer Localization",
        "cell_ids": ["axis2-exp23-md", "axis2-exp23-code"],
        "folders": ["exp0_frozen_prompts", "exp2_synonym_shift", "exp3_layer_localization"],
        "zip_name": "results_axis2_exp2_3",
    },
    {
        "file": "voxtell-oasis-exp4-mc-dropout-kaggle.ipynb",
        "title": "Experiment IV — MC Dropout Uncertainty Estimation",
        "cell_ids": ["axis2-exp4-md", "axis2-exp4-code"],
        "folders": ["exp0_frozen_prompts", "exp4_mc_dropout_uncertainty"],
        "zip_name": "results_axis2_exp4",
    },
    {
        "file": "voxtell-oasis-exp5-decoder-ablation-kaggle.ipynb",
        "title": "Experiment V — 4-Way Image vs. Text Decoder Ablation",
        "cell_ids": ["axis2-exp5-md", "axis2-exp5-code"],
        "folders": ["exp0_frozen_prompts", "exp5_decoder_ablation"],
        "zip_name": "results_axis2_exp5",
    },
]


def _src(text):
    return text.splitlines(keepends=True)


def _zip_cells(folders, zip_name):
    md = {
        "cell_type": "markdown",
        "id": f"{zip_name}-zip-md",
        "metadata": {},
        "source": _src(
            "## Package this experiment's output folders into a zip archive\n\n"
            f"Creates `/kaggle/working/{zip_name}.zip` containing: "
            + ", ".join(f"`{f}/`" for f in folders)
            + "."
        ),
    }
    code = {
        "cell_type": "code",
        "execution_count": None,
        "id": f"{zip_name}-zip-code",
        "metadata": {},
        "outputs": [],
        "source": _src(
            "import shutil, tempfile\n"
            "\n"
            f"_folders = {folders!r}\n"
            "_staging = Path(tempfile.mkdtemp())\n"
            "for _f in _folders:\n"
            "    if (RESULTS_ROOT / _f).exists():\n"
            "        shutil.copytree(RESULTS_ROOT / _f, _staging / _f)\n"
            f"zip_path = shutil.make_archive(str(WORKING_DIR / '{zip_name}'), 'zip', root_dir=str(_staging))\n"
            "shutil.rmtree(_staging, ignore_errors=True)\n"
            "print('Saved archive:', zip_path)\n"
            "for _f in _folders:\n"
            "    _d = RESULTS_ROOT / _f\n"
            "    if _d.exists():\n"
            "        print(f'  {_f}/ ->', [p.name for p in sorted(_d.iterdir())])\n"
        ),
    }
    return [md, code]


def build():
    nb = json.loads(MAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    by_id = {c.get("id"): c for c in cells}

    shared_end = next(i for i, c in enumerate(cells) if c.get("id") == SHARED_UNTIL_ID)
    shared = cells[: shared_end + 1]

    for spec in SPLITS:
        out = copy.deepcopy(nb)
        header = copy.deepcopy(shared[0])
        header["source"] = _src(
            f"# VoxTell on Neurite-OASIS — {spec['title']}\n\n"
            "Generated from `voxtell-oasis-mechanistic-experiments-kaggle.ipynb` by "
            "`scripts/build_split_kaggle_notebooks.py`. Sections 0-7 (setup) are identical across all split "
            "notebooks; only the experiment section below differs.\n"
        )
        out["cells"] = (
            [header]
            + copy.deepcopy(shared[1:])
            + [copy.deepcopy(by_id[cid]) for cid in spec["cell_ids"]]
            + _zip_cells(spec["folders"], spec["zip_name"])
        )
        for c in out["cells"]:
            if c["cell_type"] == "code":
                c["outputs"] = []
                c["execution_count"] = None
        dest = REPO_ROOT / spec["file"]
        dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[ok] {dest.name}: {len(out['cells'])} cells")


if __name__ == "__main__":
    build()
