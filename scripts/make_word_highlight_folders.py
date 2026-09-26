"""
Builds the folder tree local_plots_groundedness/word_highlight/{AMOS,OASIS}/<organ>/ with three figures per organ:
    all_prompts.png          every prompt version given to the model and its Dice
    word_attribution.png     Exact Shapley / SHAP / LIME value of each word of the prompt
    share_of_importance.png  organ name vs. position word as a percentage, per method

Usage:  python scripts/make_word_highlight_folders.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_prompt_attribution as attribution
import plot_prompt_breakdown as breakdown
import plot_share_of_importance as share

ROOT = Path(__file__).resolve().parent.parent / "local_plots_groundedness" / "word_highlight"
TREE = {
    "AMOS": ("amos", ["left kidney", "right kidney", "left adrenal gland", "right adrenal gland"]),
    # OASIS: only the structures the model actually segments (the other three have Dice ~ 0 even with the full prompt)
    "OASIS": ("oasis", ["left cerebral white matter", "right cerebral white matter",
                        "left cerebellum cortex", "right cerebellum cortex"]),
}

# replacement organ for the "all prompts" figure: the one most subjects received, unless set here
# (the left adrenal gland ties between the aorta and the pancreas; the pancreas was chosen)
LANDMARK_OVERRIDE = {"left adrenal gland": "pancreas"}

for folder, (key, organs) in TREE.items():
    for organ in organs:
        d = ROOT / folder / organ.replace(" ", "_")
        breakdown.main(key, organ, LANDMARK_OVERRIDE.get(organ, "auto"), d / "all_prompts.png")
        attribution.main(key, organ, d / "word_attribution.png")
        share.main(key, organ, d / "share_of_importance.png")
