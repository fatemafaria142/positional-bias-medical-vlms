"""
Per-prompt (P1-P4) breakout of shift-rate and wrong-side-rate, per model,
per dataset -- the data behind Results/Findings/PROMPT_WISE_SHIFT.md.

Unlike scripts/aggregate_findings.py (which pools P1-P4 into one number
per model), this keeps each prompt separate so direction-specific biases
(a model defaulting to "left" regardless of prompt, say) are visible
instead of averaging out to a misleading ~50%.

Usage:
    python scripts/aggregate_prompt_shift.py
Writes Results/Findings/PROMPT_SHIFT_DATA.csv (one row per dataset/model/prompt).
"""

from pathlib import Path

import pandas as pd

from aggregate_findings import (
    DATASETS,
    FINDINGS_DIR,
    image_size_lookup,
    load_all,
)

PROMPT_DIRECTION = {"P1": "left", "P2": "right", "P3": "upper", "P4": "lower"}


def per_prompt_rows(df: pd.DataFrame, model: str, ds: str) -> list[dict]:
    rows = []
    for pid, direction in PROMPT_DIRECTION.items():
        sub = df[df["prompt_id"] == pid]
        shift_rate = 1 - sub["no_shift"].mean() if sub["no_shift"].notna().any() else float("nan")
        wrong_rate = 1 - sub["side_correct"].mean() if sub["side_correct"].notna().any() else float("nan")
        rows.append({
            "dataset": ds,
            "model": model,
            "prompt": pid,
            "direction": direction,
            "shift_rate": shift_rate,
            "wrong_side_rate": wrong_rate,
            "n": len(sub),
        })
    return rows


def main():
    all_rows = []
    for ds in DATASETS:
        sizes = image_size_lookup(ds)
        for model, df in load_all(ds, sizes).items():
            all_rows += per_prompt_rows(df, model, ds)

    out = pd.DataFrame(all_rows)
    FINDINGS_DIR.mkdir(exist_ok=True)
    out_path = FINDINGS_DIR / "PROMPT_SHIFT_DATA.csv"
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
