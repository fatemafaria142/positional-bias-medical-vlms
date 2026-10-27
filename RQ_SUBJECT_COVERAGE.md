# RQ1–RQ4, RQ5–RQ7 Subject Coverage & Next-Steps Plan

Snapshot taken 2026-10-24 from `results/neurite-oasis/RQ*/*.csv`, re-derived directly from the
actual CSV `subject_id` columns (not assumed from earlier notes). Each RQ's notebook cell
starts its own subject loop from `OASIS_OAS1_0001_MR1` rather than continuing from a shared
pool, so "subjects used" below is specific to that RQ, not cumulative.

## Status table

| RQ | Status | Subjects used | ID range covered | Next subject to start from |
|---|---|---|---|---|
| RQ1 | Done | 100 | OASIS_OAS1_0001_MR1 – 0107_MR1 | 0108_MR1 (or next available) |
| RQ2 | Done | 100 | OASIS_OAS1_0001_MR1 – 0107_MR1 | 0108_MR1 |
| RQ3 | Done | 50 | OASIS_OAS1_0001_MR1 – 0054_MR1 | 0055_MR1 — ran via `RQ3_50subjects_kaggle.ipynb`, up from the earlier 22-subject capped run |
| RQ4 | Done | 50 | OASIS_OAS1_0001_MR1 – 0054_MR1 | 0055_MR1 — ran via `RQ4_50subjects_kaggle.ipynb`; the pre-expansion 22-subject CSVs are kept alongside as `rq4_activation_patching_old.csv` / `rq4_ground_truth_masks_old.csv` |
| RQ5 | Done, capped | 5 | OASIS_OAS1_0001_MR1 – 0005_MR1 | 0006_MR1 — `RQ5_only_50subjects_kaggle.ipynb` and the 3-part split (`RQ5_50subjects_part1/2/3_kaggle.ipynb`) exist but haven't been run yet; still the most expensive RQ (256 full predictions per organ per subject) |
| RQ6 | Done | 50 | OASIS_OAS1_0001_MR1 – 0054_MR1 | 0055_MR1 (or next available) — ran via `RQ6_only_50subjects_kaggle.ipynb`; the pre-expansion 5-subject CSVs are kept alongside as `rq6_word_shapley_summary_old.csv` / `rq6_word_shapley_coalitions_old.csv` |
| RQ7 | Run complete, pending merge | 50 (in `rq7_prompt_injection_bias_new.csv`) | OASIS_OAS1_0001_MR1 – 0054_MR1 | 0055_MR1 — the 50-subject run landed in `results/neurite-oasis/RQ7/rq7_prompt_injection_bias_new.csv`; the tracked `rq7_prompt_injection_bias.csv` is still the old 5-subject version and needs the `_new` file promoted/renamed over it |

Note: the "ID range covered" counts aren't perfectly dense, e.g. RQ1/RQ2 have 100 unique
subjects but the highest ID is `0107_MR1` (a few numbers in between are skipped in the
Neurite-OASIS staging). Use the exact unique ID set (re-derivable from the CSVs) rather than
the range bounds if you need to avoid re-running an already-covered subject exactly.

One file has no `subject_id` column at all, by design, since it is an aggregate and not per-subject rows:
`RQ1/rq1_organ_x_stage_mean_cosine.csv` (organ-level rollup).

## Priority order for moving forward

1. **Merge the RQ7 50-subject run** — promote `rq7_prompt_injection_bias_new.csv` over the
   tracked `rq7_prompt_injection_bias.csv` (and drop the `RQ4`/`RQ6` `_old` backups once
   each 50-subject result is confirmed good) so the repo state matches what's actually been run.
2. **RQ5** — still capped at 5 subjects; the standalone 50-subject notebook(s) exist but
   haven't been run. This is the most expensive RQ (256 full predictions per organ per subject),
   so expect this to take meaningfully longer than RQ3/RQ4/RQ6/RQ7 did.
3. **RQ1 / RQ2** — already the deepest at 100 subjects; lowest priority unless aiming for
   the full Neurite-OASIS cohort.

## How to re-derive the exact covered subject IDs later

```python
import csv, sys
csv.field_size_limit(2**30)  # some RQ CSVs embed RLE-encoded masks in a field
with open("results/neurite-oasis/RQ<N>/<file>.csv", encoding="utf-8") as fh:
    ids = sorted({row["subject_id"] for row in csv.DictReader(fh)})
print(len(ids), ids)
```
