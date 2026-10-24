# RQ1–RQ11 Subject Coverage & Next-Steps Plan

Snapshot taken 2026-10-23 from `results/neurite-oasis/RQ*/*.csv`. Each RQ's notebook cell
starts its own subject loop from `OASIS_OAS1_0001_MR1` rather than continuing from a shared
pool, so "subjects used" below is specific to that RQ, not cumulative.

## Status table

| RQ | Status | Subjects used | ID range covered | Next subject to start from |
|---|---|---|---|---|
| RQ1 | Done | 100 | OASIS_OAS1_0001_MR1 – 0107_MR1 | 0108_MR1 (or next available) |
| RQ2 | Done | 100 | OASIS_OAS1_0001_MR1 – 0107_MR1 | 0108_MR1 |
| RQ3 | Done, capped | 22 | OASIS_OAS1_0001_MR1 – 0023_MR1 | 0024_MR1 — **but see note below:** `RQ3_50subjects_kaggle.ipynb` exists (standalone, not yet run) to push this to 50 |
| RQ4 | Done, capped | 22 | OASIS_OAS1_0001_MR1 – 0023_MR1 | 0024_MR1 — **but see note below:** `RQ4_50subjects_kaggle.ipynb` exists (standalone, not yet run) to push this to 50 |
| RQ5 | Done | 52 | OASIS_OAS1_0001_MR1 – 0056_MR1 | 0057_MR1 |
| RQ6 | Done, capped | 21 | OASIS_OAS1_0001_MR1 – 0022_MR1 | 0023_MR1 — **but see note below:** `RQ6_50subjects_kaggle.ipynb` exists (standalone, not yet run) to push this to 50 |
| RQ8 | Notebook ready, never run | 0 | — | `RQ8_50subjects_kaggle.ipynb` exists (standalone, added in commit `0cb8bf3`) but has not been executed — `results/neurite-oasis/RQ8` still doesn't exist |
| RQ9 | Done, capped | 5 | OASIS_OAS1_0001_MR1 – 0005_MR1 | 0006_MR1 — `RQ9_only_50subjects_kaggle.ipynb` exists (standalone, not yet run) to push this to 50 |
| RQ10 | Done | 50 | OASIS_OAS1_0001_MR1 – 0054_MR1 | 0055_MR1 (or next available) — ran via `RQ10_only_50subjects_kaggle.ipynb`; the pre-expansion 5-subject CSVs are kept alongside as `rq10_word_shapley_summary_old.csv` / `rq10_word_shapley_coalitions_old.csv` |
| RQ11 | Run complete, pending merge | 50 (in `rq11_prompt_injection_bias_new.csv`) | OASIS_OAS1_0001_MR1 – 0054_MR1 | 0055_MR1 — the 50-subject run landed in `results/neurite-oasis/RQ11/rq11_prompt_injection_bias_new.csv` (untracked); the tracked `rq11_prompt_injection_bias.csv` is still the old 5-subject version and needs the `_new` file promoted/renamed over it |

Note: the "ID range covered" counts aren't perfectly dense — e.g. RQ1/RQ2 have 100 unique
subjects but the highest ID is `0107_MR1` (a few numbers in between are skipped in the
Neurite-OASIS staging). Use the exact unique ID set (re-derivable from the CSVs) rather than
the range bounds if you need to avoid re-running an already-covered subject exactly.

## Priority order for moving forward

1. **Merge the RQ11 50-subject run** — promote `rq11_prompt_injection_bias_new.csv` over the
   tracked `rq11_prompt_injection_bias.csv` (and drop the `RQ10` `_old` backups once the RQ10
   50-subject result is confirmed good) so the repo state matches what's actually been run.
2. **RQ8 (gradient/Grad-CAM saliency)** — notebook (`RQ8_50subjects_kaggle.ipynb`) exists but
   has never been executed; run it on Kaggle to produce the first `results/neurite-oasis/RQ8`
   output.
3. **RQ3 / RQ4 / RQ6** — standalone 50-subject notebooks exist for all three
   (`RQ3_50subjects_kaggle.ipynb`, `RQ4_50subjects_kaggle.ipynb`, `RQ6_50subjects_kaggle.ipynb`)
   but none have been run yet; running them extends coverage from 21–22 subjects to 50.
4. **RQ9** — `RQ9_only_50subjects_kaggle.ipynb` exists but hasn't been run; extends from 5 to 50.
5. **RQ5** — extend past `0056_MR1` to approach RQ1/RQ2's 100-subject coverage.
6. **RQ1 / RQ2** — already the deepest at 100 subjects; lowest priority unless aiming for
   the full Neurite-OASIS cohort.

## How to re-derive the exact covered subject IDs later

```python
import csv, sys
csv.field_size_limit(2**30)  # some RQ CSVs embed RLE-encoded masks in a field
with open("results/neurite-oasis/RQ<N>/<file>.csv", encoding="utf-8") as fh:
    ids = sorted({row["subject_id"] for row in csv.DictReader(fh)})
print(len(ids), ids)
```
