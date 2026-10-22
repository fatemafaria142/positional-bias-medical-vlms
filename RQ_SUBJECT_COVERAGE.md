# RQ1–RQ11 Subject Coverage & Next-Steps Plan

Snapshot taken 2026-10-23 from `results/neurite-oasis/RQ*/*.csv`. Each RQ's notebook cell
starts its own subject loop from `OASIS_OAS1_0001_MR1` rather than continuing from a shared
pool, so "subjects used" below is specific to that RQ, not cumulative.

## Status table

| RQ | Status | Subjects used | ID range covered | Next subject to start from |
|---|---|---|---|---|
| RQ1 | Done | 100 | OASIS_OAS1_0001_MR1 – 0107_MR1 | 0108_MR1 (or next available) |
| RQ2 | Done | 100 | OASIS_OAS1_0001_MR1 – 0107_MR1 | 0108_MR1 |
| RQ3 | Done | 22 | OASIS_OAS1_0001_MR1 – 0023_MR1 | 0024_MR1 |
| RQ4 | Done | 22 | OASIS_OAS1_0001_MR1 – 0023_MR1 | 0024_MR1 |
| RQ5 | Done | 52 | OASIS_OAS1_0001_MR1 – 0056_MR1 | 0057_MR1 |
| RQ6 | Done | 21 | OASIS_OAS1_0001_MR1 – 0022_MR1 | 0023_MR1 |
| **RQ7** | **Missing — never run** | 0 | — | 0001_MR1 (start from scratch) |
| **RQ8** | **Missing — never run** | 0 | — | 0001_MR1 (start from scratch) |
| RQ9 | Done, capped | 5 | OASIS_OAS1_0001_MR1 – 0005_MR1 | 0006_MR1 |
| RQ10 | Done, capped | 5 | OASIS_OAS1_0001_MR1 – 0005_MR1 | 0006_MR1 |
| RQ11 | Done, capped | 5 | OASIS_OAS1_0001_MR1 – 0005_MR1 | 0006_MR1 |

Note: the "ID range covered" counts aren't perfectly dense — e.g. RQ1/RQ2 have 100 unique
subjects but the highest ID is `0107_MR1` (a few numbers in between are skipped in the
Neurite-OASIS staging). Use the exact unique ID set (re-derivable from the CSVs) rather than
the range bounds if you need to avoid re-running an already-covered subject exactly.

## Priority order for moving forward

1. **RQ7 (per-head attention ablation)** and **RQ8 (gradient/Grad-CAM saliency)** — write
   these cells from scratch; they're specified in `RQ1_to_RQ11_Methodology.md` but have no
   notebook cells or `results/neurite-oasis/RQ7` / `RQ8` folders yet.
2. **RQ9, RQ10, RQ11** — cheapest to extend (5-subject cap was made self-contained per cell
   per the "Make RQ9/RQ10/RQ11's 5-subject cap self-contained per cell" commit). Bump the cap
   and point the loop at subjects starting from `0006_MR1`.
3. **RQ6** — extend past `0022_MR1`.
4. **RQ3 / RQ4** — extend past `0023_MR1` (these two already share the same 22-subject set).
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
