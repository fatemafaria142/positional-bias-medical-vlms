# Axis II — Deliverable 1: Frozen Prompt Catalog

## Overview
- **Total Neurite-OASIS Frozen Prompts:** 424
- **Core-7 Bilateral Structures Prompts:** 190
- **Extended-11 Bilateral Structures Prompts:** 294
- **Supplemental (AMOS / Adrenal Lesion / Lung Lobe) Prompts:** 47

## Variant Breakdown per Bilateral Structure (26 prompts per structure)
| Variant Type | Count per Target | Purpose |
|---|---:|---|
| `base` (no side) | 2 | Reference output with no laterality (`Segment the {organ} in this image.` & `{organ}`) |
| `lateralized` | 4 | Canonical left/right sentence + short `{side} {organ}` term |
| `synonym` | 10 | 5 clinical/anatomical synonyms per side (e.g., hippocampal formation, cornu ammonis, `l.`/`r.`) |
| `paraphrase` | 6 | 3 syntactic rephrasings per side (`Find the {organ} that is on the {side}.`, etc.) |
| `location_based` | 2 | 1 anatomical-location description per side (`structure in the {side} medial temporal lobe`) |
| `contradictory` | 2 | Conflicting spatial wording (`{side} {organ} located on the {opposite} hemisphere`) |

## Sample Prompts (Hippocampus + Absent-Organ Negatives)
| ID | Target | Side | Variant Type | Subtype | Prompt Text |
|---|---|---|---|---|---|
| `OASIS_P0001` | hippocampus | none | `base` | `canonical_sentence` | Segment the hippocampus in this image. |
| `OASIS_P0002` | hippocampus | none | `base` | `short_term` | hippocampus |
| `OASIS_P0003` | hippocampus | left | `lateralized` | `canonical_sentence` | Segment the left hippocampus in this image. |
| `OASIS_P0004` | hippocampus | left | `lateralized` | `short_term` | left hippocampus |
| `OASIS_P0005` | hippocampus | left | `synonym` | `clinical_synonym_1` | Segment the left hippocampal formation in this image. |
| `OASIS_P0006` | hippocampus | left | `synonym` | `clinical_synonym_2` | Segment the left archicortical hippocampus in this image. |
| `OASIS_P0007` | hippocampus | left | `synonym` | `clinical_synonym_3` | Segment the hippocampus left side in this image. |
| `OASIS_P0008` | hippocampus | left | `synonym` | `clinical_synonym_4` | Segment the l. hippocampus in this image. |
| `OASIS_P0009` | hippocampus | left | `synonym` | `clinical_synonym_5` | Segment the cornu ammonis of the left temporal lobe in this image. |
| `OASIS_P0010` | hippocampus | left | `paraphrase` | `paraphrase_1` | Find the hippocampus that is on the left. |
| `OASIS_P0011` | hippocampus | left | `paraphrase` | `paraphrase_2` | Segment the hippocampus on the patient's left side. |
| `OASIS_P0012` | hippocampus | left | `paraphrase` | `paraphrase_3` | Locate the left-sided hippocampal complex in this brain MRI. |
| `OASIS_P0013` | hippocampus | left | `location_based` | `anatomical_location_wording` | Segment the structure in the left medial temporal lobe in this image. |
| `OASIS_P0014` | hippocampus | left | `contradictory` | `conflicting_left_on_right` | Segment the left hippocampus located on the right hemisphere in this image. |
| `OASIS_P0015` | hippocampus | right | `lateralized` | `canonical_sentence` | Segment the right hippocampus in this image. |
| `OASIS_P0016` | hippocampus | right | `lateralized` | `short_term` | right hippocampus |
| `OASIS_P0017` | hippocampus | right | `synonym` | `clinical_synonym_1` | Segment the right hippocampal formation in this image. |
| `OASIS_P0018` | hippocampus | right | `synonym` | `clinical_synonym_2` | Segment the right archicortical hippocampus in this image. |
| `OASIS_P0019` | hippocampus | right | `synonym` | `clinical_synonym_3` | Segment the hippocampus right side in this image. |
| `OASIS_P0020` | hippocampus | right | `synonym` | `clinical_synonym_4` | Segment the r. hippocampus in this image. |
| `OASIS_P0021` | hippocampus | right | `synonym` | `clinical_synonym_5` | Segment the cornu ammonis of the right temporal lobe in this image. |
| `OASIS_P0022` | hippocampus | right | `paraphrase` | `paraphrase_1` | Find the hippocampus that is on the right. |
| `OASIS_P0023` | hippocampus | right | `paraphrase` | `paraphrase_2` | Segment the hippocampus on the patient's right side. |
| `OASIS_P0024` | hippocampus | right | `paraphrase` | `paraphrase_3` | Locate the right-sided hippocampal complex in this brain MRI. |
| `OASIS_P0025` | hippocampus | right | `location_based` | `anatomical_location_wording` | Segment the structure in the right medial temporal lobe in this image. |
| `OASIS_P0026` | hippocampus | right | `contradictory` | `conflicting_right_on_left` | Segment the right hippocampus located on the left hemisphere in this image. |
| `OASIS_P0417` | kidney | none | `absent_organ_negative` | `base_absent` | Segment the kidney in this image. |
| `OASIS_P0418` | kidney | left | `absent_organ_negative` | `lateralized_absent` | Segment the left kidney in this image. |
| `OASIS_P0419` | kidney | right | `absent_organ_negative` | `lateralized_absent` | Segment the right kidney in this image. |
| `OASIS_P0420` | adrenal tumor | none | `absent_organ_negative` | `lesion_base_absent` | adrenal tumor |
| `OASIS_P0421` | adrenal tumor | left | `absent_organ_negative` | `lesion_left_absent` | left adrenal tumor |
| `OASIS_P0422` | adrenal tumor | right | `absent_organ_negative` | `lesion_right_absent` | right adrenal tumor |
| `OASIS_P0423` | lung nodule | right | `absent_organ_negative` | `lung_lobe_right_upper_absent` | nodule in right upper lobe |
| `OASIS_P0424` | lung nodule | left | `absent_organ_negative` | `lung_lobe_left_lower_absent` | nodule in left lower lobe |
