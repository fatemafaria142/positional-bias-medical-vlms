"""
Axis II: Sensitivity to Prompt Variation — Frozen Prompt Catalog Generator
for Neurite-OASIS (100 subjects) with VoxTell.

Implements Deliverable 1 of Axis_II_Prompt_Sensitivity_Nipu.docx:
  - Pulls the base anatomical targets from the Neurite-OASIS label set
    (manifest_neurite_oasis_100.csv / stage_neurite_oasis_100.py).
  - Designs 5-10 prompt variants per target covering:
      1. base (no side): e.g. "hippocampus", "Segment the hippocampus in this image."
      2. lateralized: e.g. "left hippocampus", "right hippocampus" (+ sentence templates)
      3. synonyms & paraphrases (8 per side per structure):
         - VoxTell precomputed-bank clinical synonyms (e.g. "left hippocampal formation",
           "left archicortical hippocampus", "hippocampus left side", "l. hippocampus")
         - Paraphrase templates ("Find the {organ} that is on the {side}.",
           "Segment the {organ} on the patient's {side} side.")
         - Anatomical-location descriptions ("Segment the structure in the {side} medial temporal lobe.")
      4. contradictory / nonsense / negative:
         - Opposite-side reference (evaluating a "left" prompt against "right" GT and vice versa)
         - Contradictory bilateral wording ("left {organ} on the right side of the brain")
         - Absent-organ negative prompts on brain MRI ("left kidney", "right kidney",
           "left adrenal tumor", "right adrenal tumor", "nodule in right upper lobe",
           "nodule in left lower lobe")
  - Also includes the supplemental AMOS & Lung-Lobe / Lesion reference table so the
    deliverable covers all prompt sets mentioned in Section 1 of the spec.
  - Saves the frozen catalog in CSV, JSON, and Markdown formats inside
    `exp0_frozen_prompts/` for full reproducibility.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Neurite-OASIS 35-label scheme (exact match with scripts/stage_neurite_oasis_100.py)
# ---------------------------------------------------------------------------
OASIS_LABEL_MAP: Dict[int, str] = {
    1: "left cerebral white matter",
    2: "left cerebral cortex",
    3: "left lateral ventricle",
    4: "left inferior lateral ventricle",
    5: "left cerebellum white matter",
    6: "left cerebellum cortex",
    7: "left thalamus",
    8: "left caudate",
    9: "left putamen",
    10: "left pallidum",
    11: "3rd ventricle",
    12: "4th ventricle",
    13: "brain stem",
    14: "left hippocampus",
    15: "left amygdala",
    16: "left accumbens",
    17: "left ventral DC",
    18: "left vessel",
    19: "left choroid plexus",
    20: "right cerebral white matter",
    21: "right cerebral cortex",
    22: "right lateral ventricle",
    23: "right inferior lateral ventricle",
    24: "right cerebellum white matter",
    25: "right cerebellum cortex",
    26: "right thalamus",
    27: "right caudate",
    28: "right putamen",
    29: "right pallidum",
    30: "right hippocampus",
    31: "right amygdala",
    32: "right accumbens",
    33: "right ventral DC",
    34: "right vessel",
    35: "right choroid plexus",
}

# All 16 bilateral pairs in Neurite-OASIS: target_core -> (left_label_id, right_label_id)
OASIS_BILATERAL_PAIRS: Dict[str, Tuple[int, int]] = {
    "hippocampus": (14, 30),
    "thalamus": (7, 26),
    "caudate": (8, 27),
    "putamen": (9, 28),
    "pallidum": (10, 29),
    "amygdala": (15, 31),
    "accumbens": (16, 32),
    "ventral DC": (17, 33),
    "cerebral white matter": (1, 20),
    "cerebellum white matter": (5, 24),
    "cerebellum cortex": (6, 25),
    "cerebral cortex": (2, 21),
    "lateral ventricle": (3, 22),
    "inferior lateral ventricle": (4, 23),
    "choroid plexus": (19, 35),
    "vessel": (18, 34),
}

# Core 7 bilateral structures where VoxTell v1.1 achieves high baseline Dice (~0.78 - 0.92)
CORE_7_STRUCTURES: List[str] = [
    "hippocampus",
    "thalamus",
    "caudate",
    "putamen",
    "cerebral white matter",
    "cerebellum white matter",
    "cerebellum cortex",
]

# Extended 11 bilateral structures supported in VoxTell's precomputed bank
EXTENDED_11_STRUCTURES: List[str] = CORE_7_STRUCTURES + [
    "pallidum",
    "amygdala",
    "accumbens",
    "ventral DC",
]

# All 16 bilateral structures in Neurite-OASIS
ALL_16_STRUCTURES: List[str] = list(OASIS_BILATERAL_PAIRS.keys())


# ---------------------------------------------------------------------------
# Structure-specific clinical synonyms, paraphrases, and location descriptions
# Verified against VoxTell's Qwen3-Embedding-4B vocabulary & anatomical ontology
# ---------------------------------------------------------------------------
OASIS_SYNONYM_SPEC: Dict[str, Dict[str, object]] = {
    "hippocampus": {
        "clinical_synonyms": [
            "{side} hippocampal formation",
            "{side} archicortical hippocampus",
            "hippocampus {side} side",
            "{side_abbr} hippocampus",
            "cornu ammonis of the {side} temporal lobe",
        ],
        "location_description": "structure in the {side} medial temporal lobe",
        "paraphrases": [
            "Find the hippocampus that is on the {side}.",
            "Segment the hippocampus on the patient's {side} side.",
            "Locate the {side}-sided hippocampal complex in this brain MRI.",
        ],
    },
    "thalamus": {
        "clinical_synonyms": [
            "{side} dorsal thalamus",
            "{side} diencephalic thalamus",
            "{side} thalamic nucleus",
            "{side_abbr} thalamus",
            "thalamic gray matter on the {side} side",
        ],
        "location_description": "deep relay nucleus in the {side} dorsal diencephalon",
        "paraphrases": [
            "Find the thalamus that is on the {side}.",
            "Segment the thalamus on the patient's {side} side.",
            "Locate the {side}-sided thalamic region in this brain MRI.",
        ],
    },
    "caudate": {
        "clinical_synonyms": [
            "{side} caudate nucleus",
            "{side} striatal caudate",
            "{side} caudate head and body",
            "{side_abbr} caudate",
            "caudate nucleus on the {side} side",
        ],
        "location_description": "C-shaped basal ganglia nucleus bordering the {side} lateral ventricle",
        "paraphrases": [
            "Find the caudate that is on the {side}.",
            "Segment the caudate nucleus on the patient's {side} side.",
            "Locate the {side}-sided neostriatal caudate in this brain MRI.",
        ],
    },
    "putamen": {
        "clinical_synonyms": [
            "{side} lentiform putamen",
            "{side} striatal putamen",
            "{side} neostriatal putamen",
            "{side_abbr} putamen",
            "putamen on the {side} side",
        ],
        "location_description": "outermost lentiform basal ganglia nucleus in the {side} hemisphere",
        "paraphrases": [
            "Find the putamen that is on the {side}.",
            "Segment the putamen on the patient's {side} side.",
            "Locate the {side}-sided putamen in this brain MRI.",
        ],
    },
    "cerebral white matter": {
        "clinical_synonyms": [
            "{side} cerebral white matter tissue",
            "{side} hemisphere white matter",
            "{side} brain white matter",
            "{side_abbr} cerebral white matter",
            "supratentorial white matter on the {side} side",
        ],
        "location_description": "subcortical myelinated white matter tracts of the {side} cerebral hemisphere",
        "paraphrases": [
            "Find the cerebral white matter that is on the {side}.",
            "Segment the cerebral white matter on the patient's {side} side.",
            "Locate the {side}-sided hemispheric white matter in this brain MRI.",
        ],
    },
    "cerebellum white matter": {
        "clinical_synonyms": [
            "{side} cerebellar white matter",
            "{side} hemispheric cerebellar white matter",
            "{side} cerebellar central white matter",
            "{side_abbr} cerebellar white matter",
            "arbor vitae white matter of the {side} cerebellum",
        ],
        "location_description": "deep white matter core inside the {side} cerebellar hemisphere",
        "paraphrases": [
            "Find the cerebellum white matter that is on the {side}.",
            "Segment the cerebellar white matter on the patient's {side} side.",
            "Locate the {side}-sided cerebellar axonal tissue in this brain MRI.",
        ],
    },
    "cerebellum cortex": {
        "clinical_synonyms": [
            "{side} cerebellar cortex",
            "{side} hemispheric cerebellar cortex",
            "{side} cerebellar gray matter cortex",
            "{side_abbr} cerebellar cortex",
            "folia cortex of the {side} cerebellum",
        ],
        "location_description": "outer gray matter cortical layer of the {side} cerebellar hemisphere",
        "paraphrases": [
            "Find the cerebellum cortex that is on the {side}.",
            "Segment the cerebellar cortex on the patient's {side} side.",
            "Locate the {side}-sided cerebellar cortex in this brain MRI.",
        ],
    },
    "pallidum": {
        "clinical_synonyms": [
            "{side} globus pallidus",
            "{side} basal ganglia pallidum",
            "{side} lentiform pallidum",
            "{side_abbr} pallidum",
            "pallidum on the {side} side",
        ],
        "location_description": "medial lentiform nucleus medial to the {side} putamen",
        "paraphrases": [
            "Find the pallidum that is on the {side}.",
            "Segment the globus pallidus on the patient's {side} side.",
            "Locate the {side}-sided lenticular pallidum in this brain MRI.",
        ],
    },
    "amygdala": {
        "clinical_synonyms": [
            "{side} limbic amygdala",
            "{side} temporal amygdala",
            "{side} amygdaloid body",
            "{side_abbr} amygdala",
            "amygdala on the {side} side",
        ],
        "location_description": "almond-shaped limbic nucleus anterior to the {side} hippocampus",
        "paraphrases": [
            "Find the amygdala that is on the {side}.",
            "Segment the amygdala on the patient's {side} side.",
            "Locate the {side}-sided temporal amygdala in this brain MRI.",
        ],
    },
    "accumbens": {
        "clinical_synonyms": [
            "{side} nucleus accumbens",
            "{side} accumbens area",
            "{side} ventral striatal accumbens",
            "{side_abbr} nucleus accumbens",
            "accumbens nucleus on the {side} side",
        ],
        "location_description": "ventral striatum nucleus at the junction of the {side} caudate and putamen",
        "paraphrases": [
            "Find the accumbens that is on the {side}.",
            "Segment the nucleus accumbens on the patient's {side} side.",
            "Locate the {side}-sided accumbens area in this brain MRI.",
        ],
    },
    "ventral DC": {
        "clinical_synonyms": [
            "{side} ventral diencephalon",
            "{side} ventral diencephalic structure",
            "{side} hypothalamic and subthalamic region",
            "{side_abbr} ventral diencephalon",
            "ventral DC on the {side} side",
        ],
        "location_description": "ventral diencephalic region inferior to the {side} thalamus",
        "paraphrases": [
            "Find the ventral DC that is on the {side}.",
            "Segment the ventral diencephalon on the patient's {side} side.",
            "Locate the {side}-sided ventral DC in this brain MRI.",
        ],
    },
    "cerebral cortex": {
        "clinical_synonyms": [
            "{side} cerebral cortex gray matter",
            "{side} cortical ribbon",
            "{side} hemispheric cerebral cortex",
            "{side_abbr} cerebral cortex",
            "cerebral cortex on the {side} side",
        ],
        "location_description": "outer cortical gray matter sheet of the {side} cerebral hemisphere",
        "paraphrases": [
            "Find the cerebral cortex that is on the {side}.",
            "Segment the cerebral cortex on the patient's {side} side.",
            "Locate the {side}-sided cortical gray matter in this brain MRI.",
        ],
    },
    "lateral ventricle": {
        "clinical_synonyms": [
            "{side} cerebral lateral ventricle",
            "{side} lateral ventricular cavity",
            "lateral ventricle on the {side} side",
            "{side_abbr} lateral ventricle",
            "{side} hemispheric lateral ventricle",
        ],
        "location_description": "CSF-filled ventricular cavity in the {side} cerebral hemisphere",
        "paraphrases": [
            "Find the lateral ventricle that is on the {side}.",
            "Segment the lateral ventricle on the patient's {side} side.",
            "Locate the {side}-sided lateral ventricle in this brain MRI.",
        ],
    },
    "inferior lateral ventricle": {
        "clinical_synonyms": [
            "{side} temporal horn of lateral ventricle",
            "{side} inferior horn of lateral ventricle",
            "inferior lateral ventricle on the {side} side",
            "{side_abbr} inferior lateral ventricle",
            "{side} temporal ventricular horn",
        ],
        "location_description": "temporal horn of the lateral ventricle in the {side} temporal lobe",
        "paraphrases": [
            "Find the inferior lateral ventricle that is on the {side}.",
            "Segment the inferior lateral ventricle on the patient's {side} side.",
            "Locate the {side}-sided temporal ventricular horn in this brain MRI.",
        ],
    },
    "choroid plexus": {
        "clinical_synonyms": [
            "{side} ventricular choroid plexus",
            "{side} choroid plexus of lateral ventricle",
            "choroid plexus on the {side} side",
            "{side_abbr} choroid plexus",
            "{side} lateral ventricle choroid glomus",
        ],
        "location_description": "CSF-producing vascular plexus inside the {side} lateral ventricle",
        "paraphrases": [
            "Find the choroid plexus that is on the {side}.",
            "Segment the choroid plexus on the patient's {side} side.",
            "Locate the {side}-sided choroid plexus in this brain MRI.",
        ],
    },
    "vessel": {
        "clinical_synonyms": [
            "{side} intracranial vessel",
            "{side} cerebral blood vessel",
            "brain vessel on the {side} side",
            "{side_abbr} cerebral vessel",
            "{side} lenticulostriate vascular structure",
        ],
        "location_description": "vascular structure in the {side} deep cerebral parenchyma",
        "paraphrases": [
            "Find the vessel that is on the {side}.",
            "Segment the vessel on the patient's {side} side.",
            "Locate the {side}-sided brain vessel in this brain MRI.",
        ],
    },
}


# Absent-organ / cross-domain negative prompts (Section 1 & Section 2/5 of Axis II doc)
OASIS_ABSENT_ORGAN_PROMPTS: List[Dict[str, str]] = [
    {
        "target": "kidney",
        "side": "none",
        "variant_type": "absent_organ_negative",
        "variant_subtype": "base_absent",
        "prompt_text": "Segment the kidney in this image.",
        "short_term": "kidney",
        "notes": "Abdominal organ absent from brain MRI (base negative control)",
    },
    {
        "target": "kidney",
        "side": "left",
        "variant_type": "absent_organ_negative",
        "variant_subtype": "lateralized_absent",
        "prompt_text": "Segment the left kidney in this image.",
        "short_term": "left kidney",
        "notes": "Abdominal organ absent from brain MRI (left negative control)",
    },
    {
        "target": "kidney",
        "side": "right",
        "variant_type": "absent_organ_negative",
        "variant_subtype": "lateralized_absent",
        "prompt_text": "Segment the right kidney in this image.",
        "short_term": "right kidney",
        "notes": "Abdominal organ absent from brain MRI (right negative control)",
    },
    {
        "target": "adrenal tumor",
        "side": "none",
        "variant_type": "absent_organ_negative",
        "variant_subtype": "lesion_base_absent",
        "prompt_text": "adrenal tumor",
        "short_term": "adrenal tumor",
        "notes": "Axis II doc Section 1 example: base adrenal lesion absent from brain MRI",
    },
    {
        "target": "adrenal tumor",
        "side": "left",
        "variant_type": "absent_organ_negative",
        "variant_subtype": "lesion_left_absent",
        "prompt_text": "left adrenal tumor",
        "short_term": "left adrenal tumor",
        "notes": "Axis II doc Section 1 example: left adrenal lesion absent from brain MRI",
    },
    {
        "target": "adrenal tumor",
        "side": "right",
        "variant_type": "absent_organ_negative",
        "variant_subtype": "lesion_right_absent",
        "prompt_text": "right adrenal tumor",
        "short_term": "right adrenal tumor",
        "notes": "Axis II doc Section 1 example: right adrenal lesion absent from brain MRI",
    },
    {
        "target": "lung nodule",
        "side": "right",
        "variant_type": "absent_organ_negative",
        "variant_subtype": "lung_lobe_right_upper_absent",
        "prompt_text": "nodule in right upper lobe",
        "short_term": "nodule in right upper lobe",
        "notes": "Axis II doc Section 1 example: right upper lobe nodule absent from brain MRI",
    },
    {
        "target": "lung nodule",
        "side": "left",
        "variant_type": "absent_organ_negative",
        "variant_subtype": "lung_lobe_left_lower_absent",
        "prompt_text": "nodule in left lower lobe",
        "short_term": "nodule in left lower lobe",
        "notes": "Axis II doc Section 1 example: left lower lobe nodule absent from brain MRI",
    },
]


@dataclass
class PromptRecord:
    prompt_id: str
    dataset: str
    target: str
    side: str                   # 'none' | 'left' | 'right' | 'contradictory'
    variant_type: str           # 'base' | 'lateralized' | 'synonym' | 'paraphrase' | 'location_based' | 'contradictory' | 'absent_organ_negative'
    variant_subtype: str        # e.g. 'canonical_sentence', 'short_term', 'clinical_synonym_1', ...
    prompt_text: str            # Exact string passed to VoxTell
    reference_prompt_text: str  # Reference lateralized or base prompt for comparison
    gt_label_id_left: Optional[int]
    gt_label_id_right: Optional[int]
    expected_label_id: Optional[int]
    opposite_label_id: Optional[int]
    matches_image_side: str     # 'both_present_unilateral_left' | 'both_present_unilateral_right' | 'bilateral_ambiguous' | 'contradictory_mismatch' | 'target_absent'
    in_core_7: bool
    in_extended_11: bool
    notes: str


def build_oasis_prompt_catalog(
    structures: Optional[List[str]] = None,
    include_absent_negatives: bool = True,
) -> List[PromptRecord]:
    """
    Build the complete frozen Axis II prompt table for Neurite-OASIS.

    For each bilateral brain structure in `structures` (defaults to all 16 bilateral
    pairs, with flags `in_core_7` and `in_extended_11` so experiments can filter
    cleanly), generates:
      - 2 Base (no-side) prompts: canonical sentence + short anatomical term
      - 2 Lateralized Left prompts: canonical sentence + short anatomical term
      - 2 Lateralized Right prompts: canonical sentence + short anatomical term
      - 5 Clinical Synonyms per side (10 total)
      - 3 Paraphrases per side (6 total)
      - 1 Anatomical-location wording per side (2 total)
      - 2 Contradictory / nonsense spatial prompts (left-on-right, right-on-left)
    Plus 8 absent-organ negative prompts (kidney, adrenal tumor, lung lobe nodule).
    """
    if structures is None:
        structures = ALL_16_STRUCTURES

    records: List[PromptRecord] = []
    idx = 1

    for target in structures:
        left_id, right_id = OASIS_BILATERAL_PAIRS[target]
        in_core7 = target in CORE_7_STRUCTURES
        in_ext11 = target in EXTENDED_11_STRUCTURES
        spec = OASIS_SYNONYM_SPEC[target]

        base_sentence = f"Segment the {target} in this image."
        left_sentence = f"Segment the left {target} in this image."
        right_sentence = f"Segment the right {target} in this image."

        # 1. Base (no side) — Sentence & Short term
        records.append(PromptRecord(
            prompt_id=f"OASIS_P{idx:04d}",
            dataset="neurite_oasis",
            target=target,
            side="none",
            variant_type="base",
            variant_subtype="canonical_sentence",
            prompt_text=base_sentence,
            reference_prompt_text=base_sentence,
            gt_label_id_left=left_id,
            gt_label_id_right=right_id,
            expected_label_id=None,
            opposite_label_id=None,
            matches_image_side="bilateral_ambiguous",
            in_core_7=in_core7,
            in_extended_11=in_ext11,
            notes="Base prompt (sentence template, no side word)",
        ))
        idx += 1

        records.append(PromptRecord(
            prompt_id=f"OASIS_P{idx:04d}",
            dataset="neurite_oasis",
            target=target,
            side="none",
            variant_type="base",
            variant_subtype="short_term",
            prompt_text=target,
            reference_prompt_text=f"left {target}",
            gt_label_id_left=left_id,
            gt_label_id_right=right_id,
            expected_label_id=None,
            opposite_label_id=None,
            matches_image_side="bilateral_ambiguous",
            in_core_7=in_core7,
            in_extended_11=in_ext11,
            notes="Base prompt (short anatomical noun phrase, no side word)",
        ))
        idx += 1

        # 2. Lateralized, Synonyms, Paraphrases, Location, Contradictory for each side
        for side, exp_id, opp_id, ref_sent in [
            ("left", left_id, right_id, left_sentence),
            ("right", right_id, left_id, right_sentence),
        ]:
            opp_side = "right" if side == "left" else "left"
            side_abbr = "l." if side == "left" else "r."
            short_lat = f"{side} {target}"

            # 2a. Canonical lateralized sentence (exact match with manifest_neurite_oasis_100.csv)
            records.append(PromptRecord(
                prompt_id=f"OASIS_P{idx:04d}",
                dataset="neurite_oasis",
                target=target,
                side=side,
                variant_type="lateralized",
                variant_subtype="canonical_sentence",
                prompt_text=ref_sent,
                reference_prompt_text=ref_sent,
                gt_label_id_left=left_id,
                gt_label_id_right=right_id,
                expected_label_id=exp_id,
                opposite_label_id=opp_id,
                matches_image_side=f"both_present_unilateral_{side}",
                in_core_7=in_core7,
                in_extended_11=in_ext11,
                notes=f"Canonical {side} prompt from manifest_neurite_oasis_100.csv",
            ))
            idx += 1

            # 2b. Short lateralized term (matches VoxTell's native training dictionary)
            records.append(PromptRecord(
                prompt_id=f"OASIS_P{idx:04d}",
                dataset="neurite_oasis",
                target=target,
                side=side,
                variant_type="lateralized",
                variant_subtype="short_term",
                prompt_text=short_lat,
                reference_prompt_text=ref_sent,
                gt_label_id_left=left_id,
                gt_label_id_right=right_id,
                expected_label_id=exp_id,
                opposite_label_id=opp_id,
                matches_image_side=f"both_present_unilateral_{side}",
                in_core_7=in_core7,
                in_extended_11=in_ext11,
                notes=f"Short {side} anatomical term (VoxTell precomputed bank style)",
            ))
            idx += 1

            # 2c. 5 Clinical Synonyms
            for s_idx, syn_tmpl in enumerate(spec["clinical_synonyms"], 1):
                syn_phrase = syn_tmpl.format(side=side, side_abbr=side_abbr)
                syn_sentence = f"Segment the {syn_phrase} in this image."
                records.append(PromptRecord(
                    prompt_id=f"OASIS_P{idx:04d}",
                    dataset="neurite_oasis",
                    target=target,
                    side=side,
                    variant_type="synonym",
                    variant_subtype=f"clinical_synonym_{s_idx}",
                    prompt_text=syn_sentence,
                    reference_prompt_text=ref_sent,
                    gt_label_id_left=left_id,
                    gt_label_id_right=right_id,
                    expected_label_id=exp_id,
                    opposite_label_id=opp_id,
                    matches_image_side=f"both_present_unilateral_{side}",
                    in_core_7=in_core7,
                    in_extended_11=in_ext11,
                    notes=f"Clinical synonym #{s_idx} for {side} {target} ({syn_phrase})",
                ))
                idx += 1

            # 2d. 3 Paraphrases
            for p_idx, para_tmpl in enumerate(spec["paraphrases"], 1):
                para_text = para_tmpl.format(side=side)
                records.append(PromptRecord(
                    prompt_id=f"OASIS_P{idx:04d}",
                    dataset="neurite_oasis",
                    target=target,
                    side=side,
                    variant_type="paraphrase",
                    variant_subtype=f"paraphrase_{p_idx}",
                    prompt_text=para_text,
                    reference_prompt_text=ref_sent,
                    gt_label_id_left=left_id,
                    gt_label_id_right=right_id,
                    expected_label_id=exp_id,
                    opposite_label_id=opp_id,
                    matches_image_side=f"both_present_unilateral_{side}",
                    in_core_7=in_core7,
                    in_extended_11=in_ext11,
                    notes=f"Syntactic paraphrase #{p_idx} for {side} {target}",
                ))
                idx += 1

            # 2e. 1 Anatomical-location description (without naming the target directly where possible)
            loc_phrase = spec["location_description"].format(side=side)
            loc_sentence = f"Segment the {loc_phrase} in this image."
            records.append(PromptRecord(
                prompt_id=f"OASIS_P{idx:04d}",
                dataset="neurite_oasis",
                target=target,
                side=side,
                variant_type="location_based",
                variant_subtype="anatomical_location_wording",
                prompt_text=loc_sentence,
                reference_prompt_text=ref_sent,
                gt_label_id_left=left_id,
                gt_label_id_right=right_id,
                expected_label_id=exp_id,
                opposite_label_id=opp_id,
                matches_image_side=f"both_present_unilateral_{side}",
                in_core_7=in_core7,
                in_extended_11=in_ext11,
                notes=f"Anatomical location wording for {side} {target}",
            ))
            idx += 1

            # 2f. Contradictory / conflicting spatial prompt
            contra_sentence = (
                f"Segment the {side} {target} located on the {opp_side} hemisphere in this image."
            )
            records.append(PromptRecord(
                prompt_id=f"OASIS_P{idx:04d}",
                dataset="neurite_oasis",
                target=target,
                side=side,
                variant_type="contradictory",
                variant_subtype=f"conflicting_{side}_on_{opp_side}",
                prompt_text=contra_sentence,
                reference_prompt_text=ref_sent,
                gt_label_id_left=left_id,
                gt_label_id_right=right_id,
                expected_label_id=exp_id,
                opposite_label_id=opp_id,
                matches_image_side="contradictory_mismatch",
                in_core_7=in_core7,
                in_extended_11=in_ext11,
                notes=f"Self-contradictory laterality prompt ({side} organ on {opp_side} side)",
            ))
            idx += 1

    # 3. Absent-organ / cross-domain negative prompts on Neurite-OASIS brain scans
    if include_absent_negatives:
        for neg in OASIS_ABSENT_ORGAN_PROMPTS:
            records.append(PromptRecord(
                prompt_id=f"OASIS_P{idx:04d}",
                dataset="neurite_oasis",
                target=neg["target"],
                side=neg["side"],
                variant_type=neg["variant_type"],
                variant_subtype=neg["variant_subtype"],
                prompt_text=neg["prompt_text"],
                reference_prompt_text="Segment the left hippocampus in this image.",
                gt_label_id_left=None,
                gt_label_id_right=None,
                expected_label_id=None,
                opposite_label_id=None,
                matches_image_side="target_absent",
                in_core_7=True,
                in_extended_11=True,
                notes=neg["notes"],
            ))
            idx += 1

    return records


def build_supplemental_amos_lesion_lung_prompts() -> List[Dict[str, object]]:
    """
    Build the supplemental frozen prompt table for AMOS, Adrenal Lesions, and Lung Lobes
    explicitly referenced in Section 1 & Section 7 of Axis_II_Prompt_Sensitivity_Nipu.docx.
    """
    rows: List[Dict[str, object]] = []
    pid = 1

    # A. Adrenal tumor / lesion variants (from Section 1 Table of Axis II docx)
    adrenal_variants = [
        ("none", "base", "adrenal tumor", "Reference output with no laterality"),
        ("left", "lateralized", "left adrenal tumor", "Tests whether left side word changes output"),
        ("right", "lateralized", "right adrenal tumor", "Tests whether right side word changes output"),
        ("left", "synonym", "left adrenal gland mass", "Clinical synonym 1 (left)"),
        ("left", "synonym", "adrenal lesion on the left side", "Clinical paraphrase 2 (left)"),
        ("left", "synonym", "tumor of the left suprarenal gland", "Anatomical synonym 3 (left)"),
        ("left", "synonym", "left suprarenal mass lesion", "Anatomical synonym 4 (left)"),
        ("left", "synonym", "neoplasm of the left adrenal gland", "Clinical synonym 5 (left)"),
        ("right", "synonym", "right adrenal gland mass", "Clinical synonym 1 (right)"),
        ("right", "synonym", "adrenal lesion on the right side", "Clinical paraphrase 2 (right)"),
        ("right", "synonym", "tumor of the right suprarenal gland", "Anatomical synonym 3 (right)"),
        ("right", "synonym", "right suprarenal mass lesion", "Anatomical synonym 4 (right)"),
        ("right", "synonym", "neoplasm of the right adrenal gland", "Clinical synonym 5 (right)"),
        ("left", "contradictory", "left adrenal tumor on the right side", "Contradictory side mismatch"),
        ("right", "contradictory", "right adrenal tumor on the left side", "Contradictory side mismatch"),
    ]
    for side, vtype, text, notes in adrenal_variants:
        rows.append({
            "prompt_id": f"SUPP_P{pid:04d}",
            "dataset": "lesion_adrenal",
            "target": "adrenal tumor",
            "side": side,
            "variant_type": vtype,
            "prompt_text": text,
            "notes": notes,
        })
        pid += 1

    # B. Lung lobe nodule variants (from Section 1 & Section 3 of Axis II docx)
    lung_variants = [
        ("none", "base", "lung nodule", "Base prompt with no lobe/side"),
        ("right_upper", "lateralized", "nodule in right upper lobe", "Right upper lobe canonical"),
        ("left_lower", "lateralized", "nodule in left lower lobe", "Left lower lobe canonical"),
        ("left_upper", "lateralized", "nodule in left upper lobe", "Left upper lobe canonical"),
        ("right_lower", "lateralized", "nodule in right lower lobe", "Right lower lobe canonical"),
        ("right_upper", "synonym", "pulmonary nodule in the right superior lobe", "Anatomical synonym"),
        ("right_upper", "synonym", "right upper lung lobe lesion", "Clinical synonym"),
        ("right_upper", "synonym", "focal mass in the upper lobe of the right lung", "Paraphrase"),
        ("left_lower", "synonym", "pulmonary nodule in the left inferior lobe", "Anatomical synonym"),
        ("left_lower", "synonym", "left lower lung lobe lesion", "Clinical synonym"),
        ("left_lower", "synonym", "focal mass in the lower lobe of the left lung", "Paraphrase"),
        ("right_upper", "contradictory", "nodule in right upper lobe of the left lung", "Contradictory lobe/side"),
    ]
    for side, vtype, text, notes in lung_variants:
        rows.append({
            "prompt_id": f"SUPP_P{pid:04d}",
            "dataset": "lung_lobe_lesion",
            "target": "lung nodule",
            "side": side,
            "variant_type": vtype,
            "prompt_text": text,
            "notes": notes,
        })
        pid += 1

    # C. AMOS22-MRI bilateral organs (kidney, adrenal gland)
    for organ, syn_noun in [("kidney", "renal organ"), ("adrenal gland", "suprarenal gland")]:
        for side, vtype, text in [
            ("none", "base", f"Segment the {organ} in this image."),
            ("left", "lateralized", f"Segment the left {organ} in this image."),
            ("right", "lateralized", f"Segment the right {organ} in this image."),
            ("left", "synonym", f"Segment the left {syn_noun} in this image."),
            ("left", "synonym", f"Segment the {organ} on the left side in this image."),
            ("left", "paraphrase", f"Find the {organ} that is on the left."),
            ("right", "synonym", f"Segment the right {syn_noun} in this image."),
            ("right", "synonym", f"Segment the {organ} on the right side in this image."),
            ("right", "paraphrase", f"Find the {organ} that is on the right."),
            ("left", "contradictory", f"Segment the left {organ} on the right side in this image."),
        ]:
            rows.append({
                "prompt_id": f"SUPP_P{pid:04d}",
                "dataset": "amos22_mri",
                "target": organ,
                "side": side,
                "variant_type": vtype,
                "prompt_text": text,
                "notes": f"AMOS22-MRI {vtype} prompt for {organ}",
            })
            pid += 1

    return rows


def export_frozen_prompt_table(
    output_dir: Path,
    structures: Optional[List[str]] = None,
    oasis_manifest_csv: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Freeze and export the Axis II prompt catalog into `output_dir`:
      - frozen_prompts_neurite_oasis.csv (prompt-level catalog)
      - manifest_neurite_oasis_100_synonyms.csv (100-subject x prompt-variant manifest)
      - frozen_prompts_supplemental_amos_lesion_lung.csv
      - frozen_prompts_all.json
      - README_frozen_prompts.md
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oasis_records = build_oasis_prompt_catalog(structures=structures, include_absent_negatives=True)
    oasis_dicts = [asdict(r) for r in oasis_records]
    supp_dicts = build_supplemental_amos_lesion_lung_prompts()

    oasis_csv = output_dir / "frozen_prompts_neurite_oasis.csv"
    with oasis_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(oasis_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(oasis_dicts)

    supp_csv = output_dir / "frozen_prompts_supplemental_amos_lesion_lung.csv"
    with supp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(supp_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(supp_dicts)

    # Build the 100-subject x synonym/variant manifest so it can be tracked in Git
    # and loaded directly from RESEARCH_REPO in Kaggle without modifying the HF dataset.
    repo_root = Path(__file__).resolve().parent.parent
    if oasis_manifest_csv is None:
        candidate = repo_root / "data" / "neurite_oasis_100" / "manifest_neurite_oasis_100.csv"
        if candidate.exists():
            oasis_manifest_csv = candidate

    synonym_manifest_csv = output_dir / "manifest_neurite_oasis_100_synonyms.csv"
    subject_rows: List[Dict[str, object]] = []
    if oasis_manifest_csv is not None and Path(oasis_manifest_csv).exists():
        # Index original manifest by (subject_id, label_id)
        by_subj_label: Dict[Tuple[str, int], Dict[str, str]] = {}
        subjects_ordered: List[str] = []
        subj_meta: Dict[str, Dict[str, str]] = {}
        with Path(oasis_manifest_csv).open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row["subject_id"]
                lid = int(row["label_id"])
                by_subj_label[(sid, lid)] = row
                if sid not in subj_meta:
                    subjects_ordered.append(sid)
                    subj_meta[sid] = {
                        "image_path": row["image_path"],
                        "label_path": row["label_path"],
                        "volume_shape": row["volume_shape"],
                        "voxel_spacing_mm": row["voxel_spacing_mm"],
                        "axcodes": row["axcodes"],
                    }

        # Filter to Core-7 structures + absent-organ negatives for the 100-subject manifest
        # so the CSV stays compact (~19,000 rows, ~3 MB) and git-friendly while covering all
        # 100 subjects x (base, lateralized, 5 synonyms, 3 paraphrases, location, contradictory).
        core_prompts = [r for r in oasis_records if r.in_core_7]
        for sid in subjects_ordered:
            meta = subj_meta[sid]
            for pr in core_prompts:
                exp_lid = pr.expected_label_id if pr.expected_label_id is not None else pr.gt_label_id_left
                gt_row = by_subj_label.get((sid, exp_lid)) if exp_lid is not None else None
                subject_rows.append({
                    "dataset": "neurite_oasis",
                    "subject_id": sid,
                    "prompt_id": pr.prompt_id,
                    "target": pr.target,
                    "side": pr.side,
                    "variant_type": pr.variant_type,
                    "variant_subtype": pr.variant_subtype,
                    "prompt_text": pr.prompt_text,
                    "reference_prompt_text": pr.reference_prompt_text,
                    "gt_label_id_left": pr.gt_label_id_left if pr.gt_label_id_left is not None else "",
                    "gt_label_id_right": pr.gt_label_id_right if pr.gt_label_id_right is not None else "",
                    "expected_label_id": pr.expected_label_id if pr.expected_label_id is not None else "",
                    "opposite_label_id": pr.opposite_label_id if pr.opposite_label_id is not None else "",
                    "matches_image_side": pr.matches_image_side,
                    "image_path": meta["image_path"],
                    "label_path": meta["label_path"],
                    "volume_shape": meta["volume_shape"],
                    "voxel_spacing_mm": meta["voxel_spacing_mm"],
                    "axcodes": meta["axcodes"],
                    "gt_centroid_vox_0": gt_row["gt_centroid_vox_0"] if gt_row else "",
                    "gt_centroid_vox_1": gt_row["gt_centroid_vox_1"] if gt_row else "",
                    "gt_centroid_vox_2": gt_row["gt_centroid_vox_2"] if gt_row else "",
                })

        with synonym_manifest_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(subject_rows[0].keys()))
            writer.writeheader()
            writer.writerows(subject_rows)

        # Also copy to repo root `manifests/manifest_neurite_oasis_100_synonyms.csv`
        # so it is tracked in Git and immediately available when cloned in Kaggle.
        git_manifest_dir = repo_root / "manifests"
        git_manifest_dir.mkdir(parents=True, exist_ok=True)
        shutil_copy = git_manifest_dir / "manifest_neurite_oasis_100_synonyms.csv"
        shutil_copy.write_text(synonym_manifest_csv.read_text(encoding="utf-8"), encoding="utf-8")
        (git_manifest_dir / "frozen_prompts_neurite_oasis.csv").write_text(
            oasis_csv.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (git_manifest_dir / "frozen_prompts_supplemental_amos_lesion_lung.csv").write_text(
            supp_csv.read_text(encoding="utf-8"), encoding="utf-8"
        )

    json_path = output_dir / "frozen_prompts_all.json"
    payload = {
        "axis": "Axis II: Sensitivity to Prompt Variation",
        "model": "VoxTell (voxtell_v1.1 + Qwen/Qwen3-Embedding-4B)",
        "primary_dataset": "Fatema142/neurite-oasis-100-rq-analysis (100 subjects)",
        "counts": {
            "neurite_oasis_total_prompts": len(oasis_dicts),
            "neurite_oasis_core_7_prompts": sum(1 for r in oasis_dicts if r["in_core_7"]),
            "neurite_oasis_extended_11_prompts": sum(1 for r in oasis_dicts if r["in_extended_11"]),
            "supplemental_prompts": len(supp_dicts),
        },
        "core_7_structures": CORE_7_STRUCTURES,
        "extended_11_structures": EXTENDED_11_STRUCTURES,
        "all_16_structures": ALL_16_STRUCTURES,
        "neurite_oasis_prompts": oasis_dicts,
        "supplemental_prompts": supp_dicts,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = output_dir / "README_frozen_prompts.md"
    lines = [
        "# Axis II — Deliverable 1: Frozen Prompt Catalog",
        "",
        "## Overview",
        f"- **Total Neurite-OASIS Frozen Prompts:** {len(oasis_dicts)}",
        f"- **Core-7 Bilateral Structures Prompts:** {sum(1 for r in oasis_dicts if r['in_core_7'])}",
        f"- **Extended-11 Bilateral Structures Prompts:** {sum(1 for r in oasis_dicts if r['in_extended_11'])}",
        f"- **Supplemental (AMOS / Adrenal Lesion / Lung Lobe) Prompts:** {len(supp_dicts)}",
        "",
        "## Variant Breakdown per Bilateral Structure (26 prompts per structure)",
        "| Variant Type | Count per Target | Purpose |",
        "|---|---:|---|",
        "| `base` (no side) | 2 | Reference output with no laterality (`Segment the {organ} in this image.` & `{organ}`) |",
        "| `lateralized` | 4 | Canonical left/right sentence + short `{side} {organ}` term |",
        "| `synonym` | 10 | 5 clinical/anatomical synonyms per side (e.g., hippocampal formation, cornu ammonis, `l.`/`r.`) |",
        "| `paraphrase` | 6 | 3 syntactic rephrasings per side (`Find the {organ} that is on the {side}.`, etc.) |",
        "| `location_based` | 2 | 1 anatomical-location description per side (`structure in the {side} medial temporal lobe`) |",
        "| `contradictory` | 2 | Conflicting spatial wording (`{side} {organ} located on the {opposite} hemisphere`) |",
        "",
        "## Sample Prompts (Hippocampus + Absent-Organ Negatives)",
        "| ID | Target | Side | Variant Type | Subtype | Prompt Text |",
        "|---|---|---|---|---|---|",
    ]
    for r in oasis_dicts:
        if r["target"] == "hippocampus" or r["variant_type"] == "absent_organ_negative":
            lines.append(
                f"| `{r['prompt_id']}` | {r['target']} | {r['side']} | `{r['variant_type']}` | `{r['variant_subtype']}` | {r['prompt_text']} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "oasis_csv": oasis_csv,
        "synonym_manifest_csv": synonym_manifest_csv,
        "supp_csv": supp_csv,
        "json": json_path,
        "markdown": md_path,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "results_axis2" / "exp0_frozen_prompts"
    paths = export_frozen_prompt_table(out)
    for k, p in paths.items():
        print(f"[ok] {k}: {p}")
