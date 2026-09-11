# Table — Single-Region vs. Multi-Region Capability, by Model

This tracks, for the multi-region positional-bias work, which models can take **one
image + one prompt naming several regions** and produce **separate masks per named
region in that one call**, versus which only ever produce one output per call and
need a second-stage pipeline (usually SAM) bolted on to get from "located regions"
to "pixel masks." This is a different question from raw segmentation *quality* --
it's purely about architecture: does multi-region output exist natively, or does it
have to be assembled from single-region primitives.

## Single-Region vs. Multi-Region: the difference

**Table 1.** The conceptual difference between single-region and multi-region
models — what each term means, what a call actually returns, why the
architecture behaves that way, and how each side maps to what this project's
`P_ALL` manifest rows need.

| Aspect | Single-Region | Multi-Region |
|---|---|---|
| **What it means** | One prompt -> exactly one output (one mask/box/point), no matter how many objects the prompt names | One prompt -> a separate output (mask/box/point) *per named object*, all from that same one call |
| **Example prompt** | `"segment the liver"` (one object -- this is the model's intended, native use case) | `"segment liver ; right kidney ; spleen"` (or a text list passed as separate queries) -- one call, several objects named |
| **What comes back** | Exactly 1 mask/box/point -- matches the 1 object asked for | N masks/boxes/points -- one distinctly tied to each named object |
| **If you force multiple objects into one prompt anyway** | Still only 1 output back -- the model has no mechanism to emit a second one. What that single output looks like varies: CLIP-pooled models (CLIPSeg) tend to produce one blended/ambiguous region touching several named objects (the whole prompt collapsed into one embedding); other single-region models might instead just lock onto one of the named objects and ignore the rest. Either way: one output, not several | N/A -- this is the native case, not a workaround |
| **Why it happens (architecture)** | The whole prompt is pooled into **one** embedding before decoding -- there's no per-object signal left once pooling happens | The output format *repeats* a location/mask unit once per target, either in one generated sequence (token repetition) or one batched forward pass (detection-style set prediction) |
| **Can you still get N results?** | Only by calling the model N times, once per object -- one call per region | Yes, natively, in the same one call |
| **Typical model family** | CLIP-style dual-encoder models (CLIPSeg, BiomedCLIP/MedCLIP-SAMv2) | Detection/grounding models (Grounding DINO, OWLv2, Kosmos-2, Qwen2.5-VL, LocateAnything) and token-repetition models (PaliGemma 2) |
| **Does it need SAM?** | Not necessarily (some, like CLIPSeg, decode their own soft mask) -- but SAM doesn't help the single-region *limitation* either, since SAM only refines whatever region it's pointed at | Detection-style multi-region models still need SAM to turn their N boxes/points into N pixel masks; PaliGemma 2 doesn't, since it decodes pixel masks natively |
| **Fits this project's P_ALL rows?** | No -- P_ALL rows need one mask *per named region* (`target_regions`), not one merged region | Yes -- this is the capability P_ALL rows are testing for |
| **Already seen in this repo** | MedCLIP-SAMv2's P_ALL scoring note (`scripts/medclip_samv2/run_medclip_samv2.py`): one combined prompt still only gets one merged saliency blob, scored against the *union* of all named regions' masks | BiomedParse's `interactive_infer_image(model, image, prompts)` (`BiomedParse_Multi_Region_Positional_Bias.ipynb`): one image, a list of region-name prompts, one mask back per prompt in a single call |

## Why some models "require SAM"

SAM (Segment Anything Model) itself has no language understanding -- it's a
promptable mask decoder that turns a **box, point, or coarse mask prompt** into a
refined pixel mask. It's paired with a grounding/detection VLM whenever that VLM's
own output stops at coordinates (boxes or points) rather than pixels:

```
text prompt -> [grounding VLM] -> box(es)/point(s) -> [SAM] -> pixel mask(s)
```

That two-stage design is necessary because most open-vocabulary grounding models
(Grounding DINO, OWLv2, Kosmos-2, Qwen2.5-VL, LocateAnything) were trained for
**detection/localization**, not pixel-level segmentation -- their output head
predicts coordinates, and nothing in their architecture decodes a mask. SAM is
the standard way to close that gap because it accepts box/point prompts directly
and (via `predict_torch`/batched prompts) can turn *several* boxes or points from
one image into several masks in one SAM call -- so a model that finds N regions as
boxes can still end up with N real masks, just via two models instead of one.

Models that skip SAM entirely (PaliGemma 2, CLIPSeg, Florence-2's segmentation
head) instead have a **mask-producing decoder of their own** baked into the model
(a VQ-VAE token decoder, a conv decoder on top of pooled text+image features,
etc.) -- there's no coordinate intermediate to hand off.

## The 3 categories a model can fall into

For "one image, one prompt naming several regions," a model's architecture puts it
into one of three buckets -- this is what the **Region Capability** column below
calls out directly:

1. **Multi-region** -- the output format *repeats* a location/mask unit once per
   named target in a single generated sequence or a single batched forward pass.
   E.g. PaliGemma repeats its `<loc>`/`<seg>` token group once per phrase separated
   by `;` in the prompt; detection-style models (Grounding DINO, OWLv2, Kosmos-2,
   Qwen2.5-VL, LocateAnything) predict a *set* of boxes/points per forward pass by
   design (that's what open-vocabulary detection means) -- masks then come either
   natively (PaliGemma) or via a second SAM pass over that same set.
2. **Single-region only** -- the model pools the *entire* prompt into **one**
   embedding before decoding, so there is no per-sub-phrase signal left to split
   back out. A prompt naming 4 organs still produces exactly 1 blended output.
   This is exactly the CLIPSeg/BiomedCLIP limitation already documented in this
   repo for MedCLIP-SAMv2 (`scripts/medclip_samv2/run_medclip_samv2.py`, P_ALL
   scoring note).
3. **Multi-instance, not multi-region** -- the model can return several points for
   one query (e.g. "point to every lesion"), but that's multiple occurrences of
   **one** described class, not several independently named, differently
   described objects in one call. Moondream2 and Molmo2-8B fall here -- a
   different capability than what this project's P_ALL rows need (naming
   distinct organs by name in one sentence).

## Per-Model Comparison: Single-Region vs. Multi-Region Capability

**Table 2.** Single-region vs. multi-region capability, side by side, for
every surveyed model that can be given a multi-object prompt — showing what
each model natively outputs (masks, boxes, or points) and whether a second
SAM pass is needed to turn that output into a pixel mask.

| Model | Input Format | Text Prompt Content | Single-Region Capable | Multi-Region Capable | Native Output Format | Requires SAM |
|---|---|---|---|---|---|---|
| **PaliGemma 2 (3B)** | Image + Text | Multiple named objects | Yes | Yes (native masks) | Pixel masks | No |
| **Grounding DINO** | Image + Text | Multiple named objects/phrases | Yes | Yes (boxes only) | Bounding boxes | Yes |
| **OWLv2** | Image + Text | Multiple text queries | Yes | Yes (boxes only) | Bounding boxes | Yes |
| **Kosmos-2** | Image + Text | Multiple phrases | Yes | Yes (boxes/regions only) | Bounding boxes / grounded regions | Yes |
| **Qwen2.5-VL-3B** | Image + Text | Multiple named objects | Yes | Yes (boxes only) | Structured bounding boxes | Yes |
| **LocateAnything-3B (NVIDIA)** | Image + Text | Multiple named objects | Yes | Yes (points only) | Points / grounding locations | Yes |
| **Moondream2** | Image + Text | Multiple instances of the same described object | Yes | No (multi-instance of one class only, not multi-region) | Points | Yes |
| **Molmo2-8B** | Image + Text | Multiple instances of the same described object | Yes | No (multi-instance of one class only, not multi-region) | Points | Yes |
| **Florence-2** | Image + Text | Multiple objects (detection only) | Yes (segmentation task) | Yes (detection task only, not segmentation) | Boxes/regions (detection); mask (referring segmentation) | Yes (for masks) |
| **MedCLIP-SAMv2** | Image + Text | One curated object per prompt | Yes | No (multi-object prompt collapses into one blended saliency blob) | Saliency map -> box/point -> mask | Yes (own 3-stage pipeline) |
| **North-Micro-Vision-Instruct** | Image + Text | Unclear | Unconfirmed | Unconfirmed | Grounding / VQA / OCR capabilities | Unknown |

## Why each model lands where it does

- **PaliGemma 2 (3B)** -- Multi-region, no SAM needed. A prompt like
  `segment liver ; right kidney ; spleen` produces one `<loc>`+`<seg000>`-`<seg127>`
  token group *per named phrase* in the same generated sequence; each group's seg
  tokens decode through PaliGemma's own VQ-VAE into an independent pixel mask. The
  token grammar itself repeats per target, so no external mask model is needed.
- **Grounding DINO** -- Multi-region (boxes), needs SAM. Built for text-to-region
  grounding; one forward pass over a `.`-separated phrase list returns a *set* of
  boxes (one per matched phrase). It has no mask decoder of its own, so SAM's
  batched-box prompting is what turns that same set of boxes into a matching set
  of masks in one SAM call.
- **OWLv2** -- Multi-region (boxes), needs SAM. Open-vocabulary detector -- one
  image, list of text queries, one *set* of boxes back, same shape as Grounding
  DINO. No pixel decoder on board, so SAM is required to get from boxes to masks.
- **Kosmos-2** -- Multi-region (boxes/regions), needs SAM. Grounds several phrases
  to regions in one generation, as part of its image-captioning-with-grounding
  design. "Grounding" here means coordinates in its markup output, not pixels, so
  SAM is the only way to turn those regions into masks.
- **Qwen2.5-VL-3B** -- Multi-region (boxes), needs SAM. Can emit a structured
  (e.g. JSON) list of boxes for multiple named objects in one response --
  general-purpose VLM rather than a segmentation model. No native mask head, so a
  second SAM pass is the only route to pixels.
- **LocateAnything-3B (NVIDIA)** -- Multi-region (points), needs SAM. Generalist
  grounding model with explicit multi-object localization -- one call can return
  several named objects' locations. Points, not masks, so SAM's point-prompt mode
  is needed to expand each point into a mask.
- **Moondream2** -- Multi-instance only, needs SAM. Can point to every instance of
  *one* described thing ("every lesion"), which is multi-instance, not
  multi-region -- it never takes several *differently* named targets in one call
  the way this project's `target_regions` rows do. Points still need SAM to
  become masks.
- **Molmo2-8B** -- Multi-instance only, needs SAM. Same shape as Moondream2 --
  good for counting/pointing repeats of one class, not a native
  multi-named-object mask generator. Needs SAM for pixels.
- **Florence-2** -- Multi-region for detection, single-region for segmentation.
  Its open-vocabulary **detection** task natively returns boxes for several
  objects in one call. Its separate referring-expression-segmentation task is
  one-phrase-in -> one-mask-out by design, so several named organs' masks in one
  image still means either N separate referring-seg calls, or running detection
  once and handing all N boxes to SAM together.
- **MedCLIP-SAMv2** -- Single-region only, needs SAM. A 3-stage pipeline
  (BiomedCLIP saliency map -> coarse box/point -> SAM refinement) already
  implemented in this repo (`scripts/medclip_samv2/run_medclip_samv2.py`): one
  curated text prompt pools into one saliency map, so a prompt naming several
  organs still collapses into a single blended saliency blob rather than N
  separate ones -- the same CLIP-pooling ceiling CLIPSeg hits. SAM is already
  part of its own pipeline (refining the saliency-derived box/point into a
  mask), but that doesn't lift the single-region limit since SAM only refines
  whatever one region it's pointed at.
- **North-Micro-Vision-Instruct** -- Unconfirmed. Not enough public documentation
  found to confirm its grounding output format (boxes vs. points vs. masks) or
  whether one call handles multiple named targets -- flagged for follow-up rather
  than guessed at.

## Reading this against the project's P_ALL rows

This project's multi-region manifests (`dataset/acdc`, `dataset/chaos`,
`dataset/isic_task2`) test exactly the "multiple *named*, *distinct* regions in
one image, one prompt" case (`P_ALL` rows: `target_regions` lists e.g.
`liver;right_kidney;left_kidney;spleen`). Against that specific bar:

- **PaliGemma 2** is the only model above that's genuinely multi-region *and*
  mask-native in one call.
- **Grounding DINO, OWLv2, Kosmos-2, Qwen2.5-VL, LocateAnything** are
  multi-region at the box/point stage and can still reach real per-region masks --
  just via a two-stage (VLM + SAM) pipeline, the same shape already used for
  MedCLIP-SAMv2 in this repo (BiomedCLIP saliency + SAM), and comparable to the
  BiomedParse pipeline (`BiomedParse_Multi_Region_Positional_Bias.ipynb`), which
  also needs one call per region name but shares the image encoding across them.
- **Moondream2 and Molmo2-8B** don't fit the P_ALL question at all -- they
  solve a related but different problem (repeat-instance pointing).
- **Florence-2's referring segmentation** hits the same single-embedding
  ceiling that CLIPSeg/BiomedCLIP-style models run into (see MedCLIP-SAMv2's
  P_ALL notes in this repo) -- excluded from this table since it can't be
  given a multi-object prompt in the first place, but worth keeping in mind
  as the reason its segmentation task stays single-region even though its
  detection task isn't.
