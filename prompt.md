Please filter the dataset search further based on these requirements:



Exclude MRI-based datasets.

Exclude chest/thoracic imaging datasets, since this is already a well-explored domain.

Exclude dermatology datasets, including ISIC and similar skin-lesion datasets.

Please focus specifically on CT-based datasets and, if appropriate, other non-MRI modalities that fit the criteria.



There is one additional and important requirement: the anatomy being studied should be deformable.



I am exploring a research problem related to anatomical deformation, so I need datasets where the target anatomy can undergo meaningful changes in shape, position, or spatial configuration. Please prioritize datasets where deformation is biologically or clinically relevant rather than structures that are essentially rigid.



For example, consider anatomical structures or organs that can change due to:



patient positioning

physiological motion

organ filling/emptying

respiration or other motion

pathology

surgical intervention

external forces or compression

inter-patient anatomical variation

For example, consider anatomical structures or organs that can change due to:



patient positioning or changes in body posture

physiological motion

respiration and breathing cycles

cardiac motion

organ filling and emptying

bladder or bowel distension

swallowing or muscular contraction

peristalsis and gastrointestinal movement

changes in blood flow or vascular pressure

changes in tissue pressure or fluid accumulation

edema or swelling

tumor growth or regression

disease progression

inflammation or infection

tissue atrophy or degeneration

fibrosis or scarring

organ displacement caused by neighboring structures

mass effect from tumors or other lesions

anatomical changes caused by trauma

external compression or mechanical forces

surgical intervention or resection

minimally invasive procedures

radiation therapy or other treatment-related changes

preoperative versus postoperative anatomical changes

changes between different imaging time points

inter-patient anatomical variability

age-related anatomical changes

changes associated with different physiological states

changes caused by different levels of organ contraction or relaxation

deformation caused by contact with surrounding tissues or anatomical structures

changes in shape, orientation, position, or volume of the target anatomy



The dataset does not necessarily need to explicitly provide deformation annotations. However, the anatomy should have a meaningful potential to deform, move, change shape, or change position, so that it can support research into how AI models respond to anatomical and spatial changes.

Please search specifically for small-sample, publicly available, well-established benchmark datasets that satisfy these constraints.



Return the results as a table with:



| Dataset | Anatomy | Modality | Task | Number of Cases/Images | Annotation | Why the Anatomy Is Deformable | Benchmark Usage | Dataset Link |



For each dataset, please also explain what type of deformation is relevant and whether the dataset could realistically support research on deformation, spatial reasoning, localization, segmentation, or model robustness to anatomical changes.



Please prioritize CT datasets, exclude MRI, chest/thoracic, dermatology, and brain-tumor datasets, and focus on deformable anatomy with limited sample sizes and established benchmark usage.