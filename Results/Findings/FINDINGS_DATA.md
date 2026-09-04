### LGG: target is "the tumor"

| Model | Output | Detect Rate | Wrong-Side Rate (P1-P4) | No-Shift Rate (P1-P4) | Mean Norm. Distance | Mean IoU |
|---|---|---|---|---|---|---|
| Molmo2-8B | point | 90.4% | 7.1% | 45.6% | 0.109 | N/A |
| Moondream2 (point) | point | 100.0% | 8.0% | 34.0% | 0.141 | N/A |
| NorthMicroVision | bbox | 100.0% | 17.5% | 22.0% | 0.113 | N/A |
| LocateAnything-3B | bbox | 95.6% | 22.4% | 50.0% | 0.094 | N/A |
| Moondream2 (detect) | bbox | 96.8% | 42.7% | 85.4% | 0.099 | 0.184 |
| Kosmos-2 | bbox | 100.0% | 49.0% | 71.5% | 0.129 | N/A |
| CLIPSeg | mask | 56.4% | 49.0% | 56.9% | 0.085 | N/A |
| Grounding DINO | bbox | 100.0% | 49.5% | 99.5% | 0.115 | N/A |
| OWLv2 | bbox | 100.0% | 50.0% | 72.5% | 0.112 | 0.170 |
| Qwen2.5-VL-7B | bbox | 99.2% | 50.0% | 66.7% | 0.398 | 0.000 |
| Florence-2 | bbox | 100.0% | 50.5% | 99.0% | 0.115 | N/A |
| PaliGemma 2 | mask | 100.0% | 50.5% | 99.5% | 0.114 | N/A |

### KVASIR: target is "the polyp"

| Model | Output | Detect Rate | Wrong-Side Rate (P1-P4) | No-Shift Rate (P1-P4) | Mean Norm. Distance | Mean IoU |
|---|---|---|---|---|---|---|
| Moondream2 (point) | point | 100.0% | 5.5% | 29.5% | 0.182 | N/A |
| Moondream2 (detect) | bbox | 12.0% | 11.5% | 42.3% | 0.094 | 0.555 |
| Molmo2-8B | point | 99.6% | 18.1% | 55.3% | 0.107 | N/A |
| NorthMicroVision | bbox | 98.0% | 21.5% | 37.4% | 0.124 | N/A |
| Grounding DINO | bbox | 95.6% | 31.2% | 52.9% | 0.146 | N/A |
| LocateAnything-3B | bbox | 99.2% | 34.7% | 60.8% | 0.077 | N/A |
| Kosmos-2 | bbox | 100.0% | 48.0% | 86.5% | 0.127 | N/A |
| CLIPSeg | mask | 93.2% | 48.1% | 59.6% | 0.112 | N/A |
| Qwen2.5-VL-7B | bbox | 99.6% | 49.0% | 70.5% | 0.453 | 0.003 |
| PaliGemma 2 | mask | 100.0% | 50.0% | 100.0% | 0.125 | N/A |
| OWLv2 | bbox | 28.0% | 53.2% | 85.1% | 0.088 | 0.564 |
| Florence-2 | bbox | 54.4% | 53.5% | 100.0% | 0.124 | N/A |

### ISIC: target is "the lesion"

| Model | Output | Detect Rate | Wrong-Side Rate (P1-P4) | No-Shift Rate (P1-P4) | Mean Norm. Distance | Mean IoU |
|---|---|---|---|---|---|---|
| Molmo2-8B | point | 100.0% | 5.0% | 18.5% | 0.159 | N/A |
| Moondream2 (point) | point | 100.0% | 8.5% | 24.5% | 0.174 | N/A |
| NorthMicroVision | bbox | 100.0% | 17.5% | 36.5% | 0.093 | N/A |
| Moondream2 (detect) | bbox | 12.8% | 21.7% | 65.2% | 0.041 | 0.773 |
| LocateAnything-3B | bbox | 100.0% | 35.5% | 66.0% | 0.065 | N/A |
| Grounding DINO | bbox | 100.0% | 42.0% | 75.5% | 0.054 | N/A |
| Kosmos-2 | bbox | 100.0% | 48.5% | 89.5% | 0.067 | N/A |
| OWLv2 | bbox | 69.6% | 50.0% | 93.8% | 0.033 | 0.735 |
| PaliGemma 2 | mask | 100.0% | 50.5% | 95.5% | 0.030 | N/A |
| CLIPSeg | mask | 94.0% | 50.8% | 77.8% | 0.043 | N/A |
| Florence-2 | bbox | 100.0% | 51.0% | 94.0% | 0.039 | N/A |
| Qwen2.5-VL-7B | bbox | 92.4% | 52.2% | 69.0% | 0.380 | 0.012 |
