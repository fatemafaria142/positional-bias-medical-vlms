# VoxTell model components and the experiments that use them

Experiments: Replacing the Organ with Noise, Activation Patching (Switching Layer Updates to the Opposite Side), Cross-Attention Analysis, Prompt Word Removal, Word Attribution (Exact Shapley, SHAP, LIME). An empty last cell means none of the five uses that component.

| Position | Component | Code name | What it does | Used in which experiment |
|---|---|---|---|---|
| **Input (image)** | Image encoder | `encoder` | 3D convolutional network. Turns the scan into image features. It never sees the text. | Replacing the Organ with Noise |
| **Input (image)** | Image feature projection | `project_bottleneck_embed` | Linear layers. Turn the encoder features into the image tokens that the transformer decoder looks at. | Cross-Attention Analysis |
| **Input (text)** | Text projection | `project_text_embed` | Linear layers. Turn the text embedding into the starting query state of the transformer decoder. | Prompt Word Removal, Word Attribution (Exact Shapley, SHAP, LIME) |
| **Internal (middle)** | Transformer decoder | `transformer_decoder` | Six identical layers. The text query looks at the image tokens and updates itself. | Activation Patching (Switching Layer Updates to the Opposite Side), Cross-Attention Analysis |
| | ↳ Layer 1 | `transformer_decoder.layers.0` | Cross-attention, then feed-forward. | Activation Patching (Switching Layer Updates to the Opposite Side), Cross-Attention Analysis |
| | ↳ Layer 2 | `transformer_decoder.layers.1` | Cross-attention, then feed-forward. | Activation Patching (Switching Layer Updates to the Opposite Side), Cross-Attention Analysis |
| | ↳ Layer 3 | `transformer_decoder.layers.2` | Cross-attention, then feed-forward. | Activation Patching (Switching Layer Updates to the Opposite Side), Cross-Attention Analysis |
| | ↳ Layer 4 | `transformer_decoder.layers.3` | Cross-attention, then feed-forward. | Activation Patching (Switching Layer Updates to the Opposite Side), Cross-Attention Analysis |
| | ↳ Layer 5 | `transformer_decoder.layers.4` | Cross-attention, then feed-forward. | Activation Patching (Switching Layer Updates to the Opposite Side), Cross-Attention Analysis |
| | ↳ Layer 6 | `transformer_decoder.layers.5` | Cross-attention, then feed-forward. | Activation Patching (Switching Layer Updates to the Opposite Side), Cross-Attention Analysis |
| | ↳ Final normalization | `transformer_decoder.norm` | Turns the state after layer 6 into the mask embedding. | |
| **Internal to output** | Decoder channel projections | `project_to_decoder_channels` | Convert the mask embedding into the channel sizes of the convolutional decoder. | |
| **Output** | Convolutional decoder | `decoder` | Builds the 3D mask from the mask embedding and the encoder's skip features. | Replacing the Organ with Noise, Activation Patching (Switching Layer Updates to the Opposite Side), Prompt Word Removal, Word Attribution (Exact Shapley, SHAP, LIME) |

**Notes**
- The figures number the layers 1 to 6, and the code counts them 0 to 5. Figure "Layer 1" is `transformer_decoder.layers.0`.
- In Activation Patching, the text embedding is left untouched, so the text projection is not marked for it.
- LIME, SHAP and Shapley (Word Attribution) never touch a layer. They change the prompt words and read the score of the mask.
- "Used" means the visualization changes that component's input or output, or reads values from it. Every component runs in every experiment, because VoxTell needs the whole model to make a mask.
- Roles come from `scripts/voxtell_mechanistic.py`. The layer subsets in Activation Patching come from `figure_B` in `scripts/make_row14_figures.py`.
