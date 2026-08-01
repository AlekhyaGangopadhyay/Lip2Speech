# Local Image Generation Integration Plan

We will add a local Hugging Face text-to-image generation model to the Lip2Speech pipeline. This will allow the system to generate a visual depiction of the predicted corrected text (e.g. converting "place blue in at g two now" into an image of a blue object being placed in a grid).

---

## Model Recommendations

For running text-to-image locally on consumer hardware, we must balance image quality, inference speed (latency), and RAM/VRAM usage. Below is a comparison of the best free, open-source models available on Hugging Face:

| Model | Hugging Face Path | Size (GB) | Steps Required | Speed (CPU/GPU) | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SD Turbo** | `stabilityai/sd-turbo` | ~2.0 GB | 1 step | **Extremely Fast** | **Recommended** (Best for CPU/low-end GPU) |
| **Stable Diffusion 1.5** | `runwayml/stable-diffusion-v1-5` | ~4.0 GB | 20–50 steps | Moderate | Good fallback for richer/more stylized images |
| **Latent Consistency Model (LCM)** | `latent-consistency/lcm-sd15` | ~3.4 GB | 4 steps | Fast | Alternative for fast 4-step generation |

### Why SD-Turbo?
- **Speed**: It is distilled using Adversarial Diffusion Distillation (ADD), meaning it generates quality images in just **1-4 steps** instead of 25-50 steps. This is a game-changer for running on CPU or laptops.
- **Resource Footprint**: At ~2.0 GB, it fits easily into memory alongside the LipReading model and T5 corrector.

---

## Architectural Options for Prompt Expansion

The predictions from the GRID corpus are very short and structured commands (e.g. *"set blue at f two now"*). If we feed this directly to a text-to-image model, the result will be abstract or poor. We have three design options to handle prompt expansion:

> [!TIP]
> **Option A: Rule-Based / Template-Based Expansion (Recommended)**
> Parse the GRID command structured pattern and map it to a descriptive visual prompt (e.g. `"A clean high-quality studio photo of a hand placing a blue ceramic mug at grid position F2 on a white table"`).
> * **Pros**: Zero overhead, fast, highly accurate, requires 0 extra memory.
> * **Cons**: Specific to the GRID corpus pattern.

> [!NOTE]
> **Option B: Tiny Local LLM (Llama-3.2-1B-Instruct or TinyLlama)**
> Load a small local LLM to rewrite the corrected text into a detailed image prompt.
> * **Pros**: Highly flexible, handles natural language inputs if the vocabulary expands.
> * **Cons**: Adds ~2-3 GB of extra RAM/VRAM overhead and 1-2 seconds of text-generation latency.

> [!WARNING]
> **Option C: Direct Input to Text-to-Image**
> Feed the corrected text directly to the model.
> * **Pros**: Simple, zero extra code/models.
> * **Cons**: Generates lower quality/abstract images because Stable Diffusion is trained on detailed visual descriptions.

---

## User Review Required

Please review the architectural options and verify if your system has:
1. **CUDA (NVIDIA GPU)**: Diffusers can run on CPU, but GPU acceleration makes image generation take ~1 second instead of ~30-60 seconds.
2. **Library Installation**: We will need to install the Hugging Face `diffusers` library along with `accelerate`.

---

## Proposed Changes

### Dependencies

#### [MODIFY] [requirements.txt](file:///d:/Lip2Speech_Final/lip2speech/requirements.txt)
Add required packages:
```txt
diffusers>=0.20.0
accelerate>=0.20.0
```

---

### Core Image Generation

#### [NEW] [image_generator.py](file:///d:/Lip2Speech_Final/lip2speech/image_generator.py)
Create a helper module containing:
1. **`generate_image_from_text(text, output_path)`**: Loads the local Hugging Face `stabilityai/sd-turbo` (or fallback) using `diffusers.AutoPipelineForText2Image` and saves the generated image.
2. **`expand_prompt(text)`**: Expands structured GRID commands into rich visual descriptions using templates/rules.

---

### Pipeline Integration

#### [MODIFY] [inference.py](file:///d:/Lip2Speech_Final/lip2speech/inference.py)
Add a function `generate_image_prediction(text, output_path)` that coordinates the prompt expansion and image generation.

#### [MODIFY] [test_video.py](file:///d:/Lip2Speech_Final/lip2speech/test_video.py)
Add **Stage 4** to the CLI and orchestrator:
- `--stage 4`: Full pipeline including image generation.
- `--image-output`: Custom path to save the generated image (defaults to `<video_name>_output.png`).

---

## Verification Plan

### Automated Tests
We will verify by running the pipeline end-to-end:
```powershell
python test_video.py lbad8p.mpg --stage 4
```
This will:
1. Extract lips and decode text.
2. Refine text using the local T5 model.
3. Generate speech audio (`lbad8p_output.wav`).
4. Generate the corresponding visual image (`lbad8p_output.png`).

### Manual Verification
- Open the generated image file to inspect the quality and verify it matches the predicted text command.
