# 🗣️ Lip2Speech: Spatio-Temporal Lip Reading & Speech Synthesis for Assistive Communication

> **An offline, edge-deployable visual-to-speech framework** that converts silent lip movements into natural speech — designed for individuals with speech impairments (e.g., ALS patients) who can still mouth words.

<p align="center">
  <img src="static/system_architecture.png" alt="Lip2Speech System Architecture" width="85%"/>
</p>

---

## ✨ Highlights

| Feature | Detail |
|:---|:---|
| **Decoupled Pipeline** | Video → Visual Classification (Text) → LLM Refinement → Speech Synthesis |
| **Edge-Optimized** | Runs entirely offline on standard CPUs in **< 2 seconds** end-to-end latency |
| **Compiler-Independent** | Uses OpenCV Haar Cascades instead of `dlib`/CMake — zero C++ compilation required |
| **95.4% Word Accuracy** | After T5 LLM post-processing (up from 87.23% raw visual prediction) |
| **Privacy-First** | All processing is local — no cloud, no data transmission |

---

## 🏗️ Architecture

The framework solves the visual-to-speech problem through a **three-stage decoupled pipeline**, deliberately avoiding direct video-to-waveform generation (which demands massive GPU resources and exhibits high speech distortion):

```
Silent Video → [Mouth ROI Extraction] → [MobileNetV2 Spatial Features] → [BiLSTM Temporal Decoder]
                                                                                    ↓
                                                                            Raw Word Sequence
                                                                                    ↓
                                                                    [T5 LLM Grammar & Spelling Correction]
                                                                                    ↓
                                                                            Refined Sentence
                                                                                    ↓
                                                                    [Offline TTS Engine (pyttsx3)]
                                                                                    ↓
                                                                            🔊 Synthesized Speech
```

### Why Decouple?

A critical challenge in lip reading is **homophenes** — visually identical mouth shapes representing different words (e.g., "p", "b", "m"). Direct visual-to-audio models must resolve these ambiguities in pixel space, which is extremely difficult. Our decoupled design resolves homophenes at the *linguistic* level via the T5 model, leveraging robust language priors to disambiguate.

### Stage Details

| Stage | Component | Purpose |
|:---:|:---|:---|
| 1 | **MobileNetV2** (partially frozen, blocks 14–18 fine-tuned) + **2-layer BiLSTM** | Extracts spatial features from mouth crops and models temporal co-articulation across 25 frames |
| 2 | **T5 Transformer** (local, offline) | Corrects spelling/grammar errors in the raw predicted word sequence |
| 3 | **pyttsx3 TTS Engine** | Converts the refined text to audible speech entirely offline |

---

## 📊 Results

Benchmarked on the GRID Corpus validation split:

| Metric | Raw Model | After T5 Refinement |
|:---|:---:|:---:|
| **Word Accuracy Rate (WAR)** | 87.23% | **95.4%** |
| **Character Error Rate (CER)** | 6.1 | **2.3** |
| **Val Loss** | 0.2945 | N/A |
| **End-to-End Latency** | 1.8 s | **2.0 s** |

<p align="center">
  <img src="static/loss_accuracy_curves.png" alt="Training Loss and Accuracy Curves" width="48%"/>
  &nbsp;&nbsp;
  <img src="static/t5_enhancement_chart.png" alt="T5 Enhancement Impact" width="48%"/>
</p>

### Ablation Study

| Configuration | Training Loss | Epoch Latency | Edge Suitability |
|:---|:---:|:---:|:---|
| **Baseline (Selected)** — Partially Frozen MobileNetV2 (Blocks 14+), 25 frames | 1.8314 | 2.01 s | ✅ Optimal |
| Ablation A — Fully Unfrozen MobileNetV2, 25 frames | 1.7658 | 2.50 s | ❌ Marginal gain, too heavy |
| Ablation B — Partially Frozen, 10 frames | 2.8919 | 0.83 s | ❌ Fast but high loss |

<p align="center">
  <img src="static/ablation_study_chart.png" alt="Ablation Study" width="60%"/>
</p>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Trained model weights: `lipreading_model.pth` (place in project root)
- OpenCV Haar Cascade (bundled with `opencv-python`)

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/AlekhyaGangopadhyay/Lip2Speech_ReVox.git
cd Lip2Speech_ReVox

# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Inference

Run the CLI test script on a sample video:

```bash
python test_video.py lbad8p.mpg
```

To run only through a specific stage:

```bash
python test_video.py lbad8p.mpg --stage 1   # raw text only
python test_video.py lbad8p.mpg --stage 2   # + T5 refinement
python test_video.py lbad8p.mpg --stage 3   # + audio synthesis
```

Or use the inference module directly:

```bash
python inference.py --video pgby4n.mpg
```

### 3. Sample Videos

Two sample videos from the GRID Corpus are included for testing:

| File | Description |
|:---|:---|
| `lbad8p.mpg` | Sample GRID utterance |
| `pgby4n.mpg` | Sample GRID utterance |

---

## 📁 Repository Structure

```
lip2speech/
├── inference.py                # Core inference pipeline (ROI extraction → prediction → T5 → TTS)
├── test_video.py               # CLI test harness for running inference on video files
├── lipreading_model.pth        # Trained MobileNetV2-BiLSTM weights
├── requirements.txt            # Python dependencies
├── lip2speech.ipynb             # Training notebook (data loading, training loops, evaluation, ablation study)
├── docs/
│   └── RESEARCH_PAPER.md       # Full IEEE-format research paper (markdown)
├── static/
│   ├── system_architecture.png # Architecture diagram
│   ├── loss_accuracy_curves.png
│   ├── ablation_study_chart.png
│   ├── t5_enhancement_chart.png
│   ├── novelty_flowchart.png
│   └── performance_dashboard.png
├── lbad8p.mpg                  # Sample test video
├── pgby4n.mpg                  # Sample test video
└── Lip2Speech-Research Paper(5).pdf  # Research paper (PDF)
```

---

## 🔬 Research Context

### Dataset

This system is trained and evaluated on the **GRID Corpus** — a structured audiovisual dataset containing 33,000 video-transcript pairs from 34 speakers with a 51-word vocabulary in a fixed command grammar (e.g., *"lay white by j four again"*). Each clip is ~3 seconds at 25 fps; 25 representative frames are sampled per video.

> **Note:** The GRID Corpus provides a controlled environment to validate the architectural design. The constrained vocabulary makes this system ideal as a **proof-of-concept for high-frequency assistive commands** (e.g., medical assistance requests, basic control interfaces). Scaling to unconstrained vocabularies via LRS2/LRS3 datasets is a planned future direction.

### Key Design Decisions

1. **Frame-Wise Classification over CTC Loss:** Instead of using Connectionist Temporal Classification (which introduces training instability and compute overhead), we use simplified frame-wise word alignment padded with silence tokens. The T5 post-processor recovers semantic integrity without requiring heavy decoding algorithms.

2. **Standardized Assistive Voice:** The current TTS implementation uses `pyttsx3` with a standardized voice, prioritizing **intelligibility and latency** over speaker-specific characteristics — consistent with traditional assistive device design.

3. **Haar Cascades over dlib:** By replacing `dlib` (which requires CMake/C++ compilation) with OpenCV Haar Cascade classifiers, the system achieves maximum portability across operating systems and low-power hardware with zero compiler dependencies.

### Comparison with Existing Methods

| Method | Limitation | Lip2Speech Advantage |
|:---|:---|:---|
| **LipNet** (CTC-based) | Heavy compute, compile dependencies (dlib/CMake), higher WER for short words | Lightweight MobileNetV2 + BiLSTM, compiler-free, T5 refinement |
| **AV-HuBERT** (Transformer) | Large model, GPU-intensive, unsuitable for offline edge | CPU inference ~2s, fully offline |
| **Cloud Speech APIs** | Requires internet, transmits sensitive data | 100% local, privacy-guaranteed |
| **Traditional TTS Keyboards** | Requires manual typing, slow | Direct visual-to-speech, natural flow |

---

## 🔮 Future Work

- **Continuous Real-Time Streaming** — Sliding-window lip detection on live webcam feeds with on-the-fly speech synthesis
- **Unconstrained Vocabulary** — Scaling to open-vocabulary lip reading using LRS2/LRS3 datasets
- **AV-HuBERT Upgrade** — Integrating AV-HuBERT with lightweight cross-attention fusion adapters for speaker-independent generation
- **Mobile & Wearable Deployment** — Compiling models for edge NPUs and smart AR/VR glasses
- **Speaker-Specific Voice Cloning** — Zero-shot TTS models for personalized voice output

---

## 📄 Publication & Documentation

📑 **Full IEEE-Format Paper (Markdown):** [`docs/RESEARCH_PAPER.md`](docs/RESEARCH_PAPER.md) — Complete research paper with abstract, methodology, mathematical formulations, results, ablation studies, comparisons, and all references.

📕 **Research Paper (PDF):** [`Lip2Speech-Research Paper(5).pdf`](Lip2Speech-Research%20Paper(5).pdf)

### Target Venues

| Category | Venue |
|:---|:---|
| **Assistive Technology** | ACM ASSETS, IEEE TNSRE |
| **Speech & Language Processing** | Interspeech, ICASSP |
| **Mobile & Edge Computing** | ACM MobileHCI, MobiSys Workshops |

---

## 👥 Authors

| Author | Affiliation | Contact |
|:---|:---|:---|
| **Ipsita Das** | Dept. of CSE (AIML), IEM Kolkata, School of UEMK | ipsitadas2401@gmail.com |
| **Alekhya Gangopadhyay** | Dept. of CSE (AIML), IEM Kolkata, School of UEMK | iamalekhya7@gmail.com |

---

## 📚 Key References

1. Cooke et al., "An audio-visual corpus for speech perception and automatic speech recognition," *JASA*, 2006.
2. Kim et al., "Lip to speech synthesis with visual context attentional GAN," *NeurIPS*, 2021.
3. Assael et al., "LipNet: End-to-end sentence-level lipreading," *arXiv:1611.01599*, 2016.
4. Shi et al., "Robust self-supervised audio-visual speech recognition," *arXiv:2201.01763*, 2022.
5. Lu & Li, "Research on lip recognition algorithm based on MobileNet+Attention-GRU," *MBE*, 2022.

> Full reference list available in the [research paper](Lip2Speech-Research%20Paper(5).pdf).

---

<p align="center">
  <i>Built with the mission to restore voice and dignity to those who have lost the ability to speak.</i>
</p>
