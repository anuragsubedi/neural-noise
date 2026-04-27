# neural-noise

> **Controllable Music Generation via Latent Space Navigation in Diffusion Transformers**

**Course:** DSCI 498 — Deep and Generative AI
**Authors:** Anurag Subedi, Koushik Vennalakanti (Lehigh University)

---

## Overview

Modern generative audio models are **black boxes.** A user writes a text prompt, presses generate, and hopes the output matches their creative vision. There is no deterministic, fine-grained control over specific musical attributes (tempo, key, mood, instrumentation).

**neural-noise** builds an interactive Streamlit dashboard that exposes the internal control surface of a state-of-the-art Diffusion Transformer (DiT). By mapping human-interpretable musical parameters to the model's generation pipeline, we give creators real-time control over the audio synthesis process.

### The Journey (M1 → M2)

| Milestone                         | Architecture                | Representation          | Key Metric               |
| --------------------------------- | --------------------------- | ----------------------- | ------------------------ |
| **M1 — Discrete Baseline** | MicroMusicGPT (Transformer) | MIDI Tokens (388 vocab) | Perplexity: 6.3          |
| **M2 — Continuous Audio**  | ACE-Step 1.5 DiT + Qwen LM  | 48kHz Waveform Latents  | CLAP: 0.528 (top-5 mean) |

---

## Repository Structure

```
neural-noise/
│
├── m2_app/                          # ← Milestone 2: Streamlit Dashboard (primary)
│   ├── app.py                       # Streamlit entry point (multi-page)
│   ├── pages/
│   │   ├── 1_Generate.py            # Core generation UI
│   │   ├── 2_Gallery.py             # Playback gallery of pre-generated samples
│   │   ├── 3_Architecture.py        # Interactive pipeline architecture diagram
│   │   ├── 4_Metrics.py             # CLAP scores + acoustic descriptor evaluation
│   │   └── 5_MIDI_Baseline.py       # Milestone 1 discrete MIDI baseline (preserved)
│   ├── backend/
│   │   ├── ACE-Step-1.5/            # ACE-Step source (committed to Git)
│   │   │   ├── acestep/             # ACE-Step Python package
│   │   │   └── checkpoints/         # Model weights (download separately — see below)
│   │   ├── config.py                # PipelineConfig dataclass + env var mapping
│   │   ├── inference_engine.py      # Unified local/distributed inference wrapper
│   │   ├── lm_service.py            # FastAPI microservice — LM Composer (Windows)
│   │   ├── dit_service.py           # FastAPI microservice — DiT Renderer (Mac)
│   │   └── requirements.txt         # All deps (PyTorch, ACE-Step, Streamlit, etc.)
│   ├── components/                  # Waveform viz, chart renderer, preset manager
│   ├── utils/                       # Audio I/O, CLAP metrics helpers
│   ├── presets/genres.json          # Pre-built genre/mood/BPM configurations
│   ├── static/style.css             # Dark-theme custom CSS
│   └── output/                      # Pre-generated gallery samples (committed to Git)
│
├── m1_app/                          # ← Milestone 1: MicroMusicGPT (discrete baseline)
│   ├── app.py                       # Gradio dashboard
│   ├── train.py                     # Training loop
│   ├── generate.py                  # Autoregressive inference
│   ├── src/
│   │   ├── tokenizer.py             # Custom 388-token MIDI tokenizer
│   │   └── model.py                 # MicroMusicGPT Transformer architecture
│   ├── scripts/                     # Data prep, seed generation, Colab notebook builder
│   ├── MicroMusicGPT_Colab_ScaleUp.ipynb
│   └── requirements.txt             # M1 Gradio/MIDI dependencies
│
├── data/                            # M1 tokenized datasets + generated MIDI
├── 498_docs/                        # Project documentation, poster, etc
└── requirements_cloud.txt           # Lightweight deps for Streamlit Community Cloud
```

---

## Quick Start: Milestone 2 (Streamlit Dashboard)

### Option A — Read-Only / Gallery Mode (No GPU Required)

The `m2_app/output/` directory with pre-generated `.wav` files and `.json` metadata is committed to this repository. You can run the full dashboard immediately and browse all generated audio without any GPU, model download, or heavy dependencies.

**What you get:**
- **Gallery** — browse, play, and compare all pre-generated audio samples with waveform + spectrogram visualizations
- **Architecture** — interactive pipeline diagram with timing breakdown
- **Metrics** — static CLAP scores and acoustic descriptor charts
- **MIDI Baseline** — M1 perplexity training curves and comparison tables
- **Generate** — page loads but shows "Initialize Pipeline" prompt (skip in read-only mode)

```bash
# 1. Clone the repo
git clone <repo-url>
cd neural-noise

# 2. Create a lightweight virtual environment
python3 -m venv venv_gallery
source venv_gallery/bin/activate

# 3. Install lightweight frontend dependencies only
pip install streamlit plotly numpy scipy soundfile librosa Pillow python-dotenv

# 4. Launch
python3 -m streamlit run m2_app/app.py
# → http://localhost:8501
```

The Gallery loads all samples from `m2_app/output/` automatically.

---

### Option B — Full Pipeline (Local, Single Machine)

Runs the complete ACE-Step 1.5 generation pipeline on a single Apple Silicon Mac (~3 min/track with CPU offloading).

**Prerequisites:** macOS Apple Silicon (MPS), ~12 GB free disk, Python 3.10+

#### 1. Clone the Repository

```bash
git clone <repo-url>
cd neural-noise
```

#### 2. Create the Virtual Environment

> All backend dependencies live in `m2_app/backend/` — this is separate from the root-level M1 `venv`.

```bash
cd m2_app/backend
python3 -m venv acestep_env
source acestep_env/bin/activate
```

#### 3. Install Dependencies

> This installs from `m2_app/backend/requirements.txt` — **not** the root `requirements_cloud.txt`.

```bash
# (still inside m2_app/backend/, with acestep_env active)
pip install -r requirements.txt
```

#### 4. Download Model Checkpoints

The ACE-Step Python package (`acestep/`) is already committed to `m2_app/backend/ACE-Step-1.5/`. You only need to download the model weights:

```bash
cd ACE-Step-1.5

# DiT (Turbo 2B), VAE, and Text Embedder:
huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir checkpoints

# LM Planner (0.6B — fits in 6 GB VRAM):
huggingface-cli download ACE-Step/acestep-5Hz-lm-0.6B \
    --local-dir checkpoints/acestep-5Hz-lm-0.6B
```

After download, `m2_app/backend/ACE-Step-1.5/checkpoints/` should contain:

```
checkpoints/
├── acestep-v15-turbo/       # DiT weights (~4.8 GB)
├── acestep-5Hz-lm-0.6B/    # Qwen LM planner (~1.3 GB)
├── vae/                     # 1D Waveform VAE (~337 MB)
└── Qwen3-Embedding-0.6B/   # Text embedder (~1.2 GB)
```

#### 5. Launch the App

> **Always launch via `python3 -m streamlit`** — the bare `streamlit` binary inside the venv has a stale shebang path after venv creation.

```bash
# From the project root: neural-noise/
m2_app/backend/acestep_env/bin/python3 -m streamlit run m2_app/app.py
# → http://localhost:8501
```

Navigate to **Generate → select "local" → click "Initialize Pipeline"**.  
Initialization takes ~3–4 minutes on first run (model loading). Subsequent uses within the same session are instant.

---

### Option C — Distributed Mode (Two Machines, ~1 min/track)

Split the LM (Windows/CUDA) and DiT (Mac/MPS) across two machines on the same network. Both models stay permanently loaded — no offloading overhead.

**On Windows (LM Composer service):**

```bash
cd neural-noise/m2_app/backend
acestep_env\Scripts\python lm_service.py --device cuda --port 8001 --host 0.0.0.0
```

**On Mac (Streamlit + DiT Renderer):**

```bash
cd neural-noise
export NN_MODE=distributed
export NN_LM_URL=http://<WINDOWS_LOCAL_IP>:8001
m2_app/backend/acestep_env/bin/python3 -m streamlit run m2_app/app.py
```

Then in the dashboard: **Generate → select "distributed" → Initialize Pipeline**.

### Environment Variables

| Variable          | Default                   | Description                   |
| ----------------- | ------------------------- | ----------------------------- |
| `NN_MODE`       | `local`                 | `local` or `distributed`  |
| `NN_DIT_DEVICE` | `mps`                   | `mps`, `cuda`, or `cpu` |
| `NN_LM_DEVICE`  | `mps`                   | `mps`, `cuda`, or `cpu` |
| `NN_LM_URL`     | `http://localhost:8001` | Remote LM service URL         |
| `NN_DIT_URL`    | `http://localhost:8002` | Remote DiT service URL        |
| `NN_LM_MODEL`   | `acestep-5Hz-lm-0.6B`   | LM model variant              |
| `NN_DIT_MODEL`  | `acestep-v15-turbo`     | DiT model variant             |

---

## Architecture

```
User Input (Prompt, BPM, Key, Genre...)
            │
            ▼
  Qwen LM (0.6B) ── Chain-of-Thought ──► CoT Metadata + Audio Semantic Codes
            │
            ▼
  DiT (2B Turbo) ── 8-Step Diffusion ──► Rendered Latent [T, 64]
            │
            ▼
  1D VAE Decoder ──────────────────────► 48kHz Stereo Audio (.wav)
```

### Model Zoo (Selected Configuration)

| Component          | Model                    | Size   | Device                 | Reason                                        |
| ------------------ | ------------------------ | ------ | ---------------------- | --------------------------------------------- |
| **DiT**      | `acestep-v15-turbo`    | 4.8 GB | MPS (Mac)              | 4–8 steps vs 50 for base/SFT                 |
| **LM**       | `acestep-5Hz-lm-0.6B`  | 1.3 GB | CUDA (Win) / MPS (Mac) | Fits in 6 GB VRAM; adequate musical reasoning |
| **VAE**      | `vae` (1D Waveform)    | 337 MB | MPS (Mac)              | Decodes latent → 48kHz stereo                |
| **Embedder** | `Qwen3-Embedding-0.6B` | 1.2 GB | MPS (Mac)              | Text → DiT conditioning                      |

---

## Dashboard Pages

| Page                    | Description                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Home**          | Project overview, architecture diagram, quick-start links                                                          |
| **Generate**      | Core UI — genre presets, BPM, key, time signature, duration, lyrics, advanced diffusion controls                  |
| **Gallery**       | Browse and compare all pre-generated audio samples with waveform/spectrogram visualizations                        |
| **Architecture**  | Interactive pipeline diagram with timing breakdown                                                                 |
| **Metrics**       | CLAP text-audio similarity scores + acoustic descriptors (spectral centroid, bandwidth, ZCR, RMS, estimated tempo) |
| **MIDI Baseline** | Preserved M1 discrete MIDI generator for M1 → M2 comparison                                                       |

---

## Evaluation

### Primary Metric: CLAP Text-Audio Similarity

We use [LAION-CLAP (`laion/clap-htsat-unfused`)](https://huggingface.co/laion/clap-htsat-unfused) to jointly embed audio and the generation caption into a shared 512-d space and compute cosine similarity.

| Statistic                  | Value      |
| -------------------------- | ---------- |
| Mean CLAP                  | 0.465      |
| Std                        | 0.150      |
| Best                       | 0.628      |
| Top-5 Mean                 | 0.528      |
| Random / mismatch baseline | 0.05–0.20 |

A top-5 mean of 0.528 vs. a random baseline of ~0.125 confirms the model responds substantively to text prompts.

### Supporting Evidence: Acoustic Descriptors

Model-free spectral descriptors computed directly from the rendered waveform (no pre-trained scoring model): spectral centroid, spectral bandwidth, spectral rolloff (85%), zero-crossing rate, RMS energy, and estimated tempo.

> **Why not FAD?** With N = 12 samples, FAD is rank-deficient and reference-set sensitive. We report per-sample CLAP + acoustic descriptors instead — both well-defined at any N.

---

## Quick Start: Milestone 1 (MIDI Baseline)

The M1 Gradio app and training scripts use a separate, lighter dependency set.

```bash
# From: neural-noise/
python3 -m venv venv_m1
source venv_m1/bin/activate
pip install -r m1_app/requirements.txt

# 1. Extract Beethoven MIDI subset from MAESTRO v3.0:
python3 m1_app/scripts/prepare_data.py

# 2. Tokenize into train/val tensors:
python3 m1_app/src/tokenizer.py

# 3. Train locally (or open m1_app/MicroMusicGPT_Colab_ScaleUp.ipynb on A100):
python3 m1_app/train.py

# 4. Generate seed MIDI files:
python3 m1_app/scripts/generate_seeds.py

# 5. Launch the M1 Gradio dashboard:
python3 m1_app/app.py
# → http://127.0.0.1:7860
```

---

## Known Issues

| Issue                                                               | Impact                                                         | Status       |
| ------------------------------------------------------------------- | -------------------------------------------------------------- | ------------ |
| MLX DiT stream error on first call (`There is no Stream(gpu, 0)`) | ~10s extra latency, PyTorch fallback succeeds                  | Non-blocking |
| `pytorch_wavelets` DCW correction falls back to no-op             | Cosmetic warning only                                          | Non-blocking |