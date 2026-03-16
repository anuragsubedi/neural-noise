# Controllable Music Generation via Latent-Space Navigation in Diffusion Transformers

**Course:** DSCI 498 - Deep and Generative AI
**Team:** Anurag Subedi, Koushik Vennalakanti

## Project Overview

State-of-the-art music generation models like Suno and Udio produce impressive results but operate as "black-boxes" with no direct control over the internal mechanics of how sound is constructed. This prevents researchers and creators from understanding how musical attributes are encoded and how to precisely control generation beyond text prompts.

We aim to investigate ACE-Step 1.5's 64-dimensional latent space as a structured, navigable manifold for music generation. Our primary mission is exploring if we can exploit meaningful directions corresponding to interpretable musical attributes (genre, mood, key) for fine-grained control beyond what text prompting alone allows.

## Milestone 1 Deliverable: MicroMusicGPT Baseline

Before navigating the continuous Continuous Diffusion Transformers (DiTs), we established **MicroMusicGPT**: a mechanistic PyTorch Transformer baseline. Unlike ACE-Step which operates on continuous VAE compressed audio, this model operates on discrete event-based MIDI tokens, helping us establish our quantitative baseline mechanics.

### Features

- **Tiny Symphony Dataset:** Trained exclusively on Beethoven MAESTRO recordings.
- **Custom Tokenizer:** A 388-integer vocabulary capturing simultaneous polyphonic key strikes (`NOTE_ON`, `NOTE_OFF`) and time intervals (`TIME_SHIFT`) directly resolving rhythm.
- **MicroMusic Architecture:** 256-context window Transformer built specifically for M2 and RTX3060 local hardware execution.

## Repository Structure

```text
neural-noise/
├── data/
│   ├── raw_midi/                   # Processed Beethoven MAESTRO MIDI files
│   ├── seeds/                      # Seed chunks (manual chords and real excerpts)
│   ├── generated/                  # Generated .mid and synthesized .wav files
│   ├── dataset_train.pt            # Tokenized sequence tensor (Train split)
│   └── dataset_val.pt              # Tokenized sequence tensor (Validation split)
├── src/
│   ├── tokenizer.py                # Custom Event-Based MIDITokenizer implementation
│   └── model.py                    # MicroMusicGPT (Transformer) architecture class
├── checkpoints/                    # Serialized model weights (.pth)
├── scripts/
│   ├── prepare_data.py             # Parses CSV and extracts Beethoven MIDI subset
│   ├── test_tokenizer.py           # Integrity script for round-trip token reconstruction
│   ├── generate_seeds.py           # Dynamically constructs Contextual Seed MIDI files
│   ├── build_colab_notebook.py     # Compiles isolated .ipynb code for heavy A100/H100 runs
│   └── convert_colab_midi.py       # Standardizes and synthesizes .mid from Colab runs
├── train.py                        # Training optimization loop with Perplexity validation
├── generate.py                     # Autoregressive Top-K inference generator 
└── app.py                          # Gradio Dashboard with Context Seed injection & Playback Gallery
```

## Quick Start: Step-by-Step Reproduction

This pipeline is built to be run step-by-step from scratch in a fresh environment.

**1. Dependency Installation**
A `requirements.txt` is provided. We recommend using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Data Triage ("Tiny Symphony")**
Ensure the `maestro-v3.0.0` folder and `maestro-v3.0.0.csv` are in the project root. Then extract the Beethoven subset:

```bash
python scripts/prepare_data.py
```

**3. Tokenization & Validation Splits**
Build the integer token datasets. This strictly separates 90% into `dataset_train.pt` and 10% into `dataset_val.pt`:

```bash
python src/tokenizer.py
```

*(Optional) Run the Tokenizer Integrity Check to verify bidirectional encoding fidelity without bleed:*

```bash
python scripts/test_tokenizer.py
```

**4. Model Training (Local or Scaled)**
You can train the baseline model locally on Apple Silicon / standard GPUs:

```bash
python train.py
```

*(Optional) For massive scaling (1024-context, 768-embedding), compile a standalone Colab notebook and run the tensors on A100/H100 hardware:*

```bash
python scripts/build_colab_notebook.py
```

Instead of building the colab notebook from scratch, you can just review and execute the `MicroMusicGPT_Colab_ScaleUp.ipynb` file we created for the Milestone 1.

**5. Generative Prompt Seeding**
Generate the baseline chord progressions and sequence excerpts to prompt/seed the model inference:

```bash
python scripts/generate_seeds.py
```

**6. Inference & Interface Interaction**
Once weights are saved in `checkpoints/micromusicgpt_v1_final.pth`, launch the Gradio synthesis dashboard:

```bash
python app.py
```

The interface will load at `http://127.0.0.1:7860/` where you can input the seeds generated in Step 5 and synthesize the output tracks dynamically.
