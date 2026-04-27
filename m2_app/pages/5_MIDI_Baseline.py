"""
neural-noise — M1 MIDI Baseline Page
======================================

Standalone tab preserving the Milestone 1 discrete MIDI generator.
Loads the MicroMusicGPT model for comparison with M2 continuous audio.
"""

import streamlit as st
import os
import sys
from pathlib import Path

st.set_page_config(page_title="MIDI Baseline | neural-noise", page_icon="🎹", layout="wide")

_APP_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _APP_ROOT.parent
css_path = _APP_ROOT / "static" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("# Milestone 1 — Discrete MIDI Baseline")
st.markdown("### MicroMusicGPT: Autoregressive Transformer for Symbolic Music Generation")
st.markdown("---")

st.markdown("""
## About This Baseline

This page preserves the **Milestone 1** discrete music generation system for comparison
with the continuous audio pipeline in Milestone 2.

### Key Specifications

| Attribute | Value |
|-----------|-------|
| **Architecture** | Autoregressive Transformer (GPT-style) |
| **Vocabulary** | 388 tokens (NOTE_ON, NOTE_OFF, TIME_SHIFT) |
| **Block Size** | 1024 tokens |
| **Embedding Dim** | 768 |
| **Training Data** | MAESTRO v3.0 — Beethoven piano recordings |
| **Final Perplexity** | 6.3 (from ~516 initial) |
| **Output Format** | Standard MIDI (.mid) |

### The "Infinite Sustain" Problem

Early training produced a model that generated NOTE_ON events but "forgot" to generate
corresponding NOTE_OFF events — resulting in notes that sustained infinitely, creating
a cacophonous wall of sound. This was fixed by rebalancing the token vocabulary and
adding explicit temporal structure via TIME_SHIFT tokens (10ms resolution).

### Contextual Seeding

We inject the first 200 tokens of a real Beethoven excerpt into the context window
to anchor the model's generation in a specific key, tempo, and style. This "seed prompting"
technique provides deterministic control over the starting musical context.
""")

st.markdown("---")

# ---------------------------------------------------------------------------
# Interactive Generator (if M1 model is available)
# ---------------------------------------------------------------------------
st.markdown("## Generate MIDI")

checkpoint_path = _PROJECT_ROOT / "checkpoints" / "micromusicgpt_v1_final.pth"
has_model = checkpoint_path.exists()

if not has_model:
    st.warning(
        f"MicroMusicGPT checkpoint not found at `{checkpoint_path}`. "
        "The M1 model was trained on Google Colab. Place the checkpoint in "
        "`neural-noise/checkpoints/` to enable interactive generation."
    )
    st.markdown("### Pre-Generated Samples")
    st.markdown("Browse pre-generated MIDI samples from the M1 pipeline:")

    generated_dir = _PROJECT_ROOT / "data" / "generated"
    if generated_dir.exists():
        midi_files = sorted(generated_dir.glob("*.mid*"), reverse=True)
        if midi_files:
            for f in midi_files[:5]:
                wav_path = f.with_suffix(".wav")
                st.markdown(f"**{f.name}**")
                if wav_path.exists():
                    st.audio(str(wav_path), format="audio/wav")
                else:
                    st.caption("(WAV not synthesized — run the M1 app to generate audio)")
        else:
            st.caption("No pre-generated samples found.")
    else:
        st.caption("No generated directory found.")
else:
    st.info("M1 model found! Interactive MIDI generation available.")
    # In a full implementation, we'd load the model and provide the generation interface.
    # For now, we direct users to the original Gradio app.
    st.markdown(
        "For the full interactive M1 experience, run the original Gradio app:\n"
        "```bash\ncd neural-noise && python3 app.py\n```"
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Training Metrics
# ---------------------------------------------------------------------------
st.markdown("## Training Metrics")

training_img = _PROJECT_ROOT / "498_docs" / "training_metrics.png"
if training_img.exists():
    st.image(str(training_img), caption="Perplexity decay during MicroMusicGPT training (516 → 6.3)")
else:
    st.caption("Training metrics plot not found.")

st.markdown("""
### Perplexity Interpretation

| Perplexity | Meaning |
|------------|---------|
| ~516 | Random noise — model has no musical knowledge |
| ~50 | Learning basic note patterns |
| ~15 | Understanding harmonic structure |
| **6.3** | **Strong musical coherence — near-human predictability** |
""")

st.markdown("---")
st.markdown("## M1 vs M2 Comparison")
st.markdown("""
| Aspect | M1 (Discrete) | M2 (Continuous) |
|--------|---------------|-----------------|
| Representation | MIDI tokens (symbolic) | 48kHz waveform (acoustic) |
| Model | Autoregressive Transformer | Diffusion Transformer + LM |
| Control | Seed prompting (200 tokens) | Interactive params (BPM, Key, Genre) |
| Output | .mid → synthesized .wav | Native .wav (48kHz stereo) |
| Quality Metric | Perplexity: 6.3 | FAD / CLAP (TBD) |
| Instruments | Piano only (MAESTRO) | Any instrument (generative) |
| Training | Custom (A100 Colab) | Pre-trained (ACE-Step 1.5) |
""")
