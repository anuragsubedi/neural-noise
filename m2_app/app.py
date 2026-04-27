"""
neural-noise — Controllable Music Generation via Latent Space Navigation
=========================================================================

Milestone 2: Continuous Audio Synthesis with ACE-Step 1.5 DiT

Main Streamlit entry point. Configures the multi-page app, sidebar,
and global state (inference engine, preset manager).

Run with:
    streamlit run m2_app/app.py
"""

import streamlit as st
from pathlib import Path
import os

_DEPLOY_MODE = os.environ.get("DEPLOY_MODE", "")

# ---------------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="neural-noise | Controllable Music Generation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load Custom CSS
# ---------------------------------------------------------------------------
css_path = Path(__file__).parent / "static" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# neural-noise")
    st.markdown("*Controllable Music Generation*")
    st.markdown("---")

    st.markdown("### Pipeline Status" if _DEPLOY_MODE != "cloud" else "### Demo Mode")

    if _DEPLOY_MODE == "cloud":
        st.markdown(
            '<span style="background:#7c3aed;color:white;padding:3px 10px;'
            'border-radius:12px;font-size:0.8rem;">☁ Hosted Demo</span>',
            unsafe_allow_html=True,
        )
        st.caption("Gallery, Metrics & Architecture pages are fully available. "
                   "Generation requires a local GPU — see the README.")
    elif "engine_initialized" in st.session_state and st.session_state.engine_initialized:
        mode = st.session_state.get("engine_mode", "local")
        st.markdown(f'<span class="status-online">● Engine Online</span> ({mode})',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-offline">● Engine Offline</span>',
                    unsafe_allow_html=True)
        st.caption("Navigate to the Generate page to initialize.")


    st.markdown("---")

    # Generation counter
    gen_count = st.session_state.get("generation_count", 0)
    st.metric("Generations This Session", gen_count)

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "DSCI 498 — Deep and Generative AI  \n"
        "**Architecture:** ACE-Step 1.5  \n"
        "**DiT:** acestep-v15-turbo (2B)  \n"
        "**LM:** Qwen3 (0.6B)  \n"
        "**VAE:** 1D Waveform (48kHz)"
    )
    st.markdown("---")
    st.caption("Built by Anurag Subedi")


# ---------------------------------------------------------------------------
# Home Page Content
# ---------------------------------------------------------------------------

st.markdown("# neural-noise")
st.markdown("### Controllable Music Generation via Latent Space Navigation in Diffusion Transformers")

st.markdown("---")

# Hero section
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    ## The Problem

    Modern generative audio models are **black boxes**. Users type a text prompt, press generate,
    and hope the output matches their creative vision. There is no deterministic control over
    specific musical attributes — tempo, key, mood, and instrumentation are entangled inside
    opaque neural networks.

    ## Our Solution

    **neural-noise** provides an interactive interface that exposes the internal control surfaces
    of a state-of-the-art Diffusion Transformer (DiT). By mapping human-interpretable musical
    parameters to the model's generation pipeline, we give creators **fine-grained, real-time
    control** over the audio synthesis process.

    ## The Journey

    | Milestone | Architecture | Representation | Key Metric |
    |-----------|-------------|----------------|------------|
    | **M1 — Discrete Baseline** | MicroMusicGPT (Transformer) | MIDI Tokens (388 vocab) | Perplexity: 6.3 |
    | **M2 — Continuous Audio** | ACE-Step 1.5 DiT + Qwen LM | 48kHz Waveform Latents | FAD / CLAP |
    """)

with col2:
    st.markdown("### Architecture Overview")
    st.markdown("""
    ```
    ┌─────────────────────────┐
    │   User Input (Prompt,   │
    │   BPM, Key, Genre...)   │
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │   Qwen LM (0.6B)       │
    │   Chain-of-Thought      │
    │   → CoT Metadata        │
    │   → Audio Semantic Codes│
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │   DiT (2B Turbo)        │
    │   8-Step Diffusion      │
    │   → Latent Rendering    │
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │   1D VAE Decoder        │
    │   → 48kHz Stereo Audio  │
    └─────────────────────────┘
    ```
    """)

st.markdown("---")

# Quick links
st.markdown("### Quick Start")
qcol1, qcol2, qcol3, qcol4 = st.columns(4)

with qcol1:
    st.markdown("#### Generate Music")
    st.markdown("Create audio with controllable parameters — genre presets, BPM, key, and more.")
    st.page_link("pages/1_Generate.py", label="Open Generator", icon="🎹")

with qcol2:
    st.markdown("#### Gallery")
    st.markdown("Browse, replay, and compare all generated audio samples.")
    st.page_link("pages/2_Gallery.py", label="Open Gallery", icon="🎧")

with qcol3:
    st.markdown("#### Architecture")
    st.markdown("Explore the distributed inference pipeline and timing analysis.")
    st.page_link("pages/3_Architecture.py", label="View Architecture", icon="🔬")

with qcol4:
    st.markdown("#### Metrics")
    st.markdown("Quantitative and qualitative evaluation of generated audio.")
    st.page_link("pages/4_Metrics.py", label="View Metrics", icon="📊")
