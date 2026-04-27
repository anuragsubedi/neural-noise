"""
neural-noise — Gallery Page
=============================

Browse, replay, and compare all generated audio samples.
Loads audio files and their JSON sidecar metadata from the output directory.
"""

import streamlit as st
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Ensure project imports work
_APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_APP_ROOT))

from utils.audio_processing import (
    list_generated_audio,
    load_audio,
    get_audio_stats,
    load_generation_metadata,
)
from components.waveform_viz import create_waveform_figure, create_spectrogram_figure
from components.chart_renderer import render_chart

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Gallery | neural-noise",
    page_icon="🎧",
    layout="wide",
)

css_path = _APP_ROOT / "static" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page Content
# ---------------------------------------------------------------------------
st.markdown("# Playback Gallery")
st.markdown("Browse and compare all generated audio samples.")
st.markdown("---")

# Get output directory
output_dir = str(_APP_ROOT / "output")
os.makedirs(output_dir, exist_ok=True)

# List all audio files
audio_files = list_generated_audio(output_dir)

if not audio_files:
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        border: 1px dashed rgba(255,255,255,0.1);
        border-radius: 16px;
        color: #71717a;
    ">
        <h3 style="color: #a1a1aa;">No Generations Yet</h3>
        <p>Head to the <strong>Generate</strong> page to create your first audio sample.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Summary
st.markdown(f"**{len(audio_files)}** audio samples found")

# ---------------------------------------------------------------------------
# Gallery Grid
# ---------------------------------------------------------------------------
for i, audio_info in enumerate(audio_files):
    filepath = audio_info["path"]
    filename = audio_info["filename"]
    size_kb = audio_info["size_kb"]
    mod_time = datetime.fromtimestamp(audio_info["modified_time"])

    # Load metadata sidecar
    metadata = load_generation_metadata(filepath)

    with st.container():
        st.markdown(f"### {filename}")

        info_col, player_col = st.columns([1, 2])

        with info_col:
            st.caption(f"Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            st.caption(f"Size: {size_kb:.0f} KB")

            if metadata:
                params = metadata.get("params", {})
                if params.get("caption"):
                    st.markdown(f"**Caption:** *{params['caption'][:100]}...*" if len(params.get('caption', '')) > 100
                                else f"**Caption:** *{params.get('caption', 'N/A')}*")

                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.metric("BPM", params.get("bpm", "Auto"))
                    st.metric("Key", params.get("keyscale", "Auto"))
                with mcol2:
                    st.metric("Seed", metadata.get("seed", "N/A"))
                    gen_time = metadata.get("generation_time", 0)
                    st.metric("Gen Time", f"{gen_time:.1f}s" if gen_time else "N/A")

        with player_col:
            # Audio playback
            st.audio(filepath, format="audio/wav")

            # Visualizations (collapsible)
            with st.expander("Waveform & Spectrogram", expanded=False):
                audio_data, sr = load_audio(filepath)
                if audio_data is not None:
                    stats = get_audio_stats(audio_data, sr)
                    st.caption(
                        f"Duration: {stats.get('duration_seconds', 0):.1f}s | "
                        f"Sample Rate: {stats.get('sample_rate', 0)}Hz | "
                        f"Peak: {stats.get('peak_db', 0):.1f}dB"
                    )

                    viz_t1, viz_t2 = st.tabs(["Waveform", "Spectrogram"])
                    with viz_t1:
                        fig = create_waveform_figure(audio_data, sr, title=f"Waveform — {filename}")
                        render_chart(fig)
                    with viz_t2:
                        fig = create_spectrogram_figure(audio_data, sr, title=f"Spectrogram — {filename}")
                        render_chart(fig)

            # Full metadata (collapsible)
            if metadata:
                with st.expander("Full Generation Metadata", expanded=False):
                    st.json(metadata)

        st.markdown("---")

# ---------------------------------------------------------------------------
# Comparison Mode
# ---------------------------------------------------------------------------
st.markdown("## Side-by-Side Comparison")

if len(audio_files) >= 2:
    comp_col1, comp_col2 = st.columns(2)

    filenames = [f["filename"] for f in audio_files]

    with comp_col1:
        sel_a = st.selectbox("Sample A", filenames, index=0, key="comp_a")
        path_a = next(f["path"] for f in audio_files if f["filename"] == sel_a)
        st.audio(path_a, format="audio/wav")
        meta_a = load_generation_metadata(path_a)
        if meta_a and meta_a.get("params"):
            st.caption(f"Caption: {meta_a['params'].get('caption', 'N/A')[:80]}")
            st.caption(f"BPM: {meta_a['params'].get('bpm')} | Key: {meta_a['params'].get('keyscale')}")

    with comp_col2:
        sel_b = st.selectbox("Sample B", filenames, index=min(1, len(filenames) - 1), key="comp_b")
        path_b = next(f["path"] for f in audio_files if f["filename"] == sel_b)
        st.audio(path_b, format="audio/wav")
        meta_b = load_generation_metadata(path_b)
        if meta_b and meta_b.get("params"):
            st.caption(f"Caption: {meta_b['params'].get('caption', 'N/A')[:80]}")
            st.caption(f"BPM: {meta_b['params'].get('bpm')} | Key: {meta_b['params'].get('keyscale')}")
else:
    st.caption("Generate at least 2 samples to enable side-by-side comparison.")
