"""
neural-noise — Generate Music Page
====================================

The core generation interface with interactive controls mapped to
ACE-Step 1.5 GenerationParams. Supports genre presets, parameter
sliders, and real-time audio playback with waveform/spectrogram viz.
"""

import streamlit as st
import numpy as np
import time
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Ensure project imports work
_APP_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _APP_ROOT.parent
sys.path.insert(0, str(_APP_ROOT))

from components.preset_manager import get_preset_manager
from components.waveform_viz import (
    create_waveform_figure,
    create_spectrogram_figure,
    create_generation_timeline,
)
from components.chart_renderer import render_chart
from utils.audio_processing import (
    load_audio,
    get_audio_stats,
    save_generation_metadata,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Generate | neural-noise",
    page_icon="🎹",
    layout="wide",
)

# Load CSS
css_path = _APP_ROOT / "static" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cloud deployment guard — must come BEFORE any backend / torch imports
# ---------------------------------------------------------------------------
if os.environ.get("DEPLOY_MODE") == "cloud":
    st.markdown("# Generate Music")
    st.markdown("---")
    st.info(
        "**Generation is unavailable in the hosted demo.**\n\n"
        "The ACE-Step 1.5 inference pipeline requires a local GPU (Apple Silicon MPS "
        "or NVIDIA CUDA) and ~12 GB of model checkpoints — resources that are not "
        "available in the Streamlit Community Cloud environment.\n\n"
        "To run the full pipeline locally, follow the "
        "[README setup instructions](https://github.com/anuragsubedi/neural-noise"
        "#quick-start-milestone-2-streamlit-dashboard)."
    )
    st.markdown("### Explore the demo instead:")
    c1, c2 = st.columns(2)
    with c1:
        st.page_link("pages/2_Gallery.py", label="Browse the Gallery", icon="🎧")
        st.page_link("pages/4_Metrics.py", label="View Evaluation Metrics", icon="📊")
    with c2:
        st.page_link("pages/3_Architecture.py", label="Explore Architecture", icon="🔬")
        st.page_link("pages/5_MIDI_Baseline.py", label="M1 MIDI Baseline", icon="🎹")
    st.stop()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Initialize Preset Manager
# ---------------------------------------------------------------------------
preset_mgr = get_preset_manager()


# ---------------------------------------------------------------------------
# Engine Initialization (cached in session state)
# ---------------------------------------------------------------------------
def get_engine():
    """Get or initialize the inference engine (singleton in session state)."""
    if "engine" not in st.session_state or st.session_state.engine is None:
        return None
    return st.session_state.engine


def initialize_engine(mode: str = "local"):
    """Initialize the inference engine in the given mode."""
    from backend.config import PipelineConfig
    from backend.inference_engine import InferenceEngine

    config = PipelineConfig.from_env()
    config.mode = mode

    engine = InferenceEngine(config)

    # Show a status container that displays each init step
    status_container = st.status(
        f"Initializing {mode} pipeline...",
        expanded=True,
    )
    with status_container:
        st.write("Loading models — this may take 1-2 minutes on first run.")
        success = engine.initialize()

        # Display the status log from the engine
        if engine.status_log:
            for entry in engine.status_log:
                icon = "✓" if entry["level"] == "info" else "✗"
                st.write(f"`{entry['time']}` {icon} {entry['message']}")

    if success:
        status_container.update(label="Pipeline initialized", state="complete")
        st.session_state.engine = engine
        st.session_state.engine_initialized = True
        st.session_state.engine_mode = mode
    else:
        status_container.update(label="Initialization failed", state="error")
        st.error(f"Failed to initialize: {engine.initialization_error}")

    return success


# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------
st.markdown("# Generate Music")
st.markdown("Create audio with controllable parameters using the ACE-Step 1.5 pipeline.")
st.markdown("---")


# ---------------------------------------------------------------------------
# Engine Setup Section
# ---------------------------------------------------------------------------
engine = get_engine()

if engine is None:
    st.warning("The inference engine is not initialized. Initialize it below to start generating.")

    init_col1, init_col2 = st.columns([1, 2])
    with init_col1:
        mode = st.selectbox(
            "Pipeline Mode",
            ["local", "distributed"],
            help="**Local:** LM + DiT on this machine (slower, requires offloading)\n\n"
                 "**Distributed:** LM on remote GPU, DiT on this machine (faster)",
        )
    with init_col2:
        st.markdown("")  # spacing
        st.markdown("")
        if st.button("Initialize Pipeline", type="primary"):
            initialize_engine(mode)
            st.rerun()

    # Show previous init log if available
    if "engine" in st.session_state and st.session_state.engine is not None:
        eng = st.session_state.engine
        if hasattr(eng, "status_log") and eng.status_log:
            with st.expander("Last Initialization Log", expanded=False):
                for entry in eng.status_log:
                    icon = "✓" if entry["level"] == "info" else "✗"
                    st.text(f"[{entry['time']}] {icon} {entry['message']}")

    st.markdown("---")
    st.info("While the pipeline initializes, you can explore pre-generated samples in the Gallery.")
    st.stop()

# Show engine status info bar when initialized
engine_mode = st.session_state.get("engine_mode", "unknown")
with st.expander("Pipeline Status", expanded=False):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"**Mode:** `{engine_mode}`")
        st.markdown(f"**Status:** Initialized")
    with col_s2:
        if hasattr(engine, "status_log") and engine.status_log:
            for entry in engine.status_log[-3:]:  # show last 3 log entries
                icon = "✓" if entry["level"] == "info" else "✗"
                st.text(f"[{entry['time']}] {icon} {entry['message']}")


# ---------------------------------------------------------------------------
# Main Generation Interface (only shown when engine is ready)
# ---------------------------------------------------------------------------
ctrl_col, output_col = st.columns([2, 3], gap="large")

# ======================== CONTROL PANEL ========================
with ctrl_col:
    st.markdown("### Control Panel")

    # --- Preset Selection ---
    with st.expander("Genre Preset", expanded=True):
        preset_options = ["Custom"] + preset_mgr.preset_names
        # Default to first real preset so users always start with a valid caption
        default_preset_index = 1 if len(preset_options) > 1 else 0
        selected_preset = st.selectbox(
            "Select a genre preset",
            preset_options,
            index=default_preset_index,
            key="preset_selector",
            help="Presets provide curated starting points. Override any parameter below.",
        )

        # Load preset defaults
        if selected_preset != "Custom":
            preset_data = preset_mgr.get_preset(selected_preset)
            if preset_data:
                st.caption(f'*"{preset_data.get("caption", "")[:80]}..."*')

        mood_options = ["None"] + preset_mgr.moods
        selected_mood = st.selectbox(
            "Mood modifier",
            mood_options,
            index=0,
            key="mood_selector",
            help="Prepends a mood adjective to the caption for emotional steering.",
        )

    # --- Music Parameters ---
    with st.expander("Music Parameters", expanded=True):
        # Get defaults from preset
        preset_defaults = preset_mgr.get_preset(selected_preset) if selected_preset != "Custom" else {}

        bpm = st.slider(
            "BPM (Tempo)",
            min_value=60, max_value=200,
            value=preset_defaults.get("bpm", 120),
            step=5,
            key="bpm_slider",
            help="Beats per minute. 60-80: slow/ambient, 100-130: moderate, 140+: energetic.",
        )

        key_options = preset_mgr.keys
        default_key = preset_defaults.get("keyscale", "C Major")
        key_index = key_options.index(default_key) if default_key in key_options else 0
        keyscale = st.selectbox(
            "Musical Key",
            key_options,
            index=key_index,
            key="key_selector",
        )

        ts_display = list(preset_mgr.time_signatures.keys())
        ts_values = list(preset_mgr.time_signatures.values())
        default_ts_val = preset_defaults.get("timesignature", "4")
        ts_index = ts_values.index(default_ts_val) if default_ts_val in ts_values else 2
        timesig_display = st.selectbox(
            "Time Signature",
            ts_display,
            index=ts_index,
            key="timesig_selector",
        )
        timesignature = preset_mgr.time_signatures[timesig_display]

        duration = st.slider(
            "Duration (seconds)",
            min_value=10, max_value=60,
            value=30,
            step=5,
            key="duration_slider",
        )

    # --- Text Description ---
    with st.expander("Custom Text Description", expanded=False):
        custom_caption = st.text_area(
            "Override caption",
            value="",
            height=100,
            key="custom_caption",
            help="If provided, this completely replaces the preset caption.",
            placeholder="e.g., 'dark ambient drone with metallic textures and vast reverb'",
        )

    # --- Vocals & Lyrics ---
    with st.expander("Vocals & Lyrics", expanded=False):
        instrumental = st.toggle(
            "Instrumental (no vocals)",
            value=True,
            key="instrumental_toggle",
            help="When ON, the model generates instrumental music and lyrics are ignored.",
        )

        lyrics_disabled = instrumental
        custom_lyrics = st.text_area(
            "Lyrics",
            value="",
            height=140,
            key="custom_lyrics",
            disabled=lyrics_disabled,
            help="Lyrics for vocal generation. Use [verse], [chorus], [bridge] tags to "
                 "structure the song. Ignored when 'Instrumental' is ON.",
            placeholder="[verse]\nWalking through the neon haze\nLost inside a synth-wave maze\n\n[chorus]\nBurn it down, light the night...",
        )
        if lyrics_disabled:
            st.caption("Turn off the Instrumental toggle above to enable lyrics input.")

    # --- Advanced Settings ---
    with st.expander("Advanced Settings", expanded=False):
        inference_steps = st.slider(
            "Inference Steps",
            min_value=4, max_value=20,
            value=8,
            key="steps_slider",
            help="More steps = higher quality but slower. 8 is recommended for turbo model.",
        )

        shift = st.slider(
            "Timestep Shift",
            min_value=1.0, max_value=5.0,
            value=3.0, step=0.5,
            key="shift_slider",
            help="Shift factor for timestep scheduling. 3.0 recommended for turbo.",
        )

        seed = st.number_input(
            "Seed (-1 = random)",
            min_value=-1, max_value=999999999,
            value=-1,
            key="seed_input",
            help="Fixed seed for reproducible results. -1 for random.",
        )

    # --- Generate Button ---
    st.markdown("---")

    generate_clicked = st.button(
        "Generate Music",
        type="primary",
        width="stretch",
        key="generate_btn",
    )


# ======================== OUTPUT PANEL ========================
with output_col:
    st.markdown("### Output")

    if generate_clicked:
        # Build generation parameters
        params = preset_mgr.build_generation_params(
            preset_name=selected_preset if selected_preset != "Custom" else None,
            mood_override=selected_mood if selected_mood != "None" else None,
            caption_override=custom_caption if custom_caption.strip() else None,
            lyrics_override=custom_lyrics if custom_lyrics.strip() else None,
            bpm_override=bpm,
            keyscale_override=keyscale,
            timesignature_override=timesignature,
            duration=float(duration),
            inference_steps=inference_steps,
            shift=shift,
            seed=seed,
            instrumental=instrumental,
        )

        # --- Submit-time validation ---
        validation_errors = []
        # The preset_manager fills in a fallback caption, but warn the user if
        # they hit Generate with no preset, no mood, and no custom caption.
        no_preset = selected_preset == "Custom"
        no_mood = selected_mood == "None"
        no_custom_caption = not custom_caption.strip()
        if no_preset and no_mood and no_custom_caption:
            validation_errors.append(
                "No caption source provided. Pick a preset, a mood, or write a custom caption."
            )
        if not instrumental and not custom_lyrics.strip():
            validation_errors.append(
                "Vocal mode is on but no lyrics were provided. Either enable Instrumental "
                "or write some lyrics."
            )

        if validation_errors:
            for err in validation_errors:
                st.error(err)
            st.stop()

        # Show what we're generating
        with st.expander("Generation Parameters", expanded=False):
            st.json(params)

        # Run generation
        progress_bar = st.progress(0, text="Starting generation pipeline...")
        status_text = st.empty()

        try:
            status_text.markdown('<span class="status-generating">Generating audio...</span>',
                                unsafe_allow_html=True)
            progress_bar.progress(10, text="Sending to inference engine...")

            result = engine.generate(params)

            progress_bar.progress(100, text="Complete!")
            time.sleep(0.3)
            progress_bar.empty()

            if result.success and result.audio_path:
                status_text.markdown('<span class="status-online">Generation Complete!</span>',
                                    unsafe_allow_html=True)

                # Store result in session state
                st.session_state.last_result = result
                st.session_state.last_params = params
                st.session_state.generation_count = st.session_state.get("generation_count", 0) + 1

                # Save metadata sidecar
                metadata = {
                    "params": params,
                    "seed": result.seed,
                    "generation_time": result.generation_time,
                    "cot_metadata": result.cot_metadata,
                    "time_costs": result.time_costs,
                    "timestamp": datetime.now().isoformat(),
                }
                save_generation_metadata(result.audio_path, metadata)

            else:
                status_text.empty()
                st.error(f"Generation failed: {result.error}")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Generation error: {e}")
            logger.error(f"Generation error: {e}", exc_info=True)

    # --- Display Results ---
    result = st.session_state.get("last_result")
    last_params = st.session_state.get("last_params")

    if result and result.success and result.audio_path and os.path.exists(result.audio_path):
        # Audio Player
        st.markdown("#### Audio Playback")
        st.audio(result.audio_path, format="audio/wav")

        # Generation Info
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.metric("Generation Time", f"{result.generation_time:.1f}s")
        with info_col2:
            st.metric("Seed", str(result.seed))
        with info_col3:
            st.metric("Sample Rate", f"{result.sample_rate // 1000}kHz")
        with info_col4:
            # Load audio stats
            audio_data, sr = load_audio(result.audio_path)
            if audio_data is not None:
                stats = get_audio_stats(audio_data, sr)
                st.metric("Duration", f"{stats.get('duration_seconds', 0):.1f}s")

        st.markdown("---")

        # Visualizations
        if audio_data is not None:
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Waveform", "Spectrogram", "Pipeline Timing"])

            with viz_tab1:
                fig_wave = create_waveform_figure(audio_data, sr)
                render_chart(fig_wave)

            with viz_tab2:
                fig_spec = create_spectrogram_figure(audio_data, sr)
                render_chart(fig_spec)

            with viz_tab3:
                if result.time_costs:
                    fig_timeline = create_generation_timeline(result.time_costs)
                    if fig_timeline:
                        render_chart(fig_timeline)
                    else:
                        st.caption("No timing data available for this generation.")
                else:
                    st.caption("Timing breakdown not available.")

        # CoT Metadata
        if result.cot_metadata:
            with st.expander("LM Chain-of-Thought Metadata", expanded=False):
                st.json(result.cot_metadata)

        # Download
        with open(result.audio_path, "rb") as f:
            st.download_button(
                "Download Audio",
                data=f,
                file_name=Path(result.audio_path).name,
                mime="audio/wav",
            )

    elif not generate_clicked:
        # Empty state
        st.markdown("""
        <div style="
            text-align: center;
            padding: 4rem 2rem;
            border: 1px dashed rgba(255,255,255,0.1);
            border-radius: 16px;
            color: #71717a;
        ">
            <h3 style="color: #a1a1aa;">No Audio Generated Yet</h3>
            <p>Configure your parameters in the Control Panel and click <strong>Generate Music</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
