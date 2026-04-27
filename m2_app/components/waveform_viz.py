"""
neural-noise Milestone 2 — Waveform & Spectrogram Visualizations

Creates interactive figures for audio waveforms and mel-spectrograms.
Uses Plotly if available, falls back to Matplotlib.
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import plotly, fall back to matplotlib
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.info("Plotly not available, using Matplotlib for visualizations")

# Theme colors matching style.css
# Plotly accepts CSS rgba(), matplotlib needs tuples or hex
COLORS = {
    "bg": "#0a0a0f",
    "text": "#e4e4e7",
    "text_secondary": "#a1a1aa",
    "accent_violet": "#8b5cf6",
    "accent_cyan": "#06b6d4",
    "accent_emerald": "#10b981",
    "grid": "rgba(255, 255, 255, 0.05)",  # Plotly-compatible
}

# Matplotlib-compatible colors (RGBA tuples)
MPL_GRID = (1.0, 1.0, 1.0, 0.05)


def create_waveform_figure(audio: np.ndarray, sample_rate: int,
                           title: str = "Waveform", max_points: int = 10000):
    """Create an interactive waveform plot."""
    if audio.ndim == 2:
        signal = audio[0]
    else:
        signal = audio

    num_samples = len(signal)
    duration = num_samples / sample_rate

    if num_samples > max_points:
        step = num_samples // max_points
        signal = signal[::step]
    times = np.linspace(0, duration, len(signal))

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=signal, mode="lines",
            line=dict(color=COLORS["accent_violet"], width=1),
            fill="tozeroy", fillcolor="rgba(139, 92, 246, 0.1)",
            name="Amplitude",
            hovertemplate="Time: %{x:.2f}s<br>Amplitude: %{y:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
            font=dict(color=COLORS["text_secondary"], family="Inter", size=11),
            xaxis=dict(title="Time (seconds)", gridcolor=COLORS["grid"]),
            yaxis=dict(title="Amplitude", gridcolor=COLORS["grid"], range=[-1.1, 1.1]),
            margin=dict(l=50, r=20, t=40, b=40), height=250,
        )
        return fig
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 3), facecolor=COLORS["bg"])
        ax.set_facecolor(COLORS["bg"])
        ax.plot(times, signal, color=COLORS["accent_violet"], linewidth=0.5, alpha=0.9)
        ax.fill_between(times, signal, alpha=0.1, color=COLORS["accent_violet"])
        ax.set_xlabel("Time (seconds)", color=COLORS["text_secondary"], fontsize=10)
        ax.set_ylabel("Amplitude", color=COLORS["text_secondary"], fontsize=10)
        ax.set_title(title, color=COLORS["text"], fontsize=12)
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(colors=COLORS["text_secondary"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(MPL_GRID)
        plt.tight_layout()
        return fig


def create_spectrogram_figure(audio: np.ndarray, sample_rate: int,
                               title: str = "Spectrogram", n_fft: int = 2048,
                               hop_length: int = 512):
    """Create a spectrogram visualization."""
    if audio.ndim == 2:
        signal = audio[0]
    else:
        signal = audio

    # Compute spectrogram using scipy
    from scipy.signal import spectrogram as scipy_spectrogram
    freqs, times_arr, Sxx = scipy_spectrogram(
        signal.astype(np.float32), fs=sample_rate,
        nperseg=min(n_fft, len(signal)), noverlap=min(n_fft - hop_length, len(signal) - 1),
    )
    S_dB = 10 * np.log10(Sxx + 1e-10)

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=S_dB, x=times_arr, y=freqs,
            colorscale=[[0, "#0a0a0f"], [0.2, "#1e1b4b"], [0.4, "#4c1d95"],
                        [0.6, "#7c3aed"], [0.8, "#a78bfa"], [1.0, "#06b6d4"]],
            showscale=True,
            colorbar=dict(title=dict(text="dB", font=dict(color=COLORS["text_secondary"], size=10)),
                         tickfont=dict(color=COLORS["text_secondary"], size=9), thickness=12),
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
            font=dict(color=COLORS["text_secondary"], family="Inter", size=11),
            xaxis=dict(title="Time (seconds)", gridcolor=COLORS["grid"]),
            yaxis=dict(title="Frequency (Hz)", gridcolor=COLORS["grid"]),
            margin=dict(l=50, r=20, t=40, b=40), height=280,
        )
        return fig
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4), facecolor=COLORS["bg"])
        ax.set_facecolor(COLORS["bg"])
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
            "nn_cmap", ["#0a0a0f", "#1e1b4b", "#4c1d95", "#7c3aed", "#a78bfa", "#06b6d4"])
        im = ax.pcolormesh(times_arr, freqs, S_dB, cmap=cmap, shading="auto")
        ax.set_xlabel("Time (seconds)", color=COLORS["text_secondary"], fontsize=10)
        ax.set_ylabel("Frequency (Hz)", color=COLORS["text_secondary"], fontsize=10)
        ax.set_title(title, color=COLORS["text"], fontsize=12)
        ax.tick_params(colors=COLORS["text_secondary"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(MPL_GRID)
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("dB", color=COLORS["text_secondary"], fontsize=10)
        cbar.ax.tick_params(colors=COLORS["text_secondary"], labelsize=9)
        plt.tight_layout()
        return fig


def create_generation_timeline(time_costs: dict, title: str = "Pipeline Timing"):
    """Create a horizontal bar chart showing the timing breakdown."""
    label_map = {
        "lm_phase1_time": "LM Phase 1 (CoT)", "lm_phase2_time": "LM Phase 2 (Codes)",
        "encoder_time_cost": "Text Encoding", "diffusion_time_cost": "DiT Diffusion",
        "offload_time_cost": "Model Offloading", "dit_total_time_cost": "DiT Total",
        "pipeline_total_time": "Pipeline Total", "compose_time": "LM Compose (Remote)",
        "render_time": "DiT Render",
    }

    labels, values = [], []
    for key, val in time_costs.items():
        if isinstance(val, (int, float)) and val > 0 and key in label_map:
            labels.append(label_map[key])
            values.append(round(val, 2))

    if not labels:
        return None

    bar_colors = ["#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#ec4899"]

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=labels, x=values, orientation="h",
            marker=dict(color=[bar_colors[i % len(bar_colors)] for i in range(len(labels))]),
            text=[f"{v:.1f}s" for v in values], textposition="auto",
            textfont=dict(color="white", size=11),
        ))
        fig.update_layout(
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
            font=dict(color=COLORS["text_secondary"], family="Inter"),
            xaxis=dict(title="Time (seconds)", gridcolor=COLORS["grid"]),
            yaxis=dict(autorange="reversed", gridcolor=COLORS["grid"]),
            margin=dict(l=10, r=20, t=10, b=40),
            height=max(180, len(labels) * 40 + 80),
        )
        return fig
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.6)), facecolor=COLORS["bg"])
        ax.set_facecolor(COLORS["bg"])
        colors = [bar_colors[i % len(bar_colors)] for i in range(len(labels))]
        bars = ax.barh(labels, values, color=colors, height=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}s", va="center", color=COLORS["text"], fontsize=10)
        ax.set_xlabel("Time (seconds)", color=COLORS["text_secondary"], fontsize=10)
        ax.invert_yaxis()
        ax.tick_params(colors=COLORS["text_secondary"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(MPL_GRID)
        plt.tight_layout()
        return fig
