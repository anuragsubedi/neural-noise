"""
neural-noise — Architecture Page
==================================

Interactive visualization of the distributed inference pipeline.
"""

import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Architecture | neural-noise", page_icon="🔬", layout="wide")

_APP_ROOT = Path(__file__).resolve().parents[1]
css_path = _APP_ROOT / "static" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("# Inference Pipeline Architecture")
st.markdown("Understanding the distributed ACE-Step 1.5 generation pipeline.")
st.markdown("---")

arch_mode = st.radio("View Mode", ["Single Machine (Local)", "Distributed (Two Nodes)"], horizontal=True)
st.markdown("---")

if arch_mode == "Single Machine (Local)":
    st.markdown("## Single Machine Pipeline")
    st.markdown("All components run on the M2 MacBook Air with **CPU offloading**. "
                "The LM and DiT take turns using the GPU.")
    st.markdown("""
    ```
    M2 MacBook Air (16GB Unified Memory)
    ┌────────────────────────────────────────────────────────────┐
    │  Streamlit Dashboard                                       │
    │       │                                                    │
    │       ▼                                                    │
    │  LM Planner (Qwen 0.6B) ──load/offload──→ ~25s            │
    │       │ CoT metadata + audio_codes                         │
    │       ▼                                                    │
    │  Audio Code Gen (LM Phase 2) ──load/offload──→ ~17s        │
    │       │ audio_codes                                        │
    │       ▼                                                    │
    │  DiT Diffusion (2B Turbo, 8 steps) ──→ ~46s                │
    │       │ pred_latents [1, 750, 64]                          │
    │       ▼                                                    │
    │  VAE Decode (MLX) ──→ ~21s                                 │
    │       │ 48kHz stereo .wav                                  │
    │       ▼                                                    │
    │  Total: ~3 minutes (18s wasted on model offloading)        │
    └────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("### Timing Breakdown")
    import plotly.graph_objects as go
    timing = {"LM Phase 1 (CoT)": 25.44, "LM Phase 2 (Codes)": 16.99, "Text Encoding": 16.0,
              "DiT Diffusion": 45.1, "Offloading Overhead": 18.3, "VAE Decode": 21.05}
    fig = go.Figure(go.Bar(y=list(timing.keys()), x=list(timing.values()), orientation="h",
        marker=dict(color=["#8b5cf6","#a855f7","#3b82f6","#06b6d4","#ef4444","#10b981"]),
        text=[f"{v:.1f}s" for v in timing.values()], textposition="auto",
        textfont=dict(color="white", size=12)))
    fig.update_layout(paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font=dict(color="#a1a1aa", family="Inter"), xaxis_title="Time (seconds)",
        yaxis=dict(autorange="reversed"), margin=dict(l=10,r=20,t=10,b=40), height=300,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis_gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, width="stretch")
    st.warning("**Bottleneck:** 18.3s spent on model offloading. Distributed mode eliminates this.")

else:
    st.markdown("## Distributed Two-Node Pipeline")
    st.markdown("Both models stay **permanently loaded** in their respective GPUs.")
    st.markdown("""
    ```
    Windows (RTX 3060, 6GB)              M2 MacBook Air (16GB)
    ┌─────────────────────┐     HTTP     ┌──────────────────────┐
    │ LM Service (:8001)  │◄───JSON───►  │ Streamlit + DiT      │
    │ Qwen3 0.6B (CUDA)   │             │ DiT Turbo (MPS)      │
    │ Always hot in VRAM  │             │ Always hot in MPS    │
    │ ~15s compose        │             │ ~45s render          │
    └─────────────────────┘             └──────────────────────┘
                  Same WiFi Network
    ```
    """)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("LM Compose", "~15s", delta="-27s", delta_color="inverse")
    with c2: st.metric("DiT Render", "~45s", delta="-18s offload", delta_color="inverse")
    with c3: st.metric("Total", "~60s", delta="-2min", delta_color="inverse")
    st.success("No model offloading. Both GPUs fully utilized.")

st.markdown("---")
st.markdown("## Model Zoo")
st.markdown("""
| DiT Model | Params | Steps | CFG | Quality |
|-----------|--------|-------|-----|---------|
| **acestep-v15-turbo** (active) | 2B | 4-8 | No | Very High |
| acestep-v15-base | 2B | 50 | Yes | Medium |
| acestep-v15-sft | 2B | 50 | Yes | High |

| LM Model | Params | CoT | Audio Understanding |
|----------|--------|-----|---------------------|
| **acestep-5Hz-lm-0.6B** (active) | 0.6B | Yes | Medium |
| acestep-5Hz-lm-1.7B | 1.7B | Yes | Medium |
| acestep-5Hz-lm-4B | 4B | Yes | Strong |
""")

st.markdown("---")
st.markdown("## Key Concepts")
t1, t2, t3 = st.tabs(["Chain-of-Thought", "Diffusion Process", "Why Not Stream?"])
with t1:
    st.markdown("The Qwen LM first reasons about musical structure via CoT: analyze prompt → "
                "generate metadata (BPM, key) → refine caption → produce audio codes. "
                "This ensures coherent conditioning for the DiT.")
with t2:
    st.markdown("The DiT denoises the **entire** latent space [1, 750, 64] simultaneously across "
                "8 ODE steps. Unlike autoregressive models, there is no left-to-right generation. "
                "The VAE then decodes the refined latent into 48kHz stereo audio.")
with t3:
    st.markdown("Diffusion models denoise concurrently — early steps produce noise, not usable audio. "
                "For streaming, we'd need chunked autoregressive generation (10s segments). "
                "We can preview intermediate steps for interpretability.")
