"""
neural-noise Milestone 2 — Pipeline Configuration

Centralizes all paths, URLs, and defaults for the distributed inference pipeline.
Both single-machine (local) and distributed (LM on Windows, DiT on Mac) modes
are configured here.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Path resolution: ACE-Step library must be importable
# ---------------------------------------------------------------------------
# ACE-Step-1.5 lives inside m2_app/backend/ (same dir as this file)
_BACKEND_DIR = Path(__file__).resolve().parent
ACESTEP_ROOT = _BACKEND_DIR / "ACE-Step-1.5"

# Add ACE-Step to sys.path so `from acestep.xxx import ...` works
if str(ACESTEP_ROOT) not in sys.path:
    sys.path.insert(0, str(ACESTEP_ROOT))


@dataclass
class PipelineConfig:
    """Configuration for the ACE-Step inference pipeline."""

    # --- Mode ---
    mode: str = "local"  # "local" or "distributed"

    # --- Paths ---
    acestep_root: str = str(ACESTEP_ROOT)
    checkpoint_dir: str = str(ACESTEP_ROOT / "checkpoints")
    output_dir: str = str(Path(__file__).resolve().parents[1] / "output")
    presets_path: str = str(Path(__file__).resolve().parents[1] / "presets" / "genres.json")

    # --- Model Selection ---
    dit_model: str = "acestep-v15-turbo"
    lm_model: str = "acestep-5Hz-lm-0.6B"

    # --- Device ---
    dit_device: str = "mps"       # "mps" for Mac, "cuda" for NVIDIA
    lm_device: str = "mps"        # "mps" for local, "cuda" for distributed
    lm_backend: str = "pt"        # "pt" for PyTorch, "vllm" for vLLM

    # --- Distributed Mode URLs ---
    lm_service_url: str = "http://localhost:8001"
    dit_service_url: str = "http://localhost:8002"

    # --- Generation Defaults ---
    default_duration: float = 30.0
    default_bpm: int = 120
    default_keyscale: str = "C Major"
    default_timesignature: str = "4"
    default_inference_steps: int = 8
    default_shift: float = 3.0
    default_batch_size: int = 1
    default_audio_format: str = "wav"

    # --- Apple Silicon Memory ---
    mps_high_watermark_ratio: str = "0.0"  # Allow SSD swap for memory spikes

    def __post_init__(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)

    def apply_environment(self):
        """Set environment variables needed before torch import."""
        if self.dit_device == "mps":
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = self.mps_high_watermark_ratio

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create config from environment variables with sensible defaults."""
        return cls(
            mode=os.getenv("NN_MODE", "local"),
            lm_service_url=os.getenv("NN_LM_URL", "http://localhost:8001"),
            dit_service_url=os.getenv("NN_DIT_URL", "http://localhost:8002"),
            dit_device=os.getenv("NN_DIT_DEVICE", "mps"),
            lm_device=os.getenv("NN_LM_DEVICE", "mps"),
            lm_model=os.getenv("NN_LM_MODEL", "acestep-5Hz-lm-0.6B"),
            dit_model=os.getenv("NN_DIT_MODEL", "acestep-v15-turbo"),
        )


def get_config() -> PipelineConfig:
    """Singleton-style config getter."""
    return PipelineConfig.from_env()
