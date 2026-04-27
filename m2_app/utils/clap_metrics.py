"""
neural-noise Milestone 2 — CLAP Similarity Scoring

Computes text-audio similarity using LAION-CLAP via the HuggingFace transformers
ClapModel. The first call downloads the model (~614 MB) into the local HF cache.

Score interpretation
--------------------
For a single (audio, caption) pair, the score is the cosine similarity between
the L2-normalized text and audio embeddings, scaled to [-1, 1]. Empirically,
LAION-CLAP scores for well-aligned music/text pairs land in the 0.30-0.55 range;
random or mismatched pairs typically score 0.05-0.20.

We report this as the primary M2 alignment metric because, unlike FAD, CLAP is
meaningful at any sample size — each (audio, caption) pair is an independent
measurement of "did the model follow the prompt?".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_TARGET_SR = 48000  # LAION-CLAP expects 48kHz audio


_clap_state: Dict[str, object] = {
    "model": None,
    "processor": None,
    "device": None,
}


def _resample_if_needed(audio: np.ndarray, sr: int) -> np.ndarray:
    """Down-mix to mono and resample to CLAP's target sample rate if needed."""
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    if sr == CLAP_TARGET_SR:
        return audio.astype(np.float32)

    from scipy.signal import resample_poly

    # Use rational resampling to avoid floating-point ratios
    from math import gcd
    g = gcd(sr, CLAP_TARGET_SR)
    up = CLAP_TARGET_SR // g
    down = sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def is_clap_available() -> bool:
    """Check whether the transformers ClapModel API is importable."""
    try:
        from transformers import ClapModel, ClapProcessor  # noqa: F401
        return True
    except Exception:
        return False


def load_clap_model(device: str = "cpu"):
    """
    Lazily load the CLAP model + processor into a module-level cache.

    The first call may download ~614 MB on first run. Subsequent calls reuse
    the already-loaded model. We default to CPU because CLAP inference is
    one-shot per sample and MPS support for some HF audio models is fragile.
    """
    if _clap_state["model"] is not None:
        return _clap_state["model"], _clap_state["processor"], _clap_state["device"]

    from transformers import ClapModel, ClapProcessor
    import torch

    logger.info(f"Loading CLAP model ({CLAP_MODEL_ID}) on {device}...")
    model = ClapModel.from_pretrained(CLAP_MODEL_ID)
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    model = model.to(device).eval()

    _clap_state["model"] = model
    _clap_state["processor"] = processor
    _clap_state["device"] = device
    return model, processor, device


def compute_clap_score(
    audio: np.ndarray,
    sample_rate: int,
    caption: str,
    device: str = "cpu",
) -> Optional[float]:
    """
    Compute the CLAP cosine similarity between an audio array and a caption.

    Returns a float in [-1, 1], or None if CLAP is unavailable or the inputs
    are degenerate.
    """
    if not is_clap_available():
        return None
    if audio is None or audio.size == 0 or not caption or not caption.strip():
        return None

    import torch

    model, processor, _ = load_clap_model(device=device)

    audio_mono = _resample_if_needed(audio, sample_rate)

    inputs = processor(
        text=[caption.strip()],
        audio=[audio_mono],
        return_tensors="pt",
        sampling_rate=CLAP_TARGET_SR,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        audio_emb = model.get_audio_features(
            input_features=inputs["input_features"],
            is_longer=inputs.get("is_longer"),
        )
        text_emb = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )

    # Normalize and compute cosine similarity (CLAP embeddings are not pre-normalized)
    audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)
    text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
    score = float((audio_emb * text_emb).sum(dim=-1).cpu().item())
    return score


def compute_clap_scores_for_corpus(
    samples: List[Dict[str, object]],
    device: str = "cpu",
    progress_callback=None,
) -> List[Dict[str, object]]:
    """
    Compute CLAP scores for a list of {audio_path, caption} dicts.

    Each input dict should have 'audio_path' and 'caption' keys. Returns the
    same list of dicts with an added 'clap_score' key (None if computation
    failed for that sample).
    """
    from .audio_processing import load_audio

    results = []
    for i, sample in enumerate(samples):
        audio_path = sample.get("audio_path")
        caption = sample.get("caption", "") or ""
        score = None
        if audio_path and Path(audio_path).exists() and caption.strip():
            try:
                audio, sr = load_audio(audio_path)
                if audio is not None:
                    score = compute_clap_score(audio, sr, caption, device=device)
            except Exception as e:
                logger.error(f"CLAP scoring failed for {audio_path}: {e}")
        out = dict(sample)
        out["clap_score"] = score
        results.append(out)
        if progress_callback is not None:
            progress_callback(i + 1, len(samples))
    return results
