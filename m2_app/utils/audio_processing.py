"""
neural-noise Milestone 2 — Audio Processing Utilities

Handles audio I/O and basic audio statistics.
Uses soundfile if available, falls back to scipy.io.wavfile.
"""

import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Try soundfile, fall back to scipy
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    from scipy.io import wavfile as scipy_wav
    logger.info("soundfile not available, using scipy.io.wavfile (wav-only)")


def load_audio(path: str) -> Tuple[Optional[np.ndarray], int]:
    """
    Load an audio file and return (audio_array [channels, samples], sample_rate).
    """
    try:
        if HAS_SOUNDFILE:
            data, sr = sf.read(path, always_2d=True)
            return data.T, sr  # [channels, samples]
        else:
            sr, data = scipy_wav.read(path)
            data = data.astype(np.float32)
            # Normalize int16/int32 to float [-1, 1]
            if data.dtype == np.int16 or np.max(np.abs(data)) > 1.0:
                data = data / 32768.0
            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim == 2:
                data = data.T
            return data, sr
    except Exception as e:
        logger.error(f"Failed to load audio from {path}: {e}")
        return None, 0


def get_audio_stats(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """Compute basic audio statistics."""
    if audio is None or audio.size == 0:
        return {}

    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    num_channels, num_samples = audio.shape
    duration = num_samples / sample_rate
    signal = audio[0]
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    db_rms = float(20 * np.log10(rms + 1e-10))
    db_peak = float(20 * np.log10(peak + 1e-10))

    return {
        "duration_seconds": round(duration, 2),
        "sample_rate": sample_rate,
        "channels": num_channels,
        "num_samples": num_samples,
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "rms_db": round(db_rms, 2),
        "peak_db": round(db_peak, 2),
    }


def compute_acoustic_descriptors(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> Dict[str, float]:
    """
    Compute frame-level spectral descriptors and aggregate them across the clip.

    Implemented with numpy + scipy (no librosa dependency). Returns means over
    all frames so the values can be compared across clips of different lengths.

    Features:
      - spectral_centroid_hz: frequency-weighted mean of the magnitude spectrum
        (proxy for "brightness"; higher = brighter/treble-heavy).
      - spectral_bandwidth_hz: spectrum spread around the centroid (proxy for
        timbral richness; higher = wider frequency content).
      - spectral_rolloff_hz: frequency below which 85% of spectral energy lies
        (proxy for high-frequency cutoff).
      - zero_crossing_rate: fraction of samples where the signal changes sign
        (proxy for percussiveness/noisiness).
      - rms_energy: time-domain RMS averaged across frames.
      - tempo_bpm: tempo estimate from autocorrelation of the onset envelope.
    """
    from scipy import signal as sps

    if audio is None or audio.size == 0:
        return {}

    if audio.ndim == 2:
        x = audio.mean(axis=0)  # mono mix-down for analysis
    else:
        x = audio
    x = x.astype(np.float32)

    if len(x) < frame_size:
        return {}

    # Short-time Fourier transform
    f, _, Zxx = sps.stft(
        x,
        fs=sample_rate,
        nperseg=frame_size,
        noverlap=frame_size - hop_size,
        boundary=None,
    )
    mag = np.abs(Zxx)  # [freqs, frames]
    eps = 1e-10
    mag_sum = mag.sum(axis=0) + eps

    # Spectral centroid per frame, then mean across frames
    centroid = (f[:, None] * mag).sum(axis=0) / mag_sum
    centroid_mean = float(np.mean(centroid))

    # Spectral bandwidth (std around centroid, weighted by magnitude)
    bandwidth = np.sqrt(
        ((f[:, None] - centroid[None, :]) ** 2 * mag).sum(axis=0) / mag_sum
    )
    bandwidth_mean = float(np.mean(bandwidth))

    # Spectral rolloff at 85% of cumulative energy
    cum_energy = np.cumsum(mag, axis=0)
    total_energy = cum_energy[-1, :] + eps
    rolloff_mask = cum_energy >= 0.85 * total_energy[None, :]
    rolloff_indices = np.argmax(rolloff_mask, axis=0)
    rolloff_hz = f[rolloff_indices]
    rolloff_mean = float(np.mean(rolloff_hz))

    # Zero-crossing rate (frame-based, then mean)
    framed = np.lib.stride_tricks.sliding_window_view(x, frame_size)[::hop_size]
    sign_changes = np.diff(np.sign(framed), axis=1) != 0
    zcr = sign_changes.mean(axis=1)
    zcr_mean = float(np.mean(zcr))

    # RMS energy (frame-based, then mean)
    rms_frames = np.sqrt(np.mean(framed ** 2, axis=1))
    rms_mean = float(np.mean(rms_frames))

    # Tempo estimate via autocorrelation of the onset envelope
    onset_env = np.maximum(0, np.diff(rms_frames))
    if onset_env.size > 8:
        onset_env = onset_env - onset_env.mean()
        ac = np.correlate(onset_env, onset_env, mode="full")
        ac = ac[ac.size // 2 :]
        # Search lag range corresponding to 40-220 BPM
        frame_rate = sample_rate / hop_size
        min_lag = max(1, int(frame_rate * 60 / 220))
        max_lag = min(ac.size - 1, int(frame_rate * 60 / 40))
        if max_lag > min_lag:
            best_lag = int(np.argmax(ac[min_lag:max_lag])) + min_lag
            tempo_bpm = 60.0 * frame_rate / best_lag
        else:
            tempo_bpm = float("nan")
    else:
        tempo_bpm = float("nan")

    return {
        "spectral_centroid_hz": round(centroid_mean, 1),
        "spectral_bandwidth_hz": round(bandwidth_mean, 1),
        "spectral_rolloff_hz": round(rolloff_mean, 1),
        "zero_crossing_rate": round(zcr_mean, 4),
        "rms_energy": round(rms_mean, 5),
        "tempo_bpm": round(tempo_bpm, 1) if not np.isnan(tempo_bpm) else None,
    }


def list_generated_audio(output_dir: str) -> list:
    """List all generated audio files, sorted by newest first."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    audio_extensions = {".wav", ".flac", ".mp3", ".opus", ".aac"}
    files = []
    for f in output_path.iterdir():
        if f.suffix.lower() in audio_extensions and f.is_file():
            stat = f.stat()
            files.append({
                "path": str(f),
                "filename": f.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified_time": stat.st_mtime,
            })
    files.sort(key=lambda x: x["modified_time"], reverse=True)
    return files


def save_generation_metadata(audio_path: str, metadata: Dict[str, Any]):
    """Save generation metadata as a JSON sidecar file."""
    meta_path = Path(audio_path).with_suffix(".json")
    try:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save metadata to {meta_path}: {e}")


def load_generation_metadata(audio_path: str) -> Optional[Dict[str, Any]]:
    """Load generation metadata from the JSON sidecar file."""
    meta_path = Path(audio_path).with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata from {meta_path}: {e}")
        return None
