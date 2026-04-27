"""
neural-noise Milestone 2 — Preset Manager

Loads and manages genre/mood presets for the generation controls.
Presets map user-friendly selections to ACE-Step GenerationParams.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class PresetManager:
    """Manages genre/mood presets for the music generation UI."""

    def __init__(self, presets_path: Optional[str] = None):
        """
        Args:
            presets_path: Path to the genres.json file.
                         Defaults to m2_app/presets/genres.json
        """
        if presets_path is None:
            presets_path = str(
                Path(__file__).resolve().parents[1] / "presets" / "genres.json"
            )

        self._presets_path = presets_path
        self._data = self._load_presets()

    def _load_presets(self) -> dict:
        """Load presets from the JSON file."""
        try:
            with open(self._presets_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load presets from {self._presets_path}: {e}")
            return {"presets": {}, "moods": [], "keys": [], "time_signatures": {}}

    @property
    def preset_names(self) -> List[str]:
        """List of all available preset names."""
        return list(self._data.get("presets", {}).keys())

    @property
    def moods(self) -> List[str]:
        """List of all available mood options."""
        return self._data.get("moods", [])

    @property
    def keys(self) -> List[str]:
        """List of all available musical keys."""
        return self._data.get("keys", [])

    @property
    def time_signatures(self) -> Dict[str, str]:
        """Dict mapping display names to ACE-Step values."""
        return self._data.get("time_signatures", {})

    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific preset by name."""
        return self._data.get("presets", {}).get(name)

    def build_generation_params(
        self,
        preset_name: Optional[str] = None,
        mood_override: Optional[str] = None,
        caption_override: Optional[str] = None,
        lyrics_override: Optional[str] = None,
        bpm_override: Optional[int] = None,
        keyscale_override: Optional[str] = None,
        timesignature_override: Optional[str] = None,
        duration: float = 30.0,
        inference_steps: int = 8,
        shift: float = 3.0,
        seed: int = -1,
        instrumental: bool = True,
    ) -> Dict[str, Any]:
        """
        Build a generation parameters dict from a preset + user overrides.

        Priority: explicit overrides > preset values > defaults

        Args:
            preset_name: Name of the preset to use as base
            mood_override: Override the mood (appended to caption)
            caption_override: Completely replace the caption
            bpm_override: Override BPM
            keyscale_override: Override musical key
            timesignature_override: Override time signature
            duration: Audio duration in seconds
            inference_steps: DiT inference steps
            shift: Timestep shift factor
            seed: Random seed (-1 for random)
            instrumental: Whether to generate instrumental music

        Returns:
            Dict ready to pass to InferenceEngine.generate()
        """
        # Start with defaults
        params = {
            "task_type": "text2music",
            "caption": "",
            "lyrics": "",
            "instrumental": instrumental,
            "bpm": 120,
            "keyscale": "C Major",
            "timesignature": "4",
            "duration": duration,
            "inference_steps": inference_steps,
            "shift": shift,
            "seed": seed,
            "thinking": True,
        }

        # Apply preset
        if preset_name and preset_name != "Custom":
            preset = self.get_preset(preset_name)
            if preset:
                params["caption"] = preset.get("caption", "")
                params["bpm"] = preset.get("bpm", 120)
                params["keyscale"] = preset.get("keyscale", "C Major")
                params["timesignature"] = preset.get("timesignature", "4")
                params["instrumental"] = preset.get("instrumental", True)

        # Apply mood modifier to caption
        if mood_override and mood_override != "None":
            current_caption = params["caption"]
            mood_lower = mood_override.lower()
            if current_caption:
                params["caption"] = f"{mood_lower} {current_caption}"
            else:
                params["caption"] = f"{mood_lower} instrumental music"

        # Apply explicit overrides
        if caption_override and caption_override.strip():
            params["caption"] = caption_override.strip()
        if lyrics_override and lyrics_override.strip() and not params["instrumental"]:
            params["lyrics"] = lyrics_override.strip()
        if bpm_override is not None:
            params["bpm"] = bpm_override
        if keyscale_override and keyscale_override.strip():
            params["keyscale"] = keyscale_override
        if timesignature_override and timesignature_override.strip():
            params["timesignature"] = timesignature_override

        # Safety net: if caption is still empty, fall back to a generic instrumental
        # description so the LM never receives an empty prompt (which can trigger
        # device-mismatch errors on MPS during offload).
        if not params["caption"].strip():
            params["caption"] = "instrumental electronic music"

        return params


# Module-level singleton
_manager: Optional[PresetManager] = None


def get_preset_manager(presets_path: Optional[str] = None) -> PresetManager:
    """Get or create the singleton PresetManager."""
    global _manager
    if _manager is None:
        _manager = PresetManager(presets_path)
    return _manager
