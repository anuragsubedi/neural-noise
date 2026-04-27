"""
neural-noise Milestone 2 — Unified Inference Engine

Wraps the ACE-Step 1.5 API into a clean interface for the Streamlit dashboard.
Supports two modes:
  - LOCAL:       LM + DiT both run on this machine (with CPU offloading)
  - DISTRIBUTED: LM runs on a remote machine (Windows/CUDA), DiT runs locally
"""

import os
import json
import time
import base64
import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass returned to the frontend
# ---------------------------------------------------------------------------
@dataclass
class MusicGenerationResult:
    """Result object returned to the Streamlit frontend."""
    success: bool
    audio_path: Optional[str] = None
    audio_array: Optional[np.ndarray] = None
    sample_rate: int = 48000
    seed: int = -1
    generation_time: float = 0.0
    cot_metadata: Optional[Dict[str, Any]] = None
    time_costs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # numpy arrays aren't JSON serializable
        if d.get("audio_array") is not None:
            d["audio_array"] = None
        return d


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------
class InferenceEngine:
    """
    Unified interface for ACE-Step music generation.

    Usage:
        engine = InferenceEngine(config)
        engine.initialize()
        result = engine.generate({
            "caption": "ambient techno with sweeping pads",
            "bpm": 120,
            "duration": 30,
        })
    """

    def __init__(self, config):
        """
        Args:
            config: PipelineConfig instance from backend.config
        """
        self.config = config
        self._dit_handler = None
        self._llm_handler = None
        self._initialized = False
        self._initialization_error = None
        self.status_log: List[Dict[str, str]] = []

    def _log(self, message: str, level: str = "info"):
        """Append a status message to the log (visible on the frontend)."""
        import datetime
        entry = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        }
        self.status_log.append(entry)
        if level == "error":
            logger.error(message)
        else:
            logger.info(message)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        """
        Initialize the inference pipeline based on the configured mode.
        Returns True on success, False on failure.
        """
        if self._initialized:
            return True

        self.status_log.clear()
        self._log(f"Starting initialization (mode={self.config.mode})")

        try:
            if self.config.mode == "local":
                return self._initialize_local()
            elif self.config.mode == "distributed":
                return self._initialize_distributed()
            else:
                self._initialization_error = f"Unknown mode: {self.config.mode}"
                self._log(self._initialization_error, level="error")
                return False
        except Exception as e:
            self._initialization_error = str(e)
            self._log(f"Initialization failed: {e}", level="error")
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False

    def _initialize_local(self) -> bool:
        """Initialize both LM and DiT on the local machine with CPU offloading."""
        # Apply environment variables BEFORE importing torch
        self.config.apply_environment()
        self._log("Environment configured (MPS watermark, etc.)")

        self._log("Importing ACE-Step modules...")
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        self._log("ACE-Step modules imported successfully")

        self._log(f"Loading DiT model: {self.config.dit_model} on {self.config.dit_device}...")
        self._dit_handler = AceStepHandler()
        self._dit_handler.initialize_service(
            project_root=self.config.acestep_root,
            config_path=self.config.dit_model,
            device=self.config.dit_device,
            offload_to_cpu=True,
        )
        self._log("DiT (Acoustic Renderer) loaded successfully")

        self._log(f"Loading LM model: {self.config.lm_model} on {self.config.lm_device}...")
        self._llm_handler = LLMHandler()
        self._llm_handler.initialize(
            checkpoint_dir=self.config.checkpoint_dir,
            lm_model_path=self.config.lm_model,
            backend=self.config.lm_backend,
            device=self.config.lm_device,
            offload_to_cpu=True,
        )
        self._log("LM (Composer Agent) loaded successfully")

        self._initialized = True
        self._log("Local pipeline initialized — ready to generate")
        return True

    def _initialize_distributed(self) -> bool:
        """
        In distributed mode, only the DiT runs locally.
        The LM is accessed via HTTP to the remote service.
        """
        self.config.apply_environment()
        self._log("Environment configured (MPS watermark, etc.)")

        self._log("Importing ACE-Step modules...")
        from acestep.handler import AceStepHandler
        self._log("ACE-Step modules imported successfully")

        self._log(f"Loading DiT model: {self.config.dit_model} on {self.config.dit_device}...")
        self._dit_handler = AceStepHandler()
        self._dit_handler.initialize_service(
            project_root=self.config.acestep_root,
            config_path=self.config.dit_model,
            device=self.config.dit_device,
            offload_to_cpu=False,  # No offloading needed — LM is remote
        )
        self._log("DiT (Acoustic Renderer) loaded successfully")

        self._log(f"LM will be accessed remotely at {self.config.lm_service_url}")
        self._initialized = True
        self._log("Distributed pipeline initialized — ready to generate")
        return True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def initialization_error(self) -> Optional[str]:
        return self._initialization_error

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, params: Dict[str, Any]) -> MusicGenerationResult:
        """
        Generate music from the given parameters.

        Args:
            params: Dictionary of generation parameters matching GenerationParams fields.
                Required: at least one of 'caption' or 'lyrics'
                Optional: bpm, duration, keyscale, timesignature, inference_steps,
                          shift, seed, instrumental, task_type, etc.

        Returns:
            MusicGenerationResult with audio data and metadata.
        """
        if not self._initialized:
            return MusicGenerationResult(
                success=False,
                error="Engine not initialized. Call initialize() first.",
            )

        try:
            if self.config.mode == "local":
                return self._generate_local(params)
            elif self.config.mode == "distributed":
                return self._generate_distributed(params)
            else:
                return MusicGenerationResult(
                    success=False,
                    error=f"Unknown mode: {self.config.mode}",
                )
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return MusicGenerationResult(success=False, error=str(e))

    def _generate_local(self, params: Dict[str, Any]) -> MusicGenerationResult:
        """Generate using the local pipeline (LM + DiT on same machine)."""
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        start_time = time.time()

        # Build GenerationParams from the input dict
        gen_params = GenerationParams(
            task_type=params.get("task_type", "text2music"),
            caption=params.get("caption", ""),
            lyrics=params.get("lyrics", ""),
            instrumental=params.get("instrumental", True),
            bpm=params.get("bpm"),
            keyscale=params.get("keyscale", ""),
            timesignature=params.get("timesignature", ""),
            duration=params.get("duration", self.config.default_duration),
            inference_steps=params.get("inference_steps", self.config.default_inference_steps),
            shift=params.get("shift", self.config.default_shift),
            seed=params.get("seed", -1),
            thinking=params.get("thinking", True),
        )

        gen_config = GenerationConfig(
            batch_size=params.get("batch_size", self.config.default_batch_size),
            audio_format=params.get("audio_format", self.config.default_audio_format),
        )

        # Run generation
        result = generate_music(
            dit_handler=self._dit_handler,
            llm_handler=self._llm_handler,
            params=gen_params,
            config=gen_config,
            save_dir=self.config.output_dir,
        )

        elapsed = time.time() - start_time

        if not result.success:
            return MusicGenerationResult(
                success=False,
                error=result.error or result.status_message,
                generation_time=elapsed,
            )

        # Extract the first audio result
        if result.audios and len(result.audios) > 0:
            audio_info = result.audios[0]
            audio_tensor = audio_info.get("tensor")
            audio_array = None
            if audio_tensor is not None:
                audio_array = audio_tensor.numpy()

            return MusicGenerationResult(
                success=True,
                audio_path=audio_info.get("path"),
                audio_array=audio_array,
                sample_rate=audio_info.get("sample_rate", 48000),
                seed=audio_info.get("params", {}).get("seed", -1),
                generation_time=elapsed,
                cot_metadata=result.extra_outputs.get("lm_metadata"),
                time_costs=result.extra_outputs.get("time_costs"),
            )

        return MusicGenerationResult(
            success=False,
            error="Generation succeeded but no audio was returned.",
            generation_time=elapsed,
        )

    def _generate_distributed(self, params: Dict[str, Any]) -> MusicGenerationResult:
        """
        Generate using the distributed pipeline:
          1) POST to LM service -> get audio_codes + CoT metadata
          2) Feed codes to local DiT -> get rendered audio
        """
        import requests
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        start_time = time.time()

        # Step 1: Call the remote LM service
        logger.info(f"[Distributed] Calling LM service at {self.config.lm_service_url}...")
        try:
            lm_response = requests.post(
                f"{self.config.lm_service_url}/v1/compose",
                json={
                    "caption": params.get("caption", ""),
                    "lyrics": params.get("lyrics", ""),
                    "instrumental": params.get("instrumental", True),
                    "bpm": params.get("bpm"),
                    "duration": params.get("duration", self.config.default_duration),
                    "keyscale": params.get("keyscale", ""),
                    "timesignature": params.get("timesignature", ""),
                    "seed": params.get("seed", -1),
                },
                timeout=120,
            )
            lm_response.raise_for_status()
            lm_data = lm_response.json()
        except requests.exceptions.RequestException as e:
            return MusicGenerationResult(
                success=False,
                error=f"LM service error: {e}",
                generation_time=time.time() - start_time,
            )

        if not lm_data.get("success"):
            return MusicGenerationResult(
                success=False,
                error=f"LM service returned error: {lm_data.get('error', 'Unknown')}",
                generation_time=time.time() - start_time,
            )

        audio_codes = lm_data.get("audio_codes", "")
        cot_metadata = lm_data.get("cot_metadata", {})

        logger.info(f"[Distributed] LM returned {len(audio_codes)} chars of audio codes.")

        # Step 2: Feed audio_codes to local DiT for rendering
        gen_params = GenerationParams(
            task_type="text2music",
            caption=cot_metadata.get("caption", params.get("caption", "")),
            lyrics=params.get("lyrics", ""),
            instrumental=params.get("instrumental", True),
            audio_codes=audio_codes,
            bpm=cot_metadata.get("bpm", params.get("bpm")),
            keyscale=cot_metadata.get("keyscale", params.get("keyscale", "")),
            timesignature=cot_metadata.get("timesignature", params.get("timesignature", "")),
            duration=cot_metadata.get("duration", params.get("duration", self.config.default_duration)),
            inference_steps=params.get("inference_steps", self.config.default_inference_steps),
            shift=params.get("shift", self.config.default_shift),
            seed=params.get("seed", -1),
            thinking=False,  # LM already did the thinking remotely
        )

        gen_config = GenerationConfig(
            batch_size=1,
            audio_format=params.get("audio_format", self.config.default_audio_format),
        )

        # The local DiT renders using the pre-computed audio codes
        result = generate_music(
            dit_handler=self._dit_handler,
            llm_handler=None,  # No local LM needed
            params=gen_params,
            config=gen_config,
            save_dir=self.config.output_dir,
        )

        elapsed = time.time() - start_time

        if not result.success:
            return MusicGenerationResult(
                success=False,
                error=result.error or result.status_message,
                generation_time=elapsed,
            )

        if result.audios and len(result.audios) > 0:
            audio_info = result.audios[0]
            audio_tensor = audio_info.get("tensor")
            audio_array = audio_tensor.numpy() if audio_tensor is not None else None

            return MusicGenerationResult(
                success=True,
                audio_path=audio_info.get("path"),
                audio_array=audio_array,
                sample_rate=audio_info.get("sample_rate", 48000),
                seed=audio_info.get("params", {}).get("seed", -1),
                generation_time=elapsed,
                cot_metadata=cot_metadata,
                time_costs=result.extra_outputs.get("time_costs"),
            )

        return MusicGenerationResult(
            success=False,
            error="DiT rendering succeeded but no audio was returned.",
            generation_time=elapsed,
        )

    # ------------------------------------------------------------------
    # Health Checks
    # ------------------------------------------------------------------
    def health_check(self) -> Dict[str, Any]:
        """Check the health status of all pipeline components."""
        status = {
            "mode": self.config.mode,
            "initialized": self._initialized,
            "dit_status": "unknown",
            "lm_status": "unknown",
        }

        if self.config.mode == "local":
            status["dit_status"] = "loaded" if self._dit_handler else "not_loaded"
            status["lm_status"] = "loaded" if self._llm_handler else "not_loaded"
        elif self.config.mode == "distributed":
            status["dit_status"] = "loaded" if self._dit_handler else "not_loaded"
            # Check remote LM service
            try:
                import requests
                resp = requests.get(
                    f"{self.config.lm_service_url}/v1/health",
                    timeout=5,
                )
                if resp.status_code == 200:
                    status["lm_status"] = "online"
                    status["lm_info"] = resp.json()
                else:
                    status["lm_status"] = "error"
            except Exception:
                status["lm_status"] = "offline"

        return status
