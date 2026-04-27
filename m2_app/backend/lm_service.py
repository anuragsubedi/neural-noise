"""
neural-noise Milestone 2 — LM Composer Microservice

FastAPI server that runs the Qwen LM planner (0.6B or 1.7B).
Designed to run on the Windows RTX 3060 machine, but can also be tested
locally on the Mac.

Usage:
    # On Windows (CUDA):
    python3 lm_service.py --device cuda --lm-model acestep-5Hz-lm-0.6B --port 8001

    # On Mac (for local testing):
    python3 lm_service.py --device mps --lm-model acestep-5Hz-lm-0.6B --port 8001
"""

import os
import sys
import time
import argparse
import logging

# Resolve ACE-Step path (ACE-Step-1.5 is a sibling dir inside backend/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ACESTEP_ROOT = os.path.join(_SCRIPT_DIR, "ACE-Step-1.5")

if _ACESTEP_ROOT not in sys.path:
    sys.path.insert(0, _ACESTEP_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ComposeRequest(BaseModel):
    """Request body for the /v1/compose endpoint."""
    caption: str = ""
    lyrics: str = ""
    instrumental: bool = True
    bpm: Optional[int] = None
    duration: Optional[float] = 30.0
    keyscale: str = ""
    timesignature: str = ""
    seed: int = -1
    lm_temperature: float = 0.85
    lm_top_p: float = 0.9


class ComposeResponse(BaseModel):
    """Response body for the /v1/compose endpoint."""
    success: bool
    audio_codes: str = ""
    cot_metadata: Dict[str, Any] = {}
    compose_time: float = 0.0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for the /v1/health endpoint."""
    status: str
    model: str
    device: str
    backend: str


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="neural-noise LM Composer Service",
    description="Qwen LM planner for ACE-Step 1.5 distributed pipeline",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_llm_handler = None
_service_config = {}


def get_llm_handler():
    """Get the initialized LLM handler."""
    if _llm_handler is None:
        raise HTTPException(status_code=503, detail="LM not initialized")
    return _llm_handler


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if _llm_handler is not None else "not_ready",
        model=_service_config.get("lm_model", "unknown"),
        device=_service_config.get("device", "unknown"),
        backend=_service_config.get("backend", "unknown"),
    )


@app.post("/v1/compose", response_model=ComposeResponse)
async def compose(request: ComposeRequest):
    """
    Run the LM planner to generate CoT metadata and audio codes.

    This is Phase 1 + Phase 2 of the ACE-Step pipeline:
      Phase 1: Generate Chain-of-Thought metadata (BPM, key, caption, etc.)
      Phase 2: Generate audio semantic codes
    """
    handler = get_llm_handler()
    start_time = time.time()

    try:
        from acestep.inference import GenerationParams, GenerationConfig

        # Build params for the LM
        params = GenerationParams(
            task_type="text2music",
            caption=request.caption,
            lyrics=request.lyrics,
            instrumental=request.instrumental,
            bpm=request.bpm,
            keyscale=request.keyscale,
            timesignature=request.timesignature,
            duration=request.duration or 30.0,
            seed=request.seed,
            thinking=True,
            lm_temperature=request.lm_temperature,
            lm_top_p=request.lm_top_p,
            use_cot_metas=True,
            use_cot_caption=True,
            use_cot_language=True,
        )

        config = GenerationConfig(
            batch_size=1,
            use_random_seed=(request.seed == -1),
        )

        # Run the LM to get CoT + audio codes
        # We call the LLM handler directly for the CoT + codes generation
        logger.info(f"Composing: caption='{request.caption[:80]}...', bpm={request.bpm}, duration={request.duration}")

        lm_result = handler.generate_with_stop_condition(
            params=params,
            config=config,
        )

        elapsed = time.time() - start_time

        # Extract results
        audio_codes = ""
        cot_metadata = {}

        if hasattr(lm_result, "audio_codes"):
            audio_codes = lm_result.audio_codes
        if hasattr(lm_result, "cot_metadata"):
            cot_metadata = lm_result.cot_metadata

        # If the result is a tuple or dict, handle accordingly
        if isinstance(lm_result, dict):
            audio_codes = lm_result.get("audio_codes", "")
            cot_metadata = lm_result.get("cot_metadata", {})
        elif isinstance(lm_result, (list, tuple)) and len(lm_result) >= 2:
            audio_codes = lm_result[0] if isinstance(lm_result[0], str) else ""
            cot_metadata = lm_result[1] if isinstance(lm_result[1], dict) else {}

        logger.info(f"Composition complete in {elapsed:.2f}s. Codes length: {len(audio_codes)}")

        return ComposeResponse(
            success=True,
            audio_codes=audio_codes,
            cot_metadata=cot_metadata,
            compose_time=elapsed,
        )

    except Exception as e:
        logger.error(f"Composition failed: {e}", exc_info=True)
        return ComposeResponse(
            success=False,
            error=str(e),
            compose_time=time.time() - start_time,
        )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def initialize_lm(device: str, lm_model: str, backend: str, checkpoint_dir: str):
    """Initialize the LLM handler."""
    global _llm_handler, _service_config

    # Apply MPS watermark if on Mac
    if device == "mps":
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    from acestep.llm_inference import LLMHandler

    logger.info(f"Initializing LM: model={lm_model}, device={device}, backend={backend}")

    _llm_handler = LLMHandler()
    _llm_handler.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=lm_model,
        backend=backend,
        device=device,
        offload_to_cpu=False,  # Keep LM hot — this is its dedicated machine
    )

    _service_config = {
        "lm_model": lm_model,
        "device": device,
        "backend": backend,
    }

    logger.info("LM service initialized successfully.")


def main():
    parser = argparse.ArgumentParser(description="neural-noise LM Composer Service")
    parser.add_argument("--device", default="mps", choices=["cuda", "mps", "cpu"],
                        help="Device for LM inference")
    parser.add_argument("--lm-model", default="acestep-5Hz-lm-0.6B",
                        help="LM model name (in checkpoints dir)")
    parser.add_argument("--backend", default="pt", choices=["pt", "vllm"],
                        help="LM inference backend")
    parser.add_argument("--checkpoint-dir", default=os.path.join(_ACESTEP_ROOT, "checkpoints"),
                        help="Path to checkpoints directory")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    args = parser.parse_args()

    # Initialize the LM before starting the server
    initialize_lm(
        device=args.device,
        lm_model=args.lm_model,
        backend=args.backend,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Start FastAPI server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
