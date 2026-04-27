"""
neural-noise Milestone 2 — DiT Renderer Microservice

FastAPI server that runs the Diffusion Transformer (DiT) + VAE decoder.
Designed to run on the M2 MacBook Air, but can run on any machine with
sufficient memory.

Usage:
    # On Mac (MPS):
    python3 dit_service.py --device mps --port 8002

    # On a CUDA machine:
    python3 dit_service.py --device cuda --port 8002
"""

import os
import sys
import time
import argparse
import logging
import base64
import io

# Resolve ACE-Step path (ACE-Step-1.5 is a sibling dir inside backend/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ACESTEP_ROOT = os.path.join(_SCRIPT_DIR, "ACE-Step-1.5")
_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "output")

if _ACESTEP_ROOT not in sys.path:
    sys.path.insert(0, _ACESTEP_ROOT)

os.makedirs(_OUTPUT_DIR, exist_ok=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class RenderRequest(BaseModel):
    """Request body for the /v1/render endpoint."""
    audio_codes: str = ""
    caption: str = ""
    lyrics: str = ""
    instrumental: bool = True
    bpm: Optional[int] = None
    duration: Optional[float] = 30.0
    keyscale: str = ""
    timesignature: str = ""
    inference_steps: int = 8
    shift: float = 3.0
    seed: int = -1
    audio_format: str = "wav"


class RenderResponse(BaseModel):
    """Response body for the /v1/render endpoint."""
    success: bool
    audio_path: Optional[str] = None
    audio_base64: Optional[str] = None
    sample_rate: int = 48000
    seed: int = -1
    render_time: float = 0.0
    time_costs: Dict[str, Any] = {}
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for the /v1/health endpoint."""
    status: str
    model: str
    device: str


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="neural-noise DiT Renderer Service",
    description="Diffusion Transformer + VAE decoder for ACE-Step 1.5 distributed pipeline",
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
_dit_handler = None
_service_config = {}


def get_dit_handler():
    """Get the initialized DiT handler."""
    if _dit_handler is None:
        raise HTTPException(status_code=503, detail="DiT not initialized")
    return _dit_handler


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if _dit_handler is not None else "not_ready",
        model=_service_config.get("dit_model", "unknown"),
        device=_service_config.get("device", "unknown"),
    )


@app.post("/v1/render", response_model=RenderResponse)
async def render(request: RenderRequest):
    """
    Render audio from pre-computed audio codes using the DiT + VAE.

    This skips the LM phase entirely — the audio_codes and metadata
    should come from the LM Composer service.
    """
    handler = get_dit_handler()
    start_time = time.time()

    try:
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        gen_params = GenerationParams(
            task_type="text2music",
            caption=request.caption,
            lyrics=request.lyrics,
            instrumental=request.instrumental,
            audio_codes=request.audio_codes,
            bpm=request.bpm,
            keyscale=request.keyscale,
            timesignature=request.timesignature,
            duration=request.duration or 30.0,
            inference_steps=request.inference_steps,
            shift=request.shift,
            seed=request.seed,
            thinking=False,  # No LM needed — codes are pre-computed
        )

        gen_config = GenerationConfig(
            batch_size=1,
            audio_format=request.audio_format,
        )

        logger.info(f"Rendering: codes_len={len(request.audio_codes)}, steps={request.inference_steps}")

        result = generate_music(
            dit_handler=handler,
            llm_handler=None,
            params=gen_params,
            config=gen_config,
            save_dir=_OUTPUT_DIR,
        )

        elapsed = time.time() - start_time

        if not result.success:
            return RenderResponse(
                success=False,
                error=result.error or result.status_message,
                render_time=elapsed,
            )

        if result.audios and len(result.audios) > 0:
            audio_info = result.audios[0]
            audio_path = audio_info.get("path", "")

            # Also encode audio as base64 for HTTP transport
            audio_b64 = None
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            return RenderResponse(
                success=True,
                audio_path=audio_path,
                audio_base64=audio_b64,
                sample_rate=audio_info.get("sample_rate", 48000),
                seed=audio_info.get("params", {}).get("seed", -1),
                render_time=elapsed,
                time_costs=result.extra_outputs.get("time_costs", {}),
            )

        return RenderResponse(
            success=False,
            error="Rendering succeeded but no audio was produced.",
            render_time=elapsed,
        )

    except Exception as e:
        logger.error(f"Rendering failed: {e}", exc_info=True)
        return RenderResponse(
            success=False,
            error=str(e),
            render_time=time.time() - start_time,
        )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def initialize_dit(device: str, dit_model: str, acestep_root: str):
    """Initialize the DiT handler."""
    global _dit_handler, _service_config

    # Apply MPS watermark if on Mac
    if device == "mps":
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    from acestep.handler import AceStepHandler

    logger.info(f"Initializing DiT: model={dit_model}, device={device}")

    _dit_handler = AceStepHandler()
    _dit_handler.initialize_service(
        project_root=acestep_root,
        config_path=dit_model,
        device=device,
        offload_to_cpu=False,  # Keep DiT hot — this is its dedicated machine
    )

    _service_config = {
        "dit_model": dit_model,
        "device": device,
    }

    logger.info("DiT service initialized successfully.")


def main():
    parser = argparse.ArgumentParser(description="neural-noise DiT Renderer Service")
    parser.add_argument("--device", default="mps", choices=["cuda", "mps", "cpu"],
                        help="Device for DiT inference")
    parser.add_argument("--dit-model", default="acestep-v15-turbo",
                        help="DiT model name")
    parser.add_argument("--acestep-root", default=_ACESTEP_ROOT,
                        help="Path to ACE-Step-1.5 root")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8002, help="Server port")
    args = parser.parse_args()

    # Initialize DiT before starting the server
    initialize_dit(
        device=args.device,
        dit_model=args.dit_model,
        acestep_root=args.acestep_root,
    )

    # Start FastAPI server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
