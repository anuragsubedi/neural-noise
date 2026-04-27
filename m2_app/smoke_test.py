"""
neural-noise Milestone 2 — Smoke Test

Quick end-to-end test that verifies the inference pipeline works.
Generates a 10-second ambient clip and validates the output.

Usage:
    cd neural-noise
    python3 m2_app/smoke_test.py
"""

import os
import sys
import time

# Add paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

# Apply MPS watermark before any torch imports
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def run_smoke_test():
    print("=" * 60)
    print("neural-noise Milestone 2 — Smoke Test")
    print("=" * 60)

    # Step 1: Test config
    print("\n[1/5] Testing configuration...")
    from backend.config import PipelineConfig, get_config
    config = get_config()
    print(f"  ACE-Step root: {config.acestep_root}")
    print(f"  Checkpoints: {config.checkpoint_dir}")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Mode: {config.mode}")

    assert os.path.exists(config.checkpoint_dir), f"Checkpoints not found at {config.checkpoint_dir}"
    print("  ✓ Config OK")

    # Step 2: Test preset manager
    print("\n[2/5] Testing preset manager...")
    from components.preset_manager import get_preset_manager
    pm = get_preset_manager()
    print(f"  Available presets: {pm.preset_names}")
    print(f"  Available moods: {pm.moods}")
    print(f"  Available keys: {len(pm.keys)} keys")

    test_params = pm.build_generation_params(
        preset_name="Ambient Electronica",
        duration=10.0,
        inference_steps=4,  # Minimum for speed
    )
    print(f"  Test params: caption='{test_params['caption'][:50]}...', bpm={test_params['bpm']}")
    print("  ✓ Presets OK")

    # Step 3: Test audio utilities
    print("\n[3/5] Testing audio utilities...")
    from utils.audio_processing import list_generated_audio
    existing = list_generated_audio(config.output_dir)
    print(f"  Existing audio files in output: {len(existing)}")
    print("  ✓ Audio utils OK")

    # Step 4: Test visualization
    print("\n[4/5] Testing visualization...")
    import numpy as np
    from components.waveform_viz import create_waveform_figure, create_spectrogram_figure

    dummy_audio = np.random.randn(2, 48000).astype(np.float32) * 0.1
    fig_wave = create_waveform_figure(dummy_audio, 48000, title="Smoke Test Waveform")
    fig_spec = create_spectrogram_figure(dummy_audio, 48000, title="Smoke Test Spectrogram")
    print("  ✓ Visualizations OK")

    # Step 5: Test inference engine initialization
    print("\n[5/5] Testing inference engine (local mode)...")
    from backend.inference_engine import InferenceEngine

    engine = InferenceEngine(config)
    print("  Initializing pipeline (this may take 1-2 minutes)...")

    start = time.time()
    success = engine.initialize()
    init_time = time.time() - start

    if not success:
        print(f"  ✗ Initialization failed: {engine.initialization_error}")
        print("\n  This is expected if you haven't installed ACE-Step dependencies.")
        print("  The UI components are still functional — you can test them with:")
        print("    streamlit run m2_app/app.py")
        return False

    print(f"  ✓ Engine initialized in {init_time:.1f}s")

    # Generate a short clip
    print("\n  Generating 10s test clip...")
    gen_start = time.time()
    result = engine.generate({
        "caption": "ambient electronic pad with gentle reverb",
        "bpm": 90,
        "keyscale": "C Major",
        "duration": 10.0,
        "inference_steps": 4,
        "shift": 3.0,
        "instrumental": True,
    })
    gen_time = time.time() - gen_start

    if result.success:
        print(f"  ✓ Generation complete in {gen_time:.1f}s")
        print(f"  ✓ Audio saved to: {result.audio_path}")
        print(f"  ✓ Sample rate: {result.sample_rate}")
        print(f"  ✓ Seed: {result.seed}")

        # Verify the file exists and has content
        if result.audio_path and os.path.exists(result.audio_path):
            size = os.path.getsize(result.audio_path)
            print(f"  ✓ File size: {size / 1024:.0f} KB")

            # Compute stats
            audio_data, sr = None, 0
            try:
                from utils.audio_processing import load_audio, get_audio_stats
                audio_data, sr = load_audio(result.audio_path)
                if audio_data is not None:
                    stats = get_audio_stats(audio_data, sr)
                    print(f"  ✓ Audio stats: {stats}")
            except Exception as e:
                print(f"  ⚠ Stats computation failed: {e}")
        else:
            print("  ✗ Audio file not found!")
            return False
    else:
        print(f"  ✗ Generation failed: {result.error}")
        return False

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
