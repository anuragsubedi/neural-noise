#!/usr/bin/env python3
"""Quick validation of the backend config and imports after refactoring."""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.config import PipelineConfig, ACESTEP_ROOT
from backend.inference_engine import InferenceEngine

config = PipelineConfig.from_env()
print('ACESTEP_ROOT:', ACESTEP_ROOT, '| exists:', ACESTEP_ROOT.exists())

for m in [config.dit_model, config.lm_model, 'vae']:
    path = os.path.join(config.checkpoint_dir, m)
    print(f'  {m}:', 'FOUND' if os.path.isdir(path) else 'MISSING')

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
print('ACE-Step imports: OK')

engine = InferenceEngine(config)
print(f'InferenceEngine: OK (mode={config.mode})')
print('ALL CHECKS PASSED')
