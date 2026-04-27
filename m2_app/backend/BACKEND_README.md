# neural-noise Backend

Distributed inference backend for ACE-Step 1.5 music generation.

---

## Setup (From Scratch)

### 1. Create Virtual Environment

```bash
cd neural-noise/m2_app/backend
python3 -m venv acestep_env
source acestep_env/bin/activate        # macOS / Linux
# acestep_env\Scripts\activate         # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

# nano-vllm (Windows/Linux only — skip on macOS):
pip install -e ACE-Step-1.5/acestep/third_parts/nano-vllm
```

### 3. Download Model Checkpoints

```bash
cd ACE-Step-1.5

# DiT, Embedder, VAE:
huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir checkpoints

# LM (0.6B):
huggingface-cli download ACE-Step/acestep-5Hz-lm-0.6B --local-dir checkpoints/acestep-5Hz-lm-0.6B
```

After download, `checkpoints/` should contain:

```
checkpoints/
├── acestep-v15-turbo/       # DiT model
├── acestep-5Hz-lm-0.6B/    # Qwen LM 0.6B
├── vae/                     # VAE decoder
└── Qwen3-Embedding-0.6B/   # Text embedder
```

### 4. (Optional) Pre-download the CLAP scoring model

The **Metrics** page scores generated audio against its caption using
LAION-CLAP. The first visit to that page lazily downloads the model into the
HuggingFace cache (`~/.cache/huggingface/hub/`). To pre-download it before a
demo (so the first metrics view is instant), run with the venv activated:

```bash
python -c "from transformers import ClapModel, ClapProcessor; \
           ClapModel.from_pretrained('laion/clap-htsat-unfused'); \
           ClapProcessor.from_pretrained('laion/clap-htsat-unfused')"
```

| Asset                          | Size     | Cache location              |
| ------------------------------ | -------- | --------------------------- |
| `laion/clap-htsat-unfused`     | ~614 MB  | `~/.cache/huggingface/hub/` |

No additional pip packages are required — CLAP ships with the already-pinned
`transformers` package.

---

## Running

> **Important:** Always launch with `python3 -m streamlit`, not the bare `streamlit` command.

### Local Mode (Single Machine)

Both LM and DiT run on the same machine with CPU offloading.

```bash
cd neural-noise
m2_app/backend/acestep_env/bin/python3 -m streamlit run m2_app/app.py
```

Then: Navigate to **Generate** → Select **local** → Click **Initialize Pipeline**.

### Distributed Mode (Two Machines)

LM on a Windows/CUDA machine, DiT on Mac/MPS. They communicate over HTTP.

**On Windows (LM service):**

```bash
cd neural-noise/m2_app/backend
acestep_env\Scripts\python lm_service.py --device cuda --port 8001 --host 0.0.0.0
```

**On Mac (Streamlit + DiT):**

```bash
cd neural-noise
export NN_MODE=distributed
export NN_LM_URL=http://<WINDOWS_IP>:8001
m2_app/backend/acestep_env/bin/python3 -m streamlit run m2_app/app.py
```

---

## Directory Structure

```
m2_app/backend/
├── ACE-Step-1.5/
│   ├── acestep/                 # ACE-Step Python package (the only code we use)
│   └── checkpoints/             # Model weights (not in Git)
├── acestep_env/                 # Virtual environment (not in Git)
├── requirements.txt             # All dependencies (unified)
├── config.py                    # Pipeline configuration
├── inference_engine.py          # Unified inference wrapper
├── lm_service.py                # FastAPI — LM Composer microservice
├── dit_service.py               # FastAPI — DiT Renderer microservice
└── BACKEND_README.md            # This file
```

## Environment Variables

| Variable          | Default                   | Description                   |
| ----------------- | ------------------------- | ----------------------------- |
| `NN_MODE`       | `local`                 | `local` or `distributed`  |
| `NN_DIT_DEVICE` | `mps`                   | `mps`, `cuda`, or `cpu` |
| `NN_LM_DEVICE`  | `mps`                   | `mps`, `cuda`, or `cpu` |
| `NN_DIT_MODEL`  | `acestep-v15-turbo`     | DiT model name                |
| `NN_LM_MODEL`   | `acestep-5Hz-lm-0.6B`   | LM model name                 |
| `NN_LM_URL`     | `http://localhost:8001` | Remote LM service URL         |
| `NN_DIT_URL`    | `http://localhost:8002` | Remote DiT service URL        |

## Validation

```bash
cd neural-noise
m2_app/backend/acestep_env/bin/python3 -c "
import sys; sys.path.insert(0, 'm2_app')
from backend.config import PipelineConfig, ACESTEP_ROOT
from backend.inference_engine import InferenceEngine
import os

config = PipelineConfig.from_env()
print('ACESTEP_ROOT:', ACESTEP_ROOT, '| exists:', ACESTEP_ROOT.exists())
for m in [config.dit_model, config.lm_model, 'vae']:
    print(f'  {m}:', 'FOUND' if os.path.isdir(os.path.join(config.checkpoint_dir, m)) else 'MISSING')

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
print('ACE-Step imports: OK')
print('ALL CHECKS PASSED')
"
```
