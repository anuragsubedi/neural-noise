import urllib.request
import json
import os

NOTEBOOK_PATH = "MicroMusicGPT_Colab_ScaleUp.ipynb"

# Download the model code to inject into a cell
with open('src/model.py', 'r') as f:
    model_code = f.read()
    
# We will modify the config class inside the model_code for the scaled up version
scaled_config = """@dataclass
class MicroMusicGPTConfig:
    vocab_size: int = 388
    # Increased context window to 1024 for long-range classical coherency
    block_size: int = 1024 
    # Maximize batch size for A100/H100 (reduce to 32 or 16 if using T4/L4)
    batch_size: int = 64 
    # Scaled up embeddings
    n_embd: int = 768 
    # Scaled up attention heads
    n_head: int = 12 
    # Scaled up transformer layers
    n_layer: int = 12 
    dropout: float = 0.2
    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')"""

import re

# Replace the original config block
model_code = re.sub(
    r"@dataclass\nclass MicroMusicGPTConfig:.*?\ndevice:.*?$", 
    scaled_config, 
    model_code, 
    flags=re.MULTILINE | re.DOTALL
)

# Download the tokenizer code
with open('src/tokenizer.py', 'r') as f:
    tokenizer_code = f.read()
    
# Get the training loop code
with open('train.py', 'r') as f:
    train_code = f.read()

# Make the train script colab-friendly by preventing it from running if imported
train_code = train_code.replace('if __name__ == "__main__":\n    main()', '')
train_code += '\n# Start training!\n# Uncomment the line below when datasets are uploaded.\n# main()'

# Get the generate app code
with open('generate.py', 'r') as f:
    generate_code = f.read()
generate_code = generate_code.replace('if __name__ == "__main__":\n    generate_midi()', '')
generate_code += "\n# Uncomment the line below to generate MIDI after training completion.\n# generate_midi('checkpoints/micromusicgpt_v1_final.pth', 'micromusicgpt_colab_demo.mid')"


notebook = {
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "intro-markdown"
      },
      "source": [
        "# 🎵 MicroMusicGPT - Colab Pro Scale-Up\\n",
        "\\n",
        "This notebook is specifically configured to leverage high-VRAM GPUs (A100/H100) to train a scaled-up version of **MicroMusicGPT**.\\n",
        "\\n",
        "### 🧠 Why the Scale-Up?\\n",
        "Unlike text characters (e.g., Tiny Shakespeare), polyphonic MIDI music possesses complex long-range dependencies. A chord requires multiple simultaneous `NOTE_ON` events, separated spatially by `TIME_SHIFT` tokens. If the model forgets a `NOTE_OFF` event, the note sustains indefinitely (hallucination).\\n",
        "\\n",
        "By expanding the architecture from ~10M to ~100M+ parameters and doubling the `block_size` context window to 1024, the model can 'remember' long-range chord progressions and resolve melodic phrases properly.\\n",
        "\\n",
        "### 📂 Prerequisite\\n",
        "Before running the training cell, ensure you have uploaded your pre-tokenized dataset tensors to the Colab environment:\\n",
        "- Upload `data/dataset_train.pt` to `/content/data/dataset_train.pt`\\n",
        "- Upload `data/dataset_val.pt` to `/content/data/dataset_val.pt`"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "install-deps"
      },
      "outputs": [],
      "source": [
        "!pip install mido pretty_midi matplotlib torch tqdm\\n",
        "!mkdir -p data checkpoints"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tokenizer-markdown"
      },
      "source": [
        "## 1. Tokenizer Configuration\\n",
        "Here we define our custom MIDI Event Tokenizer. It maps 388 integers to NOTE_ON, NOTE_OFF, and TIME_SHIFT events."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "tokenizer-code"
      },
      "outputs": [],
      "source": [
        tokenizer_code
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "model-markdown"
      },
      "source": [
        "## 2. Scaled-up GPT Architecture\\n",
        "The `MicroMusicGPTConfig` below has been cranked up:\\n",
        "- `block_size` = 1024\\n",
        "- `n_embd` = 768\\n",
        "- `n_head` = 12\\n",
        "- `n_layer` = 12\\n",
        "\\n",
        "*(If you get Cuda Out-Of-Memory on a T4 GPU, lower the `batch_size` to 16 or 32)*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "model-code"
      },
      "outputs": [],
      "source": [
        model_code
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "train-markdown"
      },
      "source": [
        "## 3. Training Loop\\n",
        "This cell will load your uploaded `.pt` tensors and begin optimizing the transformer. We'll track both Cross-Entropy Loss and Perplexity."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "train-code"
      },
      "outputs": [],
      "source": [
        train_code
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "generate-markdown"
      },
      "source": [
        "## 4. Synthesis & Generation\\n",
        "Once training concludes, use this cell to generate a sequence using Top-K sampling and Temperature scaling to prevent runaway notes."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "generate-code"
      },
      "outputs": [],
      "source": [
        generate_code
      ]
    }
  ]
}

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully compiled standalone Colab Notebook at: {NOTEBOOK_PATH}")
