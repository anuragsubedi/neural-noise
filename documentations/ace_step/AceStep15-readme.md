# ACE-Step 1.5

[Github](https://github.com/ace-step/ACE-Step-1.5#ace-step-15)

# Pushing the Boundaries of Open-Source Music Generation

## 📝 Abstract

[abstract](https://github.com/ace-step/ACE-Step-1.5#-abstract)

🚀 We present ACE-Step v1.5, a highly efficient open-source music foundation model that brings commercial-grade generation to consumer hardware. On commonly used evaluation metrics, ACE-Step v1.5 achieves quality beyond most commercial music models while remaining extremely fast—under 2 seconds per full song on an A100 and under 10 seconds on an RTX 3090. The model runs locally with less than 4GB of VRAM, and supports lightweight personalization: users can train a LoRA from just a few songs to capture their own style.

🌉 At its core lies a novel hybrid architecture where the Language Model (LM) functions as an omni-capable planner: it transforms simple user queries into comprehensive song blueprints—scaling from short loops to 10-minute compositions—while synthesizing metadata, lyrics, and captions via Chain-of-Thought to guide the Diffusion Transformer (DiT). ⚡ Uniquely, this alignment is achieved through intrinsic reinforcement learning relying solely on the model's internal mechanisms, thereby eliminating the biases inherent in external reward models or human preferences. 🎚️

🔮 Beyond standard synthesis, ACE-Step v1.5 unifies precise stylistic control with versatile editing capabilities—such as cover generation, repainting, and vocal-to-BGM conversion—while maintaining strict adherence to prompts across 50+ languages. This paves the way for powerful tools that seamlessly integrate into the creative workflows of music artists, producers, and content creators. 🎸

### ⚡ Performance

[](https://github.com/ace-step/ACE-Step-1.5#-performance)

- ✅ **Ultra-Fast Generation** — Under 2s per full song on A100, under 10s on RTX 3090 (0.5s to 10s on A100 depending on think mode & diffusion steps)
- ✅ **Flexible Duration** — Supports 10 seconds to 10 minutes (600s) audio generation
- ✅ **Batch Generation** — Generate up to 8 songs simultaneously

### 🎵 Generation Quality

[](https://github.com/ace-step/ACE-Step-1.5#-generation-quality)

- ✅ **Commercial-Grade Output** — Quality beyond most commercial music models (between Suno v4.5 and Suno v5)
- ✅ **Rich Style Support** — 1000+ instruments and styles with fine-grained timbre description
- ✅ **Multi-Language Lyrics** — Supports 50+ languages with lyrics prompt for structure & style control

### 🎛️ Versatility & Control

[](https://github.com/ace-step/ACE-Step-1.5#%EF%B8%8F-versatility--control)

|Feature|Description|
|---|---|
|✅ Reference Audio Input|Use reference audio to guide generation style|
|✅ Cover Generation|Create covers from existing audio|
|✅ Repaint & Edit|Selective local audio editing and regeneration|
|✅ Track Separation|Separate audio into individual stems|
|✅ Multi-Track Generation|Add layers like Suno Studio's "Add Layer" feature|
|✅ Vocal2BGM|Auto-generate accompaniment for vocal tracks|
|✅ Metadata Control|Control duration, BPM, key/scale, time signature|
|✅ Simple Mode|Generate full songs from simple descriptions|
|✅ Query Rewriting|Auto LM expansion of tags and lyrics|
|✅ Audio Understanding|Extract BPM, key/scale, time signature & caption from audio|
|✅ LRC Generation|Auto-generate lyric timestamps for generated music|
|✅ LoRA Training|One-click annotation & training in Gradio. 8 songs, 1 hour on 3090 (12GB VRAM)|
|✅ Quality Scoring|Automatic quality assessment for generated audio|

### 💡 Which Model Should I Choose?

[](https://github.com/ace-step/ACE-Step-1.5#-which-model-should-i-choose)

| Your GPU VRAM | Recommended DiT                               | Recommended LM Model           | Backend | Notes                                                        |
| ------------- | --------------------------------------------- | ------------------------------ | ------- | ------------------------------------------------------------ |
| **≤6GB**      | 2B turbo                                      | None (DiT only)                | —       | LM disabled by default; INT8 quantization + full CPU offload |
| **6-8GB**     | 2B turbo                                      | `acestep-5Hz-lm-0.6B`          | `pt`    | Lightweight LM with PyTorch backend                          |
| **8-16GB**    | 2B turbo/sft                                  | `acestep-5Hz-lm-0.6B` / `1.7B` | `vllm`  | 0.6B for 8-12GB, 1.7B for 12-16GB                            |
| **16-20GB**   | 2B sft or XL turbo                            | `acestep-5Hz-lm-1.7B`          | `vllm`  | XL requires CPU offload below 20GB                           |
| **20-24GB**   | XL turbo/sft                                  | `acestep-5Hz-lm-1.7B`          | `vllm`  | XL fits without offload; 4B LM available                     |
| **≥24GB**     | XL sft (or xl-base for extract/lego/complete) | `acestep-5Hz-lm-4B`            | `vllm`  | Best quality, all models fit without offload                 |

> **XL (4B) models** (`acestep-v15-xl-*`) offer higher audio quality with ~9GB VRAM for weights (vs ~4.7GB for 2B). They require ≥12GB VRAM (with offload + quantization) or ≥20GB (without offload). All LM models are fully compatible with XL.

The UI automatically selects the best configuration for your GPU. All settings (LM model, backend, offloading, quantization) are tier-aware and pre-configured.

## 🚀 Launch Scripts

[](https://github.com/ace-step/ACE-Step-1.5#-launch-scripts)

Ready-to-use launch scripts for all platforms with auto environment detection, update checking, and dependency installation.

| Platform           | Scripts                                                 | Backend             |
| ------------------ | ------------------------------------------------------- | ------------------- |
| **Windows**        | `start_gradio_ui.bat`, `start_api_server.bat`           | CUDA                |
| **Windows (ROCm)** | `start_gradio_ui_rocm.bat`, `start_api_server_rocm.bat` | AMD ROCm            |
| **Linux**          | `start_gradio_ui.sh`, `start_api_server.sh`             | CUDA                |
| **macOS**          | `start_gradio_ui_macos.sh`, `start_api_server_macos.sh` | MLX (Apple Silicon) |

```shell
# Windows
start_gradio_ui.bat

# Linux
chmod +x start_gradio_ui.sh && ./start_gradio_ui.sh

# macOS (Apple Silicon)
chmod +x start_gradio_ui_macos.sh && ./start_gradio_ui_macos.sh
```

### ⚙️ Customizing Launch Settings

[](https://github.com/ace-step/ACE-Step-1.5#%EF%B8%8F-customizing-launch-settings)

**Recommended:** Create a `.env` file to customize models, ports, and other settings. Your `.env` configuration will survive repository updates.

```shell
# Copy the example file
cp .env.example .env

# Edit with your preferred settings
# Examples in .env:
ACESTEP_CONFIG_PATH=acestep-v15-turbo
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
PORT=7860
LANGUAGE=en
```


## 📚 Documentation

[](https://github.com/ace-step/ACE-Step-1.5#-documentation)

### Usage Guides

[](https://github.com/ace-step/ACE-Step-1.5#usage-guides)

|Method|Description|Documentation|
|---|---|---|
|🖥️ **Gradio Web UI**|Interactive web interface for music generation|[Guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GRADIO_GUIDE.md)|
|🎚️ **Studio UI**|Optional HTML frontend (DAW-like)|[Guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/studio.md)|
|🎛️ **VST3 Plugin**|Standalone VST3 plugin (C++/GGML) for DAW integration|[acestep.vst3](https://github.com/ace-step/acestep.vst3)|
|🐍 **Python API**|Programmatic access for integration|[Guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md)|
|🌐 **REST API**|HTTP-based async API for services|[Guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md)|
|⌨️ **CLI**|Interactive wizard and configuration|[Guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/CLI.md)|

### Setup & Configuration

[](https://github.com/ace-step/ACE-Step-1.5#setup--configuration)

|Topic|Documentation|
|---|---|
|📦 Installation (all platforms)|[English](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INSTALL.md) \| [中文](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/zh/INSTALL.md) \| [日本語](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/ja/INSTALL.md)|
|🎮 GPU Compatibility|[English](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GPU_COMPATIBILITY.md) \| [中文](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/zh/GPU_COMPATIBILITY.md) \| [日本語](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/ja/GPU_COMPATIBILITY.md)|
|🔧 GPU Troubleshooting|[English](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GPU_TROUBLESHOOTING.md)|
|🔬 Benchmark & Profiling|[English](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/BENCHMARK.md) \| [中文](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/zh/BENCHMARK.md)|

### Docs

[](https://github.com/ace-step/ACE-Step-1.5#multi-language-docs)

| Language     | API                                                                       | Gradio                                                                             | Inference                                                                       | Tutorial                                                                       | LoRA Training                                                                                | Install                                                                       | Benchmark                                                                       |
| ------------ | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 🇺🇸 English | [Link](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md) | [Link](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GRADIO_GUIDE.md) | [Link](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md) | [Link](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md) | [Link](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md) | [Link](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INSTALL.md) | [Link](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/BENCHMARK.md) |


## 📖 Tutorial

[](https://github.com/ace-step/ACE-Step-1.5#-tutorial)

**🎯 Must Read:** Comprehensive guide to ACE-Step 1.5's design philosophy and usage methods.

| Language     | Link                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------ |
| 🇺🇸 English | [English Tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md) |


This tutorial covers: mental models and design philosophy, model architecture and selection, input control (text and audio), inference hyperparameters, random factors and optimization strategies.

## 🔨 Train

[](https://github.com/ace-step/ACE-Step-1.5#-train)

📖 **LoRA Training Tutorial** — step-by-step guide covering data preparation, annotation, preprocessing, and training:

| Language     | Link                                                                                                           |     |
| ------------ | -------------------------------------------------------------------------------------------------------------- | --- |
| 🇺🇸 English | [LoRA Training Tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md) |     |


See also the **LoRA Training** tab in Gradio UI for one-click training, or [Gradio Guide - LoRA Training](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GRADIO_GUIDE.md#lora-training) for UI reference.

🔧 **Advanced Training with [Side-Step](https://github.com/koda-dernet/Side-Step)** — CLI-based training toolkit with corrected timestep sampling, LoKR adapters, VRAM optimization, gradient sensitivity analysis, and more. See the [Side-Step documentation](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/sidestep/Getting%20Started.md).

## 🏗️ Architecture

[](https://github.com/ace-step/ACE-Step-1.5#%EF%B8%8F-architecture)

[![ACE-Step Framework](https://github.com/ace-step/ACE-Step-1.5/raw/main/assets/ACE-Step_framework.png)](https://github.com/ace-step/ACE-Step-1.5/blob/main/assets/ACE-Step_framework.png)
## 🦁 Model Zoo

[](https://github.com/ace-step/ACE-Step-1.5#-model-zoo)

[![Model Zoo](https://github.com/ace-step/ACE-Step-1.5/raw/main/assets/model_zoo.png)](https://github.com/ace-step/ACE-Step-1.5/blob/main/assets/model_zoo.png)

### DiT Models

[](https://github.com/ace-step/ACE-Step-1.5#dit-models)

|DiT Model|Pre-Training|SFT|RL|CFG|Step|Refer audio|Text2Music|Cover|Repaint|Extract|Lego|Complete|Quality|Diversity|Fine-Tunability|Hugging Face|
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
|`acestep-v15-base`|✅|❌|❌|✅|50|✅|✅|✅|✅|✅|✅|✅|Medium|High|Easy|[Link](https://huggingface.co/ACE-Step/acestep-v15-base)|
|`acestep-v15-sft`|✅|✅|❌|✅|50|✅|✅|✅|✅|❌|❌|❌|High|Medium|Easy|[Link](https://huggingface.co/ACE-Step/acestep-v15-sft)|
|`acestep-v15-turbo`|✅|✅|❌|❌|8|✅|✅|✅|✅|❌|❌|❌|Very High|Medium|Medium|[Link](https://huggingface.co/ACE-Step/Ace-Step1.5)|

### XL (4B) DiT Models

[](https://github.com/ace-step/ACE-Step-1.5#xl-4b-dit-models)

> XL models use a larger 4B-parameter DiT decoder (~9GB bf16) for higher audio quality. They require ≥12GB VRAM (with offload + quantization) or ≥20GB (without offload). All LM models are fully compatible.

|DiT Model|Pre-Training|SFT|RL|CFG|Step|Refer audio|Text2Music|Cover|Repaint|Extract|Lego|Complete|Quality|Diversity|Fine-Tunability|Hugging Face|
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
|`acestep-v15-xl-base`|✅|❌|❌|✅|50|✅|✅|✅|✅|✅|✅|✅|High|High|Easy|[Link](https://huggingface.co/ACE-Step/acestep-v15-xl-base)|
|`acestep-v15-xl-sft`|✅|✅|❌|✅|50|✅|✅|✅|✅|❌|❌|❌|Very High|Medium|Easy|[Link](https://huggingface.co/ACE-Step/acestep-v15-xl-sft)|
|`acestep-v15-xl-turbo`|✅|✅|❌|❌|8|✅|✅|✅|✅|❌|❌|❌|Very High|Medium|Medium|[Link](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo)|

### LM Models

[](https://github.com/ace-step/ACE-Step-1.5#lm-models)

|LM Model|Pretrain from|Pre-Training|SFT|RL|CoT metas|Query rewrite|Audio Understanding|Composition Capability|Copy Melody|Hugging Face|
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
|`acestep-5Hz-lm-0.6B`|Qwen3-0.6B|✅|✅|✅|✅|✅|Medium|Medium|Weak|✅|
|`acestep-5Hz-lm-1.7B`|Qwen3-1.7B|✅|✅|✅|✅|✅|Medium|Medium|Medium|✅|
|`acestep-5Hz-lm-4B`|Qwen3-4B|✅|✅|✅|✅|✅|Strong|Strong|Strong|✅|

## 🔬 Benchmark

[](https://github.com/ace-step/ACE-Step-1.5#-benchmark)

ACE-Step 1.5 includes `profile_inference.py`, a profiling & benchmarking tool that measures LLM, DiT, and VAE timing across devices and configurations.

```shell
python profile_inference.py                        # Single-run profile
python profile_inference.py --mode benchmark       # Configuration matrix
```




### Which Model Should I Choose?

| Your GPU VRAM | Recommended DiT | Recommended LM Model           | Backend | Notes                                                        |
| ------------- | --------------- | ------------------------------ | ------- | ------------------------------------------------------------ |
| **≤6GB**      | 2B turbo        | None (DiT only)                | —       | LM disabled by default; INT8 quantization + full CPU offload |
| **6-8GB**     | 2B turbo        | `acestep-5Hz-lm-0.6B`          | `pt`    | Lightweight LM with PyTorch backend                          |
| **8-16GB**    | 2B turbo/sft    | `acestep-5Hz-lm-0.6B` / `1.7B` | `vllm`  | 0.6B for 8-12GB, 1.7B for 12-16GB                            |
