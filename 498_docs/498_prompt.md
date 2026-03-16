# Initial Instructions

Your purpose in this conversation is to help me plan step-by-step, and get started with one of the projects i am about to start working on.

First before we begin, understand that I am currently working on 2 parallel projects "latent-recommend: Music Recommendation and Similarity Search Engine Using Learned Audio Embeddings" and "neural-noise: Controllable Music Generation via Latent Space Navigation in Diffusion Transformers" for my "DSCI 441 - Statistical Machine Learning" and "DSCI 498 - Deep and Generative AI" courses respectively.

## Milestone Deliverables

Note that for both these course projects, I have to complete 3 milestones with the following key deliverables/responsibilities in each milestone:

**Milestone 0 (Proposal)** - Provide the title of the project proposal with a short description/abstract highlighting the overall goal. Also create a github repo for the project.

**Milestone 1 (Midterm)** - Expand on your problem, goals, and the next steps; organize and complete an informative ReadMe for the github repo; Create and Share a video presentation about the current state of your project and provide feedbacks on other teams' presentations.

**Milestone 2 (Final)** - Create a Poster and Presentation video for the final Project; complete code on GitHub; Create an interactive webapp illustrating your project; feedback activity.

As of now, I have completed and submitted the Milestone 0 Proposal for both these projects and I am currently about to  begin with Milestone 1, which is due right now.

I have spent some time on the DSCI 441 "latent-recommend" project and drafted the initial implementation roadmap. Now I need to spend some time on the DSCI 498 "neural-noise" project, make a step-by-step roadmap/plan and then start working on the deliverables for Milestone 1 (midterm) as its due for DSCI 498 tonight and tomorrow midnight for DSCI 441.

So remember that I'll be working on both these projects "neural-noise" and "latent-recommend" in parallel, so think about the holistic view.

## Your Goal

1. Help me implement the MicroMusicGPT baseline based on the rapid 5-step execution plan I have highlighted below.
2. Finally, help me draft the contents for the powepoint slides that I will be using to make the video presentation for Milestone 1.

---

# Context Documents

## Milestone 0 Proposals

Here are the Milestone 0 reports of for both the DSCI 441 Statistical Machine Learning and DSCI 498 DGAI project for your reference.

```
## **DSCI-441: Statistical and Machine Learning**

### **Project Milestone 0**

**Title:**   
***Music Recommendation and Similarity Search Engine Using Learned Audio Embeddings***

**Description:**  
Modern music recommendation systems (Spotify, YouTube Music, etc) overwhelmingly rely on collaborative filtering, i.e. recommending content based on what similar *users* listened to, not purely based on what the music actually *sounds like*. An alternative paradigm is **content-based recommendation**, where similarity is computed directly from the acoustic properties of the music itself. Recent advances in generative music models have produced powerful learned audio representations as a byproduct of their training. These latent vectors capture rich acoustic information like timbre, rhythm, harmony, genre, mood in a compressed form that is ideal for downstream machine learning tasks.

We aim to build a content-based music recommendation and similarity search engine by repurposing these learned latent representations of music. Rather than relying on collaborative filtering or metadata, we aim to extract compact latent vectors from a music corpus and apply classical ML techniques dimensionality reduction, unsupervised clustering, supervised classification, and nearest-neighbor retrieval to map the natural topology of musical genres and enable purely content-driven similarity search. 

Anurag Subedi  			ansc25@lehigh.edu

```

```
## **DSCI-498: Deep and Generative Learning**

### **Project Milestone 0**

**Controllable Music Generation via Latent Space Navigation in Diffusion Transformers**

State-of-the-art music generation models like Suno and Udio produce impressive results, but they are closed-source models and operate as “black-boxes” with no direct control over the internal mechanics of how sound is constructed. This prevents researchers and creators from understanding how musical attributes are encoded and how to precisely control generation beyond text prompts. ACE-Step 1.5, the current leading open-source music generation model offers a unique opportunity to investigate this question. Its architecture compresses 48kHz stereo audio into a compact 64-dimensional latent space via a 1D Variational Autoencoder (VAE), then generates audio using a 2-billion parameter Diffusion Transformer (DiT) based on text, timbre, and lyric encodings.

We aim to investigate ACE-Step 1.5's latent space as a structured, navigable manifold for music generation. The central question is: Can we discover and exploit meaningful directions in the learned latent space that correspond to interpretable musical attributes (genre, mood, key, instrumentation, energy) and use them to achieve fine-grained control over generation beyond what text prompting alone allows?


Team:
Anurag Subedi  			ansc25@lehigh.edu
Koushik Vennalakanti		kov225@lehigh.edu
```

## AceStep 1.5 - Technical Paper

As an additional context, I have also attached PDF for the technical paper of the Ace-Step 1.5 model. You will read and review this paper to understand the general nature of the foundational model we will be working with.

## Milestone 0 Professor Feedback for DGAI Project

 Please see my comment on your term project, Controllable Music Generation via Latent-Space Navigation in Diffusion Transformers:

- What would be your quantitative control metrics?
- What is your final product?

 General comment: Please address the feedback provided on your midterm report. In your midterm report, clearly state what you plan to deliver as your final product. Note that a trained model or a Jupyter notebook alone is not sufficient as a final deliverable.

## Parallel Project Context: `latent-recommend` - Current Implementation Roadmap for DSCI 441 Project

Although I haven't actually started writing any concrete code, I had spent some time planning and coming up with a strategic step-by-step plan for the DSCI 441 "latent-recommend" project. Here's the current version of the implementation roadmap for your reference:

```

### **Phase 1: The Metadata Skeleton & Bias Mitigation (Immediate Execution)**

Before processing audio, a robust, queryable metadata backbone must be established. The critical requirement of this phase is mitigating popularity bias to ensure obscure, independent tracks are represented equally alongside mainstream music.

* **Database Schema Design:** \* Initialize a relational database (SQLite for the preliminary phase, scaling to PostgreSQL with `pgvector` for embedding storage).
  * Required entities: `Track`, `Artist`, `Album`, `GenreTags`, and a dedicated `PopularityIndex` metric.
* **Data Ingestion & Stratified Sampling:**
  * Utilize APIs such as MusicBrainz (optimal for deep metadata and obscure tracks) and Last.fm (for genre tagging and popularity heuristics).
  * Implement a stratified sampling algorithm. The ingestion script must actively query for equal distributions across popularity percentiles (e.g., 20% mainstream, 40% mid-tier, 40% obscure/indie).
  * Ensure heavy sampling of specific boundary-condition genres of interest, specifically ambient soundscapes, dub-techno, and classical pieces.

### **Phase 2: Audio Acquisition & Shared Latent Extraction**

Once the metadata skeleton is populated, the pipeline must acquire the raw acoustic data and process it through the shared generative bottleneck. 
**A preliminary requirement before starting this phase will be to get the code for the Ace-Step 1.5 model up and working and explore all the available models. We should also dive deep into the DiT VAE encoder of the AceStep model and how we can capture and make sense of the latent vectors.**

* **Audio Fetching & Standardization:**
  * Programmatically fetch 30-second audio previews (via Spotify API, 7digital, or open-source datasets matching our database IDs).
  * Standardize the audio format: uniform sample rate (e.g., 44.1kHz), converted to mono, and trimmed to exact lengths.
* **Latent Embedding Generation:**
  * Pass the standardized audio arrays through the pre-trained ACE-Step 1.5 VAE encoder.
  * Extract the dense latent vectors and store them as arrays directly in the database, linked via Foreign Key to the `Track` metadata records.

### **Phase 3: Statistical Learning & Topology Mapping**

With the database populated with both metadata and latent vectors, classical SML techniques will be applied to analyze the acoustic manifold.

* **Dimensionality Reduction:**
  * Apply Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) to the latent vectors.
  * Generate 2D/3D visual mappings to evaluate if the model naturally separates distinct acoustic profiles (e.g., a dense dub-techno track vs. a sparse acoustic track) purely based on the embeddings, without metadata labels.
* **Unsupervised Clustering:**
  * Implement K-Means and Hierarchical clustering algorithms on the high-dimensional embeddings.
  * Evaluate the resulting clusters: Determine if they map to traditional metadata genres or if they discover new, purely acoustic categorizations (e.g., clustering by tempo-synced basslines or specific harmonic structures).

### **Phase 4: Retrieval Engine Implementation**

The final phase of the SML project is building the actual recommendation interface based on the generated clusters and latent geometry.

* **Similarity Search:**
  * Implement K-Nearest Neighbors (K-NN) and Cosine Similarity metrics.
  * Build a query function where a target `Track ID` is provided, and the engine retrieves the top $N$ closest vectors in the latent space.
* **Performance Evaluation:**
  * Compare the retrieved "closest" tracks against baseline collaborative filtering outputs to quantify the differences in recommendation diversity and bias mitigation.
```

## "MicroMusicGPT" (Preliminary Baseline for Milestone 1 Demo)

Before diving into the massive 2-billion parameter ACE-Step 1.5 Diffusion Transformer, I thought about building a preliminary, from-scratch autoregressive transformer called **MicroMusicGPT** inspired by the "TinyGPT"/"NanoGPT" implementation by Andrej Karpathy. This serves as a critical stepping-stone, a mechanistic baseline, and a tangible deliverable for the Milestone 1 video presentation.

While ACE-Step 1.5 operates on continuous audio latents via a VAE, MicroMusicGPT operates on discrete MIDI events. Building this helps us map out the foundational concepts of music tokenization, embedding spaces, and sequence generation, which directly informs how we will eventually manipulate the ACE-Step latent space. **CRITICAL NOTE**: This also allows us to highlight and diffrentiate between the discrete MIDI events/latents and the continuous audio latents and at the same time providing a key distinction between the delivarables in Milestone 1 and the Final Milestone.

### 1. Holistic Architecture & Modular Design

For the MicroMusicGPT, I thought of structuring the codebase as a modular Python package with the following rough directory structure:

* **`data/`**: Stores raw `.mid` files and processed tokenized datasets (numpy arrays).
* **`src/tokenizer.py`**: Handles the bidirectional translation between raw MIDI ticks/messages and our integer vocabulary.
* **`src/model.py`**: Contains the GPT transformer logic for sequence generation. Crucially, it includes `state_dict()` and serialization methods to save/load weights.
* **`train.py` & `generate.py`**: Separated execution scripts for the training loop (with validation splits and gradient monitoring) and inference (converting predicted tokens back to playable `.mid` files).
* **`checkpoints/`**: Directory for storing `.pkl` weight files at various training steps to observe *when* the model learns specific musical concepts.

### 2. The MIDI Tokenization Strategy (Event-Based)

To feed polyphonic music (chords, harmonies) and rhythm into a linear sequence model, we are using a custom 1D event vocabulary (388 tokens) rather than raw audio or standard text:

* **NOTE_ON (0-127):** Initiates a specific MIDI pitch.
* **NOTE_OFF (128-255):** Terminates a pitch (crucial for capturing duration).
* **TIME_SHIFT (256-355):** Advances the temporal clock in 10ms increments (e.g., Token 256 = 10ms, Token 257 = 20ms). This ingenious mechanic allows the model to understand simultaneous notes (chords have no time-shift between them) and rhythmic pacing.
* **Special Tokens (356-357):** BOS (Beginning of Sequence) and EOS (End of Sequence).

### 3. Training Loop & Inference Execution

* **Observable Training:** The training loop is designed to save checkpoints periodically (e.g., every 100 steps) and monitor gradient magnitudes to ensure stable learning. We hold out 10% of the MIDI data as a validation set to measure true generalization versus memorization.
* **The "Virtuoso" Inference:** For inference, the model loads a specific checkpoint and a tokenizer configuration. It can be primed with a "seed" sequence (e.g., a user playing a few notes on a MIDI keyboard), allowing it to act as an AI accompanist by autoregressively predicting the logical musical continuation and decoding it back into a standard `.mid` file.

### 4. Relevance to "neural-noise" & Professor's Feedback

* **Tangible Deliverable:** It provides a working, end-to-end model to showcase in the Milestone 1 presentation.
* **Establishing Control Metrics:** The professor asked for "quantitative control metrics." MicroMusicGPT allows us to establish baseline metrics for controllable generation. We will expand upon these evaluation concepts when we scale up to ACE-Step 1.5 in the final milestone using other audio metrics such as Frechet Audio Distance (FAD), etc.

### 5. Implementation Roadmap: MicroMusicGPT Baseline

To establish a mechanistic foundation before scaling up to the ACE-Step 1.5 DiT architecture, the first milestone is to build **MicroMusicGPT**: a lightweight, discrete-event autoregressive transformer implemented in PyTorch.

The implementation will follow a rapid 5-phase execution plan. The codebase will utilize a standard decoder-only PyTorch transformer architecture (a reference `gpt.py` script, structurally similar to Andrej Karpathy's character-level "Tiny Shakespeare" GPT, will be provided in the working directory as the baseline foundation to be modified).

#### Phase 1: Dataset Triage ("Tiny Symphony")

The target dataset is an excerpt from the MAESTRO v3.0.0 MIDI dataset. To ensure rapid training convergence while maintaining musical complexity, the dataset must be subsetted. Hence, I have decided to keep only the midi files for Ludvig Van Beethoven's pieces (i.e where the "canonical_composer" filed is "l v beethoven" in the metadata csv file).

* **Data Selection:** Extract the MIDI files strictly associated with Beethoven's pieces from the MAESTRO datasett. Filtering by a single classical composer is preferred to enforce a consistent stylistic distribution.
* **Scale:** This subsetting might yield roughly 1,000,000 musical event tokens (equivalent to the scale of the Tiny Shakespeare dataset, equating to about 6-9 hours of continuous piano).
* **Directory:** Move the selected `.mid` files into a dedicated `data/raw_midi/` directory.

#### Phase 2: The MIDI Tokenizer (Event-Based Encoding)

Standard text tokenization does not apply to polyphonic music. The agent must build a bidirectional tokenizer (`src/tokenizer.py`) using the `mido` Python library to map 2D MIDI data (pitch and time) into a 1D sequence of integers.

* **Vocabulary Definition (Size = 388):**
  * `0-127`: **NOTE_ON** (Initiates a specific MIDI pitch).
  * `128-255`: **NOTE_OFF** (Terminates a pitch).
  * `256-355`: **TIME_SHIFT** (Advances the temporal clock in 10ms bins. E.g., Token 256 = 10ms, Token 257 = 20ms). This captures both rhythm and simultaneous chord strikes.
  * `356-357`: **BOS** (Beginning of Sequence) and **EOS** (End of Sequence).
* **Batch Processing:** Parse all raw MIDI files through the encoder, concatenate the integers into a single flat sequence, and serialize it to disk as a PyTorch tensor (`data/dataset.pt`).

#### Phase 3: PyTorch Architectural Adaptation

Modify the base `gpt.py` reference script to accommodate the new musical vocabulary and prepare it for efficient GPU training.

* **Hyperparameter Configuration:** Target a lightweight footprint optimized for an Apple M2 (16GB) Chip or RTX 3060 (6GB VRAM).
  * `vocab_size = 388`
  * `block_size = 256` (yields roughly 5-8 seconds of musical context window)
  * `batch_size = 64`
  * `n_embd = 384`
  * `n_head = 6`
  * `n_layer = 6`
* **Data Loading:** Replace standard text I/O with `torch.load('data/dataset.pt')`.
* **Serialization capability:** Implement functionality to save the trained model weights via `torch.save(model.state_dict(), 'checkpoints/micromusicgpt_v1.pth')`.

#### Phase 4: Training Pipeline Execution

Execute the modified training loop to overfit the "Tiny Symphony" dataset.

* **Initialization:** Verify the starting Cross-Entropy Loss sits near the mathematical expectation of $-\ln(1/388)$ (approximately 5.96).
* **Optimization:** Utilize the `AdamW` optimizer. Monitor gradients and log the training/validation loss dynamically.
* **Checkpointing:** Save `.pth` model weights at regular intervals to allow for mechanistic inspection of the learning process over time.

#### Phase 5: Interactive UI Demo (Gradio & Synthesis)

Construct an interactive web interface (`app.py`) to demonstrate the model's generative capabilities visually and acoustically.

* **Dependencies:** Utilize `gradio`, `pretty_midi`, and `matplotlib`.
* **Inference Pipeline:**
  1. Load the trained `.pth` checkpoint.
  2. Accept a brief "seed" context (or BOS token).
  3. Autoregressively generate a sequence of ~1000 tokens.
  4. Decode the tokens back into a standard `demo.mid` file.
* **Synthesis & Visualization:** Pass the generated MIDI through `pretty_midi.Instrument.synthesize` to create a playable `.wav` audio file. Simultaneously, use `matplotlib` to plot a 2D "Piano Roll" representation, visually proving the structural coherence of the generated harmonies and rhythms.
* **Gradio Frontend:** Mount the audio player and the piano roll plot side-by-side in the Gradio app to provide a holistic, real-time evaluation dashboard.

---

# Next Steps

The Milestone 1 (midterm) is due for the DGAI project tonight. So now that you have these instructions and context, I want you to think step-by-step about the next steps for the DGAI "neural-noise" project, specifically the Milestone 1. You will think and plan about this step-by-step for implementing Milestone 1 for this project first and then later proceed to writign the actual code with my acknowledgement.
