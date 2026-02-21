## KeemenaLM.jl extended package summary (revised)

### Purpose and scope
KeemenaLM.jl makes Julia capable of:
1) Training GPT-like transformer language models end to end (starting small in v0.1).
2) Running those models for inference, generation, and minimal chat.
3) Packaging and reusing trained models as portable bundles.

The initial focus is a GPT-2 style decoder-only transformer implemented in a way the community can read, trust, and extend.

KeemenaLM.jl is expected to integrate with:
- KeemenaSubwords.jl for tokenization
- KeemenaPreprocessing.jl for text normalization/preprocessing

In v0.1, the integration points exist as simple adapter functions so downstream users can supply either Keemena components or compatible alternatives.

---

## Design principles (practical and enforceable)

### 1) Ship the golden path first
v0.1 must prioritize a single, working workflow:
- train tiny GPT-2 -> save bundle -> load bundle -> generate -> chat wrapper

Anything not required for that path is deferred or implemented as a minimal stub.

### 2) Keep core code framework-neutral
Hard rule:
- `src/core/` must not `using Flux` or `using Lux`.
- Only `src/backends/flux/` and `src/backends/lux/` can import those frameworks.

Core contains:
- configs and validation
- masks and attention helpers (pure functions)
- sampling and stopping logic
- generation and chat orchestration
- bundle schema and IO

Backends contain:
- model definitions
- weight extraction/loading
- training steps

Core calls backends only through a small set of functions (multiple dispatch), such as:
- `lm_forward(model, input_token_ids; ...)`
- `extract_weights(model)`
- `load_weights!(model, weights)`
- `train_step!(trainer, batch)`

This structure keeps the codebase ready for optional dependencies or backend split later, without introducing complex abstractions now.

### 3) Bundles are the unit of reuse
Architectures live in code. Weights live in bundles. Bundles are what users share and load.

Policy:
- Do not commit large weight files to git.
- v0.1 uses JLD2 for weights (acceptable early on).
- Plan for safetensors later via an adapter layer.

---

## Core concepts

### Architecture vs weights vs bundle
- Architecture:
  - the model definition and configuration logic
  - lives in KeemenaLM.jl source code
- Weights:
  - trained parameters for a specific model instance
  - stored outside git, inside a bundle
- Bundle:
  - directory layout capturing config, weights, tokenizer, preprocessing, and metadata
  - the portable unit used by inference and training resumption

---

## v0.1 deliverables

### 1) GPT-2 style decoder-only transformer
- Readable implementation (not maximally optimized)
- Correct causal masking
- Consistent tensor shape conventions across backends

Recommended v0.1 shape convention:
- input token ids: (sequence_length, batch_size)
- hidden states: (d_model, sequence_length, batch_size)
- logits: (vocab_size, sequence_length, batch_size)

### 2) Generation and minimal chat
- Sampling:
  - temperature
  - top_k
  - top_p (nucleus)
- Stop logic:
  - EOS token (token-level)
  - optional stop sequences (can start as token-level; text-level is acceptable for chat convenience)
- Minimal chat wrapper:
  - message history stored in memory
  - simple prompt formatting (explicitly naive in v0.1)

### 3) Bundle save/load
v0.1 must support:
- Writing:
  - JSON config
  - JSON bundle manifest (schema version)
  - JLD2 weights
- Reading:
  - bundle manifest
  - model config
  - weights dictionary

Tokenizer and preprocessing:
- Schema must support including specs or references.
- v0.1 may store placeholders and add concrete adapters as the upstream packages stabilize.

### 4) Minimal training step sanity
- A minimal `Trainer` container in core
- Backend-specific `train_step!` implementations
- One sanity test per backend:
  - forward pass produces finite loss
  - one or a few steps update parameters (and ideally reduce loss on a trivial batch)

---

## Bundle format (v0.1)

### Directory layout
Recommended bundle directory layout:

- `bundle/`
  - `bundle.json`           (required)
  - `model_config.json`     (required)
  - `weights.jld2`          (required in v0.1)
  - `tokenizer/`            (optional in v0.1)
  - `preprocessing/`        (optional in v0.1)
  - `README.md`             (optional but recommended model card)

### `bundle.json` (manifest)
Load-bearing fields:
- `schema_version` (integer)
- `architecture` (string, for v0.1: "gpt2")
- `model_config_file` (relative path)
- `weights_file` (relative path)
- `weights_format` (v0.1: "jld2")
- optional pointers for tokenizer and preprocessing
- optional metadata dictionary

### `model_config.json`
For v0.1:
- serialized `GPT2Config` (vocab size, context length, layers, heads, etc.)
- includes token ids for BOS/EOS policy if relevant

### `weights.jld2`
For v0.1:
- a portable `Dict{String, Any}` containing arrays
- backends implement mapping between model parameters and this dictionary
- future: add a `weights_safetensors.jl` adapter without changing the bundle schema

---

## Repository architecture and module layout (summary)

- `src/core/`
  - no Flux/Lux imports
  - configs, masks, generation, bundle IO, training containers
- `src/backends/flux/`
  - Flux model + weights mapping + train step
- `src/backends/lux/`
  - Lux model + weights mapping + train step
- `test/unit/`
  - masks, sampling, stopping, bundle roundtrip
- `test/integration/`
  - forward pass shape checks (Flux/Lux)
  - generation determinism (CPU)
  - training step sanity

The full scaffold is specified in `notes/repo_plan_structure.md`.

---

## Public API sketch (v0.1)

### Config
- `GPT2Config(...)`
- `validate(config)`

### Model construction
- `instantiate(config; backend = :flux or :lux, ...)`
- `instantiate(bundle; backend = :flux or :lux, ...)`

### Forward
- Backend interface used by core:
  - `lm_forward(model, input_token_ids; cache = nothing, is_training = false)`
  - `model_config(model)`

### Generation and chat
- `GenerationConfig(...)`
- `generate(model, tokenizer, preprocessing, prompt_text; generation_config = ...)`
- `ChatSession(model, tokenizer, preprocessing; system_prompt = "", generation_config = ...)`
- `chat!(session, user_text; overrides...)`

### Bundles
- `save_bundle(path, bundle)`
- `load_bundle(path)`

---

## Deferred roadmap (explicit)

### v0.2
- KV cache for efficient decoding (core types + backend support)
- streaming generation callbacks
- better evaluation utilities (perplexity, token accuracy)
- training checkpointing (optimizer state, RNG, step counters)

### v0.3+
- modern decoder components (RoPE, RMSNorm, SwiGLU)
- safetensors IO
- optional ONNX export/inference (only if it serves real use cases)
- fine-tuning utilities (LoRA) if desired

---

## Non-goals (keep the scope healthy)
- Do not try to become a full transformer framework in v0.1.
- Do not accept large weights into the git repository.
- Do not add ONNX support until core training/inference/bundles are stable.