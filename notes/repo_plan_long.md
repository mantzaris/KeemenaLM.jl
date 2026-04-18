## KeemenaLM.jl extended package summary (aligned plan)

Status note:
- the original v0.1 staged roadmap is complete through Stage 7
- this file should now be read as an architecture and scope reference, not as a pending execution checklist

### Purpose and scope
KeemenaLM.jl should make Julia capable of:
1) Training a small decoder-only transformer language model end to end.
2) Running that model for inference, generation, and minimal chat.
3) Packaging and reusing trained models as portable bundles.

The initial focus is a GPT-2 style decoder-only transformer implemented in a way the Julia community can read, trust, extend, and keep alive even if parts of the ecosystem shift over time.

KeemenaLM.jl should integrate with:
- KeemenaSubwords.jl for tokenization
- KeemenaPreprocessing.jl for text preparation

These integrations should be thin adapter points, not hard architectural dependencies for the whole package.

---

## Design principles

### 1) Ship one golden path first
v0.1 should prioritize one working workflow:
- train tiny GPT-2 with Flux on NVIDIA GPU
- save checkpoint
- save inference bundle
- load bundle from local path or artifact
- generate text
- run a REPL chat wrapper on CPU or NVIDIA GPU

Anything outside this path is secondary.

### 2) Keep core code framework-neutral
Hard rule:
- `src/core/` must not import Flux or Lux
- only `src/backends/flux/` and `src/backends/lux/` may import those frameworks

Core owns:
- config and validation
- masks and sampling
- generation and chat orchestration
- bundle schema and bundle IO
- bundle source resolution
- training container types
- checkpoint schema helpers

Backends own:
- model definitions
- forward passes
- training steps
- weight extraction/loading
- device-specific movement logic

### 3) Support backend choice without forcing parity on day one
The Julia ecosystem is fragile enough that backend-pluggability is worth the cost.

So from the start:
- public entry points accept `backend::Symbol`
- the repository contains both Flux and Lux backend folders
- the bundle format and parameter naming contract are backend-independent

But v0.1 support policy is still narrow:
- `:flux` is the supported backend for training and inference
- `:lux` has a stable repository location and API entry path, but may remain partial/stubbed until later

### 4) Bundles are the unit of reuse
Architectures live in code. Weights live in bundles. Bundles are what users load and share.

Policy:
- do not commit trained weights to git
- use `JLD2` for v0.1 weights
- allow Julia Artifacts for small official demo bundles
- do not make Artifacts the only model source

---

## Core concepts

### Architecture vs weights vs checkpoint vs bundle
- Architecture:
  - model code and config logic in KeemenaLM.jl
- Weights:
  - trained parameter arrays
- Training checkpoint:
  - local resumable training state
  - includes optimizer state, counters, and related metadata
- Inference bundle:
  - portable package for loading a model later and chatting with it
  - includes config, weights, tokenizer/preprocessing references, and metadata

### Device policy
- v0.1 inference must work on CPU
- v0.1 inference may also run on NVIDIA GPU
- v0.1 training is assumed to run on NVIDIA GPU
- non-NVIDIA GPU support is out of scope for v0.1
- CPU training is not a product target for v0.1, even if some toy tests may execute there

### Source resolution policy
Model-loading logic should resolve a source to a local bundle directory before normal load logic runs.

v0.1 sources:
- local directory
- Julia artifact

Later sources:
- direct URL
- Hugging Face-style remote repository

---

## v0.1 deliverables

### 1) GPT-2 style decoder-only transformer
- readable implementation
- correct causal masking
- one stable tensor-shape convention across backends

Recommended v0.1 shapes:
- input ids: `(sequence_length, batch_size)`
- hidden states: `(d_model, sequence_length, batch_size)`
- logits: `(vocab_size, sequence_length, batch_size)`

### 2) Generation and minimal chat
- temperature sampling
- `top_k`
- `top_p`
- EOS stopping
- optional stop sequences
- minimal in-memory chat session
- REPL-first usage, not CLI-first

### 3) Bundle save/load
v0.1 must support:
- writing:
  - `bundle.json`
  - `model_config.json`
  - `weights.jld2`
- reading:
  - bundle manifest
  - model config
  - weights dictionary
  - local and artifact bundle resolution

### 4) Checkpoint save/load
v0.1 should have a training checkpoint format distinct from inference bundles.

Minimum checkpoint contents:
- model weights
- optimizer state
- step or epoch counter
- backend symbol
- config snapshot
- RNG state if practical

### 5) Minimal backend training sanity
- a `Trainer` container in core
- Flux training step implemented
- Lux training path can remain deferred
- at least one training sanity test for the supported backend

---

## Bundle format (v0.1)

### Directory layout
- `bundle/`
  - `bundle.json`
  - `model_config.json`
  - `weights.jld2`
  - `tokenizer/` optional
  - `preprocessing/` optional
  - `README.md` optional model card

### `bundle.json`
Required fields:
- `schema_version`
- `architecture`
- `model_config_file`
- `weights_file`
- `weights_format`
- `parameter_schema`

Optional fields:
- tokenizer reference path or inline tokenizer metadata
- preprocessing reference path or inline preprocessing metadata
- distribution metadata describing source or provenance
- free-form metadata dictionary

### `weights.jld2`
For v0.1:
- store a portable `Dict{String, Any}`
- key names must follow a backend-independent GPT-2 parameter schema
- both backends should target that schema over time

---

## Repository architecture summary

- `src/core/`
  - no Flux/Lux imports
  - configs, masks, generation, bundle IO, bundle source resolution, checkpoint helpers, training containers
- `src/backends/flux/`
  - primary supported backend in v0.1
- `src/backends/lux/`
  - reserved backend path with matching architectural shape
  - may remain partial until later
- `test/unit/`
  - framework-neutral correctness tests
- `test/integration/`
  - forward-pass and generation tests
  - deterministic CPU generation tests
  - training sanity for the supported backend
  - optional NVIDIA GPU smoke tests when practical

The full scaffold is specified in `notes/repo_plan_structure.md`.

---

## Public API sketch (v0.1)

### Config and model construction
- `GPT2Config(...)`
- `validate(config)`
- `instantiate(config; backend=:flux, kwargs...)`
- `instantiate(bundle; backend=:flux, kwargs...)`

### Backend interface
- `lm_forward(model, input_token_ids; cache=nothing, is_training=false)`
- `model_config(model)`
- `extract_weights(model)`
- `load_weights!(model, weights)`
- `train_step!(trainer, input_token_ids, target_token_ids)`

### Generation and chat
- `GenerationConfig(...)`
- `generate(model, tokenizer, preprocessing, prompt_text; generation_config=...)`
- `ChatSession(model, tokenizer, preprocessing; system_prompt="", generation_config=...)`
- `chat!(session, user_text; overrides...)`

### Checkpoints and bundles
- `save_checkpoint(path, trainer, model; metadata...)`
- `load_checkpoint(path)`
- `save_bundle(path, bundle)`
- `resolve_bundle(source)`
- `load_bundle(source)`

### Published demo models
- `available_models()`
- `download_model(model_key)`

---

## Deferred roadmap

### v0.2
- Lux backend parity for the GPT-2 path
- KV cache for faster decoding
- streaming generation callbacks
- better evaluation helpers
- broader remote model sources beyond local/artifact basics

### v0.3+
- modern decoder components such as RoPE, RMSNorm, SwiGLU
- safetensors IO
- LoRA or other lightweight fine-tuning options if they serve real use cases
- ONNX only if it clearly solves a real Julia-side problem

---

## Non-goals
- Do not try to become a full transformer framework in v0.1.
- Do not promise Flux/Lux parity in v0.1.
- Do not accept large weights into git.
- Do not make Julia Artifacts the only model source.
- Do not block v0.1 on non-NVIDIA GPU support.
