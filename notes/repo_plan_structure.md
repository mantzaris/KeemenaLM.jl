## KeemenaLM.jl repository scaffold and boundaries (aligned plan)

Status note:
- the original v0.1 staged roadmap is complete through Stage 7
- this file should now be read as a structure and boundary reference, not as a pending execution checklist

This note defines the concrete repository scaffold for KeemenaLM.jl and the boundary rules that keep v0.1 shippable. It is a planning document: it specifies folders, responsibilities, and required interfaces, but not full implementations.

The v0.1 priority is a single golden path:
- train a tiny GPT-2 style decoder-only model with Flux on NVIDIA GPU
- save a resumable training checkpoint
- save a portable inference bundle
- load the bundle from a local directory or Julia artifact
- generate text and run a minimal REPL chat loop on CPU or NVIDIA GPU

Everything else is deferred unless it directly supports this path.

---

## Non-negotiable constraints

### Scope control
- v0.1 ships one architecture: GPT-2 style decoder-only transformer
- prefer correctness and readability over optimization
- REPL-first UX is the target, not CLI-first UX
- KV cache and streaming are deferred unless they do not threaten v0.1 delivery

### Backend boundaries
- `src/core/` must not import Flux or Lux
- only `src/backends/flux/` and `src/backends/lux/` may import those frameworks
- core must call backend functionality only through a small explicit interface
- public entry points must accept `backend::Symbol`

v0.1 backend support policy:
- `:flux` is the supported backend for training and inference
- `:lux` is part of the repository structure and public API plan, but may remain partial until later

### Bundle and weight policy
- trained weights must not be committed to git
- checkpoints and inference bundles are different products
- v0.1 bundle uses:
  - JSON for bundle manifest and config
  - JLD2 for weights
- small official demo bundles may be published through Julia Artifacts
- model-loading code must treat Artifacts as one source among several possible sources

### Device policy
- v0.1 training assumes NVIDIA GPU
- v0.1 inference must work on CPU
- v0.1 inference may also run on NVIDIA GPU
- non-NVIDIA GPU support is out of scope for v0.1

### Token id conventions
- KeemenaLM internal token ids are 1-based integers
- if a tokenizer produces 0-based ids, the adapter layer must shift appropriately

---

## Proposed top-level layout (v0.1)

```text
KeemenaLM.jl/
  Project.toml
  README.md
  LICENSE
  CHANGELOG.md                  # recommended
  .gitignore
  notes/                        # repository notes (not part of the package module)
    repo_plan_structure.md
    repo_plan_short.md
    repo_plan_long.md

  src/
    KeemenaLM.jl                # main module, exports, include order

    core/
      Core.jl                   # submodule KeemenaLM.Core, includes core files
      types.jl                  # core types and backend interfaces (no Flux/Lux)
      configs/
        gpt2.jl                 # GPT2Config, validation, JSON mapping
      model/
        masking.jl              # causal mask utilities (pure functions)
      generation/
        sampling.jl             # temperature, top_k, top_p (pure functions)
        stopping.jl             # EOS and stop sequences (pure functions)
        generate.jl             # generate() orchestration (calls lm_forward)
        chat.jl                 # ChatSession wrapper (minimal)
      io/
        bundle_schema.jl        # schema version and manifest structure
        model_sources.jl        # resolve local/artifact/remote-ready bundle sources
        weights_jld2.jl         # JLD2 weights adapter
        bundle_save.jl          # save_bundle()
        bundle_load.jl          # load_bundle()
      training/
        loss.jl                 # causal LM loss (framework-neutral math)
        trainer.jl              # Trainer container, dispatch to backend train_step!
        checkpoints.jl          # checkpoint save/load helpers and metadata

    backends/
      flux/
        FluxBackend.jl          # KeemenaLM.FluxBackend module entry point
        gpt2_flux.jl            # GPT-2 model definition and forward pass in Flux
        weights_flux.jl         # extract/load weight dictionary <-> Flux parameters
        train_flux.jl           # minimal Flux training step
      lux/
        LuxBackend.jl           # KeemenaLM.LuxBackend module entry point
        gpt2_lux.jl             # GPT-2 model definition and forward pass in Lux
        weights_lux.jl          # extract/load weight dictionary <-> Lux parameters/state
        train_lux.jl            # minimal Lux training step

    integrations/
      subwords.jl
      preprocessing.jl

  examples/                     # optional but recommended for the golden path
    train_tiny_gpt2_flux.jl
    train_tiny_gpt2_lux.jl
    generate_demo.jl
    chat_demo.jl
    chat_repl.jl

  test/
    runtests.jl
    unit/
      test_masking.jl
      test_sampling.jl
      test_stopping.jl
      test_bundle_schema.jl
      test_bundle_io.jl
    integration/
      test_forward_flux.jl
      test_forward_lux.jl
      test_generate_flux.jl
      test_generate_lux.jl
      test_train_step_flux.jl
      test_train_step_lux.jl
      test_checkpoint_flux.jl
```

Optional (deferred until needed):
- `artifacts/Artifacts.toml` and demo bundles shipped as artifacts
- docs and CI expansion after the golden path is stable

---

## Purpose statements and "what belongs where"

### Top-level files
- `Project.toml`
  - package dependencies
  - v0.1 may depend on Flux and Lux directly, but the source layout must still treat them as backend-only imports
- `README.md`
  - explain the golden path workflow
  - show minimal examples for training, saving checkpoints, saving bundles, loading bundles, generating text, and REPL chat
- `.gitignore`
  - must ignore weight files and local bundles
  - must ignore typical Julia build and artifact folders

### `src/KeemenaLM.jl`
Purpose:
- Public API surface.
- Includes `core/Core.jl` first, then backend modules.
Must not:
- Import Flux or Lux.
Responsibilities:
- export user-facing types/functions
- provide a single `instantiate` entry point that selects a backend
- keep backend selection explicit even before Lux parity exists

### `src/core/`
Purpose:
- framework-neutral logic
- core types, config schema, generation utilities, bundle schema and IO, source resolution, checkpoint helpers, training containers
Must not:
- import Flux or Lux
Must:
- define the backend interface functions
- keep sampling/stopping deterministic given a seed

### `src/backends/flux/`
Purpose:
- Flux-specific model construction, forward pass, training step, and weight mapping
May:
- import Flux and other Flux ecosystem dependencies needed for training
Must:
- implement the Core interfaces for Flux model types
Must not:
- reimplement sampling, stopping, or bundle schema logic
v0.1 status:
- primary supported backend

### `src/backends/lux/`
Purpose:
- Lux-specific model construction, forward pass, training step, and weight mapping
May:
- import Lux and its training ecosystem dependencies
Must:
- implement the Core interfaces for Lux model types
Must not:
- reimplement sampling, stopping, or bundle schema logic
v0.1 status:
- reserved backend path
- can remain partial/stubbed while Flux is stabilized

### `src/integrations/`
Purpose:
- thin adapters for sibling Keemena packages

Rules:
- integration helpers should be optional
- core generation should still work with any object implementing the tokenizer/preprocessing hooks

### `examples/`
Purpose:
- scripts that demonstrate the golden path
Rules:
- examples may be simple and imperative
- examples can import Flux/Lux directly if needed, but should primarily use the KeemenaLM public API
- at least one example should show REPL-first chat usage

### `test/`
Purpose:
- staged tests:
  - unit tests: core, framework-neutral
  - integration tests: backend behavior and minimal training/generation sanity
Rules:
- unit tests must not require building Flux/Lux models
- Flux integration tests should become real early
- Lux tests may begin as placeholders/skips until Lux support is active

---

## Module boundaries and import rules (enforced by structure)

### Main module layering
- `KeemenaLM` (top-level)
  - includes `KeemenaLM.Core`
  - includes `KeemenaLM.FluxBackend`
  - includes `KeemenaLM.LuxBackend`

### What imports what
- `KeemenaLM` may import:
  - `KeemenaLM.Core`
  - `KeemenaLM.FluxBackend` (as a module)
  - `KeemenaLM.LuxBackend` (as a module)
- `KeemenaLM.Core` may import JSON3, StructTypes, JLD2, Random, LinearAlgebra, Statistics, Downloads, Artifacts-related helpers, and similar general-purpose packages
- `KeemenaLM.Core` must not import Flux or Lux
- `KeemenaLM.FluxBackend` may import Flux and NVIDIA-related support used by Flux
- `KeemenaLM.LuxBackend` may import Lux and NVIDIA-related support used by Lux

---

## Core-backend interface contract (required signatures, no implementations)

The point of this contract is to keep core logic independent of Flux/Lux, while still enabling model execution, weight IO, and training.

### Core model and config abstractions
- `AbstractModelConfig`
- `AbstractCausalLM`

### Required shape conventions (v0.1)
- Token ids:
  - `input_token_ids` shape: `(sequence_length, batch_size)`
  - element type: integer token ids (1-based)
- Logits:
  - output `logits` shape: `(vocab_size, sequence_length, batch_size)`
  - element type: floating point

### Backend-required functions
These functions are declared in core and implemented (methods added) in each backend.

```julia
# Construct model from config or bundle
instantiate(config::GPT2Config; kwargs...) -> AbstractCausalLM
instantiate(bundle::Bundle; kwargs...) -> AbstractCausalLM

# Query model config
model_config(model::AbstractCausalLM) -> AbstractModelConfig

# Forward pass (no caching required in v0.1)
lm_forward(
    model::AbstractCausalLM,
    input_token_ids::AbstractMatrix{<:Integer};
    cache = nothing,
    is_training::Bool = false
) -> (logits, updated_cache)

# Weights portability (v0.1 uses Dict{String, Any})
extract_weights(model::AbstractCausalLM) -> Dict{String, Any}
load_weights!(model::AbstractCausalLM, weights::Dict{String, Any}) -> model

# Training step (minimal; backend owns optimizer mechanics)
train_step!(
    trainer::Trainer,
    input_token_ids::AbstractMatrix{<:Integer},
    target_token_ids::AbstractMatrix{<:Integer}
) -> NamedTuple  # e.g. (loss = ..., metrics = ...)
```

Notes:
- `cache` can remain `nothing` in v0.1
- if a backend wants to introduce a cache type later, the same signature supports it
- weight dictionary keys must follow a backend-independent parameter schema for GPT-2

### Tokenizer and preprocessing integration points
Core defines adapter hooks that upstream Keemena packages can extend.

```julia
preprocess_text(preprocessing, text::AbstractString) -> AbstractString
tokenizer_encode(tokenizer, text::AbstractString) -> Vector{Int}
tokenizer_decode(tokenizer, token_ids::AbstractVector{<:Integer}) -> String
```

v0.1 requirement:
- generation and chat must work with any tokenizer/preprocessing object that implements these methods

---

## Generation API requirements (v0.1)

Core generation should be usable regardless of backend.

### Generation config
- `GenerationConfig` should include:
  - `max_new_tokens::Int`
  - `temperature::Float64`
  - `top_k::Int` (0 disables)
  - `top_p::Float64` (1.0 disables)
  - `seed::Union{Nothing, Int}`
  - `eos_token_id::Union{Nothing, Int}`
  - `stop_sequences::Vector{String}` (optional; acceptable to implement as a simple text-level stop in v0.1)

### Required generation entry point
```julia
generate(
    model::AbstractCausalLM,
    tokenizer,
    preprocessing,
    prompt_text::AbstractString;
    generation_config::GenerationConfig = GenerationConfig()
) -> String
```

v0.1 behavior requirements:
- deterministic sampling when `seed` is provided and running on CPU
- truncate prompt to the model context length
- stop on EOS token when `eos_token_id` is provided
- callable on CPU
- callable on NVIDIA GPU when supported by the active backend

### Minimal chat wrapper
```julia
ChatSession(model, tokenizer, preprocessing; system_prompt = "", generation_config = GenerationConfig())
chat!(session::ChatSession, user_text::AbstractString; overrides...) -> String
```

v0.1 behavior requirements:
- maintain a simple in-memory message history
- build a plain-text prompt format that is easy to replace later
- prefer REPL-first ergonomics

---

## Checkpoints and bundles

Training checkpoints and inference bundles should stay distinct.

### Checkpoint requirements
Purpose:
- resume training

Should include:
- model weights
- optimizer state
- step or epoch counters
- backend symbol
- config snapshot
- RNG state if practical

Recommended entry points:
```julia
save_checkpoint(path, trainer, model; metadata...) -> path
load_checkpoint(path)
```

### Bundle requirements
Purpose:
- inference
- sharing
- artifact-published demo models

Bundles are directories on disk. v0.1 requires round-trip correctness:
- save bundle -> load bundle -> instantiate -> generate

### Bundle directory layout (v0.1 recommended)
- `bundle/`
  - `bundle.json`           required
  - `model_config.json`     required
  - `weights.jld2`          required
  - `tokenizer/`            optional in v0.1 (schema supports it)
  - `preprocessing/`        optional in v0.1 (schema supports it)
  - `README.md`             optional model card

### Manifest requirements (`bundle.json`)
Fields (minimum viable set):
- `schema_version::Int` (v0.1 uses 1)
- `architecture::String` (v0.1: "gpt2")
- `model_config_file::String` (relative path)
- `weights_file::String` (relative path)
- `weights_format::String` (v0.1: "jld2")
- `parameter_schema::String`
Optional fields:
- tokenizer reference path or inline metadata pointer
- preprocessing reference path or inline metadata pointer
- distribution metadata for local/artifact/remote provenance
- `metadata::Dict{String, Any}` (training details optional)

### Required IO entry points
```julia
save_bundle(directory_path::AbstractString, bundle::Bundle) -> directory_path
load_bundle(directory_path::AbstractString) -> Bundle
```

### Bundle source resolution
The package should resolve a source to a local bundle directory before normal load logic runs.

Recommended entry points:
```julia
resolve_bundle(source) -> local_directory_path
load_bundle(source) -> Bundle
```

v0.1 supported sources:
- local directory
- Julia artifact

Later sources:
- direct URL
- Hugging Face-style remote repository

### Weights portability requirement
- `extract_weights` must produce a stable dictionary whose keys do not depend on Flux vs Lux specifics.
- `load_weights!` must accept that dictionary and populate the model.

v0.1 allowance:
- start with one stable naming convention that both backends implement, then iterate

---

## Tests (staged and practical)

### Unit tests (framework-neutral)
Located under `test/unit/`.
Must cover:
- causal masks correctness
- sampling correctness (temperature/top_k/top_p edge cases)
- stopping correctness (EOS and any stop-sequence logic included)
- bundle schema and bundle save/load roundtrip using dummy weights

Constraints:
- Unit tests must not require Flux or Lux model construction.

### Integration tests (backend behavior)
Located under `test/integration/`.
Must cover:
- forward pass shape conventions for Flux
- deterministic generation on CPU with fixed seed
- minimal training step sanity:
  - finite loss
  - parameters update
  - ideally loss decreases on a trivial batch after a few steps
- checkpoint roundtrip for the supported backend
- optional NVIDIA GPU smoke tests when practical

Practical staging:
- integration tests may start as skipped/broken until the backend implementations exist
- Flux integration tests should become active first
- Lux integration tests should activate as Lux support lands

---

## CI matrix suggestion (when you add CI)

- OS:
  - Linux (required)
  - macOS or Windows (optional)
- Julia versions:
  - current stable
  - LTS if you intend to support it
- Jobs:
  - core unit tests
  - Flux integration tests
  - Lux integration tests when Lux is active
  - optional NVIDIA GPU smoke job if runners exist

---

## v0.1 implementation plan: feature -> file mapping

This mapping is the checklist for v0.1 work.

### 1) GPT-2 config and validation
- Files:
  - `src/core/configs/gpt2.jl`
- Requirements:
  - validate hyperparameters
  - JSON mapping for saving/loading

### 2) Causal mask utilities
- Files:
  - `src/core/model/masking.jl`
- Requirements:
  - causal mask construction for a given sequence length
  - unit tests

### 3) GPT-2 model (Flux backend)
- Files:
  - `src/backends/flux/gpt2_flux.jl`
- Requirements:
  - implement `lm_forward` for the Flux model type
  - enforce shape conventions
  - support training on NVIDIA GPU
  - support inference on CPU and NVIDIA GPU
  - integration forward test

### 4) GPT-2 model (Lux backend)
- Files:
  - `src/backends/lux/gpt2_lux.jl`
- Requirements:
  - preserve matching module and API shape
  - preserve the backend-independent parameter naming contract
  - may remain a stub until later

### 5) Generation loop and sampling
- Files:
  - `src/core/generation/sampling.jl`
  - `src/core/generation/stopping.jl`
  - `src/core/generation/generate.jl`
  - `src/core/generation/chat.jl`
- Requirements:
  - deterministic generation on CPU with a seed
  - EOS stopping support

### 6) Bundle schema and save/load (JSON + JLD2)
- Files:
  - `src/core/io/bundle_schema.jl`
  - `src/core/io/model_sources.jl`
  - `src/core/io/weights_jld2.jl`
  - `src/core/io/bundle_save.jl`
  - `src/core/io/bundle_load.jl`
- Requirements:
  - load/save roundtrip tests (dummy weights)
  - instantiate from loaded bundle must be supported (backend code)
  - local and artifact source resolution

### 7) Weight mapping for both backends
- Files:
  - `src/backends/flux/weights_flux.jl`
  - `src/backends/lux/weights_lux.jl`
- Requirements:
  - `extract_weights` and `load_weights!` implemented for Flux
  - Lux path may begin as a naming-contract stub
  - dictionary keys must be stable and documented (even if minimal)

### 8) Checkpoint save/load
- Files:
  - `src/core/training/checkpoints.jl`
- Requirements:
  - persist enough state to resume training later
  - clearly separate checkpoint semantics from inference bundle semantics

### 9) Minimal training step sanity
- Files:
  - `src/core/training/loss.jl`
  - `src/core/training/trainer.jl`
  - `src/backends/flux/train_flux.jl`
  - `src/backends/lux/train_lux.jl`
- Requirements:
  - Flux path: one training step updates parameters on the supported NVIDIA GPU path
  - Lux path may be deferred
  - basic integration tests

---

## Deferred items (explicitly not required for v0.1)

- KV cache and fast decoding
- Streaming generation callbacks
- Full training loops and evaluation metrics suite
- ONNX export/inference
- Lux backend parity
- Direct URL and Hugging Face-style model distribution

v0.1 should remain small, explicit, and testable.
