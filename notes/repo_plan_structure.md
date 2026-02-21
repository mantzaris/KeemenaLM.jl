## KeemenaLM.jl repository scaffold and boundaries (guiding document, no implementations)

This note defines the concrete repository scaffold for KeemenaLM.jl and the boundary rules that keep v0.1 shippable. It is intentionally a guiding document: it specifies folders, responsibilities, and required interfaces, but does not include function bodies or full code.

The v0.1 priority is a single golden path:
- Train a tiny GPT-2 style decoder-only model (Flux or Lux).
- Save a portable bundle (JSON config + JLD2 weights).
- Load the bundle.
- Generate text and run a minimal chat loop.

Everything else is deferred unless it directly supports this path.

---

## Non-negotiable constraints

### Scope control
- v0.1 ships one architecture: GPT-2 style decoder-only transformer.
- Prefer correctness and readability over performance optimizations.
- KV cache and streaming are allowed only if they do not risk shipping v0.1.

### Backend boundaries
- `src/core/` must not import Flux or Lux.
  - No `using Flux`
  - No `using Lux`
- Only:
  - `src/backends/flux/`
  - `src/backends/lux/`
  may import Flux/Lux.
- Core must call backend functionality only through a small, explicit interface (multiple dispatch).

### Bundle and weight policy
- Trained weights must not be committed to git.
- v0.1 bundle uses:
  - JSON for bundle manifest and model config
  - JLD2 for weights
- Plan for safetensors later via an adapter file, but do not block v0.1 on it.

### Token id conventions
- KeemenaLM internal token ids are treated as 1-based integers (Julia-native indexing).
- If a tokenizer produces 0-based ids, the adapter layer must shift appropriately.

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
        weights_jld2.jl         # JLD2 weights adapter
        bundle_save.jl          # save_bundle()
        bundle_load.jl          # load_bundle()
      training/
        loss.jl                 # causal LM loss (framework-neutral math)
        trainer.jl              # Trainer container, dispatch to backend train_step!

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

  examples/                     # optional but recommended for the golden path
    train_tiny_gpt2_flux.jl
    train_tiny_gpt2_lux.jl
    generate_demo.jl
    chat_demo.jl

  test/
    runtests.jl
    unit/
      test_masking.jl
      test_sampling.jl
      test_stopping.jl
      test_bundle_io.jl
    integration/
      test_forward_flux.jl
      test_forward_lux.jl
      test_generate_flux.jl
      test_generate_lux.jl
      test_train_step_flux.jl
      test_train_step_lux.jl
```

Optional (deferred until needed):
- `docs/` (Documenter.jl)
- `artifacts/Artifacts.toml` and demo bundles shipped as artifacts
- `.github/workflows/ci.yml` once tests are stable

---

## Purpose statements and "what belongs where"

### Top-level files
- `Project.toml`
  - Package dependencies.
  - v0.1 may depend on Flux and Lux directly, but the source layout must still treat them as backend-only imports.
- `README.md`
  - Explain the golden path workflow.
  - Show minimal examples for training, saving bundle, loading bundle, generating text.
- `.gitignore`
  - Must ignore weight files and local bundles.
  - Must ignore typical Julia build and artifact folders.

### `src/KeemenaLM.jl`
Purpose:
- Public API surface.
- Includes `core/Core.jl` first, then backend modules.
Must not:
- Import Flux or Lux.
Responsibilities:
- Export user-facing types/functions.
- Provide a single `instantiate` entry point that selects a backend.

### `src/core/`
Purpose:
- Framework-neutral logic.
- Core types, config schema, generation utilities, bundle schema and IO, training containers.
Must not:
- Import Flux or Lux.
Must:
- Define the backend interface functions (signatures and expectations).
- Keep sampling/stopping deterministic given a seed.

### `src/backends/flux/`
Purpose:
- Flux-specific model construction, forward pass, training step, and weight mapping.
May:
- Import Flux and other Flux ecosystem dependencies needed for training.
Must:
- Implement the Core interfaces for Flux model types.
Must not:
- Reimplement sampling, stopping, or bundle schema logic.

### `src/backends/lux/`
Purpose:
- Lux-specific model construction, forward pass, training step, and weight mapping.
May:
- Import Lux and its training ecosystem dependencies.
Must:
- Implement the Core interfaces for Lux model types.
Must not:
- Reimplement sampling, stopping, or bundle schema logic.

### `examples/`
Purpose:
- Scripts that demonstrate the golden path.
Rules:
- Examples may be simple and imperative.
- Examples can import Flux/Lux directly if needed, but should primarily use the KeemenaLM public API.

### `test/`
Purpose:
- Staged tests:
  - Unit tests: core, framework-neutral.
  - Integration tests: backend behavior and minimal training/generation sanity.
Rules:
- Unit tests must not require building Flux/Lux models.
- Integration tests can be skipped or marked broken initially, but should become real as the backends land.

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
- `KeemenaLM.Core` may import:
  - JSON3, StructTypes, JLD2, Random, LinearAlgebra, Statistics, etc.
  - It must not import Flux or Lux.
- `KeemenaLM.FluxBackend` may import Flux and extend Core interface methods.
- `KeemenaLM.LuxBackend` may import Lux and extend Core interface methods.

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
- `cache` can remain `nothing` in v0.1.
- If a backend wants to introduce a cache type later, the same signature supports it.

### Tokenizer and preprocessing integration points
Core defines adapter hooks that upstream Keemena packages can extend.

```julia
preprocess_text(preprocessing, text::AbstractString) -> AbstractString
tokenizer_encode(tokenizer, text::AbstractString) -> Vector{Int}
tokenizer_decode(tokenizer, token_ids::AbstractVector{<:Integer}) -> String
```

v0.1 requirement:
- Generation and chat must work with any tokenizer/preprocessing object that implements these methods.

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
- Deterministic sampling when `seed` is provided and running on CPU.
- Truncate prompt to the model context length.
- Stop on EOS token when `eos_token_id` is provided.

### Minimal chat wrapper
```julia
ChatSession(model, tokenizer, preprocessing; system_prompt = "", generation_config = GenerationConfig())
chat!(session::ChatSession, user_text::AbstractString; overrides...) -> String
```

v0.1 behavior requirements:
- Maintain a simple in-memory message history.
- Build a plain-text prompt format that is easy to replace later.

---

## Bundle format and IO requirements (v0.1)

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
Optional fields:
- tokenizer reference path or inline metadata pointer
- preprocessing reference path or inline metadata pointer
- `metadata::Dict{String, Any}` (training details optional)

### Required IO entry points
```julia
save_bundle(directory_path::AbstractString, bundle::Bundle) -> directory_path
load_bundle(directory_path::AbstractString) -> Bundle
```

### Weights portability requirement
- `extract_weights` must produce a stable dictionary whose keys do not depend on Flux vs Lux specifics.
- `load_weights!` must accept that dictionary and populate the model.

v0.1 allowance:
- Start with one stable naming convention that both backends implement, then iterate.

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
- forward pass shape conventions for Lux
- deterministic generation on CPU with fixed seed (Flux and Lux)
- minimal training step sanity:
  - finite loss
  - parameters update
  - ideally loss decreases on a trivial batch after a few steps

Practical staging:
- Integration tests may start as skipped/broken until the backend implementations exist.
- As soon as a backend forward pass exists, its integration tests should become active.

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
  - Lux integration tests

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
  - integration forward test

### 4) GPT-2 model (Lux backend)
- Files:
  - `src/backends/lux/gpt2_lux.jl`
- Requirements:
  - implement `lm_forward` for the Lux model type
  - enforce shape conventions
  - integration forward test

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
  - `src/core/io/weights_jld2.jl`
  - `src/core/io/bundle_save.jl`
  - `src/core/io/bundle_load.jl`
- Requirements:
  - load/save roundtrip tests (dummy weights)
  - instantiate from loaded bundle must be supported (backend code)

### 7) Weight mapping for both backends
- Files:
  - `src/backends/flux/weights_flux.jl`
  - `src/backends/lux/weights_lux.jl`
- Requirements:
  - `extract_weights` and `load_weights!` implemented for both backends
  - dictionary keys must be stable and documented (even if minimal)

### 8) Minimal training step sanity
- Files:
  - `src/core/training/loss.jl`
  - `src/core/training/trainer.jl`
  - `src/backends/flux/train_flux.jl`
  - `src/backends/lux/train_lux.jl`
- Requirements:
  - one training step updates parameters
  - basic integration tests

---

## Deferred items (explicitly not required for v0.1)

- KV cache and fast decoding
- Streaming generation callbacks
- Full training loops, checkpointing, evaluation metrics suite
- Artifact-distributed demo bundles
- ONNX export/inference
- Full docs site with Documenter

v0.1 should remain small and correct. Add features only when the golden path is stable and tested.