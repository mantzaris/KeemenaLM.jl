## KeemenaLM.jl staged roadmap and recommendation

This note is the direct execution roadmap for getting KeemenaLM.jl from scaffold state to a usable proof-of-concept training and chat package.

It is intentionally staged rather than granular. Each stage should end with a clear outcome and a small set of tests that prove the stage is done well enough to build on.

---

## Overall recommendation

Build the package in this order:
1. Make the Flux GPT-2 path correct on CPU first.
2. Make bundle save/load and generation work on CPU.
3. Make the same Flux path run on NVIDIA GPU for training.
4. Make the REPL chat flow usable from a saved bundle.
5. Only then expand Lux support and model distribution polish.

This order keeps the risk low:
- correctness before acceleration
- one supported backend before two
- inference usability before broader packaging work
- explicit test gates before moving to the next stage

---

## Coordination guidance for files and function scope

This section is here to reduce drift. The package should stay easy to reason about by keeping each file narrow in purpose and by keeping new functions close to the layer that actually owns the behavior.

### Main trunk rule
Every new file or function should serve the current stage on the main path:
- Flux GPT-2
- checkpoint or bundle persistence
- CPU generation
- NVIDIA GPU training
- REPL chat

If a new file does not clearly support one of those, it is probably early.

### File ownership guidance

#### `src/core/`
This layer owns framework-neutral logic only.

Allowed here:
- configs and validation
- masks
- sampling and stopping
- generation orchestration
- chat state and chat prompt formatting
- bundle schema and bundle source resolution
- checkpoint schema and metadata
- pure loss math
- abstract interfaces used by backends

Not allowed here:
- `using Flux`
- `using Lux`
- backend-specific optimizer logic
- CUDA or NVIDIA device calls
- parameter traversal code tied to one backend

Rule of thumb:
If a function should work the same for Flux and Lux, it belongs in `core`.

#### `src/backends/flux/`
This layer owns Flux-specific behavior.

Allowed here:
- Flux model structs and layers
- Flux forward pass
- Flux parameter extraction and loading
- Flux training step
- Flux device movement for CPU or NVIDIA GPU

Not allowed here:
- redefining bundle schema
- redefining generation sampling
- redefining stop logic
- redefining generic chat behavior

Rule of thumb:
If a function depends on Flux types or Flux parameter layout, it belongs here.

#### `src/backends/lux/`
This layer should mirror the Flux backend structure, even before it reaches feature parity.

Allowed here:
- Lux-specific model construction
- Lux forward pass
- Lux parameter extraction and loading
- Lux training step when implemented

Rule of thumb:
Lux files should conform to the same conceptual contract as Flux files, not invent a parallel architecture.

#### `src/integrations/`
This layer owns optional adapters to sibling packages.

Allowed here:
- `KeemenaSubwords.jl` adapters
- `KeemenaPreprocessing.jl` adapters
- translation glue between external package types and KeemenaLM hooks

Not allowed here:
- moving core logic out of `core`
- backend-specific model logic

Rule of thumb:
If removing the sibling package should not break the rest of the architecture, the code is in the right place.

#### `examples/`
Examples should prove the golden path, not become a second API surface.

Allowed here:
- imperative scripts
- explicit device choices
- tiny training or chat workflows

Not allowed here:
- hidden production logic that should live in `src/`

Rule of thumb:
If an example contains reusable logic, move that logic into package code and keep the example thin.

#### `test/`
Tests should track architectural boundaries, not fight them.

Rules:
- `test/unit/` should test framework-neutral behavior
- `test/integration/` should test backend behavior and end-to-end flows
- if a test requires Flux or Lux model construction, it is not a unit test

### Function scope guidance

#### Prefer narrow orchestration functions in core
Core functions should usually:
- validate inputs
- call backend interface functions
- apply framework-neutral logic
- return plain Julia data or core types

Core should not absorb backend details just because the Flux path lands first.

#### Prefer backend leaf functions for backend details
Backend functions should usually:
- own model construction
- own forward execution
- own parameter extraction/loading
- own training step mechanics
- own device movement

Do not push Flux-specific shortcuts into core just to move faster.

#### Keep persistence functions small
Persistence code should split into:
- schema definition
- source resolution
- actual read/write

That means:
- `bundle_schema.jl` defines structure and versioning
- `model_sources.jl` resolves local path or artifact sources
- `bundle_save.jl` writes bundles
- `bundle_load.jl` reads bundles
- `checkpoints.jl` handles resumable training state

Do not let one large save/load file absorb all persistence behavior.

#### Keep chat and generation separate
Generation and chat are related but not the same.

Recommended boundary:
- `generate.jl` handles token generation from a prompt
- `chat.jl` handles message history, prompt assembly, and chat session flow

This matters because chat prompt formatting will change faster than raw generation.

### Trunk-safety rules

#### Add one responsibility per file
When a file starts doing two or three unrelated jobs, split it before adding more.

#### Avoid speculative abstractions
Do not create generic plugin-like abstractions for features that only have one real implementation today.

Good abstraction now:
- backend interface for model execution
- source resolution interface for bundles

Bad abstraction now:
- full architecture registries
- many-layer device frameworks
- highly generic trainer systems

#### Keep the public API smaller than the internal implementation
New internal helper functions are fine. New exported functions should be added slowly and only when they support the main user path.

#### Prefer extending an existing file before creating parallel variants
If a new helper clearly belongs to an existing file, add it there first. New files should appear when the responsibility is genuinely new, not just because a name felt convenient.

#### Do not let tests drift from the real user path
The most important integration tests should always cover:
- train or step
- save
- load
- generate
- chat

If tests are only proving isolated helper behavior, the package can drift without anyone noticing.

### Recommended coordination by stage

#### Stages 1 to 2
Most changes should be in:
- `src/core/`
- `src/backends/flux/`
- `test/unit/`
- `test/integration/`

Avoid touching Lux unless the change is needed to preserve matching interfaces.

#### Stages 3 to 4
Most changes should be in:
- `src/core/training/`
- `src/core/io/`
- `src/backends/flux/`
- `examples/`

Keep the training path explicit and avoid hiding too much state in helper layers.

#### Stages 5 to 6
Most changes should be in:
- `src/core/generation/`
- `src/core/io/`
- `src/integrations/`
- `examples/`

Avoid expanding backend complexity here unless it blocks actual user usage.

#### Stage 7
Most changes should be in:
- `src/backends/lux/`
- Lux integration tests

The goal is Lux conformity to the existing trunk decisions, not re-opening core architecture.

---

## Stage 1: Finish the core CPU path

### Goal
Get one backend-neutral end-to-end path working on CPU with the Flux backend:
- instantiate GPT-2 model
- run forward pass
- compute loss
- run generation

### Recommendation
Do not start with GPU-specific work here. Keep this stage CPU-first so failures are easier to debug and test deterministically.

Focus on:
- `GPT2Config` validation
- causal masking correctness
- Flux forward pass shape correctness
- core loss function
- sampling and stop logic
- `generate()` on CPU with deterministic seed behavior

### Files to create or complete
- `src/core/configs/gpt2.jl`
- `src/core/model/masking.jl`
- `src/core/generation/sampling.jl`
- `src/core/generation/stopping.jl`
- `src/core/generation/generate.jl`
- `src/core/training/loss.jl`
- `src/backends/flux/gpt2_flux.jl`
- `src/backends/flux/FluxBackend.jl`
- `test/unit/test_configs.jl`
- `test/unit/test_masking.jl`
- `test/unit/test_sampling.jl`
- `test/unit/test_stopping.jl`
- `test/integration/test_forward_flux.jl`
- `test/integration/test_generate_flux.jl`

### Functions to add or complete
- in `src/core/configs/gpt2.jl`:
  - `validate(config::GPT2Config)`
- in `src/core/model/masking.jl`:
  - causal mask helpers used by both backends
- in `src/core/generation/sampling.jl`:
  - token sampling helpers for temperature, `top_k`, and `top_p`
- in `src/core/generation/stopping.jl`:
  - EOS and stop-sequence helpers
- in `src/core/generation/generate.jl`:
  - `generate(model, tokenizer, preprocessing, prompt_text; generation_config=...)`
- in `src/core/training/loss.jl`:
  - causal language-model loss over logits and targets
- in `src/backends/flux/gpt2_flux.jl`:
  - Flux GPT-2 model construction
  - `lm_forward(model, input_token_ids; ...)`
  - `model_config(model)`
- in `src/backends/flux/FluxBackend.jl`:
  - `instantiate(config::GPT2Config; ...)`

### Call and dependency flow
- `instantiate(config; backend=:flux)` in the top-level module should delegate to the Flux backend constructor.
- Flux model construction should depend on validated `GPT2Config`.
- `generate(...)` in core should:
  - preprocess text
  - encode tokens
  - call `lm_forward(...)`
  - call sampling helpers
  - call stopping helpers
  - decode the final tokens
- loss computation should remain in core and operate on logits returned by backend `lm_forward(...)`.

### Dependency rule for this stage
- `generate.jl` may depend on `sampling.jl`, `stopping.jl`, and tokenizer/preprocessing hooks.
- `generate.jl` must not depend on Flux internals.
- `gpt2_flux.jl` may depend on masking helpers, but masking helpers must not depend on Flux.

### Expected outcome
You can build a tiny Flux GPT-2 model, feed token ids through it, get logits with the expected shape, compute finite loss, and generate text deterministically on CPU.

### Tests to require before moving on
- unit:
  - config validation tests
  - masking tests
  - sampling edge-case tests
  - stopping tests
- integration:
  - Flux forward pass returns expected logits shape
  - loss is finite on a trivial batch
  - CPU generation is deterministic when `seed` is fixed

### Stop condition for this stage
Do not move to GPU training or bundle work until CPU forward, loss, and generation are stable.

---

## Stage 2: Finish portable inference bundles

### Goal
Make the package able to save a trained or randomly initialized model as a portable inference bundle and load it back correctly.

### Recommendation
Separate inference bundles from training checkpoints now, before the persistence code grows. That avoids a lot of confusion later.

Focus on:
- `bundle.json`
- `model_config.json`
- `weights.jld2`
- stable parameter naming contract
- `save_bundle`
- `load_bundle`
- `resolve_bundle` for local paths first

If artifact support is easy at this point, add the resolution abstraction now, but keep actual artifact-published demo models for a later stage.

### Files to create or complete
- `src/core/io/bundle_schema.jl`
- `src/core/io/model_sources.jl`
- `src/core/io/weights_jld2.jl`
- `src/core/io/bundle_save.jl`
- `src/core/io/bundle_load.jl`
- `src/backends/flux/weights_flux.jl`
- `test/unit/test_bundle_schema.jl`
- `test/unit/test_bundle_io.jl`

### Functions to add or complete
- in `src/core/io/bundle_schema.jl`:
  - manifest and bundle types
  - schema version helpers if needed
- in `src/core/io/model_sources.jl`:
  - `resolve_bundle(source)`
- in `src/core/io/weights_jld2.jl`:
  - JLD2 read/write helpers for portable weight dictionaries
- in `src/core/io/bundle_save.jl`:
  - `save_bundle(path, bundle)`
- in `src/core/io/bundle_load.jl`:
  - `load_bundle(source)`
- in `src/backends/flux/weights_flux.jl`:
  - `extract_weights(model)`
  - `load_weights!(model, weights)`

### Call and dependency flow
- `save_bundle(...)` should depend on:
  - bundle schema definitions
  - config serialization
  - backend `extract_weights(model)`
  - JLD2 helper code
- `load_bundle(source)` should:
  - call `resolve_bundle(source)`
  - read bundle metadata and config
  - read portable weights
  - return a core `Bundle` object
- `instantiate(bundle; backend=:flux)` should:
  - build a Flux model from the config
  - call backend `load_weights!(...)`

### Dependency rule for this stage
- bundle save/load code should know about bundle structure, not Flux parameter internals
- backend weight files should know about Flux parameter internals, not JSON/JLD2 bundle layout decisions beyond the portable dictionary contract

### Expected outcome
A bundle saved on disk can be loaded back, instantiated, and used for generation without relying on the original training session state.

### Tests to require before moving on
- unit:
  - bundle schema validation
  - bundle save/load roundtrip with dummy weights
- integration:
  - save bundle -> load bundle -> instantiate -> generate on CPU
  - loaded model produces consistent shapes and finite outputs

### Stop condition for this stage
Do not build the user-facing REPL chat flow until bundle roundtrips are reliable.

---

## Stage 3: Add resumable training checkpoints

### Goal
Support training resumption without mixing that concern into the inference bundle format.

### Recommendation
Keep checkpoint code small and pragmatic. A checkpoint only needs enough data to resume training safely:
- model weights
- optimizer state
- step or epoch counter
- backend symbol
- config snapshot
- RNG state if practical

Do not overdesign checkpoint metadata in v0.1.

### Files to create or complete
- `src/core/training/checkpoints.jl`
- `src/core/training/trainer.jl`
- `src/backends/flux/train_flux.jl`
- `test/integration/test_checkpoint_flux.jl`

### Functions to add or complete
- in `src/core/training/checkpoints.jl`:
  - `save_checkpoint(path, trainer, model; metadata...)`
  - `load_checkpoint(path)`
- in `src/core/training/trainer.jl`:
  - `Trainer` type carrying optimizer and training metadata needed by the supported path
- in `src/backends/flux/train_flux.jl`:
  - `train_step!(trainer, input_token_ids, target_token_ids)`

### Call and dependency flow
- `train_step!(...)` should:
  - call backend `lm_forward(...)`
  - call core loss helpers
  - update backend-owned parameters via Flux machinery
- `save_checkpoint(...)` should gather:
  - model weights via backend extraction
  - trainer state
  - checkpoint metadata
- `load_checkpoint(...)` should return enough information for training code to reconstruct trainer and model state cleanly

### Dependency rule for this stage
- checkpoint code may depend on bundle-style weight serialization helpers if that reduces duplication
- checkpoint code should not be the same API surface as bundle IO, because the semantics are different

### Expected outcome
A training run can stop, save a checkpoint, reload, and continue from roughly the same state.

### Tests to require before moving on
- integration:
  - checkpoint save/load roundtrip for Flux
  - after reload, model parameters match saved parameters
  - training step counter resumes correctly

### Stop condition for this stage
Do not call the training workflow usable until checkpoints can roundtrip.

---

## Stage 4: Make Flux training usable on NVIDIA GPU

### Goal
Train the supported backend on NVIDIA GPU in a way that is realistic for the project scope.

### Recommendation
Assume:
- training is NVIDIA GPU only in v0.1
- inference remains available on CPU and optionally NVIDIA GPU

Do not try to optimize heavily yet. The priority is:
- model moves to device cleanly
- batches move to device cleanly
- one training step updates parameters
- short training run is numerically sane

Keep the code path explicit. Avoid hiding too much device logic.

### Files to create or complete
- `src/backends/flux/train_flux.jl`
- `src/backends/flux/gpt2_flux.jl`
- `examples/train_tiny_gpt2_flux.jl`
- `test/integration/test_train_step_flux.jl`

### Functions to add or complete
- in `src/backends/flux/train_flux.jl`:
  - Flux training-step path that works on the supported NVIDIA GPU setup
  - explicit device-move helpers if needed
- in `src/backends/flux/gpt2_flux.jl`:
  - model/device movement helpers if needed by training and inference
- in `examples/train_tiny_gpt2_flux.jl`:
  - small reference training script using the supported training path

### Call and dependency flow
- training example should:
  - build or load tokenizer/preprocessing
  - create config
  - call `instantiate(...; backend=:flux)`
  - build `Trainer`
  - call `train_step!(...)` in a loop
  - call `save_checkpoint(...)`
  - optionally call `save_bundle(...)`
- device movement should stay inside Flux backend code or the training example glue, not leak into core generation logic

### Dependency rule for this stage
- keep GPU-specific code in Flux backend files and top-level training scripts
- core types may carry device-agnostic metadata, but should not own CUDA calls

### Expected outcome
You can run a tiny Flux GPT-2 training script on an NVIDIA GPU, see finite losses, update parameters, save a checkpoint, and export a bundle.

### Tests to require before moving on
- integration:
  - Flux training step updates parameters on the supported path
  - a short toy run does not produce NaNs
  - checkpoint can be saved after GPU training
- optional smoke tests:
  - a very small NVIDIA GPU test in CI if infrastructure exists

### Stop condition for this stage
Do not expand Lux or model distribution until the supported Flux training path works in practice.

---

## Stage 5: Make the package usable from the REPL

### Goal
Deliver the first real user-facing workflow:
- load a bundle
- construct a chat session
- talk to the model in the Julia REPL

### Recommendation
Keep the interface REPL-first and minimal. A good first target is:
- `load_bundle(source)`
- `instantiate(bundle; backend=:flux)`
- `ChatSession(...)`
- `chat!(...)`
- `chat_repl(...)`

Prompt formatting can stay simple in v0.1. The important thing is that a user can actually use the package without writing much glue code.

### Files to create or complete
- `src/core/generation/chat.jl`
- `examples/chat_demo.jl`
- `examples/chat_repl.jl`
- `test/integration/test_chat_flux.jl`

### Functions to add or complete
- in `src/core/generation/chat.jl`:
  - `ChatSession(...)`
  - `chat!(session, user_text; overrides...)`
  - optional helper to format message history into a prompt
- in `examples/chat_repl.jl`:
  - `chat_repl(...)` or equivalent thin REPL loop wrapper

### Call and dependency flow
- `chat!(...)` should:
  - append the user message to session history
  - build a prompt from session state
  - call `generate(...)`
  - append the assistant reply to session history
  - return the reply text
- `chat_repl(...)` should be a thin example-facing loop around `chat!(...)`, not a second chat implementation

### Dependency rule for this stage
- `chat.jl` may depend on `generate.jl`
- `generate.jl` must not depend on chat session types

### Expected outcome
A user can install the package, load a local or published demo bundle, and chat with the model from the Julia REPL on CPU.

### Tests to require before moving on
- integration:
  - bundle load -> chat session creation -> one chat turn
  - multi-turn chat keeps message history correctly
  - CPU inference still behaves deterministically when requested

### Stop condition for this stage
At this point the package is a real proof of concept and can reasonably be shown to users.

---

## Stage 6: Publish one official demo model

### Goal
Prove the model distribution story with one small official model.

### Recommendation
Use Julia Artifacts for the first official tiny demo model. Design the code so artifacts are only one source type, but use them first because they fit a Julia-native proof of concept well.

The first published model should be:
- small
- stable
- documented
- easy to download from Julia

Do not publish a large or ambitious model first.

### Files to create or complete
- `src/core/io/model_sources.jl`
- optional `artifacts/Artifacts.toml`
- documentation or note describing the demo model key and provenance
- `test/integration/test_artifact_bundle_flux.jl`

### Functions to add or complete
- in `src/core/io/model_sources.jl`:
  - source-resolution logic for official artifact-backed models
  - `available_models()`
  - `download_model(model_key)` if you want an explicit fetch API rather than implicit resolution

### Call and dependency flow
- `available_models()` should expose package-known model identifiers
- `download_model(model_key)` or `resolve_bundle(model_key)` should materialize the artifact locally
- `load_bundle(source)` should not care whether the final source came from a local path or an artifact once resolution is done

### Dependency rule for this stage
- artifact logic belongs in source resolution, not in bundle parsing or backend code

### Expected outcome
Users can call a package API to discover or download a tiny official model and then chat with it locally.

### Tests to require before moving on
- integration:
  - resolve artifact model -> load bundle -> instantiate -> generate
  - artifact-backed chat example works on CPU

### Stop condition for this stage
After this stage, the project has a coherent training, packaging, distribution, and usage story.

---

## Stage 7: Expand Lux support

### Goal
Bring the Lux backend up to the same architectural contract after the Flux path is already real and usable.

### Recommendation
Do not chase Lux parity in parallel with Flux stabilization. Use Flux to settle:
- parameter naming contract
- bundle schema
- checkpoint expectations
- generation behavior

Then make Lux conform to those decisions.

The first Lux target should be:
- instantiate
- forward pass
- load/save shared weight schema
- generate

Training parity can follow after inference parity.

### Files to create or complete
- `src/backends/lux/LuxBackend.jl`
- `src/backends/lux/gpt2_lux.jl`
- `src/backends/lux/weights_lux.jl`
- `src/backends/lux/train_lux.jl`
- `test/integration/test_forward_lux.jl`
- `test/integration/test_generate_lux.jl`
- later `test/integration/test_train_step_lux.jl`

### Functions to add or complete
- in `src/backends/lux/LuxBackend.jl`:
  - `instantiate(config::GPT2Config; ...)`
  - `instantiate(bundle::Bundle; ...)`
- in `src/backends/lux/gpt2_lux.jl`:
  - Lux GPT-2 model construction
  - `lm_forward(...)`
  - `model_config(...)`
- in `src/backends/lux/weights_lux.jl`:
  - `extract_weights(...)`
  - `load_weights!(...)`
- in `src/backends/lux/train_lux.jl`:
  - later `train_step!(...)`

### Call and dependency flow
- Lux instantiate/load flow should mirror the Flux flow:
  - config or bundle in
  - Lux model constructed
  - shared parameter schema loaded
  - core generation reused unchanged
- Lux backend should adapt to the shared core interfaces rather than asking core to branch deeply on backend

### Dependency rule for this stage
- do not reopen bundle schema, chat behavior, or source resolution to suit Lux unless a genuine architectural bug is found

### Expected outcome
Lux becomes a second real backend rather than a speculative scaffold.

### Tests to require before calling this stage done
- integration:
  - Lux forward pass shape test
  - Lux bundle load using shared parameter schema
  - Lux CPU generation smoke test

Later parity tests:
- Lux training sanity
- Lux checkpoint roundtrip

---

## Recommended milestones

Use these milestones rather than many tiny issues:

### Milestone A: CPU correctness
- Stage 1 complete

### Milestone B: Portable model reuse
- Stage 2 complete
- Stage 3 complete

### Milestone C: Supported training path
- Stage 4 complete

### Milestone D: User-facing proof of concept
- Stage 5 complete
- Stage 6 complete

### Milestone E: Backend resilience
- Stage 7 underway or complete

---

## What counts as "usable"

The package should be considered usable when all of the following are true:
- a tiny model can be trained with Flux on NVIDIA GPU
- the model can be checkpointed and resumed
- an inference bundle can be saved and loaded
- generation works on CPU
- a user can chat with a saved bundle from the REPL
- one official demo model can be resolved and loaded through the package API

That is the right proof-of-concept target for v0.1.

---

## What not to do too early

- Do not optimize performance before correctness is proven.
- Do not chase Flux/Lux parity before Flux is usable.
- Do not make CLI tooling a priority over REPL usage.
- Do not make Julia Artifacts the only distribution story.
- Do not take on non-NVIDIA GPU support in v0.1.
- Do not expand to more architectures before GPT-2 is stable.
