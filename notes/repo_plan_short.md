## KeemenaLM.jl concise package summary (revised)

### What KeemenaLM.jl is
KeemenaLM.jl is the model layer in the Keemena stack. It provides Julia-native transformer language models (decoder-only first), plus the utilities needed to train, run inference, and package models as reusable bundles.

Initial target (v0.1): a simple, correct GPT-2 style decoder-only transformer that the community can read and extend.

### The golden path (v0.1)
1) Train a tiny GPT-2 model (Flux or Lux).
2) Save a bundle (JSON config + JLD2 weights + tokenizer/preproc references).
3) Load the bundle.
4) Generate text and run a minimal chat session.

### What v0.1 includes
- Architecture: GPT-2 style decoder-only transformer
- Backends: Flux and Lux model implementations
- Core (framework-neutral):
  - causal mask utilities
  - sampling (temperature, top_k, top_p)
  - stop conditions (EOS; optional token-sequence stops)
  - generation loop and minimal chat wrapper
  - bundle schema and bundle save/load
- Training: minimal training step sanity (not a full training framework yet)

### What v0.1 does not include (deferred)
- KV cache and fast decoding
- Streaming generation callbacks
- Full training orchestration, checkpointing, and evaluation suites
- ONNX export or ONNX inference

### Backend boundary rule (important)
- `src/core/` must not import Flux or Lux.
- Only `src/backends/flux/` and `src/backends/lux/` may import Flux/Lux.
- Core calls model execution through a small interface (multiple dispatch), so backends stay thin and separable later.

### Bundles and weights policy
- Weights are not committed to git.
- v0.1 weight format: JLD2.
- Bundle format uses JSON for portability and future-proofing:
  - `bundle.json` (manifest, schema version)
  - `model_config.json` (GPT2Config)
  - `weights.jld2` (portable weight dictionary)
  - tokenizer and preprocessing specs or references (supported by schema; adapters can evolve)
  - optional `README.md` model card