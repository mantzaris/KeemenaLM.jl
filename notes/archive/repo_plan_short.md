## KeemenaLM.jl concise package summary (aligned plan)

Status note:
- the original v0.1 staged roadmap is complete through Stage 7
- this file should now be read as an architecture and scope reference, not as a pending execution checklist

### What KeemenaLM.jl is
KeemenaLM.jl is the model layer in the Keemena stack. It provides Julia-native decoder-only transformer language models, training and inference utilities, and portable model bundles for reuse later.

Initial target: a readable GPT-2 style decoder-only transformer in pure Julia. The goal is proof of concept and community groundwork, not ChatGPT-class capability.

### The golden path (v0.1)
1) Train a tiny GPT-2 style model with the Flux backend on NVIDIA GPU.
2) Save:
   - a resumable training checkpoint
   - a portable inference bundle
3) Load the inference bundle from a local path or Julia artifact.
4) Generate text and run a minimal REPL chat session on CPU or NVIDIA GPU.

### What v0.1 includes
- Architecture:
  - GPT-2 style decoder-only transformer
- Backend architecture:
  - all public APIs accept `backend::Symbol`
  - Flux is the supported v0.1 backend
  - Lux has a reserved backend path and module layout from day one, but may remain partial/stubbed in v0.1
- Core, framework-neutral logic:
  - causal masking
  - sampling (`temperature`, `top_k`, `top_p`)
  - stopping logic
  - generation loop
  - minimal chat wrapper
  - bundle schema and bundle IO
  - bundle source resolution
  - checkpoint metadata
- Runtime policy:
  - inference must work on CPU
  - inference may also run on NVIDIA GPU
  - training is assumed to run on NVIDIA GPU in v0.1

### What v0.1 does not include
- Flux/Lux feature parity
- non-NVIDIA GPU support
- KV cache and fast decoding
- streaming generation callbacks
- full evaluation suite
- ONNX export or inference

### Backend boundary rule
- `src/core/` must not import Flux or Lux.
- Only `src/backends/flux/` and `src/backends/lux/` may import Flux/Lux.
- Core talks to backends through a small interface defined in core.

### Weights, checkpoints, and bundles
- Weights are not committed to git.
- v0.1 weight format is `JLD2`.
- Keep two persistence formats:
  - checkpoint: for resuming training, includes optimizer state and counters
  - bundle: for user-facing inference and sharing
- Small official demo bundles can be distributed through Julia Artifacts.
- The loader design should treat Artifacts as one source among several:
  - local directory
  - Julia artifact
  - direct URL later
  - Hugging Face-style remote source later

### Integration stance
- `KeemenaSubwords.jl` is the preferred tokenizer integration.
- `KeemenaPreprocessing.jl` is useful for corpus preparation, but should remain optional for basic model loading and chat.
