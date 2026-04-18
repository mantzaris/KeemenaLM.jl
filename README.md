# KeemenaLM.jl

[![CI](https://github.com/mantzaris/KeemenaLM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mantzaris/KeemenaLM.jl/actions/workflows/CI.yml)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://mantzaris.github.io/KeemenaLM.jl/dev/)

KeemenaLM.jl is a Julia proof-of-concept language-model package centered on a small GPT-2 style decoder-only model with portable bundles, resumable checkpoints, REPL chat, and a second inference backend.

## Status

Current supported state:
- Flux inference on CPU
- Flux training path, including NVIDIA/CUDA training support
- portable inference bundles
- resumable training checkpoints
- REPL chat from a saved bundle
- official demo model flow through local Julia artifact registration
- Lux inference parity on CPU

Not yet supported:
- Lux training parity
- tokenizer/preprocessing persistence inside bundles
- remote official model hosting or download integration

## Supported Workflows

Training and export with Flux:
```bash
julia --project=. examples/train_tiny_gpt2_flux.jl
```

Register the official local demo artifact:
```bash
julia --project=. tools/build_public_model_artifact.jl
```

One-turn chat from a bundle directory or official model key:
```bash
julia --project=. examples/chat_demo.jl tiny-demo
```

REPL chat from a bundle directory or official model key:
```bash
julia --project=. examples/chat_repl.jl tiny-demo
```

## Notes

- Official model keys such as `tiny-demo` are supported through local artifact registration in this repo setup. They are not a fresh-user remote download path yet.
- Tokenizer and preprocessing objects are still supplied explicitly by the caller.
- Lux currently supports instantiate, forward pass, shared bundle weights, and CPU generation only.
