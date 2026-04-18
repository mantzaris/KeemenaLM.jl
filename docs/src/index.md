# KeemenaLM.jl

KeemenaLM.jl is a Julia proof-of-concept language-model package for a small GPT-2 style decoder-only model.

## Supported v0.1 state

- Flux inference on CPU
- Flux training path, including checkpoints and NVIDIA/CUDA support
- portable bundles and bundle load/save
- REPL chat from a saved bundle
- official demo model flow through local artifact registration
- Lux inference on CPU using the shared portable weight schema

## Not yet supported

- Lux training parity
- tokenizer/preprocessing persistence inside bundles
- remote official model hosting or download integration

## API

See the generated API reference page for exported types and functions:
- [API Reference](api.md)
