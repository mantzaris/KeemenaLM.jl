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

## Current project progress

- The original staged proof-of-concept roadmap is complete through the planned v0.1 scope.
- The synthetic CFG benchmark phase completed successfully and established the basic learning pattern for the tiny model.
- Controlled sweeps showed that complexity hurts learning materially, extra epochs help only a little at the degraded point, and width helped more than depth under the fixed synthetic recipe.
- Prepared-corpus real-text sweeps identified `Flux.Adam(0.001)` as a much better training path than plain gradient descent for the current tiny model.
- A first trained demo baseline was completed on the prepared better local real-text corpus at `context_length = 48`, `embedding_size = 128`, `ffn_hidden_size = 256`, and `epochs = 38`.
- Current real-text quality is still weak and domain-narrow, so this baseline is best understood as a proof-of-concept trained artifact rather than a good chatbot.

## Immediate next focus

- keep the architecture stable
- either run one more bounded Adam budget extension to check for flattening, or use the current trained baseline for user-facing demo/documentation work
- treat broader corpus/tokenizer changes as the next quality-improvement branch after the current baseline is fully documented and evaluated

## API

See the generated API reference page for exported types and functions:
- [API Reference](api.md)
