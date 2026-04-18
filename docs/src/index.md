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
- Small and larger local real-text sanity checks also completed successfully with the same training, checkpoint, bundle, reload, and generation path.
- Current real-text quality on the tiny local corpora is still weak, which points to data scale and variety as the next bottleneck rather than pipeline correctness.

## Immediate next focus

- keep the architecture stable
- improve the real-text corpus used for experiments
- rerun the same training, checkpoint, bundle, and generation path on better local real-text data before starting another major implementation stage

## API

See the generated API reference page for exported types and functions:
- [API Reference](api.md)
