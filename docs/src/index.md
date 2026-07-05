# KeemenaLM.jl

KeemenaLM.jl is a Julia proof-of-concept for training and running small GPT-style
decoder language models from scratch. The package includes model configuration,
Flux and Lux backends, bundle IO, checkpointing, generation, chat helpers, and a
simple behavior-gate evaluator.

The active research focus is the scratch-chatbot pipeline used for the broad v9
336M baseline. That baseline demonstrates that the Julia training, export,
tokenizer sidecar, behavior evaluation, and REPL path work at a larger scale, but
it does not produce a reliable assistant.

## Current Supported State

- Flux instantiate, forward pass, generation, and training
- NVIDIA/CUDA training through the Flux path
- Lux instantiate, forward pass, shared weights, and CPU generation
- portable model bundles with JLD2 weights
- resumable checkpoints
- local official-model artifact registration
- chatbot behavior scoring helpers
- one-turn REPL for saved chatbot bundles

## Current Baseline

The strongest current scratch-chatbot run is the v9 broad 336M baseline:

- `336,488,448` estimated parameters
- `24` layers, `16` heads, `1024` embedding, `4096` FFN
- context length `512`
- about `1.5B` pretraining characters
- `39,584` SFT examples
- final test losses: `pretrain=2.0753`, `sft=3.1421`
- behavior gate: `17/22`

The model still fails basic common sense, arithmetic, factual grounding,
unknown-private-fact handling, and repetition checks. Treat it as a research
baseline, not a usable chatbot.

See [Current Chatbot Baseline](chatbot.md) for commands and interpretation.

## Next Research Direction

The next useful work is not a blind larger run. The current diagnosis is that the
project is data- and evaluation-limited. The recommended direction is a v10 data
and behavior-eval pass, then a short continuation or smoke run to verify that the
new data moves the known failures before spending another multi-day training run.

## Package API

See [API Reference](api.md) for exported package types and functions.
