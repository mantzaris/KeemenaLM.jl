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

## Current Progress

The original staged proof-of-concept roadmap is complete through the planned v0.1 scope.

What has been demonstrated:
- the synthetic CFG benchmark phase ran end to end and is complete enough for this proof-of-concept stage
- controlled sweeps showed that CFG complexity hurts learning materially
- at the degraded synthetic point, extra epochs help only a little
- width helped more than depth under the fixed synthetic recipe
- local real-text sanity checks, prepared-corpus sweeps, optimizer sweeps, and final-run preflight all ran end to end using the same checkpoint, bundle, and reload path
- a first trained demo baseline was produced on the prepared better local real-text corpus using `Flux.Adam(0.001)` with `context_length = 48`, `embedding_size = 128`, `ffn_hidden_size = 256`, and `epochs = 38`

Current interpretation:
- the package pipeline is working
- the current trained baseline is a valid proof-of-concept training/export/load artifact
- qualitative generation is still weak, domain-narrow, and not chatbot-quality
- the corrected UltraChat GPU `v2_fixed` run reached validation loss `2.6447` and test loss `2.6688`, materially better than `v1`
- despite the numeric improvement, `v2_fixed` still drifts off-task and repeats, so it is not chatbot-quality
- large `v2_fixed` checkpoint/model weight artifacts were removed after inspection; see `notes/tiny_chatbot_ultrachat_subword_v2_fixed.md`

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

No-retraining decoding evaluation for the current UltraChat subword candidate:
```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_ultrachat_decoding_eval.jl
```

Interactive chat with the current UltraChat subword candidate:
```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_ultrachat_chat_repl.jl
```

## Notes

- Official model keys such as `tiny-demo` are supported through local artifact registration in this repo setup. They are not a fresh-user remote download path yet.
- Tokenizer and preprocessing objects are still supplied explicitly by the caller.
- Lux currently supports instantiate, forward pass, shared bundle weights, and CPU generation only.
- The current best local real-text baseline uses `Flux.Adam(0.001)` and produces a real trained demo artifact, but it is still not a strong conversational model.
- The next recommended work is either one more bounded Adam budget extension to look for flattening, or user-facing polish/documentation around the current trained baseline before broader data/tokenizer changes.
