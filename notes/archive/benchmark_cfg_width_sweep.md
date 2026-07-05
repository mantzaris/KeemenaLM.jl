# BenchmarkDataNLP CFG width sweep

This is a small controlled model-width sweep built on top of the existing CFG experiment pipeline.

Purpose:
- hold the degraded point fixed at `complexity = 5`
- vary only embedding width
- check whether the remaining degradation looks capacity-limited

Sweep settings:
- generator: `CFG`
- fixed `complexity = 5`
- fixed `num_sentences = 4_000`
- fixed training:
  - backend `:flux`
  - `epochs = 2`
  - `batch_size = 16`
  - `learning_rate = 0.01f0`
- same tokenizer approach as the base experiment
- same Flux training/checkpoint/bundle/generation path
- same depth and heads:
  - `context_length = 48`
  - `num_layers = 2`
  - `num_heads = 2`
- widths:
  - `embedding_size = 32`, `ffn_hidden_size = 64`
  - `embedding_size = 64`, `ffn_hidden_size = 128`
  - `embedding_size = 96`, `ffn_hidden_size = 192`

Why FFN scales:
- the sweep varies width only in practice
- `ffn_hidden_size` keeps the same `2x` ratio used by the current baseline model

Run it:

```bash
julia --project=tools/benchmark_cfg tools/run_benchmark_cfg_width_sweep.jl
```

Outputs:
- one run directory per width under the sweep output root
- `summary.json`
- `summary.md`
