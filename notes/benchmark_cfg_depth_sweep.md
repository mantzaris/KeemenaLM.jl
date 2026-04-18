# BenchmarkDataNLP CFG depth sweep

This is a small controlled model-depth sweep built on top of the existing CFG experiment pipeline.

Purpose:
- hold the degraded point fixed at `complexity = 5`
- vary only depth
- check whether the remaining degradation is better addressed by more sequential depth than by more width

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
- fixed width:
  - `embedding_size = 64`
  - `ffn_hidden_size = 128`
- fixed:
  - `context_length = 48`
  - `num_heads = 2`
- depths:
  - `num_layers = 2`
  - `num_layers = 3`
  - `num_layers = 4`

Why this width:
- it is the established baseline width
- the earlier width sweep showed `64 -> 96` helped only a little, so `64` is the cleanest fixed reference point for isolating depth effects

Run it:

```bash
julia --project=tools/benchmark_cfg tools/run_benchmark_cfg_depth_sweep.jl
```

Outputs:
- one run directory per depth under the sweep output root
- `summary.json`
- `summary.md`
