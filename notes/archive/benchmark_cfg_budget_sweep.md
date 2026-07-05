# BenchmarkDataNLP CFG budget sweep

This is a small controlled budget sweep built on top of the existing CFG experiment pipeline.

Purpose:
- hold the degraded point from the first complexity sweep fixed at `complexity = 5`
- vary only training budget
- check whether the earlier degradation looks more like under-training than a pipeline failure

Sweep settings:
- generator: `CFG`
- fixed `complexity = 5`
- fixed `num_sentences = 4_000`
- `enable_polysemy = false`
- same tokenizer approach as the base experiment
- same Flux training/checkpoint/bundle/generation path
- same model:
  - `context_length = 48`
  - `num_layers = 2`
  - `num_heads = 2`
  - `embedding_size = 64`
  - `ffn_hidden_size = 128`
- same training except epochs:
  - epochs: `1`, `2`, `4`
  - `batch_size = 16`
  - `learning_rate = 0.01f0`

Run it:

```bash
julia --project=tools/benchmark_cfg tools/run_benchmark_cfg_budget_sweep.jl
```

Outputs:
- one run directory per epoch value under the sweep output root
- `summary.json`
- `summary.md`
