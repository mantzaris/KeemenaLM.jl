# BenchmarkDataNLP CFG complexity sweep

This is a small controlled sweep built on top of the existing CFG experiment pipeline.

Purpose:
- keep the model and training path fixed
- vary only CFG complexity
- check whether the current tiny GPT-2 setup still learns as synthetic grammar complexity rises

Sweep settings:
- generator: `CFG`
- complexities: `3`, `5`, `7`
- `num_sentences = 4_000` for every run
- `enable_polysemy = false`
- same tokenizer approach as the base experiment
- same Flux training/checkpoint/bundle/generation path
- same model:
  - `context_length = 48`
  - `num_layers = 2`
  - `num_heads = 2`
  - `embedding_size = 64`
  - `ffn_hidden_size = 128`
- same training:
  - `batch_size = 16`
  - `epochs = 2`
  - `learning_rate = 0.01f0`

Why this stops at `7` for the first pass:
- a quick fixed-sentence-count probe showed `complexity = 9` produces much larger text volume
- that would make the first sweep less interpretable as a clean complexity-only comparison

Run it:

```bash
julia --project=tools/benchmark_cfg tools/run_benchmark_cfg_sweep.jl
```

Outputs:
- one run directory per complexity under the sweep output root
- `summary.json`
- `summary.md`
