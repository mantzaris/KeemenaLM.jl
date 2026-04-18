# BenchmarkDataNLP CFG controlled complexity sweep

This is a cleaner complexity comparison built on top of the existing CFG experiment pipeline.

Purpose:
- revisit the earlier CFG complexity comparison
- control effective training volume more carefully than the first sentence-count sweep
- measure how much complexity itself still hurts learning after that control

Control method:
- match the **training token stream length**
- use a target of `72_001` training tokens for every run
- with `context_length = 48`, that also yields exactly `1_500` LM examples per run

Sweep settings:
- generator: `CFG`
- complexities: `3`, `5`, `7`
- fixed `num_sentences = 4_000` at dataset generation time
- fixed model:
  - `context_length = 48`
  - `num_layers = 2`
  - `num_heads = 2`
  - `embedding_size = 64`
  - `ffn_hidden_size = 128`
- fixed training:
  - backend `:flux`
  - `epochs = 2`
  - `batch_size = 16`
  - `learning_rate = 0.01f0`
- same tokenizer approach as the base experiment
- same Flux training/checkpoint/bundle/generation path

Why this is cleaner than the first complexity sweep:
- the first sweep kept sentence count fixed
- sentence length changed substantially with complexity
- that changed both the training token volume and the number of LM examples
- this sweep holds the training token stream constant instead

Run it:

```bash
julia --project=tools/benchmark_cfg tools/run_benchmark_cfg_controlled_complexity_sweep.jl
```

Outputs:
- one run directory per complexity under the sweep output root
- `summary.json`
- `summary.md`
