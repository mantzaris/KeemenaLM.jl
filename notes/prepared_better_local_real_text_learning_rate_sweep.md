# Prepared Better Local Real-Text Learning-Rate Sweep

Purpose: test whether a small learning-rate change improves convergence more efficiently than simply adding more epochs at the current best real-text recipe.

This sweep keeps fixed:
- corpus: `tmp/better_local_real_text_corpus_prepared/dataset`
- backend: `:flux`
- tokenizer path: char-level experiment-local tokenizer
- optimizer family: `Flux.Descent`
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `batch_size = 16`
- `epochs = 22`
- same deterministic seed style as earlier prepared-corpus runs

This sweep varies only:
- `learning_rate = 0.01`
- `learning_rate = 0.005`
- `learning_rate = 0.002`

Run with:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_learning_rate_sweep.jl
```

Optional custom output directory:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_learning_rate_sweep.jl tmp/better_local_real_text_corpus_prepared/dataset tmp/prepared_better_local_real_text_learning_rate_sweep_custom
```

Outputs:
- per-run directories under `tmp/prepared_better_local_real_text_learning_rate_sweep/`
- `summary.json`
- `summary.md`

Interpretation target:
- if a smaller learning rate beats the current `22`-epoch baseline, optimization is now a stronger bottleneck than raw budget
- if it does not, then continuing budget or a different optimizer family may be more promising than a simple LR reduction
