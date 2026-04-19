# Prepared Better Local Real-Text Optimizer Sweep

Purpose: test whether optimizer family now matters more than raw extra budget at the current best char-level real-text recipe.

This sweep keeps fixed:
- corpus: `tmp/better_local_real_text_corpus_prepared/dataset`
- backend: `:flux`
- tokenizer path: char-level experiment-local tokenizer
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `batch_size = 16`
- `epochs = 22`
- same deterministic seed style as earlier prepared-corpus runs

This sweep varies only:
- optimizer family

Runs:
- `Flux.Descent(learning_rate = 0.01)`
- `Flux.Adam(learning_rate = 0.001)`

Comparison policy:
- optimizer-specific standard learning rates
- this is a first family comparison, not a full per-optimizer LR tuning study

Run with:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_optimizer_sweep.jl
```

Optional custom output directory:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_optimizer_sweep.jl tmp/better_local_real_text_corpus_prepared/dataset tmp/prepared_better_local_real_text_optimizer_sweep_custom
```

Outputs:
- per-run directories under `tmp/prepared_better_local_real_text_optimizer_sweep/`
- `summary.json`
- `summary.md`

Interpretation target:
- if `Adam` clearly beats the `Flux.Descent` baseline, optimizer dynamics are now a stronger bottleneck than raw extra epochs
- if it does not, then more budget or a later LR schedule may still be the cleaner next step
