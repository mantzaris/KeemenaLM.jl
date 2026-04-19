# Prepared Better Local Real-Text Adam Budget Sweep

Purpose: test whether the new Adam baseline is still under-trained beyond `22` epochs or whether it is beginning to approach saturation.

This sweep keeps fixed:
- corpus: `tmp/better_local_real_text_corpus_prepared/dataset`
- backend: `:flux`
- optimizer family: `Flux.Adam`
- optimizer hyperparameters: `learning_rate = 0.001`
- tokenizer path: char-level experiment-local tokenizer
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `batch_size = 16`
- same deterministic seed style as earlier prepared-corpus runs

This sweep varies only:
- `epochs = 22`
- `epochs = 26`
- `epochs = 30`

Run with:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_adam_budget_sweep.jl
```

Optional custom output directory:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_adam_budget_sweep.jl tmp/better_local_real_text_corpus_prepared/dataset tmp/prepared_better_local_real_text_adam_budget_sweep_custom
```

Outputs:
- per-run directories under `tmp/prepared_better_local_real_text_adam_budget_sweep/`
- `summary.json`
- `summary.md`

Interpretation target:
- if validation and test keep improving, Adam still has useful budget headroom
- if gains flatten or reverse, the setup may be approaching saturation and a final long run becomes more defensible
