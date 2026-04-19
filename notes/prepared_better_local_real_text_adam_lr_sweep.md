# Prepared Better Local Real-Text Adam LR Sweep

Purpose: test whether the new Adam baseline can be improved further with a small Adam-only learning-rate tuning pass before any longer final training run.

This sweep keeps fixed:
- corpus: `tmp/better_local_real_text_corpus_prepared/dataset`
- backend: `:flux`
- optimizer family: `Flux.Adam`
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
- Adam learning rate

Runs:
- `0.001`
- `0.0005`
- `0.0002`

Run with:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_adam_lr_sweep.jl
```

Optional custom output directory:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_adam_lr_sweep.jl tmp/better_local_real_text_corpus_prepared/dataset tmp/prepared_better_local_real_text_adam_lr_sweep_custom
```

Outputs:
- per-run directories under `tmp/prepared_better_local_real_text_adam_lr_sweep/`
- `summary.json`
- `summary.md`

Interpretation target:
- if one Adam LR clearly beats the current `0.001` Adam baseline, use that for any longer follow-up training
- if `0.001` remains best, the next step should be a budget extension on the current Adam baseline rather than more immediate LR tinkering
