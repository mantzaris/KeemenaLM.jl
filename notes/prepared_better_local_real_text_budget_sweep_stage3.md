# Prepared Better Local Real-Text Budget Sweep Stage 3

Purpose: test whether the current best char-level real-text recipe is still improving materially beyond 10 epochs, or whether it is approaching saturation.

## Sweep Design

Only one variable changes:

- `epochs`

Runs:

- `epochs = 10`
- `epochs = 12`
- `epochs = 14`

## Fixed Settings

- backend `:flux`
- char-level tokenizer path
- prepared corpus: `tmp/better_local_real_text_corpus_prepared/dataset`
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `batch_size = 16`
- `learning_rate = 0.01f0`
- same deterministic seed style as the current best run

## Run

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_budget_sweep_stage3.jl
```

Optional custom paths:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_budget_sweep_stage3.jl \
  tmp/better_local_real_text_corpus_prepared/dataset \
  tmp/my_prepared_budget_sweep_stage3
```

## Outputs

The sweep writes one directory per run plus:

- `summary.json`
- `summary.md`

The summary compares train/validation/test loss, training-budget counts, and sample-output paths against the current best `128 / 256`, `10`-epoch baseline.
