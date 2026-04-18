# Prepared Better Local Real-Text Budget Sweep

Purpose: test whether additional training budget helps more than further architecture tweaks at the current best width on the prepared better local real-text corpus.

## Sweep Design

Only one variable changes:

- `epochs`

Runs:

- `epochs = 2`
- `epochs = 4`
- `epochs = 6`

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
- same deterministic seed style as the prepared-corpus baseline

## Run

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_budget_sweep.jl
```

Optional custom paths:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_budget_sweep.jl \
  tmp/better_local_real_text_corpus_prepared/dataset \
  tmp/my_prepared_budget_sweep
```

## Outputs

The sweep writes one directory per run plus:

- `summary.json`
- `summary.md`

The summary compares train/validation/test loss, training-budget counts, and sample-output paths against the current `128 / 256` width baseline.
