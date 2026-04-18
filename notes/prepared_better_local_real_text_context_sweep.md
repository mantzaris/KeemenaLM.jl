# Prepared Better Local Real-Text Context Sweep

Purpose: test whether longer context is the next meaningful bottleneck on the prepared better local real-text corpus at the current best width and training budget.

## Sweep Design

Only one variable changes:

- `context_length`

Runs:

- `context_length = 48`
- `context_length = 64`
- `context_length = 96`

## Fixed Settings

- backend `:flux`
- char-level tokenizer path
- prepared corpus: `tmp/better_local_real_text_corpus_prepared/dataset`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `epochs = 6`
- `batch_size = 16`
- `learning_rate = 0.01f0`
- same deterministic seed style as the current best prepared-corpus run

## Caveat

Changing context length also changes LM example construction and therefore train example counts and training-batch counts. This makes the comparison somewhat less tightly controlled than the width or budget sweeps.

## Run

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_context_sweep.jl
```

Optional custom paths:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_context_sweep.jl \
  tmp/better_local_real_text_corpus_prepared/dataset \
  tmp/my_prepared_context_sweep
```

## Outputs

The sweep writes one directory per run plus:

- `summary.json`
- `summary.md`

The summary compares train/validation/test loss, training-budget counts, and sample-output paths against the current `128 / 256`, `6`-epoch baseline.
