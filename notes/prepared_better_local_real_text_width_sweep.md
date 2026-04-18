# Prepared Better Local Real-Text Width Sweep

Purpose: test whether model width is now the main bottleneck on the prepared better local real-text corpus while keeping the current char-level path unchanged.

## Sweep Design

Only one variable changes:

- `embedding_size`

The feed-forward width scales in the simplest consistent way:

- `ffn_hidden_size = 2 * embedding_size`

Runs:

- `embedding_size = 64`, `ffn_hidden_size = 128`
- `embedding_size = 96`, `ffn_hidden_size = 192`
- `embedding_size = 128`, `ffn_hidden_size = 256`

## Fixed Settings

- backend `:flux`
- char-level tokenizer path
- prepared corpus: `tmp/better_local_real_text_corpus_prepared/dataset`
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `epochs = 2`
- `batch_size = 16`
- `learning_rate = 0.01f0`
- same deterministic seed style as the prepared-corpus baseline

## Run

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_width_sweep.jl
```

Optional custom paths:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_width_sweep.jl \
  tmp/better_local_real_text_corpus_prepared/dataset \
  tmp/my_prepared_width_sweep
```

## Outputs

The sweep writes one directory per run plus:

- `summary.json`
- `summary.md`

The summary compares train/validation/test loss, run settings, training-budget counts, and sample-output paths against the current prepared-corpus char baseline.
