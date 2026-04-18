# Prepared Better Local Real-Text Width Sweep Stage 2

Purpose: test whether model capacity is still the main bottleneck once the current best training budget is applied on the prepared better local real-text corpus.

## Sweep Design

Only one variable changes:

- `embedding_size`

The feed-forward width continues to scale as:

- `ffn_hidden_size = 2 * embedding_size`

Runs:

- `embedding_size = 128`, `ffn_hidden_size = 256`
- `embedding_size = 160`, `ffn_hidden_size = 320`
- `embedding_size = 192`, `ffn_hidden_size = 384`

## Fixed Settings

- backend `:flux`
- char-level tokenizer path
- prepared corpus: `tmp/better_local_real_text_corpus_prepared/dataset`
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `epochs = 6`
- `batch_size = 16`
- `learning_rate = 0.01f0`
- same deterministic seed style as the current best run

## Run

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_width_sweep_stage2.jl
```

Optional custom paths:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_width_sweep_stage2.jl \
  tmp/better_local_real_text_corpus_prepared/dataset \
  tmp/my_prepared_width_sweep_stage2
```

## Outputs

The sweep writes one directory per run plus:

- `summary.json`
- `summary.md`

The summary compares train/validation/test loss, training-budget counts, and sample-output paths against the current best `128 / 256`, `6`-epoch baseline.
