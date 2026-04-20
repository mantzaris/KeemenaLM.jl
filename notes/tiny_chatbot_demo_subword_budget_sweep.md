# Tiny Chatbot Demo Subword Budget Sweep

This is a bounded budget-tuning pass on the first proper subword conversational chatbot path.

## Question

Was the original `50`-epoch subword conversational run simply overtrained, and does the subword path look more viable in the earlier validation-minimum range?

## Fixed recipe

- dataset: `tmp/tiny_chatbot_demo_dataset_v1`
- tokenizer package: `KeemenaSubwords.jl`
- tokenizer trainer: `:hf_gpt2_bytebpe`
- `vocab_size = 512`
- `min_frequency = 2`
- chat markers kept literally in the text and also trained as added special tokens:
  - `User:`
  - `Assistant:`
  - `<END_ASSISTANT>`
  - `<CHAT_END>`
- backend: Flux
- optimizer: `Flux.Adam(0.001)`
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `batch_size = 16`

## Sweep

Vary only:

- `epochs`

Runs:

- `28`
- `30`
- `32`

This range is narrower than the original suggestion because the first subword run's validation curve bottomed out around the high 20s / low 30s, with clearer degradation later.

## Output

Default sweep root:

```text
tmp/tiny_chatbot_demo_subword_budget_sweep
```

Summary files:

- `summary.json`
- `summary.md`

Each run writes the full artifact set in its own subdirectory.
