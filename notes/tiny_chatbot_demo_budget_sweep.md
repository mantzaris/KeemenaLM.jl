# Tiny Chatbot Demo Budget Sweep

This is a bounded budget-extension study on the first tiny conversational chatbot baseline.

## Question

Does the conversational chatbot baseline still improve materially with more training budget, or is it already close to saturation after `24` epochs?

## Fixed recipe

- dataset: `tmp/tiny_chatbot_demo_dataset_v1`
- backend: Flux
- optimizer: `Flux.Adam(0.001)`
- tokenizer: experiment-local char tokenizer
- explicit in-data markers:
  - `User:`
  - `Assistant:`
  - `<END_ASSISTANT>`
  - `<CHAT_END>`
- training-time join separator: blank line (`\n\n`)
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

- `24`
- `32`
- `40`

## Output

Default sweep root:

```text
tmp/tiny_chatbot_demo_budget_sweep
```

Summary files:

- `summary.json`
- `summary.md`

Each run writes the full artifact set in its own subdirectory.
