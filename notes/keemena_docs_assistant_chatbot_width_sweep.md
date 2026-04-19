# Keemena Docs Assistant Chatbot Width Sweep

This sweep tests whether modestly wider models help the boundary-aware char-level chatbot path.

## Fixed setup

- dataset: `tmp/keemena_docs_assistant_dataset_v2`
- tokenizer: experiment-local char tokenizer
- training-stream separator:

```text
<CHAT_END>
```

- optimizer: `Flux.Adam(0.001)`
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `batch_size = 16`
- `epochs = 30`

## Width runs

- `embedding_size = 128`, `ffn_hidden_size = 256`
- `embedding_size = 160`, `ffn_hidden_size = 320`
- `embedding_size = 192`, `ffn_hidden_size = 384`

`ffn_hidden_size` continues to scale as `2 * embedding_size`.

## Output

Default output directory:

```text
tmp/keemena_docs_assistant_chatbot_width_sweep
```

The sweep writes per-run output directories plus:

- `summary.json`
- `summary.md`

This is a bounded capacity comparison only. It does not change tokenizer path, optimizer family, context length, or boundary handling.
