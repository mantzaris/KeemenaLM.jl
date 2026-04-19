# Keemena Docs Assistant Chatbot PoC on v3

This run retrains the boundary-aware char-level chatbot proof of concept on the improved `v3` dataset.

## Dataset

- input dataset: `tmp/keemena_docs_assistant_dataset_v3`
- format: chat-style QA pairs rendered as

```text
User: ...
Assistant: ...
```

## Training recipe

- backend: Flux
- optimizer: `Flux.Adam(0.001)`
- tokenizer: experiment-local char tokenizer
- explicit training-stream separator: `\n<CHAT_END>\n`
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `batch_size = 16`
- epoch target: `30`

The recipe intentionally matches the boundary-aware `v2` chatbot baseline so the dataset is the main change. Because `v3` is smaller than `v2`, the final step count is expected to be lower at the same epoch target.

## Output directory

Default output directory:

```text
tmp/keemena_docs_assistant_chatbot_poc_run_v3
```

## Artifacts

- `checkpoints/`
- `checkpoints/final_checkpoint.jld2`
- `bundle/`
- `tokenizer.json`
- `metrics.json`
- `sample_outputs.txt`
- `evaluation_prompts.txt`
- `evaluation_prompts.json`
- `run_recipe.json`

## Launch

```bash
julia --project=. tools/run_keemena_docs_assistant_chatbot_poc.jl \
  --dataset-dir tmp/keemena_docs_assistant_dataset_v3 \
  --output-dir tmp/keemena_docs_assistant_chatbot_poc_run_v3 \
  --epochs 30
```

This remains a narrow technical-docs assistant experiment, not an open-domain chatbot benchmark.
