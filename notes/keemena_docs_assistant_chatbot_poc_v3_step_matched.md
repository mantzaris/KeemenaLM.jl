# Keemena Docs Assistant Chatbot PoC on v3, Step-Matched

This run is the fairer budget-controlled follow-up for the cleaned `v3` chatbot dataset.

## Control method

- keep the boundary-aware char-level chatbot recipe fixed
- keep `tmp/keemena_docs_assistant_dataset_v3`
- increase epochs only enough to match the earlier `v2` boundary-aware chatbot baseline more closely by final optimizer steps

The `v2` baseline used:

- `30` epochs
- `70` train batches per epoch
- `2100` final steps

The `v3` dataset is smaller and yields:

- `40` train batches per epoch

So the step-matched follow-up uses:

- `53` epochs
- expected final step count: `2120`

## Fixed recipe

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

## Output directory

```text
tmp/keemena_docs_assistant_chatbot_poc_run_v3_step_matched
```

## Launch

```bash
julia --project=. tools/run_keemena_docs_assistant_chatbot_poc.jl \
  --dataset-dir tmp/keemena_docs_assistant_dataset_v3 \
  --output-dir tmp/keemena_docs_assistant_chatbot_poc_run_v3_step_matched \
  --epochs 53 \
  --experiment-name keemena_docs_assistant_chatbot_poc_run_v3_step_matched \
  --purpose "step-matched boundary-aware v3 chatbot retrain for fairer comparison against the v2 baseline"
```
