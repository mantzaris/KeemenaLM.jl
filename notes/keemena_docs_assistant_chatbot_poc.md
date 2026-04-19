# Keemena Docs Assistant Chatbot PoC

This run is the first narrow-domain chatbot-style proof of concept trained on the cleaned `Keemena Docs Assistant` dataset.

## Dataset

- input dataset: `tmp/keemena_docs_assistant_dataset_v2`
- format: chat-style QA pairs rendered as

```text
User: ...
Assistant: ...
```

## Training recipe

- backend: Flux
- optimizer: `Flux.Adam(0.001)`
- tokenizer: experiment-local char tokenizer
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `batch_size = 16`
- first PoC epoch target: `30`

## Output directory

Default output directory:

```text
tmp/keemena_docs_assistant_chatbot_poc_run_with_boundaries
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
julia --project=. tools/run_keemena_docs_assistant_chatbot_poc.jl
```

This run is intended to produce the first chatbot-style artifact for a narrow technical docs assistant. It is not an open-domain chatbot benchmark and it should not be interpreted as a polished conversational model.

## Boundary handling update

The chatbot dataset itself is kept unchanged. The training path now inserts an explicit deterministic separator between chat examples when building the LM token stream:

```text
<CHAT_END>
```

This makes example boundaries more visible than a bare newline join and is intended to give the char-level model a cleaner signal for `User:` / `Assistant:` turn structure without redesigning the package or changing the core model recipe.
