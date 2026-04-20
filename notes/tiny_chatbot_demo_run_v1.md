# Tiny Chatbot Demo Run v1

This run is the first tiny conversational chatbot demo training path.

## Dataset

- input dataset: `tmp/tiny_chatbot_demo_dataset_v1`
- format:

```text
User: ...
Assistant: ...
<END_ASSISTANT>
<CHAT_END>
```

The dataset already carries explicit answer and chat-end markers, so the training path does not inject another `<CHAT_END>` separator between examples. It only joins samples with a blank-line separator.

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
- first conversational demo epoch target: `24`

`24` epochs is a bounded first-pass target for this larger conversational dataset. It keeps the first demo run practical while still giving the model enough passes to learn the explicit chat markers and the assistant-style local core.

## Output directory

Default output directory:

```text
tmp/tiny_chatbot_demo_run_v1
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
julia --project=. tools/run_tiny_chatbot_demo_v1.jl
```

This is a conversational demo run, not a docs-assistant benchmark and not a production chatbot recipe.
