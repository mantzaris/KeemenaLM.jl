# Tiny Chatbot Real Subword Run v1

This is the first serious subword conversational training run on the real downloaded OASST1-dominant corpus.

## Tokenizer path

- package: `KeemenaSubwords.jl`
- trainer: `:hf_gpt2_bytebpe`
- vocab size: `4096`
- min frequency: `2`

Chat markers stay literally in the dataset text and are also trained as added special tokens with stable single ids:

- `User:`
- `Assistant:`
- `<END_ASSISTANT>`
- `<CHAT_END>`

## Model/run policy

Intended target attempted first:

- `context_length = 128`
- `num_layers = 8`
- `num_heads = 8`
- `embedding_size = 512`
- `ffn_hidden_size = 2048`

That exact model fit, but epoch throughput was too slow for a bounded single run in this environment.

So the one fallback used for the completed run was:

- `context_length = 128`
- `num_layers = 6`
- `num_heads = 6`
- `embedding_size = 384`
- `ffn_hidden_size = 1536`
- `batch_size = 8`
- `optimizer = Flux.Adam(0.0003)`
- `epochs = 3`

## Output

Run directory:

```text
tmp/tiny_chatbot_real_subword_run_v1
```

Artifacts:

- `checkpoints/`
- `checkpoints/final_checkpoint.jld2`
- `bundle/`
- `tokenizer_bundle/`
- `metrics.json`
- `sample_outputs.txt`
- `evaluation_prompts.txt`
- `evaluation_prompts.json`
- `run_recipe.json`

## Result

The run completed end to end, but the saved samples still collapse badly. This is a real non-toy chatbot training path for the repo, but not yet a good conversational model.
