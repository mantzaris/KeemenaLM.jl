# Tiny Chatbot Real Subword Run v2

This is the throughput-aware serious retrain on the real downloaded conversational corpus.

## Why this run exists

The previous serious run only reached `3` epochs and still looked clearly undertrained.

The next step here was not to scale the model up again. It was to prioritize usable training throughput and get a more honestly trained baseline on the same real corpus and tokenizer path.

## Recipe choice

The current `6 x 384` fallback from the first serious run still looked clearly undertrained:

- training loss kept dropping strongly
- validation loss kept dropping strongly
- there was no saturation signal

So this retrain keeps the same serious fallback model and trains it longer instead of shrinking again.

## Completed recipe

- corpus: `tmp/tiny_chatbot_real_corpus_v1`
- tokenizer: `KeemenaSubwords.jl`
- trainer: `:hf_gpt2_bytebpe`
- vocab size: `4096`
- min frequency: `2`
- chat markers kept in text and also added as special tokens
- backend: `:flux`
- optimizer: `Flux.Adam(0.0003)`
- context length: `128`
- num layers: `6`
- num heads: `6`
- embedding size: `384`
- ffn hidden size: `1536`
- batch size: `8`

## Output

Run directory:

```text
tmp/tiny_chatbot_real_subword_run_v2
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

## Interpretation

This run is meant to answer a narrower question than the first serious run:

- is the current serious subword path mainly failing because it is undertrained?

It is not meant to claim chatbot quality by itself.
