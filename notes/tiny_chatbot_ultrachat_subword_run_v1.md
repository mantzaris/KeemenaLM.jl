# Tiny Chatbot UltraChat Subword Run v1

This is the first serious UltraChat-based subword conversational training run.

## Corpus

- source dataset directory: `tmp/tiny_chatbot_ultrachat_corpus_v1`
- source corpus: `HuggingFaceH4/ultrachat_200k`
- selected source splits:
  - `train_sft`
  - `test_sft`

## Tokenizer

- package: `KeemenaSubwords.jl`
- trainer: `:hf_gpt2_bytebpe`
- vocab size: `8192`
- min frequency: `2`
- chat markers remain in the text and are also added as special tokens:
  - `User:`
  - `Assistant:`
  - `<END_ASSISTANT>`
  - `<CHAT_END>`

## Model

- backend: `:flux`
- optimizer: `Flux.Adam(0.0003)`
- `context_length = 128`
- `num_layers = 6`
- `num_heads = 6`
- `embedding_size = 384`
- `ffn_hidden_size = 1536`
- `batch_size = 16`
- `epochs = 1`

## Bounded throughput policy

The full UltraChat corpus is too large to run as a full-epoch local training job in a bounded way with the current runner and environment, so this first serious run used deterministic bounded slices:

- tokenizer training texts: `20000`
- train texts: `3000`
- validation texts: `500`
- test texts: `500`

An earlier larger launch at `6000 / 1000 / 1000` was abandoned because it was still too slow to complete a bounded single-epoch run here.

## Result

- final step count: `1670`
- train loss: `6.4555`
- validation loss: `5.9185`
- test loss: `5.9370`

## Readout

This run is clearly beyond the earlier toy and tiny-corpus stages in terms of source corpus scale, tokenizer path, and general setup seriousness.

But the qualitative outputs are still weak. The model learned some assistant-shell behavior and common reply starters like `Certainly.` and `Yes,`, but it still collapses into repetitive fragments, numbered-list loops, and malformed continuations instead of giving stable useful answers.

So this is a more serious conversational baseline, but not yet a good chatbot.
