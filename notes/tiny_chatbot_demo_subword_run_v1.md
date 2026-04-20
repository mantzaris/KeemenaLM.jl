# Tiny Chatbot Demo Subword Run v1

This is the first proper subword-tokenized conversational chatbot run for the tiny demo path.

## Dataset

- input dataset: `tmp/tiny_chatbot_demo_dataset_v1`
- rendered format:

```text
User: ...
Assistant: ...
<END_ASSISTANT>
<CHAT_END>
```

## Tokenizer choice

- package: `KeemenaSubwords.jl`
- trainer: `:hf_gpt2_bytebpe`
- first-pass vocabulary size: `512`
- minimum frequency: `2`

## Chat-marker handling

The conversational markers stay present literally in the dataset text, but they are also trained as tokenizer added special tokens so each marker gets a single stable token id:

- `User:`
- `Assistant:`
- `<END_ASSISTANT>`
- `<CHAT_END>`

This is the chosen middle ground for the first proper subword path:

- better than letting the markers fragment into arbitrary subword pieces
- simpler than redesigning the model around automatic inserted BOS/EOS chat templates
- explicit and inspectable in the tokenizer bundle

The run still encodes dataset text with `add_special_tokens=false`; it does not rely on runtime auto-insertion. The markers are part of the text and part of the tokenizer vocabulary.

## Training recipe

- backend: Flux
- optimizer: `Flux.Adam(0.001)`
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `batch_size = 16`
- first-pass epoch target: `50`

`50` epochs is a bounded first-pass target chosen to roughly step-match the earlier char-level conversational baseline. A quick token-count probe showed the subword path yields far fewer train batches per epoch, so keeping `24` epochs would have been an unfairly under-budgeted comparison.

## Output directory

Default output directory:

```text
tmp/tiny_chatbot_demo_subword_run_v1
```

## Artifacts

- `checkpoints/`
- `checkpoints/final_checkpoint.jld2`
- `bundle/`
- `tokenizer_bundle/`
- `metrics.json`
- `sample_outputs.txt`
- `evaluation_prompts.txt`
- `evaluation_prompts.json`
- `run_recipe.json`

## Launch

```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_demo_subword_v1.jl
```

Tokenizer artifacts are still saved separately from the KeemenaLM model bundle.
