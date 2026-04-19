# Keemena Docs Assistant Subword Chatbot PoC

This run is a single tokenizer-path comparison against the boundary-aware char-level chatbot baseline.

## Intent

- same chatbot dataset: `tmp/keemena_docs_assistant_dataset_v2`
- same optimizer family: `Flux.Adam(0.001)`
- same small GPT-2 shape
- same explicit training-stream separator:

```text
<CHAT_END>
```

- tokenizer path changes from the experiment-local char tokenizer to a single subword tokenizer path from `KeemenaSubwords.jl`

## Tokenizer choice

- package: `KeemenaSubwords.jl`
- trainer: `:hf_gpt2_bytebpe`
- requested vocabulary size: `384`
- minimum frequency: `2`

This keeps the comparison decoder-friendly and uses the same local tokenizer tooling that already exists in the repository.

## Output

Run with the dedicated subword experiment environment:

```bash
julia --project=tools/subword_real_text tools/run_keemena_docs_assistant_chatbot_subword_poc.jl
```

Default output directory:

```text
tmp/keemena_docs_assistant_chatbot_subword_poc_run
```

Artifacts written:

- `checkpoints/`
- `checkpoints/final_checkpoint.jld2`
- `bundle/`
- `tokenizer_bundle/`
- `metrics.json`
- `sample_outputs.txt`
- `evaluation_prompts.txt`
- `evaluation_prompts.json`
- `run_recipe.json`

## Scope note

This remains a bounded tokenizer comparison only. Tokenizer artifacts are still stored separately from the KeemenaLM model bundle, and this run should be compared against the boundary-aware char chatbot baseline with the understanding that token-level loss is not perfectly apples-to-apples across different tokenizations.
