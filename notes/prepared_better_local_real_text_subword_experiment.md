# Prepared Better Local Real-Text Subword Experiment

Purpose: test whether tokenizer quality is now the main bottleneck on the prepared better local real-text corpus, without changing the current Flux training, checkpoint, bundle, and generation path more than necessary.

## Tokenizer Path

This experiment uses `KeemenaSubwords.jl` directly in a dedicated local experiment environment.

Chosen path:

- tokenizer trainer: `:hf_gpt2_bytebpe`
- package: `KeemenaSubwords.jl`
- tokenizer artifact persistence: separate tokenizer training bundle under the experiment output directory

Why this path:

- it is local and inspectable
- it matches a decoder-style LM better than a character tokenizer
- it gives a reproducible tokenizer bundle via `quick_train_bundle`
- it avoids redesigning KeemenaLM bundle/checkpoint formats

Important current limitation:

- the tokenizer is still saved separately from the KeemenaLM model bundle
- this experiment does **not** imply that tokenizer payload persistence is solved in the main package format

## Environment

Instantiate the dedicated environment:

```bash
julia --project=tools/subword_real_text -e 'using Pkg; Pkg.instantiate()'
```

## Run

```bash
julia --project=tools/subword_real_text tools/run_prepared_better_local_real_text_subword_experiment.jl
```

Optional custom paths:

```bash
julia --project=tools/subword_real_text tools/run_prepared_better_local_real_text_subword_experiment.jl \
  tmp/better_local_real_text_corpus_prepared/dataset \
  tmp/my_prepared_corpus_subword_run
```

## Fixed Model And Training Settings

- backend `:flux`
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 64`
- `ffn_hidden_size = 128`
- `epochs = 2`
- `batch_size = 16`
- `learning_rate = 0.01f0`

## Outputs

The run writes:

- tokenizer training bundle
- model checkpoint(s)
- model bundle
- `metrics.json`
- `sample_outputs.txt`

This experiment should be compared directly against the char-level prepared-corpus baseline to judge whether subword tokenization materially improves loss and sample quality, and to record any practical integration pain points.

## Fairer Follow-Up

If the first subword run yields fewer optimizer steps than the char baseline, rerun with an explicit step budget:

```bash
julia --project=tools/subword_real_text tools/run_prepared_better_local_real_text_subword_experiment.jl \
  --epochs 3 \
  --target-final-step 160
```

This keeps the tokenizer family, model, and corpus fixed while matching the char prepared-corpus baseline on optimizer-step budget more closely.
