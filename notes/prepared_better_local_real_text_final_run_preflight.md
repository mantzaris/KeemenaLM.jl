# Prepared Better Local Real-Text Final Run Preflight And Baseline

Purpose: verify that the current best real-text recipe is ready for one explicit best-effort training run, and record the resulting first trained demo baseline.

## Intended Final Recipe

- corpus: `tmp/better_local_real_text_corpus_prepared/dataset`
- backend: `:flux`
- optimizer family: `Flux.Adam`
- optimizer hyperparameters: `learning_rate = 0.001`
- tokenizer path: char-level experiment-local tokenizer
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `batch_size = 16`
- intended full-run baseline: `epochs = 38`
- deterministic seed style unchanged from the earlier prepared-corpus runs

## Recommended Output Layout

For the eventual full run:
- `checkpoints/`
  - one checkpoint per epoch
  - `final_checkpoint.jld2`
- `bundle/`
- `tokenizer.json`
- `metrics.json`
- `sample_outputs.txt`
- `evaluation_prompts.txt`
- `evaluation_prompts.json`
- `run_recipe.json`

For the preflight:
- `fresh_run/` with the same layout as above
- `resume_smoke/`
  - `checkpoints/`
  - `resume_smoke_metrics.json`
- `preflight_summary.json`

## Checkpoint Cadence

Recommended cadence: every epoch.

Reason:
- the run is long enough that resume matters
- per-epoch checkpoints are already supported and keep the resume path simple

## Evaluation Prompt Policy

Save the fixed prompts actually used by the generation step:
- deterministic prompt prefixes derived from the test split
- written to `evaluation_prompts.txt` and `evaluation_prompts.json`

This keeps later sample comparison explicit instead of relying only on `sample_outputs.txt`.

## Commands

Preflight:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_final_run.jl --preflight
```

Eventual full run:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_final_run.jl
```

Optional custom full-run output directory:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_final_run.jl --output-dir tmp/prepared_better_local_real_text_final_run_custom
```

## Preflight Scope

The preflight is intentionally bounded:
- short run only
- checkpoint creation
- final bundle export
- bundle reload
- sample generation
- resume-from-checkpoint smoke test

It does not replace the eventual full training run.

## Final Demo Run Outcome

Final run output directory:
- `tmp/prepared_better_local_real_text_final_demo_run`

Final run recipe used:
- corpus: `tmp/better_local_real_text_corpus_prepared/dataset`
- backend: `:flux`
- optimizer family: `Flux.Adam`
- optimizer hyperparameters: `learning_rate = 0.001`
- tokenizer path: char-level experiment-local tokenizer
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `batch_size = 16`
- `epochs = 38`

Final run results:
- train loss `1.6413`
- validation loss `1.8590`
- test loss `1.8998`
- final step count `3040`

Final run artifacts:
- `checkpoints/` with per-epoch checkpoints and `final_checkpoint.jld2`
- `bundle/`
- `tokenizer.json`
- `metrics.json`
- `sample_outputs.txt`
- `evaluation_prompts.txt`
- `evaluation_prompts.json`
- `run_recipe.json`

Qualitative assessment:
- this is a valid trained proof-of-concept baseline for the current pipeline
- it is not a strong chatbot and still produces weak, repetitive, domain-narrow text
- it is useful as the repo's first trained demo artifact and as a reference point for later comparisons

Remaining limitations:
- tokenizer persistence remains separate from bundle semantics
- qualitative generation is still weak and malformed in places
- the run is a reproducible training/export/load demo, not a quality benchmark or conversational baseline
