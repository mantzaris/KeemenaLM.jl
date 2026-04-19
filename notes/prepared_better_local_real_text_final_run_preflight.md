# Prepared Better Local Real-Text Final Run Preflight

Purpose: verify that the current best real-text recipe is ready for one explicit best-effort training run, without starting that long run yet.

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
