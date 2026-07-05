# Better local real-text sanity-check experiment

This is the next bounded local real-text transfer sanity check for `KeemenaLM.jl`.

Purpose:
- test the stable training/checkpoint/bundle/generation pipeline on a better local corpus
- improve on the earlier markdown-only runs by adding selected codebase docstrings
- keep the corpus entirely local, coherent, and inspectable

Selected corpus source:
- markdown files:
  - `README.md`
  - `docs/src/index.md`
  - `notes/repo_plan_short.md`
  - `notes/repo_plan_structure.md`
  - `notes/todo_staged_roadmap.md`
  - `notes/repo_plan_long.md`
  - `notes/next_planned_experiments.md`
  - `notes/official_models.md`
- selected source docstrings:
  - `src/core/types.jl`
  - `src/core/configs/gpt2.jl`
  - `src/core/model/masking.jl`
  - `src/core/generation/sampling.jl`
  - `src/core/generation/stopping.jl`
  - `src/core/generation/generate.jl`
  - `src/core/generation/chat.jl`
  - `src/core/io/bundle_schema.jl`
  - `src/core/io/model_sources.jl`
  - `src/core/io/weights_jld2.jl`
  - `src/core/io/bundle_save.jl`
  - `src/core/io/bundle_load.jl`
  - `src/core/training/loss.jl`
  - `src/core/training/trainer.jl`
  - `src/backends/flux/gpt2_flux.jl`

Why this corpus is better:
- still entirely local
- still narrow and technical
- adds interface/documentation prose with more varied sentence shapes than the prior markdown-only corpus
- avoids external data and noisy generated outputs

Corpus preparation:
- same deterministic markdown cleaning and paragraph splitting as the earlier local-text runs
- docstrings are extracted from Julia source files using triple-quoted blocks and lightly normalized
- deterministic 80/10/10 split by paragraph order across the combined entry list

Model and training settings:
- backend `:flux`
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 64`
- `ffn_hidden_size = 128`
- `epochs = 2`
- `batch_size = 16`
- `learning_rate = 0.01f0`
- deterministic seed style matching the earlier experiments

Run it:

```bash
julia --project=tools/benchmark_cfg tools/run_local_text_corpus_experiment_better.jl
```

Outputs:
- prepared split files in `dataset/`
- `corpus_metadata.json`
- checkpoints
- bundle
- `tokenizer.json`
- `metrics.json`
- `sample_outputs.txt`
