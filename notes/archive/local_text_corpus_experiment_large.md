# Larger local real-text sanity-check experiment

This is the second bounded local real-text transfer sanity check for `KeemenaLM.jl`.

Purpose:
- extend the first local-text experiment with a larger but still coherent local technical-writing corpus
- keep the same training/checkpoint/bundle/generation pipeline
- check whether more local corpus volume materially improves behavior

Selected corpus source:
- `README.md`
- `docs/src/index.md`
- `notes/repo_plan_short.md`
- `notes/repo_plan_structure.md`
- `notes/todo_staged_roadmap.md`
- `notes/repo_plan_long.md`
- `notes/next_planned_experiments.md`
- `notes/official_models.md`

Why this larger corpus:
- stays in the same technical/documentation style as the first run
- materially increases local corpus volume without pulling in noisy scaffold/change-log text
- remains fully local and inspectable

Corpus preparation:
- same deterministic markdown cleaning and paragraph splitting as the first local-text experiment
- deterministic 80/10/10 split by paragraph order

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
julia --project=tools/benchmark_cfg tools/run_local_text_corpus_experiment_large.jl
```

Outputs:
- prepared split files in `dataset/`
- `corpus_metadata.json`
- checkpoints
- bundle
- `tokenizer.json`
- `metrics.json`
- `sample_outputs.txt`

Observed outcome:
- the larger corpus improved loss modestly relative to the first local-text run
- the operational path remained stable: deterministic prep, training, checkpointing, bundle export, bundle reload, and generation all worked
- generation quality was still weak and repetitive
- the main bottleneck after this run is still corpus size and variety rather than package mechanics
