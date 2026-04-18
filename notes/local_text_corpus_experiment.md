# Local real-text sanity-check experiment

This is the first bounded real-text transfer sanity check for `KeemenaLM.jl`.

Purpose:
- test the existing training/checkpoint/bundle/generation pipeline on a small local real-text corpus
- keep the corpus fully local and inspectable
- not measure chatbot quality
- not build a large data pipeline

Selected corpus source:
- `README.md`
- `docs/src/index.md`
- `notes/repo_plan_short.md`
- `notes/repo_plan_structure.md`
- `notes/todo_staged_roadmap.md`

Why this corpus:
- narrow, consistent technical style
- locally controlled
- enough prose to exercise the full pipeline without becoming a large dataset task

Corpus preparation:
- read the selected markdown files in the listed order
- remove fenced code blocks
- apply light markdown normalization:
  - strip heading and bullet markers
  - strip inline backticks
  - replace markdown links with their visible text
- split into paragraphs
- drop very short fragments
- create a deterministic 80/10/10 split by paragraph order

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
- deterministic seed style matching the synthetic experiments

Run it:

```bash
julia --project=tools/benchmark_cfg tools/run_local_text_corpus_experiment.jl
```

Outputs:
- prepared split files in `dataset/`
- `corpus_metadata.json`
- checkpoints
- bundle
- `tokenizer.json`
- `metrics.json`
- `sample_outputs.txt`
