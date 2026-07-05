# Better Local Real-Text Corpus Preparation

Purpose: prepare a better narrow-domain real-text corpus for the next KeemenaLM experiment without changing the current training, checkpoint, bundle, or generation pipeline.

## Corpus Choice

The prepared corpus is a curated local prose subset from:

- `../KeemenaPreprocessing.jl`
- `../KeemenaSubwords.jl`

Included content is limited to README, overview, paper, and guide-style documentation that reads as coherent technical prose. It deliberately excludes:

- private or operational notes in unrelated repos
- provenance-unclear local text dumps
- API reference or generated inventory pages that are more list-like than prose-like

This is a better next-step corpus than the current KeemenaLM repo-docs corpus because it is:

- still fully local and inspectable
- materially larger
- narrowly focused on NLP/preprocessing/tokenization
- more sentence-like and less roadmap/planning-oriented

## Preparation Script

Run:

```bash
julia --project=. tools/prepare_better_local_real_text_corpus.jl
```

Optional custom output directory:

```bash
julia --project=. tools/prepare_better_local_real_text_corpus.jl tmp/my_better_corpus
```

## Cleaning And Splitting

The script:

1. removes fenced code blocks
2. strips markdown markers and inline backticks
3. replaces markdown links with visible link text
4. collapses whitespace
5. splits into paragraphs
6. drops paragraphs shorter than five words
7. computes a deterministic SHA1 ordering key from `source_file`, `paragraph_index`, and cleaned paragraph text
8. writes deterministic `80/10/10` train/validation/test splits from that stable ordering

## Outputs

The prepared dataset is written under:

- `dataset/training.txt`
- `dataset/validation.txt`
- `dataset/testing.txt`
- `dataset/corpus_metadata.json`

The metadata file records:

- exact source files
- excluded nearby sources and reasons
- cleaning steps
- split policy and split method details
- per-file size stats
- per-split size stats

## Experiment Status

The prepared-real-text training scripts and generated outputs were removed after
the project direction moved to the UltraChat subword chatbot path. This note is
kept only as corpus-preparation history.

## Why This Corpus

This corpus is intended as the next real-text training target because it stays within a coherent Julia NLP tooling domain while offering more natural prose variety than repo planning/docs text. It remains small and controlled enough to inspect directly before the next training run.
