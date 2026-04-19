# Keemena Docs Assistant Dataset

This dataset is the first tiny chatbot-ready corpus for a narrow-domain `Keemena Docs Assistant`.

## Scope

- KeemenaLM.jl
- KeemenaPreprocessing.jl
- KeemenaSubwords.jl

The target assistant is intentionally narrow. It should answer factual, procedural, limitation, and troubleshooting questions about training, checkpoints, bundles, official model flow, preprocessing, tokenizer loading, tokenizer formats, offsets, alignment, and related workflows across the three local Keemena projects.

## Source selection

The preparation script uses only local controlled material:

- user-facing docs from this repository
- local docs from `../KeemenaPreprocessing.jl`
- local docs from `../KeemenaSubwords.jl`
- a very small set of short KeemenaLM docstrings for `load_bundle`, `save_bundle`, model-source resolution, and chat session behavior

It intentionally avoids experiment sweep notes and unrelated generated output so the corpus stays closer to a real docs assistant than to a benchmark diary.

## Dataset format

Primary split files are plain chat text:

```text
User: ...
Assistant: ...
```

Each split is also written as JSONL with source provenance and category labels.

## Split policy

- deterministic QA-pair generation from the curated source set
- deterministic SHA1 ordering by pair id
- fixed `80/10/10` split into training, validation, and testing

## Output

Run:

```bash
julia --project=. tools/prepare_keemena_docs_assistant_dataset.jl
```

Default output directory for the cleaned chatbot-ready pass:

```text
tmp/keemena_docs_assistant_dataset_v4
```

Files written:

- `training.txt`
- `validation.txt`
- `testing.txt`
- `training.jsonl`
- `validation.jsonl`
- `testing.jsonl`
- `metadata.json`

## Intended use

This is suitable for a first narrow docs-assistant proof of concept. It is not an open-domain chat corpus and it should not be treated as a polished customer-support dataset. The dataset is only as good as the local docs it was derived from, so expect narrow coverage and occasional phrasing that sounds like documentation rather than natural dialogue.

## Cleanup policy for v4

The earlier chatbot datasets were structurally correct but still too documentation-shaped for answer generation. The v4 pass keeps the pipeline deterministic and scriptable while improving chat supervision directly:

- keep one best deterministic answer per question so training sees a more consistent assistant target
- append an explicit assistant end marker to every sample:

```text
<END_ASSISTANT>
```

- limit auto-derived pairs to the first substantive paragraph per section
- shorten auto-derived answers toward one or two direct sentences instead of full docs fragments
- filter more low-value answer shapes such as function-signature fragments, docs-index fragments, contract dumps, and repeated code-oriented explanations
- keep filtering table-heavy answers, walkthrough scaffolding, and clearly truncated stubs
- add a deterministic curated supplement with clearer assistant-style answers for:
  - KeemenaLM bundle, checkpoint, chat, official-model, and limitation questions
  - KeemenaPreprocessing streaming, offsets, alignments, and bundle save/load questions
  - KeemenaSubwords loading, format, offset, troubleshooting, and integration questions

The v4 dataset may be smaller than earlier passes. That is intentional if the result is more direct, more assistant-like, and a better teaching signal for answer generation.
