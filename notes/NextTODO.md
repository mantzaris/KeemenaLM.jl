# Next TODO

Date updated: 2026-07-04

This is the current restart point for the repository.

## Current State

The latest completed scratch-chatbot experiment is the broad v9 336M run:

- run directory: `tmp/tiny_chatbot_v9_broad_336m_run`
- final bundle: `tmp/tiny_chatbot_v9_broad_336m_run/bundle`
- tokenizer bundle: `tmp/tiny_chatbot_v9_broad_336m_run/tokenizer_bundle`
- parameter count: `336,488,448`
- pretraining text: about `1.5B` characters
- SFT examples: `39,584`
- test losses: `pretrain=2.0753`, `sft=3.1421`
- behavior gate: `17/22`

The model is not a usable assistant. It is a publishable proof-of-concept and a
baseline for collaboration.

Read first:

- `notes/tiny_chatbot_v9_broad_336m_current_state.md`
- `README.md`
- `CONTRIBUTING.md`
- `notes/README.md`

## Active Tool Surface

Use these current files:

- `tools/prepare_tiny_chatbot_v9_broad_corpus.sh`
- `tools/prepare_tiny_chatbot_real_chat_corpus.py`
- `tools/prepare_tiny_chatbot_v8_direct_answer_corpus.py`
- `tools/run_tiny_chatbot_v8_scratch.jl`
- `tools/run_tiny_chatbot_v8_behavior_eval.jl`
- `tools/run_tiny_chatbot_v8_prompt_probe.jl`
- `tools/run_tiny_chatbot_v8_chat_repl.jl`
- `tools/package_tiny_chatbot_v9_release_artifact.sh`
- `tools/tiny_chatbot_training_common.jl`

The v8 names are historical. They are still the current v8/v9-style runner and
evaluation tools. Retired v4-v7 and UltraChat standalone runner files have been
removed.

## Do Not Do Next

Do not launch a blind 0.5B or longer v9-style run with the same data. The current
result suggests the project is more data- and evaluation-limited than
parameter-limited.

Do not restart the old repair-loop path. It produced exact-prompt improvements
but poor nearby generalization.

Do not commit generated corpora, tokenizer bundles, checkpoints, or weights to
git.

## Recommended Next Experiment

1. Expand the behavior suite beyond the current 22 prompts.
2. Build a v10 data pass that directly covers the v9 failures:
   - arithmetic in varied phrasings
   - common-sense ability questions
   - edible vs unsafe object choices
   - safe ordinary preferences
   - object affordances
   - city/country/book/movie/person distinctions
   - short factual QA
   - unknown private facts with explicit `I do not know` answers
   - repetition and contradiction traps
3. Audit decoded training examples and assistant-loss masks.
4. Run a tiny smoke training job.
5. Run behavior eval and prompt probe before any multi-day job.
6. Try a short continuation from the final v9 bundle only if the data/eval smoke
   pass improves the known failures.
7. Consider a larger model only after smaller verification runs demonstrate that
   the new data and evaluation are moving quality.

## Current Commands

Behavior gate:

```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_behavior_eval.jl \
  --run-dir tmp/tiny_chatbot_v9_broad_336m_run
```

Prompt probe:

```bash
CUDA_VISIBLE_DEVICES=0 julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_prompt_probe.jl \
  --run-dir tmp/tiny_chatbot_v9_broad_336m_run \
  --bundle-dir tmp/tiny_chatbot_v9_broad_336m_run/bundle \
  --tokenizer-bundle-dir tmp/tiny_chatbot_v9_broad_336m_run/tokenizer_bundle \
  --device gpu \
  --temperature 0.0 \
  --top-k 0 \
  --top-p 1.0 \
  --max-new-tokens 120
```

REPL:

```bash
CUDA_VISIBLE_DEVICES=0 julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_chat_repl.jl \
  --run-dir tmp/tiny_chatbot_v9_broad_336m_run \
  --bundle-dir tmp/tiny_chatbot_v9_broad_336m_run/bundle \
  --tokenizer-bundle-dir tmp/tiny_chatbot_v9_broad_336m_run/tokenizer_bundle \
  --device gpu \
  --temperature 0.0 \
  --top-k 0 \
  --top-p 1.0 \
  --max-new-tokens 120
```

Package release artifact:

```bash
tools/package_tiny_chatbot_v9_release_artifact.sh
```

Rebuild v9-style corpus:

```bash
tools/prepare_tiny_chatbot_v9_broad_corpus.sh
```

## Verification

Before committing code changes, run:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

For docs changes, run:

```bash
julia --project=docs docs/make.jl
```
