# Contributing

KeemenaLM.jl is currently best treated as a Julia research proof-of-concept for
training GPT-style decoder models from scratch. The active research baseline is
the v9 broad 336M scratch chatbot, which is useful as a reproducible partial
result but is not a usable assistant.

## Read First

Before proposing another expensive run, read:

- `README.md`
- `notes/README.md`
- `notes/tiny_chatbot_v9_broad_336m_current_state.md`
- `notes/NextTODO.md`

The current diagnosis is data- and evaluation-limited. A larger or longer run is
not the immediate recommendation unless a smaller smoke or continuation run first
shows that improved data and evaluation are fixing the known failures.

## Useful Contributions

Good contributions include:

- cleaner corpus preparation scripts and filters
- compact, license-aware dataset manifests
- behavior-evaluation cases that are held out from training
- experiment notes with exact commands, hardware, runtime, and results
- small training, generation, tokenizer, or checkpoint fixes
- tests for masking, stopping, sampling, bundle IO, and behavior scoring
- report tooling that summarizes `metrics.json`, `behavior_eval.json`, and
  sample outputs

Avoid submitting:

- raw downloaded datasets
- generated `tmp/` corpora
- model weights or checkpoints directly in git
- tokenizer bundles generated under `tmp/`
- private chat logs or data with secrets
- one-off scripts that cannot be reproduced

Large model artifacts should be published through a release or external
large-file host, not normal git history.

## Current Active Tools

Use these for current chatbot work:

- `tools/prepare_tiny_chatbot_v9_broad_corpus.sh`
- `tools/prepare_tiny_chatbot_real_chat_corpus.py`
- `tools/prepare_tiny_chatbot_v8_direct_answer_corpus.py`
- `tools/run_tiny_chatbot_v8_scratch.jl`
- `tools/run_tiny_chatbot_v8_behavior_eval.jl`
- `tools/run_tiny_chatbot_v8_prompt_probe.jl`
- `tools/run_tiny_chatbot_v8_chat_repl.jl`
- `tools/package_tiny_chatbot_v9_release_artifact.sh`

The v8 names remain because the runner evolved into the v9 broad experiment.
Retired v4-v7 and UltraChat standalone runners were removed from the active tool
surface.

## Experiment Reports

For each meaningful experiment, add a note under `notes/` with:

- date recorded
- goal and hypothesis
- commit or branch, if known
- exact data source and preparation command
- exact training command
- hardware and approximate runtime
- model dimensions and parameter count
- training, validation, and test losses
- behavior-eval pass rate and failed cases
- manual prompt samples, including failures
- artifact retention state
- conclusion and recommended next step

Do not report loss alone as model quality. Include qualitative prompt outputs and
behavior-gate results.

## Data Contributions

Prefer scripts and manifests over committed data. Every data contribution should
explain:

- upstream source
- license or terms to check
- filtering rules
- language filter
- length limits
- deduplication policy
- categories kept and removed
- whether assistant-only loss masking is expected

For the next chatbot pass, prioritize broad varied coverage over exact prompt
repair. Useful categories include common-sense yes/no, safe object choices,
edible vs unsafe objects, small arithmetic in varied phrasings, city/country and
book/author distinctions, everyday object affordances, short factual QA, user
provided fact recall, and explicit `I do not know` responses for private
unknowns.

## Evaluation Contributions

Behavior tests should be held out and varied. Avoid adding only the exact prompt
a model just failed.

Useful checks include:

- role leakage
- empty completions
- repetition loops
- too-long answers
- arithmetic correctness
- common-sense ability questions
- edible vs unsafe objects
- factual category distinctions
- private unknowns
- contradiction checks
- hallucination checks
- manual transcripts for human inspection

A good evaluation contribution should make it harder for the model to pass by
keyword coincidence alone.

## Artifact Policy

The repository ignores generated corpora, weights, checkpoints, tokenizer
bundles, and most run outputs by default. Keep these out of normal commits:

- `tmp/` runs
- checkpoints and optimizer states
- `weights.jld2`
- tokenizer bundles
- cached third-party datasets
- generated prepared corpora

To package the current v9 local model for a release artifact, use:

```bash
tools/package_tiny_chatbot_v9_release_artifact.sh
```

A published model artifact should include the model card/current-state note,
data source summary, known failures, intended use, and non-goals.

## Code Checks

Before opening a pull request, run the relevant tests when possible:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

For chatbot-runner changes, also run a tiny smoke experiment before launching a
large GPU job. Confirm that:

- tokenizer training or loading works
- assistant-loss masks decode correctly
- checkpoint retention is bounded
- bundle export works
- behavior eval runs
- prompt probe runs
- the REPL can load the exported bundle
