# Future Chatbot Improvement Handoff

Date updated: 2026-07-04

## Read This First

The Julia LLM pipeline works end to end, but the project does not yet have a
decent chatbot. The current strongest model is the v9 broad 336M scratch run,
documented in `notes/tiny_chatbot_v9_broad_336m_current_state.md`.

Current result:

- `336,488,448` estimated parameters
- about `1.5B` pretraining characters
- `39,584` SFT examples
- test losses: `pretrain=2.0753`, `sft=3.1421`
- behavior gate: `17/22`
- retained local run: `tmp/tiny_chatbot_v9_broad_336m_run`

The result is better than older repair-loop experiments but still fails basic
chatbot expectations. Do not spend the next session simply increasing model size
or extending the same training recipe.

## Current Diagnosis

The bottleneck appears to be data and evaluation quality more than raw model
capacity.

Known failures:

- brittle arithmetic
- weak common sense
- unsafe or nonsensical object-choice answers
- poor factual grounding
- hallucinated private unknowns
- repetition loops
- keyword-based behavior-gate success that does not generalize enough

Likely causes:

- `39,584` SFT examples is small for a 336M scratch assistant
- broad chat data is noisy and often teaches vague responses
- the 5,000 synthetic anchors are useful but too narrow
- the behavior gate is too small and can be gamed by keyword overlap
- validation loss is not a good proxy for interactive assistant quality

## Active Code State

Current chatbot files:

- `tools/prepare_tiny_chatbot_v9_broad_corpus.sh`
- `tools/prepare_tiny_chatbot_real_chat_corpus.py`
- `tools/prepare_tiny_chatbot_v8_direct_answer_corpus.py`
- `tools/run_tiny_chatbot_v8_scratch.jl`
- `tools/run_tiny_chatbot_v8_behavior_eval.jl`
- `tools/run_tiny_chatbot_v8_prompt_probe.jl`
- `tools/run_tiny_chatbot_v8_chat_repl.jl`
- `tools/package_tiny_chatbot_v9_release_artifact.sh`

The v8 filenames remain because that runner became the current v8/v9 training
path. Retired UltraChat and v4-v7 standalone runners were removed.

## Recommended Next Direction

Build a v10 data and evaluation pass before any long run.

Data priorities:

- many varied short-answer examples
- common-sense yes/no tasks
- edible vs unsafe object distinctions
- safe ordinary choice prompts
- small arithmetic across many phrasings
- everyday object affordances
- city/country/person/book/movie distinctions
- short factual QA
- user-provided fact recall
- explicit `I do not know` answers for private unknowns
- examples that avoid template closers and repeated slogans

Evaluation priorities:

- expand beyond the current 22 prompts
- include held-out variants, not exact training prompts
- score repetition, role leakage, contradiction, unsafe choices, and hallucinated
  private facts
- keep qualitative sample transcripts with every run
- avoid treating loss as quality by itself

Training priorities:

- smoke-test data and masks before a full run
- try short continuations from the final v9 bundle only after the data/eval pass
- compare replay ratios and lower learning rates on short runs
- keep checkpoint retention bounded
- package only final bundles and metadata for release

## Success Criteria For The Next Attempt

A useful next model does not need to be benchmark competitive. It should at
least:

- answer greetings normally
- avoid continuing as `User:`
- stop after one assistant response
- avoid obvious repetition loops
- answer small arithmetic reliably
- say that humans and cats cannot fly unaided
- prefer edible safe objects over unsafe ones
- answer common city/country/capital/category questions
- say it does not know private facts not given by the user

If a smoke run fails these checks, inspect the decoded training examples and
assistant-loss masks before changing model size.
