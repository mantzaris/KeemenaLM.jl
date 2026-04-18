# Next planned experiments

This note records the next intended experiment sequence after the completed synthetic CFG phase and the first two local real-text sanity checks.

Current completed experiment work:
- first bounded CFG experiment
- fixed-sentence-count CFG complexity sweep
- fixed-complexity budget sweep at `complexity = 5`
- fixed-complexity width sweep at `complexity = 5`
- fixed-complexity depth sweep at `complexity = 5`
- token-controlled CFG complexity sweep
- first small local real-text sanity check
- second larger local real-text sanity check

Current interpretation:
- the pipeline works end to end
- the tiny Flux GPT-2 clearly learns low-complexity CFG structure
- learning degrades as CFG complexity rises, even after controlling training token volume
- at `complexity = 5`, more epochs help a little, width helps more than depth, and the setup still looks capacity-limited
- the real-text path also works end to end, but the two local markdown corpora are too small and repetitive to produce useful text quality
- the next bottleneck is better real-text data, not more architecture churn

---

## Immediate next experiment

### Better local real-text corpus

Purpose:
- keep the current training, checkpoint, bundle, and generation path stable
- use a somewhat larger and cleaner local real-text corpus in the same general technical-writing style
- test whether more natural-text volume and variety improve behavior more than the tiny markdown-only sanity-check corpora did

Recommended control:
- keep the current tiny model and training recipe fixed first
- add more locally controlled prose before changing architecture again
- prefer one coherent corpus extension over a mixed noisy grab bag
- keep deterministic split creation and the same simple tokenizer path for the next run

Primary outputs:
- train, validation, and test loss
- sample outputs after bundle reload
- token and example-count stats
- a direct comparison against the current small and larger local-text baselines

Decision target:
- if a better local corpus materially improves results, the next bottleneck is still mostly data
- if a better local corpus barely helps, the next bottleneck is more likely tokenizer quality, training recipe limits, or model capacity on natural text

---

## Second follow-up experiment

### Small external real-text dataset or better curated local corpus

Only do this after the better local real-text corpus run above.

Purpose:
- move beyond repo docs and notes while still keeping the experiment small, explicit, and legally clear

Recommendation:
- use a narrow real-text corpus with clearer sentence-level variety
- keep the same tiny model and training path first
- use it as a transfer sanity check, not as a chatbot benchmark

---

## Later experiment track

### Architecture or tokenizer follow-up on real text

Only do this after the better real-text data step.

Possible directions:
- tokenizer and preprocessing integration polish
- a slightly larger real-text model at fixed corpus and budget
- a cleaner real-text corpus curation pass
- a small model-capacity follow-up on real text if the data improvement alone is not enough

---

## Not the next experiment

Do not prioritize these before the next real-text data step:
- new architecture stages
- another synthetic sweep unless a very specific question appears
- chatbot-style evaluation
- tokenizer persistence redesign
- large benchmark framework creation
- remote model distribution work
- Lux training parity

Those are important later, but they are not the next best way to learn from the current results.
