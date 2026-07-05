## KeemenaLM.jl staged roadmap and recommendation

This note is now the post-v0.1 roadmap document.

Stages 1 through 7 are complete and are no longer listed here as pending execution stages.
The earlier stage sequence delivered the current proof-of-concept baseline:
- Flux CPU inference
- Flux NVIDIA/CUDA training
- portable bundles
- resumable checkpoints
- REPL chat
- local artifact-backed official demo model flow
- Lux CPU inference parity

Use the other planning notes as architectural reference:
- `repo_plan_short.md`
- `repo_plan_long.md`
- `repo_plan_structure.md`

This file should now only track future work after the completed v0.1 roadmap.

---

## Current baseline

The repository has reached the original usable proof-of-concept target.

Current supported state:
- GPT-2 style decoder-only architecture
- Flux training and inference
- Lux inference
- bundle save/load
- checkpoint save/load
- REPL chat
- local artifact registration for the official tiny demo model

Current explicit limits:
- Lux training parity is still deferred
- tokenizer and preprocessing payloads are still supplied explicitly by the user
- official model distribution is still local artifact registration only
- no non-NVIDIA GPU support

Current experiment status:
- the synthetic CFG benchmark phase is complete enough for this proof-of-concept stage
- local real-text sanity checks and prepared-corpus sweeps completed successfully
- optimizer and budget sweeps established `Flux.Adam(0.001)` as the best current training path for the tiny real-text setup
- a first trained demo baseline was produced successfully on the prepared better local real-text corpus
- the pipeline is now proven end to end for training, checkpointing, bundle export/load, and saved-prompt generation
- current qualitative generation is still weak and domain-narrow, so the trained baseline is a proof-of-concept artifact rather than a good chatbot

---

## Next-stage guidance

Future stages should be added only when they are concrete enough to execute.

Rules for new stages:
- keep each stage outcome testable
- keep stage boundaries explicit
- do not reopen completed architecture decisions without a concrete reason
- prefer user-facing simplification before speculative expansion

Recommended priority order for future work:
1. either one more bounded Adam budget extension or stabilization/documentation around the first trained demo baseline
2. larger and cleaner real-text data work using the current stable pipeline
3. tokenizer and preprocessing integration polish
4. real remote official model distribution
5. Lux training parity

Immediate experiment continuation is documented separately in:
- `next_planned_experiments.md`
- `local_text_corpus_experiment.md`
- `local_text_corpus_experiment_large.md`

---

## Future stages

## Stage 8: TBD

### Goal
TBD

### Recommendation
TBD

### Files to create or complete
- TBD

### Functions to add or complete
- TBD

### Call and dependency flow
- TBD

### Dependency rule for this stage
- TBD

### Expected outcome
TBD

### Tests to require before calling this stage done
- TBD

---

## Stage 9: TBD

### Goal
TBD

### Recommendation
TBD

### Files to create or complete
- TBD

### Functions to add or complete
- TBD

### Call and dependency flow
- TBD

### Dependency rule for this stage
- TBD

### Expected outcome
TBD

### Tests to require before calling this stage done
- TBD

---

## Stage 10: TBD

### Goal
TBD

### Recommendation
TBD

### Files to create or complete
- TBD

### Functions to add or complete
- TBD

### Call and dependency flow
- TBD

### Dependency rule for this stage
- TBD

### Expected outcome
TBD

### Tests to require before calling this stage done
- TBD

---

## Recommended milestones

### v0.1 complete
- original Stages 1 through 7 complete

### Post-v0.1 milestone A: usability polish
- tokenizer and preprocessing integration is less manual

### Post-v0.1 milestone B: distribution maturity
- official model flow works for a fresh user without local artifact registration

### Post-v0.1 milestone C: backend parity
- Lux training and checkpoint parity reach the supported-path standard

---

## What counts as usable now

The package currently meets the original proof-of-concept usability target:
- a tiny model can be trained with Flux on NVIDIA GPU
- the model can be checkpointed and resumed
- an inference bundle can be saved and loaded
- generation works on CPU
- a user can chat with a saved bundle from the REPL
- one official demo model can be resolved and loaded through the package API

Future stages should improve ergonomics, distribution, and backend depth rather than recreate the already-completed baseline.
The immediate next learning task is either to confirm Adam-budget flattening or to stabilize and compare against the new trained baseline, not to reopen core architecture.
