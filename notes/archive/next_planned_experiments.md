# Next Planned Experiments

This note records the current state of the bounded experiment track after the synthetic CFG studies, the local real-text corpus preparation work, and the prepared-corpus real-text sweeps.

## Completed Experiment Trail

Synthetic CFG phase:
- first bounded CFG experiment
- fixed-sentence-count CFG complexity sweep
- fixed-complexity budget sweep at `complexity = 5`
- fixed-complexity width sweep at `complexity = 5`
- fixed-complexity depth sweep at `complexity = 5`
- token-controlled CFG complexity sweep

Local real-text phase:
- first small local markdown/text sanity check
- second larger local markdown/text sanity check
- better local real-text corpus selection and preparation
- prepared better local real-text char-level baseline run
- prepared-corpus subword experiment and step-matched subword follow-up
- prepared-corpus width sweep
- prepared-corpus budget sweep
- prepared-corpus context-length sweep
- prepared-corpus second width sweep
- prepared-corpus budget extension sweeps through `22` epochs under `Flux.Descent`
- prepared-corpus learning-rate sweep under `Flux.Descent`
- prepared-corpus optimizer-family sweep
- prepared-corpus Adam learning-rate sweep
- prepared-corpus Adam budget sweeps through `38` epochs
- final-run preflight and final trained demo baseline run

## Current Best Real-Text Recipe

Current best tested recipe:
- corpus: `tmp/better_local_real_text_corpus_prepared/dataset`
- backend: `:flux`
- optimizer family: `Flux.Adam`
- optimizer hyperparameters: `learning_rate = 0.001`
- tokenizer path: char-level experiment-local tokenizer
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `batch_size = 16`
- deterministic seed style unchanged across prepared-corpus comparisons

Current best bounded result:
- `epochs = 38`
- train loss `1.6413`
- validation loss `1.8590`
- test loss `1.8998`

Current trained demo baseline:
- output directory: `tmp/prepared_better_local_real_text_final_demo_run`
- per-epoch checkpoints, final checkpoint, bundle, tokenizer artifact, metrics, saved evaluation prompts, and sample outputs were all produced successfully
- qualitative generation is still weak and repetitive, so this is a trained proof-of-concept baseline rather than a strong chatbot demo

## Current Interpretation

- the package pipeline is stable end to end for training, checkpoints, bundle export/load, and generation
- the synthetic CFG phase established that the tiny model learns controlled structure and responds to capacity and budget changes
- the prepared better local real-text corpus is the best current real-text setup
- the tested subword path did not beat the char-level baseline under the tested conditions
- width helped up to `128 / 256`, then flattened or turned over
- longer context did not help in the tested setup
- optimizer choice mattered a lot: `Flux.Adam(0.001)` clearly beat `Flux.Descent`
- training budget under Adam remained the strongest positive signal through `38` epochs
- the current baseline is operationally solid but still not close to chatbot-quality generation

## Best Next Step

### One more bounded Adam budget extension or stop and treat the current run as the first official baseline

Recommended if you still want one more clean training experiment:
- keep the current Adam recipe fixed
- extend epochs one more bounded step beyond `38`
- stop if validation/test gains flatten or begin to reverse

Recommended if you want to pause exploration and stabilize the project state:
- treat `tmp/prepared_better_local_real_text_final_demo_run` as the first trained demo baseline
- document its recipe, artifacts, and limitations clearly
- use that baseline for later comparison when data/tokenizer improvements resume

Decision target:
- if another bounded Adam extension still improves cleanly, the active bottleneck is still raw training budget
- if gains flatten, the next clean branch is training-recipe or data/tokenizer work rather than more architecture sweeps

## Not The Immediate Next Step

Do not prioritize these immediately:
- new architecture stages
- width or context sweeps on the current char-level baseline
- tokenizer persistence redesign
- claiming chatbot-quality results from the current trained baseline
- remote data/model work
- Lux training parity
- broad experiment-framework work

Those may matter later, but they are not the cleanest next lever from the current results.
