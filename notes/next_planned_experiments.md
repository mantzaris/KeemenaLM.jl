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
- prepared-corpus budget extension sweeps through `22` epochs

## Current Best Real-Text Recipe

Current best tested recipe:
- corpus: `tmp/better_local_real_text_corpus_prepared/dataset`
- backend: `:flux`
- tokenizer path: char-level experiment-local tokenizer
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `batch_size = 16`
- `learning_rate = 0.01f0`
- deterministic seed style unchanged across prepared-corpus comparisons

Current best bounded result:
- `epochs = 22`
- validation loss `2.8881`
- test loss `2.8768`

## Current Interpretation

- the package pipeline is stable end to end for training, checkpoints, bundle export/load, and generation
- the synthetic CFG phase established that the tiny model learns controlled structure and responds to capacity and budget changes
- the prepared better local real-text corpus is the best current real-text setup
- the first subword path did not beat the char-level baseline under the tested conditions
- width helped up to `128 / 256`, then flattened or turned over
- longer context did not help in the tested setup
- training budget remains the strongest positive signal, though gains are now showing diminishing returns

## Best Next Experiment

### Small optimizer or training-recipe change at the current best recipe

Purpose:
- test whether the current recipe is now limited more by optimization than by raw epoch count
- keep architecture, tokenizer path, corpus, checkpoint flow, and bundle flow stable

Recommended controls:
- start from the current best `128 / 256`, `context_length = 48` recipe
- change only one training-recipe variable at a time
- examples:
  - learning-rate sweep
  - optimizer choice or optimizer hyperparameter change
  - simple learning-rate decay if kept narrowly scoped

Decision target:
- if a small training-recipe change beats the current `22`-epoch baseline cleanly, the next bottleneck is optimization rather than architecture
- if it does not, then one more bounded budget extension is still defensible before bigger changes

## Alternate Conservative Next Step

### One more bounded budget extension

If you want a cleaner saturation curve before changing optimizer behavior:
- keep the current best recipe fixed
- extend epochs one more step beyond `22`
- stop if validation/test gains become negligible or begin to reverse

## Not The Next Step

Do not prioritize these next:
- new architecture stages
- tokenizer persistence redesign
- chatbot-style evaluation
- remote data/model work
- Lux training parity
- broad experiment-framework work

Those may matter later, but they are not the cleanest next lever from the current results.
