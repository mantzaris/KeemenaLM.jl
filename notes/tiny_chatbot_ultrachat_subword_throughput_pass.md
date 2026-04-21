# Tiny Chatbot UltraChat Subword Throughput Pass

This note records the bounded engineering pass on the UltraChat subword runner.

## What changed

- added explicit CLI controls for:
  - tokenizer training text limit
  - train / validation / test text limits
  - validation / test batch limits
  - tokenizer bundle reuse directory
- added tokenizer bundle reuse so repeated UltraChat runs do not retrain the tokenizer by default when a compatible bundle already exists
- removed an unnecessary intermediate `inputs` / `targets` materialization step in batch construction
- kept the same model family, same corpus, same tokenizer family, and same artifact flow

## Why

The previous UltraChat path was spending too much time in repeated setup work and still had all scale controls effectively buried in hardcoded defaults.

The goal of this pass was not to redesign the trainer, but to make larger serious runs more practical and more explicit.

## What the proof attempts showed

The improved runner does clear setup much faster when it can reuse the tokenizer bundle.

However, materially larger proof runs such as:

- `train_text_limit = 4000`, `epochs = 2`
- `train_text_limit = 4000`, `epochs = 1`, `batch_size = 24`
- `train_text_limit = 3500`, `epochs = 1`, `batch_size = 32`

still remained too slow to finish cleanly in this environment within a bounded pass.

## Readout

The engineering changes help with:

- repeated tokenizer work
- explicit run sizing
- setup overhead

But they do not solve the main remaining cost:

- the actual training-loop throughput on larger UltraChat slices

So this pass is useful, but only partial. The next bottleneck is no longer hidden setup waste. It is the per-epoch optimization cost on a serious corpus slice.
