# Next planned experiments

This note records the next intended experiment sequence after the completed BenchmarkDataNLP CFG runs.

Current completed benchmark work:
- first bounded CFG experiment
- fixed-sentence-count CFG complexity sweep
- fixed-complexity budget sweep at `complexity = 5`

Current interpretation:
- the pipeline works end to end
- the tiny Flux GPT-2 clearly learns low-complexity CFG structure
- learning degrades as CFG complexity rises
- at `complexity = 5`, more epochs help a little, but not enough to remove degradation
- the current setup now looks more capacity-limited than pipeline-broken

---

## Immediate next experiment

### Model-size sweep at fixed complexity

Purpose:
- keep the degraded point from the earlier sweeps fixed at `complexity = 5`
- keep dataset and training budget fixed
- vary only one model-capacity axis
- test whether the remaining degradation is mainly capacity-bound

Recommended control:
- fix `complexity = 5`
- fix `num_sentences = 4_000`
- fix `epochs = 2`
- fix `batch_size = 16`
- fix `learning_rate = 0.01f0`
- fix tokenizer approach
- fix backend `:flux`

Recommended sweep axis:
- vary `embedding_size` only

Recommended first values:
- `embedding_size = 32`
- `embedding_size = 64`
- `embedding_size = 96`

Keep these fixed to avoid mixing variables:
- `num_layers = 2`
- `num_heads = 2`
- `ffn_hidden_size` scaled in the simplest consistent way for the chosen embedding size, or held fixed if you want a stricter width-only control

Primary outputs:
- train, validation, and test loss
- perplexity-style signal
- final step count
- sample output paths
- token or example-count stats for interpretability

Decision target:
- if a modest size increase materially improves loss and samples, the next bottleneck is probably capacity
- if a size increase barely helps, the next bottleneck is more likely data formulation, tokenizer limits, or training recipe limits

---

## Second follow-up experiment

### Architectural capacity sweep

Only do this after the width sweep above.

Purpose:
- if width helps, test whether depth helps further

Recommendation:
- keep `complexity = 5`
- keep dataset and budget fixed
- hold width fixed at the best result from the first size sweep
- vary `num_layers` only

Possible values:
- `num_layers = 2`
- `num_layers = 3`
- `num_layers = 4`

Do not mix width and depth in the same first follow-up sweep.

---

## Later experiment track

### Cleaner complexity comparison

The first complexity sweep was useful, but fixed sentence count allowed effective training volume to drift with complexity.

Later, run a cleaner complexity sweep with one of these controls:
- fixed total training token stream length
- fixed total number of LM examples

That will answer the complexity question more cleanly than the first exploratory sweep.

---

## Not the next experiment

Do not prioritize these before the model-size sweep:
- chatbot-style evaluation
- tokenizer persistence redesign
- large benchmark framework creation
- remote model distribution work
- Lux training parity

Those are important later, but they are not the next best way to learn from the current benchmark results.
