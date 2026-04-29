# Next TODO

This file is the restart point for the next session. The old toy/docs/OASST
experiment files and generated outputs were removed. The current path is the
UltraChat subword chatbot path only.

## Current State

Keep working from:

- Corpus prep: `tools/prepare_tiny_chatbot_ultrachat_corpus.py`
- Training runner: `tools/run_tiny_chatbot_ultrachat_subword_v1.jl`
- Corpus note: `notes/tiny_chatbot_ultrachat_corpus_v1.md`
- Prepared corpus: `tmp/tiny_chatbot_ultrachat_corpus_v1`
- Last successful run: `tmp/tiny_chatbot_ultrachat_subword_candidate_run_v1`

The last successful run completed cleanly and now keeps only final artifacts:

- `bundle/`
- `tokenizer_bundle/`
- `checkpoints/final_checkpoint.jld2`
- `metrics.json`
- `progress.json`
- `run_recipe.json`
- `sample_outputs.txt`
- `evaluation_prompts.txt`
- `evaluation_prompts.json`

Intermediate step and epoch checkpoints were deleted because they consumed about
`38G` and are not needed once the final checkpoint and bundle exist.

## Last Successful Run

Dataset:

- `tmp/tiny_chatbot_ultrachat_corpus_v1`
- Source: `HuggingFaceH4/ultrachat_200k`
- License recorded in the corpus note: MIT
- Prepared examples: about `140895`
- Training split used in the run: `50000` conversations
- Validation/test caps used in the run: `1000` each

Tokenizer:

- Package: `KeemenaSubwords.jl`
- Trainer: `:hf_gpt2_bytebpe`
- Vocab size: `8192`
- Min frequency: `2`
- Chat markers kept in text and trained as special tokens:
  - `User:`
  - `Assistant:`
  - `<END_ASSISTANT>`
  - `<CHAT_END>`

Model:

- `context_length = 128`
- `num_layers = 8`
- `num_heads = 8`
- `embedding_size = 512`
- `ffn_hidden_size = 2048`

Training:

- Backend: `:flux`
- Optimizer: `Flux.Adam(0.0003)`
- Batch size: `16`
- Epochs: `2`
- Final step: `55560`

Final metrics:

- Train loss: about `3.1615`
- Validation loss: about `2.9413`
- Test loss: about `3.0498`

This was a large numeric improvement over the previous `6x384` UltraChat runs
whose validation/test losses were around `3.84`/`3.92`.

## What Was Good

- The run completed successfully.
- Checkpointing, final bundle export, tokenizer bundle persistence, metrics,
  progress reporting, and sample generation all worked.
- The larger `8x512` model and `50000`-conversation training slice were
  materially better than the smaller `6x384` runs.
- Validation was still improving at the end, with no obvious overtraining signal.
- The model learned a recognizable assistant shell more often than earlier runs.

## What Was Not Good

The model is still not a usable chatbot.

Observed sample failures:

- Wrong-task drift: answers often start in an unrelated domain.
- Repetition: local loops like repeated nouns, repeated list items, or repeated
  template phrases.
- Weak stop/turn control: outputs may continue into extra `User:` or
  `Assistant:` turns.
- Generic list/guide mode: the model often writes a generic instructional blob
  instead of answering the prompt directly.

Important caveat: the current sample generation path is too naive, so it may make
the model look worse than it is.

Current sample generation uses:

- `temperature = 0.0`
- fixed `max_new_tokens = 200`
- no explicit stop on `<END_ASSISTANT>`
- no explicit stop on `<CHAT_END>`
- no explicit stop on a new `User:` turn
- no exposed `top_k`, `top_p`, or repetition controls

Fixing evaluation decoding is the next best truth-finding step before launching
another expensive training run.

## How It Was Run

The runner defaults now match the last successful candidate recipe. To rerun the
same recipe explicitly:

```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_ultrachat_subword_v1.jl \
  --dataset-dir tmp/tiny_chatbot_ultrachat_corpus_v1 \
  --output-dir tmp/tiny_chatbot_ultrachat_subword_candidate_run_v1 \
  --context-length 128 \
  --num-layers 8 \
  --num-heads 8 \
  --embedding-size 512 \
  --ffn-hidden-size 2048 \
  --batch-size 16 \
  --epochs 2 \
  --learning-rate 0.0003 \
  --tokenizer-training-text-limit 20000 \
  --train-text-limit 50000 \
  --validation-text-limit 1000 \
  --test-text-limit 1000 \
  --log-every-steps 50 \
  --checkpoint-every-steps 500
```

For the next long training run, use a new output directory rather than
overwriting the current candidate.

## Monitoring

Simple loop that prints whether a run is still active every 10 minutes:

```bash
while pgrep -af 'run_tiny_chatbot_ultrachat_subword_v1.jl' >/dev/null; do date; echo RUNNING; sleep 600; done; date; echo STOPPED
```

Inspect persisted progress:

```bash
cat tmp/tiny_chatbot_ultrachat_subword_candidate_run_v1/progress.json
```

Inspect disk usage:

```bash
df -h .
du -sh tmp/* 2>/dev/null | sort -hr
du -sh tmp/tiny_chatbot_ultrachat_subword_candidate_run_v1/checkpoints
```

## Checkpoint Policy Next Time

Use much sparser checkpoints. The last run wrote many `step_*.jld2` files and
filled disk quickly.

Recommendation:

- For a serious multi-day run, use `--checkpoint-every-steps 5000` or higher.
- Keep per-epoch/final checkpoints.
- After a run completes and the bundle exports successfully, keep:
  - `bundle/`
  - `tokenizer_bundle/`
  - `checkpoints/final_checkpoint.jld2`
  - `metrics.json`
  - `progress.json`
  - `run_recipe.json`
  - sample/prompt files
- Delete intermediate `checkpoints/step_*.jld2` and `checkpoints/epoch_*.jld2`
  unless they are needed for resume/debugging.

Manual cleanup command after a completed run:

```bash
find tmp/tiny_chatbot_ultrachat_subword_candidate_run_v1/checkpoints -maxdepth 1 -type f \( -name 'step_*.jld2' -o -name 'epoch_*.jld2' \) -delete
```

## Next Improvements

Do this first:

1. Add a separate resampling/evaluation path that loads the existing candidate
   bundle and tokenizer without retraining.
2. Generate new samples with stop control:
   - stop at `<END_ASSISTANT>`
   - stop at `<CHAT_END>`
   - stop at a new `\nUser:` turn
   - optionally stop at a repeated `\nAssistant:` turn if it begins a new block
3. Expose supported generation knobs:
   - `max_new_tokens`
   - `temperature`
   - `top_k` or `top_p` if supported by `GenerationConfig`
   - repetition control only if already supported cleanly
4. Compare:
   - current greedy/fixed-length behavior
   - stop-controlled greedy behavior
   - one mild sampling setting, if supported
5. Write results next to the current run, for example:
   - `tmp/tiny_chatbot_ultrachat_subword_candidate_run_v1/sample_outputs_decoding_eval.txt`

Only after that, decide the next expensive training run.

Likely next training directions:

- If decoding improves samples materially: continue this `8x512` path with more
  epochs or a larger training slice.
- If decoding does not help enough but validation was still improving: run a
  longer `8x512` training job with sparse checkpoints.
- If the model remains too generic after better decoding and longer training:
  consider larger context length, more data exposure, or a larger model, but only
  after the inference path is fair.

## Session Conclusions

Use this file as the authoritative handoff for the next session. It is intended
to be enough context to continue without reading the removed experiment trail.

Conclusions from this session:

- The earlier docs-assistant, tiny handcrafted chatbot, OASST-only, and
  prepared-real-text experiments were useful exploration but are no longer the
  active direction.
- The active direction is a real downloaded conversational corpus plus a
  tokenizer-aware subword chatbot runner.
- UltraChat is currently the right corpus path because it is much larger and more
  conversational than the small local/OASST-only corpora.
- The successful `8x512` UltraChat candidate run is the best result so far.
- The run is a meaningful numeric improvement, but it still does not produce a
  usable chatbot.
- The next bottleneck should not be assumed to be model size or data size until
  inference quality is measured with proper stop control.
- The immediate next task is therefore an evaluation/decoding improvement, not
  another long training launch.
- The next long training run should use a new output directory and sparse
  checkpointing.
- Do not recreate the removed small experiment scripts unless there is a clear
  reason; they were removed to reduce confusion and disk/source bloat.
- Keep generated run artifacts in `tmp/` only when they are current or needed for
  comparison. Otherwise keep notes/metrics and delete bulky model artifacts.

Minimal next-session plan:

1. Read this file.
2. Inspect `tools/run_tiny_chatbot_ultrachat_subword_v1.jl`.
3. Add or create a no-training decoding evaluation path for the existing
   candidate bundle.
4. Generate stop-controlled samples.
5. Decide whether the next expensive run should be more epochs, more corpus
   exposure, larger context/model, or data/filtering changes.
