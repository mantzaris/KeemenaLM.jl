# Future Codex Handoff: Chatbot Improvement

Date recorded: 2026-05-17

## Read This First

The Julia training pipeline works, but the project does not yet have a decent
chatbot. Do not spend the next session simply repeating the last training run.

The generated corpus, tokenizer bundles, checkpoints, model weights, and run
directories under `tmp/` were deleted to free disk space. Only source code and
notes remain. Any future run must regenerate or redownload data and train/export
new artifacts.

Useful context notes:

- `notes/tiny_chatbot_ultrachat_subword_v2_fixed.md`
- `notes/tiny_chatbot_ultrachat_subword_v3_assistant_loss.md`
- `notes/tiny_chatbot_ultrachat_corpus_v1.md`
- `notes/NextTODO.md`

## Current Code State

Primary files:

- `tools/prepare_tiny_chatbot_ultrachat_corpus.py`
- `tools/run_tiny_chatbot_ultrachat_subword_v1.jl`
- `tools/run_tiny_chatbot_ultrachat_decoding_eval.jl`
- `tools/run_tiny_chatbot_ultrachat_chat_repl.jl`
- `src/core/training/loss.jl`
- `src/backends/flux/train_flux.jl`
- `test/unit/test_loss.jl`

Important implemented changes:

- GPU training works when launched with the correct CUDA device.
- The training runner supports `--device gpu`.
- Checkpointing is sparse and retained by limit, so it should not fill the disk
  with unlimited step checkpoints unless options are overridden.
- The default chatbot training objective is now `--loss-mode assistant_only`.
- Assistant-only loss uses a token mask so user prompts, role markers, and other
  context are not optimized as target text.
- `--loss-mode all_tokens` still exists for comparison only.
- Decoding evaluation and chat REPL scripts exist for testing a saved run.
- The trained model can be stopped in the REPL with `/exit` or `/quit`.

## Last Known Results

### v2 Fixed All-Token Run

- Run directory was `tmp/tiny_chatbot_ultrachat_subword_candidate_run_v2_fixed`.
- Artifacts were deleted.
- Final validation loss: `2.6447167355808423`
- Final test loss: `2.668845098776673`
- Qualitative result: not chatbot-quality. It drifted off task and repeated.

### v3 Assistant-Loss Run

- Run directory was `tmp/tiny_chatbot_ultrachat_subword_candidate_run_v3_assistant_loss`.
- Artifacts were deleted.
- Final step: `106044`
- Epochs: `2`
- Model: `8` layers, `8` heads, `512` embedding, `2048` FFN, context `128`
- Batch size: `16`
- Learning rate: `0.0003`
- Final train loss: `2.74744721791506`
- Final validation loss: `2.617841907558477`
- Final test loss: `2.6358939453654227`
- Qualitative result: still failed. Greedy decoding produced repeated fragments;
  sampling produced nonsensical text and template/list drift.

The assistant-only objective is still the right direction, but it did not fix
the core quality problem.

## Current Diagnosis

The bottleneck is probably not just "more epochs." The next likely issues are:

- Training data is too broad and noisy for a small chatbot demo.
- UltraChat contains template-like, document-like, list-heavy, non-chatty, and
  sometimes multilingual material that a tiny model overlearns.
- Context length `128` is short, so examples are often truncated or learned as
  fragments.
- The model is trained from scratch; it needs a cleaner, narrower supervised
  target than a broad general corpus.
- Generation quality should be judged with stop-controlled evaluation, not only
  fixed-length sampling.

## Recommended Next Direction

Build a cleaner small SFT dataset before launching another long training run.
The goal should be a modest but coherent chatbot, not broad knowledge.

Recommended data rules:

- Keep English conversational examples only.
- Prefer short single-turn or clean two-turn conversations.
- Prefer direct assistant answers over essays, lists, templates, code blocks,
  roleplay, documents, or marketing copy.
- Filter out examples with placeholders like `[Name]`, `[Address]`, repeated
  headings, tables, scripts, lyrics, and boilerplate letters.
- Target assistant replies that are complete but short, roughly `20` to `200`
  words.
- Use a clear format:
  - `User: ...`
  - `Assistant: ...`
  - `<END_ASSISTANT>`
  - `<CHAT_END>`
- Keep assistant-only loss masking.
- Do not train loss on user turns.

Before full training, add or run a data/mask audit that prints random tokenized
training windows with:

- decoded input text
- decoded target text
- which spans are masked as assistant loss
- counts of assistant target tokens per batch

If the audit shows broken examples, repeated templates, or mask errors, fix the
data before training.

## Suggested Future Plan

1. Regenerate or rebuild the corpus because `tmp/` was intentionally cleaned.
2. Add a filtered corpus variant, for example:
   `tmp/tiny_chatbot_clean_sft_corpus_v1`.
3. Run a small CPU/GPU smoke training first with a tiny data limit to verify:
   - tokenizer training or loading
   - assistant masks
   - progress logs
   - checkpoint retention
   - bundle export
   - REPL loading
4. Train a new candidate only after the smoke run produces sensible decoded
   samples.
5. Try `context_length = 256` if the 32GB GPU has enough memory. If it OOMs,
   reduce batch size before shrinking the model.
6. Evaluate with `tools/run_tiny_chatbot_ultrachat_decoding_eval.jl`.
7. Use the REPL only after reading the decoding report.

Possible next run name:

```text
tmp/tiny_chatbot_clean_sft_candidate_run_v4
```

Possible training shape:

```text
device: gpu
loss_mode: assistant_only
context_length: 256
num_layers: 8
num_heads: 8
embedding_size: 512
ffn_hidden_size: 2048
batch_size: 8 or 16
epochs: 2 to 4
learning_rate: 0.0003
checkpoint_every_steps: 5000
max_step_checkpoints: 2
max_epoch_checkpoints: 2
```

Do not reuse old tokenizer paths from deleted `tmp/` directories. Either train a
new tokenizer bundle or deliberately keep a freshly generated tokenizer bundle
for the next run.

## What Success Should Look Like

For this stage, success is not benchmark-grade intelligence. A decent demo
chatbot should:

- answer greetings normally
- remember the immediate user message within a short exchange
- avoid continuing as `User:`
- stop after one assistant response
- avoid obvious repetition loops
- avoid template placeholders and unrelated list/document output
- give short direct answers to simple prompts

If the next model still fails these checks, inspect decoded training examples
and mask spans before changing model size.
