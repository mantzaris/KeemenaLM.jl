# Next TODO

This file is the restart point for the next session.

Current canonical handoff:

- `notes/future_codex_chatbot_improvement_note.md`
- `notes/tiny_chatbot_ultrachat_subword_v3_assistant_loss.md`

Important cleanup state:

- All generated `tmp/` corpora, tokenizer bundles, model weights, checkpoints,
  and run directories from the previous chatbot experiments were intentionally
  deleted to free disk space.
- Only source code and notes remain.
- Do not run commands that reuse tokenizer or run paths under the old `tmp/`
  directories unless those artifacts have been regenerated.

## Current State

Keep working from:

- Corpus prep: `tools/prepare_tiny_chatbot_ultrachat_corpus.py`
- Training runner: `tools/run_tiny_chatbot_ultrachat_subword_v1.jl`
- No-training decoding evaluator: `tools/run_tiny_chatbot_ultrachat_decoding_eval.jl`
- Candidate chat REPL: `tools/run_tiny_chatbot_ultrachat_chat_repl.jl`
- Corpus note: `notes/tiny_chatbot_ultrachat_corpus_v1.md`
- Latest completed run note: `notes/tiny_chatbot_ultrachat_subword_v3_assistant_loss.md`
- Future-session improvement note: `notes/future_codex_chatbot_improvement_note.md`

## Latest Completed Run

The GPU `v3_assistant_loss` run completed on 2026-05-17 and is documented in
`notes/tiny_chatbot_ultrachat_subword_v3_assistant_loss.md`.

Final metrics:

- final step: `106044`
- epoch 2 train loss: `2.74744721791506`
- validation loss: `2.617841907558477`
- test loss: `2.6358939453654227`

Interpretation:

- assistant-only loss is the right training objective for this chatbot path
- the model still is not chatbot-quality
- generated samples and REPL output still drift, repeat, and produce template or
  nonsense fragments
- more epochs on the same broad data are unlikely to be the best next step
- the next improvement should be a cleaner, narrower SFT corpus plus a data/mask
  audit before training

Important caveat:

- the old `tmp/` artifacts are gone, including the v1 tokenizer bundle and all
  v2/v3 model weights. Regenerate data and tokenizer artifacts before training.

## Next Training Direction

The runner default is assistant-only supervised chat loss:

- `--loss-mode assistant_only` is now the default.
- User turns, prompts, role markers, and document separators are context only.
- Loss is applied only to assistant response tokens and chat-ending target
  markers.
- `--loss-mode all_tokens` remains available only for comparison with older
  runs.

This should directly target two observed failure modes from `v2_fixed`:

- continuing into fake `User:` turns
- learning too much generic prompt/user/document continuation behavior

Important comparison caveat:

- validation/test losses from assistant-only runs are not directly comparable to
  `v1` or `v2_fixed`, because the loss denominator now includes only assistant
  target tokens.

Recommended next run name after building a cleaner corpus:

```text
tmp/tiny_chatbot_clean_sft_candidate_run_v4
```

The next session should not start by rerunning the stale v3 command. Instead:

1. regenerate or rebuild a corpus
2. filter it toward short, clean, English assistant responses
3. audit decoded token windows and assistant-loss masks
4. run a tiny smoke training
5. only then launch the next full GPU training run

## Checkpoint Policy

Use sparse checkpoints with retention. The last run wrote many `step_*.jld2`
files and filled disk quickly.

Recommendation:

- The runner now defaults to `--checkpoint-every-steps 5000`.
- The runner now keeps only the latest two step checkpoints and latest two epoch
  checkpoints by default.
- For a serious multi-day run, keep `--checkpoint-every-steps 5000` or higher.
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

## Minimum Next-Session Plan

1. Read this file.
2. Read `notes/future_codex_chatbot_improvement_note.md`.
3. Regenerate or rebuild a training corpus.
4. Create a cleaner filtered SFT variant.
5. Audit decoded examples and assistant-loss masks.
6. Run a tiny smoke training job.
7. Launch a full GPU run only if the smoke output and audit look sane.
8. Evaluate with `tools/run_tiny_chatbot_ultrachat_decoding_eval.jl`.
9. Test interactively with `tools/run_tiny_chatbot_ultrachat_chat_repl.jl`.

The REPL exits with `/exit` or `/quit`.
