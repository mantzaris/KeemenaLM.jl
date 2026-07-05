# Real Chat Message Importer

Date recorded: 2026-06-16

## Why This Exists

Most public chat datasets are not stored in the exact token sequence used by a
model. They are usually stored as structured messages, ShareGPT-style
conversations, prompt/response rows, or tree-structured messages.

Common Python training stacks hide the conversion through chat templates. This
repo now has the same idea as a Julia-compatible data-prep step:

```text
real chat rows
-> normalized User/Assistant messages
-> prompt_text + assistant target_text JSONL
-> existing assistant-only SFT batching
```

The runner still computes assistant-only loss through the existing clean SFT
batcher. The importer only prepares the data.

## Added Tool

```text
tools/prepare_tiny_chatbot_real_chat_corpus.py
```

Supported input shapes:

- `messages: [{role, content}, ...]`
- `conversation` / `conversations` / `turns` / `dialog` / `chat`
- ShareGPT-style `{from, value}` turns
- prompt/response rows such as `prompt` + `response`, `question` + `answer`, or `instruction` + `output`
- OASST-style message trees with `message_id`, `parent_id`, `role`, `text`

Supported file types:

- `.parquet`
- `.jsonl`
- `.jsonl.gz`
- `.json`

Supported Hugging Face presets:

- `wildchat` -> `allenai/WildChat`
- `lmsys-chat-1m` -> `lmsys/lmsys-chat-1m`
- `oasst1` -> `OpenAssistant/oasst1`
- `wildchat-oasst1`

Some datasets may be gated or have license/use restrictions. Accept those terms
outside this repo before downloading.

## Smoke Test

Use a local file and tiny pretrain text:

```bash
python3 tools/prepare_tiny_chatbot_real_chat_corpus.py   --source-file tmp/real_chat_importer_smoke.jsonl   --pretrain-source tiny_local   --output-dir tmp/real_chat_importer_smoke_corpus
```

## Real Chat Pilot

The first WildChat pilot finished at 3/8 behavior cases with high SFT loss. It
proved the pipeline works, but it also showed that the default import was too
permissive for a tiny model: long marketing/social-media prompts, jailbreak
style rows, and generic AI-disclaimer answers survived the filter. The importer
now defaults to shorter first-assistant-turn examples, rejects more prompt
injection and marketing prompts, and keeps synthetic behavior anchors before
real rows when the SFT cap is applied.

Recommended WildChat-only pilot v2:

```bash
python3 tools/prepare_tiny_chatbot_real_chat_corpus.py   --preset wildchat   --hf-max-files 2   --pretrain-max-characters 200000000   --max-sft-examples 30000   --max-examples-per-category 8000   --anchor-examples 3000   --output-dir tmp/tiny_chatbot_real_chat_wildchat_pilot_corpus_v2
```

Then train with the existing v8 runner:

```bash
CUDA_VISIBLE_DEVICES=GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_scratch.jl   --dataset-dir tmp/tiny_chatbot_real_chat_wildchat_pilot_corpus_v2   --output-dir tmp/tiny_chatbot_real_chat_wildchat_pilot_run_v2   --device gpu   --pretrain-epochs 1   --behavior-max-updates 5000   --behavior-min-updates 1000   --validation-every-updates 500   --direct-sft-parts 7   --pretrain-replay-parts 1   --no-base-checkpoint
```

This keeps disk use low. Do not pass `--save-final-checkpoint` unless an extra
large checkpoint is needed. The runner exports the final lightweight bundle and
skips the base checkpoint with `--no-base-checkpoint`.

## Real Chat Pilot v3

Pilot v2 improved from 3/8 to 4/8 behavior cases and removed the long capital
loop, but the remaining failures were deterministic basics: arithmetic, name
recall, France capital, and kind rewrite. The v2 corpus had 12,699 SFT examples
with 3,000 behavior anchors, so each anchor skill only had about 200 examples
against much larger generic real-chat categories. For this tiny model, the next
pilot should be anchor-heavy and should preserve the best validation bundle.

Prepare v3 with about 10k behavior anchors and about 6k filtered real-chat rows:

```bash
python3 tools/prepare_tiny_chatbot_real_chat_corpus.py   --preset wildchat   --hf-max-files 2   --pretrain-max-characters 200000000   --max-sft-examples 16000   --max-examples-per-category 1400   --anchor-examples 10000   --max-user-words 60   --max-assistant-words 80   --max-user-chars 500   --max-assistant-chars 700   --max-prompt-chars 1000   --output-dir tmp/tiny_chatbot_real_chat_wildchat_pilot_corpus_v3
```

Train v3 with a stronger SFT ratio, more frequent validation, and one extra
overwritten best-behavior bundle:

```bash
CUDA_VISIBLE_DEVICES=GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_scratch.jl   --dataset-dir tmp/tiny_chatbot_real_chat_wildchat_pilot_corpus_v3   --output-dir tmp/tiny_chatbot_real_chat_wildchat_pilot_run_v3   --device gpu   --pretrain-epochs 1   --behavior-max-updates 8000   --behavior-min-updates 1000   --validation-every-updates 250   --direct-sft-parts 11   --pretrain-replay-parts 1   --save-best-behavior-bundle   --no-base-checkpoint
```

This stores no base checkpoint and no final checkpoint. It writes the normal
final `bundle/` plus, if enabled, one overwritten `best_behavior_bundle/`. That
stays within the one-or-two-artifact disk rule while avoiding the v2 problem
where the best behavior pass rate happened before the final update.

## Real Chat Pilot v4

Pilot v3 ended at 3/8, but the saved best-behavior bundle reached 5/8 at update
6250. The remaining best-bundle failures were arithmetic, name recall, and
France capital. The important finding was that 10k anchors did not produce 10k
useful basics: the old generator saturated at only 234 arithmetic examples, 300
capital examples, and 270 color examples. The v4 generator fixes that by
weighting and expanding deterministic basic categories. A 10k anchor smoke check
now produces roughly 1.1k arithmetic, 1.1k name-recall, 940 color, and 890
capital examples.

To avoid another large download, build v4 from the already downloaded v3 source
files and reuse the v3 pretrain splits:

```bash
python3 tools/prepare_tiny_chatbot_real_chat_corpus.py   --source-name wildchat_v3_cache   --source-file tmp/tiny_chatbot_real_chat_wildchat_pilot_corpus_v3/cache/allenai_wildchat/data/train-00000-of-00006.parquet   --source-file tmp/tiny_chatbot_real_chat_wildchat_pilot_corpus_v3/cache/allenai_wildchat/data/train-00001-of-00006.parquet   --pretrain-source existing   --existing-pretrain-dir tmp/tiny_chatbot_real_chat_wildchat_pilot_corpus_v3/pretrain   --max-sft-examples 22000   --max-examples-per-category 2200   --anchor-examples 18000   --max-user-words 50   --max-assistant-words 70   --max-user-chars 450   --max-assistant-chars 650   --max-prompt-chars 900   --output-dir tmp/tiny_chatbot_real_chat_wildchat_pilot_corpus_v4
```

Then train v4 with the same disk-safe artifact policy:

```bash
CUDA_VISIBLE_DEVICES=GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_scratch.jl   --dataset-dir tmp/tiny_chatbot_real_chat_wildchat_pilot_corpus_v4   --output-dir tmp/tiny_chatbot_real_chat_wildchat_pilot_run_v4   --device gpu   --pretrain-epochs 1   --behavior-max-updates 9000   --behavior-min-updates 1000   --validation-every-updates 250   --direct-sft-parts 15   --pretrain-replay-parts 1   --save-best-behavior-bundle   --no-base-checkpoint
```

After v4 corpus prep succeeds, the v4 trainer does not need any `cache/`
directory. If disk gets tight, remove old downloaded cache directories first.

Pilot v4 reached 5/8 at update 1000 and stopped early on the pretrain-regression
guard. Its one-off continuation wrapper has been retired; future continuation
runs should use the current v8 scratch runner directly with an explicit
`--initial-bundle-dir` and matching tokenizer bundle.

## Full Real Chat Direction

After the pilot, prefer a real-chat corpus over generated-direct data:

```bash
python3 tools/prepare_tiny_chatbot_real_chat_corpus.py   --preset wildchat-oasst1   --pretrain-max-characters 1500000000   --max-sft-examples 200000   --max-examples-per-category 50000   --anchor-examples 5000   --output-dir tmp/tiny_chatbot_real_chat_corpus
```

Only reduce the anchor slice after the behavior gate is stable. For the current
tiny model, v3 showed that too few deterministic anchors cause basic fact,
arithmetic, and recall failures even when SFT loss improves.

## Notes

- Conversion is expected. The important change is that conversion is now generic and message-schema based.
- Assistant-only masking remains in the runner, not in the importer.
- The importer writes the same `sft/training.jsonl`, `sft/validation.jsonl`, and `sft/testing.jsonl` layout that `tools/run_tiny_chatbot_v8_scratch.jl` already understands.
- For tiny models, prefer quality and fit over raw chat volume. First-turn, short-answer real chat plus a large enough balanced behavior-anchor slice is a better pilot shape than a very broad, noisy 200k-row chat dump. v2 used 3k anchors and was still diluted; v3 intentionally raises that to 10k.

## Real Chat Pilot v5 Repair

The v4 continuation produced the strongest checkpoint so far. The final bundle
was 6/8, but `best_behavior_bundle/` reached 7/8 at update 4750. That best
bundle failed only `usa_democracy`. Manual samples also showed that the gate was
still too forgiving on two quality points: green could pass without preserving
"green", and dinner ideas could repeat the same food twice.

The v5 repair corpus avoids another raw dataset download. It reuses the prepared
v4 corpus, hardlinks the v4 pretrain splits when possible, keeps a balanced v4
SFT subset, and adds targeted repair examples for:

- USA democracy wording
- exact green color identity
- the `You forgot again.` kind rewrite
- two distinct dinner ideas

Prepare the compact repair corpus:

```text
retired: legacy v5 repair corpus wrapper removed from active tool surface
```

Historical note: this continued from an old v4 continuation bundle. The wrapper and local run path are retired:

```text
retired: legacy v5 repair runner wrapper removed from active tool surface
```

The v5 run uses `--keep-best-behavior-only`, so after final evaluation it keeps
only the overwritten `best_behavior_bundle/` model export in the new run
directory. Metrics, samples, tokenizer bundle, and behavior eval files remain.
This is the preferred low-disk continuation command.

Result: v5 repair reached 8/8 behavior cases at update 2500. Validation losses
were `pretrain=3.0802` and `sft=2.3161`; test losses were `pretrain=3.0547` and
`sft=2.2434`. The final bundle was removed after evaluation, so the active model
is:

```text
tmp/tiny_chatbot_real_chat_wildchat_pilot_run_v5_repair/best_behavior_bundle
```

Start an interactive v8 chat session with:

```text
retired: legacy v5 chat wrapper removed from active tool surface
```

Use greedy decoding first. To experiment with sampling, override environment
variables, for example the current v8 chat REPL with equivalent decoding flags.

## Real Chat Pilot v6 Repair

Interactive testing exposed that v5 was only an anchor pass, not yet a robust
chatbot. The original 8-case gate passed, but nearby prompts failed:

- `what is 2 plus 2?` -> answered `0`
- `is blue a color?` -> answered about green
- `what is the capital of Japan?` -> answered Australia/Canberra
- `is John a name?` -> answered a math pattern
- `I like cats` -> fell back to greeting

The behavior gate now includes these nearby cases, raising the gate from 8 to 13
cases. Under the widened gate, the v5 best bundle scores 8/13.

Prepare the v6 repair corpus, which reuses v5 data and adds small-sum, capital,
color subject, name-fact, simple preference chat, and memory phrasing repairs:

```text
retired: legacy v6 repair corpus wrapper removed from active tool surface
```

Continue from the v5 best bundle:

```text
retired: legacy v6 repair runner wrapper removed from active tool surface
```

This run keeps the same low-disk policy: no base checkpoint, no final checkpoint,
and `--keep-best-behavior-only` so only the best behavior bundle is retained.

## Real Chat Pilot v7 Repair

v6 passed the 13-case widened gate and fixed the v5 failures for exact probe
prompts. Fresh one-turn probing showed good answers for `2+2`, blue, Japan with
capitalized `Japan`, name recall, rewrite, and dinner. However, REPL testing and
targeted probes still exposed brittle casing/generalization:

- `what is the capital of japan?` -> answered Australia/Canberra
- `what is the capital of canada?` -> answered Australia/Canberra
- `what is 3+4?` -> answered `12`
- `is john a name?` -> answered a math pattern

The v8 REPL now defaults to stateless one-turn prompts because arbitrary carried
history can corrupt later answers. Stateful mode is still available with
stateful mode in the current v8 chat REPL, but it is not the
recommended default for this tiny model.

The v7 repair gate adds lowercase country/name checks and `3+4`. The v7 corpus
reuses v6 data and adds dense small addition pairs, lowercase/titlecase capital
prompts, lowercase/titlecase name facts, color casing, and simple preference
chat variants.

Prepare v7 data:

```text
retired: legacy v7 repair corpus wrapper removed from active tool surface
```

Continue from v6 best:

```text
retired: legacy v7 repair runner wrapper removed from active tool surface
```

After training, use:

```text
retired: legacy v7 chat wrapper removed from active tool surface
```

## General Repair Stage

v7 improved the targeted probes but still showed template collapse on broader
chat prompts:

- `My name is Alex.` -> answered with a color fact
- `Who was Plato?` -> answered with arithmetic
- `How many wheels on a car?` -> answered with a days-in-week fact
- `What is the meaning of life?` -> answered with a capital fact

This means the repair loop had become too specific. The next stage is not another
single-prompt patch; it is a class-based general repair corpus. The behavior gate
now has 22 cases and includes statement acknowledgments, small arithmetic,
lowercase capital/name prompts, simple general facts, open-ended prompts, and
unknown private facts.

Prepare the general repair corpus:

```text
retired: legacy general repair corpus wrapper removed from active tool surface
```

Continue from v7 best:

```text
retired: legacy general repair runner wrapper removed from active tool surface
```

After training, test with:

```bash
retired: legacy general repair chat wrapper removed from active tool surface
```

This run keeps the low-disk policy: no base checkpoint, no final checkpoint, and
only one overwritten `best_behavior_bundle/` model artifact.

## General Repair Result And Direction

The general repair run improved the exact behavior gate but did not solve the
chatbot problem. Final behavior pass rate was 20/22 (`0.909`), with failures on
`canada_capital_lowercase` and `plato_fact`. The saved best bundle reached
21/22 (`0.955`) but still failed `canada_capital_lowercase`.

Manual REPL testing remained the more important signal:

- `Is earth a planet?` -> answered with a color template
- `I slept well today.` -> answered with an arithmetic template
- earlier broad prompts such as `Who was Plato?`, `How many wheels on a car?`,
  and `What is the meaning of life?` exposed the same nearest-template failure
  before they were added to the repair set

Conclusion: the repair-loop approach has become overfit. The model is fitting
synthetic short-answer classes and choosing among memorized templates rather
than learning broad conversational behavior. Lower SFT loss and higher behavior
gate pass rate are no longer reliable signs of chatbot quality under this data
mix.

Do not keep extending the repair chain with v8/v9/v10 single-prompt fixes. The
next serious attempt should be a broader from-scratch run:

- Broad pretraining first, with more varied real text and topics.
- Broad real chat / QA SFT second, where real examples dominate the corpus.
- Synthetic anchors should be a small guardrail slice, roughly 5-15%, not the
  main training distribution.
- Increase pretrain replay during SFT so the model does not forget general
  language behavior while learning assistant responses.
- Keep behavior gates as regression tests, not as the primary training corpus.
- Add held-out random variants and ordinary chat prompts to evaluation, because
  exact gate prompts are too easy to memorize.

Recommended next direction: build a v9-style broad corpus from scratch instead
of continuing from `tmp/tiny_chatbot_real_chat_wildchat_pilot_run_general_repair`.
