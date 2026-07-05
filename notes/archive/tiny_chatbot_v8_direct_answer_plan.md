# Tiny Chatbot v8 Direct-Answer Plan

Date recorded: 2026-06-15

## Purpose

V8 changes the final chatbot objective instead of scaling v7. The v7 run proved
that the scratch Julia pipeline can train a large GPT-style model and drive
losses down, but low chat/SFT loss did not produce reliable chatbot behavior.

The v8 hypothesis is:

```text
The scratch base model can learn language.
The chatbot failure is caused by the final assistant objective and evaluation.
Fix the final objective with direct-answer assistant-only data and behavior gates.
```

## Added Files

- `tools/prepare_tiny_chatbot_v8_direct_answer_corpus.py`
- `tools/run_tiny_chatbot_v8_scratch.jl`
- `tools/run_tiny_chatbot_v8_behavior_eval.jl`
- `src/core/generation/behavior_gate.jl`
- `test/unit/test_behavior_gate.jl`

## Corpus

Prepare a smoke corpus without network downloads:

```bash
python3 tools/prepare_tiny_chatbot_v8_direct_answer_corpus.py \
  --pretrain-source tiny_local \
  --direct-sft-examples 200 \
  --output-dir tmp/tiny_chatbot_v8_direct_answer_corpus_smoke
```

Prepare the full corpus:

```bash
python3 tools/prepare_tiny_chatbot_v8_direct_answer_corpus.py \
  --output-dir tmp/tiny_chatbot_v8_direct_answer_corpus
```

The SFT split uses direct-answer assistant examples for greetings, arithmetic,
colors, yes/no facts, country/capital facts, common facts, immediate user-fact
recall, short rewrites, simple planning, dinner ideas, and uncertainty handling.
It does not use the v6/v7 repeated synthetic closers.

## Training

Smoke run shape:

```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_scratch.jl \
  --dataset-dir tmp/tiny_chatbot_v8_direct_answer_corpus_smoke \
  --output-dir tmp/tiny_chatbot_v8_direct_answer_smoke_run \
  --device cpu \
  --context-length 96 \
  --num-layers 2 \
  --num-heads 2 \
  --embedding-size 128 \
  --ffn-hidden-size 256 \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --pretrain-epochs 1 \
  --behavior-max-updates 20 \
  --behavior-min-updates 5 \
  --validation-every-updates 5 \
  --tokenizer-vocab-size 2048 \
  --tokenizer-training-text-limit 400 \
  --validation-batch-limit 4 \
  --test-batch-limit 4 \
  --behavior-prompt-limit 4
```

Full run defaults keep the v7-scale architecture, but the final stage is now:

```text
streamed_pretrain_all_tokens -> behavior_direct_sft_with_pretrain_replay
```

The raw `chat_lm` all-token continuation stage is intentionally disabled.

## Behavior Gate

The behavior gate rejects outputs with fake role markers, repetition loops,
empty completions, missing required simple answers, forbidden phrases, or
excessive length. Gate prompts cover:

- greeting
- `1 + 1`
- `is green a color?`
- immediate name recall
- USA democracy/direct factual answer
- France capital
- kind rewrite
- two simple dinner ideas

Run the gate after training:

```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_behavior_eval.jl \
  --run-dir tmp/tiny_chatbot_v8_direct_answer_candidate_run
```

A v8 checkpoint should not be treated as promising unless the behavior gate
passes or the failures are narrow and explainable from the decoded data audit.
