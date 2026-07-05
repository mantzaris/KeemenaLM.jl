#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-tmp/tiny_chatbot_v9_broad_corpus_5k_anchor}"

python3 tools/prepare_tiny_chatbot_real_chat_corpus.py \
  --preset wildchat-oasst1 \
  --pretrain-source downloaded \
  --pretrain-max-characters 1500000000 \
  --max-sft-examples 200000 \
  --max-examples-per-category 50000 \
  --anchor-examples 5000 \
  --max-assistant-turn-index 1 \
  --min-user-words 1 \
  --max-user-words 100 \
  --min-assistant-words 1 \
  --max-assistant-words 100 \
  --max-user-chars 800 \
  --max-assistant-chars 900 \
  --max-prompt-chars 1400 \
  --max-user-lines 4 \
  --max-assistant-lines 8 \
  --train-fraction 0.96 \
  --validation-fraction 0.02 \
  --seed 20260616 \
  --output-dir "$OUTPUT_DIR"
