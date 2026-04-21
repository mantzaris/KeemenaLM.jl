# Tiny Chatbot Real Corpus v1

This is the first non-toy downloaded conversational corpus for the tiny chatbot path.

## Source strategy

- dominant source: `OpenAssistant/oasst1`
- file: `2023-04-12_oasst_ready.messages.jsonl.gz`
- license: `Apache-2.0`
- small local anchor: the existing curated conversational core

The local anchor stays small and intentional. The downloaded corpus dominates the final mix.

## Filtering policy

- English only
- reviewed, non-deleted, non-synthetic
- accept top reviewed assistant replies up to rank `2`
- extract:
  - ordinary single-turn assistant exchanges
  - short clean multi-turn dialogues
- reject:
  - roleplay-heavy / fiction-heavy / theatrical samples
  - sexual content
  - self-harm / violence / illegal-help content
  - hateful content
  - policy-heavy / citation-heavy / formatting-heavy noise
  - browser/version lookup noise
  - code-helpdesk-heavy samples for this first serious chatbot corpus
  - joke-only / trivia-heavy / oddball examples surfaced during inspection
  - very long essay-like answers

## Format

All examples are normalized to:

```text
User: ...
Assistant: ...
<END_ASSISTANT>
<CHAT_END>
```

Multi-turn examples stay coherent in that same format and end on an assistant turn.

## Split policy

- deterministic stable-hash split by dialogue id / group
- fixed `80/10/10`
- multi-turn dialogues stay grouped together

## Result

Final corpus counts:

- total: `6630`
- training: `5304`
- validation: `663`
- testing: `663`

Source-group counts:

- local curated core: `230`
- OASST1 single-turn: `6041`
- OASST1 multi-turn: `359`

This did not reach the aspirational `20k+` range while staying clean. It is the largest OASST1-only filtered subset from the current path that stayed defensible for ordinary assistant behavior without broadening into noisy helpdesk/trivia chat.
