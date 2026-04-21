# Tiny Chatbot Real Corpus v2

This is the next larger real conversational corpus for the serious subword chatbot path.

## Source strategy

- dominant source: `OpenAssistant/oasst1`
- file: `2023-04-12_oasst_ready.messages.jsonl.gz`
- license: `Apache-2.0`
- small local anchor retained: the existing curated conversational core

The downloaded corpus still dominates strongly. The local anchor stays small and is only there to keep a clear assistant tone anchor.

## Expansion policy from v1

- keep the explicit chat format unchanged
- keep OASST1 as the only external source in this bounded pass
- widen acceptance from reviewed ranks up to `2` to reviewed ranks up to `4`
- widen prompt/answer length bounds slightly for ordinary assistant exchanges
- keep the broad safety and noise rejects from `v1`
- keep multi-turn extraction limited to short clean follow-up dialogues that end on an assistant turn

## Filtering policy

- English only
- reviewed, non-deleted, non-synthetic
- accept reviewed alternatives up to rank `4`
- extract:
  - ordinary single-turn assistant exchanges
  - short clean multi-turn dialogues
- reject:
  - roleplay-heavy / fiction-heavy / theatrical samples
  - sexual content
  - self-harm / violence / illegal-help content
  - hateful content
  - policy-heavy / citation-heavy / formatting-heavy noise
  - code-helpdesk-heavy samples for this main conversational path
  - browser/version lookup noise
  - trivia-heavy / oddball samples surfaced during inspection
  - URL-heavy, overly long, or messy answers

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

- total: `8575`
- training: `6860`
- validation: `857`
- testing: `858`

Source-group counts:

- local curated core: `230`
- OASST1 single-turn: `7796`
- OASST1 multi-turn: `549`

Multi-turn coverage:

- multi-turn examples: `591`
- average turns per example: `1.07`

## Readout

This is materially larger than `tiny_chatbot_real_corpus_v1`, but it also makes the next constraint clear: OASST1 alone does not cleanly reach the aspirational `20k+` range for this assistant-style target under the current quality bar.

So `v2` is a real expansion, but it is still not the larger serious corpus that would make the chatbot path feel comfortably out of the data-starved regime. The next real corpus expansion will likely need either:

- a second clearly licensed conversational source, or
- a more permissive OASST1 policy than this pass used, with the risk of noticeably more noise.
