# Tiny Chatbot Demo Dataset v3

This dataset is the third conversational corpus for the tiny chatbot demo path.

## Goal

Make the corpus less brittle and more conversation-first than `v1` or `v2` by:

- keeping the local curated core as a tone anchor
- keeping the external source license-simple
- filtering harder against trivia, code-helpdesk, theatrical, and oddball exchanges
- increasing the share of short clean multi-turn dialogues without turning the corpus into noisy tree fragments

## Source strategy

The dataset stays hybrid:

- local curated conversational core
- filtered `OpenAssistant/oasst1` supplement

No second external source was added for `v3`.

Reason:

- `OpenAssistant/oasst1` is still the cleanest clearly licensed external option already in use
- `Anthropic/hh-rlhf` is MIT-licensed, but its own dataset card says it is not meant for supervised dialogue-agent training
- `DailyDialog` is more conversational, but its `CC BY-NC-SA 4.0` license is less attractive for the mainline demo path

## External source

- source: `OpenAssistant/oasst1`
- file: `2023-04-12_oasst_ready.messages.jsonl.gz`
- license: `Apache-2.0`

The external slice includes:

- filtered single-turn `User -> Assistant` examples
- filtered short multi-turn `User -> Assistant -> User -> Assistant` dialogues

## Filtering policy

Compared with `v2`, `v3` filters harder for ordinary assistant behavior:

- English only
- reviewed, non-deleted, non-synthetic
- prefer rank `0` or missing rank
- first user turn must look like an ordinary assistant/help request
- follow-up user turns use a softer conversational filter so short natural back-and-forth exchanges can survive
- short to medium prompt and answer lengths
- reject roleplay, fiction, sexual content, self-harm, violence, illegal help, hateful content, policy-heavy shapes, citation-heavy shapes, and code-heavy helpdesk patterns
- reject trivia-heavy, browser/version lookup, fandom/game, theatrical, and other oddball examples surfaced during inspection
- reject URL-heavy, formatting-heavy, or overly long answers

## Output format

Every example is normalized to:

```text
User: ...
Assistant: ...
<END_ASSISTANT>
<CHAT_END>
```

Multi-turn examples stay coherent inside the same format and end on an assistant turn.

## Split policy

- deterministic stable-hash split by `dialogue_group`
- fixed `80/10/10`
- multi-turn dialogues stay grouped together

## Build

Run:

```bash
julia --project=. tools/prepare_tiny_chatbot_demo_dataset.jl
```

Default output directory:

```text
tmp/tiny_chatbot_demo_dataset_v3
```

## Intended use

This corpus is for the next subword tiny-chatbot retrain. The target is a more stable small conversational model, not broader factual coverage.
