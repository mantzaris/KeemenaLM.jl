# Tiny Chatbot Demo Dataset v2

This dataset is the second conversational corpus for the tiny chatbot demo path.

## Goal

Make the training corpus more conversational than `v1` by:

- keeping the local curated core as a style anchor
- expanding the external assistant-style slice materially
- adding more short multi-turn dialogues
- reducing formal, trivia-heavy, or essay-shaped assistant behavior

## Target chatbot

- small, polite, plainspoken assistant
- short back-and-forth conversation
- simple explanations
- light planning and brainstorming
- clarification and harmless requests
- a small amount of Keemena-flavored technical help as secondary flavor

This is still a bounded demo corpus, not a production chatbot dataset.

## Source strategy

The dataset is hybrid:

- local curated conversational core
- filtered `OpenAssistant/oasst1` supplement

The local curated core remains visible in the final corpus so the assistant style does not collapse into generic benchmark tone.

## External source

- source: `OpenAssistant/oasst1`
- file: `2023-04-12_oasst_ready.messages.jsonl.gz`
- license: `Apache-2.0`

The external slice now includes:

- filtered single-turn `User -> Assistant` examples
- filtered short multi-turn `User -> Assistant -> User -> Assistant` dialogues

## Filtering policy

The OASST1 slice is filtered to prefer ordinary assistant behavior:

- English only
- reviewed, non-deleted, non-synthetic
- prefer rank `0` or missing rank
- short to medium prompt and answer lengths
- reject roleplay, fiction, sexual content, self-harm, violence, illegal help, hateful content, bizarre outputs, policy-heavy shapes, citation-heavy requests, and code-heavy exchanges
- reject URL-heavy, very long, or noisy formatting-heavy examples
- prefer ordinary assistant/help exchanges over essay, trivia, or academic-style requests

## Output format

Every example is normalized to:

```text
User: ...
Assistant: ...
<END_ASSISTANT>
<CHAT_END>
```

Multi-turn examples stay coherent inside that same format and end on an assistant turn.

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
tmp/tiny_chatbot_demo_dataset_v2
```

Files written:

- `training.txt`
- `validation.txt`
- `testing.txt`
- `training.jsonl`
- `validation.jsonl`
- `testing.jsonl`
- `metadata.json`

## Intended use

This corpus is for the next subword tiny-chatbot retrain. It is meant to give the small model a better chance of learning recognizable conversational turn-taking and short assistant replies than `v1` did.
