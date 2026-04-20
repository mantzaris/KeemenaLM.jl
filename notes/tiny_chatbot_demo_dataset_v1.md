# Tiny Chatbot Demo Dataset v1

This dataset is the first tiny conversational chatbot corpus for the demo path.

## Target chatbot

- small, polite, plainspoken assistant
- short back-and-forth conversation
- simple explanations
- light brainstorming
- clarification and harmless requests
- only a small amount of Keemena-flavored technical help as secondary flavor

This is not a docs-assistant dataset and not an open-domain production corpus.

## Data strategy

The dataset is hybrid:

- a local curated conversational core
- plus a very small filtered external supplement from `OpenAssistant/oasst1`

The local curated core should remain visibly present in the final dataset so the tone does not drift into a generic benchmark flavor.

## Output format

Every example uses the same explicit rendered chat format:

```text
User: ...
Assistant: ...
<END_ASSISTANT>
<CHAT_END>
```

Some local examples are short multi-turn dialogues, but they are flattened into the same explicit format and still end with an assistant turn.

## External source

Recommended external source:

- `OpenAssistant/oasst1`
- file: `2023-04-12_oasst_ready.messages.jsonl.gz`
- license: `Apache-2.0`

The external slice is intentionally small and heavily filtered:

- English only
- assistant-style prompt/reply pairs
- short to medium exchanges
- reject roleplay, fiction, erotic/sexual, self-harm, violent, illegal, hateful, bizarre, code-heavy, and very long messy examples

## Local curated core

The local core is intentionally authored for assistant behavior rather than extracted from documentation prose.

It includes:

- greetings
- simple explanations
- everyday task help
- rewrite / clarify requests
- light brainstorming
- planning help
- polite limitations
- a small amount of Keemena-flavored technical help

The Keemena-flavored slice is grounded in local controlled sources such as:

- `docs/src/index.md`
- `notes/official_models.md`
- `../KeemenaPreprocessing.jl/docs/src/index.md`
- selected preprocessing guides
- `../KeemenaSubwords.jl/docs/src/index.md`
- selected subwords loading/integration docs

## Split policy

- deterministic stable-hash split by dialogue id
- fixed `80/10/10`

## Build

Run:

```bash
julia --project=. tools/prepare_tiny_chatbot_demo_dataset.jl
```

Default output directory:

```text
tmp/tiny_chatbot_demo_dataset_v1
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

This dataset is for a first tiny GPT-2-like chatbot demo in Julia. It is intended to teach a recognizable small assistant feel, not strong factual coverage or open-domain knowledge.
