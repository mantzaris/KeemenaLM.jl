# Tiny Chatbot Dataset Plan

This plan is for a tiny conversational chatbot demo, not a package QA assistant.

## Target chatbot

- Persona: a small, polite, plainspoken general assistant
- Scope: short everyday chat, simple help, lightweight explanations, harmless brainstorming, and brief follow-up turns
- Non-goals:
  - deep factual accuracy
  - package-specific QA as the main behavior
  - web-scale knowledge
  - production support quality

The desired feel is "small GPT-style chat" rather than "technical documentation lookup."

## Recommended data strategy

Use a hybrid corpus:

- local curated core
  - hand-curated chat pairs and short multi-turn dialogues written specifically for the target style
  - small amount of adapted local Keemena material only where it helps teach helpful assistant behavior
- tiny external conversational supplement
  - a tightly filtered English subset of `OpenAssistant/oasst1`
  - use only short, safe, assistant-style exchanges with clean prompt/reply structure

Why hybrid:

- local material alone is legally safest, but the currently available local text is mostly docs- and workflow-shaped, not conversational
- a tiny external supplement is the cleanest way to teach general assistant rhythm, tone, and turn-taking without downloading or depending on a giant corpus

## Source plan

### 1. Local curated core

Target about `150-250` examples.

Composition:

- greeting / small-task chat
- simple explanation requests
- harmless brainstorming
- rewrite / summarize / clarify requests
- polite refusals and limitations
- a small number of narrow technical help turns derived from Keemena material

The local core should be intentionally authored or tightly curated so it sounds like an assistant, not like copied documentation.

### 2. External supplement

Target about `250-500` examples after filtering.

Recommended source:

- `OpenAssistant/oasst1` English subset

Selection policy:

- single-turn and short two-turn prompt/assistant exchanges only
- short to medium length replies
- clear assistant tone
- remove roleplay, creative-fiction threads, unsafe content, policy-heavy moderation cases, and very domain-specific knowledge prompts
- avoid very long answers, verbose lists, and noisy tree branches

Rejected or lower-priority options:

- `DailyDialog`
  - conversationally attractive, but license is `CC BY-NC-SA 4.0`, so it is not the clean choice for a repo artifact
- `databricks-dolly-15k`
  - license is usable, but it is more instruction-style than conversational and less useful than OASST1 for short back-and-forth feel
- existing Keemena docs corpora alone
  - safe and local, but still too docs-shaped for the main chatbot target

## Dataset format

Use one explicit chat format everywhere:

```text
User: ...
Assistant: ...
<END_ASSISTANT>
<CHAT_END>
```

Rules:

- one assistant answer per training example
- keep the explicit assistant terminator
- keep the explicit chat boundary marker
- allow a small number of multi-turn dialogues, but flatten them into the same explicit format

This keeps continuity with the current training path while making the new corpus intentionally conversational.

## Realistic target size

Recommended first target:

- total `450-700` chat examples
- roughly:
  - `200` high-quality local curated examples
  - `250-500` filtered OASST1 examples

That is still small enough to inspect manually and large enough to teach more natural chat rhythm than the docs-assistant data.

## Split policy

- deterministic split by chat example id
- stable hash ordering, then fixed `80/10/10`
- keep multi-turn conversations grouped so turns from the same dialogue do not split across train and eval

## Quality bar before training

Do not train until the prepared dataset meets these checks:

- no markdown tables or documentation dump answers
- no duplicate questions with conflicting answers
- assistant answers are short, direct, and answer-like
- at least `80%` of samples read like plausible assistant replies
- obvious unsafe or bizarre external examples are filtered out
- the local curated core is visibly present and not drowned out by external data

## Main risks

- too much external data will wash out the desired simple assistant tone
- too little external data will keep the model trapped in docs-shaped behavior
- tiny char-level training may still bottleneck answer quality even with better supervision
- local hand-curation takes real effort if we want the assistant to sound consistent

## Recommended next implementation task

Prepare `tmp/tiny_chatbot_demo_dataset_v1` with:

- a deterministic local curated core file under `tools/` or `notes/` input assets
- a deterministic preparation script under `tools/`
- optional external OASST1 subset ingestion behind an explicit flag or documented step
- output files:
  - `training.txt`
  - `validation.txt`
  - `testing.txt`
  - `training.jsonl`
  - `validation.jsonl`
  - `testing.jsonl`
  - `metadata.json`

The implementation task should stay focused on dataset preparation only, not training.
