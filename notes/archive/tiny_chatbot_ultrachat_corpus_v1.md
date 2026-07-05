# Tiny Chatbot UltraChat Corpus v1

This is the first serious single-source conversational corpus built from `HuggingFaceH4/ultrachat_200k`.

## Source strategy

- source dataset: `HuggingFaceH4/ultrachat_200k`
- dataset card: <https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k>
- license: `MIT`
- selected splits:
  - `train_sft`
  - `test_sft`

This corpus intentionally uses the SFT-style UltraChat conversations only. It does not mix in the `gen` splits, and it does not include the earlier local curated anchor.

## Why this split choice

- the SFT splits are the cleanest first serious supervised-chat source in the dataset
- the goal here is a practical chatbot-training corpus, not the largest raw UltraChat dump
- keeping the source effectively single-source makes the next serious retrain easier to interpret

## Format

All conversations are normalized to the repo’s explicit chat format:

```text
User: ...
Assistant: ...
<END_ASSISTANT>
<CHAT_END>
```

Multi-turn dialogues are preserved in-order and required to end on an assistant turn.

## Filtering policy

- keep only user/assistant dialogues
- require conversations to start with `User` and end with `Assistant`
- keep `1` to `6` assistant turns per example
- reject extremely short, extremely long, or obviously malformed turns
- reject URL-heavy, markdown-heavy, table-heavy, or formatting-heavy samples
- reject code-dump-heavy samples and conversations dominated by programming syntax
- reject obviously pathological or off-target roleplay, erotic, self-harm, violence, illegal-help, hateful, or prompt-injection-style prompts
- do not overfilter toward a tiny handcrafted corpus; keep broad ordinary assistant behavior, including practical and factual dialogues

## Split policy

- deterministic stable-hash split by dialogue id / group
- fixed `80/10/10`
- conversation/dialogue units stay together

## Result

Final corpus counts:

- total: `140895`
- training: `112716`
- validation: `14089`
- testing: `14090`

Source-group counts:

- `ultrachat_sft`: `140895`

Multi-turn coverage:

- multi-turn examples: `140895`
- average turns per example: `3.14`

## Readout

This corpus is materially larger and more stable than the earlier OASST1-based real corpus. It decisively leaves the repo’s tiny-corpus regime and gives the next serious subword chatbot retrain a much more realistic amount of conversational supervision.

At the same time, it is not a perfectly “ordinary casual assistant” corpus. Even after filtering, UltraChat still contains:

- a fair amount of long-form instructional behavior
- some specialized factual/help content
- some list-heavy or guide-style answers that are still conversational enough to keep, but not especially casual

So the main remaining risk before training is no longer “too little data.” It is tone mix and answer style mix inside the much larger corpus.
