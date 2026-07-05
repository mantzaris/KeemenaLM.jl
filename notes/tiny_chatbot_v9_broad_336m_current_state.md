# Tiny Chatbot v9 Broad 336M Current State

Date recorded: 2026-07-04

## Summary

This run is the strongest scratch-chatbot experiment in the repository so far,
but it is still not a usable chatbot.

The result is publishable as a proof-of-concept and negative/partial result:
the Julia training, export, tokenizer bundle, behavior evaluation, and REPL
paths all worked at a 336M parameter scale, but the trained assistant still
fails basic common-sense, arithmetic, factual-grounding, and safety-adjacent
questions.

The current interpretation is that the project is more data- and
objective-limited than parameter-limited. A larger model might help later, but
another week-long run with the same kind of data would probably make the same
failure modes more fluent instead of fixing them.

## Run Identity

- Dataset directory: `tmp/tiny_chatbot_v9_broad_corpus_5k_anchor`
- Run directory: `tmp/tiny_chatbot_v9_broad_336m_run`
- Final inference bundle: `tmp/tiny_chatbot_v9_broad_336m_run/bundle`
- Best behavior-gate bundle: `tmp/tiny_chatbot_v9_broad_336m_run/best_behavior_bundle` (removed after cleanup)
- Tokenizer bundle: `tmp/tiny_chatbot_v9_broad_336m_run/tokenizer_bundle`
- Metrics file: `tmp/tiny_chatbot_v9_broad_336m_run/metrics.json`
- Behavior eval file: `tmp/tiny_chatbot_v9_broad_336m_run/behavior_eval.json`
- Samples file: `tmp/tiny_chatbot_v9_broad_336m_run/sample_outputs.txt`

Generated `tmp/` data, bundles, and weights are local artifacts and are ignored
by git. They should not be committed to the repository.

After documentation, redundant local artifacts were removed to free disk space:

- removed old smoke run: `tmp/tiny_chatbot_v9_smoke_run`
- removed regenerated training corpus: `tmp/tiny_chatbot_v9_broad_corpus_5k_anchor`
- removed duplicate saved behavior bundle:
  `tmp/tiny_chatbot_v9_broad_336m_run/best_behavior_bundle`

The retained local run state is the final model bundle, tokenizer bundle, and
small run metadata under `tmp/tiny_chatbot_v9_broad_336m_run`.

## Model

- Backend: Flux
- Device: GPU
- Layers: `24`
- Attention heads: `16`
- Embedding size: `1024`
- FFN hidden size: `4096`
- Context length: `512`
- Vocabulary size: `32768`
- Estimated parameter count: `336,488,448`
- Optimizer: `Flux.Adam(0.00006)`
- Batch size: `1`
- Gradient accumulation: `4`
- Effective batch size: `4`

## Dataset

The corpus mixed broad pretraining text, real chat/instruction examples, and a
small direct-answer anchor set.

Recorded counts:

- Pretrain characters: `1,501,199,296`
- Pretrain split:
  - train documents: `3,756,878`
  - validation documents: `18,850`
  - test documents: `18,569`
- SFT total examples: `39,584`
- SFT split:
  - train examples: `38,056`
  - validation examples: `789`
  - test examples: `739`
- Synthetic direct-answer anchors: `5,000`
- Synthetic anchor share of SFT: about `12.6%`

The SFT data came mostly from filtered WildChat and OpenAssistant/OASST-style
rows, with the synthetic anchor examples covering simple tasks such as
greetings, arithmetic, colors, capitals, name recall, rewrites, dinner ideas,
and uncertainty handling.

Important limitation: the broad real-chat examples were noisy and uneven. The
anchor set was useful, but too small and too narrow to reliably teach basic
assistant behavior to a scratch 336M model.

## Training Recipe

Curriculum:

1. `streamed_pretrain_all_tokens`
2. `behavior_direct_sft_with_pretrain_replay`

Disabled curriculum:

- raw chat-LM all-token continuation

Behavior-stage ratio:

- direct SFT parts: `3`
- pretrain replay parts: `2`

Other behavior-stage settings:

- maximum behavior updates: `35,000`
- minimum behavior updates: `2,000`
- validation every updates: `1,000`
- behavior minimum pass rate target: `1.0`
- best behavior bundle saved
- base checkpoint not saved
- final checkpoint not saved

The run stopped because it reached `max_behavior_updates_reached`, not because
the behavior gate passed.

## Quantitative Results

Pretrain stage:

- optimizer updates: `166,296`
- microbatches: `665,183`
- train loss: `3.0982`
- validation loss: `3.3402`
- validation perplexity: `28.2260`
- loss target count: `336,816,820`

Behavior stage:

- behavior updates: `35,000`
- final global step: `201,296`
- train loss: `2.1155`
- validation pretrain loss: `2.0983`
- validation SFT loss: `3.1157`
- test pretrain loss: `2.0753`
- test SFT loss: `3.1421`
- behavior pass rate: `17/22`, or `0.7727`

The final bundle failed these behavior-gate cases:

- `arithmetic_one_plus_one`
- `arithmetic_three_plus_four_symbol`
- `car_wheels`
- `private_unknown`
- `john_is_name`

The saved `best_behavior_bundle` also reached `17/22`, but with a different
failure set. Manual testing suggests the final `bundle` is the better default
for qualitative testing because the best-behavior bundle loops more often.

## Qualitative Result

The final bundle is clearly better than earlier v7/v8 overfit repair attempts:

- It usually avoids fake `User:` role continuation.
- It can answer some simple direct facts, such as `London is a city`.
- It can answer some capital and color prompts.
- It gives recognizable greetings.
- It handles some short rewrite and dinner-idea prompts.

It is still not reliable:

- Common sense is weak.
  - It said paper could be eaten.
  - It treated metal as if it were an edible or ordinary answer.
  - It said humans and cats can fly.
- Arithmetic is brittle.
  - `1 plus 1` produced `4`.
  - `3+4` produced `4`.
  - The best-behavior bundle collapsed into a long arithmetic repetition loop.
- Factual grounding is weak.
  - It invented a book title.
  - It produced nonsense for Japan being famous.
  - It gave a fluent but false Plato answer.
  - It repeated itself badly on Star Trek.
- Unknown-private-fact handling is weak.
  - It hallucinated an answer for `What is my favorite food?`
- Repetition remains a problem.
  - Some answers repeat phrases such as `constitutional republic`,
    `physical world`, or `hunting`.
- Safety-adjacent preference learning is not robust.
  - It does not consistently prefer safe ordinary objects over unsafe ones.

Representative final-bundle behavior-gate samples:

```text
User: hello
Assistant: Hello !  How  can  I  help  you ?

User: what is 1 plus 1?
Assistant: 1  plus  1  is  equal  to  4 .

User: what is the capital of France?
Assistant: The  capital  of  France  is  Paris .

User: How many wheels on a car?
Assistant: A  car  on  a  car  can  have  five  wheels  on  its  own .

User: What is my favorite food?
Assistant: My  favorite  food  is  a  large  language  model  that  is  designed
to  help  you  with  something .  It  is  designed  to  help  you  with  something .
```

## Interpretation

This result supports the current data-limited diagnosis.

The model learned enough language structure to produce short assistant-like
responses, but not enough grounded behavior to be dependable. The biggest
problem is not that the training loop cannot run. The problem is that the data
and evaluation objective do not yet put enough pressure on the specific behavior
wanted from a small scratch assistant.

Likely causes:

- `39,584` SFT examples is small for a 336M scratch assistant.
- Broad chat data is noisy and often teaches vague or overlong assistant style.
- The `5,000` synthetic anchor examples are useful but too narrow.
- The behavior gate is too small and too easy to satisfy through keyword hits.
- Validation loss does not capture common-sense or factual reliability.
- The training data does not provide enough repeated, varied coverage for simple
  facts, everyday affordances, arithmetic, safe choices, and unknown-private-fact
  refusal.

## Publication Positioning

This repository should not claim to ship a good LLM or a usable assistant.

Accurate positioning:

- Julia proof-of-concept for training a GPT-style decoder from scratch
- End-to-end package pipeline: training, checkpointing policy, tokenizer bundle,
  inference bundle, behavior gate, and chat REPL
- 336M scratch-chatbot baseline that demonstrates partial instruction-following
  and clear failure modes
- Open research scaffold for improving small scratch assistants through data,
  evaluation, and training recipe changes

Avoid claiming:

- general chatbot quality
- safety
- factual reliability
- competitive LLM performance
- that increasing parameter count alone is the next fix

If weights or prepared datasets are shared outside the repo, check upstream
dataset licenses and privacy constraints first. The safest default is to publish
source code, preparation scripts, exact run recipes, metrics, and sample outputs,
while keeping raw downloaded datasets and generated `tmp/` corpora out of git.

## Recommended Collaboration Path

The best way for other researchers to help is to make failures reproducible and
easy to target.

High-value collaboration areas:

- Data curation:
  - build a cleaner v10 SFT set with many varied short answers
  - add common-sense yes/no examples
  - add safe-choice examples such as food vs non-food objects
  - add broad arithmetic phrasings
  - add everyday object affordances
  - add explicit unknown-private-fact answers
- Evaluation:
  - expand behavior cases from 22 prompts to a larger held-out suite
  - separate keyword-pass scoring from human-readable qualitative samples
  - add repetition, contradiction, unsafe-choice, and hallucination checks
- Training:
  - test short continuations from the final bundle before another full run
  - compare replay ratios and lower learning rates
  - improve early stopping around manual-quality proxies
- Infrastructure:
  - make corpus manifests easier to reproduce
  - keep model bundles and tokenizer bundles exportable without committing
    large files
  - add scripts that produce compact experiment reports from `metrics.json`

Recommended next experiment:

1. Do not start a 0.5B full run yet.
2. Build a v10 data pass with much better coverage of the current failures.
3. Expand the behavior eval before training.
4. Continue from the final v9 bundle for a short 5k-10k update run with lower
   learning rate and pretrain replay.
5. Only run a larger model after the smaller continuation shows the data and
   evaluation are moving the right failures.

## Commands For Local Evaluation

Behavior gate:

```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_behavior_eval.jl \
  --run-dir tmp/tiny_chatbot_v9_broad_336m_run
```

Greedy one-turn REPL with the final bundle:

```bash
CUDA_VISIBLE_DEVICES=0 julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_chat_repl.jl \
  --run-dir tmp/tiny_chatbot_v9_broad_336m_run \
  --bundle-dir tmp/tiny_chatbot_v9_broad_336m_run/bundle \
  --tokenizer-bundle-dir tmp/tiny_chatbot_v9_broad_336m_run/tokenizer_bundle \
  --device gpu \
  --temperature 0.0 \
  --top-k 0 \
  --top-p 1.0 \
  --max-new-tokens 120
```

Prompt probe:

```bash
CUDA_VISIBLE_DEVICES=0 julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_prompt_probe.jl \
  --run-dir tmp/tiny_chatbot_v9_broad_336m_run \
  --bundle-dir tmp/tiny_chatbot_v9_broad_336m_run/bundle \
  --tokenizer-bundle-dir tmp/tiny_chatbot_v9_broad_336m_run/tokenizer_bundle \
  --device gpu \
  --temperature 0.0 \
  --top-k 0 \
  --top-p 1.0 \
  --max-new-tokens 120
```

## Release Artifact Packaging

Do not commit model weights to git. To publish the final model, create a Julia-artifact-friendly release tarball and upload it to a GitHub Release or another large-file host:

```bash
tools/package_tiny_chatbot_v9_release_artifact.sh
```

The package contains:

- `bundle/`
- `tokenizer_bundle/`
- run metadata
- behavior eval
- sample outputs
- this current-state note as `MODEL_CARD.md`

The package does not contain raw training data, regenerated corpora, optimizer states, or redundant checkpoints. Temporary staging is removed by default. After upload, bind the release URL into `artifacts/Artifacts.toml` with:

```bash
ARTIFACT_URL=https://example.invalid/keemenalm-tiny-chatbot-v9-broad-336m.tar.gz \
UPDATE_ARTIFACTS_TOML=1 \
tools/package_tiny_chatbot_v9_release_artifact.sh
```

Fresh clones can then use `download_model("tiny-chatbot-v9-broad-336m")` and `resolve_tokenizer_bundle("tiny-chatbot-v9-broad-336m")`.

## Data Rebuild Script

The raw data should not be committed. A v9-style broad corpus can be rebuilt
from public dataset sources with:

```bash
tools/prepare_tiny_chatbot_v9_broad_corpus.sh
```

This uses the `wildchat-oasst1` preset in
`tools/prepare_tiny_chatbot_real_chat_corpus.py` plus downloaded broad pretrain
text and 5,000 synthetic direct-answer anchors. Some upstream datasets may be
gated or have license/use restrictions. Accept those terms outside this repo
before downloading.

The rebuild may not be byte-for-byte identical if upstream datasets change, but
it records the intended data recipe without committing the data itself.
