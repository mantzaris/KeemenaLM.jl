# Tiny Chatbot v7 Scratch Postmortem

Date recorded: 2026-06-10

## Purpose

V7 tested whether a larger scratch model could avoid the v6 synthetic-SFT
collapse by using:

- more general pretraining data
- much less synthetic SFT
- a mixed replay final stage instead of a long SFT-only stage
- early stopping before SFT loss collapsed to near-zero

The goal was still a pure-from-scratch Julia chatbot, not a pretrained-model
import.

## Training Recipe

- Corpus directory: `tmp/tiny_chatbot_v7_scratch_corpus`
- Run directory: `tmp/tiny_chatbot_v7_scratch_candidate_run`
- Device: GPU, RTX 5000 Ada by UUID
- Context length: `512`
- Model:
  - layers: `24`
  - heads: `16`
  - embedding size: `1024`
  - FFN hidden size: `4096`
  - actual vocab size: `30630`
  - estimated params: `334299136`
- Optimizer: `Flux.Adam(0.00006)`
- Batch size: `1`
- Gradient accumulation: `4`
- Effective batch size: `4`

Corpus:

- pretrain characters: `1499999783`
- pretrain training docs: `3756878`
- pretrain validation docs: `18850`
- pretrain testing docs: `18569`
- SFT source groups:
  - synthetic clean assistant examples: `12000`
  - filtered UltraChat examples: `854`
- SFT split:
  - training: `12743`
  - validation: `60`
  - testing: `51`

Training stages:

- `streamed_pretrain_all_tokens`
- `mixed_replay`

Mixed replay ratio:

- `70%` pretrain replay
- `20%` chat LM
- `10%` assistant-only SFT

Mixed replay was capped at `35000` updates, with validation every `2000`
updates. It stopped at the first validation because the SFT target was already
reached without the configured replay-regression limits firing.

## Results

Pretrain stage:

- optimizer updates: `172056`
- microbatches: `688221`
- train loss: `3.0249129467007685`
- validation loss: `3.393557557542607`
- validation perplexity: `29.771678554789357`

Mixed replay stage:

- mixed updates: `2000`
- final global step: `174056`
- train loss: `1.5966031041931128`
- validation losses:
  - pretrain: `2.1754043473357223`
  - chat LM: `0.6600972718141673`
  - SFT: `0.5042859352290912`
- stop reason: `sft_target_reached_without_replay_regression`

Final test losses:

- pretrain: `2.1555328555713302`
- chat LM: `0.4798316456489021`
- SFT: `0.4176116555735922`

Important caveat:

These losses looked much healthier than v6, but they did not predict acceptable
chatbot behavior. Low chat/SFT losses still mostly measured matching the narrow
training distribution, not general assistant quality.

## Qualitative Result

V7 was not chatbot-quality.

Saved samples were more coherent than v6, but still template-heavy. Common
phrases included:

- `Tell me the rough situation and what outcome would help most.`
- `Keep the plan smaller than your first instinct.`
- `Start with the rough version.`

Manual REPL tests failed basic chatbot checks:

- `hello` caused the model to emit a synthetic training-record-like prompt and
  then answer it.
- `my name is alex, and I live in the USA. Is the USA a democracy or a dictatorship?`
  caused the model to emit an unrelated synthetic anxiety/support example.
- `what is 1 plus 1?` collapsed into a repeated `q, q, q ...` style loop.
- `is green a color?` produced a garbled partial answer.

Conclusion:

V7 was a real improvement over v6 as a training experiment, but it still failed
the product goal. It learned surface chat style and some fluent fragments, not a
reliable assistant.

## Cleanup State

After inspection, generated artifacts were deleted to free disk space:

- `tmp/tiny_chatbot_v7_scratch_corpus`
- `tmp/tiny_chatbot_v7_scratch_candidate_run`

The older generated v4/v5/v6 corpora and run directories were also deleted.
`tmp/` was left essentially empty. Only source code and notes remain.

## Diagnosis

The main failure was not lack of model capacity. A `334M` parameter model should
be able to learn basic answers such as `1 + 1 = 2` and `green is a color` if
the objective strongly teaches that behavior.

The likely failure causes are:

- Synthetic assistant examples still dominated the final assistant style even
  after reducing them from v6.
- The synthetic templates used repeated closers and generic advice phrases; the
  model learned those as high-probability assistant behavior.
- The chat LM objective trained raw transcript continuation, so emitting
  `User:` and `Assistant:` records remained a valid continuation pattern.
- The final validation losses measured distribution fit, not whether the model
  answers simple interactive prompts.
- The training/eval loop did not include behavioral guardrails such as
  arithmetic, factual yes/no, color questions, user-name recall, or fake-role
  continuation detection.

## Recommended v8 Direction

Do not train v8 by simply increasing model size or running v7 longer.

The next useful experiment should change the final objective and evaluation
more aggressively:

1. Remove synthetic template closers entirely.
   Do not include repeated slogans like `Keep the plan smaller than your first
   instinct`.
2. Do not use raw chat transcript continuation as a major final-stage objective.
   Avoid teaching the model that producing a fresh `User:` turn is normal.
3. Build a direct-answer assistant curriculum with assistant-only targets.
   Include many short examples for:
   - greetings
   - arithmetic
   - colors
   - yes/no facts
   - country/capital questions
   - common world facts
   - user-provided facts and immediate recall
   - short rewrites
   - simple planning
4. Keep general pretrain replay, but make assistant behavior data more varied
   and direct.
5. Add a behavioral eval gate during training.
   Stop or reject runs that:
   - output `User:` or `Assistant:` inside the assistant response
   - loop on a token or phrase
   - fail `1 + 1`
   - fail `is green a color?`
   - fail a simple country/capital question
   - ignore a user-provided name/fact
6. Prefer a smaller, higher-quality final instruction set over broad noisy SFT.

The v8 hypothesis should be:

```text
The scratch base model can learn language.
The chatbot failure is caused by the final assistant objective.
Fix the final objective with direct-answer assistant-only data and behavioral
eval gates before changing model size again.
```
