# Current Chatbot Baseline

The current chatbot baseline is the broad v9 336M scratch run documented in
`notes/tiny_chatbot_v9_broad_336m_current_state.md`.

## Result Summary

This run is publishable as a Julia proof-of-concept and partial/negative result:
training, checkpoint policy, tokenizer bundle export, model bundle export,
behavior evaluation, and REPL loading all worked. The trained model is still not
a usable assistant.

Key numbers:

- run directory: `tmp/tiny_chatbot_v9_broad_336m_run`
- final bundle: `tmp/tiny_chatbot_v9_broad_336m_run/bundle`
- tokenizer bundle: `tmp/tiny_chatbot_v9_broad_336m_run/tokenizer_bundle`
- estimated parameters: `336,488,448`
- pretraining characters: about `1.5B`
- SFT examples: `39,584`
- pretrain test loss: `2.0753`
- SFT test loss: `3.1421`
- behavior pass rate: `17/22`

Known failures include arithmetic, common-sense ability questions, edible vs
unsafe objects, private unknowns, factual grounding, and repetition loops.

## Artifact-Backed Usage

The `tiny-chatbot-v9-broad-336m` artifact is bound in `artifacts/Artifacts.toml`, so the tools can load weights and tokenizer sidecar by key:

```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_prompt_probe.jl \
  --model-key tiny-chatbot-v9-broad-336m \
  --device auto
```

Use the local `--run-dir` form below while developing against an unpublished run.

## Behavior Evaluation

Run the behavior gate against the retained local model:

```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_behavior_eval.jl \
  --run-dir tmp/tiny_chatbot_v9_broad_336m_run
```

## Prompt Probe

Use deterministic decoding for quick comparisons:

```bash
CUDA_VISIBLE_DEVICES=0 julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_prompt_probe.jl \
  --run-dir tmp/tiny_chatbot_v9_broad_336m_run \
  --bundle-dir tmp/tiny_chatbot_v9_broad_336m_run/bundle \
  --tokenizer-bundle-dir tmp/tiny_chatbot_v9_broad_336m_run/tokenizer_bundle \
  --device auto \
  --temperature 0.0 \
  --top-k 0 \
  --top-p 1.0 \
  --max-new-tokens 120
```

## REPL

Start the one-turn REPL:

```bash
CUDA_VISIBLE_DEVICES=0 julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_chat_repl.jl \
  --run-dir tmp/tiny_chatbot_v9_broad_336m_run \
  --bundle-dir tmp/tiny_chatbot_v9_broad_336m_run/bundle \
  --tokenizer-bundle-dir tmp/tiny_chatbot_v9_broad_336m_run/tokenizer_bundle \
  --device auto \
  --temperature 0.0 \
  --top-k 0 \
  --top-p 1.0 \
  --max-new-tokens 120
```

`--device auto` falls back to CPU if CUDA is unavailable. Type `/exit` or `/quit`
to leave the REPL.

## Training Direction

Do not treat the current result as evidence that a larger model alone is the
next fix. The recommended next step is:

1. Build a cleaner v10 data pass with much broader failure coverage.
2. Expand the behavior suite before training.
3. Run a tiny smoke experiment and inspect decoded masks and outputs.
4. Try a short continuation from the v9 bundle only after the data/eval pass.
5. Consider a larger model only if smaller verification runs improve the known
   failures.
