# KeemenaLM.jl

[![CI](https://github.com/mantzaris/KeemenaLM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mantzaris/KeemenaLM.jl/actions/workflows/CI.yml)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://mantzaris.github.io/KeemenaLM.jl/dev/)

KeemenaLM.jl is a Julia proof-of-concept for building, training, exporting, and
running small GPT-style decoder language models from scratch. The current
research focus is a tiny scratch-trained chatbot pipeline: corpus preparation,
tokenizer training, Flux training, checkpoint/bundle export, behavior
evaluation, and an interactive REPL.

This repository does not ship a production chatbot. The strongest current model
is a partial 336M parameter baseline that proves the Julia pipeline works and
documents clear data and evaluation failures.

## Current Baseline

The current scratch-chatbot baseline is the broad v9 336M run:

- model size: `336,488,448` estimated parameters
- architecture: `24` layers, `16` heads, `1024` embedding, `4096` FFN
- context length: `512`
- tokenizer vocabulary: `32,768`
- pretraining data: about `1.5B` characters
- SFT examples: `39,584`
- final test losses: `pretrain=2.0753`, `sft=3.1421`
- behavior gate: `17/22`
- retained local run: `tmp/tiny_chatbot_v9_broad_336m_run`

The model is better than the older overfit repair attempts, but it is still not
a usable assistant. It fails common sense, arithmetic, factual grounding,
unknown-private-fact handling, safe object choice, and repetition checks.

Read the full model note before treating the result as successful:

- `notes/tiny_chatbot_v9_broad_336m_current_state.md`
- `notes/current_data_and_artifacts.md`
- `notes/NextTODO.md`

## What Works

Current supported package and research workflow:

- Flux model instantiate, forward pass, generation, and training
- NVIDIA/CUDA training through the Flux path
- Lux instantiate, forward pass, shared weights, and CPU generation
- portable model bundles with JLD2 weights
- resumable checkpoints
- official local demo artifact registration
- tokenizer bundle sidecars for chatbot runs
- behavior scoring for simple chatbot regression checks
- one-turn chatbot REPL from a saved bundle

Known limitations:

- Lux training parity is not implemented.
- Tokenizer/preprocessing payloads are not embedded inside `Bundle`; chatbot
  runs keep a separate tokenizer bundle directory.
- Official model download is local-artifact based, not remote-hosted.
- The v9 chatbot baseline is not reliable enough for real users.

## Repository Layout

- `src/`: package code for model config, bundle IO, generation, training, and
  Flux/Lux backends
- `tools/run_tiny_chatbot_v8_scratch.jl`: current scratch-chatbot trainer used
  for v8/v9-style runs
- `tools/run_tiny_chatbot_v8_behavior_eval.jl`: behavior-gate evaluator
- `tools/run_tiny_chatbot_v8_chat_repl.jl`: one-turn interactive chat REPL
- `tools/run_tiny_chatbot_v8_prompt_probe.jl`: fixed prompt probe for quick
  qualitative checks
- `tools/prepare_tiny_chatbot_real_chat_corpus.py`: current real-chat corpus
  importer
- `tools/prepare_tiny_chatbot_v9_broad_corpus.sh`: v9-style corpus rebuild
  wrapper
- `tools/package_tiny_chatbot_v9_release_artifact.sh`: release artifact packager
- `examples/`: small package demos
- `notes/`: current handoff plus archived experiment notes
- `docs/`: Documenter.jl source

## Two Ways To Use It

### 1. Clone The Repo And Run The Chatbot Tools

This is the best path for trying the v9 chatbot, running the REPL, rebuilding
data, or experimenting with training.

```bash
git clone https://github.com/mantzaris/KeemenaLM.jl
cd KeemenaLM.jl

julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=tools/subword_real_text -e 'using Pkg; Pkg.instantiate()'
```

After the v9 model artifact is published and bound in `artifacts/Artifacts.toml`,
run:

```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_chat_repl.jl \
  --model-key tiny-chatbot-v9-broad-336m \
  --device cpu \
  --temperature 0.0 \
  --top-k 0 \
  --top-p 1.0 \
  --max-new-tokens 120
```

Use `--device gpu` with `CUDA_VISIBLE_DEVICES=0` on a compatible NVIDIA setup.

### 2. Add The Package From GitHub In Julia

This is the package/API path. It is useful for loading bundles, using the model
API, and resolving published artifacts from Julia code.

```julia
using Pkg
Pkg.add(url = "https://github.com/mantzaris/KeemenaLM.jl")

using KeemenaLM
bundle_dir = download_model("tiny-chatbot-v9-broad-336m")
tokenizer_dir = resolve_tokenizer_bundle("tiny-chatbot-v9-broad-336m")
```

The tokenizer sidecar is needed for the v9 chatbot tools. For the full REPL and
prompt-probe workflow, cloning the repository is still the most direct path
because those scripts live under `tools/`.

## Get The Model Artifact

Weights are not committed to git. The streamlined distribution path is a Julia artifact named `tiny-chatbot-v9-broad-336m`. Once the release tarball URL is bound in `artifacts/Artifacts.toml`, users can fetch the model bundle from Julia:

```julia
using KeemenaLM
bundle_dir = download_model("tiny-chatbot-v9-broad-336m")
tokenizer_dir = resolve_tokenizer_bundle("tiny-chatbot-v9-broad-336m")
```

The chatbot tools can then load both the weights and tokenizer sidecar by key:

```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_prompt_probe.jl \
  --model-key tiny-chatbot-v9-broad-336m \
  --device cpu
```

Until a hosted release URL is bound, use the retained local run under `tmp/tiny_chatbot_v9_broad_336m_run`.

## Test The Current Local Model

The current local model is expected at:

```text
tmp/tiny_chatbot_v9_broad_336m_run
```

Run the behavior gate:

```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_behavior_eval.jl \
  --run-dir tmp/tiny_chatbot_v9_broad_336m_run
```

Run a deterministic prompt probe:

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

Run the one-turn REPL:

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

Use `--device cpu` if no compatible GPU is available. Type `/exit` or `/quit`
to leave the REPL.

## Rebuild Data

Raw downloaded datasets and generated corpora are not committed. Rebuild a
v9-style broad corpus with:

```bash
tools/prepare_tiny_chatbot_v9_broad_corpus.sh
```

The wrapper calls `tools/prepare_tiny_chatbot_real_chat_corpus.py` with the
`wildchat-oasst1` preset, downloaded broad pretraining text, and 5,000 synthetic
direct-answer anchors. Check and accept upstream dataset licenses or terms
outside this repository before downloading.

## Train A New Candidate

After preparing a compatible corpus:

```bash
CUDA_VISIBLE_DEVICES=0 julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_scratch.jl \
  --dataset-dir tmp/tiny_chatbot_v9_broad_corpus_5k_anchor \
  --output-dir tmp/tiny_chatbot_v9_broad_336m_run_next \
  --device gpu \
  --pretrain-epochs 1 \
  --behavior-max-updates 35000 \
  --behavior-min-updates 2000 \
  --validation-every-updates 1000 \
  --direct-sft-parts 3 \
  --pretrain-replay-parts 2 \
  --save-best-behavior-bundle \
  --no-base-checkpoint
```

Do not launch another multi-day run blindly. The documented next step is a v10
data and evaluation pass, followed by a short continuation or smoke run before
any larger model experiment.

## Package Weights For Release

Do not commit model weights, checkpoints, tokenizer bundles, or generated
corpora to git. To create a tarball for a GitHub Release or another large-file
host:

```bash
tools/package_tiny_chatbot_v9_release_artifact.sh
```

The artifact contains the final model bundle, tokenizer bundle, run metadata, behavior eval, sample outputs, and model card. It intentionally excludes raw training data and removes temporary staging by default. Set `ARTIFACT_URL=... UPDATE_ARTIFACTS_TOML=1` after uploading the tarball to bind it for fresh-clone downloads.

## Package Demos

The small package demos still exercise the generic bundle and generation API:

```bash
julia --project=. examples/train_tiny_gpt2_flux.jl
julia --project=. tools/build_public_model_artifact.jl
julia --project=. examples/chat_demo.jl tiny-demo
julia --project=. examples/chat_repl.jl tiny-demo
```

These demos are not the v9 chatbot baseline; they are compact package smoke
workflows.

## Contributing

Research contributions are welcome when they are reproducible and honest about
model quality. Useful areas are data curation, license-aware dataset manifests,
behavior evaluation, training reports, and small infrastructure fixes.

Start with:

- `CONTRIBUTING.md`
- `notes/README.md`
- `notes/tiny_chatbot_v9_broad_336m_current_state.md`
- `notes/current_data_and_artifacts.md`
- `notes/NextTODO.md`

Run the package tests before submitting code changes:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
