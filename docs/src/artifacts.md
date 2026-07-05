# Data And Artifacts

Generated data and model artifacts are intentionally kept out of git. The
repository keeps scripts, metadata, notes, and tests; local corpora and weights
live under ignored directories such as `tmp/`.

## Julia Artifact Model Distribution

The streamlined model distribution path is the Julia artifact key:

```julia
using KeemenaLM
bundle_dir = download_model("tiny-chatbot-v9-broad-336m")
tokenizer_dir = resolve_tokenizer_bundle("tiny-chatbot-v9-broad-336m")
```

`download_model` returns the KeemenaLM bundle directory containing the weights.
`resolve_tokenizer_bundle` returns the matching tokenizer sidecar needed by the
chatbot tools.

This works for fresh clones because the v9 release tarball URL and checksum are
bound in `artifacts/Artifacts.toml`.

## Package A Release Artifact

Do not commit weights to git. Package the retained local v9 run with:

```bash
tools/package_tiny_chatbot_v9_release_artifact.sh
```

The packager creates a Julia-artifact-friendly tarball containing:

- `bundle/`
- `tokenizer_bundle/`
- run metadata
- behavior eval
- sample outputs
- `MODEL_CARD.md`

It excludes raw training data and regenerated corpora. Temporary staging is
removed by default so the packager does not leave another full copy of the model
on disk.

After uploading a future tarball, bind it for fresh-clone downloads:

```bash
ARTIFACT_URL=https://github.com/mantzaris/KeemenaLM.jl/releases/download/vNEXT/keemenalm-tiny-chatbot-v9-broad-336m.tar.gz \
UPDATE_ARTIFACTS_TOML=1 \
tools/package_tiny_chatbot_v9_release_artifact.sh
```

The script also writes a standalone `*.Artifacts.toml` snippet next to the
tarball.

## Use The Artifact In Tools

Current chatbot tools can use the model key directly:

```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_prompt_probe.jl \
  --model-key tiny-chatbot-v9-broad-336m \
  --device auto
```

The same `--model-key` option works for the behavior evaluator and chat REPL.

## Rebuild A v9-Style Corpus

Use the wrapper:

```bash
tools/prepare_tiny_chatbot_v9_broad_corpus.sh
```

It calls `tools/prepare_tiny_chatbot_real_chat_corpus.py` with the
`wildchat-oasst1` preset, downloaded broad pretraining text, and 5,000 synthetic
direct-answer anchors. Upstream datasets may require accepting licenses or use
terms outside this repository.

The rebuild is a recipe, not a guaranteed byte-for-byte reproduction. Upstream
datasets can change.

## Train A Candidate

After preparing a compatible corpus, the current trainer is:

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

The v8 filename is historical; this runner is the current v8/v9-style trainer.

## Official Demo Artifact

The package still includes a tiny local official-model flow for API testing:

```bash
julia --project=. tools/build_public_model_artifact.jl
```

That registers a small local demo artifact such as `tiny-demo`. It is separate
from the v9 chatbot baseline and cannot be used to bind v9 weights.
