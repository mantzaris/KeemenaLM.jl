# Current Data And Artifact Policy

Date updated: 2026-07-04

## Current Local Model

The retained local v9 model is expected at:

```text
tmp/tiny_chatbot_v9_broad_336m_run
```

Required subdirectories for inference:

- `bundle/`
- `tokenizer_bundle/`

Useful metadata files:

- `metrics.json`
- `behavior_eval.json`
- `sample_outputs.txt`
- `run_recipe.json`
- `sft_data_mask_audit.txt`

## Julia Artifact Distribution

The intended fresh-clone distribution path is the Julia artifact key:

```julia
using KeemenaLM
bundle_dir = download_model("tiny-chatbot-v9-broad-336m")
tokenizer_dir = resolve_tokenizer_bundle("tiny-chatbot-v9-broad-336m")
```

The artifact root contains only what users need for inference and inspection:

- `bundle/`
- `tokenizer_bundle/`
- metadata files
- `MODEL_CARD.md`
- checksums

It does not contain raw training data, regenerated corpora, optimizer states, or
redundant checkpoints.

## Package The Current Model

Package the current local v9 run with:

```bash
tools/package_tiny_chatbot_v9_release_artifact.sh
```

The script writes:

- `tmp/release_artifacts/keemenalm-tiny-chatbot-v9-broad-336m.tar.gz`
- `tmp/release_artifacts/keemenalm-tiny-chatbot-v9-broad-336m.tar.gz.sha256`
- `tmp/release_artifacts/keemenalm-tiny-chatbot-v9-broad-336m.Artifacts.toml`

Temporary staging is removed by default to avoid leaving another full copy of
the model on disk. Set `KEEP_STAGING=1` only when you need to inspect the staged
layout.

After uploading the tarball, bind it into `artifacts/Artifacts.toml`:

```bash
ARTIFACT_URL=https://example.invalid/keemenalm-tiny-chatbot-v9-broad-336m.tar.gz \
UPDATE_ARTIFACTS_TOML=1 \
tools/package_tiny_chatbot_v9_release_artifact.sh
```

Use the real release URL. Fresh clones can then fetch the weights through Julia
artifacts instead of carrying weights in git.

## Use The Bound Artifact

Prompt probe:

```bash
julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_prompt_probe.jl \
  --model-key tiny-chatbot-v9-broad-336m \
  --device cpu
```

Behavior eval and the REPL accept the same `--model-key` option.

## Rebuild A v9-Style Corpus

Raw data and generated corpora are not committed. Rebuild the current broad
corpus recipe with:

```bash
tools/prepare_tiny_chatbot_v9_broad_corpus.sh
```

This wrapper uses `tools/prepare_tiny_chatbot_real_chat_corpus.py` with the
`wildchat-oasst1` preset, downloaded broad pretraining text, and 5,000 synthetic
direct-answer anchors.

Check upstream dataset licenses and terms outside this repository before
downloading. A rebuild may not be byte-for-byte identical if upstream datasets
change.

## What To Keep Locally

After a serious run, keep:

- final `bundle/`
- matching `tokenizer_bundle/`
- `metrics.json`
- `behavior_eval.json`
- `run_recipe.json`
- sample outputs and prompt files
- mask/data audit files

Delete redundant intermediate checkpoints and old run directories once the final
bundle loads and metadata is documented.
