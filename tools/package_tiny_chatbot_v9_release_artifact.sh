#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-tmp/tiny_chatbot_v9_broad_336m_run}"
OUTPUT_DIR="${OUTPUT_DIR:-tmp/release_artifacts}"
ARTIFACT_NAME="${ARTIFACT_NAME:-keemenalm-tiny-chatbot-v9-broad-336m}"
JULIA_ARTIFACT_NAME="${JULIA_ARTIFACT_NAME:-keemenalm_tiny_chatbot_v9_broad_336m}"
ARTIFACT_URL="${ARTIFACT_URL:-}"
UPDATE_ARTIFACTS_TOML="${UPDATE_ARTIFACTS_TOML:-0}"
KEEP_STAGING="${KEEP_STAGING:-0}"
ARTIFACTS_TOML="${ARTIFACTS_TOML:-artifacts/Artifacts.toml}"

required_paths=(
  "$RUN_DIR/bundle/bundle.json"
  "$RUN_DIR/bundle/model_config.json"
  "$RUN_DIR/bundle/weights.jld2"
  "$RUN_DIR/tokenizer_bundle/tokenizer.json"
  "$RUN_DIR/tokenizer_bundle/keemena_training_manifest.json"
  "$RUN_DIR/run_recipe.json"
  "$RUN_DIR/metrics.json"
  "$RUN_DIR/behavior_eval.json"
  "notes/tiny_chatbot_v9_broad_336m_current_state.md"
)

for path in "${required_paths[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required artifact input: $path" >&2
    exit 1
  fi
done

rm -rf "$OUTPUT_DIR/$ARTIFACT_NAME"
mkdir -p "$OUTPUT_DIR/$ARTIFACT_NAME"

cp -R "$RUN_DIR/bundle" "$OUTPUT_DIR/$ARTIFACT_NAME/bundle"
cp -R "$RUN_DIR/tokenizer_bundle" "$OUTPUT_DIR/$ARTIFACT_NAME/tokenizer_bundle"

mkdir -p "$OUTPUT_DIR/$ARTIFACT_NAME/metadata"
for file_name in \
  run_recipe.json \
  metrics.json \
  behavior_eval.json \
  sample_outputs.txt \
  evaluation_prompts.txt \
  evaluation_prompts.json \
  sft_data_mask_audit.txt \
  progress.json
do
  if [[ -f "$RUN_DIR/$file_name" ]]; then
    cp "$RUN_DIR/$file_name" "$OUTPUT_DIR/$ARTIFACT_NAME/metadata/$file_name"
  fi
done

cp notes/tiny_chatbot_v9_broad_336m_current_state.md "$OUTPUT_DIR/$ARTIFACT_NAME/MODEL_CARD.md"

cat > "$OUTPUT_DIR/$ARTIFACT_NAME/README.md" <<'INNER_EOF'
# KeemenaLM Tiny Chatbot v9 Broad 336M

This artifact contains the final v9 broad 336M model bundle, tokenizer bundle,
and run metadata for KeemenaLM.jl.

It is a research baseline, not a reliable chatbot.

## Layout

- `bundle/`: KeemenaLM inference bundle with `weights.jld2`
- `tokenizer_bundle/`: tokenizer used by the v8/v9 chat runner
- `metadata/`: metrics, behavior eval, run recipe, samples, and audits
- `MODEL_CARD.md`: current-state note and known failures

## Run

From the KeemenaLM.jl repository root, after extracting this artifact into a
directory:

```bash
CUDA_VISIBLE_DEVICES=0 julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_chat_repl.jl \
  --run-dir path/to/extracted-artifact \
  --bundle-dir path/to/extracted-artifact/bundle \
  --tokenizer-bundle-dir path/to/extracted-artifact/tokenizer_bundle \
  --device gpu \
  --temperature 0.0 \
  --top-k 0 \
  --top-p 1.0 \
  --max-new-tokens 120
```

Use `--device cpu` if no compatible GPU is available.
INNER_EOF

(
  cd "$OUTPUT_DIR/$ARTIFACT_NAME"
  find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

TAR_PATH="$OUTPUT_DIR/$ARTIFACT_NAME.tar"
TAR_GZ_PATH="$OUTPUT_DIR/$ARTIFACT_NAME.tar.gz"
SHA256_PATH="$TAR_GZ_PATH.sha256"
ARTIFACT_ENTRY_PATH="$OUTPUT_DIR/$ARTIFACT_NAME.Artifacts.toml"

rm -f "$TAR_PATH" "$TAR_GZ_PATH" "$SHA256_PATH" "$ARTIFACT_ENTRY_PATH"
tar -C "$OUTPUT_DIR/$ARTIFACT_NAME" -cf "$TAR_PATH" .
GIT_TREE_SHA1=$(julia --startup-file=no -e 'using Tar; print(Tar.tree_hash(ARGS[1]; algorithm = "git-sha1"))' "$TAR_PATH")
gzip -n -c "$TAR_PATH" > "$TAR_GZ_PATH"
rm -f "$TAR_PATH"
TARBALL_SHA256=$(sha256sum "$TAR_GZ_PATH" | awk '{print $1}')
printf '%s  %s\n' "$TARBALL_SHA256" "$TAR_GZ_PATH" > "$SHA256_PATH"

{
  printf '[%s]\n' "$JULIA_ARTIFACT_NAME"
  printf 'git-tree-sha1 = "%s"\n' "$GIT_TREE_SHA1"
  printf 'lazy = true\n'
  if [[ -n "$ARTIFACT_URL" ]]; then
    printf '\n[[%s.download]]\n' "$JULIA_ARTIFACT_NAME"
    printf 'url = "%s"\n' "$ARTIFACT_URL"
    printf 'sha256 = "%s"\n' "$TARBALL_SHA256"
  else
    printf '\n# Add a download block after uploading the tarball:\n'
    printf '# [[%s.download]]\n' "$JULIA_ARTIFACT_NAME"
    printf '# url = "https://example.invalid/%s.tar.gz"\n' "$ARTIFACT_NAME"
    printf '# sha256 = "%s"\n' "$TARBALL_SHA256"
  fi
} > "$ARTIFACT_ENTRY_PATH"

if [[ "$UPDATE_ARTIFACTS_TOML" == "1" ]]; then
  if [[ -z "$ARTIFACT_URL" ]]; then
    echo "UPDATE_ARTIFACTS_TOML=1 requires ARTIFACT_URL so fresh clones can download the artifact." >&2
    exit 1
  fi
  julia --project=. -e 'using Pkg; Pkg.Artifacts.bind_artifact!(ARGS[1], ARGS[2], Base.SHA1(ARGS[3]); download_info = [(ARGS[4], ARGS[5])], lazy = true, force = true)' \
    "$ARTIFACTS_TOML" "$JULIA_ARTIFACT_NAME" "$GIT_TREE_SHA1" "$ARTIFACT_URL" "$TARBALL_SHA256"
fi

if [[ "$KEEP_STAGING" != "1" ]]; then
  rm -rf "$OUTPUT_DIR/$ARTIFACT_NAME"
fi

echo "release artifact: $TAR_GZ_PATH"
echo "checksum: $SHA256_PATH"
echo "julia artifact entry: $ARTIFACT_ENTRY_PATH"
echo "julia artifact name: $JULIA_ARTIFACT_NAME"
echo "git-tree-sha1: $GIT_TREE_SHA1"
echo "tarball sha256: $TARBALL_SHA256"
if [[ "$KEEP_STAGING" == "1" ]]; then
  echo "staging directory: $OUTPUT_DIR/$ARTIFACT_NAME"
else
  echo "staging directory removed; set KEEP_STAGING=1 to keep it"
fi
if [[ "$UPDATE_ARTIFACTS_TOML" == "1" ]]; then
  echo "updated artifacts registry: $ARTIFACTS_TOML"
else
  echo "upload the tarball, then add $ARTIFACT_ENTRY_PATH to $ARTIFACTS_TOML or rerun with ARTIFACT_URL=... UPDATE_ARTIFACTS_TOML=1"
fi
