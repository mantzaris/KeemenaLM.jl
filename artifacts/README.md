# Model Artifacts

`Artifacts.toml` is the Julia artifact registry used by `KeemenaLM.available_models`,
`KeemenaLM.download_model`, and related helpers.

Current artifact keys:

- `tiny-demo`: tiny local toy bundle for API smoke tests
- `tiny-chatbot-v9-broad-336m`: current v9 chatbot research baseline, expected
  to be distributed as a GitHub Release artifact

The v9 weights are not committed to git. To publish them:

1. Run `tools/package_tiny_chatbot_v9_release_artifact.sh`.
2. Upload `tmp/release_artifacts/keemenalm-tiny-chatbot-v9-broad-336m.tar.gz`
   to a GitHub Release or another large-file host.
3. Add the generated `*.Artifacts.toml` entry to `artifacts/Artifacts.toml`, or
   rerun the packager with `ARTIFACT_URL=... UPDATE_ARTIFACTS_TOML=1`.

After that, fresh clones can fetch the model through Julia artifacts:

```julia
using KeemenaLM
bundle_dir = download_model("tiny-chatbot-v9-broad-336m")
tokenizer_dir = resolve_tokenizer_bundle("tiny-chatbot-v9-broad-336m")
```
