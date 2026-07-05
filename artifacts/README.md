# Model Artifacts

`Artifacts.toml` is the Julia artifact registry used by `KeemenaLM.available_models`,
`KeemenaLM.download_model`, and related helpers.

Current artifact keys:

- `tiny-demo`: tiny local toy bundle for API smoke tests
- `tiny-chatbot-v9-broad-336m`: current v9 chatbot research baseline,
  distributed as a lazy GitHub Release artifact

The v9 weights are not committed to git. The current artifact is hosted at:

```text
https://github.com/mantzaris/KeemenaLM.jl/releases/download/v0.1.0/keemenalm-tiny-chatbot-v9-broad-336m.tar.gz
```

Fresh clones can fetch the model through Julia artifacts:

```julia
using KeemenaLM
bundle_dir = download_model("tiny-chatbot-v9-broad-336m")
tokenizer_dir = resolve_tokenizer_bundle("tiny-chatbot-v9-broad-336m")
```

To publish a future replacement:

1. Run `tools/package_tiny_chatbot_v9_release_artifact.sh`.
2. Upload `tmp/release_artifacts/keemenalm-tiny-chatbot-v9-broad-336m.tar.gz`
   to a GitHub Release or another large-file host.
3. Add the generated `*.Artifacts.toml` entry to `artifacts/Artifacts.toml`, or
   rerun the packager with `ARTIFACT_URL=... UPDATE_ARTIFACTS_TOML=1`.
