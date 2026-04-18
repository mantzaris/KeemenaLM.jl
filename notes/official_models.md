# Official Models

## `tiny-demo`

- Purpose: first Stage 6 official GPT-2 demo bundle distributed through Julia Artifacts.
- Stage 6 install semantics: local artifact registration only in this repo setup, not a fresh-user remote download path.
- Build/register locally: `julia --project=. tools/build_public_model_artifact.jl`
- Resolve/materialize locally: `available_models()`, `download_model("tiny-demo")`, `load_bundle("tiny-demo")`
- Local path precedence: if a local bundle directory exists with the same name as an official model key, local directory resolution wins.
- Important limitation: tokenizer and preprocessing payloads are still not persisted in bundles, so callers must still supply the matching tokenizer/preprocessing convention explicitly.
