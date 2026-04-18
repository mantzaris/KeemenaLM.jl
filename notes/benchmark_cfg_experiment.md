# BenchmarkDataNLP CFG experiment

This is the first bounded synthetic-language-model experiment for `KeemenaLM.jl`.

Purpose:
- validate the existing pipeline end to end
- not measure chatbot quality
- not establish a general evaluation suite

What it exercises:
- dataset generation with `BenchmarkDataNLP.generate_corpus_CFG(...)`
- train/validation/test split loading from the generated `.jsonl` files
- Flux training through `train_step!`
- resumable checkpoint save
- portable bundle export
- bundle reload and model instantiation
- deterministic sample generation
- simple loss / perplexity-style reporting

Default settings:
- generator: `CFG`
- complexity: `5`
- `num_sentences = 8_000`
- `enable_polysemy = false`
- train/test/validation: package-native `80/10/10`
- tokenizer: experiment-local char tokenizer built from the generated corpus
- model:
  - `context_length = 48`
  - `num_layers = 2`
  - `num_heads = 2`
  - `embedding_size = 64`
  - `ffn_hidden_size = 128`
- training:
  - backend `:flux`
  - device: CPU by default for the first reproducible local run
  - `batch_size = 16`
  - `epochs = 2`
  - `learning_rate = 0.01f0`

Run it:

```bash
julia --project=tools/benchmark_cfg -e 'using Pkg; Pkg.instantiate()'
julia --project=tools/benchmark_cfg tools/run_benchmark_cfg_experiment.jl
```

Optional output directory:

```bash
julia --project=tools/benchmark_cfg tools/run_benchmark_cfg_experiment.jl tmp/my_cfg_run
```

Dependency provenance:
- this experiment uses a dedicated Julia environment at `tools/benchmark_cfg/Project.toml`
- `BenchmarkDataNLP.jl` is sourced explicitly from:
  - `https://github.com/mantzaris/BenchmarkDataNLP.jl.git`
  - `rev = "main"`
- the local `KeemenaLM.jl` checkout is sourced explicitly via:
  - `KeemenaLM = { path = "../.." }`

Outputs written under the run directory:
- `dataset/`
- `checkpoints/`
- `bundle/`
- `tokenizer.json`
- `metrics.json`
- `sample_outputs.txt`

Important current limitation:
- tokenizer and preprocessing are still not persisted inside the bundle itself
- this experiment saves the tokenizer separately as an experiment artifact
