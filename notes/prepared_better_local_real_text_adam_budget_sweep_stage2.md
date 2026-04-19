# Prepared Better Local Real-Text Adam Budget Sweep Stage 2

Purpose: test whether the current best Adam recipe is still under-trained beyond `30` epochs or whether it is beginning to approach saturation closely enough to justify a final demo-training run.

This sweep keeps fixed:
- corpus: `tmp/better_local_real_text_corpus_prepared/dataset`
- backend: `:flux`
- optimizer family: `Flux.Adam`
- optimizer hyperparameters: `learning_rate = 0.001`
- tokenizer path: char-level experiment-local tokenizer
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `batch_size = 16`
- same deterministic seed style as earlier prepared-corpus runs

This sweep varies only:
- `epochs = 30`
- `epochs = 34`
- `epochs = 38`

Run with:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_adam_budget_sweep_stage2.jl
```

Optional custom output directory:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_adam_budget_sweep_stage2.jl tmp/better_local_real_text_corpus_prepared/dataset tmp/prepared_better_local_real_text_adam_budget_sweep_stage2_custom
```

Outputs:
- per-run directories under `tmp/prepared_better_local_real_text_adam_budget_sweep_stage2/`
- `summary.json`
- `summary.md`

Interpretation target:
- if validation and test keep improving, Adam still has useful budget headroom beyond `30` epochs
- if gains flatten enough, the next step can reasonably be a final longer demo-training run
