# Prepared Better Local Real-Text Budget Sweep Stage 4

Purpose: extend the current best char-level real-text recipe beyond the 14-epoch baseline to test whether training still improves materially or is finally nearing saturation.

This sweep keeps fixed:
- corpus: `tmp/better_local_real_text_corpus_prepared/dataset`
- backend: `:flux`
- tokenizer path: char-level experiment-local tokenizer
- `context_length = 48`
- `num_layers = 2`
- `num_heads = 2`
- `embedding_size = 128`
- `ffn_hidden_size = 256`
- `batch_size = 16`
- `learning_rate = 0.01f0`
- same deterministic seed style as earlier prepared-corpus runs

This sweep varies only:
- `epochs = 14`
- `epochs = 16`
- `epochs = 18`

Run with:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_budget_sweep_stage4.jl
```

Optional custom output directory:

```bash
julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_budget_sweep_stage4.jl tmp/better_local_real_text_corpus_prepared/dataset tmp/prepared_better_local_real_text_budget_sweep_stage4_custom
```

Outputs:
- per-run directories under `tmp/prepared_better_local_real_text_budget_sweep_stage4/`
- `summary.json`
- `summary.md`

Interpretation target:
- if validation and test keep improving, the recipe is still under-trained
- if gains flatten, the setup may be nearing saturation
- if validation worsens while train improves, overfitting may be starting
