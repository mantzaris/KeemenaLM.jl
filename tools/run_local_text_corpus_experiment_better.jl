#!/usr/bin/env julia

include("run_local_text_corpus_experiment.jl")

const BETTER_LOCAL_TEXT_DEFAULT_OUTPUT_DIR = joinpath(pwd(), "tmp", "local_text_corpus_experiment_better")
const BETTER_LOCAL_TEXT_MARKDOWN_FILES = [
    "README.md",
    "docs/src/index.md",
    "notes/repo_plan_short.md",
    "notes/repo_plan_structure.md",
    "notes/todo_staged_roadmap.md",
    "notes/repo_plan_long.md",
    "notes/next_planned_experiments.md",
    "notes/official_models.md",
]
const BETTER_LOCAL_TEXT_DOCSTRING_FILES = [
    "src/core/types.jl",
    "src/core/configs/gpt2.jl",
    "src/core/model/masking.jl",
    "src/core/generation/sampling.jl",
    "src/core/generation/stopping.jl",
    "src/core/generation/generate.jl",
    "src/core/generation/chat.jl",
    "src/core/io/bundle_schema.jl",
    "src/core/io/model_sources.jl",
    "src/core/io/weights_jld2.jl",
    "src/core/io/bundle_save.jl",
    "src/core/io/bundle_load.jl",
    "src/core/training/loss.jl",
    "src/core/training/trainer.jl",
    "src/backends/flux/gpt2_flux.jl",
]

function main(args)
    length(args) <= 1 || error("usage: julia --project=tools/benchmark_cfg tools/run_local_text_corpus_experiment_better.jl [output_dir]")
    output_dir = length(args) == 1 ? abspath(args[1]) : BETTER_LOCAL_TEXT_DEFAULT_OUTPUT_DIR
    run_local_text_experiment(
        BETTER_LOCAL_TEXT_MARKDOWN_FILES,
        output_dir;
        experiment_name = "local_text_corpus_experiment_better",
        purpose = "better local real-text transfer sanity check, not chatbot benchmarking",
        corpus_source_label = "local_markdown_plus_docstrings",
        docstring_source_files = BETTER_LOCAL_TEXT_DOCSTRING_FILES,
    )
end

main(ARGS)
