#!/usr/bin/env julia

include("run_local_text_corpus_experiment.jl")

const LARGE_LOCAL_TEXT_DEFAULT_OUTPUT_DIR = joinpath(pwd(), "tmp", "local_text_corpus_experiment_large")
const LARGE_LOCAL_TEXT_CORPUS_FILES = [
    "README.md",
    "docs/src/index.md",
    "notes/repo_plan_short.md",
    "notes/repo_plan_structure.md",
    "notes/todo_staged_roadmap.md",
    "notes/repo_plan_long.md",
    "notes/next_planned_experiments.md",
    "notes/official_models.md",
]

function main(args)
    length(args) <= 1 || error("usage: julia --project=tools/benchmark_cfg tools/run_local_text_corpus_experiment_large.jl [output_dir]")
    output_dir = length(args) == 1 ? abspath(args[1]) : LARGE_LOCAL_TEXT_DEFAULT_OUTPUT_DIR
    run_local_text_experiment(
        LARGE_LOCAL_TEXT_CORPUS_FILES,
        output_dir;
        experiment_name = "local_text_corpus_experiment_large",
        purpose = "larger local real-text transfer sanity check, not chatbot benchmarking",
        corpus_source_label = "local_markdown_repo_docs_extended",
    )
end

main(ARGS)
