#!/usr/bin/env julia

using JSON3
using Printf

include("run_prepared_better_local_real_text_experiment.jl")

const BUDGET_SWEEP_OUTPUT_DIR = joinpath(pwd(), "tmp", "prepared_better_local_real_text_budget_sweep")
const BUDGET_SWEEP_SUMMARY_JSON = "summary.json"
const BUDGET_SWEEP_SUMMARY_MD = "summary.md"

const BUDGET_SWEEP_RUNS = [
    (epochs = 2,),
    (epochs = 4,),
    (epochs = 6,),
]

function main(args)
    length(args) <= 2 || error("usage: julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_budget_sweep.jl [prepared_dataset_dir] [output_dir]")
    dataset_dir = length(args) >= 1 ? abspath(args[1]) : PREPARED_CORPUS_INPUT_DIR
    output_dir = length(args) == 2 ? abspath(args[2]) : BUDGET_SWEEP_OUTPUT_DIR
    run_prepared_better_local_real_text_budget_sweep(dataset_dir, output_dir)
end

function run_prepared_better_local_real_text_budget_sweep(dataset_dir::AbstractString, output_dir::AbstractString)
    mkpath(output_dir)

    baseline_metrics_path = joinpath(
        pwd(),
        "tmp",
        "prepared_better_local_real_text_width_sweep",
        "embed_128_ffn_256",
        "metrics.json",
    )
    isfile(baseline_metrics_path) || throw(ArgumentError("128/256 width baseline metrics not found: $(baseline_metrics_path)"))
    baseline_metrics = JSON3.read(read(baseline_metrics_path, String))

    run_summaries = Dict{String, Any}[]
    for run in BUDGET_SWEEP_RUNS
        run_name = @sprintf("epochs_%02d", run.epochs)
        run_output_dir = joinpath(output_dir, run_name)
        settings = merge_settings(
            ExperimentSettings();
            complexity = 0,
            num_sentences = 0,
            prompt_prefix_characters = 24,
            embedding_size = 128,
            ffn_hidden_size = 256,
            epochs = run.epochs,
        )

        metrics = run_prepared_better_local_real_text_experiment(
            dataset_dir,
            run_output_dir;
            settings = settings,
            experiment_name = "prepared_better_local_real_text_budget_" * run_name,
            purpose = "prepared better local real-text budget sweep at 128/256 width, not chatbot benchmarking",
        )

        push!(
            run_summaries,
            Dict(
                "run_name" => run_name,
                "output_dir" => run_output_dir,
                "epochs" => run.epochs,
                "embedding_size" => 128,
                "ffn_hidden_size" => 256,
                "train_loss" => metrics["epoch_metrics"][end]["train_loss"],
                "validation_loss" => metrics["epoch_metrics"][end]["validation_loss"],
                "test_loss" => metrics["training"]["test_loss"],
                "test_perplexity" => metrics["training"]["test_perplexity"],
                "train_example_count" => metrics["training"]["train_example_count"],
                "train_batches" => metrics["training"]["train_batches"],
                "final_step" => metrics["training"]["final_step"],
                "sample_outputs_path" => metrics["artifacts"]["sample_outputs_path"],
                "metrics_path" => joinpath(run_output_dir, "metrics.json"),
            ),
        )
    end

    summary = Dict(
        "experiment" => "prepared_better_local_real_text_budget_sweep",
        "purpose" => "controlled training-budget sweep on the prepared better local real-text corpus at 128/256 width",
        "dataset_dir" => dataset_dir,
        "fixed_settings" => Dict(
            "backend" => "flux",
            "tokenizer" => "char-level experiment-local tokenizer",
            "context_length" => 48,
            "num_layers" => 2,
            "num_heads" => 2,
            "embedding_size" => 128,
            "ffn_hidden_size" => 256,
            "batch_size" => 16,
            "learning_rate" => 0.01,
            "seed_style" => "same deterministic seeds as the prepared-corpus baseline",
        ),
        "sweep_axis" => "epochs",
        "runs" => run_summaries,
        "baseline" => Dict(
            "metrics_path" => baseline_metrics_path,
            "epochs" => baseline_metrics["training"]["epochs"],
            "embedding_size" => baseline_metrics["model"]["embedding_size"],
            "ffn_hidden_size" => baseline_metrics["model"]["ffn_hidden_size"],
            "train_loss" => baseline_metrics["epoch_metrics"][end]["train_loss"],
            "validation_loss" => baseline_metrics["epoch_metrics"][end]["validation_loss"],
            "test_loss" => baseline_metrics["training"]["test_loss"],
            "test_perplexity" => baseline_metrics["training"]["test_perplexity"],
            "train_example_count" => baseline_metrics["training"]["train_example_count"],
            "train_batches" => baseline_metrics["training"]["train_batches"],
            "final_step" => baseline_metrics["training"]["final_step"],
            "sample_outputs_path" => baseline_metrics["artifacts"]["sample_outputs_path"],
        ),
    )

    open(joinpath(output_dir, BUDGET_SWEEP_SUMMARY_JSON), "w") do io
        JSON3.write(io, summary)
    end
    write_budget_sweep_summary_md(joinpath(output_dir, BUDGET_SWEEP_SUMMARY_MD), summary)

    println("budget sweep summary: ", joinpath(output_dir, BUDGET_SWEEP_SUMMARY_JSON))
    println("budget sweep markdown: ", joinpath(output_dir, BUDGET_SWEEP_SUMMARY_MD))
    return summary
end

function write_budget_sweep_summary_md(path::AbstractString, summary)
    baseline = summary["baseline"]
    runs = summary["runs"]
    open(path, "w") do io
        println(io, "# Prepared Better Local Real-Text Budget Sweep")
        println(io)
        println(io, "Baseline: epochs=$(baseline["epochs"]), embed=$(baseline["embedding_size"]), ffn=$(baseline["ffn_hidden_size"]), test_loss=$(round(baseline["test_loss"], digits=4))")
        println(io)
        println(io, "| run | epochs | train_loss | val_loss | test_loss | train_examples | train_batches | final_step |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for run in runs
            println(
                io,
                "| $(run["run_name"]) | $(run["epochs"]) | $(round(run["train_loss"], digits=4)) | " *
                "$(round(run["validation_loss"], digits=4)) | $(round(run["test_loss"], digits=4)) | " *
                "$(run["train_example_count"]) | $(run["train_batches"]) | $(run["final_step"]) |",
            )
        end
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
