#!/usr/bin/env julia

using Flux
using JSON3
using Printf

include("run_prepared_better_local_real_text_experiment.jl")

const OPTIMIZER_SWEEP_OUTPUT_DIR = joinpath(pwd(), "tmp", "prepared_better_local_real_text_optimizer_sweep")
const OPTIMIZER_SWEEP_SUMMARY_JSON = "summary.json"
const OPTIMIZER_SWEEP_SUMMARY_MD = "summary.md"

const OPTIMIZER_SWEEP_RUNS = [
    (
        label = "descent_lr_0p01",
        optimizer_name = "Flux.Descent",
        learning_rate = 0.01f0,
        optimizer_builder = settings -> Flux.Descent(settings.learning_rate),
        optimizer_hparams = Dict("learning_rate" => 0.01f0),
    ),
    (
        label = "adam_lr_0p001",
        optimizer_name = "Flux.Adam",
        learning_rate = 0.001f0,
        optimizer_builder = settings -> Flux.Adam(settings.learning_rate),
        optimizer_hparams = Dict("learning_rate" => 0.001f0),
    ),
]

function main(args)
    length(args) <= 2 || error("usage: julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_optimizer_sweep.jl [prepared_dataset_dir] [output_dir]")
    dataset_dir = length(args) >= 1 ? abspath(args[1]) : PREPARED_CORPUS_INPUT_DIR
    output_dir = length(args) == 2 ? abspath(args[2]) : OPTIMIZER_SWEEP_OUTPUT_DIR
    run_prepared_better_local_real_text_optimizer_sweep(dataset_dir, output_dir)
end

function run_prepared_better_local_real_text_optimizer_sweep(dataset_dir::AbstractString, output_dir::AbstractString)
    mkpath(output_dir)

    baseline_metrics_path = joinpath(
        pwd(),
        "tmp",
        "prepared_better_local_real_text_budget_sweep_stage5",
        "epochs_22",
        "metrics.json",
    )
    isfile(baseline_metrics_path) || throw(ArgumentError("22-epoch baseline metrics not found: $(baseline_metrics_path)"))
    baseline_metrics = JSON3.read(read(baseline_metrics_path, String))

    run_summaries = Dict{String, Any}[]
    for run in OPTIMIZER_SWEEP_RUNS
        run_output_dir = joinpath(output_dir, run.label)
        run_metrics_path = joinpath(run_output_dir, "metrics.json")
        settings = merge_settings(
            ExperimentSettings();
            complexity = 0,
            num_sentences = 0,
            prompt_prefix_characters = 24,
            context_length = 48,
            embedding_size = 128,
            ffn_hidden_size = 256,
            epochs = 22,
            learning_rate = run.learning_rate,
        )

        metrics = if isfile(run_metrics_path)
            JSON3.read(read(run_metrics_path, String))
        else
            run_prepared_better_local_real_text_experiment(
                dataset_dir,
                run_output_dir;
                settings = settings,
                experiment_name = "prepared_better_local_real_text_optimizer_sweep_" * run.label,
                purpose = "first optimizer-family comparison on the prepared better local real-text corpus at the current best recipe, not chatbot benchmarking",
                optimizer_builder = run.optimizer_builder,
                optimizer_name = run.optimizer_name,
                optimizer_hparams = run.optimizer_hparams,
            )
        end

        push!(
            run_summaries,
            Dict(
                "run_name" => run.label,
                "output_dir" => run_output_dir,
                "optimizer_name" => run.optimizer_name,
                "optimizer_hyperparameters" => Dict(String(key) => value for (key, value) in pairs(run.optimizer_hparams)),
                "learning_rate" => Float64(run.learning_rate),
                "embedding_size" => 128,
                "ffn_hidden_size" => 256,
                "context_length" => 48,
                "epochs" => 22,
                "train_loss" => metrics["epoch_metrics"][end]["train_loss"],
                "validation_loss" => metrics["epoch_metrics"][end]["validation_loss"],
                "test_loss" => metrics["training"]["test_loss"],
                "test_perplexity" => metrics["training"]["test_perplexity"],
                "train_example_count" => metrics["training"]["train_example_count"],
                "train_batches" => metrics["training"]["train_batches"],
                "final_step" => metrics["training"]["final_step"],
                "sample_outputs_path" => metrics["artifacts"]["sample_outputs_path"],
                "metrics_path" => run_metrics_path,
            ),
        )
    end

    summary = Dict(
        "experiment" => "prepared_better_local_real_text_optimizer_sweep",
        "purpose" => "optimizer-family comparison on the prepared better local real-text corpus at the current best recipe",
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
            "epochs" => 22,
            "seed_style" => "same deterministic seeds as the current best prepared-corpus run",
        ),
        "comparison_policy" => "optimizer-family comparison with optimizer-specific standard learning rates",
        "sweep_axis" => "optimizer_family",
        "runs" => run_summaries,
        "baseline" => Dict(
            "metrics_path" => baseline_metrics_path,
            "optimizer_name" => get(baseline_metrics["training"], :optimizer_name, "Flux.Descent"),
            "optimizer_hyperparameters" => get(baseline_metrics["training"], :optimizer_hyperparameters, Dict("learning_rate" => baseline_metrics["training"]["learning_rate"])),
            "epochs" => baseline_metrics["training"]["epochs"],
            "learning_rate" => baseline_metrics["training"]["learning_rate"],
            "embedding_size" => baseline_metrics["model"]["embedding_size"],
            "ffn_hidden_size" => baseline_metrics["model"]["ffn_hidden_size"],
            "context_length" => baseline_metrics["model"]["context_length"],
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

    open(joinpath(output_dir, OPTIMIZER_SWEEP_SUMMARY_JSON), "w") do io
        JSON3.write(io, summary)
    end
    write_optimizer_sweep_summary_md(joinpath(output_dir, OPTIMIZER_SWEEP_SUMMARY_MD), summary)

    println("optimizer sweep summary: ", joinpath(output_dir, OPTIMIZER_SWEEP_SUMMARY_JSON))
    println("optimizer sweep markdown: ", joinpath(output_dir, OPTIMIZER_SWEEP_SUMMARY_MD))
    return summary
end

function write_optimizer_sweep_summary_md(path::AbstractString, summary)
    baseline = summary["baseline"]
    runs = summary["runs"]
    open(path, "w") do io
        println(io, "# Prepared Better Local Real-Text Optimizer Sweep")
        println(io)
        println(io, "Baseline: optimizer=$(baseline["optimizer_name"]), lr=$(baseline["learning_rate"]), epochs=$(baseline["epochs"]), test_loss=$(round(baseline["test_loss"], digits=4))")
        println(io)
        println(io, "| run | optimizer | learning_rate | train_loss | val_loss | test_loss | train_examples | train_batches | final_step |")
        println(io, "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for run in runs
            println(
                io,
                "| $(run["run_name"]) | $(run["optimizer_name"]) | $(run["learning_rate"]) | $(round(run["train_loss"], digits=4)) | " *
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
