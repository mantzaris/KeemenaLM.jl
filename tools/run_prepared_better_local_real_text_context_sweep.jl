#!/usr/bin/env julia

using JSON3
using Printf

include("run_prepared_better_local_real_text_experiment.jl")

const CONTEXT_SWEEP_OUTPUT_DIR = joinpath(pwd(), "tmp", "prepared_better_local_real_text_context_sweep")
const CONTEXT_SWEEP_SUMMARY_JSON = "summary.json"
const CONTEXT_SWEEP_SUMMARY_MD = "summary.md"

const CONTEXT_SWEEP_RUNS = [
    (context_length = 48,),
    (context_length = 64,),
    (context_length = 96,),
]

function main(args)
    length(args) <= 2 || error("usage: julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_context_sweep.jl [prepared_dataset_dir] [output_dir]")
    dataset_dir = length(args) >= 1 ? abspath(args[1]) : PREPARED_CORPUS_INPUT_DIR
    output_dir = length(args) == 2 ? abspath(args[2]) : CONTEXT_SWEEP_OUTPUT_DIR
    run_prepared_better_local_real_text_context_sweep(dataset_dir, output_dir)
end

function run_prepared_better_local_real_text_context_sweep(dataset_dir::AbstractString, output_dir::AbstractString)
    mkpath(output_dir)

    baseline_metrics_path = joinpath(
        pwd(),
        "tmp",
        "prepared_better_local_real_text_budget_sweep",
        "epochs_06",
        "metrics.json",
    )
    isfile(baseline_metrics_path) || throw(ArgumentError("128/256, 6-epoch baseline metrics not found: $(baseline_metrics_path)"))
    baseline_metrics = JSON3.read(read(baseline_metrics_path, String))

    run_summaries = Dict{String, Any}[]
    for run in CONTEXT_SWEEP_RUNS
        run_name = @sprintf("context_%03d", run.context_length)
        run_output_dir = joinpath(output_dir, run_name)
        settings = merge_settings(
            ExperimentSettings();
            complexity = 0,
            num_sentences = 0,
            prompt_prefix_characters = 24,
            embedding_size = 128,
            ffn_hidden_size = 256,
            epochs = 6,
            context_length = run.context_length,
        )

        metrics = run_prepared_better_local_real_text_experiment(
            dataset_dir,
            run_output_dir;
            settings = settings,
            experiment_name = "prepared_better_local_real_text_context_" * run_name,
            purpose = "prepared better local real-text context-length sweep at 128/256 width and 6 epochs, not chatbot benchmarking",
        )

        push!(
            run_summaries,
            Dict(
                "run_name" => run_name,
                "output_dir" => run_output_dir,
                "context_length" => run.context_length,
                "embedding_size" => 128,
                "ffn_hidden_size" => 256,
                "epochs" => 6,
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
        "experiment" => "prepared_better_local_real_text_context_sweep",
        "purpose" => "controlled context-length sweep on the prepared better local real-text corpus at 128/256 width and 6 epochs",
        "dataset_dir" => dataset_dir,
        "fixed_settings" => Dict(
            "backend" => "flux",
            "tokenizer" => "char-level experiment-local tokenizer",
            "num_layers" => 2,
            "num_heads" => 2,
            "embedding_size" => 128,
            "ffn_hidden_size" => 256,
            "epochs" => 6,
            "batch_size" => 16,
            "learning_rate" => 0.01,
            "seed_style" => "same deterministic seeds as the current best prepared-corpus run",
        ),
        "sweep_axis" => "context_length",
        "caveat" => "changing context length changes LM example construction and train example counts, so this sweep is less perfectly controlled than width or epoch sweeps",
        "runs" => run_summaries,
        "baseline" => Dict(
            "metrics_path" => baseline_metrics_path,
            "context_length" => baseline_metrics["model"]["context_length"],
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

    open(joinpath(output_dir, CONTEXT_SWEEP_SUMMARY_JSON), "w") do io
        JSON3.write(io, summary)
    end
    write_context_sweep_summary_md(joinpath(output_dir, CONTEXT_SWEEP_SUMMARY_MD), summary)

    println("context sweep summary: ", joinpath(output_dir, CONTEXT_SWEEP_SUMMARY_JSON))
    println("context sweep markdown: ", joinpath(output_dir, CONTEXT_SWEEP_SUMMARY_MD))
    return summary
end

function write_context_sweep_summary_md(path::AbstractString, summary)
    baseline = summary["baseline"]
    runs = summary["runs"]
    open(path, "w") do io
        println(io, "# Prepared Better Local Real-Text Context Sweep")
        println(io)
        println(io, "Caveat: changing context length also changes LM example construction and train example counts.")
        println(io)
        println(io, "Baseline: context=$(baseline["context_length"]), epochs=$(baseline["epochs"]), embed=$(baseline["embedding_size"]), ffn=$(baseline["ffn_hidden_size"]), test_loss=$(round(baseline["test_loss"], digits=4))")
        println(io)
        println(io, "| run | context | train_loss | val_loss | test_loss | train_examples | train_batches | final_step |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for run in runs
            println(
                io,
                "| $(run["run_name"]) | $(run["context_length"]) | $(round(run["train_loss"], digits=4)) | " *
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
