#!/usr/bin/env julia

using JSON3
using Printf

include("run_tiny_chatbot_demo_v1.jl")

const SWEEP_OUTPUT_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_demo_budget_sweep")
const SWEEP_EPOCHS = [24, 32, 40]

function main(args)
    length(args) == 0 || error("usage: julia --project=. tools/run_tiny_chatbot_demo_budget_sweep.jl")
    run_tiny_chatbot_demo_budget_sweep()
end

function run_tiny_chatbot_demo_budget_sweep()
    mkpath(SWEEP_OUTPUT_DIR)
    runs = Dict{String, Any}[]

    for epochs in SWEEP_EPOCHS
        run_name = @sprintf("epochs_%02d", epochs)
        output_dir = joinpath(SWEEP_OUTPUT_DIR, run_name)
        metrics = run_tiny_chatbot_demo(
            TINY_CHATBOT_DATASET_DIR,
            output_dir;
            epochs = epochs,
            experiment_name = "tiny_chatbot_demo_budget_sweep_" * run_name,
            purpose = "tiny conversational chatbot budget sweep at fixed recipe, epochs=$(epochs)",
        )

        push!(runs, Dict(
            "epochs" => epochs,
            "output_dir" => output_dir,
            "train_loss" => metrics["epoch_metrics"][end]["train_loss"],
            "validation_loss" => metrics["epoch_metrics"][end]["validation_loss"],
            "test_loss" => metrics["training"]["test_loss"],
            "final_step" => metrics["training"]["final_step"],
            "train_example_count" => metrics["training"]["train_example_count"],
            "train_batches" => metrics["training"]["train_batches"],
            "sample_outputs_path" => metrics["artifacts"]["sample_outputs_path"],
            "metrics_path" => joinpath(output_dir, "metrics.json"),
        ))
    end

    summary = Dict(
        "experiment" => "tiny_chatbot_demo_budget_sweep",
        "dataset_dir" => TINY_CHATBOT_DATASET_DIR,
        "fixed_recipe" => Dict(
            "optimizer" => "Flux.Adam(0.001)",
            "tokenizer" => "char-level experiment-local tokenizer",
            "document_separator" => TINY_CHATBOT_DOCUMENT_SEPARATOR,
            "context_length" => 48,
            "num_layers" => 2,
            "num_heads" => 2,
            "embedding_size" => 128,
            "ffn_hidden_size" => 256,
            "batch_size" => 16,
        ),
        "varied_only" => "epochs",
        "runs" => runs,
    )

    write_json(joinpath(SWEEP_OUTPUT_DIR, "summary.json"), summary)
    write_summary_markdown(joinpath(SWEEP_OUTPUT_DIR, "summary.md"), summary)
    return summary
end

function write_summary_markdown(path::AbstractString, summary)
    open(path, "w") do io
        println(io, "# Tiny Chatbot Demo Budget Sweep")
        println(io)
        println(io, "- dataset: `", summary["dataset_dir"], "`")
        println(io, "- varied only: `epochs`")
        println(io, "- fixed recipe:")
        fixed = summary["fixed_recipe"]
        println(io, "  - optimizer: `", fixed["optimizer"], "`")
        println(io, "  - tokenizer: `", fixed["tokenizer"], "`")
        println(io, "  - separator: `", repr(fixed["document_separator"]), "`")
        println(io, "  - context_length: `", fixed["context_length"], "`")
        println(io, "  - num_layers: `", fixed["num_layers"], "`")
        println(io, "  - num_heads: `", fixed["num_heads"], "`")
        println(io, "  - embedding_size: `", fixed["embedding_size"], "`")
        println(io, "  - ffn_hidden_size: `", fixed["ffn_hidden_size"], "`")
        println(io, "  - batch_size: `", fixed["batch_size"], "`")
        println(io)

        for run in summary["runs"]
            println(io, "## epochs = `", run["epochs"], "`")
            println(io)
            println(io, "- train loss: `", @sprintf("%.4f", run["train_loss"]), "`")
            println(io, "- validation loss: `", @sprintf("%.4f", run["validation_loss"]), "`")
            println(io, "- test loss: `", @sprintf("%.4f", run["test_loss"]), "`")
            println(io, "- final step: `", run["final_step"], "`")
            println(io, "- train example count: `", run["train_example_count"], "`")
            println(io, "- train batches: `", run["train_batches"], "`")
            println(io, "- output dir: `", run["output_dir"], "`")
            println(io, "- sample outputs: `", run["sample_outputs_path"], "`")
            println(io)
        end
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
