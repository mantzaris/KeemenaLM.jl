#!/usr/bin/env julia

using JSON3
using Printf

include("run_tiny_chatbot_demo_subword_v1.jl")

const SUBWORD_SWEEP_OUTPUT_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_demo_subword_budget_sweep")
const SUBWORD_SWEEP_EPOCHS = [28, 30, 32]

function main(args)
    length(args) == 0 || error("usage: julia --project=tools/subword_real_text tools/run_tiny_chatbot_demo_subword_budget_sweep.jl")
    run_tiny_chatbot_demo_subword_budget_sweep()
end

function run_tiny_chatbot_demo_subword_budget_sweep()
    mkpath(SUBWORD_SWEEP_OUTPUT_DIR)
    runs = Dict{String, Any}[]

    for epochs in SUBWORD_SWEEP_EPOCHS
        run_name = @sprintf("epochs_%02d", epochs)
        output_dir = joinpath(SUBWORD_SWEEP_OUTPUT_DIR, run_name)
        metrics = run_tiny_chatbot_subword_demo(
            TINY_CHATBOT_DATASET_DIR,
            output_dir;
            settings = TinyChatbotSubwordSettings(epochs = epochs),
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
        "experiment" => "tiny_chatbot_demo_subword_budget_sweep",
        "dataset_dir" => TINY_CHATBOT_DATASET_DIR,
        "comparison_baseline" => Dict(
            "original_run_dir" => TINY_CHATBOT_SUBWORD_OUTPUT_DIR,
            "epochs" => 50,
        ),
        "fixed_recipe" => Dict(
            "optimizer" => "Flux.Adam(0.001)",
            "tokenizer_package" => "KeemenaSubwords.jl",
            "tokenizer_trainer" => "hf_gpt2_bytebpe",
            "tokenizer_vocab_size" => 512,
            "tokenizer_min_frequency" => 2,
            "document_separator" => TINY_CHATBOT_DOCUMENT_SEPARATOR,
            "context_length" => 48,
            "num_layers" => 2,
            "num_heads" => 2,
            "embedding_size" => 128,
            "ffn_hidden_size" => 256,
            "batch_size" => 16,
            "chat_marker_strategy" => "literal markers in dataset text plus tokenizer added special tokens",
        ),
        "varied_only" => "epochs",
        "runs" => runs,
    )

    write_json(joinpath(SUBWORD_SWEEP_OUTPUT_DIR, "summary.json"), summary)
    write_summary_markdown(joinpath(SUBWORD_SWEEP_OUTPUT_DIR, "summary.md"), summary)
    return summary
end

function write_summary_markdown(path::AbstractString, summary)
    open(path, "w") do io
        println(io, "# Tiny Chatbot Demo Subword Budget Sweep")
        println(io)
        println(io, "- dataset: `", summary["dataset_dir"], "`")
        println(io, "- varied only: `epochs`")
        println(io, "- comparison baseline: original `50`-epoch subword run")
        println(io)

        fixed = summary["fixed_recipe"]
        println(io, "- fixed recipe:")
        println(io, "  - optimizer: `", fixed["optimizer"], "`")
        println(io, "  - tokenizer package: `", fixed["tokenizer_package"], "`")
        println(io, "  - tokenizer trainer: `", fixed["tokenizer_trainer"], "`")
        println(io, "  - tokenizer vocab size: `", fixed["tokenizer_vocab_size"], "`")
        println(io, "  - tokenizer min frequency: `", fixed["tokenizer_min_frequency"], "`")
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
