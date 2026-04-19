#!/usr/bin/env julia

using JSON3
using Printf

include("run_keemena_docs_assistant_chatbot_poc.jl")

const CHATBOT_WIDTH_SWEEP_OUTPUT_DIR = joinpath(pwd(), "tmp", "keemena_docs_assistant_chatbot_width_sweep")

function main(args)
    dataset_dir = CHATBOT_DATASET_DIR
    output_dir = CHATBOT_WIDTH_SWEEP_OUTPUT_DIR

    argument_index = 1
    while argument_index <= length(args)
        argument = args[argument_index]
        if argument == "--dataset-dir"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --dataset-dir")
            dataset_dir = abspath(args[argument_index])
        elseif argument == "--output-dir"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --output-dir")
            output_dir = abspath(args[argument_index])
        else
            error("usage: julia --project=. tools/run_keemena_docs_assistant_chatbot_width_sweep.jl [--dataset-dir DIR] [--output-dir DIR]")
        end
        argument_index += 1
    end

    run_chatbot_width_sweep(dataset_dir, output_dir)
end

function run_chatbot_width_sweep(dataset_dir::AbstractString, output_dir::AbstractString)
    output_dir = abspath(output_dir)
    mkpath(output_dir)

    runs = [
        (embedding_size = 128, ffn_hidden_size = 256),
        (embedding_size = 160, ffn_hidden_size = 320),
        (embedding_size = 192, ffn_hidden_size = 384),
    ]

    results = Dict{String, Any}[]
    for run in runs
        label = "embed_$(run.embedding_size)_ffn_$(run.ffn_hidden_size)"
        run_output_dir = joinpath(output_dir, label)
        metrics = run_chatbot_poc(
            dataset_dir,
            run_output_dir;
            epochs = CHATBOT_EPOCHS,
            embedding_size = run.embedding_size,
            ffn_hidden_size = run.ffn_hidden_size,
        )

        push!(results, Dict(
            "label" => label,
            "embedding_size" => run.embedding_size,
            "ffn_hidden_size" => run.ffn_hidden_size,
            "output_dir" => run_output_dir,
            "metrics_path" => joinpath(run_output_dir, "metrics.json"),
            "sample_outputs_path" => String(metrics["artifacts"]["sample_outputs_path"]),
            "train_loss" => metrics["epoch_metrics"][end]["train_loss"],
            "validation_loss" => metrics["epoch_metrics"][end]["validation_loss"],
            "test_loss" => metrics["training"]["test_loss"],
            "train_example_count" => metrics["training"]["train_example_count"],
            "train_batches" => metrics["training"]["train_batches"],
            "final_step" => metrics["training"]["final_step"],
        ))
    end

    summary = Dict(
        "experiment" => "keemena_docs_assistant_chatbot_width_sweep",
        "dataset_dir" => abspath(dataset_dir),
        "fixed_settings" => Dict(
            "optimizer" => "Flux.Adam(0.001)",
            "tokenizer" => "char-level experiment-local tokenizer",
            "document_separator" => CHATBOT_DOCUMENT_SEPARATOR,
            "context_length" => 48,
            "num_layers" => 2,
            "num_heads" => 2,
            "batch_size" => 16,
            "epochs" => 30,
        ),
        "results" => results,
    )

    write_json(joinpath(output_dir, "summary.json"), summary)
    write_summary_markdown(joinpath(output_dir, "summary.md"), summary)
    return summary
end

function write_summary_markdown(path::AbstractString, summary::Dict{String, Any})
    open(path, "w") do io
        println(io, "# Keemena Docs Assistant Chatbot Width Sweep")
        println(io)
        println(io, "- dataset: `", summary["dataset_dir"], "`")
        println(io, "- fixed optimizer: `Flux.Adam(0.001)`")
        println(io, "- fixed boundary separator: `<CHAT_END>`")
        println(io)
        println(io, "| run | embed | ffn | train_loss | validation_loss | test_loss | final_step |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for result in summary["results"]
            println(
                io,
                "| `", result["label"], "` | ",
                result["embedding_size"], " | ",
                result["ffn_hidden_size"], " | ",
                @sprintf("%.4f", result["train_loss"]), " | ",
                @sprintf("%.4f", result["validation_loss"]), " | ",
                @sprintf("%.4f", result["test_loss"]), " | ",
                result["final_step"], " |",
            )
        end
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
