#!/usr/bin/env julia

using JSON3
using Printf

include("run_benchmark_cfg_experiment.jl")

const DEFAULT_SWEEP_OUTPUT_DIR = joinpath(pwd(), "tmp", "benchmark_cfg_width_sweep")
const SWEEP_COMPLEXITY = 5
const SWEEP_NUM_SENTENCES = 4_000
const SWEEP_EPOCHS = 2
const SWEEP_BATCH_SIZE = 16
const SWEEP_LEARNING_RATE = 0.01f0
const WIDTH_CONFIGS = [
    (embedding_size = 32, ffn_hidden_size = 64),
    (embedding_size = 64, ffn_hidden_size = 128),
    (embedding_size = 96, ffn_hidden_size = 192),
]

function main(args)
    length(args) <= 1 || error("usage: julia --project=tools/benchmark_cfg tools/run_benchmark_cfg_width_sweep.jl [output_dir]")

    sweep_output_dir = length(args) == 1 ? abspath(args[1]) : DEFAULT_SWEEP_OUTPUT_DIR
    mkpath(sweep_output_dir)

    base_settings = merge_settings(
        ExperimentSettings();
        complexity = SWEEP_COMPLEXITY,
        num_sentences = SWEEP_NUM_SENTENCES,
        epochs = SWEEP_EPOCHS,
        batch_size = SWEEP_BATCH_SIZE,
        learning_rate = SWEEP_LEARNING_RATE,
    )

    run_summaries = Dict{String, Any}[]
    for width in WIDTH_CONFIGS
        run_name = "cfg_complexity_$(SWEEP_COMPLEXITY)_embed_$(width.embedding_size)"
        run_output_dir = joinpath(sweep_output_dir, run_name)
        settings = merge_settings(
            base_settings;
            embedding_size = width.embedding_size,
            ffn_hidden_size = width.ffn_hidden_size,
        )
        metrics = run_experiment(
            settings,
            run_output_dir;
            experiment_name = run_name,
            purpose = "fixed-complexity synthetic CFG width sweep for pipeline evaluation",
        )
        push!(run_summaries, summarize_run(metrics))
    end

    summary = Dict(
        "experiment" => "benchmark_cfg_width_sweep",
        "purpose" => "controlled embedding-width sweep at fixed CFG complexity for the existing KeemenaLM experiment pipeline",
        "sweep_variable" => "embedding_size",
        "ffn_scaling" => "ffn_hidden_size = 2 * embedding_size",
        "held_fixed" => Dict(
            "complexity" => SWEEP_COMPLEXITY,
            "num_sentences" => SWEEP_NUM_SENTENCES,
            "epochs" => SWEEP_EPOCHS,
            "batch_size" => SWEEP_BATCH_SIZE,
            "learning_rate" => SWEEP_LEARNING_RATE,
            "enable_polysemy" => false,
            "backend" => "flux",
            "context_length" => base_settings.context_length,
            "num_layers" => base_settings.num_layers,
            "num_heads" => base_settings.num_heads,
            "dataset_seed" => base_settings.dataset_seed,
            "model_seed" => base_settings.model_seed,
            "generation_seed" => base_settings.generation_seed,
        ),
        "runs" => run_summaries,
    )

    summary_json_path = joinpath(sweep_output_dir, "summary.json")
    open(summary_json_path, "w") do io
        JSON3.write(io, summary)
    end

    summary_markdown_path = joinpath(sweep_output_dir, "summary.md")
    write_summary_markdown(summary_markdown_path, summary)

    println("sweep summary json: $(summary_json_path)")
    println("sweep summary markdown: $(summary_markdown_path)")
end

function summarize_run(metrics)::Dict{String, Any}
    final_epoch = metrics["epoch_metrics"][end]
    train_text_stats = split_text_stats(metrics["dataset"]["training_file"])
    validation_text_stats = split_text_stats(metrics["dataset"]["validation_file"])
    test_text_stats = split_text_stats(metrics["dataset"]["testing_file"])

    return Dict(
        "run_name" => metrics["experiment"],
        "embedding_size" => metrics["model"]["embedding_size"],
        "ffn_hidden_size" => metrics["model"]["ffn_hidden_size"],
        "complexity" => metrics["dataset"]["complexity"],
        "num_sentences" => metrics["dataset"]["num_sentences"],
        "vocab_size" => metrics["model"]["vocab_size"],
        "train_loss" => final_epoch["train_loss"],
        "train_perplexity" => final_epoch["train_perplexity"],
        "validation_loss" => final_epoch["validation_loss"],
        "validation_perplexity" => final_epoch["validation_perplexity"],
        "test_loss" => metrics["training"]["test_loss"],
        "test_perplexity" => metrics["training"]["test_perplexity"],
        "train_batches" => metrics["training"]["train_batches"],
        "validation_batches" => metrics["training"]["validation_batches"],
        "test_batches" => metrics["training"]["test_batches"],
        "final_step" => metrics["training"]["final_step"],
        "train_split_stats" => train_text_stats,
        "validation_split_stats" => validation_text_stats,
        "test_split_stats" => test_text_stats,
        "bundle_dir" => metrics["artifacts"]["bundle_dir"],
        "sample_outputs_path" => metrics["artifacts"]["sample_outputs_path"],
        "samples" => metrics["samples"],
    )
end

function split_text_stats(path::AbstractString)::Dict{String, Any}
    texts = read_jsonl_texts(path)
    total_characters = sum(length, texts)
    token_stream_length = total_characters + max(length(texts) - 1, 0)
    return Dict(
        "lines" => length(texts),
        "total_characters" => total_characters,
        "token_stream_length" => token_stream_length,
        "mean_characters_per_line" => total_characters / length(texts),
    )
end

function write_summary_markdown(path::AbstractString, summary::Dict{String, Any})
    open(path, "w") do io
        println(io, "# BenchmarkDataNLP CFG width sweep")
        println(io)
        println(io, summary["purpose"])
        println(io)
        println(io, "- Sweep variable: `", summary["sweep_variable"], "`")
        println(io, "- FFN scaling: `", summary["ffn_scaling"], "`")
        println(
            io,
            "- Held fixed: `complexity = ",
            summary["held_fixed"]["complexity"],
            "`, `num_sentences = ",
            summary["held_fixed"]["num_sentences"],
            "`, `epochs = ",
            summary["held_fixed"]["epochs"],
            "`, same Flux training path, same checkpoint/bundle flow",
        )
        println(io)
        println(io, "| Embedding | FFN | Train Loss | Validation Loss | Test Loss | Test PPL | Final Step | Train Tokens | Sample Outputs |")
        println(io, "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for run in summary["runs"]
            println(
                io,
                "| ",
                run["embedding_size"],
                " | ",
                run["ffn_hidden_size"],
                " | ",
                @sprintf("%.4f", run["train_loss"]),
                " | ",
                @sprintf("%.4f", run["validation_loss"]),
                " | ",
                @sprintf("%.4f", run["test_loss"]),
                " | ",
                @sprintf("%.4f", run["test_perplexity"]),
                " | ",
                run["final_step"],
                " | ",
                run["train_split_stats"]["token_stream_length"],
                " | `",
                run["sample_outputs_path"],
                "` |",
            )
        end
    end
    return path
end

main(ARGS)
