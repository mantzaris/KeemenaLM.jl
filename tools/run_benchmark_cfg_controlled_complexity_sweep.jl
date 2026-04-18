#!/usr/bin/env julia

using JSON3
using Printf

include("run_benchmark_cfg_experiment.jl")

const DEFAULT_SWEEP_OUTPUT_DIR = joinpath(pwd(), "tmp", "benchmark_cfg_controlled_complexity_sweep")
const SWEEP_COMPLEXITIES = [3, 5, 7]
const SWEEP_NUM_SENTENCES = 4_000
const TRAIN_TOKEN_STREAM_TARGET = 72_001
const TRAIN_EXAMPLE_TARGET = fld(TRAIN_TOKEN_STREAM_TARGET - 1, 48)

function main(args)
    length(args) <= 1 || error("usage: julia --project=tools/benchmark_cfg tools/run_benchmark_cfg_controlled_complexity_sweep.jl [output_dir]")

    sweep_output_dir = length(args) == 1 ? abspath(args[1]) : DEFAULT_SWEEP_OUTPUT_DIR
    mkpath(sweep_output_dir)

    base_settings = merge_settings(
        ExperimentSettings();
        num_sentences = SWEEP_NUM_SENTENCES,
        train_token_stream_limit = TRAIN_TOKEN_STREAM_TARGET,
    )

    run_summaries = Dict{String, Any}[]
    for complexity in SWEEP_COMPLEXITIES
        run_name = "cfg_complexity_$(complexity)_train_tokens_$(TRAIN_TOKEN_STREAM_TARGET)"
        run_output_dir = joinpath(sweep_output_dir, run_name)
        settings = merge_settings(base_settings; complexity = complexity)
        metrics = run_experiment(
            settings,
            run_output_dir;
            experiment_name = run_name,
            purpose = "token-controlled synthetic CFG complexity sweep for pipeline evaluation",
        )
        push!(run_summaries, summarize_run(metrics))
    end

    summary = Dict(
        "experiment" => "benchmark_cfg_controlled_complexity_sweep",
        "purpose" => "controlled complexity sweep with matched training token stream length for the existing KeemenaLM experiment pipeline",
        "control_method" => "training token stream length",
        "train_token_stream_target" => TRAIN_TOKEN_STREAM_TARGET,
        "train_example_target" => TRAIN_EXAMPLE_TARGET,
        "sweep_variable" => "complexity",
        "held_fixed" => Dict(
            "num_sentences" => SWEEP_NUM_SENTENCES,
            "enable_polysemy" => false,
            "backend" => "flux",
            "context_length" => base_settings.context_length,
            "num_layers" => base_settings.num_layers,
            "num_heads" => base_settings.num_heads,
            "embedding_size" => base_settings.embedding_size,
            "ffn_hidden_size" => base_settings.ffn_hidden_size,
            "batch_size" => base_settings.batch_size,
            "epochs" => base_settings.epochs,
            "learning_rate" => base_settings.learning_rate,
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
        "train_token_stream_length" => metrics["training"]["train_token_stream_length"],
        "train_example_count" => metrics["training"]["train_example_count"],
        "train_token_stream_limit" => metrics["training"]["train_token_stream_limit"],
        "raw_train_split_stats" => train_text_stats,
        "validation_split_stats" => validation_text_stats,
        "test_split_stats" => test_text_stats,
        "final_step" => metrics["training"]["final_step"],
        "bundle_dir" => metrics["artifacts"]["bundle_dir"],
        "sample_outputs_path" => metrics["artifacts"]["sample_outputs_path"],
        "samples" => metrics["samples"],
    )
end

function split_text_stats(path::AbstractString)::Dict{String, Any}
    texts = read_jsonl_texts(path)
    total_characters = sum(length, texts)
    token_stream_length = total_characters + max(length(texts) - 1, 0)
    example_count = fld(token_stream_length - 1, 48)
    return Dict(
        "lines" => length(texts),
        "total_characters" => total_characters,
        "token_stream_length" => token_stream_length,
        "example_count" => example_count,
        "mean_characters_per_line" => total_characters / length(texts),
    )
end

function write_summary_markdown(path::AbstractString, summary::Dict{String, Any})
    open(path, "w") do io
        println(io, "# BenchmarkDataNLP CFG controlled complexity sweep")
        println(io)
        println(io, summary["purpose"])
        println(io)
        println(io, "- Control method: `", summary["control_method"], "`")
        println(io, "- Training token target: `", summary["train_token_stream_target"], "`")
        println(io, "- Training example target: `", summary["train_example_target"], "`")
        println(io, "- Sweep variable: `", summary["sweep_variable"], "`")
        println(io)
        println(io, "| Complexity | Raw Train Tokens | Controlled Train Tokens | Controlled Examples | Test Loss | Test PPL | Sample Outputs |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for run in summary["runs"]
            println(
                io,
                "| ",
                run["complexity"],
                " | ",
                run["raw_train_split_stats"]["token_stream_length"],
                " | ",
                run["train_token_stream_length"],
                " | ",
                run["train_example_count"],
                " | ",
                @sprintf("%.4f", run["test_loss"]),
                " | ",
                @sprintf("%.4f", run["test_perplexity"]),
                " | `",
                run["sample_outputs_path"],
                "` |",
            )
        end
    end
    return path
end

main(ARGS)
