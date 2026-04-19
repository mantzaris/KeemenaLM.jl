#!/usr/bin/env julia

using Flux
using JSON3
using KeemenaLM
using Printf
using Random

include("experiment_common.jl")

const PREPARED_CORPUS_INPUT_DIR = joinpath(pwd(), "tmp", "better_local_real_text_corpus_prepared", "dataset")
const PREPARED_CORPUS_OUTPUT_DIR = joinpath(pwd(), "tmp", "prepared_better_local_real_text_experiment")
const PREPARED_CORPUS_EXPERIMENT_NAME = "prepared_better_local_real_text_experiment"
const PREPARED_CORPUS_EXPERIMENT_PURPOSE = "first training run on the prepared better local real-text corpus, not chatbot benchmarking"

function main(args)
    length(args) <= 2 || error("usage: julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_experiment.jl [prepared_dataset_dir] [output_dir]")
    dataset_dir = length(args) >= 1 ? abspath(args[1]) : PREPARED_CORPUS_INPUT_DIR
    output_dir = length(args) == 2 ? abspath(args[2]) : PREPARED_CORPUS_OUTPUT_DIR
    run_prepared_better_local_real_text_experiment(dataset_dir, output_dir)
end

function run_prepared_better_local_real_text_experiment(
    dataset_dir::AbstractString,
    output_dir::AbstractString;
    settings::ExperimentSettings = merge_settings(
        ExperimentSettings();
        complexity = 0,
        num_sentences = 0,
        prompt_prefix_characters = 24,
    ),
    experiment_name::AbstractString = PREPARED_CORPUS_EXPERIMENT_NAME,
    purpose::AbstractString = PREPARED_CORPUS_EXPERIMENT_PURPOSE,
    optimizer_builder = nothing,
    optimizer_name::AbstractString = "Flux.Descent",
    optimizer_hparams::AbstractDict = Dict{String, Any}(),
    document_separator::AbstractString = "\n",
    source_type::AbstractString = "prepared_local_text_dataset",
    dataset_format::AbstractString = "plain local text splits",
)
    training_path = joinpath(dataset_dir, "training.txt")
    validation_path = joinpath(dataset_dir, "validation.txt")
    testing_path = joinpath(dataset_dir, "testing.txt")
    metadata_path = resolve_prepared_metadata_path(dataset_dir)

    isfile(training_path) || throw(ArgumentError("prepared training split does not exist: $(training_path)"))
    isfile(validation_path) || throw(ArgumentError("prepared validation split does not exist: $(validation_path)"))
    isfile(testing_path) || throw(ArgumentError("prepared testing split does not exist: $(testing_path)"))

    output_dir = abspath(output_dir)
    checkpoint_dir = joinpath(output_dir, "checkpoints")
    bundle_dir = joinpath(output_dir, "bundle")
    tokenizer_path = joinpath(output_dir, "tokenizer.json")
    metrics_path = joinpath(output_dir, "metrics.json")
    sample_path = joinpath(output_dir, "sample_outputs.txt")

    mkpath(output_dir)
    mkpath(checkpoint_dir)

    split_texts = load_prepared_split_texts(dataset_dir)
    corpus_metadata = JSON3.read(read(metadata_path, String))
    tokenizer = build_tokenizer(split_texts; extra_texts = isempty(document_separator) ? String[] : [document_separator])
    save_tokenizer(tokenizer_path, tokenizer)

    train_batches, train_stats = build_lm_batches(
        split_texts.training,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        document_separator = document_separator,
    )
    validation_batches, validation_stats = build_lm_batches(
        split_texts.validation,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        document_separator = document_separator,
    )
    test_batches, test_stats = build_lm_batches(
        split_texts.testing,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        document_separator = document_separator,
    )

    config = GPT2Config(
        vocab_size = length(tokenizer.alphabet),
        context_length = settings.context_length,
        num_layers = settings.num_layers,
        num_heads = settings.num_heads,
        embedding_size = settings.embedding_size,
        ffn_hidden_size = settings.ffn_hidden_size,
    )

    Random.seed!(settings.model_seed)
    model = instantiate(config; backend = :flux, seed = settings.model_seed)
    resolved_optimizer_builder = isnothing(optimizer_builder) ? (experiment_settings -> Flux.Descent(experiment_settings.learning_rate)) : optimizer_builder
    resolved_optimizer_hparams = isempty(optimizer_hparams) ? Dict{String, Any}("learning_rate" => settings.learning_rate) : Dict{String, Any}(String(key) => value for (key, value) in pairs(optimizer_hparams))
    trainer = KeemenaLM.Core.Trainer(
        model;
        optimizer = resolved_optimizer_builder(settings),
        backend = :flux,
        metadata = Dict(
            "experiment" => experiment_name,
            "corpus_source" => source_type,
            "tokenizer_path" => tokenizer_path,
            "prepared_corpus_metadata_path" => metadata_path,
            "optimizer_name" => optimizer_name,
            "document_separator" => document_separator,
        ),
    )

    initial_validation_loss = mean_loss(model, validation_batches)
    println("== KeemenaLM prepared better local real-text experiment ==")
    println("prepared_dataset_dir: $(dataset_dir)")
    println("output_dir: $(output_dir)")
    println(@sprintf("initial validation loss: %.4f", initial_validation_loss))

    epoch_metrics = Dict{String, Any}[]
    for epoch in 1:settings.epochs
        epoch_losses = Float64[]
        for (input_batch, target_batch) in train_batches
            step_result = KeemenaLM.Core.train_step!(trainer, input_batch, target_batch)
            push!(epoch_losses, step_result.loss)
        end

        trainer.epoch = epoch
        train_loss = sum(epoch_losses) / length(epoch_losses)
        validation_loss = mean_loss(model, validation_batches)
        checkpoint_path = joinpath(checkpoint_dir, @sprintf("epoch_%02d_checkpoint.jld2", epoch))
        save_checkpoint(checkpoint_path, trainer, model; experiment = experiment_name, epoch = epoch)

        push!(
            epoch_metrics,
            Dict(
                "epoch" => epoch,
                "step" => trainer.step,
                "train_loss" => train_loss,
                "train_perplexity" => exp(train_loss),
                "validation_loss" => validation_loss,
                "validation_perplexity" => exp(validation_loss),
                "checkpoint_path" => checkpoint_path,
            ),
        )

        println(
            @sprintf(
                "epoch %d/%d  train_loss=%.4f  validation_loss=%.4f  checkpoint=%s",
                epoch,
                settings.epochs,
                train_loss,
                validation_loss,
                checkpoint_path,
            ),
        )
    end

    final_checkpoint_path = joinpath(checkpoint_dir, "final_checkpoint.jld2")
    save_checkpoint(final_checkpoint_path, trainer, model; experiment = experiment_name, stage = "final")

    bundle = Bundle(
        model_config = KeemenaLM.Core.model_config(model),
        weights = KeemenaLM.Core.extract_weights(model),
    )
    save_bundle(bundle_dir, bundle)

    reloaded_bundle = load_bundle(bundle_dir)
    reloaded_model = instantiate(reloaded_bundle; backend = :flux)
    reloaded_tokenizer = load_tokenizer(tokenizer_path)

    test_loss = mean_loss(reloaded_model, test_batches)
    prompts = sample_prompts(split_texts.testing; count = 3, prefix_characters = settings.prompt_prefix_characters)
    samples = generate_samples(reloaded_model, reloaded_tokenizer, prompts; settings = settings)
    write_samples(sample_path, samples)

    metrics = Dict(
        "experiment" => experiment_name,
        "purpose" => purpose,
        "corpus" => Dict(
            "source_type" => source_type,
            "prepared_dataset_dir" => dataset_dir,
            "corpus_metadata_file" => metadata_path,
            "training_file" => training_path,
            "validation_file" => validation_path,
            "testing_file" => testing_path,
            "split_policy" => metadata_value(corpus_metadata, "split_policy", "unspecified"),
            "split_method" => metadata_value(corpus_metadata, "split_method", "deterministic_fixed_split"),
            "corpus_name" => metadata_string(corpus_metadata, "corpus_name", basename(dataset_dir)),
            "dataset_format" => dataset_format,
            "document_separator" => document_separator,
        ),
        "model" => Dict(
            "backend" => "flux",
            "vocab_size" => config.vocab_size,
            "context_length" => config.context_length,
            "num_layers" => config.num_layers,
            "num_heads" => config.num_heads,
            "embedding_size" => config.embedding_size,
            "ffn_hidden_size" => config.ffn_hidden_size,
            "model_seed" => settings.model_seed,
        ),
        "training" => Dict(
            "optimizer_name" => optimizer_name,
            "optimizer_hyperparameters" => resolved_optimizer_hparams,
            "batch_size" => settings.batch_size,
            "epochs" => settings.epochs,
            "learning_rate" => settings.learning_rate,
            "train_batches" => length(train_batches),
            "validation_batches" => length(validation_batches),
            "test_batches" => length(test_batches),
            "train_token_stream_length" => train_stats.token_stream_length,
            "train_example_count" => train_stats.example_count,
            "validation_token_stream_length" => validation_stats.token_stream_length,
            "validation_example_count" => validation_stats.example_count,
            "test_token_stream_length" => test_stats.token_stream_length,
            "test_example_count" => test_stats.example_count,
            "final_step" => trainer.step,
            "final_epoch" => trainer.epoch,
            "initial_validation_loss" => initial_validation_loss,
            "test_loss" => test_loss,
            "test_perplexity" => exp(test_loss),
        ),
        "artifacts" => Dict(
            "tokenizer_path" => tokenizer_path,
            "final_checkpoint" => final_checkpoint_path,
            "bundle_dir" => bundle_dir,
            "sample_outputs_path" => sample_path,
        ),
        "epoch_metrics" => epoch_metrics,
        "split_stats" => Dict(
            "training" => split_text_stats(split_texts.training, settings.context_length),
            "validation" => split_text_stats(split_texts.validation, settings.context_length),
            "testing" => split_text_stats(split_texts.testing, settings.context_length),
        ),
        "samples" => samples,
    )

    open(metrics_path, "w") do io
        JSON3.write(io, metrics)
    end

    println(@sprintf("final test loss: %.4f", test_loss))
    println("bundle export: $(bundle_dir)")
    println("checkpoint: $(final_checkpoint_path)")
    println("tokenizer: $(tokenizer_path)")
    println("metrics: $(metrics_path)")
    println("samples: $(sample_path)")
    return metrics
end

function resolve_prepared_metadata_path(dataset_dir::AbstractString)::String
    candidates = (
        joinpath(dataset_dir, "corpus_metadata.json"),
        joinpath(dataset_dir, "metadata.json"),
    )
    for path in candidates
        isfile(path) && return path
    end
    throw(ArgumentError("prepared dataset metadata does not exist in $(dataset_dir); expected one of $(collect(candidates))"))
end

function metadata_value(metadata, key::AbstractString, default)
    return haskey(metadata, key) ? metadata[key] : default
end

function metadata_string(metadata, key::AbstractString, default::AbstractString)::String
    value = metadata_value(metadata, key, default)
    return value isa AbstractString ? String(value) : string(value)
end

function load_prepared_split_texts(dataset_dir::AbstractString)
    return (
        training = read_prepared_paragraphs(joinpath(dataset_dir, "training.txt")),
        validation = read_prepared_paragraphs(joinpath(dataset_dir, "validation.txt")),
        testing = read_prepared_paragraphs(joinpath(dataset_dir, "testing.txt")),
    )
end

function read_prepared_paragraphs(path::AbstractString)::Vector{String}
    contents = read(path, String)
    paragraphs = [strip(paragraph) for paragraph in split(contents, r"\n\s*\n")]
    paragraphs = filter(!isempty, paragraphs)
    isempty(paragraphs) && throw(ArgumentError("prepared split is empty: $(path)"))
    return paragraphs
end

function split_text_stats(texts::Vector{String}, context_length::Int)::Dict{String, Any}
    token_stream_length = sum(length, texts) + max(length(texts) - 1, 0)
    example_count = fld(max(token_stream_length - 1, 0), context_length)
    return Dict(
        "paragraph_count" => length(texts),
        "token_stream_length" => token_stream_length,
        "example_count" => example_count,
        "mean_characters_per_paragraph" => sum(length, texts) / length(texts),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
