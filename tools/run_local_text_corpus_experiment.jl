#!/usr/bin/env julia

using Flux
using JSON3
using KeemenaLM
using Printf
using Random

include("run_benchmark_cfg_experiment.jl")

const LOCAL_TEXT_DEFAULT_OUTPUT_DIR = joinpath(pwd(), "tmp", "local_text_corpus_experiment")
const EXPERIMENT_NAME = "local_text_corpus_experiment"
const CORPUS_FILES = [
    "README.md",
    "docs/src/index.md",
    "notes/repo_plan_short.md",
    "notes/repo_plan_structure.md",
    "notes/todo_staged_roadmap.md",
]

function main(args)
    length(args) <= 1 || error("usage: julia --project=tools/benchmark_cfg tools/run_local_text_corpus_experiment.jl [output_dir]")

    output_dir = length(args) == 1 ? abspath(args[1]) : LOCAL_TEXT_DEFAULT_OUTPUT_DIR
    dataset_dir = joinpath(output_dir, "dataset")
    checkpoint_dir = joinpath(output_dir, "checkpoints")
    bundle_dir = joinpath(output_dir, "bundle")
    tokenizer_path = joinpath(output_dir, "tokenizer.json")
    metrics_path = joinpath(output_dir, "metrics.json")
    sample_path = joinpath(output_dir, "sample_outputs.txt")
    dataset_metadata_path = joinpath(dataset_dir, "corpus_metadata.json")

    mkpath(output_dir)
    mkpath(dataset_dir)
    mkpath(checkpoint_dir)

    settings = merge_settings(
        ExperimentSettings();
        complexity = 0, # unused for this experiment
        num_sentences = 0, # unused for this experiment
        prompt_prefix_characters = 24,
    )

    println("== KeemenaLM local text corpus experiment ==")
    println("output_dir: $(output_dir)")
    println("purpose: small real-text transfer sanity check, not chatbot benchmarking")

    corpus_entries = collect_corpus_entries(CORPUS_FILES)
    split_entries = split_entries_deterministically(corpus_entries)
    split_texts = (
        training = [entry.text for entry in split_entries.training],
        validation = [entry.text for entry in split_entries.validation],
        testing = [entry.text for entry in split_entries.testing],
    )
    split_paths = write_split_files(dataset_dir, split_texts)
    corpus_metadata = write_corpus_metadata(dataset_metadata_path, corpus_entries, split_entries)

    tokenizer = build_tokenizer(split_texts)
    save_tokenizer(tokenizer_path, tokenizer)

    train_batches, train_stats = build_lm_batches(
        split_texts.training,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
    )
    validation_batches, validation_batch_stats = build_lm_batches(
        split_texts.validation,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
    )
    test_batches, test_batch_stats = build_lm_batches(
        split_texts.testing,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
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
    trainer = KeemenaLM.Core.Trainer(
        model;
        optimizer = Flux.Descent(settings.learning_rate),
        backend = :flux,
        metadata = Dict(
            "experiment" => EXPERIMENT_NAME,
            "corpus_source" => "local_markdown_repo_docs",
            "tokenizer_path" => tokenizer_path,
        ),
    )

    initial_validation_loss = mean_loss(model, validation_batches)
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
        save_checkpoint(checkpoint_path, trainer, model; experiment = EXPERIMENT_NAME, epoch = epoch)

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
    save_checkpoint(final_checkpoint_path, trainer, model; experiment = EXPERIMENT_NAME, stage = "final")

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
        "experiment" => EXPERIMENT_NAME,
        "purpose" => "small real-text transfer sanity check only",
        "corpus" => Dict(
            "source_type" => "local_markdown_files",
            "source_files" => CORPUS_FILES,
            "corpus_metadata_file" => dataset_metadata_path,
            "training_file" => split_paths.training,
            "validation_file" => split_paths.validation,
            "testing_file" => split_paths.testing,
            "split_policy" => "deterministic paragraph-order split 80/10/10 after fenced-code removal and markdown-light normalization",
            "dataset_seed" => settings.dataset_seed,
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
            "batch_size" => settings.batch_size,
            "epochs" => settings.epochs,
            "learning_rate" => settings.learning_rate,
            "train_batches" => length(train_batches),
            "validation_batches" => length(validation_batches),
            "test_batches" => length(test_batches),
            "train_token_stream_length" => train_stats.token_stream_length,
            "train_example_count" => train_stats.example_count,
            "validation_token_stream_length" => validation_batch_stats.token_stream_length,
            "validation_example_count" => validation_batch_stats.example_count,
            "test_token_stream_length" => test_batch_stats.token_stream_length,
            "test_example_count" => test_batch_stats.example_count,
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
    println("prepared dataset metadata: $(corpus_metadata)")
    println("bundle export: $(bundle_dir)")
    println("checkpoint: $(final_checkpoint_path)")
    println("tokenizer: $(tokenizer_path)")
    println("metrics: $(metrics_path)")
    println("samples: $(sample_path)")
end

Base.@kwdef struct CorpusEntry
    source_file::String
    paragraph_index::Int
    text::String
end

function collect_corpus_entries(relative_paths::Vector{String})::Vector{CorpusEntry}
    entries = CorpusEntry[]
    for relative_path in relative_paths
        absolute_path = joinpath(pwd(), relative_path)
        isfile(absolute_path) || throw(ArgumentError("corpus file does not exist: $(absolute_path)"))
        paragraphs = markdown_to_paragraphs(read(absolute_path, String))
        for (paragraph_index, paragraph) in enumerate(paragraphs)
            push!(entries, CorpusEntry(relative_path, paragraph_index, paragraph))
        end
    end
    isempty(entries) && throw(ArgumentError("no usable paragraphs were extracted from the selected corpus files"))
    return entries
end

function markdown_to_paragraphs(markdown_text::AbstractString)::Vector{String}
    lines = split(markdown_text, '\n')
    paragraphs = String[]
    paragraph_lines = String[]
    in_code_fence = false

    for raw_line in lines
        line = replace(raw_line, '\r' => "")
        stripped = strip(line)

        if startswith(stripped, "```")
            in_code_fence = !in_code_fence
            continue
        end
        in_code_fence && continue

        normalized_line = normalize_markdown_line(line)
        if isempty(strip(normalized_line))
            maybe_push_paragraph!(paragraphs, paragraph_lines)
        else
            push!(paragraph_lines, normalized_line)
        end
    end
    maybe_push_paragraph!(paragraphs, paragraph_lines)
    return paragraphs
end

function normalize_markdown_line(line::AbstractString)::String
    normalized = strip(line)
    isempty(normalized) && return ""
    startswith(normalized, "[![") && return ""

    normalized = replace(normalized, r"^#+\s*" => "")
    normalized = replace(normalized, r"^[-*]\s+" => "")
    normalized = replace(normalized, r"^\d+\.\s+" => "")
    normalized = replace(normalized, r"\[([^\]]+)\]\([^)]+\)" => s"\1")
    normalized = replace(normalized, '`' => "")
    normalized = replace(normalized, r"\s+" => " ")

    return strip(normalized)
end

function maybe_push_paragraph!(paragraphs::Vector{String}, paragraph_lines::Vector{String})
    isempty(paragraph_lines) && return nothing
    paragraph = join(paragraph_lines, " ")
    word_count = length(split(paragraph))
    if word_count >= 5
        push!(paragraphs, paragraph)
    end
    empty!(paragraph_lines)
    return nothing
end

function split_entries_deterministically(entries::Vector{CorpusEntry})
    total_entries = length(entries)
    total_entries >= 10 || throw(ArgumentError("need at least 10 extracted paragraphs for a stable 80/10/10 split"))

    training_count = max(1, floor(Int, 0.8 * total_entries))
    validation_count = max(1, floor(Int, 0.1 * total_entries))
    testing_count = total_entries - training_count - validation_count
    testing_count >= 1 || throw(ArgumentError("not enough corpus entries to create a test split"))

    training_entries = entries[1:training_count]
    validation_entries = entries[(training_count + 1):(training_count + validation_count)]
    testing_entries = entries[(training_count + validation_count + 1):end]

    return (
        training = training_entries,
        validation = validation_entries,
        testing = testing_entries,
    )
end

function write_split_files(dataset_dir::AbstractString, split_texts)
    paths = (
        training = joinpath(dataset_dir, "training.txt"),
        validation = joinpath(dataset_dir, "validation.txt"),
        testing = joinpath(dataset_dir, "testing.txt"),
    )
    open(paths.training, "w") do io
        write(io, join(split_texts.training, "\n\n"))
    end
    open(paths.validation, "w") do io
        write(io, join(split_texts.validation, "\n\n"))
    end
    open(paths.testing, "w") do io
        write(io, join(split_texts.testing, "\n\n"))
    end
    return paths
end

function write_corpus_metadata(path::AbstractString, corpus_entries, split_entries)
    metadata = Dict(
        "source_files" => CORPUS_FILES,
        "total_paragraphs" => length(corpus_entries),
        "split_counts" => Dict(
            "training" => length(split_entries.training),
            "validation" => length(split_entries.validation),
            "testing" => length(split_entries.testing),
        ),
    )
    open(path, "w") do io
        JSON3.write(io, metadata)
    end
    return path
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

main(ARGS)
