#!/usr/bin/env julia

using BenchmarkDataNLP
using Flux
using JSON3
using KeemenaLM
using Printf
using Random

const DEFAULT_OUTPUT_DIR = joinpath(pwd(), "tmp", "benchmark_cfg_first_experiment")
const DEFAULT_EXPERIMENT_NAME = "benchmark_cfg_first_experiment"

Base.@kwdef struct ExperimentSettings
    dataset_seed::Int = 20260417
    model_seed::Int = 20260418
    generation_seed::Int = 20260419
    complexity::Int = 5
    num_sentences::Int = 8_000
    enable_polysemy::Bool = false
    context_length::Int = 48
    batch_size::Int = 16
    epochs::Int = 2
    learning_rate::Float32 = 0.01f0
    num_layers::Int = 2
    num_heads::Int = 2
    embedding_size::Int = 64
    ffn_hidden_size::Int = 128
    prompt_prefix_characters::Int = 12
    sample_generation_tokens::Int = 32
end

struct ExperimentCharTokenizer
    alphabet::Vector{Char}
    token_to_id::Dict{Char, Int}
    id_to_token::Dict{Int, Char}
end

function ExperimentCharTokenizer(alphabet::AbstractVector{Char})
    token_to_id = Dict(character => index for (index, character) in enumerate(alphabet))
    id_to_token = Dict(index => character for (index, character) in enumerate(alphabet))
    return ExperimentCharTokenizer(collect(alphabet), token_to_id, id_to_token)
end

KeemenaLM.Core.tokenizer_encode(tokenizer::ExperimentCharTokenizer, text::AbstractString) =
    [get(tokenizer.token_to_id, character, 1) for character in text]

KeemenaLM.Core.tokenizer_decode(tokenizer::ExperimentCharTokenizer, token_ids::AbstractVector{<:Integer}) =
    String([get(tokenizer.id_to_token, Int(token_id), '?') for token_id in token_ids])

function run_experiment(
    settings::ExperimentSettings,
    output_dir::AbstractString;
    experiment_name::AbstractString = DEFAULT_EXPERIMENT_NAME,
    purpose::AbstractString = "pipeline validation, not chatbot-quality benchmarking",
)
    output_dir = abspath(output_dir)
    dataset_dir = joinpath(output_dir, "dataset")
    checkpoint_dir = joinpath(output_dir, "checkpoints")
    bundle_dir = joinpath(output_dir, "bundle")
    tokenizer_path = joinpath(output_dir, "tokenizer.json")
    metrics_path = joinpath(output_dir, "metrics.json")
    sample_path = joinpath(output_dir, "sample_outputs.txt")
    dataset_base_filename = "cfg_complexity_$(settings.complexity)_n$(settings.num_sentences)"

    mkpath(output_dir)
    mkpath(dataset_dir)
    mkpath(checkpoint_dir)

    println("== KeemenaLM BenchmarkDataNLP CFG experiment ==")
    println("output_dir: $(output_dir)")
    println("purpose: $(purpose)")

    dataset_paths = generate_cfg_dataset(settings, dataset_dir, dataset_base_filename)
    split_texts = load_split_texts(dataset_paths)
    tokenizer = build_tokenizer(split_texts)
    save_tokenizer(tokenizer_path, tokenizer)

    train_batches = build_lm_batches(split_texts.training, tokenizer; context_length = settings.context_length, batch_size = settings.batch_size)
    validation_batches = build_lm_batches(split_texts.validation, tokenizer; context_length = settings.context_length, batch_size = settings.batch_size)
    test_batches = build_lm_batches(split_texts.testing, tokenizer; context_length = settings.context_length, batch_size = settings.batch_size)

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
            "experiment" => experiment_name,
            "dataset_generator" => "BenchmarkDataNLP.generate_corpus_CFG",
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
        save_checkpoint(checkpoint_path, trainer, model; experiment = experiment_name, epoch = epoch)

        epoch_summary = Dict(
            "epoch" => epoch,
            "step" => trainer.step,
            "train_loss" => train_loss,
            "train_perplexity" => exp(train_loss),
            "validation_loss" => validation_loss,
            "validation_perplexity" => exp(validation_loss),
            "checkpoint_path" => checkpoint_path,
        )
        push!(epoch_metrics, epoch_summary)

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
        "dataset" => Dict(
            "generator" => "CFG",
            "package" => "BenchmarkDataNLP.jl",
            "complexity" => settings.complexity,
            "num_sentences" => settings.num_sentences,
            "enable_polysemy" => settings.enable_polysemy,
            "dataset_seed" => settings.dataset_seed,
            "training_file" => dataset_paths.training,
            "validation_file" => dataset_paths.validation,
            "testing_file" => dataset_paths.testing,
            "metadata_file" => dataset_paths.metadata,
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

function main(args)
    settings = ExperimentSettings()
    output_dir = DEFAULT_OUTPUT_DIR
    experiment_name = DEFAULT_EXPERIMENT_NAME

    argument_index = 1
    while argument_index <= length(args)
        argument = args[argument_index]
        if argument == "--output-dir"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --output-dir")
            output_dir = abspath(args[argument_index])
        elseif argument == "--complexity"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --complexity")
            settings = merge_settings(settings; complexity = parse(Int, args[argument_index]))
        elseif argument == "--num-sentences"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --num-sentences")
            settings = merge_settings(settings; num_sentences = parse(Int, args[argument_index]))
        elseif argument == "--experiment-name"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --experiment-name")
            experiment_name = args[argument_index]
        elseif startswith(argument, "--")
            error("unknown argument $(argument)")
        elseif argument_index == length(args)
            output_dir = abspath(argument)
        else
            error("usage: julia --project=tools/benchmark_cfg tools/run_benchmark_cfg_experiment.jl [--output-dir DIR] [--complexity N] [--num-sentences N] [--experiment-name NAME] [output_dir]")
        end
        argument_index += 1
    end

    run_experiment(settings, output_dir; experiment_name = experiment_name)
end

function merge_settings(
    settings::ExperimentSettings;
    dataset_seed::Int = settings.dataset_seed,
    model_seed::Int = settings.model_seed,
    generation_seed::Int = settings.generation_seed,
    complexity::Int = settings.complexity,
    num_sentences::Int = settings.num_sentences,
    enable_polysemy::Bool = settings.enable_polysemy,
    context_length::Int = settings.context_length,
    batch_size::Int = settings.batch_size,
    epochs::Int = settings.epochs,
    learning_rate::Float32 = settings.learning_rate,
    num_layers::Int = settings.num_layers,
    num_heads::Int = settings.num_heads,
    embedding_size::Int = settings.embedding_size,
    ffn_hidden_size::Int = settings.ffn_hidden_size,
    prompt_prefix_characters::Int = settings.prompt_prefix_characters,
    sample_generation_tokens::Int = settings.sample_generation_tokens,
)
    return ExperimentSettings(
        dataset_seed = dataset_seed,
        model_seed = model_seed,
        generation_seed = generation_seed,
        complexity = complexity,
        num_sentences = num_sentences,
        enable_polysemy = enable_polysemy,
        context_length = context_length,
        batch_size = batch_size,
        epochs = epochs,
        learning_rate = learning_rate,
        num_layers = num_layers,
        num_heads = num_heads,
        embedding_size = embedding_size,
        ffn_hidden_size = ffn_hidden_size,
        prompt_prefix_characters = prompt_prefix_characters,
        sample_generation_tokens = sample_generation_tokens,
    )
end

function generate_cfg_dataset(settings::ExperimentSettings, dataset_dir::AbstractString, base_filename::AbstractString)
    Random.seed!(settings.dataset_seed)
    BenchmarkDataNLP.generate_corpus_CFG(
        complexity = settings.complexity,
        num_sentences = settings.num_sentences,
        enable_polysemy = settings.enable_polysemy,
        output_dir = dataset_dir,
        base_filename = base_filename,
    )

    return (
        training = joinpath(dataset_dir, base_filename * "_training.jsonl"),
        testing = joinpath(dataset_dir, base_filename * "_testing.jsonl"),
        validation = joinpath(dataset_dir, base_filename * "_validation.jsonl"),
        metadata = joinpath(dataset_dir, base_filename * "_metadata.json"),
    )
end

function load_split_texts(dataset_paths)
    return (
        training = read_jsonl_texts(dataset_paths.training),
        testing = read_jsonl_texts(dataset_paths.testing),
        validation = read_jsonl_texts(dataset_paths.validation),
    )
end

function read_jsonl_texts(path::AbstractString)::Vector{String}
    isfile(path) || throw(ArgumentError("dataset file does not exist: $(path)"))

    texts = String[]
    for line in eachline(path)
        entry = JSON3.read(line)
        hasproperty(entry, :text) || throw(ArgumentError("dataset line is missing a text field in $(path)"))
        push!(texts, String(entry.text))
    end
    isempty(texts) && throw(ArgumentError("dataset split is empty: $(path)"))
    return texts
end

function build_tokenizer(split_texts)
    alphabet = Set{Char}(['\n'])
    for texts in values(split_texts)
        for text in texts
            for character in text
                push!(alphabet, character)
            end
        end
    end
    return ExperimentCharTokenizer(sort!(collect(alphabet)))
end

function save_tokenizer(path::AbstractString, tokenizer::ExperimentCharTokenizer)
    open(path, "w") do io
        JSON3.write(io, Dict("alphabet" => [string(character) for character in tokenizer.alphabet]))
    end
    return path
end

function load_tokenizer(path::AbstractString)::ExperimentCharTokenizer
    isfile(path) || throw(ArgumentError("tokenizer file does not exist: $(path)"))
    tokenizer_data = JSON3.read(read(path, String))
    hasproperty(tokenizer_data, :alphabet) || throw(ArgumentError("tokenizer file is missing alphabet"))
    alphabet = [only(String(character_string)) for character_string in tokenizer_data.alphabet]
    return ExperimentCharTokenizer(alphabet)
end

function build_lm_batches(
    texts::Vector{String},
    tokenizer::ExperimentCharTokenizer;
    context_length::Int,
    batch_size::Int,
)
    corpus = join(texts, "\n")
    token_ids = KeemenaLM.Core.tokenizer_encode(tokenizer, corpus)
    length(token_ids) > context_length ||
        throw(ArgumentError("dataset split is too small for context_length=$(context_length)"))

    example_count = fld(length(token_ids) - 1, context_length)
    example_count > 0 || throw(ArgumentError("dataset split did not yield any LM examples"))

    inputs = Vector{Vector{Int32}}(undef, example_count)
    targets = Vector{Vector{Int32}}(undef, example_count)

    for example_index in 1:example_count
        offset = (example_index - 1) * context_length
        input_slice = token_ids[(offset + 1):(offset + context_length)]
        target_slice = token_ids[(offset + 2):(offset + context_length + 1)]
        inputs[example_index] = Int32.(input_slice)
        targets[example_index] = Int32.(target_slice)
    end

    batches = Tuple{Matrix{Int32}, Matrix{Int32}}[]
    for batch_start in 1:batch_size:example_count
        batch_end = min(batch_start + batch_size - 1, example_count)
        actual_batch_size = batch_end - batch_start + 1
        input_batch = Matrix{Int32}(undef, context_length, actual_batch_size)
        target_batch = Matrix{Int32}(undef, context_length, actual_batch_size)

        for (column_index, example_index) in enumerate(batch_start:batch_end)
            input_batch[:, column_index] = inputs[example_index]
            target_batch[:, column_index] = targets[example_index]
        end

        push!(batches, (input_batch, target_batch))
    end

    return batches
end

function mean_loss(model, batches)::Float64
    total_loss = 0.0
    for (input_batch, target_batch) in batches
        logits, _ = KeemenaLM.Core.lm_forward(model, input_batch; cache = nothing, is_training = false)
        total_loss += Float64(KeemenaLM.Core.causal_lm_cross_entropy(logits, target_batch))
    end
    return total_loss / length(batches)
end

function sample_prompts(texts::Vector{String}; count::Int, prefix_characters::Int)
    prompts = String[]
    for text in texts
        isempty(strip(text)) && continue
        push!(prompts, first(text, min(prefix_characters, length(text))))
        length(prompts) == count && break
    end
    isempty(prompts) && throw(ArgumentError("unable to derive non-empty prompts from the test split"))
    return prompts
end

function generate_samples(model, tokenizer, prompts::Vector{String}; settings::ExperimentSettings)
    generation_config = GenerationConfig(
        max_new_tokens = settings.sample_generation_tokens,
        temperature = 0.0,
        seed = settings.generation_seed,
    )

    return [
        Dict(
            "prompt" => prompt,
            "output" => generate(model, tokenizer, nothing, prompt; generation_config = generation_config),
        ) for prompt in prompts
    ]
end

function write_samples(path::AbstractString, samples)
    open(path, "w") do io
        for sample in samples
            println(io, "prompt> ", sample["prompt"])
            println(io, "output> ", sample["output"])
            println(io)
        end
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
