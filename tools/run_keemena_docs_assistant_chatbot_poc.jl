#!/usr/bin/env julia

using Flux
using JSON3
using KeemenaLM

include("run_prepared_better_local_real_text_experiment.jl")

const CHATBOT_DATASET_DIR = joinpath(pwd(), "tmp", "keemena_docs_assistant_dataset_v2")
const CHATBOT_OUTPUT_DIR = joinpath(pwd(), "tmp", "keemena_docs_assistant_chatbot_poc_run_with_boundaries")
const CHATBOT_EXPERIMENT_NAME = "keemena_docs_assistant_chatbot_poc_run_with_boundaries"
const CHATBOT_PURPOSE = "first narrow-domain Keemena Docs Assistant chatbot proof-of-concept run with explicit chat example boundaries"
const CHATBOT_EPOCHS = 30
const CHATBOT_DOCUMENT_SEPARATOR = "\n<CHAT_END>\n"

function main(args)
    dataset_dir = CHATBOT_DATASET_DIR
    output_dir = CHATBOT_OUTPUT_DIR
    epochs = CHATBOT_EPOCHS
    embedding_size = 128
    ffn_hidden_size = 256
    experiment_name = CHATBOT_EXPERIMENT_NAME
    purpose = CHATBOT_PURPOSE

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
        elseif argument == "--epochs"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --epochs")
            epochs = parse(Int, args[argument_index])
        elseif argument == "--embedding-size"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --embedding-size")
            embedding_size = parse(Int, args[argument_index])
        elseif argument == "--ffn-hidden-size"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --ffn-hidden-size")
            ffn_hidden_size = parse(Int, args[argument_index])
        elseif argument == "--experiment-name"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --experiment-name")
            experiment_name = args[argument_index]
        elseif argument == "--purpose"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --purpose")
            purpose = args[argument_index]
        else
            error("usage: julia --project=. tools/run_keemena_docs_assistant_chatbot_poc.jl [--dataset-dir DIR] [--output-dir DIR] [--epochs N] [--embedding-size N] [--ffn-hidden-size N] [--experiment-name NAME] [--purpose TEXT]")
        end
        argument_index += 1
    end

    run_chatbot_poc(
        dataset_dir,
        output_dir;
        epochs = epochs,
        embedding_size = embedding_size,
        ffn_hidden_size = ffn_hidden_size,
        experiment_name = experiment_name,
        purpose = purpose,
    )
end

function chatbot_settings(; epochs::Int, embedding_size::Int = 128, ffn_hidden_size::Int = 256)
    return merge_settings(
        ExperimentSettings();
        complexity = 0,
        num_sentences = 0,
        prompt_prefix_characters = 24,
        context_length = 48,
        num_layers = 2,
        num_heads = 2,
        embedding_size = embedding_size,
        ffn_hidden_size = ffn_hidden_size,
        batch_size = 16,
        epochs = epochs,
        learning_rate = 0.001f0,
        sample_generation_tokens = 96,
    )
end

function chatbot_recipe_dict(
    dataset_dir::AbstractString,
    output_dir::AbstractString;
    epochs::Int,
    experiment_name::AbstractString = CHATBOT_EXPERIMENT_NAME,
    purpose::AbstractString = CHATBOT_PURPOSE,
    embedding_size::Int = 128,
    ffn_hidden_size::Int = 256,
)
    return Dict(
        "experiment" => String(experiment_name),
        "purpose" => String(purpose),
        "dataset_dir" => abspath(dataset_dir),
        "output_dir" => abspath(output_dir),
        "backend" => "flux",
        "optimizer_name" => "Flux.Adam",
        "optimizer_hyperparameters" => Dict("learning_rate" => 0.001),
        "tokenizer" => "char-level experiment-local tokenizer",
        "context_length" => 48,
        "num_layers" => 2,
        "num_heads" => 2,
        "embedding_size" => embedding_size,
        "ffn_hidden_size" => ffn_hidden_size,
        "batch_size" => 16,
        "epochs" => epochs,
        "checkpoint_cadence" => "every epoch",
        "sample_generation_tokens" => 96,
        "seed_style" => "same deterministic seeds as the prepared-corpus experiments",
        "dataset_format" => "chat-style User/Assistant QA pairs",
        "document_separator" => CHATBOT_DOCUMENT_SEPARATOR,
    )
end

function chatbot_evaluation_prompts()::Vector{String}
    return [
        "User: How do I load a saved bundle in KeemenaLM.jl?\nAssistant:",
        "User: What is not supported yet in KeemenaLM.jl?\nAssistant:",
        "User: How do checkpoints work in KeemenaLM.jl?\nAssistant:",
        "User: What tokenizer support does KeemenaSubwords.jl provide?\nAssistant:",
        "User: What offset convention does KeemenaPreprocessing.jl use?\nAssistant:",
    ]
end

function run_chatbot_poc(
    dataset_dir::AbstractString,
    output_dir::AbstractString;
    epochs::Int = CHATBOT_EPOCHS,
    experiment_name::AbstractString = CHATBOT_EXPERIMENT_NAME,
    purpose::AbstractString = CHATBOT_PURPOSE,
    embedding_size::Int = 128,
    ffn_hidden_size::Int = 256,
)
    output_dir = abspath(output_dir)
    mkpath(output_dir)

    recipe = chatbot_recipe_dict(
        dataset_dir,
        output_dir;
        epochs = epochs,
        experiment_name = experiment_name,
        purpose = purpose,
        embedding_size = embedding_size,
        ffn_hidden_size = ffn_hidden_size,
    )
    write_json(joinpath(output_dir, "run_recipe.json"), recipe)

    metrics = run_prepared_better_local_real_text_experiment(
        dataset_dir,
        output_dir;
        settings = chatbot_settings(; epochs = epochs, embedding_size = embedding_size, ffn_hidden_size = ffn_hidden_size),
        experiment_name = String(experiment_name),
        purpose = String(purpose),
        optimizer_builder = settings -> Flux.Adam(settings.learning_rate),
        optimizer_name = "Flux.Adam",
        optimizer_hparams = Dict("learning_rate" => 0.001f0),
        document_separator = CHATBOT_DOCUMENT_SEPARATOR,
        source_type = "prepared_local_chatbot_docs_dataset",
        dataset_format = "User/Assistant chat-style QA pairs with explicit training-stream boundary separators",
    )

    prompts = chatbot_evaluation_prompts()
    write_prompt_files(output_dir, prompts)

    bundle_dir = String(metrics["artifacts"]["bundle_dir"])
    tokenizer_path = String(metrics["artifacts"]["tokenizer_path"])
    sample_path = String(metrics["artifacts"]["sample_outputs_path"])
    metrics_path = joinpath(output_dir, "metrics.json")

    reloaded_bundle = load_bundle(bundle_dir)
    reloaded_model = instantiate(reloaded_bundle; backend = :flux)
    reloaded_tokenizer = load_tokenizer(tokenizer_path)
    samples = generate_samples(
        reloaded_model,
        reloaded_tokenizer,
        prompts;
        settings = chatbot_settings(; epochs = epochs, embedding_size = embedding_size, ffn_hidden_size = ffn_hidden_size),
    )
    write_samples(sample_path, samples)

    metrics["samples"] = samples
    metrics["evaluation_prompts"] = prompts
    metrics["corpus"]["source_type"] = "prepared_local_chatbot_docs_dataset"
    metrics["corpus"]["dataset_format"] = "User/Assistant chat-style QA pairs with explicit training-stream boundary separators"
    metrics["corpus"]["document_separator"] = CHATBOT_DOCUMENT_SEPARATOR
    metrics["artifacts"]["evaluation_prompts_txt"] = joinpath(output_dir, "evaluation_prompts.txt")
    metrics["artifacts"]["evaluation_prompts_json"] = joinpath(output_dir, "evaluation_prompts.json")

    open(metrics_path, "w") do io
        JSON3.write(io, metrics)
    end

    return metrics
end

function write_prompt_files(output_dir::AbstractString, prompts::Vector{String})
    write_json(joinpath(output_dir, "evaluation_prompts.json"), Dict("prompts" => prompts))
    open(joinpath(output_dir, "evaluation_prompts.txt"), "w") do io
        for prompt in prompts
            println(io, prompt)
            println(io)
        end
    end
end

function write_json(path::AbstractString, value)
    open(path, "w") do io
        JSON3.write(io, value)
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
