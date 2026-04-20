#!/usr/bin/env julia

using Flux
using JSON3
using KeemenaLM

include("run_prepared_better_local_real_text_experiment.jl")

const TINY_CHATBOT_DATASET_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_demo_dataset_v1")
const TINY_CHATBOT_OUTPUT_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_demo_run_v1")
const TINY_CHATBOT_EXPERIMENT_NAME = "tiny_chatbot_demo_run_v1"
const TINY_CHATBOT_PURPOSE = "first tiny conversational chatbot demo run on the hybrid local-plus-oasst1 dataset"
const TINY_CHATBOT_EPOCHS = 24
const TINY_CHATBOT_DOCUMENT_SEPARATOR = "\n\n"

function main(args)
    dataset_dir = TINY_CHATBOT_DATASET_DIR
    output_dir = TINY_CHATBOT_OUTPUT_DIR
    epochs = TINY_CHATBOT_EPOCHS
    embedding_size = 128
    ffn_hidden_size = 256
    experiment_name = TINY_CHATBOT_EXPERIMENT_NAME
    purpose = TINY_CHATBOT_PURPOSE

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
            error("usage: julia --project=. tools/run_tiny_chatbot_demo_v1.jl [--dataset-dir DIR] [--output-dir DIR] [--epochs N] [--embedding-size N] [--ffn-hidden-size N] [--experiment-name NAME] [--purpose TEXT]")
        end
        argument_index += 1
    end

    run_tiny_chatbot_demo(
        dataset_dir,
        output_dir;
        epochs = epochs,
        embedding_size = embedding_size,
        ffn_hidden_size = ffn_hidden_size,
        experiment_name = experiment_name,
        purpose = purpose,
    )
end

function tiny_chatbot_settings(; epochs::Int, embedding_size::Int = 128, ffn_hidden_size::Int = 256)
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

function tiny_chatbot_recipe_dict(
    dataset_dir::AbstractString,
    output_dir::AbstractString;
    epochs::Int,
    experiment_name::AbstractString = TINY_CHATBOT_EXPERIMENT_NAME,
    purpose::AbstractString = TINY_CHATBOT_PURPOSE,
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
        "seed_style" => "same deterministic seeds as the prepared-corpus/chatbot experiments",
        "dataset_format" => "conversational User/Assistant chat pairs with explicit <END_ASSISTANT> and <CHAT_END> markers in the data",
        "document_separator" => TINY_CHATBOT_DOCUMENT_SEPARATOR,
    )
end

function tiny_chatbot_evaluation_prompts()::Vector{String}
    return [
        "User: Hi there.\nAssistant:",
        "User: Can you help me plan a calm evening?\nAssistant:",
        "User: Can you rewrite this to sound kinder: 'You forgot again.'\nAssistant:",
        "User: I'm feeling stuck and I don't know where to start.\nAssistant:",
        "User: Can you give me three low-stress weekend ideas?\nAssistant:",
        "User: What is a bundle in KeemenaLM.jl, in simple terms?\nAssistant:",
    ]
end

function run_tiny_chatbot_demo(
    dataset_dir::AbstractString,
    output_dir::AbstractString;
    epochs::Int = TINY_CHATBOT_EPOCHS,
    experiment_name::AbstractString = TINY_CHATBOT_EXPERIMENT_NAME,
    purpose::AbstractString = TINY_CHATBOT_PURPOSE,
    embedding_size::Int = 128,
    ffn_hidden_size::Int = 256,
)
    output_dir = abspath(output_dir)
    mkpath(output_dir)

    recipe = tiny_chatbot_recipe_dict(
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
        settings = tiny_chatbot_settings(; epochs = epochs, embedding_size = embedding_size, ffn_hidden_size = ffn_hidden_size),
        experiment_name = String(experiment_name),
        purpose = String(purpose),
        optimizer_builder = settings -> Flux.Adam(settings.learning_rate),
        optimizer_name = "Flux.Adam",
        optimizer_hparams = Dict("learning_rate" => 0.001f0),
        document_separator = TINY_CHATBOT_DOCUMENT_SEPARATOR,
        source_type = "tiny_conversational_chatbot_demo_dataset",
        dataset_format = "User/Assistant chat examples with explicit <END_ASSISTANT> and <CHAT_END> markers already present in each sample",
    )

    prompts = tiny_chatbot_evaluation_prompts()
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
        settings = tiny_chatbot_settings(; epochs = epochs, embedding_size = embedding_size, ffn_hidden_size = ffn_hidden_size),
    )
    write_samples(sample_path, samples)

    metrics["samples"] = samples
    metrics["evaluation_prompts"] = prompts
    metrics["corpus"]["source_type"] = "tiny_conversational_chatbot_demo_dataset"
    metrics["corpus"]["dataset_format"] = "User/Assistant chat examples with explicit <END_ASSISTANT> and <CHAT_END> markers already present in each sample"
    metrics["corpus"]["document_separator"] = TINY_CHATBOT_DOCUMENT_SEPARATOR
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
