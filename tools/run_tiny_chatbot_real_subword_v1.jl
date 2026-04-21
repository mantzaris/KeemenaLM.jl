#!/usr/bin/env julia

using Flux
using JSON3
using KeemenaLM
using KeemenaSubwords
using Printf
using Random

const TINY_CHATBOT_DATASET_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_real_corpus_v1")
const TINY_CHATBOT_SUBWORD_OUTPUT_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_real_subword_run_v2")
const TINY_CHATBOT_SUBWORD_EXPERIMENT_NAME = "tiny_chatbot_real_subword_run_v2"
const TINY_CHATBOT_SUBWORD_PURPOSE = "throughput-aware serious subword conversational chatbot retrain on the real downloaded OASST1-dominant corpus"
const TINY_CHATBOT_DOCUMENT_SEPARATOR = "\n\n"
const CHAT_MARKERS = (
    user = "User:",
    assistant = "Assistant:",
    end_assistant = "<END_ASSISTANT>",
    chat_end = "<CHAT_END>",
)

Base.@kwdef struct TinyChatbotSubwordSettings
    model_seed::Int = 20260418
    generation_seed::Int = 20260419
    context_length::Int = 128
    batch_size::Int = 8
    epochs::Int = 3
    learning_rate::Float32 = 0.0003f0
    num_layers::Int = 6
    num_heads::Int = 6
    embedding_size::Int = 384
    ffn_hidden_size::Int = 1536
    sample_generation_tokens::Int = 160
    tokenizer_trainer::Symbol = :hf_gpt2_bytebpe
    tokenizer_vocab_size::Int = 4096
    tokenizer_min_frequency::Int = 2
    tokenizer_model_name::String = "tiny_chatbot_real_gpt2_bytebpe"
    document_separator::String = TINY_CHATBOT_DOCUMENT_SEPARATOR
    chat_special_tokens::Dict{Symbol,String} = Dict(
        :unk => "<|endoftext|>",
        :user => CHAT_MARKERS.user,
        :assistant => CHAT_MARKERS.assistant,
        :end_assistant => CHAT_MARKERS.end_assistant,
        :chat_end => CHAT_MARKERS.chat_end,
    )
end

KeemenaLM.Core.tokenizer_encode(tokenizer::KeemenaSubwords.AbstractSubwordTokenizer, text::AbstractString) =
    KeemenaSubwords.encode(tokenizer, text; add_special_tokens = false)

KeemenaLM.Core.tokenizer_decode(
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer,
    token_ids::AbstractVector{<:Integer},
) = KeemenaSubwords.decode(tokenizer, Int[Int(token_id) for token_id in token_ids])

function main(args)
    dataset_dir = TINY_CHATBOT_DATASET_DIR
    output_dir = TINY_CHATBOT_SUBWORD_OUTPUT_DIR
    settings = TinyChatbotSubwordSettings()

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
            settings = TinyChatbotSubwordSettings(;
                model_seed = settings.model_seed,
                generation_seed = settings.generation_seed,
                context_length = settings.context_length,
                batch_size = settings.batch_size,
                epochs = parse(Int, args[argument_index]),
                learning_rate = settings.learning_rate,
                num_layers = settings.num_layers,
                num_heads = settings.num_heads,
                embedding_size = settings.embedding_size,
                ffn_hidden_size = settings.ffn_hidden_size,
                sample_generation_tokens = settings.sample_generation_tokens,
                tokenizer_trainer = settings.tokenizer_trainer,
                tokenizer_vocab_size = settings.tokenizer_vocab_size,
                tokenizer_min_frequency = settings.tokenizer_min_frequency,
                tokenizer_model_name = settings.tokenizer_model_name,
                document_separator = settings.document_separator,
                chat_special_tokens = settings.chat_special_tokens,
            )
        elseif argument == "--batch-size"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --batch-size")
            settings = TinyChatbotSubwordSettings(;
                model_seed = settings.model_seed,
                generation_seed = settings.generation_seed,
                context_length = settings.context_length,
                batch_size = parse(Int, args[argument_index]),
                epochs = settings.epochs,
                learning_rate = settings.learning_rate,
                num_layers = settings.num_layers,
                num_heads = settings.num_heads,
                embedding_size = settings.embedding_size,
                ffn_hidden_size = settings.ffn_hidden_size,
                sample_generation_tokens = settings.sample_generation_tokens,
                tokenizer_trainer = settings.tokenizer_trainer,
                tokenizer_vocab_size = settings.tokenizer_vocab_size,
                tokenizer_min_frequency = settings.tokenizer_min_frequency,
                tokenizer_model_name = settings.tokenizer_model_name,
                document_separator = settings.document_separator,
                chat_special_tokens = settings.chat_special_tokens,
            )
        else
            error("usage: julia --project=tools/subword_real_text tools/run_tiny_chatbot_real_subword_v1.jl [--dataset-dir DIR] [--output-dir DIR] [--epochs N] [--batch-size N]")
        end
        argument_index += 1
    end

    run_tiny_chatbot_subword_demo(dataset_dir, output_dir; settings = settings)
end

function run_tiny_chatbot_subword_demo(
    dataset_dir::AbstractString,
    output_dir::AbstractString;
    settings::TinyChatbotSubwordSettings = TinyChatbotSubwordSettings(),
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
    tokenizer_bundle_dir = joinpath(output_dir, "tokenizer_bundle")
    metrics_path = joinpath(output_dir, "metrics.json")
    sample_path = joinpath(output_dir, "sample_outputs.txt")

    mkpath(output_dir)
    mkpath(checkpoint_dir)

    recipe = subword_recipe_dict(dataset_dir, output_dir, settings)
    write_json(joinpath(output_dir, "run_recipe.json"), recipe)

    split_texts = load_prepared_split_texts(dataset_dir)
    corpus_metadata = JSON3.read(read(metadata_path, String))

    tokenizer_training_texts = [text * settings.document_separator for text in split_texts.training]
    Random.seed!(settings.model_seed)
    tokenizer_training_output = KeemenaSubwords.quick_train_bundle(
        settings.tokenizer_trainer,
        tokenizer_training_texts;
        bundle_directory = tokenizer_bundle_dir,
        overwrite = true,
        export_format = :hf_tokenizer_json,
        vocab_size = settings.tokenizer_vocab_size,
        min_frequency = settings.tokenizer_min_frequency,
        model_name = settings.tokenizer_model_name,
        special_tokens = settings.chat_special_tokens,
        sanity_text = "User: Hi there.\nAssistant: Hello.\n<END_ASSISTANT>\n<CHAT_END>",
    )
    tokenizer = tokenizer_training_output.tokenizer

    marker_token_ids = Dict(
        string(key) => KeemenaSubwords.token_to_id(tokenizer, token_string) for (key, token_string) in settings.chat_special_tokens if key != :unk
    )

    train_batches, train_stats = build_subword_lm_batches(
        split_texts.training,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        document_separator = settings.document_separator,
    )
    validation_batches, validation_stats = build_subword_lm_batches(
        split_texts.validation,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        document_separator = settings.document_separator,
    )
    test_batches, test_stats = build_subword_lm_batches(
        split_texts.testing,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        document_separator = settings.document_separator,
    )

    config = GPT2Config(
        vocab_size = KeemenaSubwords.vocab_size(tokenizer),
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
        optimizer = Flux.Adam(settings.learning_rate),
        backend = :flux,
        metadata = Dict(
            "experiment" => TINY_CHATBOT_SUBWORD_EXPERIMENT_NAME,
            "corpus_source" => "tiny_chatbot_real_corpus_v1",
            "tokenizer_bundle_dir" => tokenizer_bundle_dir,
            "prepared_corpus_metadata_path" => metadata_path,
            "document_separator" => settings.document_separator,
            "optimizer_name" => "Flux.Adam",
            "chat_marker_representation" => "added special tokens present literally in training text",
        ),
    )

    initial_validation_loss = mean_loss(model, validation_batches)
    println("== Real conversational chatbot subword run ==")
    println("prepared_dataset_dir: $(dataset_dir)")
    println("output_dir: $(output_dir)")
    println("tokenizer_bundle_dir: $(tokenizer_bundle_dir)")
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
        save_checkpoint(checkpoint_path, trainer, model; experiment = TINY_CHATBOT_SUBWORD_EXPERIMENT_NAME, epoch = epoch)

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

        println(@sprintf(
            "epoch %d/%d  train_loss=%.4f  validation_loss=%.4f  checkpoint=%s",
            epoch,
            settings.epochs,
            train_loss,
            validation_loss,
            checkpoint_path,
        ))
    end

    final_checkpoint_path = joinpath(checkpoint_dir, "final_checkpoint.jld2")
    save_checkpoint(final_checkpoint_path, trainer, model; experiment = TINY_CHATBOT_SUBWORD_EXPERIMENT_NAME, stage = "final")

    bundle = Bundle(
        model_config = KeemenaLM.Core.model_config(model),
        weights = KeemenaLM.Core.extract_weights(model),
    )
    save_bundle(bundle_dir, bundle)

    reloaded_bundle = load_bundle(bundle_dir)
    reloaded_model = instantiate(reloaded_bundle; backend = :flux)
    reloaded_tokenizer = KeemenaSubwords.load_training_bundle(tokenizer_bundle_dir)

    prompts = tiny_chatbot_evaluation_prompts()
    write_prompt_files(output_dir, prompts)

    test_loss = mean_loss(reloaded_model, test_batches)
    samples = generate_samples(
        reloaded_model,
        reloaded_tokenizer,
        prompts;
        generation_seed = settings.generation_seed,
        max_new_tokens = settings.sample_generation_tokens,
    )
    write_samples(sample_path, samples)

    metrics = Dict(
        "experiment" => TINY_CHATBOT_SUBWORD_EXPERIMENT_NAME,
        "purpose" => TINY_CHATBOT_SUBWORD_PURPOSE,
        "comparison_note" => "Loss is not directly apples-to-apples with the char-level conversational baseline because tokenization changed. The more useful comparison is step budget plus qualitative outputs.",
        "corpus" => Dict(
            "source_type" => "tiny_chatbot_real_corpus_v1",
            "dataset_format" => "User/Assistant chat examples with explicit <END_ASSISTANT> and <CHAT_END> markers already present in each sample",
            "prepared_dataset_dir" => dataset_dir,
            "corpus_metadata_file" => metadata_path,
            "training_file" => training_path,
            "validation_file" => validation_path,
            "testing_file" => testing_path,
            "split_policy" => metadata_value(corpus_metadata, "split_policy", "unspecified"),
            "split_method" => metadata_value(corpus_metadata, "split_method", "deterministic_fixed_split"),
            "corpus_name" => metadata_string(corpus_metadata, "dataset_name", basename(dataset_dir)),
            "document_separator" => settings.document_separator,
        ),
        "tokenizer" => Dict(
            "package" => "KeemenaSubwords.jl",
            "trainer" => String(settings.tokenizer_trainer),
            "vocab_size_requested" => settings.tokenizer_vocab_size,
            "vocab_size_actual" => KeemenaSubwords.vocab_size(tokenizer),
            "min_frequency" => settings.tokenizer_min_frequency,
            "model_name" => settings.tokenizer_model_name,
            "bundle_directory" => tokenizer_bundle_dir,
            "tokenizer_json_path" => joinpath(tokenizer_bundle_dir, "tokenizer.json"),
            "chat_marker_strategy" => "Markers are present literally in the dataset and also trained as added special tokens so they get single stable token ids.",
            "chat_marker_strings" => Dict(string(key) => value for (key, value) in settings.chat_special_tokens),
            "chat_marker_token_ids" => marker_token_ids,
            "training_summary" => Dict(
                "trainer" => String(tokenizer_training_output.training_summary.trainer),
                "tokenizer_type" => tokenizer_training_output.training_summary.tokenizer_type,
                "config_type" => tokenizer_training_output.training_summary.config_type,
                "model_name" => tokenizer_training_output.training_summary.model_name,
                "version" => tokenizer_training_output.training_summary.version,
                "vocab_size" => tokenizer_training_output.training_summary.vocab_size,
            ),
            "bundle_persistence_note" => "Tokenizer bundle is saved separately; KeemenaLM model bundles still do not persist tokenizer payloads.",
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
            "optimizer_name" => "Flux.Adam",
            "optimizer_hyperparameters" => Dict("learning_rate" => settings.learning_rate),
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
            "tokenizer_bundle_dir" => tokenizer_bundle_dir,
            "final_checkpoint" => final_checkpoint_path,
            "bundle_dir" => bundle_dir,
            "sample_outputs_path" => sample_path,
            "evaluation_prompts_txt" => joinpath(output_dir, "evaluation_prompts.txt"),
            "evaluation_prompts_json" => joinpath(output_dir, "evaluation_prompts.json"),
        ),
        "epoch_metrics" => epoch_metrics,
        "split_stats" => Dict(
            "training" => split_text_stats(split_texts.training, settings.context_length, tokenizer; document_separator = settings.document_separator),
            "validation" => split_text_stats(split_texts.validation, settings.context_length, tokenizer; document_separator = settings.document_separator),
            "testing" => split_text_stats(split_texts.testing, settings.context_length, tokenizer; document_separator = settings.document_separator),
        ),
        "samples" => samples,
        "evaluation_prompts" => prompts,
    )

    open(metrics_path, "w") do io
        JSON3.write(io, metrics)
    end

    println(@sprintf("final test loss: %.4f", test_loss))
    println("tokenizer bundle: $(tokenizer_bundle_dir)")
    println("bundle export: $(bundle_dir)")
    println("checkpoint: $(final_checkpoint_path)")
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

function build_subword_lm_batches(
    texts::Vector{String},
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer;
    context_length::Int,
    batch_size::Int,
    document_separator::AbstractString,
)
    corpus = join(texts, document_separator)
    token_ids = KeemenaSubwords.encode(tokenizer, corpus; add_special_tokens = false)
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

    stats = (
        token_stream_length = length(token_ids),
        example_count = example_count,
    )
    return batches, stats
end

function mean_loss(model, batches)::Float64
    total_loss = 0.0
    for (input_batch, target_batch) in batches
        logits, _ = KeemenaLM.Core.lm_forward(model, input_batch; cache = nothing, is_training = false)
        total_loss += Float64(KeemenaLM.Core.causal_lm_cross_entropy(logits, target_batch))
    end
    return total_loss / length(batches)
end

function generate_samples(
    model,
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer,
    prompts::Vector{String};
    generation_seed::Int,
    max_new_tokens::Int,
)
    generation_config = GenerationConfig(
        max_new_tokens = max_new_tokens,
        temperature = 0.0,
        seed = generation_seed,
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

function split_text_stats(
    texts::Vector{String},
    context_length::Int,
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer;
    document_separator::AbstractString,
)::Dict{String, Any}
    token_stream_length = length(KeemenaSubwords.encode(tokenizer, join(texts, document_separator); add_special_tokens = false))
    example_count = fld(max(token_stream_length - 1, 0), context_length)
    return Dict(
        "paragraph_count" => length(texts),
        "token_stream_length" => token_stream_length,
        "example_count" => example_count,
        "mean_characters_per_paragraph" => sum(length, texts) / length(texts),
    )
end

function tiny_chatbot_evaluation_prompts()::Vector{String}
    return [
        "User: Hi there.\nAssistant:",
        "User: Can you help me plan a calm evening?\nAssistant:",
        "User: Can you rewrite this to sound kinder: 'You forgot again.'\nAssistant:",
        "User: I'm overwhelmed and I don't know where to start.\nAssistant:",
        "User: Can you give me three low-stress weekend ideas?\nAssistant:",
        "User: I need a short follow-up message after no reply yet.\nAssistant:",
        "User: I need a short follow-up message after no reply yet.\nAssistant: Sure. Try: 'Just checking in on this when you have a moment.'\n<END_ASSISTANT>\n<CHAT_END>\nUser: Can you make it warmer?\nAssistant:",
        "User: Can you help me think of two simple dinner ideas?\nAssistant:",
    ]
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

function subword_recipe_dict(dataset_dir::AbstractString, output_dir::AbstractString, settings::TinyChatbotSubwordSettings)
    return Dict(
        "experiment" => TINY_CHATBOT_SUBWORD_EXPERIMENT_NAME,
        "purpose" => TINY_CHATBOT_SUBWORD_PURPOSE,
        "dataset_dir" => abspath(dataset_dir),
        "output_dir" => abspath(output_dir),
        "backend" => "flux",
        "optimizer_name" => "Flux.Adam",
        "optimizer_hyperparameters" => Dict("learning_rate" => settings.learning_rate),
        "tokenizer_package" => "KeemenaSubwords.jl",
        "tokenizer_trainer" => String(settings.tokenizer_trainer),
        "tokenizer_vocab_size" => settings.tokenizer_vocab_size,
        "tokenizer_min_frequency" => settings.tokenizer_min_frequency,
        "tokenizer_model_name" => settings.tokenizer_model_name,
        "chat_marker_strategy" => "literal markers in text plus tokenizer added special tokens",
        "chat_marker_strings" => Dict(string(key) => value for (key, value) in settings.chat_special_tokens),
        "context_length" => settings.context_length,
        "num_layers" => settings.num_layers,
        "num_heads" => settings.num_heads,
        "embedding_size" => settings.embedding_size,
        "ffn_hidden_size" => settings.ffn_hidden_size,
        "batch_size" => settings.batch_size,
        "epochs" => settings.epochs,
        "sample_generation_tokens" => settings.sample_generation_tokens,
        "document_separator" => settings.document_separator,
    )
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
