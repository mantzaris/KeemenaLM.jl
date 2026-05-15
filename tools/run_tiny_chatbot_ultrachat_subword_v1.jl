#!/usr/bin/env julia

using Flux
using JSON3
using KeemenaLM
using KeemenaSubwords
using Printf
using Random

const TINY_CHATBOT_DATASET_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_ultrachat_corpus_v1")
const TINY_CHATBOT_SUBWORD_OUTPUT_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_ultrachat_subword_candidate_run_v1")
const TINY_CHATBOT_SUBWORD_EXPERIMENT_NAME = "tiny_chatbot_ultrachat_subword_candidate_run_v1"
const TINY_CHATBOT_SUBWORD_PURPOSE = "current UltraChat-based subword conversational chatbot candidate training run"
const TINY_CHATBOT_DOCUMENT_SEPARATOR = "\n\n"
const CHAT_MARKERS = (
    user = "User:",
    assistant = "Assistant:",
    end_assistant = "<END_ASSISTANT>",
    chat_end = "<CHAT_END>",
)
const CHAT_DECODING_STOP_SEQUENCES = String[
    CHAT_MARKERS.end_assistant,
    CHAT_MARKERS.chat_end,
    "\nUser:",
    "\nAssistant:",
    "\nSystem:",
]

Base.@kwdef struct TinyChatbotSubwordSettings
    model_seed::Int = 20260418
    generation_seed::Int = 20260419
    device::Symbol = :auto
    context_length::Int = 128
    batch_size::Int = 16
    epochs::Int = 2
    learning_rate::Float32 = 0.0003f0
    num_layers::Int = 8
    num_heads::Int = 8
    embedding_size::Int = 512
    ffn_hidden_size::Int = 2048
    sample_generation_tokens::Int = 200
    tokenizer_trainer::Symbol = :hf_gpt2_bytebpe
    tokenizer_vocab_size::Int = 8192
    tokenizer_min_frequency::Int = 2
    tokenizer_model_name::String = "tiny_chatbot_ultrachat_gpt2_bytebpe"
    tokenizer_training_text_limit::Int = 20000
    train_text_limit::Int = 50000
    validation_text_limit::Int = 1000
    test_text_limit::Int = 1000
    validation_batch_limit::Int = 0
    test_batch_limit::Int = 0
    log_every_steps::Int = 50
    checkpoint_every_steps::Int = 5000
    max_step_checkpoints::Int = 2
    max_epoch_checkpoints::Int = 2
    reuse_tokenizer_bundle_dir::String = ""
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

function updated_settings(settings::TinyChatbotSubwordSettings; kwargs...)
    options = Dict{Symbol,Any}(kwargs)
    return TinyChatbotSubwordSettings(
        model_seed = get(options, :model_seed, settings.model_seed),
        generation_seed = get(options, :generation_seed, settings.generation_seed),
        device = get(options, :device, settings.device),
        context_length = get(options, :context_length, settings.context_length),
        batch_size = get(options, :batch_size, settings.batch_size),
        epochs = get(options, :epochs, settings.epochs),
        learning_rate = get(options, :learning_rate, settings.learning_rate),
        num_layers = get(options, :num_layers, settings.num_layers),
        num_heads = get(options, :num_heads, settings.num_heads),
        embedding_size = get(options, :embedding_size, settings.embedding_size),
        ffn_hidden_size = get(options, :ffn_hidden_size, settings.ffn_hidden_size),
        sample_generation_tokens = get(options, :sample_generation_tokens, settings.sample_generation_tokens),
        tokenizer_trainer = get(options, :tokenizer_trainer, settings.tokenizer_trainer),
        tokenizer_vocab_size = get(options, :tokenizer_vocab_size, settings.tokenizer_vocab_size),
        tokenizer_min_frequency = get(options, :tokenizer_min_frequency, settings.tokenizer_min_frequency),
        tokenizer_model_name = get(options, :tokenizer_model_name, settings.tokenizer_model_name),
        tokenizer_training_text_limit = get(options, :tokenizer_training_text_limit, settings.tokenizer_training_text_limit),
        train_text_limit = get(options, :train_text_limit, settings.train_text_limit),
        validation_text_limit = get(options, :validation_text_limit, settings.validation_text_limit),
        test_text_limit = get(options, :test_text_limit, settings.test_text_limit),
        validation_batch_limit = get(options, :validation_batch_limit, settings.validation_batch_limit),
        test_batch_limit = get(options, :test_batch_limit, settings.test_batch_limit),
        log_every_steps = get(options, :log_every_steps, settings.log_every_steps),
        checkpoint_every_steps = get(options, :checkpoint_every_steps, settings.checkpoint_every_steps),
        max_step_checkpoints = get(options, :max_step_checkpoints, settings.max_step_checkpoints),
        max_epoch_checkpoints = get(options, :max_epoch_checkpoints, settings.max_epoch_checkpoints),
        reuse_tokenizer_bundle_dir = get(options, :reuse_tokenizer_bundle_dir, settings.reuse_tokenizer_bundle_dir),
        document_separator = get(options, :document_separator, settings.document_separator),
        chat_special_tokens = get(options, :chat_special_tokens, settings.chat_special_tokens),
    )
end

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
        elseif argument == "--device"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --device")
            settings = updated_settings(settings; device = parse_device(args[argument_index]))
        elseif argument == "--epochs"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --epochs")
            settings = updated_settings(settings; epochs = parse(Int, args[argument_index]))
        elseif argument == "--context-length"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --context-length")
            settings = updated_settings(settings; context_length = parse(Int, args[argument_index]))
        elseif argument == "--batch-size"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --batch-size")
            settings = updated_settings(settings; batch_size = parse(Int, args[argument_index]))
        elseif argument == "--learning-rate"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --learning-rate")
            settings = updated_settings(settings; learning_rate = parse(Float32, args[argument_index]))
        elseif argument == "--num-layers"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --num-layers")
            settings = updated_settings(settings; num_layers = parse(Int, args[argument_index]))
        elseif argument == "--num-heads"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --num-heads")
            settings = updated_settings(settings; num_heads = parse(Int, args[argument_index]))
        elseif argument == "--embedding-size"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --embedding-size")
            settings = updated_settings(settings; embedding_size = parse(Int, args[argument_index]))
        elseif argument == "--ffn-hidden-size"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --ffn-hidden-size")
            settings = updated_settings(settings; ffn_hidden_size = parse(Int, args[argument_index]))
        elseif argument == "--tokenizer-training-text-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --tokenizer-training-text-limit")
            settings = updated_settings(settings; tokenizer_training_text_limit = parse(Int, args[argument_index]))
        elseif argument == "--train-text-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --train-text-limit")
            settings = updated_settings(settings; train_text_limit = parse(Int, args[argument_index]))
        elseif argument == "--validation-text-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --validation-text-limit")
            settings = updated_settings(settings; validation_text_limit = parse(Int, args[argument_index]))
        elseif argument == "--test-text-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --test-text-limit")
            settings = updated_settings(settings; test_text_limit = parse(Int, args[argument_index]))
        elseif argument == "--validation-batch-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --validation-batch-limit")
            settings = updated_settings(settings; validation_batch_limit = parse(Int, args[argument_index]))
        elseif argument == "--test-batch-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --test-batch-limit")
            settings = updated_settings(settings; test_batch_limit = parse(Int, args[argument_index]))
        elseif argument == "--log-every-steps"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --log-every-steps")
            settings = updated_settings(settings; log_every_steps = parse(Int, args[argument_index]))
        elseif argument == "--checkpoint-every-steps"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --checkpoint-every-steps")
            settings = updated_settings(settings; checkpoint_every_steps = parse(Int, args[argument_index]))
        elseif argument == "--max-step-checkpoints"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --max-step-checkpoints")
            settings = updated_settings(settings; max_step_checkpoints = parse(Int, args[argument_index]))
        elseif argument == "--max-epoch-checkpoints"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --max-epoch-checkpoints")
            settings = updated_settings(settings; max_epoch_checkpoints = parse(Int, args[argument_index]))
        elseif argument == "--reuse-tokenizer-bundle-dir"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --reuse-tokenizer-bundle-dir")
            settings = updated_settings(settings; reuse_tokenizer_bundle_dir = abspath(args[argument_index]))
        else
            error("usage: julia --project=tools/subword_real_text tools/run_tiny_chatbot_ultrachat_subword_v1.jl [--dataset-dir DIR] [--output-dir DIR] [--device auto|cpu|gpu] [--epochs N] [--context-length N] [--batch-size N] [--learning-rate X] [--num-layers N] [--num-heads N] [--embedding-size N] [--ffn-hidden-size N] [--tokenizer-training-text-limit N] [--train-text-limit N] [--validation-text-limit N] [--test-text-limit N] [--validation-batch-limit N] [--test-batch-limit N] [--log-every-steps N] [--checkpoint-every-steps N] [--max-step-checkpoints N] [--max-epoch-checkpoints N] [--reuse-tokenizer-bundle-dir DIR]")
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
    validate_tiny_chatbot_settings(settings)

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
    progress_path = joinpath(output_dir, "progress.json")

    mkpath(output_dir)
    mkpath(checkpoint_dir)

    recipe = subword_recipe_dict(dataset_dir, output_dir, settings)
    write_json(joinpath(output_dir, "run_recipe.json"), recipe)

    split_texts = load_prepared_split_texts(dataset_dir; settings = settings)
    corpus_metadata = JSON3.read(read(metadata_path, String))

    tokenizer, tokenizer_training_summary, tokenizer_bundle_source = prepare_tokenizer_bundle(
        tokenizer_bundle_dir,
        split_texts.tokenizer_training,
        settings,
    )

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
    model = KeemenaLM.FluxBackend.move_model_to_device(
        instantiate(config; backend = :flux, seed = settings.model_seed);
        device = settings.device,
    )
    resolved_device = KeemenaLM.FluxBackend.is_cuda_array(model.token_embedding) ? :gpu : :cpu
    trainer = KeemenaLM.Core.Trainer(
        model;
        optimizer = Flux.Adam(settings.learning_rate),
        backend = :flux,
        metadata = Dict(
            "experiment" => TINY_CHATBOT_SUBWORD_EXPERIMENT_NAME,
            "corpus_source" => "tiny_chatbot_ultrachat_corpus_v1",
            "tokenizer_bundle_dir" => tokenizer_bundle_dir,
            "prepared_corpus_metadata_path" => metadata_path,
            "document_separator" => settings.document_separator,
            "optimizer_name" => "Flux.Adam",
            "requested_device" => String(settings.device),
            "resolved_device" => String(resolved_device),
            "chat_marker_representation" => "added special tokens present literally in training text",
        ),
    )

    validation_batches_for_eval = limit_batches(validation_batches, settings.validation_batch_limit)
    test_batches_for_eval = limit_batches(test_batches, settings.test_batch_limit)

    initial_validation_loss = mean_loss(model, validation_batches_for_eval)
    println("== Real conversational chatbot subword run ==")
    println("prepared_dataset_dir: $(dataset_dir)")
    println("output_dir: $(output_dir)")
    println("tokenizer_bundle_dir: $(tokenizer_bundle_dir)")
    println("requested_device: $(settings.device)")
    println("resolved_device: $(resolved_device)")
    println(@sprintf("initial validation loss: %.4f", initial_validation_loss))

    epoch_metrics = Dict{String, Any}[]
    step_checkpoint_metrics = Dict{String, Any}[]
    latest_checkpoint_path = nothing
    write_progress(
        progress_path;
        status = "running",
        epoch = 0,
        step = trainer.step,
        latest_train_loss = nothing,
        latest_validation_loss = initial_validation_loss,
        latest_checkpoint = nothing,
    )
    for epoch in 1:settings.epochs
        epoch_losses = Float64[]
        for (input_batch, target_batch) in train_batches
            step_result = KeemenaLM.Core.train_step!(trainer, input_batch, target_batch)
            push!(epoch_losses, step_result.loss)

            if settings.log_every_steps > 0 && trainer.step % settings.log_every_steps == 0
                recent_count = min(length(epoch_losses), settings.log_every_steps)
                recent_window = epoch_losses[(end - recent_count + 1):end]
                recent_train_loss = sum(recent_window) / length(recent_window)
                write_progress(
                    progress_path;
                    status = "running",
                    epoch = epoch,
                    step = trainer.step,
                    latest_train_loss = recent_train_loss,
                    latest_validation_loss = nothing,
                    latest_checkpoint = latest_checkpoint_path,
                )
                println(@sprintf(
                    "step %d  epoch %d/%d  recent_train_loss=%.4f",
                    trainer.step,
                    epoch,
                    settings.epochs,
                    recent_train_loss,
                ))
            end

            if settings.checkpoint_every_steps > 0 && trainer.step % settings.checkpoint_every_steps == 0
                step_validation_loss = mean_loss(model, validation_batches_for_eval)
                step_checkpoint_path = joinpath(checkpoint_dir, @sprintf("step_%06d_checkpoint.jld2", trainer.step))
                save_checkpoint(
                    step_checkpoint_path,
                    trainer,
                    model;
                    experiment = TINY_CHATBOT_SUBWORD_EXPERIMENT_NAME,
                    epoch = epoch,
                    step = trainer.step,
                    stage = "step_interval",
                )
                push!(
                    step_checkpoint_metrics,
                    Dict(
                        "epoch" => epoch,
                        "step" => trainer.step,
                        "train_loss_running_epoch_mean" => sum(epoch_losses) / length(epoch_losses),
                        "validation_loss" => step_validation_loss,
                        "validation_perplexity" => exp(step_validation_loss),
                        "checkpoint_path" => step_checkpoint_path,
                    ),
                )
                latest_checkpoint_path = step_checkpoint_path
                deleted_step_checkpoints = prune_matching_checkpoints!(
                    checkpoint_dir,
                    r"^step_.*_checkpoint\.jld2$",
                    settings.max_step_checkpoints;
                    keep_path = step_checkpoint_path,
                )
                write_progress(
                    progress_path;
                    status = "running",
                    epoch = epoch,
                    step = trainer.step,
                    latest_train_loss = sum(epoch_losses) / length(epoch_losses),
                    latest_validation_loss = step_validation_loss,
                    latest_checkpoint = step_checkpoint_path,
                )
                println(@sprintf(
                    "step checkpoint  epoch %d/%d  step=%d  validation_loss=%.4f  checkpoint=%s",
                    epoch,
                    settings.epochs,
                    trainer.step,
                    step_validation_loss,
                    step_checkpoint_path,
                ))
                if !isempty(deleted_step_checkpoints)
                    println("pruned old step checkpoints: $(length(deleted_step_checkpoints))")
                end
            end
        end

        trainer.epoch = epoch
        train_loss = sum(epoch_losses) / length(epoch_losses)
        validation_loss = mean_loss(model, validation_batches_for_eval)
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
        deleted_epoch_checkpoints = prune_matching_checkpoints!(
            checkpoint_dir,
            r"^epoch_.*_checkpoint\.jld2$",
            settings.max_epoch_checkpoints;
            keep_path = checkpoint_path,
        )
        if !isempty(deleted_epoch_checkpoints)
            println("pruned old epoch checkpoints: $(length(deleted_epoch_checkpoints))")
        end
        latest_checkpoint_path = checkpoint_path
        write_progress(
            progress_path;
            status = "running",
            epoch = epoch,
            step = trainer.step,
            latest_train_loss = train_loss,
            latest_validation_loss = validation_loss,
            latest_checkpoint = checkpoint_path,
        )
    end

    final_checkpoint_path = joinpath(checkpoint_dir, "final_checkpoint.jld2")
    save_checkpoint(final_checkpoint_path, trainer, model; experiment = TINY_CHATBOT_SUBWORD_EXPERIMENT_NAME, stage = "final")

    bundle = Bundle(
        model_config = KeemenaLM.Core.model_config(model),
        weights = KeemenaLM.Core.extract_weights(model),
    )
    save_bundle(bundle_dir, bundle)

    reloaded_bundle = load_bundle(bundle_dir)
    reloaded_model = KeemenaLM.FluxBackend.move_model_to_device(
        instantiate(reloaded_bundle; backend = :flux);
        device = settings.device,
    )
    reloaded_tokenizer = KeemenaSubwords.load_training_bundle(tokenizer_bundle_dir)

    prompts = tiny_chatbot_evaluation_prompts()
    write_prompt_files(output_dir, prompts)

    test_loss = mean_loss(reloaded_model, test_batches_for_eval)
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
            "source_type" => "tiny_chatbot_ultrachat_corpus_v1",
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
            "tokenizer_training_text_limit" => settings.tokenizer_training_text_limit,
            "train_text_limit" => settings.train_text_limit,
            "validation_text_limit" => settings.validation_text_limit,
            "test_text_limit" => settings.test_text_limit,
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
                "trainer" => tokenizer_training_summary["trainer"],
                "tokenizer_type" => tokenizer_training_summary["tokenizer_type"],
                "config_type" => tokenizer_training_summary["config_type"],
                "model_name" => tokenizer_training_summary["model_name"],
                "version" => tokenizer_training_summary["version"],
                "vocab_size" => tokenizer_training_summary["vocab_size"],
            ),
            "bundle_source" => tokenizer_bundle_source,
            "bundle_persistence_note" => "Tokenizer bundle is saved separately; KeemenaLM model bundles still do not persist tokenizer payloads.",
        ),
        "model" => Dict(
            "backend" => "flux",
            "requested_device" => String(settings.device),
            "resolved_device" => String(resolved_device),
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
            "validation_batches_for_eval" => length(validation_batches_for_eval),
            "test_batches_for_eval" => length(test_batches_for_eval),
            "log_every_steps" => settings.log_every_steps,
            "checkpoint_every_steps" => settings.checkpoint_every_steps,
            "max_step_checkpoints" => settings.max_step_checkpoints,
            "max_epoch_checkpoints" => settings.max_epoch_checkpoints,
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
            "progress_path" => progress_path,
        ),
        "epoch_metrics" => epoch_metrics,
        "step_checkpoint_metrics" => step_checkpoint_metrics,
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
    write_progress(
        progress_path;
        status = "completed",
        epoch = trainer.epoch,
        step = trainer.step,
        latest_train_loss = isempty(epoch_metrics) ? nothing : epoch_metrics[end]["train_loss"],
        latest_validation_loss = isempty(epoch_metrics) ? initial_validation_loss : epoch_metrics[end]["validation_loss"],
        latest_checkpoint = final_checkpoint_path,
    )

    println(@sprintf("final test loss: %.4f", test_loss))
    println("tokenizer bundle: $(tokenizer_bundle_dir)")
    println("bundle export: $(bundle_dir)")
    println("checkpoint: $(final_checkpoint_path)")
    println("metrics: $(metrics_path)")
    println("samples: $(sample_path)")
    return metrics
end

function validate_tiny_chatbot_settings(settings::TinyChatbotSubwordSettings)
    settings.device in (:auto, :cpu, :gpu) ||
        throw(ArgumentError("device must be one of :auto, :cpu, or :gpu"))
    settings.checkpoint_every_steps >= 0 ||
        throw(ArgumentError("checkpoint_every_steps must be >= 0"))
    settings.max_step_checkpoints >= 1 ||
        throw(ArgumentError("max_step_checkpoints must be >= 1"))
    settings.max_epoch_checkpoints >= 1 ||
        throw(ArgumentError("max_epoch_checkpoints must be >= 1"))
    return settings
end

function parse_device(value::AbstractString)::Symbol
    normalized_value = lowercase(strip(value))
    if normalized_value == "auto"
        return :auto
    elseif normalized_value == "cpu"
        return :cpu
    elseif normalized_value == "gpu"
        return :gpu
    else
        throw(ArgumentError("--device must be one of auto, cpu, or gpu"))
    end
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

function load_prepared_split_texts(dataset_dir::AbstractString; settings::TinyChatbotSubwordSettings)
    return (
        tokenizer_training = read_prepared_paragraphs(
            joinpath(dataset_dir, "training.txt");
            limit = settings.tokenizer_training_text_limit,
        ),
        training = read_prepared_paragraphs(
            joinpath(dataset_dir, "training.txt");
            limit = settings.train_text_limit,
        ),
        validation = read_prepared_paragraphs(
            joinpath(dataset_dir, "validation.txt");
            limit = settings.validation_text_limit,
        ),
        testing = read_prepared_paragraphs(
            joinpath(dataset_dir, "testing.txt");
            limit = settings.test_text_limit,
        ),
    )
end

function read_prepared_paragraphs(path::AbstractString; limit::Union{Nothing,Int} = nothing)::Vector{String}
    paragraphs = String[]
    buffer = IOBuffer()

    open(path, "r") do io
        while !eof(io)
            line = readline(io; keep = true)
            if isempty(strip(line))
                paragraph = strip(String(take!(buffer)))
                if !isempty(paragraph)
                    push!(paragraphs, paragraph)
                    limit !== nothing && length(paragraphs) >= limit && break
                end
            else
                write(buffer, line)
            end
        end
    end

    if (limit === nothing || length(paragraphs) < limit) && position(buffer) > 0
        paragraph = strip(String(take!(buffer)))
        isempty(paragraph) || push!(paragraphs, paragraph)
    end

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

    batches = Tuple{Matrix{Int32}, Matrix{Int32}}[]
    for batch_start in 1:batch_size:example_count
        batch_end = min(batch_start + batch_size - 1, example_count)
        actual_batch_size = batch_end - batch_start + 1
        input_batch = Matrix{Int32}(undef, context_length, actual_batch_size)
        target_batch = Matrix{Int32}(undef, context_length, actual_batch_size)

        for (column_index, example_index) in enumerate(batch_start:batch_end)
            offset = (example_index - 1) * context_length
            input_batch[:, column_index] = Int32.(token_ids[(offset + 1):(offset + context_length)])
            target_batch[:, column_index] = Int32.(token_ids[(offset + 2):(offset + context_length + 1)])
        end

        push!(batches, (input_batch, target_batch))
    end

    stats = (
        token_stream_length = length(token_ids),
        example_count = example_count,
    )
    return batches, stats
end

function limit_batches(batches, limit::Int)
    (limit <= 0 || limit >= length(batches)) && return batches
    return batches[1:limit]
end

function prepare_tokenizer_bundle(
    tokenizer_bundle_dir::AbstractString,
    tokenizer_training_texts::Vector{String},
    settings::TinyChatbotSubwordSettings,
)
    bundle_source = "trained_new_bundle"
    requested_reuse_dir = settings.reuse_tokenizer_bundle_dir
    reuse_dir = isempty(requested_reuse_dir) ? tokenizer_bundle_dir : requested_reuse_dir
    tokenizer_json_path = joinpath(reuse_dir, "tokenizer.json")
    manifest_path = joinpath(reuse_dir, "keemena_training_manifest.json")

    if isfile(tokenizer_json_path) && isfile(manifest_path)
        tokenizer = KeemenaSubwords.load_training_bundle(reuse_dir)
        summary = Dict(
            "trainer" => String(settings.tokenizer_trainer),
            "tokenizer_type" => "HuggingFaceJSONTokenizer",
            "config_type" => "reused_bundle",
            "model_name" => settings.tokenizer_model_name,
            "version" => "reused",
            "vocab_size" => KeemenaSubwords.vocab_size(tokenizer),
        )
        bundle_source = abspath(reuse_dir)
        if abspath(reuse_dir) != abspath(tokenizer_bundle_dir)
            mkpath(tokenizer_bundle_dir)
            cp(tokenizer_json_path, joinpath(tokenizer_bundle_dir, "tokenizer.json"); force = true)
            cp(manifest_path, joinpath(tokenizer_bundle_dir, "keemena_training_manifest.json"); force = true)
        end
        return tokenizer, summary, bundle_source
    end

    training_texts = [text * settings.document_separator for text in tokenizer_training_texts]
    Random.seed!(settings.model_seed)
    tokenizer_training_output = KeemenaSubwords.quick_train_bundle(
        settings.tokenizer_trainer,
        training_texts;
        bundle_directory = tokenizer_bundle_dir,
        overwrite = true,
        export_format = :hf_tokenizer_json,
        vocab_size = settings.tokenizer_vocab_size,
        min_frequency = settings.tokenizer_min_frequency,
        model_name = settings.tokenizer_model_name,
        special_tokens = settings.chat_special_tokens,
        sanity_text = "User: Hi there.\nAssistant: Hello.\n<END_ASSISTANT>\n<CHAT_END>",
    )
    summary = Dict(
        "trainer" => String(tokenizer_training_output.training_summary.trainer),
        "tokenizer_type" => tokenizer_training_output.training_summary.tokenizer_type,
        "config_type" => tokenizer_training_output.training_summary.config_type,
        "model_name" => tokenizer_training_output.training_summary.model_name,
        "version" => tokenizer_training_output.training_summary.version,
        "vocab_size" => tokenizer_training_output.training_summary.vocab_size,
    )
    return tokenizer_training_output.tokenizer, summary, bundle_source
end

function mean_loss(model, batches)::Float64
    total_loss = 0.0
    for (input_batch, target_batch) in batches
        logits, _ = KeemenaLM.Core.lm_forward(model, input_batch; cache = nothing, is_training = false)
        loss_target_batch = KeemenaLM.FluxBackend.move_like(target_batch, model.token_embedding)
        total_loss += Float64(KeemenaLM.Core.causal_lm_cross_entropy(logits, loss_target_batch))
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
        stop_sequences = copy(CHAT_DECODING_STOP_SEQUENCES),
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
        "User: Can you help me plan a calm evening after a long day?\nAssistant:",
        "User: Can you rewrite this to sound kinder: 'You forgot again.'\nAssistant:",
        "User: I'm overwhelmed and I don't know where to start.\nAssistant:",
        "User: Can you give me three low-stress weekend ideas?\nAssistant:",
        "User: I need a short follow-up message after no reply yet.\nAssistant:",
        "User: I need a short follow-up message after no reply yet.\nAssistant: Sure. Try: 'Just checking in on this when you have a moment.'\n<END_ASSISTANT>\n<CHAT_END>\nUser: Can you make it warmer?\nAssistant:",
        "User: Can you help me think of two simple dinner ideas?\nAssistant:",
        "User: Can you help me brainstorm a simple birthday plan for a friend?\nAssistant:",
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
        "device" => String(settings.device),
        "optimizer_name" => "Flux.Adam",
        "optimizer_hyperparameters" => Dict("learning_rate" => settings.learning_rate),
        "tokenizer_package" => "KeemenaSubwords.jl",
        "tokenizer_trainer" => String(settings.tokenizer_trainer),
        "tokenizer_vocab_size" => settings.tokenizer_vocab_size,
        "tokenizer_min_frequency" => settings.tokenizer_min_frequency,
        "tokenizer_model_name" => settings.tokenizer_model_name,
        "tokenizer_training_text_limit" => settings.tokenizer_training_text_limit,
        "reuse_tokenizer_bundle_dir" => settings.reuse_tokenizer_bundle_dir,
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
        "train_text_limit" => settings.train_text_limit,
        "validation_text_limit" => settings.validation_text_limit,
        "test_text_limit" => settings.test_text_limit,
        "validation_batch_limit" => settings.validation_batch_limit,
        "test_batch_limit" => settings.test_batch_limit,
        "log_every_steps" => settings.log_every_steps,
        "checkpoint_every_steps" => settings.checkpoint_every_steps,
        "max_step_checkpoints" => settings.max_step_checkpoints,
        "max_epoch_checkpoints" => settings.max_epoch_checkpoints,
        "document_separator" => settings.document_separator,
    )
end

function prune_matching_checkpoints!(
    checkpoint_dir::AbstractString,
    filename_pattern::Regex,
    retention_limit::Int;
    keep_path::Union{Nothing,AbstractString} = nothing,
)::Vector{String}
    retention_limit <= 0 && return String[]
    isdir(checkpoint_dir) || return String[]

    checkpoint_paths = String[]
    for filename in readdir(checkpoint_dir)
        occursin(filename_pattern, filename) || continue
        checkpoint_path = joinpath(checkpoint_dir, filename)
        isfile(checkpoint_path) || continue
        push!(checkpoint_paths, checkpoint_path)
    end

    length(checkpoint_paths) <= retention_limit && return String[]

    keep_absolute_path = keep_path === nothing ? nothing : abspath(keep_path)
    sort!(checkpoint_paths; by = checkpoint_path -> stat(checkpoint_path).mtime)

    deleted_paths = String[]
    delete_count = length(checkpoint_paths) - retention_limit
    for checkpoint_path in checkpoint_paths
        delete_count <= 0 && break
        if keep_absolute_path !== nothing && abspath(checkpoint_path) == keep_absolute_path
            continue
        end

        rm(checkpoint_path; force = true)
        push!(deleted_paths, checkpoint_path)
        delete_count -= 1
    end

    return deleted_paths
end

function write_progress(
    path::AbstractString;
    status::AbstractString,
    epoch,
    step,
    latest_train_loss,
    latest_validation_loss,
    latest_checkpoint,
)
    return write_json(path, Dict(
        "status" => status,
        "latest_epoch" => epoch,
        "latest_step" => step,
        "latest_train_loss" => latest_train_loss,
        "latest_validation_loss" => latest_validation_loss,
        "latest_checkpoint" => latest_checkpoint,
        "updated_at_unix" => time(),
    ))
end

function write_json(path::AbstractString, value)
    open(path, "w") do io
        JSON3.write(io, value)
    end
    return path
end

function command_allows_gpu_device(args)::Bool
    device = :auto
    argument_index = 1
    while argument_index <= length(args)
        if args[argument_index] == "--device"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --device")
            device = parse_device(args[argument_index])
        end
        argument_index += 1
    end
    return device !== :cpu
end

if abspath(PROGRAM_FILE) == @__FILE__
    command_allows_gpu_device(ARGS) && KeemenaLM.FluxBackend.has_functional_cuda_gpu()
    Base.invokelatest(main, ARGS)
end
