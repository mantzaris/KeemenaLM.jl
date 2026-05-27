#!/usr/bin/env julia

include(joinpath(@__DIR__, "run_tiny_chatbot_ultrachat_subword_v1.jl"))

using Flux
using JSON3
using KeemenaLM
using KeemenaSubwords
using Printf
using Random

const CLEAN_SFT_V4_DATASET_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_clean_sft_corpus_v1")
const CLEAN_SFT_V4_OUTPUT_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_clean_sft_candidate_run_v4")
const CLEAN_SFT_V4_EXPERIMENT_NAME = "tiny_chatbot_clean_sft_candidate_run_v4"
const CLEAN_SFT_V4_PURPOSE = "v4 clean short-answer SFT chatbot run with example-based assistant-only batches"

Base.@kwdef struct CleanSFTV4Settings
    model_seed::Int = 20260518
    generation_seed::Int = 20260519
    device::Symbol = :auto
    context_length::Int = 256
    batch_size::Int = 4
    epochs::Int = 3
    learning_rate::Float32 = 0.0002f0
    num_layers::Int = 12
    num_heads::Int = 12
    embedding_size::Int = 768
    ffn_hidden_size::Int = 3072
    sample_generation_tokens::Int = 180
    tokenizer_trainer::Symbol = :hf_gpt2_bytebpe
    tokenizer_vocab_size::Int = 16_384
    tokenizer_min_frequency::Int = 2
    tokenizer_model_name::String = "tiny_chatbot_clean_sft_v4_gpt2_bytebpe"
    tokenizer_training_example_limit::Int = 50_000
    train_example_limit::Int = 0
    validation_example_limit::Int = 2_000
    test_example_limit::Int = 2_000
    validation_batch_limit::Int = 0
    test_batch_limit::Int = 0
    log_every_steps::Int = 25
    checkpoint_every_steps::Int = 0
    max_step_checkpoints::Int = 1
    max_epoch_checkpoints::Int = 1
    audit_example_count::Int = 8
    shuffle_batches::Bool = true
    reuse_tokenizer_bundle_dir::String = ""
    chat_special_tokens::Dict{Symbol,String} = Dict(
        :unk => "<|endoftext|>",
        :user => CHAT_MARKERS.user,
        :assistant => CHAT_MARKERS.assistant,
        :end_assistant => CHAT_MARKERS.end_assistant,
        :chat_end => CHAT_MARKERS.chat_end,
    )
end

struct CleanSFTExample
    id::String
    prompt_text::String
    target_text::String
    assistant_text::String
    chat_text::String
end

function main_v4(args)
    dataset_dir = CLEAN_SFT_V4_DATASET_DIR
    output_dir = CLEAN_SFT_V4_OUTPUT_DIR
    settings = CleanSFTV4Settings()

    argument_index = 1
    while argument_index <= length(args)
        argument = args[argument_index]
        if argument in ("--help", "-h")
            print_v4_usage()
            return nothing
        elseif argument == "--dataset-dir"
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
            settings = update_v4_settings(settings; device = parse_device(args[argument_index]))
        elseif argument == "--epochs"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --epochs")
            settings = update_v4_settings(settings; epochs = parse(Int, args[argument_index]))
        elseif argument == "--context-length"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --context-length")
            settings = update_v4_settings(settings; context_length = parse(Int, args[argument_index]))
        elseif argument == "--batch-size"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --batch-size")
            settings = update_v4_settings(settings; batch_size = parse(Int, args[argument_index]))
        elseif argument == "--learning-rate"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --learning-rate")
            settings = update_v4_settings(settings; learning_rate = parse(Float32, args[argument_index]))
        elseif argument == "--num-layers"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --num-layers")
            settings = update_v4_settings(settings; num_layers = parse(Int, args[argument_index]))
        elseif argument == "--num-heads"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --num-heads")
            settings = update_v4_settings(settings; num_heads = parse(Int, args[argument_index]))
        elseif argument == "--embedding-size"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --embedding-size")
            settings = update_v4_settings(settings; embedding_size = parse(Int, args[argument_index]))
        elseif argument == "--ffn-hidden-size"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --ffn-hidden-size")
            settings = update_v4_settings(settings; ffn_hidden_size = parse(Int, args[argument_index]))
        elseif argument == "--tokenizer-vocab-size"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --tokenizer-vocab-size")
            settings = update_v4_settings(settings; tokenizer_vocab_size = parse(Int, args[argument_index]))
        elseif argument == "--tokenizer-training-example-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --tokenizer-training-example-limit")
            settings = update_v4_settings(settings; tokenizer_training_example_limit = parse(Int, args[argument_index]))
        elseif argument == "--train-example-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --train-example-limit")
            settings = update_v4_settings(settings; train_example_limit = parse(Int, args[argument_index]))
        elseif argument == "--validation-example-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --validation-example-limit")
            settings = update_v4_settings(settings; validation_example_limit = parse(Int, args[argument_index]))
        elseif argument == "--test-example-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --test-example-limit")
            settings = update_v4_settings(settings; test_example_limit = parse(Int, args[argument_index]))
        elseif argument == "--validation-batch-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --validation-batch-limit")
            settings = update_v4_settings(settings; validation_batch_limit = parse(Int, args[argument_index]))
        elseif argument == "--test-batch-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --test-batch-limit")
            settings = update_v4_settings(settings; test_batch_limit = parse(Int, args[argument_index]))
        elseif argument == "--log-every-steps"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --log-every-steps")
            settings = update_v4_settings(settings; log_every_steps = parse(Int, args[argument_index]))
        elseif argument == "--checkpoint-every-steps"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --checkpoint-every-steps")
            settings = update_v4_settings(settings; checkpoint_every_steps = parse(Int, args[argument_index]))
        elseif argument == "--max-step-checkpoints"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --max-step-checkpoints")
            settings = update_v4_settings(settings; max_step_checkpoints = parse(Int, args[argument_index]))
        elseif argument == "--max-epoch-checkpoints"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --max-epoch-checkpoints")
            settings = update_v4_settings(settings; max_epoch_checkpoints = parse(Int, args[argument_index]))
        elseif argument == "--audit-example-count"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --audit-example-count")
            settings = update_v4_settings(settings; audit_example_count = parse(Int, args[argument_index]))
        elseif argument == "--reuse-tokenizer-bundle-dir"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --reuse-tokenizer-bundle-dir")
            settings = update_v4_settings(settings; reuse_tokenizer_bundle_dir = abspath(args[argument_index]))
        elseif argument == "--no-shuffle-batches"
            settings = update_v4_settings(settings; shuffle_batches = false)
        else
            error("unknown argument $(argument). Run with --help for usage.")
        end
        argument_index += 1
    end

    return run_clean_sft_v4(dataset_dir, output_dir; settings = settings)
end

function print_v4_usage()
    println("""
usage: julia --project=tools/subword_real_text tools/run_tiny_chatbot_clean_sft_v4.jl [options]

Required before training:
  python tools/prepare_tiny_chatbot_clean_sft_corpus.py --output-dir tmp/tiny_chatbot_clean_sft_corpus_v1

Important options:
  --dataset-dir DIR                         Clean SFT corpus directory. Defaults to tmp/tiny_chatbot_clean_sft_corpus_v1.
  --output-dir DIR                          Run output directory. Defaults to tmp/tiny_chatbot_clean_sft_candidate_run_v4.
  --device auto|cpu|gpu                     Defaults to auto.
  --context-length N                        Defaults to 256.
  --batch-size N                            Defaults to 4.
  --epochs N                                Defaults to 3.
  --learning-rate X                         Defaults to 0.0002.
  --num-layers N                            Defaults to 12.
  --num-heads N                             Defaults to 12.
  --embedding-size N                        Defaults to 768.
  --ffn-hidden-size N                       Defaults to 3072.
  --tokenizer-vocab-size N                  Defaults to 16384.
  --checkpoint-every-steps N                Defaults to 0, so only epoch/final checkpoints are written.
  --max-epoch-checkpoints N                 Defaults to 1.
  --reuse-tokenizer-bundle-dir DIR          Optional freshly generated tokenizer bundle to reuse.
""")
end

function update_v4_settings(settings::CleanSFTV4Settings; kwargs...)
    options = Dict{Symbol,Any}(kwargs)
    return CleanSFTV4Settings(
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
        tokenizer_training_example_limit = get(options, :tokenizer_training_example_limit, settings.tokenizer_training_example_limit),
        train_example_limit = get(options, :train_example_limit, settings.train_example_limit),
        validation_example_limit = get(options, :validation_example_limit, settings.validation_example_limit),
        test_example_limit = get(options, :test_example_limit, settings.test_example_limit),
        validation_batch_limit = get(options, :validation_batch_limit, settings.validation_batch_limit),
        test_batch_limit = get(options, :test_batch_limit, settings.test_batch_limit),
        log_every_steps = get(options, :log_every_steps, settings.log_every_steps),
        checkpoint_every_steps = get(options, :checkpoint_every_steps, settings.checkpoint_every_steps),
        max_step_checkpoints = get(options, :max_step_checkpoints, settings.max_step_checkpoints),
        max_epoch_checkpoints = get(options, :max_epoch_checkpoints, settings.max_epoch_checkpoints),
        audit_example_count = get(options, :audit_example_count, settings.audit_example_count),
        shuffle_batches = get(options, :shuffle_batches, settings.shuffle_batches),
        reuse_tokenizer_bundle_dir = get(options, :reuse_tokenizer_bundle_dir, settings.reuse_tokenizer_bundle_dir),
        chat_special_tokens = get(options, :chat_special_tokens, settings.chat_special_tokens),
    )
end

function run_clean_sft_v4(
    dataset_dir::AbstractString,
    output_dir::AbstractString;
    settings::CleanSFTV4Settings = CleanSFTV4Settings(),
)
    validate_v4_settings(settings)

    training_jsonl_path = joinpath(dataset_dir, "training.jsonl")
    validation_jsonl_path = joinpath(dataset_dir, "validation.jsonl")
    testing_jsonl_path = joinpath(dataset_dir, "testing.jsonl")

    isdir(dataset_dir) || throw(ArgumentError(
        "clean SFT dataset directory does not exist: $(dataset_dir). " *
        "Prepare it first with: python3 tools/prepare_tiny_chatbot_clean_sft_corpus.py --output-dir tmp/tiny_chatbot_clean_sft_corpus_v1",
    ))
    isfile(training_jsonl_path) || throw(ArgumentError("clean SFT training split does not exist: $(training_jsonl_path)"))
    isfile(validation_jsonl_path) || throw(ArgumentError("clean SFT validation split does not exist: $(validation_jsonl_path)"))
    isfile(testing_jsonl_path) || throw(ArgumentError("clean SFT testing split does not exist: $(testing_jsonl_path)"))
    metadata_path = resolve_clean_sft_metadata_path(dataset_dir)

    output_dir = abspath(output_dir)
    checkpoint_dir = joinpath(output_dir, "checkpoints")
    bundle_dir = joinpath(output_dir, "bundle")
    tokenizer_bundle_dir = joinpath(output_dir, "tokenizer_bundle")
    metrics_path = joinpath(output_dir, "metrics.json")
    sample_path = joinpath(output_dir, "sample_outputs.txt")
    progress_path = joinpath(output_dir, "progress.json")
    audit_path = joinpath(output_dir, "data_mask_audit.txt")

    mkpath(output_dir)
    mkpath(checkpoint_dir)
    write_json(joinpath(output_dir, "run_recipe.json"), clean_sft_v4_recipe_dict(dataset_dir, output_dir, settings))

    training_examples = read_clean_sft_examples(training_jsonl_path; limit = settings.train_example_limit)
    validation_examples = read_clean_sft_examples(validation_jsonl_path; limit = settings.validation_example_limit)
    testing_examples = read_clean_sft_examples(testing_jsonl_path; limit = settings.test_example_limit)
    corpus_metadata = JSON3.read(read(metadata_path, String))

    tokenizer_training_examples = limit_examples(training_examples, settings.tokenizer_training_example_limit)
    tokenizer_settings = TinyChatbotSubwordSettings(
        model_seed = settings.model_seed,
        generation_seed = settings.generation_seed,
        device = settings.device,
        loss_mode = :assistant_only,
        context_length = settings.context_length,
        batch_size = settings.batch_size,
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
        tokenizer_training_text_limit = length(tokenizer_training_examples),
        train_text_limit = length(training_examples),
        validation_text_limit = length(validation_examples),
        test_text_limit = length(testing_examples),
        validation_batch_limit = settings.validation_batch_limit,
        test_batch_limit = settings.test_batch_limit,
        log_every_steps = settings.log_every_steps,
        checkpoint_every_steps = settings.checkpoint_every_steps,
        max_step_checkpoints = settings.max_step_checkpoints,
        max_epoch_checkpoints = settings.max_epoch_checkpoints,
        reuse_tokenizer_bundle_dir = settings.reuse_tokenizer_bundle_dir,
        document_separator = TINY_CHATBOT_DOCUMENT_SEPARATOR,
        chat_special_tokens = settings.chat_special_tokens,
    )
    tokenizer, tokenizer_training_summary, tokenizer_bundle_source = prepare_tokenizer_bundle(
        tokenizer_bundle_dir,
        [example.chat_text for example in tokenizer_training_examples],
        tokenizer_settings,
    )

    marker_token_ids = Dict(
        string(key) => KeemenaSubwords.token_to_id(tokenizer, token_string) for (key, token_string) in settings.chat_special_tokens if key != :unk
    )
    pad_token_id = KeemenaSubwords.token_to_id(tokenizer, settings.chat_special_tokens[:unk])

    train_batches, train_stats = build_clean_sft_batches(
        training_examples,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        pad_token_id = pad_token_id,
    )
    validation_batches, validation_stats = build_clean_sft_batches(
        validation_examples,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        pad_token_id = pad_token_id,
    )
    test_batches, test_stats = build_clean_sft_batches(
        testing_examples,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        pad_token_id = pad_token_id,
    )

    write_clean_sft_audit(
        audit_path,
        training_examples,
        tokenizer;
        context_length = settings.context_length,
        pad_token_id = pad_token_id,
        audit_example_count = settings.audit_example_count,
        seed = settings.model_seed,
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
            "experiment" => CLEAN_SFT_V4_EXPERIMENT_NAME,
            "corpus_source" => "tiny_chatbot_clean_sft_corpus_v1",
            "tokenizer_bundle_dir" => tokenizer_bundle_dir,
            "prepared_corpus_metadata_path" => metadata_path,
            "optimizer_name" => "Flux.Adam",
            "requested_device" => String(settings.device),
            "resolved_device" => String(resolved_device),
            "loss_mode" => "assistant_only",
            "batching" => "example_based_padded_sft",
            "chat_marker_representation" => "added special tokens present literally in target text",
        ),
    )

    validation_batches_for_eval = limit_batches(validation_batches, settings.validation_batch_limit)
    test_batches_for_eval = limit_batches(test_batches, settings.test_batch_limit)

    initial_validation_loss = masked_mean_loss(model, validation_batches_for_eval)
    println("== Clean SFT v4 chatbot run ==")
    println("prepared_dataset_dir: $(dataset_dir)")
    println("output_dir: $(output_dir)")
    println("tokenizer_bundle_dir: $(tokenizer_bundle_dir)")
    println("requested_device: $(settings.device)")
    println("resolved_device: $(resolved_device)")
    println("model_shape: layers=$(settings.num_layers) heads=$(settings.num_heads) emb=$(settings.embedding_size) ffn=$(settings.ffn_hidden_size) context=$(settings.context_length)")
    println("batching: example_based_padded_sft")
    println(@sprintf("initial validation loss: %.4f", initial_validation_loss))
    println("audit: $(audit_path)")

    epoch_metrics = Dict{String,Any}[]
    step_checkpoint_metrics = Dict{String,Any}[]
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

    batch_rng = Random.MersenneTwister(settings.model_seed + 11)
    for epoch in 1:settings.epochs
        epoch_losses = Float64[]
        epoch_batches = settings.shuffle_batches ? Random.shuffle(batch_rng, train_batches) : train_batches
        for (input_batch, target_batch, loss_mask_batch) in epoch_batches
            step_result = KeemenaLM.Core.train_step!(trainer, input_batch, target_batch, loss_mask_batch)
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
                step_validation_loss = masked_mean_loss(model, validation_batches_for_eval)
                step_checkpoint_path = joinpath(checkpoint_dir, @sprintf("step_%06d_checkpoint.jld2", trainer.step))
                save_checkpoint(
                    step_checkpoint_path,
                    trainer,
                    model;
                    experiment = CLEAN_SFT_V4_EXPERIMENT_NAME,
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
                prune_matching_checkpoints!(
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
            end
        end

        trainer.epoch = epoch
        train_loss = sum(epoch_losses) / length(epoch_losses)
        validation_loss = masked_mean_loss(model, validation_batches_for_eval)
        checkpoint_path = joinpath(checkpoint_dir, @sprintf("epoch_%02d_checkpoint.jld2", epoch))
        save_checkpoint(checkpoint_path, trainer, model; experiment = CLEAN_SFT_V4_EXPERIMENT_NAME, epoch = epoch)
        prune_matching_checkpoints!(
            checkpoint_dir,
            r"^epoch_.*_checkpoint\.jld2$",
            settings.max_epoch_checkpoints;
            keep_path = checkpoint_path,
        )

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
    save_checkpoint(final_checkpoint_path, trainer, model; experiment = CLEAN_SFT_V4_EXPERIMENT_NAME, stage = "final")

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

    test_loss = masked_mean_loss(reloaded_model, test_batches_for_eval)
    samples = generate_samples(
        reloaded_model,
        reloaded_tokenizer,
        prompts;
        generation_seed = settings.generation_seed,
        max_new_tokens = settings.sample_generation_tokens,
    )
    write_samples(sample_path, samples)

    metrics = Dict(
        "experiment" => CLEAN_SFT_V4_EXPERIMENT_NAME,
        "purpose" => CLEAN_SFT_V4_PURPOSE,
        "comparison_note" => "This run uses clean example-based SFT batches, so losses are not directly comparable to the previous flattened UltraChat stream runs.",
        "corpus" => Dict(
            "source_type" => "tiny_chatbot_clean_sft_corpus_v1",
            "prepared_dataset_dir" => dataset_dir,
            "corpus_metadata_file" => metadata_path,
            "training_jsonl" => training_jsonl_path,
            "validation_jsonl" => validation_jsonl_path,
            "testing_jsonl" => testing_jsonl_path,
            "dataset_counts" => metadata_value(corpus_metadata, "dataset_counts", "unspecified"),
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
            "estimated_parameter_count" => estimate_gpt2_parameter_count(config),
            "model_seed" => settings.model_seed,
        ),
        "training" => Dict(
            "optimizer_name" => "Flux.Adam",
            "optimizer_hyperparameters" => Dict("learning_rate" => settings.learning_rate),
            "batching" => "example_based_padded_sft",
            "batch_size" => settings.batch_size,
            "epochs" => settings.epochs,
            "learning_rate" => settings.learning_rate,
            "loss_mode" => "assistant_only",
            "train_batches" => length(train_batches),
            "validation_batches" => length(validation_batches),
            "test_batches" => length(test_batches),
            "validation_batches_for_eval" => length(validation_batches_for_eval),
            "test_batches_for_eval" => length(test_batches_for_eval),
            "log_every_steps" => settings.log_every_steps,
            "checkpoint_every_steps" => settings.checkpoint_every_steps,
            "max_step_checkpoints" => settings.max_step_checkpoints,
            "max_epoch_checkpoints" => settings.max_epoch_checkpoints,
            "train_stats" => Dict(pairs(train_stats)),
            "validation_stats" => Dict(pairs(validation_stats)),
            "test_stats" => Dict(pairs(test_stats)),
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
            "data_mask_audit_path" => audit_path,
            "evaluation_prompts_txt" => joinpath(output_dir, "evaluation_prompts.txt"),
            "evaluation_prompts_json" => joinpath(output_dir, "evaluation_prompts.json"),
            "progress_path" => progress_path,
        ),
        "epoch_metrics" => epoch_metrics,
        "step_checkpoint_metrics" => step_checkpoint_metrics,
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

function resolve_clean_sft_metadata_path(dataset_dir::AbstractString)::String
    candidates = (
        joinpath(dataset_dir, "metadata.json"),
        joinpath(dataset_dir, "corpus_metadata.json"),
    )
    for path in candidates
        isfile(path) && return path
    end
    throw(ArgumentError(
        "clean SFT dataset metadata does not exist in $(dataset_dir); expected metadata.json. " *
        "Regenerate the corpus with: python3 tools/prepare_tiny_chatbot_clean_sft_corpus.py --output-dir tmp/tiny_chatbot_clean_sft_corpus_v1",
    ))
end

function validate_v4_settings(settings::CleanSFTV4Settings)
    settings.device in (:auto, :cpu, :gpu) ||
        throw(ArgumentError("device must be one of :auto, :cpu, or :gpu"))
    settings.context_length > 0 || throw(ArgumentError("context_length must be > 0"))
    settings.batch_size > 0 || throw(ArgumentError("batch_size must be > 0"))
    settings.epochs > 0 || throw(ArgumentError("epochs must be > 0"))
    settings.learning_rate > 0 || throw(ArgumentError("learning_rate must be > 0"))
    settings.num_layers > 0 || throw(ArgumentError("num_layers must be > 0"))
    settings.num_heads > 0 || throw(ArgumentError("num_heads must be > 0"))
    settings.embedding_size % settings.num_heads == 0 ||
        throw(ArgumentError("embedding_size must be divisible by num_heads"))
    settings.ffn_hidden_size > 0 || throw(ArgumentError("ffn_hidden_size must be > 0"))
    settings.tokenizer_vocab_size > 0 || throw(ArgumentError("tokenizer_vocab_size must be > 0"))
    settings.tokenizer_training_example_limit >= 0 ||
        throw(ArgumentError("tokenizer_training_example_limit must be >= 0"))
    settings.train_example_limit >= 0 || throw(ArgumentError("train_example_limit must be >= 0"))
    settings.validation_example_limit >= 0 || throw(ArgumentError("validation_example_limit must be >= 0"))
    settings.test_example_limit >= 0 || throw(ArgumentError("test_example_limit must be >= 0"))
    settings.validation_batch_limit >= 0 || throw(ArgumentError("validation_batch_limit must be >= 0"))
    settings.test_batch_limit >= 0 || throw(ArgumentError("test_batch_limit must be >= 0"))
    settings.checkpoint_every_steps >= 0 || throw(ArgumentError("checkpoint_every_steps must be >= 0"))
    settings.max_step_checkpoints >= 1 || throw(ArgumentError("max_step_checkpoints must be >= 1"))
    settings.max_epoch_checkpoints >= 1 || throw(ArgumentError("max_epoch_checkpoints must be >= 1"))
    settings.audit_example_count >= 0 || throw(ArgumentError("audit_example_count must be >= 0"))
    return settings
end

function read_clean_sft_examples(path::AbstractString; limit::Int = 0)::Vector{CleanSFTExample}
    examples = CleanSFTExample[]
    open(path, "r") do io
        while !eof(io)
            line = strip(readline(io))
            isempty(line) && continue
            row = JSON3.read(line)
            prompt_text = String(row.prompt_text)
            assistant_text = String(row.assistant_text)
            target_text = hasproperty(row, :target_text) ?
                String(row.target_text) :
                string(" ", assistant_text, "\n", CHAT_MARKERS.end_assistant, "\n", CHAT_MARKERS.chat_end)
            chat_text = hasproperty(row, :chat_text) ? String(row.chat_text) : prompt_text * target_text
            push!(
                examples,
                CleanSFTExample(
                    hasproperty(row, :id) ? String(row.id) : string("example_", length(examples) + 1),
                    prompt_text,
                    target_text,
                    assistant_text,
                    chat_text,
                ),
            )
            limit > 0 && length(examples) >= limit && break
        end
    end
    isempty(examples) && throw(ArgumentError("clean SFT split is empty: $(path)"))
    return examples
end

function limit_examples(examples::Vector{CleanSFTExample}, limit::Int)::Vector{CleanSFTExample}
    (limit <= 0 || limit >= length(examples)) && return examples
    return examples[1:limit]
end

function build_clean_sft_batches(
    examples::Vector{CleanSFTExample},
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer;
    context_length::Int,
    batch_size::Int,
    pad_token_id::Int,
)
    batches = Tuple{Matrix{Int32},Matrix{Int32},Matrix{Float32}}[]
    input_batch = fill(Int32(pad_token_id), context_length, batch_size)
    target_batch = fill(Int32(pad_token_id), context_length, batch_size)
    loss_mask_batch = zeros(Float32, context_length, batch_size)

    batch_column = 0
    kept_examples = 0
    skipped_too_long_target = 0
    skipped_empty = 0
    truncated_prompt_examples = 0
    loss_target_count = 0
    total_prompt_tokens = 0
    total_target_tokens = 0

    for example in examples
        window = encode_clean_sft_window(example, tokenizer; context_length = context_length, pad_token_id = pad_token_id)
        if window.status === :too_long_target
            skipped_too_long_target += 1
            continue
        elseif window.status === :empty
            skipped_empty += 1
            continue
        elseif window.status !== :ok
            error("unexpected clean SFT window status $(window.status)")
        end

        batch_column += 1
        input_batch[:, batch_column] = window.input_ids
        target_batch[:, batch_column] = window.target_ids
        loss_mask_batch[:, batch_column] = window.loss_mask

        kept_examples += 1
        truncated_prompt_examples += window.prompt_truncated ? 1 : 0
        loss_target_count += window.loss_target_count
        total_prompt_tokens += window.prompt_token_count
        total_target_tokens += window.target_token_count

        if batch_column == batch_size
            push!(batches, (copy(input_batch), copy(target_batch), copy(loss_mask_batch)))
            fill!(input_batch, Int32(pad_token_id))
            fill!(target_batch, Int32(pad_token_id))
            fill!(loss_mask_batch, 0.0f0)
            batch_column = 0
        end
    end

    if batch_column > 0
        push!(
            batches,
            (
                copy(input_batch[:, 1:batch_column]),
                copy(target_batch[:, 1:batch_column]),
                copy(loss_mask_batch[:, 1:batch_column]),
            ),
        )
    end

    kept_examples > 0 || throw(ArgumentError("clean SFT split did not yield any trainable examples"))
    stats = (
        source_example_count = length(examples),
        example_count = kept_examples,
        batch_count = length(batches),
        skipped_too_long_target = skipped_too_long_target,
        skipped_empty = skipped_empty,
        truncated_prompt_examples = truncated_prompt_examples,
        loss_target_count = loss_target_count,
        mean_prompt_tokens = total_prompt_tokens / kept_examples,
        mean_target_tokens = total_target_tokens / kept_examples,
    )
    return batches, stats
end

function encode_clean_sft_window(
    example::CleanSFTExample,
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer;
    context_length::Int,
    pad_token_id::Int,
)
    prompt_token_ids = KeemenaSubwords.encode(tokenizer, example.prompt_text; add_special_tokens = false)
    target_token_ids = KeemenaSubwords.encode(tokenizer, example.target_text; add_special_tokens = false)

    if isempty(prompt_token_ids) || isempty(target_token_ids)
        return (status = :empty,)
    end

    max_full_length = context_length + 1
    if length(target_token_ids) >= max_full_length
        return (status = :too_long_target,)
    end

    prompt_truncated = false
    prompt_keep_count = length(prompt_token_ids)
    if length(prompt_token_ids) + length(target_token_ids) > max_full_length
        prompt_keep_count = max_full_length - length(target_token_ids)
        prompt_keep_count <= 0 && return (status = :too_long_target,)
        prompt_token_ids = last(prompt_token_ids, prompt_keep_count)
        prompt_truncated = true
    end

    full_token_ids = vcat(prompt_token_ids, target_token_ids)
    loss_flags = vcat(zeros(Float32, length(prompt_token_ids)), ones(Float32, length(target_token_ids)))
    sequence_length = length(full_token_ids) - 1
    sequence_length > 0 || return (status = :empty,)

    input_ids = fill(Int32(pad_token_id), context_length)
    target_ids = fill(Int32(pad_token_id), context_length)
    loss_mask = zeros(Float32, context_length)

    input_ids[1:sequence_length] = Int32.(full_token_ids[1:(end - 1)])
    target_ids[1:sequence_length] = Int32.(full_token_ids[2:end])
    loss_mask[1:sequence_length] = loss_flags[2:end]
    loss_target_count = Int(sum(loss_mask))
    loss_target_count > 0 || return (status = :empty,)

    return (
        status = :ok,
        input_ids = input_ids,
        target_ids = target_ids,
        loss_mask = loss_mask,
        sequence_length = sequence_length,
        prompt_truncated = prompt_truncated,
        prompt_token_count = prompt_keep_count,
        target_token_count = length(target_token_ids),
        loss_target_count = loss_target_count,
    )
end

function write_clean_sft_audit(
    path::AbstractString,
    examples::Vector{CleanSFTExample},
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer;
    context_length::Int,
    pad_token_id::Int,
    audit_example_count::Int,
    seed::Int,
)
    mkpath(dirname(path))
    sample_count = min(audit_example_count, length(examples))
    rng = Random.MersenneTwister(seed + 23)
    selected_indices = sample_count == length(examples) ? collect(eachindex(examples)) : sort(Random.randperm(rng, length(examples))[1:sample_count])

    open(path, "w") do io
        println(io, "# Clean SFT v4 data/mask audit")
        println(io, "context_length: ", context_length)
        println(io, "sample_count: ", sample_count)
        println(io)

        for (audit_index, example_index) in enumerate(selected_indices)
            example = examples[example_index]
            window = encode_clean_sft_window(example, tokenizer; context_length = context_length, pad_token_id = pad_token_id)
            println(io, "## audit_example ", audit_index)
            println(io, "id: ", example.id)
            println(io, "status: ", window.status)
            println(io, "prompt>")
            println(io, example.prompt_text)
            println(io, "target>")
            println(io, example.target_text)
            if window.status === :ok
                live_input_ids = Int.(window.input_ids[1:window.sequence_length])
                live_target_ids = Int.(window.target_ids[1:window.sequence_length])
                live_mask = window.loss_mask[1:window.sequence_length]
                masked_target_ids = Int[live_target_ids[index] for index in eachindex(live_target_ids) if live_mask[index] > 0.0f0]
                println(io, "prompt_truncated: ", window.prompt_truncated)
                println(io, "sequence_length: ", window.sequence_length)
                println(io, "loss_target_count: ", window.loss_target_count)
                println(io, "decoded_input_window>")
                println(io, KeemenaSubwords.decode(tokenizer, live_input_ids))
                println(io, "decoded_loss_targets>")
                println(io, KeemenaSubwords.decode(tokenizer, masked_target_ids))
            end
            println(io)
        end
    end
    return path
end

function masked_mean_loss(model, batches)::Float64
    total_loss = 0.0
    total_weight = 0.0
    for (input_batch, target_batch, loss_mask_batch) in batches
        batch_weight = Float64(sum(loss_mask_batch))
        batch_weight > 0 || continue
        logits, _ = KeemenaLM.Core.lm_forward(model, input_batch; cache = nothing, is_training = false)
        loss_target_batch = KeemenaLM.FluxBackend.move_like(target_batch, model.token_embedding)
        loss_weights = KeemenaLM.FluxBackend.move_like(loss_mask_batch, model.token_embedding)
        batch_loss = Float64(KeemenaLM.Core.causal_lm_cross_entropy(logits, loss_target_batch, loss_weights))
        total_loss += batch_loss * batch_weight
        total_weight += batch_weight
    end
    total_weight > 0 || throw(ArgumentError("batches did not include any loss targets"))
    return total_loss / total_weight
end

function estimate_gpt2_parameter_count(config::GPT2Config)::Int
    embedding_parameters = config.embedding_size * config.vocab_size
    position_parameters = config.embedding_size * config.context_length
    per_layer_parameters =
        4 * (config.embedding_size * config.embedding_size + config.embedding_size) +
        (config.ffn_hidden_size * config.embedding_size + config.ffn_hidden_size) +
        (config.embedding_size * config.ffn_hidden_size + config.embedding_size) +
        4 * 2 * config.embedding_size
    final_norm_parameters = 2 * config.embedding_size
    return embedding_parameters + position_parameters + config.num_layers * per_layer_parameters + final_norm_parameters
end

function clean_sft_v4_recipe_dict(dataset_dir::AbstractString, output_dir::AbstractString, settings::CleanSFTV4Settings)
    config = GPT2Config(
        vocab_size = settings.tokenizer_vocab_size,
        context_length = settings.context_length,
        num_layers = settings.num_layers,
        num_heads = settings.num_heads,
        embedding_size = settings.embedding_size,
        ffn_hidden_size = settings.ffn_hidden_size,
    )
    return Dict(
        "experiment" => CLEAN_SFT_V4_EXPERIMENT_NAME,
        "purpose" => CLEAN_SFT_V4_PURPOSE,
        "dataset_dir" => abspath(dataset_dir),
        "output_dir" => abspath(output_dir),
        "backend" => "flux",
        "device" => String(settings.device),
        "loss_mode" => "assistant_only",
        "batching" => "example_based_padded_sft",
        "optimizer_name" => "Flux.Adam",
        "optimizer_hyperparameters" => Dict("learning_rate" => settings.learning_rate),
        "tokenizer_package" => "KeemenaSubwords.jl",
        "tokenizer_trainer" => String(settings.tokenizer_trainer),
        "tokenizer_vocab_size" => settings.tokenizer_vocab_size,
        "tokenizer_min_frequency" => settings.tokenizer_min_frequency,
        "tokenizer_model_name" => settings.tokenizer_model_name,
        "tokenizer_training_example_limit" => settings.tokenizer_training_example_limit,
        "reuse_tokenizer_bundle_dir" => settings.reuse_tokenizer_bundle_dir,
        "chat_marker_strategy" => "literal markers in target text plus tokenizer added special tokens",
        "chat_marker_strings" => Dict(string(key) => value for (key, value) in settings.chat_special_tokens),
        "context_length" => settings.context_length,
        "num_layers" => settings.num_layers,
        "num_heads" => settings.num_heads,
        "embedding_size" => settings.embedding_size,
        "ffn_hidden_size" => settings.ffn_hidden_size,
        "estimated_parameter_count_at_requested_vocab" => estimate_gpt2_parameter_count(config),
        "batch_size" => settings.batch_size,
        "epochs" => settings.epochs,
        "sample_generation_tokens" => settings.sample_generation_tokens,
        "train_example_limit" => settings.train_example_limit,
        "validation_example_limit" => settings.validation_example_limit,
        "test_example_limit" => settings.test_example_limit,
        "validation_batch_limit" => settings.validation_batch_limit,
        "test_batch_limit" => settings.test_batch_limit,
        "log_every_steps" => settings.log_every_steps,
        "checkpoint_every_steps" => settings.checkpoint_every_steps,
        "max_step_checkpoints" => settings.max_step_checkpoints,
        "max_epoch_checkpoints" => settings.max_epoch_checkpoints,
        "audit_example_count" => settings.audit_example_count,
        "shuffle_batches" => settings.shuffle_batches,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !any(argument -> argument in ("--help", "-h"), ARGS)
        command_allows_gpu_device(ARGS) && KeemenaLM.FluxBackend.has_functional_cuda_gpu()
    end
    Base.invokelatest(main_v4, ARGS)
end
