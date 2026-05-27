#!/usr/bin/env julia

include(joinpath(@__DIR__, "run_tiny_chatbot_clean_sft_v4.jl"))

using Flux
using JSON3
using KeemenaLM
using KeemenaSubwords
using Printf
using Random

const V5_DATASET_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_v5_scratch_corpus")
const V5_OUTPUT_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_v5_scratch_candidate_run")
const V5_EXPERIMENT_NAME = "tiny_chatbot_v5_scratch_candidate_run"
const V5_PURPOSE = "v5 stronger pure-scratch curriculum: prose pretrain, chat LM continuation, assistant-only SFT"

Base.@kwdef struct V5ScratchSettings
    model_seed::Int = 20260520
    generation_seed::Int = 20260521
    device::Symbol = :auto
    context_length::Int = 384
    batch_size::Int = 2
    pretrain_epochs::Int = 1
    chat_lm_epochs::Int = 1
    sft_epochs::Int = 2
    learning_rate::Float32 = 0.00015f0
    num_layers::Int = 16
    num_heads::Int = 12
    embedding_size::Int = 768
    ffn_hidden_size::Int = 3072
    sample_generation_tokens::Int = 180
    tokenizer_trainer::Symbol = :hf_gpt2_bytebpe
    tokenizer_vocab_size::Int = 32_768
    tokenizer_min_frequency::Int = 2
    tokenizer_model_name::String = "tiny_chatbot_v5_scratch_gpt2_bytebpe"
    tokenizer_training_text_limit::Int = 80_000
    pretrain_text_limit::Int = 0
    chat_lm_text_limit::Int = 0
    sft_train_example_limit::Int = 0
    validation_text_limit::Int = 1_000
    test_text_limit::Int = 1_000
    sft_validation_example_limit::Int = 1_000
    sft_test_example_limit::Int = 1_000
    validation_batch_limit::Int = 0
    test_batch_limit::Int = 0
    log_every_steps::Int = 25
    audit_example_count::Int = 8
    shuffle_batches::Bool = true
    save_mid_checkpoint::Bool = true
    save_final_checkpoint::Bool = false
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

function main_v5(args)
    dataset_dir = V5_DATASET_DIR
    output_dir = V5_OUTPUT_DIR
    settings = V5ScratchSettings()

    argument_index = 1
    while argument_index <= length(args)
        argument = args[argument_index]
        if argument in ("--help", "-h")
            print_v5_usage()
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
            settings = update_v5_settings(settings; device = parse_device(args[argument_index]))
        elseif argument == "--context-length"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --context-length")
            settings = update_v5_settings(settings; context_length = parse(Int, args[argument_index]))
        elseif argument == "--batch-size"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --batch-size")
            settings = update_v5_settings(settings; batch_size = parse(Int, args[argument_index]))
        elseif argument == "--pretrain-epochs"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --pretrain-epochs")
            settings = update_v5_settings(settings; pretrain_epochs = parse(Int, args[argument_index]))
        elseif argument == "--chat-lm-epochs"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --chat-lm-epochs")
            settings = update_v5_settings(settings; chat_lm_epochs = parse(Int, args[argument_index]))
        elseif argument == "--sft-epochs"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --sft-epochs")
            settings = update_v5_settings(settings; sft_epochs = parse(Int, args[argument_index]))
        elseif argument == "--learning-rate"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --learning-rate")
            settings = update_v5_settings(settings; learning_rate = parse(Float32, args[argument_index]))
        elseif argument == "--num-layers"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --num-layers")
            settings = update_v5_settings(settings; num_layers = parse(Int, args[argument_index]))
        elseif argument == "--num-heads"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --num-heads")
            settings = update_v5_settings(settings; num_heads = parse(Int, args[argument_index]))
        elseif argument == "--embedding-size"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --embedding-size")
            settings = update_v5_settings(settings; embedding_size = parse(Int, args[argument_index]))
        elseif argument == "--ffn-hidden-size"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --ffn-hidden-size")
            settings = update_v5_settings(settings; ffn_hidden_size = parse(Int, args[argument_index]))
        elseif argument == "--tokenizer-vocab-size"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --tokenizer-vocab-size")
            settings = update_v5_settings(settings; tokenizer_vocab_size = parse(Int, args[argument_index]))
        elseif argument == "--tokenizer-training-text-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --tokenizer-training-text-limit")
            settings = update_v5_settings(settings; tokenizer_training_text_limit = parse(Int, args[argument_index]))
        elseif argument == "--pretrain-text-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --pretrain-text-limit")
            settings = update_v5_settings(settings; pretrain_text_limit = parse(Int, args[argument_index]))
        elseif argument == "--chat-lm-text-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --chat-lm-text-limit")
            settings = update_v5_settings(settings; chat_lm_text_limit = parse(Int, args[argument_index]))
        elseif argument == "--sft-train-example-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --sft-train-example-limit")
            settings = update_v5_settings(settings; sft_train_example_limit = parse(Int, args[argument_index]))
        elseif argument == "--validation-batch-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --validation-batch-limit")
            settings = update_v5_settings(settings; validation_batch_limit = parse(Int, args[argument_index]))
        elseif argument == "--test-batch-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --test-batch-limit")
            settings = update_v5_settings(settings; test_batch_limit = parse(Int, args[argument_index]))
        elseif argument == "--log-every-steps"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --log-every-steps")
            settings = update_v5_settings(settings; log_every_steps = parse(Int, args[argument_index]))
        elseif argument == "--audit-example-count"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --audit-example-count")
            settings = update_v5_settings(settings; audit_example_count = parse(Int, args[argument_index]))
        elseif argument == "--reuse-tokenizer-bundle-dir"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --reuse-tokenizer-bundle-dir")
            settings = update_v5_settings(settings; reuse_tokenizer_bundle_dir = abspath(args[argument_index]))
        elseif argument == "--no-shuffle-batches"
            settings = update_v5_settings(settings; shuffle_batches = false)
        elseif argument == "--no-mid-checkpoint"
            settings = update_v5_settings(settings; save_mid_checkpoint = false)
        elseif argument == "--save-final-checkpoint"
            settings = update_v5_settings(settings; save_final_checkpoint = true)
        else
            error("unknown argument $(argument). Run with --help for usage.")
        end
        argument_index += 1
    end

    return run_v5_scratch(dataset_dir, output_dir; settings = settings)
end

function print_v5_usage()
    println("""
usage: julia --project=tools/subword_real_text tools/run_tiny_chatbot_v5_scratch.jl [options]

Prepare the corpus first:
  python3 tools/prepare_tiny_chatbot_v5_scratch_corpus.py --output-dir tmp/tiny_chatbot_v5_scratch_corpus

Important defaults:
  stronger model: 16 layers, 12 heads, 768 embedding, 3072 FFN, context 384, vocab 32768
  stages: 1 pretrain epoch, 1 chat-LM epoch, 2 assistant-only SFT epochs
  checkpointing: only one mid_run_checkpoint.jld2 by default; final output is bundle weights

Options:
  --dataset-dir DIR
  --output-dir DIR
  --device auto|cpu|gpu
  --context-length N
  --batch-size N
  --pretrain-epochs N
  --chat-lm-epochs N
  --sft-epochs N
  --learning-rate X
  --num-layers N
  --num-heads N
  --embedding-size N
  --ffn-hidden-size N
  --tokenizer-vocab-size N
  --tokenizer-training-text-limit N
  --no-mid-checkpoint
  --save-final-checkpoint
""")
end

function update_v5_settings(settings::V5ScratchSettings; kwargs...)
    options = Dict{Symbol,Any}(kwargs)
    return V5ScratchSettings(
        model_seed = get(options, :model_seed, settings.model_seed),
        generation_seed = get(options, :generation_seed, settings.generation_seed),
        device = get(options, :device, settings.device),
        context_length = get(options, :context_length, settings.context_length),
        batch_size = get(options, :batch_size, settings.batch_size),
        pretrain_epochs = get(options, :pretrain_epochs, settings.pretrain_epochs),
        chat_lm_epochs = get(options, :chat_lm_epochs, settings.chat_lm_epochs),
        sft_epochs = get(options, :sft_epochs, settings.sft_epochs),
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
        pretrain_text_limit = get(options, :pretrain_text_limit, settings.pretrain_text_limit),
        chat_lm_text_limit = get(options, :chat_lm_text_limit, settings.chat_lm_text_limit),
        sft_train_example_limit = get(options, :sft_train_example_limit, settings.sft_train_example_limit),
        validation_text_limit = get(options, :validation_text_limit, settings.validation_text_limit),
        test_text_limit = get(options, :test_text_limit, settings.test_text_limit),
        sft_validation_example_limit = get(options, :sft_validation_example_limit, settings.sft_validation_example_limit),
        sft_test_example_limit = get(options, :sft_test_example_limit, settings.sft_test_example_limit),
        validation_batch_limit = get(options, :validation_batch_limit, settings.validation_batch_limit),
        test_batch_limit = get(options, :test_batch_limit, settings.test_batch_limit),
        log_every_steps = get(options, :log_every_steps, settings.log_every_steps),
        audit_example_count = get(options, :audit_example_count, settings.audit_example_count),
        shuffle_batches = get(options, :shuffle_batches, settings.shuffle_batches),
        save_mid_checkpoint = get(options, :save_mid_checkpoint, settings.save_mid_checkpoint),
        save_final_checkpoint = get(options, :save_final_checkpoint, settings.save_final_checkpoint),
        reuse_tokenizer_bundle_dir = get(options, :reuse_tokenizer_bundle_dir, settings.reuse_tokenizer_bundle_dir),
        document_separator = get(options, :document_separator, settings.document_separator),
        chat_special_tokens = get(options, :chat_special_tokens, settings.chat_special_tokens),
    )
end

function run_v5_scratch(
    dataset_dir::AbstractString,
    output_dir::AbstractString;
    settings::V5ScratchSettings = V5ScratchSettings(),
)
    validate_v5_settings(settings)
    dataset_dir = abspath(dataset_dir)
    output_dir = abspath(output_dir)

    paths = v5_dataset_paths(dataset_dir)
    for path in values(paths)
        isfile(path) || throw(ArgumentError("v5 dataset file does not exist: $(path). Prepare v5 data first."))
    end

    checkpoint_dir = joinpath(output_dir, "checkpoints")
    bundle_dir = joinpath(output_dir, "bundle")
    tokenizer_bundle_dir = joinpath(output_dir, "tokenizer_bundle")
    metrics_path = joinpath(output_dir, "metrics.json")
    progress_path = joinpath(output_dir, "progress.json")
    sample_path = joinpath(output_dir, "sample_outputs.txt")
    audit_path = joinpath(output_dir, "sft_data_mask_audit.txt")
    mkpath(output_dir)
    mkpath(checkpoint_dir)

    metadata = JSON3.read(read(paths.metadata, String))
    write_json(joinpath(output_dir, "run_recipe.json"), v5_recipe_dict(dataset_dir, output_dir, settings))

    pretrain_texts = read_prepared_paragraphs(paths.pretrain_training; limit = optional_text_limit(settings.pretrain_text_limit))
    pretrain_validation_texts = read_prepared_paragraphs(paths.pretrain_validation; limit = optional_text_limit(settings.validation_text_limit))
    pretrain_testing_texts = read_prepared_paragraphs(paths.pretrain_testing; limit = optional_text_limit(settings.test_text_limit))
    chat_lm_texts = read_prepared_paragraphs(paths.chat_training; limit = optional_text_limit(settings.chat_lm_text_limit))
    chat_validation_texts = read_prepared_paragraphs(paths.chat_validation; limit = optional_text_limit(settings.validation_text_limit))
    chat_testing_texts = read_prepared_paragraphs(paths.chat_testing; limit = optional_text_limit(settings.test_text_limit))
    sft_examples = read_clean_sft_examples(paths.sft_training; limit = settings.sft_train_example_limit)
    sft_validation_examples = read_clean_sft_examples(paths.sft_validation; limit = settings.sft_validation_example_limit)
    sft_testing_examples = read_clean_sft_examples(paths.sft_testing; limit = settings.sft_test_example_limit)

    tokenizer_training_texts = mixed_tokenizer_training_texts(
        pretrain_texts,
        chat_lm_texts,
        sft_examples,
        settings.tokenizer_training_text_limit,
    )
    tokenizer_settings = TinyChatbotSubwordSettings(
        model_seed = settings.model_seed,
        generation_seed = settings.generation_seed,
        device = settings.device,
        loss_mode = :assistant_only,
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        epochs = settings.pretrain_epochs + settings.chat_lm_epochs + settings.sft_epochs,
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
        tokenizer_training_text_limit = length(tokenizer_training_texts),
        reuse_tokenizer_bundle_dir = settings.reuse_tokenizer_bundle_dir,
        document_separator = settings.document_separator,
        chat_special_tokens = settings.chat_special_tokens,
    )
    tokenizer, tokenizer_training_summary, tokenizer_bundle_source = prepare_tokenizer_bundle(
        tokenizer_bundle_dir,
        tokenizer_training_texts,
        tokenizer_settings,
    )
    pad_token_id = KeemenaSubwords.token_to_id(tokenizer, settings.chat_special_tokens[:unk])

    pretrain_batches, pretrain_stats = build_subword_lm_batches(
        pretrain_texts,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        document_separator = settings.document_separator,
        loss_mode = :all_tokens,
        chat_special_tokens = settings.chat_special_tokens,
    )
    pretrain_validation_batches, pretrain_validation_stats = build_subword_lm_batches(
        pretrain_validation_texts,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        document_separator = settings.document_separator,
        loss_mode = :all_tokens,
        chat_special_tokens = settings.chat_special_tokens,
    )
    pretrain_test_batches, pretrain_test_stats = build_subword_lm_batches(
        pretrain_testing_texts,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        document_separator = settings.document_separator,
        loss_mode = :all_tokens,
        chat_special_tokens = settings.chat_special_tokens,
    )
    chat_batches, chat_stats = build_subword_lm_batches(
        chat_lm_texts,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        document_separator = settings.document_separator,
        loss_mode = :all_tokens,
        chat_special_tokens = settings.chat_special_tokens,
    )
    chat_validation_batches, chat_validation_stats = build_subword_lm_batches(
        chat_validation_texts,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        document_separator = settings.document_separator,
        loss_mode = :all_tokens,
        chat_special_tokens = settings.chat_special_tokens,
    )
    chat_test_batches, chat_test_stats = build_subword_lm_batches(
        chat_testing_texts,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        document_separator = settings.document_separator,
        loss_mode = :all_tokens,
        chat_special_tokens = settings.chat_special_tokens,
    )
    sft_batches, sft_stats = build_clean_sft_batches(
        sft_examples,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        pad_token_id = pad_token_id,
    )
    sft_validation_batches, sft_validation_stats = build_clean_sft_batches(
        sft_validation_examples,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        pad_token_id = pad_token_id,
    )
    sft_test_batches, sft_test_stats = build_clean_sft_batches(
        sft_testing_examples,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        pad_token_id = pad_token_id,
    )
    write_clean_sft_audit(
        audit_path,
        sft_examples,
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
            "experiment" => V5_EXPERIMENT_NAME,
            "corpus_source" => "tiny_chatbot_v5_scratch_corpus",
            "tokenizer_bundle_dir" => tokenizer_bundle_dir,
            "prepared_corpus_metadata_path" => paths.metadata,
            "optimizer_name" => "Flux.Adam",
            "requested_device" => String(settings.device),
            "resolved_device" => String(resolved_device),
            "training_curriculum" => "pretrain_all_tokens -> chat_lm_all_tokens -> sft_assistant_only",
        ),
    )

    validation_sets = Dict(
        "pretrain" => limit_batches(pretrain_validation_batches, settings.validation_batch_limit),
        "chat_lm" => limit_batches(chat_validation_batches, settings.validation_batch_limit),
        "sft" => limit_batches(sft_validation_batches, settings.validation_batch_limit),
    )
    test_sets = Dict(
        "pretrain" => limit_batches(pretrain_test_batches, settings.test_batch_limit),
        "chat_lm" => limit_batches(chat_test_batches, settings.test_batch_limit),
        "sft" => limit_batches(sft_test_batches, settings.test_batch_limit),
    )
    total_planned_steps =
        settings.pretrain_epochs * length(pretrain_batches) +
        settings.chat_lm_epochs * length(chat_batches) +
        settings.sft_epochs * length(sft_batches)
    mid_checkpoint_step = cld(total_planned_steps, 2)
    mid_checkpoint_path = joinpath(checkpoint_dir, "mid_run_checkpoint.jld2")
    mid_checkpoint_saved = Ref(false)

    println("== v5 scratch chatbot curriculum ==")
    println("prepared_dataset_dir: $(dataset_dir)")
    println("output_dir: $(output_dir)")
    println("tokenizer_bundle_dir: $(tokenizer_bundle_dir)")
    println("requested_device: $(settings.device)")
    println("resolved_device: $(resolved_device)")
    println("model_shape: layers=$(settings.num_layers) heads=$(settings.num_heads) emb=$(settings.embedding_size) ffn=$(settings.ffn_hidden_size) context=$(settings.context_length)")
    println("estimated_parameters: $(estimate_gpt2_parameter_count(config))")
    println("planned_steps: $(total_planned_steps)")
    println("mid_checkpoint_step: $(mid_checkpoint_step)")
    println("checkpoint_policy: one mid-run checkpoint; final model is bundle weights")
    println("sft_audit: $(audit_path)")

    initial_validation = Dict(
        stage => masked_mean_loss(model, batches) for (stage, batches) in validation_sets
    )
    write_progress(
        progress_path;
        status = "running",
        epoch = 0,
        step = trainer.step,
        latest_train_loss = nothing,
        latest_validation_loss = initial_validation,
        latest_checkpoint = nothing,
    )

    stage_metrics = Dict{String,Any}[]
    batch_rng = Random.MersenneTwister(settings.model_seed + 101)
    append!(
        stage_metrics,
        train_v5_stage!(
            trainer,
            model,
            "pretrain",
            pretrain_batches,
            validation_sets["pretrain"],
            settings.pretrain_epochs,
            settings,
            batch_rng,
            progress_path,
            checkpoint_dir,
            mid_checkpoint_path,
            mid_checkpoint_step,
            mid_checkpoint_saved,
        ),
    )
    append!(
        stage_metrics,
        train_v5_stage!(
            trainer,
            model,
            "chat_lm",
            chat_batches,
            validation_sets["chat_lm"],
            settings.chat_lm_epochs,
            settings,
            batch_rng,
            progress_path,
            checkpoint_dir,
            mid_checkpoint_path,
            mid_checkpoint_step,
            mid_checkpoint_saved,
        ),
    )
    append!(
        stage_metrics,
        train_v5_stage!(
            trainer,
            model,
            "sft",
            sft_batches,
            validation_sets["sft"],
            settings.sft_epochs,
            settings,
            batch_rng,
            progress_path,
            checkpoint_dir,
            mid_checkpoint_path,
            mid_checkpoint_step,
            mid_checkpoint_saved,
        ),
    )

    final_checkpoint_path = nothing
    if settings.save_final_checkpoint
        final_checkpoint_path = joinpath(checkpoint_dir, "final_checkpoint.jld2")
        save_checkpoint(final_checkpoint_path, trainer, model; experiment = V5_EXPERIMENT_NAME, stage = "final")
    end

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
    samples = generate_samples(
        reloaded_model,
        reloaded_tokenizer,
        prompts;
        generation_seed = settings.generation_seed,
        max_new_tokens = settings.sample_generation_tokens,
    )
    write_samples(sample_path, samples)

    test_losses = Dict(stage => masked_mean_loss(reloaded_model, batches) for (stage, batches) in test_sets)
    metrics = Dict(
        "experiment" => V5_EXPERIMENT_NAME,
        "purpose" => V5_PURPOSE,
        "corpus_metadata" => metadata,
        "tokenizer" => Dict(
            "package" => "KeemenaSubwords.jl",
            "trainer" => String(settings.tokenizer_trainer),
            "vocab_size_requested" => settings.tokenizer_vocab_size,
            "vocab_size_actual" => KeemenaSubwords.vocab_size(tokenizer),
            "min_frequency" => settings.tokenizer_min_frequency,
            "model_name" => settings.tokenizer_model_name,
            "bundle_directory" => tokenizer_bundle_dir,
            "bundle_source" => tokenizer_bundle_source,
            "training_summary" => Dict(
                "trainer" => tokenizer_training_summary["trainer"],
                "tokenizer_type" => tokenizer_training_summary["tokenizer_type"],
                "config_type" => tokenizer_training_summary["config_type"],
                "model_name" => tokenizer_training_summary["model_name"],
                "version" => tokenizer_training_summary["version"],
                "vocab_size" => tokenizer_training_summary["vocab_size"],
            ),
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
            "curriculum" => ["pretrain", "chat_lm", "sft"],
            "optimizer_name" => "Flux.Adam",
            "learning_rate" => settings.learning_rate,
            "batch_size" => settings.batch_size,
            "pretrain_epochs" => settings.pretrain_epochs,
            "chat_lm_epochs" => settings.chat_lm_epochs,
            "sft_epochs" => settings.sft_epochs,
            "planned_steps" => total_planned_steps,
            "final_step" => trainer.step,
            "initial_validation_losses" => initial_validation,
            "test_losses" => test_losses,
            "mid_checkpoint_step" => mid_checkpoint_step,
            "mid_checkpoint_saved" => mid_checkpoint_saved[],
        ),
        "batch_stats" => Dict(
            "pretrain" => Dict(pairs(pretrain_stats)),
            "pretrain_validation" => Dict(pairs(pretrain_validation_stats)),
            "pretrain_test" => Dict(pairs(pretrain_test_stats)),
            "chat_lm" => Dict(pairs(chat_stats)),
            "chat_lm_validation" => Dict(pairs(chat_validation_stats)),
            "chat_lm_test" => Dict(pairs(chat_test_stats)),
            "sft" => Dict(pairs(sft_stats)),
            "sft_validation" => Dict(pairs(sft_validation_stats)),
            "sft_test" => Dict(pairs(sft_test_stats)),
        ),
        "artifacts" => Dict(
            "tokenizer_bundle_dir" => tokenizer_bundle_dir,
            "bundle_dir" => bundle_dir,
            "mid_checkpoint" => mid_checkpoint_saved[] ? mid_checkpoint_path : nothing,
            "final_checkpoint" => final_checkpoint_path,
            "sample_outputs_path" => sample_path,
            "evaluation_prompts_txt" => joinpath(output_dir, "evaluation_prompts.txt"),
            "evaluation_prompts_json" => joinpath(output_dir, "evaluation_prompts.json"),
            "sft_data_mask_audit_path" => audit_path,
            "progress_path" => progress_path,
        ),
        "stage_metrics" => stage_metrics,
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
        latest_train_loss = isempty(stage_metrics) ? nothing : stage_metrics[end]["train_loss"],
        latest_validation_loss = isempty(stage_metrics) ? initial_validation : stage_metrics[end]["validation_loss"],
        latest_checkpoint = mid_checkpoint_saved[] ? mid_checkpoint_path : nothing,
    )

    println("test_losses: $(test_losses)")
    println("tokenizer bundle: $(tokenizer_bundle_dir)")
    println("bundle export: $(bundle_dir)")
    println("mid checkpoint: $(mid_checkpoint_saved[] ? mid_checkpoint_path : "not saved")")
    println("metrics: $(metrics_path)")
    println("samples: $(sample_path)")
    return metrics
end

function train_v5_stage!(
    trainer,
    model,
    stage_name::AbstractString,
    train_batches,
    validation_batches,
    epochs::Int,
    settings::V5ScratchSettings,
    batch_rng,
    progress_path::AbstractString,
    checkpoint_dir::AbstractString,
    mid_checkpoint_path::AbstractString,
    mid_checkpoint_step::Int,
    mid_checkpoint_saved::Base.RefValue{Bool},
)
    metrics = Dict{String,Any}[]
    epochs <= 0 && return metrics
    for epoch in 1:epochs
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
                    epoch = trainer.epoch,
                    step = trainer.step,
                    latest_train_loss = recent_train_loss,
                    latest_validation_loss = nothing,
                    latest_checkpoint = mid_checkpoint_saved[] ? mid_checkpoint_path : nothing,
                )
                println(@sprintf(
                    "stage %s  step %d  epoch %d/%d  recent_train_loss=%.4f",
                    stage_name,
                    trainer.step,
                    epoch,
                    epochs,
                    recent_train_loss,
                ))
            end

            if settings.save_mid_checkpoint && !mid_checkpoint_saved[] && trainer.step >= mid_checkpoint_step
                save_checkpoint(
                    mid_checkpoint_path,
                    trainer,
                    model;
                    experiment = V5_EXPERIMENT_NAME,
                    stage = "mid_run",
                    curriculum_stage = stage_name,
                    step = trainer.step,
                )
                mid_checkpoint_saved[] = true
                println("saved only mid-run checkpoint: $(mid_checkpoint_path)")
            end
        end

        trainer.epoch += 1
        train_loss = sum(epoch_losses) / length(epoch_losses)
        validation_loss = masked_mean_loss(model, validation_batches)
        push!(
            metrics,
            Dict(
                "stage" => String(stage_name),
                "stage_epoch" => epoch,
                "global_epoch" => trainer.epoch,
                "step" => trainer.step,
                "train_loss" => train_loss,
                "train_perplexity" => exp(train_loss),
                "validation_loss" => validation_loss,
                "validation_perplexity" => exp(validation_loss),
            ),
        )
        write_progress(
            progress_path;
            status = "running",
            epoch = trainer.epoch,
            step = trainer.step,
            latest_train_loss = train_loss,
            latest_validation_loss = validation_loss,
            latest_checkpoint = mid_checkpoint_saved[] ? mid_checkpoint_path : nothing,
        )
        println(@sprintf(
            "stage %s  epoch %d/%d  train_loss=%.4f  validation_loss=%.4f",
            stage_name,
            epoch,
            epochs,
            train_loss,
            validation_loss,
        ))
    end
    return metrics
end

function v5_dataset_paths(dataset_dir::AbstractString)
    return (
        metadata = joinpath(dataset_dir, "metadata.json"),
        pretrain_training = joinpath(dataset_dir, "pretrain", "training.txt"),
        pretrain_validation = joinpath(dataset_dir, "pretrain", "validation.txt"),
        pretrain_testing = joinpath(dataset_dir, "pretrain", "testing.txt"),
        chat_training = joinpath(dataset_dir, "chat_lm", "training.txt"),
        chat_validation = joinpath(dataset_dir, "chat_lm", "validation.txt"),
        chat_testing = joinpath(dataset_dir, "chat_lm", "testing.txt"),
        sft_training = joinpath(dataset_dir, "sft", "training.jsonl"),
        sft_validation = joinpath(dataset_dir, "sft", "validation.jsonl"),
        sft_testing = joinpath(dataset_dir, "sft", "testing.jsonl"),
    )
end

function mixed_tokenizer_training_texts(
    pretrain_texts::Vector{String},
    chat_lm_texts::Vector{String},
    sft_examples::Vector{CleanSFTExample},
    limit::Int,
)::Vector{String}
    texts = String[]
    pretrain_limit = limit <= 0 ? length(pretrain_texts) : min(length(pretrain_texts), max(1, round(Int, limit * 0.70)))
    chat_limit = limit <= 0 ? length(chat_lm_texts) : min(length(chat_lm_texts), max(1, round(Int, limit * 0.15)))
    sft_limit = limit <= 0 ? length(sft_examples) : min(length(sft_examples), max(1, limit - pretrain_limit - chat_limit))
    append!(texts, pretrain_texts[1:pretrain_limit])
    append!(texts, chat_lm_texts[1:chat_limit])
    append!(texts, [example.chat_text for example in sft_examples[1:sft_limit]])
    isempty(texts) && throw(ArgumentError("tokenizer training texts are empty"))
    return texts
end

optional_text_limit(limit::Int) = limit <= 0 ? nothing : limit

function validate_v5_settings(settings::V5ScratchSettings)
    settings.device in (:auto, :cpu, :gpu) || throw(ArgumentError("device must be one of :auto, :cpu, or :gpu"))
    settings.context_length > 0 || throw(ArgumentError("context_length must be > 0"))
    settings.batch_size > 0 || throw(ArgumentError("batch_size must be > 0"))
    settings.pretrain_epochs >= 0 || throw(ArgumentError("pretrain_epochs must be >= 0"))
    settings.chat_lm_epochs >= 0 || throw(ArgumentError("chat_lm_epochs must be >= 0"))
    settings.sft_epochs >= 0 || throw(ArgumentError("sft_epochs must be >= 0"))
    settings.pretrain_epochs + settings.chat_lm_epochs + settings.sft_epochs > 0 ||
        throw(ArgumentError("at least one training stage epoch count must be > 0"))
    settings.learning_rate > 0 || throw(ArgumentError("learning_rate must be > 0"))
    settings.num_layers > 0 || throw(ArgumentError("num_layers must be > 0"))
    settings.num_heads > 0 || throw(ArgumentError("num_heads must be > 0"))
    settings.embedding_size % settings.num_heads == 0 ||
        throw(ArgumentError("embedding_size must be divisible by num_heads"))
    settings.ffn_hidden_size > 0 || throw(ArgumentError("ffn_hidden_size must be > 0"))
    settings.tokenizer_vocab_size > 0 || throw(ArgumentError("tokenizer_vocab_size must be > 0"))
    settings.tokenizer_training_text_limit >= 0 || throw(ArgumentError("tokenizer_training_text_limit must be >= 0"))
    settings.pretrain_text_limit >= 0 || throw(ArgumentError("pretrain_text_limit must be >= 0"))
    settings.chat_lm_text_limit >= 0 || throw(ArgumentError("chat_lm_text_limit must be >= 0"))
    settings.sft_train_example_limit >= 0 || throw(ArgumentError("sft_train_example_limit must be >= 0"))
    settings.validation_text_limit > 0 || throw(ArgumentError("validation_text_limit must be > 0"))
    settings.test_text_limit > 0 || throw(ArgumentError("test_text_limit must be > 0"))
    settings.validation_batch_limit >= 0 || throw(ArgumentError("validation_batch_limit must be >= 0"))
    settings.test_batch_limit >= 0 || throw(ArgumentError("test_batch_limit must be >= 0"))
    settings.audit_example_count >= 0 || throw(ArgumentError("audit_example_count must be >= 0"))
    return settings
end

function v5_recipe_dict(dataset_dir::AbstractString, output_dir::AbstractString, settings::V5ScratchSettings)
    config = GPT2Config(
        vocab_size = settings.tokenizer_vocab_size,
        context_length = settings.context_length,
        num_layers = settings.num_layers,
        num_heads = settings.num_heads,
        embedding_size = settings.embedding_size,
        ffn_hidden_size = settings.ffn_hidden_size,
    )
    return Dict(
        "experiment" => V5_EXPERIMENT_NAME,
        "purpose" => V5_PURPOSE,
        "dataset_dir" => abspath(dataset_dir),
        "output_dir" => abspath(output_dir),
        "backend" => "flux",
        "device" => String(settings.device),
        "curriculum" => ["pretrain_all_tokens", "chat_lm_all_tokens", "sft_assistant_only"],
        "optimizer_name" => "Flux.Adam",
        "optimizer_hyperparameters" => Dict("learning_rate" => settings.learning_rate),
        "tokenizer_package" => "KeemenaSubwords.jl",
        "tokenizer_trainer" => String(settings.tokenizer_trainer),
        "tokenizer_vocab_size" => settings.tokenizer_vocab_size,
        "tokenizer_min_frequency" => settings.tokenizer_min_frequency,
        "tokenizer_model_name" => settings.tokenizer_model_name,
        "tokenizer_training_text_limit" => settings.tokenizer_training_text_limit,
        "reuse_tokenizer_bundle_dir" => settings.reuse_tokenizer_bundle_dir,
        "context_length" => settings.context_length,
        "num_layers" => settings.num_layers,
        "num_heads" => settings.num_heads,
        "embedding_size" => settings.embedding_size,
        "ffn_hidden_size" => settings.ffn_hidden_size,
        "estimated_parameter_count_at_requested_vocab" => estimate_gpt2_parameter_count(config),
        "batch_size" => settings.batch_size,
        "pretrain_epochs" => settings.pretrain_epochs,
        "chat_lm_epochs" => settings.chat_lm_epochs,
        "sft_epochs" => settings.sft_epochs,
        "sample_generation_tokens" => settings.sample_generation_tokens,
        "checkpoint_policy" => "one mid-run checkpoint by default; final bundle only unless --save-final-checkpoint is passed",
        "save_mid_checkpoint" => settings.save_mid_checkpoint,
        "save_final_checkpoint" => settings.save_final_checkpoint,
        "shuffle_batches" => settings.shuffle_batches,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !any(argument -> argument in ("--help", "-h"), ARGS)
        command_allows_gpu_device(ARGS) && KeemenaLM.FluxBackend.has_functional_cuda_gpu()
    end
    Base.invokelatest(main_v5, ARGS)
end
