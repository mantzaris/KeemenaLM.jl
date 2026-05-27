#!/usr/bin/env julia

include(joinpath(@__DIR__, "run_tiny_chatbot_v5_scratch.jl"))

using Flux
using JSON3
using KeemenaLM
using KeemenaSubwords
using Printf
using Random

const V6_DATASET_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_v6_scratch_corpus")
const V6_OUTPUT_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_v6_scratch_candidate_run")
const V6_EXPERIMENT_NAME = "tiny_chatbot_v6_scratch_candidate_run"
const V6_PURPOSE = "v6 larger pure-scratch chatbot: streamed large pretrain, chat LM, synthetic-dominant clean assistant SFT"

Base.@kwdef mutable struct V6ScratchSettings
    model_seed::Int = 20260523
    generation_seed::Int = 20260524
    device::Symbol = :auto
    context_length::Int = 512
    batch_size::Int = 1
    gradient_accumulation_steps::Int = 4
    pretrain_epochs::Int = 1
    chat_lm_epochs::Int = 1
    sft_epochs::Int = 2
    learning_rate::Float32 = 0.00008f0
    num_layers::Int = 24
    num_heads::Int = 16
    embedding_size::Int = 1024
    ffn_hidden_size::Int = 4096
    sample_generation_tokens::Int = 180
    tokenizer_trainer::Symbol = :hf_gpt2_bytebpe
    tokenizer_vocab_size::Int = 32_768
    tokenizer_min_frequency::Int = 2
    tokenizer_model_name::String = "tiny_chatbot_v6_scratch_gpt2_bytebpe"
    tokenizer_training_text_limit::Int = 160_000
    pretrain_text_limit::Int = 0
    chat_lm_text_limit::Int = 0
    sft_train_example_limit::Int = 0
    validation_text_limit::Int = 1_000
    test_text_limit::Int = 1_000
    sft_validation_example_limit::Int = 2_000
    sft_test_example_limit::Int = 2_000
    validation_batch_limit::Int = 400
    test_batch_limit::Int = 400
    log_every_updates::Int = 25
    audit_example_count::Int = 10
    save_base_checkpoint::Bool = true
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

function main_v6(args)
    dataset_dir = V6_DATASET_DIR
    output_dir = V6_OUTPUT_DIR
    settings = V6ScratchSettings()

    argument_index = 1
    while argument_index <= length(args)
        argument = args[argument_index]
        if argument in ("--help", "-h")
            print_v6_usage()
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
            settings.device = parse_device(args[argument_index])
        elseif argument == "--context-length"
            argument_index += 1
            settings.context_length = parse(Int, args[argument_index])
        elseif argument == "--batch-size"
            argument_index += 1
            settings.batch_size = parse(Int, args[argument_index])
        elseif argument == "--gradient-accumulation-steps"
            argument_index += 1
            settings.gradient_accumulation_steps = parse(Int, args[argument_index])
        elseif argument == "--pretrain-epochs"
            argument_index += 1
            settings.pretrain_epochs = parse(Int, args[argument_index])
        elseif argument == "--chat-lm-epochs"
            argument_index += 1
            settings.chat_lm_epochs = parse(Int, args[argument_index])
        elseif argument == "--sft-epochs"
            argument_index += 1
            settings.sft_epochs = parse(Int, args[argument_index])
        elseif argument == "--learning-rate"
            argument_index += 1
            settings.learning_rate = parse(Float32, args[argument_index])
        elseif argument == "--num-layers"
            argument_index += 1
            settings.num_layers = parse(Int, args[argument_index])
        elseif argument == "--num-heads"
            argument_index += 1
            settings.num_heads = parse(Int, args[argument_index])
        elseif argument == "--embedding-size"
            argument_index += 1
            settings.embedding_size = parse(Int, args[argument_index])
        elseif argument == "--ffn-hidden-size"
            argument_index += 1
            settings.ffn_hidden_size = parse(Int, args[argument_index])
        elseif argument == "--sample-generation-tokens"
            argument_index += 1
            settings.sample_generation_tokens = parse(Int, args[argument_index])
        elseif argument == "--tokenizer-vocab-size"
            argument_index += 1
            settings.tokenizer_vocab_size = parse(Int, args[argument_index])
        elseif argument == "--tokenizer-training-text-limit"
            argument_index += 1
            settings.tokenizer_training_text_limit = parse(Int, args[argument_index])
        elseif argument == "--pretrain-text-limit"
            argument_index += 1
            settings.pretrain_text_limit = parse(Int, args[argument_index])
        elseif argument == "--chat-lm-text-limit"
            argument_index += 1
            settings.chat_lm_text_limit = parse(Int, args[argument_index])
        elseif argument == "--sft-train-example-limit"
            argument_index += 1
            settings.sft_train_example_limit = parse(Int, args[argument_index])
        elseif argument == "--validation-text-limit"
            argument_index += 1
            settings.validation_text_limit = parse(Int, args[argument_index])
        elseif argument == "--test-text-limit"
            argument_index += 1
            settings.test_text_limit = parse(Int, args[argument_index])
        elseif argument == "--sft-validation-example-limit"
            argument_index += 1
            settings.sft_validation_example_limit = parse(Int, args[argument_index])
        elseif argument == "--sft-test-example-limit"
            argument_index += 1
            settings.sft_test_example_limit = parse(Int, args[argument_index])
        elseif argument == "--validation-batch-limit"
            argument_index += 1
            settings.validation_batch_limit = parse(Int, args[argument_index])
        elseif argument == "--test-batch-limit"
            argument_index += 1
            settings.test_batch_limit = parse(Int, args[argument_index])
        elseif argument == "--log-every-updates"
            argument_index += 1
            settings.log_every_updates = parse(Int, args[argument_index])
        elseif argument == "--audit-example-count"
            argument_index += 1
            settings.audit_example_count = parse(Int, args[argument_index])
        elseif argument == "--reuse-tokenizer-bundle-dir"
            argument_index += 1
            settings.reuse_tokenizer_bundle_dir = abspath(args[argument_index])
        elseif argument == "--no-base-checkpoint"
            settings.save_base_checkpoint = false
        elseif argument == "--save-final-checkpoint"
            settings.save_final_checkpoint = true
        else
            error("unknown argument $(argument). Run with --help for usage.")
        end
        argument_index += 1
    end

    return run_v6_scratch(dataset_dir, output_dir; settings = settings)
end

function print_v6_usage()
    println("""
usage: julia --project=tools/subword_real_text tools/run_tiny_chatbot_v6_scratch.jl [options]

Prepare the corpus first:
  python3 tools/prepare_tiny_chatbot_v6_scratch_corpus.py --output-dir tmp/tiny_chatbot_v6_scratch_corpus

Important defaults:
  larger model: 24 layers, 16 heads, 1024 embedding, 4096 FFN, context 512, vocab 32768
  streamed training: train batches are generated from files instead of fully materialized in memory
  stages: 1 pretrain epoch, 1 chat-LM epoch, 2 assistant-only SFT epochs
  checkpointing: one base_before_sft_checkpoint.jld2 by default; final output is bundle weights

Options:
  --dataset-dir DIR
  --output-dir DIR
  --device auto|cpu|gpu
  --context-length N
  --batch-size N
  --gradient-accumulation-steps N
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
  --validation-batch-limit N
  --test-batch-limit N
  --no-base-checkpoint
  --save-final-checkpoint
""")
end

function run_v6_scratch(
    dataset_dir::AbstractString,
    output_dir::AbstractString;
    settings::V6ScratchSettings = V6ScratchSettings(),
)
    validate_v6_settings(settings)
    dataset_dir = abspath(dataset_dir)
    output_dir = abspath(output_dir)

    paths = v6_dataset_paths(dataset_dir)
    for path in values(paths)
        isfile(path) || throw(ArgumentError("v6 dataset file does not exist: $(path). Prepare v6 data first."))
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
    write_json(joinpath(output_dir, "run_recipe.json"), v6_recipe_dict(dataset_dir, output_dir, settings))

    tokenizer_training_texts = v6_tokenizer_training_texts(paths, settings)
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

    pretrain_validation_texts = read_prepared_paragraphs(paths.pretrain_validation; limit = settings.validation_text_limit)
    pretrain_testing_texts = read_prepared_paragraphs(paths.pretrain_testing; limit = settings.test_text_limit)
    chat_validation_texts = read_prepared_paragraphs(paths.chat_validation; limit = settings.validation_text_limit)
    chat_testing_texts = read_prepared_paragraphs(paths.chat_testing; limit = settings.test_text_limit)
    sft_audit_examples = read_clean_sft_examples(paths.sft_training; limit = max(settings.audit_example_count, 1))
    sft_validation_examples = read_clean_sft_examples(paths.sft_validation; limit = settings.sft_validation_example_limit)
    sft_testing_examples = read_clean_sft_examples(paths.sft_testing; limit = settings.sft_test_example_limit)

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
        sft_audit_examples,
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
            "experiment" => V6_EXPERIMENT_NAME,
            "corpus_source" => "tiny_chatbot_v6_scratch_corpus",
            "tokenizer_bundle_dir" => tokenizer_bundle_dir,
            "prepared_corpus_metadata_path" => paths.metadata,
            "optimizer_name" => "Flux.Adam",
            "requested_device" => String(settings.device),
            "resolved_device" => String(resolved_device),
            "training_curriculum" => "streamed_pretrain_all_tokens -> streamed_chat_lm_all_tokens -> streamed_sft_assistant_only",
            "gradient_accumulation_steps" => settings.gradient_accumulation_steps,
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

    base_checkpoint_path = joinpath(checkpoint_dir, "base_before_sft_checkpoint.jld2")
    final_checkpoint_path = nothing
    base_checkpoint_saved = Ref(false)

    println("== v6 scratch chatbot curriculum ==")
    println("prepared_dataset_dir: $(dataset_dir)")
    println("output_dir: $(output_dir)")
    println("tokenizer_bundle_dir: $(tokenizer_bundle_dir)")
    println("requested_device: $(settings.device)")
    println("resolved_device: $(resolved_device)")
    println("model_shape: layers=$(settings.num_layers) heads=$(settings.num_heads) emb=$(settings.embedding_size) ffn=$(settings.ffn_hidden_size) context=$(settings.context_length)")
    println("estimated_parameters: $(estimate_gpt2_parameter_count(config))")
    println("batch_size: $(settings.batch_size)")
    println("gradient_accumulation_steps: $(settings.gradient_accumulation_steps)")
    println("checkpoint_policy: one base-before-SFT checkpoint; final model is bundle weights")
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
    append!(
        stage_metrics,
        train_v6_streaming_stage!(
            trainer,
            "pretrain",
            () -> stream_lm_batches_from_text_file(
                paths.pretrain_training,
                tokenizer;
                context_length = settings.context_length,
                batch_size = settings.batch_size,
                document_separator = settings.document_separator,
                loss_mode = :all_tokens,
                chat_special_tokens = settings.chat_special_tokens,
                text_limit = settings.pretrain_text_limit,
            ),
            validation_sets["pretrain"],
            settings.pretrain_epochs,
            settings,
            progress_path,
            base_checkpoint_path,
            base_checkpoint_saved,
        ),
    )
    append!(
        stage_metrics,
        train_v6_streaming_stage!(
            trainer,
            "chat_lm",
            () -> stream_lm_batches_from_text_file(
                paths.chat_training,
                tokenizer;
                context_length = settings.context_length,
                batch_size = settings.batch_size,
                document_separator = settings.document_separator,
                loss_mode = :all_tokens,
                chat_special_tokens = settings.chat_special_tokens,
                text_limit = settings.chat_lm_text_limit,
            ),
            validation_sets["chat_lm"],
            settings.chat_lm_epochs,
            settings,
            progress_path,
            base_checkpoint_path,
            base_checkpoint_saved,
        ),
    )

    if settings.save_base_checkpoint && !base_checkpoint_saved[]
        save_checkpoint(
            base_checkpoint_path,
            trainer,
            model;
            experiment = V6_EXPERIMENT_NAME,
            stage = "base_before_sft",
            step = trainer.step,
        )
        base_checkpoint_saved[] = true
        println("saved base-before-SFT checkpoint: $(base_checkpoint_path)")
    end

    append!(
        stage_metrics,
        train_v6_streaming_stage!(
            trainer,
            "sft",
            () -> stream_clean_sft_batches_from_jsonl(
                paths.sft_training,
                tokenizer;
                context_length = settings.context_length,
                batch_size = settings.batch_size,
                pad_token_id = pad_token_id,
                example_limit = settings.sft_train_example_limit,
            ),
            validation_sets["sft"],
            settings.sft_epochs,
            settings,
            progress_path,
            base_checkpoint_path,
            base_checkpoint_saved,
        ),
    )

    if settings.save_final_checkpoint
        final_checkpoint_path = joinpath(checkpoint_dir, "final_checkpoint.jld2")
        save_checkpoint(final_checkpoint_path, trainer, model; experiment = V6_EXPERIMENT_NAME, stage = "final")
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
        "experiment" => V6_EXPERIMENT_NAME,
        "purpose" => V6_PURPOSE,
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
            "curriculum" => ["streamed_pretrain", "streamed_chat_lm", "streamed_sft"],
            "optimizer_name" => "Flux.Adam",
            "learning_rate" => settings.learning_rate,
            "batch_size" => settings.batch_size,
            "gradient_accumulation_steps" => settings.gradient_accumulation_steps,
            "effective_batch_size" => settings.batch_size * settings.gradient_accumulation_steps,
            "pretrain_epochs" => settings.pretrain_epochs,
            "chat_lm_epochs" => settings.chat_lm_epochs,
            "sft_epochs" => settings.sft_epochs,
            "final_step" => trainer.step,
            "initial_validation_losses" => initial_validation,
            "test_losses" => test_losses,
            "base_checkpoint_saved" => base_checkpoint_saved[],
        ),
        "batch_stats" => Dict(
            "pretrain_validation" => Dict(pairs(pretrain_validation_stats)),
            "pretrain_test" => Dict(pairs(pretrain_test_stats)),
            "chat_lm_validation" => Dict(pairs(chat_validation_stats)),
            "chat_lm_test" => Dict(pairs(chat_test_stats)),
            "sft_validation" => Dict(pairs(sft_validation_stats)),
            "sft_test" => Dict(pairs(sft_test_stats)),
        ),
        "artifacts" => Dict(
            "tokenizer_bundle_dir" => tokenizer_bundle_dir,
            "bundle_dir" => bundle_dir,
            "base_checkpoint" => base_checkpoint_saved[] ? base_checkpoint_path : nothing,
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
        latest_checkpoint = base_checkpoint_saved[] ? base_checkpoint_path : nothing,
    )

    println("test_losses: $(test_losses)")
    println("tokenizer bundle: $(tokenizer_bundle_dir)")
    println("bundle export: $(bundle_dir)")
    println("base checkpoint: $(base_checkpoint_saved[] ? base_checkpoint_path : "not saved")")
    println("metrics: $(metrics_path)")
    println("samples: $(sample_path)")
    return metrics
end

function train_v6_streaming_stage!(
    trainer,
    stage_name::AbstractString,
    batch_source::Function,
    validation_batches,
    epochs::Int,
    settings::V6ScratchSettings,
    progress_path::AbstractString,
    base_checkpoint_path::AbstractString,
    base_checkpoint_saved::Base.RefValue{Bool},
)
    metrics = Dict{String,Any}[]
    epochs <= 0 && return metrics
    for epoch in 1:epochs
        epoch_losses = Float64[]
        update_losses = Float64[]
        microbatch_count = 0
        loss_target_count = 0
        accumulated_gradient = nothing
        accumulated_count = 0
        accumulated_loss = 0.0

        for (input_batch, target_batch, loss_mask_batch) in batch_source()
            batch_loss_targets = Int(sum(loss_mask_batch))
            batch_loss_targets > 0 || continue
            loss_value, gradient = v6_loss_and_gradient(trainer, input_batch, target_batch, loss_mask_batch)
            push!(epoch_losses, loss_value)
            microbatch_count += 1
            loss_target_count += batch_loss_targets
            accumulated_loss += loss_value
            accumulated_count += 1
            accumulated_gradient = accumulated_gradient === nothing ? gradient : v6_gradient_add(accumulated_gradient, gradient)

            if accumulated_count >= settings.gradient_accumulation_steps
                update_loss = accumulated_loss / accumulated_count
                v6_apply_gradient_update!(trainer, accumulated_gradient, accumulated_count)
                push!(update_losses, update_loss)
                accumulated_gradient = nothing
                accumulated_count = 0
                accumulated_loss = 0.0

                if settings.log_every_updates > 0 && trainer.step % settings.log_every_updates == 0
                    recent_count = min(length(update_losses), settings.log_every_updates)
                    recent_window = update_losses[(end - recent_count + 1):end]
                    recent_train_loss = sum(recent_window) / length(recent_window)
                    write_progress(
                        progress_path;
                        status = "running",
                        epoch = trainer.epoch,
                        step = trainer.step,
                        latest_train_loss = recent_train_loss,
                        latest_validation_loss = nothing,
                        latest_checkpoint = base_checkpoint_saved[] ? base_checkpoint_path : nothing,
                    )
                    println(@sprintf(
                        "stage %s  update %d  epoch %d/%d  recent_train_loss=%.4f",
                        stage_name,
                        trainer.step,
                        epoch,
                        epochs,
                        recent_train_loss,
                    ))
                end
            end
        end

        if accumulated_count > 0
            update_loss = accumulated_loss / accumulated_count
            v6_apply_gradient_update!(trainer, accumulated_gradient, accumulated_count)
            push!(update_losses, update_loss)
        end

        isempty(epoch_losses) && throw(ArgumentError("stage $(stage_name) epoch $(epoch) did not yield any trainable batches"))
        trainer.epoch += 1
        train_loss = sum(epoch_losses) / length(epoch_losses)
        validation_loss = masked_mean_loss(trainer.model, validation_batches)
        push!(
            metrics,
            Dict(
                "stage" => String(stage_name),
                "stage_epoch" => epoch,
                "global_epoch" => trainer.epoch,
                "step" => trainer.step,
                "microbatch_count" => microbatch_count,
                "optimizer_update_count" => length(update_losses),
                "loss_target_count" => loss_target_count,
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
            latest_checkpoint = base_checkpoint_saved[] ? base_checkpoint_path : nothing,
        )
        println(@sprintf(
            "stage %s  epoch %d/%d  microbatches=%d  updates=%d  train_loss=%.4f  validation_loss=%.4f",
            stage_name,
            epoch,
            epochs,
            microbatch_count,
            length(update_losses),
            train_loss,
            validation_loss,
        ))
    end
    return metrics
end

function v6_loss_and_gradient(
    trainer,
    input_token_ids::AbstractMatrix{<:Integer},
    target_token_ids::AbstractMatrix{<:Integer},
    loss_mask::AbstractMatrix{<:Real},
)
    size(input_token_ids) == size(target_token_ids) ||
        throw(ArgumentError("input_token_ids and target_token_ids must have the same shape"))
    size(loss_mask) == size(target_token_ids) ||
        throw(ArgumentError("loss_mask must have the same shape as target_token_ids"))
    trainer.optimizer === nothing && (trainer.optimizer = Flux.Descent(0.01f0))
    trainer.optimizer_state === nothing && (trainer.optimizer_state = Flux.setup(trainer.optimizer, trainer.model))
    trainer.backend == :unknown && (trainer.backend = :flux)

    loss_target_ids = KeemenaLM.FluxBackend.move_like(target_token_ids, trainer.model.token_embedding)
    loss_weights = KeemenaLM.FluxBackend.move_like(Float32.(loss_mask), trainer.model.token_embedding)
    loss_function(model) = begin
        logits, _ = KeemenaLM.Core.lm_forward(model, input_token_ids; cache = nothing, is_training = false)
        return KeemenaLM.Core.causal_lm_cross_entropy(logits, loss_target_ids, loss_weights)
    end
    loss_value, model_gradients = Flux.withgradient(loss_function, trainer.model)
    isfinite(loss_value) || throw(ArgumentError("training loss is not finite"))
    return Float64(loss_value), model_gradients[1]
end

function v6_apply_gradient_update!(trainer, accumulated_gradient, accumulation_count::Int)
    accumulation_count > 0 || throw(ArgumentError("accumulation_count must be > 0"))
    averaged_gradient = v6_gradient_scale(accumulated_gradient, Float32(1 / accumulation_count))
    Flux.update!(trainer.optimizer_state, trainer.model, averaged_gradient)
    trainer.step += 1
    return trainer.step
end

v6_gradient_add(::Nothing, ::Nothing) = nothing
v6_gradient_add(a::Nothing, b) = b
v6_gradient_add(a, b::Nothing) = a
v6_gradient_add(a::Number, b::Number) = a + b
v6_gradient_add(a::AbstractArray{<:Number}, b::AbstractArray{<:Number}) = a .+ b

function v6_gradient_add(a::NamedTuple, b::NamedTuple)
    keys(a) == keys(b) || throw(ArgumentError("cannot add gradients with different NamedTuple keys"))
    return NamedTuple{keys(a)}(ntuple(index -> v6_gradient_add(getfield(a, keys(a)[index]), getfield(b, keys(b)[index])), length(keys(a))))
end

function v6_gradient_add(a::AbstractVector{<:NamedTuple}, b::AbstractVector{<:NamedTuple})
    length(a) == length(b) || throw(ArgumentError("cannot add gradient vectors with different lengths"))
    return [v6_gradient_add(a[index], b[index]) for index in eachindex(a)]
end

v6_gradient_scale(a::Nothing, scale::Real) = nothing
v6_gradient_scale(a::Number, scale::Real) = a * scale
v6_gradient_scale(a::AbstractArray{<:Number}, scale::Real) = a .* scale

function v6_gradient_scale(a::NamedTuple, scale::Real)
    return NamedTuple{keys(a)}(ntuple(index -> v6_gradient_scale(getfield(a, keys(a)[index]), scale), length(keys(a))))
end

function v6_gradient_scale(a::AbstractVector{<:NamedTuple}, scale::Real)
    return [v6_gradient_scale(value, scale) for value in a]
end

function stream_lm_batches_from_text_file(
    path::AbstractString,
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer;
    context_length::Int,
    batch_size::Int,
    document_separator::AbstractString,
    loss_mode::Symbol,
    chat_special_tokens::Dict{Symbol,String},
    text_limit::Int = 0,
)
    return Channel{Tuple{Matrix{Int32},Matrix{Int32},Matrix{Float32}}}(1) do channel
        token_buffer = Int[]
        mask_buffer = Float32[]
        separator_token_ids = KeemenaSubwords.encode(tokenizer, document_separator; add_special_tokens = false)
        input_batch = Matrix{Int32}(undef, context_length, batch_size)
        target_batch = Matrix{Int32}(undef, context_length, batch_size)
        loss_mask_batch = Matrix{Float32}(undef, context_length, batch_size)
        batch_column = 0

        function flush_batch!()
            if batch_column > 0
                put!(channel, (copy(input_batch[:, 1:batch_column]), copy(target_batch[:, 1:batch_column]), copy(loss_mask_batch[:, 1:batch_column])))
            end
            return nothing
        end

        function maybe_emit_examples!()
            while length(token_buffer) > context_length
                target_loss_mask = Float32.(mask_buffer[2:(context_length + 1)])
                target_loss_count = sum(target_loss_mask)
                if target_loss_count > 0
                    batch_column += 1
                    input_batch[:, batch_column] = Int32.(token_buffer[1:context_length])
                    target_batch[:, batch_column] = Int32.(token_buffer[2:(context_length + 1)])
                    loss_mask_batch[:, batch_column] = target_loss_mask
                    if batch_column == batch_size
                        put!(channel, (copy(input_batch), copy(target_batch), copy(loss_mask_batch)))
                        batch_column = 0
                    end
                end
                deleteat!(token_buffer, 1:context_length)
                deleteat!(mask_buffer, 1:context_length)
            end
            return nothing
        end

        text_count = 0
        foreach_prepared_paragraph(path) do text
            text_count += 1
            if text_limit > 0 && text_count > text_limit
                return false
            end
            text_token_ids = KeemenaSubwords.encode(tokenizer, text; add_special_tokens = false)
            text_loss_mask = token_loss_mask_for_text(
                text_token_ids,
                tokenizer;
                loss_mode = loss_mode,
                chat_special_tokens = chat_special_tokens,
            )
            append!(token_buffer, text_token_ids)
            append!(mask_buffer, text_loss_mask)
            if !isempty(separator_token_ids)
                append!(token_buffer, separator_token_ids)
                append!(mask_buffer, zeros(Float32, length(separator_token_ids)))
            end
            maybe_emit_examples!()
            return true
        end
        flush_batch!()
    end
end

function stream_clean_sft_batches_from_jsonl(
    path::AbstractString,
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer;
    context_length::Int,
    batch_size::Int,
    pad_token_id::Int,
    example_limit::Int = 0,
)
    return Channel{Tuple{Matrix{Int32},Matrix{Int32},Matrix{Float32}}}(1) do channel
        input_batch = fill(Int32(pad_token_id), context_length, batch_size)
        target_batch = fill(Int32(pad_token_id), context_length, batch_size)
        loss_mask_batch = zeros(Float32, context_length, batch_size)
        batch_column = 0
        example_count = 0

        function flush_batch!()
            if batch_column > 0
                put!(channel, (copy(input_batch[:, 1:batch_column]), copy(target_batch[:, 1:batch_column]), copy(loss_mask_batch[:, 1:batch_column])))
            end
            return nothing
        end

        open(path, "r") do io
            for line in eachline(io)
                stripped = strip(line)
                isempty(stripped) && continue
                example_count += 1
                if example_limit > 0 && example_count > example_limit
                    break
                end
                row = JSON3.read(stripped)
                example = CleanSFTExample(
                    hasproperty(row, :id) ? String(row.id) : string("example_", example_count),
                    String(row.prompt_text),
                    String(row.target_text),
                    String(row.assistant_text),
                    String(row.chat_text),
                )
                window = encode_clean_sft_window(example, tokenizer; context_length = context_length, pad_token_id = pad_token_id)
                window.status === :ok || continue

                batch_column += 1
                input_batch[:, batch_column] = window.input_ids
                target_batch[:, batch_column] = window.target_ids
                loss_mask_batch[:, batch_column] = window.loss_mask
                if batch_column == batch_size
                    put!(channel, (copy(input_batch), copy(target_batch), copy(loss_mask_batch)))
                    fill!(input_batch, Int32(pad_token_id))
                    fill!(target_batch, Int32(pad_token_id))
                    fill!(loss_mask_batch, 0.0f0)
                    batch_column = 0
                end
            end
        end
        flush_batch!()
    end
end

function foreach_prepared_paragraph(callback::Function, path::AbstractString)
    open(path, "r") do io
        buffer = IOBuffer()
        for line in eachline(io, keep = true)
            if isempty(strip(line))
                paragraph = strip(String(take!(buffer)))
                if !isempty(paragraph)
                    callback(paragraph) || return nothing
                end
            else
                write(buffer, line)
            end
        end
        if position(buffer) > 0
            paragraph = strip(String(take!(buffer)))
            if !isempty(paragraph)
                callback(paragraph)
            end
        end
    end
    return nothing
end

function v6_tokenizer_training_texts(paths, settings::V6ScratchSettings)::Vector{String}
    limit = settings.tokenizer_training_text_limit
    pretrain_limit = limit <= 0 ? 0 : max(1, round(Int, limit * 0.65))
    chat_limit = limit <= 0 ? 0 : max(1, round(Int, limit * 0.20))
    sft_limit = limit <= 0 ? 0 : max(1, limit - pretrain_limit - chat_limit)

    pretrain_texts = read_prepared_paragraphs(paths.pretrain_training; limit = optional_text_limit(pretrain_limit))
    chat_lm_texts = read_prepared_paragraphs(paths.chat_training; limit = optional_text_limit(chat_limit))
    sft_examples = read_clean_sft_examples(paths.sft_training; limit = sft_limit)
    texts = String[]
    append!(texts, pretrain_texts)
    append!(texts, chat_lm_texts)
    append!(texts, [example.chat_text for example in sft_examples])
    isempty(texts) && throw(ArgumentError("tokenizer training texts are empty"))
    return limit <= 0 ? texts : texts[1:min(limit, length(texts))]
end

function v6_dataset_paths(dataset_dir::AbstractString)
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

function validate_v6_settings(settings::V6ScratchSettings)
    settings.device in (:auto, :cpu, :gpu) || throw(ArgumentError("device must be one of :auto, :cpu, or :gpu"))
    settings.context_length > 0 || throw(ArgumentError("context_length must be > 0"))
    settings.batch_size > 0 || throw(ArgumentError("batch_size must be > 0"))
    settings.gradient_accumulation_steps > 0 || throw(ArgumentError("gradient_accumulation_steps must be > 0"))
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
    settings.sft_validation_example_limit > 0 || throw(ArgumentError("sft_validation_example_limit must be > 0"))
    settings.sft_test_example_limit > 0 || throw(ArgumentError("sft_test_example_limit must be > 0"))
    settings.validation_batch_limit >= 0 || throw(ArgumentError("validation_batch_limit must be >= 0"))
    settings.test_batch_limit >= 0 || throw(ArgumentError("test_batch_limit must be >= 0"))
    settings.log_every_updates >= 0 || throw(ArgumentError("log_every_updates must be >= 0"))
    settings.audit_example_count >= 0 || throw(ArgumentError("audit_example_count must be >= 0"))
    return settings
end

function v6_recipe_dict(dataset_dir::AbstractString, output_dir::AbstractString, settings::V6ScratchSettings)
    config = GPT2Config(
        vocab_size = settings.tokenizer_vocab_size,
        context_length = settings.context_length,
        num_layers = settings.num_layers,
        num_heads = settings.num_heads,
        embedding_size = settings.embedding_size,
        ffn_hidden_size = settings.ffn_hidden_size,
    )
    return Dict(
        "experiment" => V6_EXPERIMENT_NAME,
        "purpose" => V6_PURPOSE,
        "dataset_dir" => abspath(dataset_dir),
        "output_dir" => abspath(output_dir),
        "backend" => "flux",
        "device" => String(settings.device),
        "curriculum" => ["streamed_pretrain_all_tokens", "streamed_chat_lm_all_tokens", "streamed_sft_assistant_only"],
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
        "gradient_accumulation_steps" => settings.gradient_accumulation_steps,
        "effective_batch_size" => settings.batch_size * settings.gradient_accumulation_steps,
        "pretrain_epochs" => settings.pretrain_epochs,
        "chat_lm_epochs" => settings.chat_lm_epochs,
        "sft_epochs" => settings.sft_epochs,
        "sample_generation_tokens" => settings.sample_generation_tokens,
        "checkpoint_policy" => "one base-before-SFT checkpoint by default; final bundle only unless --save-final-checkpoint is passed",
        "save_base_checkpoint" => settings.save_base_checkpoint,
        "save_final_checkpoint" => settings.save_final_checkpoint,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !any(argument -> argument in ("--help", "-h"), ARGS)
        command_allows_gpu_device(ARGS) && KeemenaLM.FluxBackend.has_functional_cuda_gpu()
    end
    Base.invokelatest(main_v6, ARGS)
end
