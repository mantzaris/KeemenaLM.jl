#!/usr/bin/env julia

include(joinpath(@__DIR__, "run_tiny_chatbot_v6_scratch.jl"))

using Flux
using JSON3
using KeemenaLM
using KeemenaSubwords
using Printf
using Random

const V7_DATASET_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_v7_scratch_corpus")
const V7_OUTPUT_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_v7_scratch_candidate_run")
const V7_EXPERIMENT_NAME = "tiny_chatbot_v7_scratch_candidate_run"
const V7_PURPOSE = "v7 scratch chatbot: larger pretrain data, small SFT steering, mixed replay final stage with early stopping"

Base.@kwdef mutable struct V7ScratchSettings
    model_seed::Int = 20260527
    generation_seed::Int = 20260528
    device::Symbol = :auto
    context_length::Int = 512
    batch_size::Int = 1
    gradient_accumulation_steps::Int = 4
    pretrain_epochs::Int = 1
    mixed_max_updates::Int = 35_000
    mixed_min_updates::Int = 2_000
    validation_every_updates::Int = 2_000
    learning_rate::Float32 = 0.00006f0
    num_layers::Int = 24
    num_heads::Int = 16
    embedding_size::Int = 1024
    ffn_hidden_size::Int = 4096
    mixed_pretrain_parts::Int = 7
    mixed_chat_lm_parts::Int = 2
    mixed_sft_parts::Int = 1
    sft_early_stop_loss::Float64 = 1.20
    pretrain_max_relative_degradation::Float64 = 1.25
    chat_lm_max_relative_degradation::Float64 = 1.35
    sample_generation_tokens::Int = 180
    tokenizer_trainer::Symbol = :hf_gpt2_bytebpe
    tokenizer_vocab_size::Int = 32_768
    tokenizer_min_frequency::Int = 2
    tokenizer_model_name::String = "tiny_chatbot_v7_scratch_gpt2_bytebpe"
    tokenizer_training_text_limit::Int = 180_000
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

function main_v7(args)
    dataset_dir = V7_DATASET_DIR
    output_dir = V7_OUTPUT_DIR
    settings = V7ScratchSettings()

    argument_index = 1
    while argument_index <= length(args)
        argument = args[argument_index]
        if argument in ("--help", "-h")
            print_v7_usage()
            return nothing
        elseif argument == "--dataset-dir"
            argument_index += 1
            dataset_dir = abspath(args[argument_index])
        elseif argument == "--output-dir"
            argument_index += 1
            output_dir = abspath(args[argument_index])
        elseif argument == "--device"
            argument_index += 1
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
        elseif argument == "--mixed-max-updates"
            argument_index += 1
            settings.mixed_max_updates = parse(Int, args[argument_index])
        elseif argument == "--mixed-min-updates"
            argument_index += 1
            settings.mixed_min_updates = parse(Int, args[argument_index])
        elseif argument == "--validation-every-updates"
            argument_index += 1
            settings.validation_every_updates = parse(Int, args[argument_index])
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
        elseif argument == "--mixed-pretrain-parts"
            argument_index += 1
            settings.mixed_pretrain_parts = parse(Int, args[argument_index])
        elseif argument == "--mixed-chat-lm-parts"
            argument_index += 1
            settings.mixed_chat_lm_parts = parse(Int, args[argument_index])
        elseif argument == "--mixed-sft-parts"
            argument_index += 1
            settings.mixed_sft_parts = parse(Int, args[argument_index])
        elseif argument == "--sft-early-stop-loss"
            argument_index += 1
            settings.sft_early_stop_loss = parse(Float64, args[argument_index])
        elseif argument == "--pretrain-max-relative-degradation"
            argument_index += 1
            settings.pretrain_max_relative_degradation = parse(Float64, args[argument_index])
        elseif argument == "--chat-lm-max-relative-degradation"
            argument_index += 1
            settings.chat_lm_max_relative_degradation = parse(Float64, args[argument_index])
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
        elseif argument == "--validation-batch-limit"
            argument_index += 1
            settings.validation_batch_limit = parse(Int, args[argument_index])
        elseif argument == "--test-batch-limit"
            argument_index += 1
            settings.test_batch_limit = parse(Int, args[argument_index])
        elseif argument == "--log-every-updates"
            argument_index += 1
            settings.log_every_updates = parse(Int, args[argument_index])
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

    return run_v7_scratch(dataset_dir, output_dir; settings = settings)
end

function print_v7_usage()
    println("""
usage: julia --project=tools/subword_real_text tools/run_tiny_chatbot_v7_scratch.jl [options]

Prepare the corpus first:
  python3 tools/prepare_tiny_chatbot_v7_scratch_corpus.py --output-dir tmp/tiny_chatbot_v7_scratch_corpus

Important defaults:
  model: 24 layers, 16 heads, 1024 embedding, 4096 FFN, context 512
  pretrain: streamed pretrain epoch
  final stage: mixed replay, default 70% pretrain / 20% chat LM / 10% SFT
  early stopping: stop mixed stage when SFT is good enough without pretrain/chat regression
  checkpointing: one base_before_mixed_checkpoint.jld2 by default; final output is bundle weights
""")
end

function run_v7_scratch(
    dataset_dir::AbstractString,
    output_dir::AbstractString;
    settings::V7ScratchSettings = V7ScratchSettings(),
)
    validate_v7_settings(settings)
    dataset_dir = abspath(dataset_dir)
    output_dir = abspath(output_dir)

    paths = v7_dataset_paths(dataset_dir)
    for path in values(paths)
        isfile(path) || throw(ArgumentError("v7 dataset file does not exist: $(path). Prepare v7 data first."))
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
    write_json(joinpath(output_dir, "run_recipe.json"), v7_recipe_dict(dataset_dir, output_dir, settings))

    tokenizer_training_texts = v7_tokenizer_training_texts(paths, settings)
    tokenizer_settings = TinyChatbotSubwordSettings(
        model_seed = settings.model_seed,
        generation_seed = settings.generation_seed,
        device = settings.device,
        loss_mode = :assistant_only,
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        epochs = settings.pretrain_epochs + 1,
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

    validation_sets, test_sets, batch_stats = v7_build_eval_sets(paths, tokenizer, pad_token_id, settings)
    sft_audit_examples = read_clean_sft_examples(paths.sft_training; limit = max(settings.audit_example_count, 1))
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
            "experiment" => V7_EXPERIMENT_NAME,
            "corpus_source" => "tiny_chatbot_v7_scratch_corpus",
            "tokenizer_bundle_dir" => tokenizer_bundle_dir,
            "prepared_corpus_metadata_path" => paths.metadata,
            "optimizer_name" => "Flux.Adam",
            "requested_device" => String(settings.device),
            "resolved_device" => String(resolved_device),
            "training_curriculum" => "streamed_pretrain_all_tokens -> mixed_replay(pretrain/chat_lm/sft)",
            "gradient_accumulation_steps" => settings.gradient_accumulation_steps,
        ),
    )

    base_checkpoint_path = joinpath(checkpoint_dir, "base_before_mixed_checkpoint.jld2")
    final_checkpoint_path = nothing
    base_checkpoint_saved = Ref(false)

    println("== v7 scratch chatbot curriculum ==")
    println("prepared_dataset_dir: $(dataset_dir)")
    println("output_dir: $(output_dir)")
    println("requested_device: $(settings.device)")
    println("resolved_device: $(resolved_device)")
    println("model_shape: layers=$(settings.num_layers) heads=$(settings.num_heads) emb=$(settings.embedding_size) ffn=$(settings.ffn_hidden_size) context=$(settings.context_length)")
    println("estimated_parameters: $(estimate_gpt2_parameter_count(config))")
    println("batch_size: $(settings.batch_size)")
    println("gradient_accumulation_steps: $(settings.gradient_accumulation_steps)")
    println("mixed_ratio_parts: pretrain=$(settings.mixed_pretrain_parts) chat_lm=$(settings.mixed_chat_lm_parts) sft=$(settings.mixed_sft_parts)")
    println("mixed_max_updates: $(settings.mixed_max_updates)")
    println("checkpoint_policy: one base-before-mixed checkpoint; final model is bundle weights")
    println("sft_audit: $(audit_path)")

    initial_validation = Dict(stage => masked_mean_loss(model, batches) for (stage, batches) in validation_sets)
    write_progress(
        progress_path;
        status = "running",
        epoch = 0,
        step = trainer.step,
        latest_train_loss = nothing,
        latest_validation_loss = initial_validation,
        latest_checkpoint = nothing,
    )

    v6_settings = v7_as_v6_settings(settings)
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
            v6_settings,
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
            experiment = V7_EXPERIMENT_NAME,
            stage = "base_before_mixed",
            step = trainer.step,
        )
        base_checkpoint_saved[] = true
        println("saved base-before-mixed checkpoint: $(base_checkpoint_path)")
    end

    mixed_result = train_v7_mixed_stage!(
        trainer,
        tokenizer,
        paths,
        validation_sets,
        settings,
        progress_path,
        base_checkpoint_path,
        base_checkpoint_saved,
        pad_token_id,
    )
    push!(stage_metrics, mixed_result.summary)

    if settings.save_final_checkpoint
        final_checkpoint_path = joinpath(checkpoint_dir, "final_checkpoint.jld2")
        save_checkpoint(final_checkpoint_path, trainer, model; experiment = V7_EXPERIMENT_NAME, stage = "final")
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
        "experiment" => V7_EXPERIMENT_NAME,
        "purpose" => V7_PURPOSE,
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
            "curriculum" => ["streamed_pretrain", "mixed_replay"],
            "optimizer_name" => "Flux.Adam",
            "learning_rate" => settings.learning_rate,
            "batch_size" => settings.batch_size,
            "gradient_accumulation_steps" => settings.gradient_accumulation_steps,
            "effective_batch_size" => settings.batch_size * settings.gradient_accumulation_steps,
            "pretrain_epochs" => settings.pretrain_epochs,
            "mixed_max_updates" => settings.mixed_max_updates,
            "mixed_min_updates" => settings.mixed_min_updates,
            "mixed_ratio_parts" => Dict(
                "pretrain" => settings.mixed_pretrain_parts,
                "chat_lm" => settings.mixed_chat_lm_parts,
                "sft" => settings.mixed_sft_parts,
            ),
            "final_step" => trainer.step,
            "initial_validation_losses" => initial_validation,
            "mixed_validation_history" => mixed_result.validation_history,
            "mixed_stop_reason" => mixed_result.stop_reason,
            "test_losses" => test_losses,
            "base_checkpoint_saved" => base_checkpoint_saved[],
        ),
        "batch_stats" => batch_stats,
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
        latest_validation_loss = isempty(mixed_result.validation_history) ? initial_validation : mixed_result.validation_history[end]["validation_loss"],
        latest_checkpoint = base_checkpoint_saved[] ? base_checkpoint_path : nothing,
    )

    println("test_losses: $(test_losses)")
    println("mixed_stop_reason: $(mixed_result.stop_reason)")
    println("tokenizer bundle: $(tokenizer_bundle_dir)")
    println("bundle export: $(bundle_dir)")
    println("base checkpoint: $(base_checkpoint_saved[] ? base_checkpoint_path : "not saved")")
    println("metrics: $(metrics_path)")
    println("samples: $(sample_path)")
    return metrics
end

mutable struct V7RestartingBatchSource
    name::Symbol
    factory::Function
    channel::Any
    iterator_state::Any
end

function V7RestartingBatchSource(name::Symbol, factory::Function)
    return V7RestartingBatchSource(name, factory, factory(), nothing)
end

function next_v7_batch!(source::V7RestartingBatchSource)
    result = source.iterator_state === nothing ? iterate(source.channel) : iterate(source.channel, source.iterator_state)
    if result === nothing
        source.channel = source.factory()
        source.iterator_state = nothing
        result = iterate(source.channel)
        result === nothing && throw(ArgumentError("mixed source $(source.name) did not yield any batches"))
    end
    batch, state = result
    source.iterator_state = state
    return batch
end

function train_v7_mixed_stage!(
    trainer,
    tokenizer,
    paths,
    validation_sets,
    settings::V7ScratchSettings,
    progress_path::AbstractString,
    base_checkpoint_path::AbstractString,
    base_checkpoint_saved::Base.RefValue{Bool},
    pad_token_id::Int,
)
    pretrain_source = V7RestartingBatchSource(
        :pretrain,
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
    )
    chat_source = V7RestartingBatchSource(
        :chat_lm,
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
    )
    sft_source = V7RestartingBatchSource(
        :sft,
        () -> stream_clean_sft_batches_from_jsonl(
            paths.sft_training,
            tokenizer;
            context_length = settings.context_length,
            batch_size = settings.batch_size,
            pad_token_id = pad_token_id,
            example_limit = settings.sft_train_example_limit,
        ),
    )
    sources = Dict(:pretrain => pretrain_source, :chat_lm => chat_source, :sft => sft_source)
    cycle = v7_mixed_cycle(settings)
    validation_baseline = Dict(stage => masked_mean_loss(trainer.model, batches) for (stage, batches) in validation_sets)

    epoch_losses = Float64[]
    update_losses = Float64[]
    source_microbatches = Dict(:pretrain => 0, :chat_lm => 0, :sft => 0)
    source_loss_targets = Dict(:pretrain => 0, :chat_lm => 0, :sft => 0)
    accumulated_gradient = nothing
    accumulated_count = 0
    accumulated_loss = 0.0
    mixed_updates = 0
    microbatch_index = 0
    validation_history = Dict{String,Any}[]
    stop_reason = "max_mixed_updates_reached"

    while mixed_updates < settings.mixed_max_updates
        microbatch_index += 1
        source_name = cycle[((microbatch_index - 1) % length(cycle)) + 1]
        input_batch, target_batch, loss_mask_batch = next_v7_batch!(sources[source_name])
        batch_loss_targets = Int(sum(loss_mask_batch))
        batch_loss_targets > 0 || continue

        loss_value, gradient = v6_loss_and_gradient(trainer, input_batch, target_batch, loss_mask_batch)
        push!(epoch_losses, loss_value)
        source_microbatches[source_name] += 1
        source_loss_targets[source_name] += batch_loss_targets
        accumulated_loss += loss_value
        accumulated_count += 1
        accumulated_gradient = accumulated_gradient === nothing ? gradient : v6_gradient_add(accumulated_gradient, gradient)

        if accumulated_count >= settings.gradient_accumulation_steps
            update_loss = accumulated_loss / accumulated_count
            v6_apply_gradient_update!(trainer, accumulated_gradient, accumulated_count)
            mixed_updates += 1
            push!(update_losses, update_loss)
            accumulated_gradient = nothing
            accumulated_count = 0
            accumulated_loss = 0.0

            if settings.log_every_updates > 0 && mixed_updates % settings.log_every_updates == 0
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
                    "stage mixed  update %d/%d  global_step %d  recent_train_loss=%.4f",
                    mixed_updates,
                    settings.mixed_max_updates,
                    trainer.step,
                    recent_train_loss,
                ))
            end

            if settings.validation_every_updates > 0 && mixed_updates % settings.validation_every_updates == 0
                validation_loss = Dict(stage => masked_mean_loss(trainer.model, batches) for (stage, batches) in validation_sets)
                record = Dict(
                    "mixed_updates" => mixed_updates,
                    "global_step" => trainer.step,
                    "validation_loss" => validation_loss,
                )
                push!(validation_history, record)
                println(@sprintf(
                    "stage mixed  validation update %d/%d  pretrain=%.4f  chat_lm=%.4f  sft=%.4f",
                    mixed_updates,
                    settings.mixed_max_updates,
                    validation_loss["pretrain"],
                    validation_loss["chat_lm"],
                    validation_loss["sft"],
                ))
                maybe_stop_reason = v7_mixed_stop_reason(validation_loss, validation_baseline, mixed_updates, settings)
                if maybe_stop_reason !== nothing
                    stop_reason = maybe_stop_reason
                    break
                end
            end
        end
    end

    if accumulated_count > 0 && mixed_updates < settings.mixed_max_updates
        update_loss = accumulated_loss / accumulated_count
        v6_apply_gradient_update!(trainer, accumulated_gradient, accumulated_count)
        mixed_updates += 1
        push!(update_losses, update_loss)
    end

    isempty(epoch_losses) && throw(ArgumentError("mixed stage did not yield any trainable batches"))
    final_validation = Dict(stage => masked_mean_loss(trainer.model, batches) for (stage, batches) in validation_sets)
    push!(
        validation_history,
        Dict(
            "mixed_updates" => mixed_updates,
            "global_step" => trainer.step,
            "validation_loss" => final_validation,
            "final" => true,
        ),
    )
    trainer.epoch += 1
    train_loss = sum(epoch_losses) / length(epoch_losses)
    summary = Dict(
        "stage" => "mixed_replay",
        "global_epoch" => trainer.epoch,
        "step" => trainer.step,
        "mixed_updates" => mixed_updates,
        "microbatch_count" => sum(values(source_microbatches)),
        "source_microbatches" => Dict(String(key) => value for (key, value) in pairs(source_microbatches)),
        "source_loss_targets" => Dict(String(key) => value for (key, value) in pairs(source_loss_targets)),
        "train_loss" => train_loss,
        "train_perplexity" => exp(train_loss),
        "validation_loss" => final_validation,
        "stop_reason" => stop_reason,
    )
    write_progress(
        progress_path;
        status = "running",
        epoch = trainer.epoch,
        step = trainer.step,
        latest_train_loss = train_loss,
        latest_validation_loss = final_validation,
        latest_checkpoint = base_checkpoint_saved[] ? base_checkpoint_path : nothing,
    )
    println(@sprintf(
        "stage mixed  completed updates=%d  train_loss=%.4f  pretrain_val=%.4f  chat_lm_val=%.4f  sft_val=%.4f  stop_reason=%s",
        mixed_updates,
        train_loss,
        final_validation["pretrain"],
        final_validation["chat_lm"],
        final_validation["sft"],
        stop_reason,
    ))
    return (summary = summary, validation_history = validation_history, stop_reason = stop_reason)
end

function v7_mixed_stop_reason(validation_loss, validation_baseline, mixed_updates::Int, settings::V7ScratchSettings)
    mixed_updates < settings.mixed_min_updates && return nothing
    pretrain_limit = validation_baseline["pretrain"] * settings.pretrain_max_relative_degradation
    chat_limit = validation_baseline["chat_lm"] * settings.chat_lm_max_relative_degradation

    validation_loss["pretrain"] > pretrain_limit && return "pretrain_validation_regression"
    validation_loss["chat_lm"] > chat_limit && return "chat_lm_validation_regression"

    if validation_loss["sft"] <= settings.sft_early_stop_loss &&
       validation_loss["pretrain"] <= pretrain_limit &&
       validation_loss["chat_lm"] <= chat_limit
        return "sft_target_reached_without_replay_regression"
    end

    return nothing
end

function v7_mixed_cycle(settings::V7ScratchSettings)::Vector{Symbol}
    settings.mixed_pretrain_parts >= 0 || throw(ArgumentError("mixed_pretrain_parts must be >= 0"))
    settings.mixed_chat_lm_parts >= 0 || throw(ArgumentError("mixed_chat_lm_parts must be >= 0"))
    settings.mixed_sft_parts >= 0 || throw(ArgumentError("mixed_sft_parts must be >= 0"))
    cycle = Symbol[]
    append!(cycle, fill(:pretrain, settings.mixed_pretrain_parts))
    append!(cycle, fill(:chat_lm, settings.mixed_chat_lm_parts))
    append!(cycle, fill(:sft, settings.mixed_sft_parts))
    isempty(cycle) && throw(ArgumentError("at least one mixed source part must be > 0"))
    return cycle
end

function v7_build_eval_sets(paths, tokenizer, pad_token_id::Int, settings::V7ScratchSettings)
    pretrain_validation_texts = read_prepared_paragraphs(paths.pretrain_validation; limit = settings.validation_text_limit)
    pretrain_testing_texts = read_prepared_paragraphs(paths.pretrain_testing; limit = settings.test_text_limit)
    chat_validation_texts = read_prepared_paragraphs(paths.chat_validation; limit = settings.validation_text_limit)
    chat_testing_texts = read_prepared_paragraphs(paths.chat_testing; limit = settings.test_text_limit)
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
    batch_stats = Dict(
        "pretrain_validation" => Dict(pairs(pretrain_validation_stats)),
        "pretrain_test" => Dict(pairs(pretrain_test_stats)),
        "chat_lm_validation" => Dict(pairs(chat_validation_stats)),
        "chat_lm_test" => Dict(pairs(chat_test_stats)),
        "sft_validation" => Dict(pairs(sft_validation_stats)),
        "sft_test" => Dict(pairs(sft_test_stats)),
    )
    return validation_sets, test_sets, batch_stats
end

function v7_tokenizer_training_texts(paths, settings::V7ScratchSettings)::Vector{String}
    limit = settings.tokenizer_training_text_limit
    pretrain_limit = limit <= 0 ? 0 : max(1, round(Int, limit * 0.75))
    chat_limit = limit <= 0 ? 0 : max(1, round(Int, limit * 0.18))
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

function v7_as_v6_settings(settings::V7ScratchSettings)::V6ScratchSettings
    return V6ScratchSettings(
        model_seed = settings.model_seed,
        generation_seed = settings.generation_seed,
        device = settings.device,
        context_length = settings.context_length,
        batch_size = settings.batch_size,
        gradient_accumulation_steps = settings.gradient_accumulation_steps,
        pretrain_epochs = settings.pretrain_epochs,
        chat_lm_epochs = 0,
        sft_epochs = 0,
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
        tokenizer_training_text_limit = settings.tokenizer_training_text_limit,
        pretrain_text_limit = settings.pretrain_text_limit,
        chat_lm_text_limit = settings.chat_lm_text_limit,
        sft_train_example_limit = settings.sft_train_example_limit,
        validation_text_limit = settings.validation_text_limit,
        test_text_limit = settings.test_text_limit,
        sft_validation_example_limit = settings.sft_validation_example_limit,
        sft_test_example_limit = settings.sft_test_example_limit,
        validation_batch_limit = settings.validation_batch_limit,
        test_batch_limit = settings.test_batch_limit,
        log_every_updates = settings.log_every_updates,
        audit_example_count = settings.audit_example_count,
        save_base_checkpoint = settings.save_base_checkpoint,
        save_final_checkpoint = settings.save_final_checkpoint,
        reuse_tokenizer_bundle_dir = settings.reuse_tokenizer_bundle_dir,
        document_separator = settings.document_separator,
        chat_special_tokens = settings.chat_special_tokens,
    )
end

function v7_dataset_paths(dataset_dir::AbstractString)
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

function validate_v7_settings(settings::V7ScratchSettings)
    settings.device in (:auto, :cpu, :gpu) || throw(ArgumentError("device must be one of :auto, :cpu, or :gpu"))
    settings.context_length > 0 || throw(ArgumentError("context_length must be > 0"))
    settings.batch_size > 0 || throw(ArgumentError("batch_size must be > 0"))
    settings.gradient_accumulation_steps > 0 || throw(ArgumentError("gradient_accumulation_steps must be > 0"))
    settings.pretrain_epochs >= 0 || throw(ArgumentError("pretrain_epochs must be >= 0"))
    settings.mixed_max_updates > 0 || throw(ArgumentError("mixed_max_updates must be > 0"))
    settings.mixed_min_updates >= 0 || throw(ArgumentError("mixed_min_updates must be >= 0"))
    settings.validation_every_updates >= 0 || throw(ArgumentError("validation_every_updates must be >= 0"))
    settings.learning_rate > 0 || throw(ArgumentError("learning_rate must be > 0"))
    settings.num_layers > 0 || throw(ArgumentError("num_layers must be > 0"))
    settings.num_heads > 0 || throw(ArgumentError("num_heads must be > 0"))
    settings.embedding_size % settings.num_heads == 0 ||
        throw(ArgumentError("embedding_size must be divisible by num_heads"))
    settings.ffn_hidden_size > 0 || throw(ArgumentError("ffn_hidden_size must be > 0"))
    settings.sft_early_stop_loss > 0 || throw(ArgumentError("sft_early_stop_loss must be > 0"))
    settings.pretrain_max_relative_degradation >= 1.0 ||
        throw(ArgumentError("pretrain_max_relative_degradation must be >= 1.0"))
    settings.chat_lm_max_relative_degradation >= 1.0 ||
        throw(ArgumentError("chat_lm_max_relative_degradation must be >= 1.0"))
    settings.tokenizer_vocab_size > 0 || throw(ArgumentError("tokenizer_vocab_size must be > 0"))
    settings.tokenizer_training_text_limit >= 0 || throw(ArgumentError("tokenizer_training_text_limit must be >= 0"))
    settings.pretrain_text_limit >= 0 || throw(ArgumentError("pretrain_text_limit must be >= 0"))
    settings.chat_lm_text_limit >= 0 || throw(ArgumentError("chat_lm_text_limit must be >= 0"))
    settings.sft_train_example_limit >= 0 || throw(ArgumentError("sft_train_example_limit must be >= 0"))
    settings.validation_text_limit > 0 || throw(ArgumentError("validation_text_limit must be > 0"))
    settings.test_text_limit > 0 || throw(ArgumentError("test_text_limit must be > 0"))
    settings.validation_batch_limit >= 0 || throw(ArgumentError("validation_batch_limit must be >= 0"))
    settings.test_batch_limit >= 0 || throw(ArgumentError("test_batch_limit must be >= 0"))
    settings.log_every_updates >= 0 || throw(ArgumentError("log_every_updates must be >= 0"))
    v7_mixed_cycle(settings)
    return settings
end

function v7_recipe_dict(dataset_dir::AbstractString, output_dir::AbstractString, settings::V7ScratchSettings)
    config = GPT2Config(
        vocab_size = settings.tokenizer_vocab_size,
        context_length = settings.context_length,
        num_layers = settings.num_layers,
        num_heads = settings.num_heads,
        embedding_size = settings.embedding_size,
        ffn_hidden_size = settings.ffn_hidden_size,
    )
    return Dict(
        "experiment" => V7_EXPERIMENT_NAME,
        "purpose" => V7_PURPOSE,
        "dataset_dir" => abspath(dataset_dir),
        "output_dir" => abspath(output_dir),
        "backend" => "flux",
        "device" => String(settings.device),
        "curriculum" => ["streamed_pretrain_all_tokens", "mixed_replay"],
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
        "mixed_max_updates" => settings.mixed_max_updates,
        "mixed_min_updates" => settings.mixed_min_updates,
        "validation_every_updates" => settings.validation_every_updates,
        "mixed_ratio_parts" => Dict(
            "pretrain" => settings.mixed_pretrain_parts,
            "chat_lm" => settings.mixed_chat_lm_parts,
            "sft" => settings.mixed_sft_parts,
        ),
        "sft_early_stop_loss" => settings.sft_early_stop_loss,
        "checkpoint_policy" => "one base-before-mixed checkpoint by default; final bundle only unless --save-final-checkpoint is passed",
        "save_base_checkpoint" => settings.save_base_checkpoint,
        "save_final_checkpoint" => settings.save_final_checkpoint,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !any(argument -> argument in ("--help", "-h"), ARGS)
        command_allows_gpu_device(ARGS) && KeemenaLM.FluxBackend.has_functional_cuda_gpu()
    end
    Base.invokelatest(main_v7, ARGS)
end
