#!/usr/bin/env julia

include(joinpath(@__DIR__, "tiny_chatbot_training_common.jl"))
include(joinpath(@__DIR__, "run_tiny_chatbot_v8_behavior_eval.jl"))

using Flux
using JSON3
using KeemenaLM
using KeemenaSubwords
using Printf
using Random

const V8_DATASET_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_v9_broad_corpus_5k_anchor")
const V8_OUTPUT_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_v9_broad_336m_run_next")
const V8_EXPERIMENT_NAME = "tiny_chatbot_v9_broad_336m_candidate_run"
const V8_PURPOSE = "v8/v9 scratch chatbot: broad pretrain plus behavior-gated direct-answer assistant SFT with pretrain replay"

Base.@kwdef mutable struct V8ScratchSettings
    model_seed::Int = 20260615
    generation_seed::Int = 20260616
    device::Symbol = :auto
    context_length::Int = 512
    batch_size::Int = 1
    gradient_accumulation_steps::Int = 4
    pretrain_epochs::Int = 1
    behavior_max_updates::Int = 35_000
    behavior_min_updates::Int = 2_000
    validation_every_updates::Int = 1_000
    learning_rate::Float32 = 0.00006f0
    num_layers::Int = 24
    num_heads::Int = 16
    embedding_size::Int = 1024
    ffn_hidden_size::Int = 4096
    direct_sft_parts::Int = 3
    pretrain_replay_parts::Int = 2
    direct_sft_early_stop_loss::Float64 = 1.20
    pretrain_max_relative_degradation::Float64 = 1.25
    stop_on_pretrain_regression::Bool = true
    behavior_min_pass_rate::Float64 = 1.0
    behavior_max_new_tokens::Int = 100
    behavior_prompt_limit::Int = 0
    sample_generation_tokens::Int = 120
    tokenizer_trainer::Symbol = :hf_gpt2_bytebpe
    tokenizer_vocab_size::Int = 32_768
    tokenizer_min_frequency::Int = 2
    tokenizer_model_name::String = "tiny_chatbot_v9_broad_gpt2_bytebpe"
    tokenizer_training_text_limit::Int = 180_000
    pretrain_text_limit::Int = 0
    sft_train_example_limit::Int = 0
    validation_text_limit::Int = 1_000
    test_text_limit::Int = 1_000
    sft_validation_example_limit::Int = 2_000
    sft_test_example_limit::Int = 2_000
    validation_batch_limit::Int = 400
    test_batch_limit::Int = 400
    log_every_updates::Int = 25
    audit_example_count::Int = 12
    save_base_checkpoint::Bool = true
    save_final_checkpoint::Bool = false
    save_best_behavior_bundle::Bool = false
    keep_best_behavior_only::Bool = false
    reuse_tokenizer_bundle_dir::String = ""
    initial_bundle_dir::String = ""
    document_separator::String = TINY_CHATBOT_DOCUMENT_SEPARATOR
    chat_special_tokens::Dict{Symbol,String} = Dict(
        :unk => "<|endoftext|>",
        :user => CHAT_MARKERS.user,
        :assistant => CHAT_MARKERS.assistant,
        :end_assistant => CHAT_MARKERS.end_assistant,
        :chat_end => CHAT_MARKERS.chat_end,
    )
end

function main_v8(args)
    dataset_dir = V8_DATASET_DIR
    output_dir = V8_OUTPUT_DIR
    settings = V8ScratchSettings()

    argument_index = 1
    while argument_index <= length(args)
        argument = args[argument_index]
        if argument in ("--help", "-h")
            print_v8_usage()
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
        elseif argument == "--behavior-max-updates"
            argument_index += 1
            settings.behavior_max_updates = parse(Int, args[argument_index])
        elseif argument == "--behavior-min-updates"
            argument_index += 1
            settings.behavior_min_updates = parse(Int, args[argument_index])
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
        elseif argument == "--direct-sft-parts"
            argument_index += 1
            settings.direct_sft_parts = parse(Int, args[argument_index])
        elseif argument == "--pretrain-replay-parts"
            argument_index += 1
            settings.pretrain_replay_parts = parse(Int, args[argument_index])
        elseif argument == "--direct-sft-early-stop-loss"
            argument_index += 1
            settings.direct_sft_early_stop_loss = parse(Float64, args[argument_index])
        elseif argument == "--pretrain-max-relative-degradation"
            argument_index += 1
            settings.pretrain_max_relative_degradation = parse(Float64, args[argument_index])
        elseif argument == "--disable-pretrain-regression-stop"
            settings.stop_on_pretrain_regression = false
        elseif argument == "--behavior-min-pass-rate"
            argument_index += 1
            settings.behavior_min_pass_rate = parse(Float64, args[argument_index])
        elseif argument == "--behavior-max-new-tokens"
            argument_index += 1
            settings.behavior_max_new_tokens = parse(Int, args[argument_index])
        elseif argument == "--behavior-prompt-limit"
            argument_index += 1
            settings.behavior_prompt_limit = parse(Int, args[argument_index])
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
        elseif argument == "--initial-bundle-dir"
            argument_index += 1
            settings.initial_bundle_dir = abspath(args[argument_index])
        elseif argument == "--no-base-checkpoint"
            settings.save_base_checkpoint = false
        elseif argument == "--save-final-checkpoint"
            settings.save_final_checkpoint = true
        elseif argument == "--save-best-behavior-bundle"
            settings.save_best_behavior_bundle = true
        elseif argument == "--keep-best-behavior-only"
            settings.keep_best_behavior_only = true
        else
            error("unknown argument $(argument). Run with --help for usage.")
        end
        argument_index += 1
    end

    return run_v8_scratch(dataset_dir, output_dir; settings = settings)
end

function print_v8_usage()
    println("""
usage: julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_scratch.jl [options]

Prepare a full v9-style corpus first:
  tools/prepare_tiny_chatbot_v9_broad_corpus.sh

For smoke tests, prepare a small compatible corpus with tools/prepare_tiny_chatbot_real_chat_corpus.py and pass --dataset-dir explicitly

Important defaults:
  model: 24 layers, 16 heads, 1024 embedding, 4096 FFN, context 512
  curriculum: streamed pretrain all_tokens -> behavior stage with direct SFT plus pretrain replay
  no raw chat_lm stage is used
  stop condition: behavior gate passes, direct-SFT loss is low enough, and pretrain validation has not regressed
  disk: pass --no-base-checkpoint and optionally --save-best-behavior-bundle to keep at most final bundle + best behavior bundle
  low disk: add --keep-best-behavior-only with --save-best-behavior-bundle to delete the final bundle after evaluation when a best bundle exists
  continuation: pass --initial-bundle-dir DIR, --pretrain-epochs 0, and --reuse-tokenizer-bundle-dir DIR to continue behavior training from a saved bundle
""")
end

function run_v8_scratch(
    dataset_dir::AbstractString,
    output_dir::AbstractString;
    settings::V8ScratchSettings = V8ScratchSettings(),
)
    validate_v8_settings(settings)
    dataset_dir = abspath(dataset_dir)
    output_dir = abspath(output_dir)

    paths = v8_dataset_paths(dataset_dir)
    for path in values(paths)
        isfile(path) || throw(ArgumentError("v8/v9 dataset file does not exist: $(path). Prepare compatible v8/v9 data first."))
    end

    checkpoint_dir = joinpath(output_dir, "checkpoints")
    bundle_dir = joinpath(output_dir, "bundle")
    best_behavior_bundle_dir = joinpath(output_dir, "best_behavior_bundle")
    tokenizer_bundle_dir = joinpath(output_dir, "tokenizer_bundle")
    metrics_path = joinpath(output_dir, "metrics.json")
    progress_path = joinpath(output_dir, "progress.json")
    sample_path = joinpath(output_dir, "sample_outputs.txt")
    audit_path = joinpath(output_dir, "sft_data_mask_audit.txt")
    behavior_eval_path = joinpath(output_dir, "behavior_eval.json")
    mkpath(output_dir)
    mkpath(checkpoint_dir)

    metadata = JSON3.read(read(paths.metadata, String))
    write_json(joinpath(output_dir, "run_recipe.json"), v8_recipe_dict(dataset_dir, output_dir, settings))

    tokenizer_training_texts = v8_tokenizer_training_texts(paths, settings)
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

    validation_sets, test_sets, batch_stats = v8_build_eval_sets(paths, tokenizer, pad_token_id, settings)
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
    model = if isempty(settings.initial_bundle_dir)
        KeemenaLM.FluxBackend.move_model_to_device(
            instantiate(config; backend = :flux, seed = settings.model_seed);
            device = settings.device,
        )
    else
        loaded_model = load_model(settings.initial_bundle_dir; backend = :flux)
        loaded_config = KeemenaLM.Core.model_config(loaded_model)
        validate_v8_initial_bundle_config(loaded_config, config, settings.initial_bundle_dir)
        config = loaded_config
        KeemenaLM.FluxBackend.move_model_to_device(loaded_model; device = settings.device)
    end
    resolved_device = KeemenaLM.FluxBackend.is_cuda_array(model.token_embedding) ? :gpu : :cpu
    trainer = KeemenaLM.Core.Trainer(
        model;
        optimizer = Flux.Adam(settings.learning_rate),
        backend = :flux,
        metadata = Dict(
            "experiment" => V8_EXPERIMENT_NAME,
            "corpus_source" => "tiny_chatbot_v9_broad_corpus",
            "tokenizer_bundle_dir" => tokenizer_bundle_dir,
            "prepared_corpus_metadata_path" => paths.metadata,
            "optimizer_name" => "Flux.Adam",
            "requested_device" => String(settings.device),
            "resolved_device" => String(resolved_device),
            "training_curriculum" => "streamed_pretrain_all_tokens -> behavior_direct_sft_with_pretrain_replay",
            "gradient_accumulation_steps" => settings.gradient_accumulation_steps,
            "initial_bundle_dir" => settings.initial_bundle_dir,
        ),
    )

    base_checkpoint_path = joinpath(checkpoint_dir, "base_before_behavior_checkpoint.jld2")
    final_checkpoint_path = nothing
    base_checkpoint_saved = Ref(false)

    println("== v8/v9 broad scratch chatbot curriculum ==")
    println("prepared_dataset_dir: $(dataset_dir)")
    println("output_dir: $(output_dir)")
    println("requested_device: $(settings.device)")
    println("resolved_device: $(resolved_device)")
    println("model_shape: layers=$(settings.num_layers) heads=$(settings.num_heads) emb=$(settings.embedding_size) ffn=$(settings.ffn_hidden_size) context=$(settings.context_length)")
    println("estimated_parameters: $(estimate_gpt2_parameter_count(config))")
    println("behavior_ratio_parts: direct_sft=$(settings.direct_sft_parts) pretrain_replay=$(settings.pretrain_replay_parts)")
    println("initial_bundle_dir: $(isempty(settings.initial_bundle_dir) ? "none" : settings.initial_bundle_dir)")
    println("pretrain_regression_stop: $(settings.stop_on_pretrain_regression)")
    println("raw_chat_lm_stage: disabled")
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

    streaming_settings = v8_streaming_stage_settings(settings)
    stage_metrics = Dict{String,Any}[]
    append!(
        stage_metrics,
        train_streaming_stage!(
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
            streaming_settings,
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
            experiment = V8_EXPERIMENT_NAME,
            stage = "base_before_behavior",
            step = trainer.step,
        )
        base_checkpoint_saved[] = true
        println("saved base-before-behavior checkpoint: $(base_checkpoint_path)")
    end

    behavior_result = train_v8_behavior_stage!(
        trainer,
        tokenizer,
        paths,
        validation_sets,
        settings,
        progress_path,
        base_checkpoint_path,
        base_checkpoint_saved,
        pad_token_id,
        best_behavior_bundle_dir,
    )
    push!(stage_metrics, behavior_result.summary)

    if settings.save_final_checkpoint
        final_checkpoint_path = joinpath(checkpoint_dir, "final_checkpoint.jld2")
        save_checkpoint(final_checkpoint_path, trainer, model; experiment = V8_EXPERIMENT_NAME, stage = "final")
    end

    save_v8_model_bundle(
        bundle_dir,
        model;
        metadata = Dict(
            "kind" => "final",
            "experiment" => V8_EXPERIMENT_NAME,
            "step" => trainer.step,
        ),
    )
    reloaded_bundle = load_bundle(bundle_dir)
    reloaded_model = KeemenaLM.FluxBackend.move_model_to_device(
        instantiate(reloaded_bundle; backend = :flux);
        device = settings.device,
    )
    reloaded_tokenizer = KeemenaSubwords.load_training_bundle(tokenizer_bundle_dir)

    behavior_cases = v8_behavior_cases(settings)
    prompts = String[String(getfield(case, :prompt)) for case in behavior_cases]
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
    final_behavior_report = v8_score_model_behavior(
        reloaded_model,
        reloaded_tokenizer;
        cases = behavior_cases,
        max_new_tokens = settings.behavior_max_new_tokens,
        seed = settings.generation_seed,
    )
    open(behavior_eval_path, "w") do io
        JSON3.write(io, final_behavior_report)
    end

    final_bundle_removed_after_eval = settings.keep_best_behavior_only && behavior_result.best_behavior_bundle_dir !== nothing
    bundle_artifact_dir = final_bundle_removed_after_eval ? nothing : bundle_dir

    metrics = Dict(
        "experiment" => V8_EXPERIMENT_NAME,
        "purpose" => V8_PURPOSE,
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
            "curriculum" => ["streamed_pretrain", "behavior_direct_sft_with_pretrain_replay"],
            "disabled_curriculum" => ["raw_chat_lm_all_tokens"],
            "optimizer_name" => "Flux.Adam",
            "learning_rate" => settings.learning_rate,
            "batch_size" => settings.batch_size,
            "gradient_accumulation_steps" => settings.gradient_accumulation_steps,
            "effective_batch_size" => settings.batch_size * settings.gradient_accumulation_steps,
            "pretrain_epochs" => settings.pretrain_epochs,
            "behavior_max_updates" => settings.behavior_max_updates,
            "behavior_min_updates" => settings.behavior_min_updates,
            "behavior_ratio_parts" => Dict("direct_sft" => settings.direct_sft_parts, "pretrain_replay" => settings.pretrain_replay_parts),
            "final_step" => trainer.step,
            "initial_validation_losses" => initial_validation,
            "behavior_validation_history" => behavior_result.validation_history,
            "behavior_stop_reason" => behavior_result.stop_reason,
            "best_behavior_record" => behavior_result.best_behavior_record,
            "test_losses" => test_losses,
            "final_behavior_summary" => final_behavior_report["summary"],
            "base_checkpoint_saved" => base_checkpoint_saved[],
            "keep_best_behavior_only" => settings.keep_best_behavior_only,
            "final_bundle_removed_after_eval" => final_bundle_removed_after_eval,
        ),
        "batch_stats" => batch_stats,
        "artifacts" => Dict(
            "tokenizer_bundle_dir" => tokenizer_bundle_dir,
            "bundle_dir" => bundle_artifact_dir,
            "final_bundle_dir_before_cleanup" => final_bundle_removed_after_eval ? bundle_dir : nothing,
            "best_behavior_bundle_dir" => behavior_result.best_behavior_bundle_dir,
            "base_checkpoint" => base_checkpoint_saved[] ? base_checkpoint_path : nothing,
            "final_checkpoint" => final_checkpoint_path,
            "sample_outputs_path" => sample_path,
            "behavior_eval_path" => behavior_eval_path,
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
        latest_validation_loss = isempty(behavior_result.validation_history) ? initial_validation : behavior_result.validation_history[end]["validation_loss"],
        latest_checkpoint = base_checkpoint_saved[] ? base_checkpoint_path : nothing,
    )

    if final_bundle_removed_after_eval
        rm(bundle_dir; force = true, recursive = true)
    end

    bundle_export_message = final_bundle_removed_after_eval ? string(bundle_dir, " (removed after evaluation; use best behavior bundle)") : bundle_dir

    println("test_losses: $(test_losses)")
    println("behavior_stop_reason: $(behavior_result.stop_reason)")
    println("final_behavior_summary: $(final_behavior_report["summary"])")
    println("tokenizer bundle: $(tokenizer_bundle_dir)")
    println("bundle export: $(bundle_export_message)")
    println("best behavior bundle: $(behavior_result.best_behavior_bundle_dir === nothing ? "not saved" : behavior_result.best_behavior_bundle_dir)")
    println("base checkpoint: $(base_checkpoint_saved[] ? base_checkpoint_path : "not saved")")
    println("behavior eval: $(behavior_eval_path)")
    println("metrics: $(metrics_path)")
    println("samples: $(sample_path)")
    return metrics
end

function train_v8_behavior_stage!(
    trainer,
    tokenizer,
    paths,
    validation_sets,
    settings::V8ScratchSettings,
    progress_path::AbstractString,
    base_checkpoint_path::AbstractString,
    base_checkpoint_saved::Base.RefValue{Bool},
    pad_token_id::Int,
    best_behavior_bundle_dir::AbstractString,
)
    pretrain_source = RestartingBatchSource(
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
    sft_source = RestartingBatchSource(
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
    sources = Dict(:pretrain => pretrain_source, :sft => sft_source)
    cycle = v8_behavior_cycle(settings)
    validation_baseline = Dict(stage => masked_mean_loss(trainer.model, batches) for (stage, batches) in validation_sets)
    behavior_cases = v8_behavior_cases(settings)

    epoch_losses = Float64[]
    update_losses = Float64[]
    source_microbatches = Dict(:pretrain => 0, :sft => 0)
    source_loss_targets = Dict(:pretrain => 0, :sft => 0)
    accumulated_gradient = nothing
    accumulated_count = 0
    accumulated_loss = 0.0
    behavior_updates = 0
    microbatch_index = 0
    validation_history = Dict{String,Any}[]
    best_behavior_record = nothing
    saved_best_behavior_bundle_dir = nothing
    stop_reason = "max_behavior_updates_reached"

    while behavior_updates < settings.behavior_max_updates
        microbatch_index += 1
        source_name = cycle[((microbatch_index - 1) % length(cycle)) + 1]
        input_batch, target_batch, loss_mask_batch = next_restarting_batch!(sources[source_name])
        batch_loss_targets = Int(sum(loss_mask_batch))
        batch_loss_targets > 0 || continue

        loss_value, gradient = training_loss_and_gradient(trainer, input_batch, target_batch, loss_mask_batch)
        push!(epoch_losses, loss_value)
        source_microbatches[source_name] += 1
        source_loss_targets[source_name] += batch_loss_targets
        accumulated_loss += loss_value
        accumulated_count += 1
        accumulated_gradient = accumulated_gradient === nothing ? gradient : gradient_tree_add(accumulated_gradient, gradient)

        if accumulated_count >= settings.gradient_accumulation_steps
            update_loss = accumulated_loss / accumulated_count
            apply_accumulated_gradient_update!(trainer, accumulated_gradient, accumulated_count)
            behavior_updates += 1
            push!(update_losses, update_loss)
            accumulated_gradient = nothing
            accumulated_count = 0
            accumulated_loss = 0.0

            if settings.log_every_updates > 0 && behavior_updates % settings.log_every_updates == 0
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
                    "stage behavior  update %d/%d  global_step %d  recent_train_loss=%.4f",
                    behavior_updates,
                    settings.behavior_max_updates,
                    trainer.step,
                    recent_train_loss,
                ))
            end

            if settings.validation_every_updates > 0 && behavior_updates % settings.validation_every_updates == 0
                validation_loss = Dict(stage => masked_mean_loss(trainer.model, batches) for (stage, batches) in validation_sets)
                behavior_report = v8_score_model_behavior(
                    trainer.model,
                    tokenizer;
                    cases = behavior_cases,
                    max_new_tokens = settings.behavior_max_new_tokens,
                    seed = settings.generation_seed,
                )
                record = Dict(
                    "behavior_updates" => behavior_updates,
                    "global_step" => trainer.step,
                    "validation_loss" => validation_loss,
                    "behavior_summary" => behavior_report["summary"],
                )
                if v8_behavior_record_is_better(record, best_behavior_record)
                    best_behavior_record = deepcopy(record)
                    if settings.save_best_behavior_bundle
                        save_v8_model_bundle(
                            best_behavior_bundle_dir,
                            trainer.model;
                            metadata = Dict(
                                "kind" => "best_behavior",
                                "experiment" => V8_EXPERIMENT_NAME,
                                "behavior_updates" => behavior_updates,
                                "global_step" => trainer.step,
                                "validation_loss" => validation_loss,
                                "behavior_summary" => behavior_report["summary"],
                            ),
                        )
                        saved_best_behavior_bundle_dir = best_behavior_bundle_dir
                        record["best_behavior_bundle_dir"] = best_behavior_bundle_dir
                        println(@sprintf(
                            "stage behavior  best update %d/%d  behavior_pass_rate=%.3f  saved_bundle=%s",
                            behavior_updates,
                            settings.behavior_max_updates,
                            Float64(behavior_report["summary"]["pass_rate"]),
                            best_behavior_bundle_dir,
                        ))
                    end
                end
                push!(validation_history, record)
                println(@sprintf(
                    "stage behavior  validation update %d/%d  pretrain=%.4f  sft=%.4f  behavior_pass_rate=%.3f",
                    behavior_updates,
                    settings.behavior_max_updates,
                    validation_loss["pretrain"],
                    validation_loss["sft"],
                    Float64(behavior_report["summary"]["pass_rate"]),
                ))
                maybe_stop_reason = v8_behavior_stop_reason(validation_loss, behavior_report["summary"], validation_baseline, behavior_updates, settings)
                if maybe_stop_reason !== nothing
                    stop_reason = maybe_stop_reason
                    break
                end
            end
        end
    end

    if accumulated_count > 0 && behavior_updates < settings.behavior_max_updates
        update_loss = accumulated_loss / accumulated_count
        apply_accumulated_gradient_update!(trainer, accumulated_gradient, accumulated_count)
        behavior_updates += 1
        push!(update_losses, update_loss)
    end

    isempty(epoch_losses) && throw(ArgumentError("behavior stage did not yield any trainable batches"))
    final_validation = Dict(stage => masked_mean_loss(trainer.model, batches) for (stage, batches) in validation_sets)
    final_behavior_report = v8_score_model_behavior(
        trainer.model,
        tokenizer;
        cases = behavior_cases,
        max_new_tokens = settings.behavior_max_new_tokens,
        seed = settings.generation_seed,
    )
    final_record = Dict(
        "behavior_updates" => behavior_updates,
        "global_step" => trainer.step,
        "validation_loss" => final_validation,
        "behavior_summary" => final_behavior_report["summary"],
        "final" => true,
    )
    if v8_behavior_record_is_better(final_record, best_behavior_record)
        best_behavior_record = deepcopy(final_record)
        if settings.save_best_behavior_bundle
            save_v8_model_bundle(
                best_behavior_bundle_dir,
                trainer.model;
                metadata = Dict(
                    "kind" => "best_behavior",
                    "experiment" => V8_EXPERIMENT_NAME,
                    "behavior_updates" => behavior_updates,
                    "global_step" => trainer.step,
                    "validation_loss" => final_validation,
                    "behavior_summary" => final_behavior_report["summary"],
                ),
            )
            saved_best_behavior_bundle_dir = best_behavior_bundle_dir
            final_record["best_behavior_bundle_dir"] = best_behavior_bundle_dir
        end
    end
    push!(validation_history, final_record)
    trainer.epoch += 1
    train_loss = sum(epoch_losses) / length(epoch_losses)
    summary = Dict(
        "stage" => "behavior_direct_sft_with_pretrain_replay",
        "global_epoch" => trainer.epoch,
        "step" => trainer.step,
        "behavior_updates" => behavior_updates,
        "microbatch_count" => sum(values(source_microbatches)),
        "source_microbatches" => Dict(String(key) => value for (key, value) in pairs(source_microbatches)),
        "source_loss_targets" => Dict(String(key) => value for (key, value) in pairs(source_loss_targets)),
        "train_loss" => train_loss,
        "train_perplexity" => exp(train_loss),
        "validation_loss" => final_validation,
        "behavior_summary" => final_behavior_report["summary"],
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
        "stage behavior  completed updates=%d  train_loss=%.4f  pretrain_val=%.4f  sft_val=%.4f  behavior_pass_rate=%.3f  stop_reason=%s",
        behavior_updates,
        train_loss,
        final_validation["pretrain"],
        final_validation["sft"],
        Float64(final_behavior_report["summary"]["pass_rate"]),
        stop_reason,
    ))
    return (
        summary = summary,
        validation_history = validation_history,
        stop_reason = stop_reason,
        best_behavior_record = best_behavior_record,
        best_behavior_bundle_dir = saved_best_behavior_bundle_dir,
    )
end

function save_v8_model_bundle(directory_path::AbstractString, model; metadata = Dict{String,Any}())
    bundle = Bundle(
        manifest = BundleManifest(metadata = metadata),
        model_config = KeemenaLM.Core.model_config(model),
        weights = KeemenaLM.Core.extract_weights(model),
    )
    save_bundle(directory_path, bundle)
    return directory_path
end

function v8_behavior_record_is_better(candidate::AbstractDict, current)::Bool
    current === nothing && return true
    candidate_summary = candidate["behavior_summary"]
    current_summary = current["behavior_summary"]
    candidate_pass_rate = Float64(candidate_summary["pass_rate"])
    current_pass_rate = Float64(current_summary["pass_rate"])
    candidate_pass_rate != current_pass_rate && return candidate_pass_rate > current_pass_rate

    candidate_sft_loss = Float64(candidate["validation_loss"]["sft"])
    current_sft_loss = Float64(current["validation_loss"]["sft"])
    candidate_sft_loss != current_sft_loss && return candidate_sft_loss < current_sft_loss

    return Int(candidate["behavior_updates"]) > Int(current["behavior_updates"])
end

function v8_behavior_stop_reason(validation_loss, behavior_summary, validation_baseline, behavior_updates::Int, settings::V8ScratchSettings)
    behavior_updates < settings.behavior_min_updates && return nothing
    if settings.stop_on_pretrain_regression
        pretrain_limit = validation_baseline["pretrain"] * settings.pretrain_max_relative_degradation
        validation_loss["pretrain"] > pretrain_limit && return "pretrain_validation_regression"
    end

    behavior_pass_rate = Float64(behavior_summary["pass_rate"])
    behavior_passed = behavior_pass_rate >= settings.behavior_min_pass_rate
    if behavior_passed && validation_loss["sft"] <= settings.direct_sft_early_stop_loss
        return "behavior_gate_passed_without_pretrain_regression"
    end
    return nothing
end

function validate_v8_initial_bundle_config(loaded_config, expected_config::GPT2Config, bundle_dir::AbstractString)
    loaded_config isa GPT2Config || throw(ArgumentError("initial bundle must contain a GPT2Config: $(bundle_dir)"))
    loaded_config.vocab_size == expected_config.vocab_size || throw(ArgumentError("initial bundle vocab_size $(loaded_config.vocab_size) does not match tokenizer vocab $(expected_config.vocab_size): $(bundle_dir)"))
    loaded_config.context_length == expected_config.context_length || throw(ArgumentError("initial bundle context_length $(loaded_config.context_length) does not match --context-length $(expected_config.context_length): $(bundle_dir)"))
    loaded_config.num_layers == expected_config.num_layers || throw(ArgumentError("initial bundle num_layers $(loaded_config.num_layers) does not match --num-layers $(expected_config.num_layers): $(bundle_dir)"))
    loaded_config.num_heads == expected_config.num_heads || throw(ArgumentError("initial bundle num_heads $(loaded_config.num_heads) does not match --num-heads $(expected_config.num_heads): $(bundle_dir)"))
    loaded_config.embedding_size == expected_config.embedding_size || throw(ArgumentError("initial bundle embedding_size $(loaded_config.embedding_size) does not match --embedding-size $(expected_config.embedding_size): $(bundle_dir)"))
    loaded_config.ffn_hidden_size == expected_config.ffn_hidden_size || throw(ArgumentError("initial bundle ffn_hidden_size $(loaded_config.ffn_hidden_size) does not match --ffn-hidden-size $(expected_config.ffn_hidden_size): $(bundle_dir)"))
    return loaded_config
end

function v8_behavior_cycle(settings::V8ScratchSettings)::Vector{Symbol}
    settings.direct_sft_parts >= 0 || throw(ArgumentError("direct_sft_parts must be >= 0"))
    settings.pretrain_replay_parts >= 0 || throw(ArgumentError("pretrain_replay_parts must be >= 0"))
    cycle = Symbol[]
    append!(cycle, fill(:sft, settings.direct_sft_parts))
    append!(cycle, fill(:pretrain, settings.pretrain_replay_parts))
    isempty(cycle) && throw(ArgumentError("at least one behavior source part must be > 0"))
    return cycle
end

function v8_behavior_cases(settings::V8ScratchSettings)
    cases = KeemenaLM.Core.chatbot_behavior_cases()
    settings.behavior_prompt_limit <= 0 && return cases
    return cases[1:min(settings.behavior_prompt_limit, length(cases))]
end

function v8_build_eval_sets(paths, tokenizer, pad_token_id::Int, settings::V8ScratchSettings)
    pretrain_validation_texts = read_prepared_paragraphs(paths.pretrain_validation; limit = settings.validation_text_limit)
    pretrain_testing_texts = read_prepared_paragraphs(paths.pretrain_testing; limit = settings.test_text_limit)
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
        "sft" => limit_batches(sft_validation_batches, settings.validation_batch_limit),
    )
    test_sets = Dict(
        "pretrain" => limit_batches(pretrain_test_batches, settings.test_batch_limit),
        "sft" => limit_batches(sft_test_batches, settings.test_batch_limit),
    )
    batch_stats = Dict(
        "pretrain_validation" => Dict(pairs(pretrain_validation_stats)),
        "pretrain_test" => Dict(pairs(pretrain_test_stats)),
        "sft_validation" => Dict(pairs(sft_validation_stats)),
        "sft_test" => Dict(pairs(sft_test_stats)),
    )
    return validation_sets, test_sets, batch_stats
end

function v8_tokenizer_training_texts(paths, settings::V8ScratchSettings)::Vector{String}
    limit = settings.tokenizer_training_text_limit
    pretrain_limit = limit <= 0 ? 0 : max(1, round(Int, limit * 0.75))
    sft_limit = limit <= 0 ? 0 : max(1, limit - pretrain_limit)

    pretrain_texts = read_prepared_paragraphs(paths.pretrain_training; limit = optional_text_limit(pretrain_limit))
    sft_examples = read_clean_sft_examples(paths.sft_training; limit = sft_limit)
    texts = String[]
    append!(texts, pretrain_texts)
    append!(texts, [example.chat_text for example in sft_examples])
    isempty(texts) && throw(ArgumentError("tokenizer training texts are empty"))
    return limit <= 0 ? texts : texts[1:min(limit, length(texts))]
end

function v8_streaming_stage_settings(settings::V8ScratchSettings)::StreamingStageSettings
    return StreamingStageSettings(
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
        chat_lm_text_limit = 0,
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

function v8_dataset_paths(dataset_dir::AbstractString)
    return (
        metadata = joinpath(dataset_dir, "metadata.json"),
        pretrain_training = joinpath(dataset_dir, "pretrain", "training.txt"),
        pretrain_validation = joinpath(dataset_dir, "pretrain", "validation.txt"),
        pretrain_testing = joinpath(dataset_dir, "pretrain", "testing.txt"),
        sft_training = joinpath(dataset_dir, "sft", "training.jsonl"),
        sft_validation = joinpath(dataset_dir, "sft", "validation.jsonl"),
        sft_testing = joinpath(dataset_dir, "sft", "testing.jsonl"),
    )
end

function validate_v8_settings(settings::V8ScratchSettings)
    settings.device in (:auto, :cpu, :gpu) || throw(ArgumentError("device must be one of :auto, :cpu, or :gpu"))
    settings.context_length > 0 || throw(ArgumentError("context_length must be > 0"))
    settings.batch_size > 0 || throw(ArgumentError("batch_size must be > 0"))
    settings.gradient_accumulation_steps > 0 || throw(ArgumentError("gradient_accumulation_steps must be > 0"))
    settings.pretrain_epochs >= 0 || throw(ArgumentError("pretrain_epochs must be >= 0"))
    settings.pretrain_epochs > 0 || settings.pretrain_replay_parts > 0 || throw(ArgumentError("v8 needs either pretrain_epochs > 0 or pretrain replay > 0"))
    settings.behavior_max_updates > 0 || throw(ArgumentError("behavior_max_updates must be > 0"))
    settings.behavior_min_updates >= 0 || throw(ArgumentError("behavior_min_updates must be >= 0"))
    settings.validation_every_updates >= 0 || throw(ArgumentError("validation_every_updates must be >= 0"))
    settings.learning_rate > 0 || throw(ArgumentError("learning_rate must be > 0"))
    settings.num_layers > 0 || throw(ArgumentError("num_layers must be > 0"))
    settings.num_heads > 0 || throw(ArgumentError("num_heads must be > 0"))
    settings.embedding_size % settings.num_heads == 0 || throw(ArgumentError("embedding_size must be divisible by num_heads"))
    settings.ffn_hidden_size > 0 || throw(ArgumentError("ffn_hidden_size must be > 0"))
    settings.direct_sft_early_stop_loss > 0 || throw(ArgumentError("direct_sft_early_stop_loss must be > 0"))
    settings.pretrain_max_relative_degradation >= 1.0 || throw(ArgumentError("pretrain_max_relative_degradation must be >= 1.0"))
    isempty(settings.initial_bundle_dir) || isdir(settings.initial_bundle_dir) || throw(ArgumentError("--initial-bundle-dir does not exist: $(settings.initial_bundle_dir)"))
    0.0 < settings.behavior_min_pass_rate <= 1.0 || throw(ArgumentError("behavior_min_pass_rate must be in (0, 1]"))
    settings.behavior_max_new_tokens >= 0 || throw(ArgumentError("behavior_max_new_tokens must be >= 0"))
    settings.behavior_prompt_limit >= 0 || throw(ArgumentError("behavior_prompt_limit must be >= 0"))
    settings.tokenizer_vocab_size > 0 || throw(ArgumentError("tokenizer_vocab_size must be > 0"))
    settings.tokenizer_training_text_limit >= 0 || throw(ArgumentError("tokenizer_training_text_limit must be >= 0"))
    settings.pretrain_text_limit >= 0 || throw(ArgumentError("pretrain_text_limit must be >= 0"))
    settings.sft_train_example_limit >= 0 || throw(ArgumentError("sft_train_example_limit must be >= 0"))
    settings.validation_text_limit > 0 || throw(ArgumentError("validation_text_limit must be > 0"))
    settings.test_text_limit > 0 || throw(ArgumentError("test_text_limit must be > 0"))
    settings.sft_validation_example_limit > 0 || throw(ArgumentError("sft_validation_example_limit must be > 0"))
    settings.sft_test_example_limit > 0 || throw(ArgumentError("sft_test_example_limit must be > 0"))
    settings.validation_batch_limit >= 0 || throw(ArgumentError("validation_batch_limit must be >= 0"))
    settings.test_batch_limit >= 0 || throw(ArgumentError("test_batch_limit must be >= 0"))
    settings.log_every_updates >= 0 || throw(ArgumentError("log_every_updates must be >= 0"))
    settings.audit_example_count >= 0 || throw(ArgumentError("audit_example_count must be >= 0"))
    v8_behavior_cycle(settings)
    return settings
end

function v8_recipe_dict(dataset_dir::AbstractString, output_dir::AbstractString, settings::V8ScratchSettings)
    config = GPT2Config(
        vocab_size = settings.tokenizer_vocab_size,
        context_length = settings.context_length,
        num_layers = settings.num_layers,
        num_heads = settings.num_heads,
        embedding_size = settings.embedding_size,
        ffn_hidden_size = settings.ffn_hidden_size,
    )
    return Dict(
        "experiment" => V8_EXPERIMENT_NAME,
        "purpose" => V8_PURPOSE,
        "dataset_dir" => abspath(dataset_dir),
        "output_dir" => abspath(output_dir),
        "backend" => "flux",
        "device" => String(settings.device),
        "curriculum" => ["streamed_pretrain_all_tokens", "behavior_direct_sft_with_pretrain_replay"],
        "disabled_curriculum" => ["raw_chat_lm_all_tokens"],
        "optimizer_name" => "Flux.Adam",
        "optimizer_hyperparameters" => Dict("learning_rate" => settings.learning_rate),
        "tokenizer_package" => "KeemenaSubwords.jl",
        "tokenizer_trainer" => String(settings.tokenizer_trainer),
        "tokenizer_vocab_size" => settings.tokenizer_vocab_size,
        "tokenizer_min_frequency" => settings.tokenizer_min_frequency,
        "tokenizer_model_name" => settings.tokenizer_model_name,
        "tokenizer_training_text_limit" => settings.tokenizer_training_text_limit,
        "reuse_tokenizer_bundle_dir" => settings.reuse_tokenizer_bundle_dir,
        "initial_bundle_dir" => settings.initial_bundle_dir,
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
        "behavior_max_updates" => settings.behavior_max_updates,
        "behavior_min_updates" => settings.behavior_min_updates,
        "validation_every_updates" => settings.validation_every_updates,
        "behavior_ratio_parts" => Dict("direct_sft" => settings.direct_sft_parts, "pretrain_replay" => settings.pretrain_replay_parts),
        "direct_sft_early_stop_loss" => settings.direct_sft_early_stop_loss,
        "pretrain_max_relative_degradation" => settings.pretrain_max_relative_degradation,
        "stop_on_pretrain_regression" => settings.stop_on_pretrain_regression,
        "behavior_min_pass_rate" => settings.behavior_min_pass_rate,
        "checkpoint_policy" => "one base-before-behavior checkpoint by default; final bundle only unless --save-final-checkpoint is passed; optional best behavior bundle overwrites one extra bundle directory",
        "save_base_checkpoint" => settings.save_base_checkpoint,
        "save_final_checkpoint" => settings.save_final_checkpoint,
        "save_best_behavior_bundle" => settings.save_best_behavior_bundle,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !any(argument -> argument in ("--help", "-h"), ARGS)
        command_allows_gpu_device(ARGS) && KeemenaLM.FluxBackend.has_functional_cuda_gpu()
    end
    Base.invokelatest(main_v8, ARGS)
end
