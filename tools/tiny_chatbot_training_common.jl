#!/usr/bin/env julia

# Shared implementation for the current tiny-chatbot v8/v9 tools.
# This was factored out of historical runner files so old version-named entrypoints can be removed.


# ---- Shared base tokenizer, batching, training, checkpoint, and bundle helpers ----


using Flux
using JSON3
using KeemenaLM
using KeemenaSubwords
using Printf
using Random

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
    loss_mode::Symbol = :assistant_only
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
    tokenizer_model_name::String = "tiny_chatbot_gpt2_bytebpe"
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


function parse_device(value::AbstractString)::Symbol
    normalized_value = lowercase(strip(value))
    if normalized_value == "auto"
        return :auto
    elseif normalized_value == "cpu"
        return :cpu
    elseif normalized_value == "gpu"
        return :gpu
    elseif normalized_value == "metal"
        return :metal
    else
        throw(ArgumentError("--device must be one of auto, cpu, gpu, or metal"))
    end
end

function parse_loss_mode(value::AbstractString)::Symbol
    normalized_value = lowercase(strip(value))
    if normalized_value in ("assistant_only", "assistant-only", "assistant")
        return :assistant_only
    elseif normalized_value in ("all_tokens", "all-tokens", "all")
        return :all_tokens
    else
        throw(ArgumentError("--loss-mode must be one of assistant_only or all_tokens"))
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
    loss_mode::Symbol,
    chat_special_tokens::Dict{Symbol,String},
)
    token_ids, token_loss_mask = build_training_token_stream(
        texts,
        tokenizer;
        document_separator = document_separator,
        loss_mode = loss_mode,
        chat_special_tokens = chat_special_tokens,
    )
    length(token_ids) > context_length ||
        throw(ArgumentError("dataset split is too small for context_length=$(context_length)"))

    candidate_example_count = fld(length(token_ids) - 1, context_length)
    candidate_example_count > 0 || throw(ArgumentError("dataset split did not yield any LM examples"))

    batches = Tuple{Matrix{Int32}, Matrix{Int32}, Matrix{Float32}}[]
    input_batch = Matrix{Int32}(undef, context_length, batch_size)
    target_batch = Matrix{Int32}(undef, context_length, batch_size)
    loss_mask_batch = Matrix{Float32}(undef, context_length, batch_size)
    batch_column = 0
    example_count = 0
    loss_target_count = 0

    for example_index in 1:candidate_example_count
        offset = (example_index - 1) * context_length
        target_loss_mask = Float32.(token_loss_mask[(offset + 2):(offset + context_length + 1)])
        target_loss_count = sum(target_loss_mask)
        target_loss_count > 0 || continue

        batch_column += 1
        input_batch[:, batch_column] = Int32.(token_ids[(offset + 1):(offset + context_length)])
        target_batch[:, batch_column] = Int32.(token_ids[(offset + 2):(offset + context_length + 1)])
        loss_mask_batch[:, batch_column] = target_loss_mask
        example_count += 1
        loss_target_count += Int(target_loss_count)

        if batch_column == batch_size
            push!(batches, (copy(input_batch), copy(target_batch), copy(loss_mask_batch)))
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

    example_count > 0 || throw(ArgumentError("dataset split did not yield any LM examples with loss targets"))

    stats = (
        token_stream_length = length(token_ids),
        candidate_example_count = candidate_example_count,
        example_count = example_count,
        loss_target_count = loss_target_count,
    )
    return batches, stats
end

function build_training_token_stream(
    texts::Vector{String},
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer;
    document_separator::AbstractString,
    loss_mode::Symbol,
    chat_special_tokens::Dict{Symbol,String},
)
    token_ids = Int[]
    token_loss_mask = Float32[]
    separator_token_ids = KeemenaSubwords.encode(tokenizer, document_separator; add_special_tokens = false)

    for (text_index, text) in enumerate(texts)
        text_token_ids = KeemenaSubwords.encode(tokenizer, text; add_special_tokens = false)
        text_loss_mask = token_loss_mask_for_text(
            text_token_ids,
            tokenizer;
            loss_mode = loss_mode,
            chat_special_tokens = chat_special_tokens,
        )
        append!(token_ids, text_token_ids)
        append!(token_loss_mask, text_loss_mask)

        if text_index < length(texts) && !isempty(separator_token_ids)
            append!(token_ids, separator_token_ids)
            append!(token_loss_mask, zeros(Float32, length(separator_token_ids)))
        end
    end

    return token_ids, token_loss_mask
end

function token_loss_mask_for_text(
    token_ids::Vector{Int},
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer;
    loss_mode::Symbol,
    chat_special_tokens::Dict{Symbol,String},
)::Vector{Float32}
    if loss_mode === :all_tokens
        return ones(Float32, length(token_ids))
    elseif loss_mode !== :assistant_only
        throw(ArgumentError("unsupported loss_mode $(loss_mode); expected :assistant_only or :all_tokens"))
    end

    assistant_token_id = KeemenaSubwords.token_to_id(tokenizer, chat_special_tokens[:assistant])
    user_token_id = KeemenaSubwords.token_to_id(tokenizer, chat_special_tokens[:user])
    chat_end_token_id = KeemenaSubwords.token_to_id(tokenizer, chat_special_tokens[:chat_end])

    loss_mask = zeros(Float32, length(token_ids))
    in_assistant_turn = false
    for (index, token_id) in enumerate(token_ids)
        if token_id == assistant_token_id
            in_assistant_turn = true
            loss_mask[index] = 0.0f0
        elseif token_id == user_token_id
            in_assistant_turn = false
            loss_mask[index] = 0.0f0
        elseif in_assistant_turn
            loss_mask[index] = 1.0f0
            token_id == chat_end_token_id && (in_assistant_turn = false)
        end
    end

    return loss_mask
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
    for (input_batch, target_batch, loss_mask_batch) in batches
        logits, _ = KeemenaLM.Core.lm_forward(model, input_batch; cache = nothing, is_training = false)
        loss_target_batch = KeemenaLM.FluxBackend.move_like(target_batch, model.token_embedding)
        loss_weights = KeemenaLM.FluxBackend.move_like(loss_mask_batch, model.token_embedding)
        total_loss += Float64(KeemenaLM.Core.causal_lm_cross_entropy(logits, loss_target_batch, loss_weights))
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


# ---- Shared assistant-only SFT example and mask helpers ----



using Flux
using JSON3
using KeemenaLM
using KeemenaSubwords
using Printf
using Random


struct CleanSFTExample
    id::String
    prompt_text::String
    target_text::String
    assistant_text::String
    chat_text::String
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




Base.@kwdef mutable struct StreamingStageSettings
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
    tokenizer_model_name::String = "tiny_chatbot_streaming_gpt2_bytebpe"
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


function train_streaming_stage!(
    trainer,
    stage_name::AbstractString,
    batch_source::Function,
    validation_batches,
    epochs::Int,
    settings::StreamingStageSettings,
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
            loss_value, gradient = training_loss_and_gradient(trainer, input_batch, target_batch, loss_mask_batch)
            push!(epoch_losses, loss_value)
            microbatch_count += 1
            loss_target_count += batch_loss_targets
            accumulated_loss += loss_value
            accumulated_count += 1
            accumulated_gradient = accumulated_gradient === nothing ? gradient : gradient_tree_add(accumulated_gradient, gradient)

            if accumulated_count >= settings.gradient_accumulation_steps
                update_loss = accumulated_loss / accumulated_count
                apply_accumulated_gradient_update!(trainer, accumulated_gradient, accumulated_count)
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
            apply_accumulated_gradient_update!(trainer, accumulated_gradient, accumulated_count)
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

function training_loss_and_gradient(
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

function apply_accumulated_gradient_update!(trainer, accumulated_gradient, accumulation_count::Int)
    accumulation_count > 0 || throw(ArgumentError("accumulation_count must be > 0"))
    averaged_gradient = gradient_tree_scale(accumulated_gradient, Float32(1 / accumulation_count))
    Flux.update!(trainer.optimizer_state, trainer.model, averaged_gradient)
    trainer.step += 1
    return trainer.step
end

gradient_tree_add(::Nothing, ::Nothing) = nothing
gradient_tree_add(a::Nothing, b) = b
gradient_tree_add(a, b::Nothing) = a
gradient_tree_add(a::Number, b::Number) = a + b
gradient_tree_add(a::AbstractArray{<:Number}, b::AbstractArray{<:Number}) = a .+ b

function gradient_tree_add(a::NamedTuple, b::NamedTuple)
    keys(a) == keys(b) || throw(ArgumentError("cannot add gradients with different NamedTuple keys"))
    return NamedTuple{keys(a)}(ntuple(index -> gradient_tree_add(getfield(a, keys(a)[index]), getfield(b, keys(b)[index])), length(keys(a))))
end

function gradient_tree_add(a::AbstractVector{<:NamedTuple}, b::AbstractVector{<:NamedTuple})
    length(a) == length(b) || throw(ArgumentError("cannot add gradient vectors with different lengths"))
    return [gradient_tree_add(a[index], b[index]) for index in eachindex(a)]
end

gradient_tree_scale(a::Nothing, scale::Real) = nothing
gradient_tree_scale(a::Number, scale::Real) = a * scale
gradient_tree_scale(a::AbstractArray{<:Number}, scale::Real) = a .* scale

function gradient_tree_scale(a::NamedTuple, scale::Real)
    return NamedTuple{keys(a)}(ntuple(index -> gradient_tree_scale(getfield(a, keys(a)[index]), scale), length(keys(a))))
end

function gradient_tree_scale(a::AbstractVector{<:NamedTuple}, scale::Real)
    return [gradient_tree_scale(value, scale) for value in a]
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



mutable struct RestartingBatchSource
    name::Symbol
    factory::Function
    channel::Any
    iterator_state::Any
end

function RestartingBatchSource(name::Symbol, factory::Function)
    return RestartingBatchSource(name, factory, factory(), nothing)
end

function next_restarting_batch!(source::RestartingBatchSource)
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
