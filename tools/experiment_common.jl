using JSON3
using KeemenaLM

Base.@kwdef struct ExperimentSettings
    dataset_seed::Int = 20260417
    model_seed::Int = 20260418
    generation_seed::Int = 20260419
    complexity::Int = 5
    num_sentences::Int = 8_000
    enable_polysemy::Bool = false
    context_length::Int = 48
    batch_size::Int = 16
    epochs::Int = 2
    learning_rate::Float32 = 0.01f0
    num_layers::Int = 2
    num_heads::Int = 2
    embedding_size::Int = 64
    ffn_hidden_size::Int = 128
    prompt_prefix_characters::Int = 12
    sample_generation_tokens::Int = 32
    train_token_stream_limit::Union{Nothing, Int} = nothing
end

struct ExperimentCharTokenizer
    alphabet::Vector{Char}
    token_to_id::Dict{Char, Int}
    id_to_token::Dict{Int, Char}
end

function ExperimentCharTokenizer(alphabet::AbstractVector{Char})
    token_to_id = Dict(character => index for (index, character) in enumerate(alphabet))
    id_to_token = Dict(index => character for (index, character) in enumerate(alphabet))
    return ExperimentCharTokenizer(collect(alphabet), token_to_id, id_to_token)
end

KeemenaLM.Core.tokenizer_encode(tokenizer::ExperimentCharTokenizer, text::AbstractString) =
    [get(tokenizer.token_to_id, character, 1) for character in text]

KeemenaLM.Core.tokenizer_decode(tokenizer::ExperimentCharTokenizer, token_ids::AbstractVector{<:Integer}) =
    String([get(tokenizer.id_to_token, Int(token_id), '?') for token_id in token_ids])

function merge_settings(
    settings::ExperimentSettings;
    dataset_seed::Int = settings.dataset_seed,
    model_seed::Int = settings.model_seed,
    generation_seed::Int = settings.generation_seed,
    complexity::Int = settings.complexity,
    num_sentences::Int = settings.num_sentences,
    enable_polysemy::Bool = settings.enable_polysemy,
    context_length::Int = settings.context_length,
    batch_size::Int = settings.batch_size,
    epochs::Int = settings.epochs,
    learning_rate::Float32 = settings.learning_rate,
    num_layers::Int = settings.num_layers,
    num_heads::Int = settings.num_heads,
    embedding_size::Int = settings.embedding_size,
    ffn_hidden_size::Int = settings.ffn_hidden_size,
    prompt_prefix_characters::Int = settings.prompt_prefix_characters,
    sample_generation_tokens::Int = settings.sample_generation_tokens,
    train_token_stream_limit::Union{Nothing, Int} = settings.train_token_stream_limit,
)
    return ExperimentSettings(
        dataset_seed = dataset_seed,
        model_seed = model_seed,
        generation_seed = generation_seed,
        complexity = complexity,
        num_sentences = num_sentences,
        enable_polysemy = enable_polysemy,
        context_length = context_length,
        batch_size = batch_size,
        epochs = epochs,
        learning_rate = learning_rate,
        num_layers = num_layers,
        num_heads = num_heads,
        embedding_size = embedding_size,
        ffn_hidden_size = ffn_hidden_size,
        prompt_prefix_characters = prompt_prefix_characters,
        sample_generation_tokens = sample_generation_tokens,
        train_token_stream_limit = train_token_stream_limit,
    )
end

function build_tokenizer(split_texts; extra_texts::AbstractVector{<:AbstractString} = String[])
    alphabet = Set{Char}(['\n'])
    for texts in values(split_texts)
        for text in texts
            for character in text
                push!(alphabet, character)
            end
        end
    end
    for text in extra_texts
        for character in text
            push!(alphabet, character)
        end
    end
    return ExperimentCharTokenizer(sort!(collect(alphabet)))
end

function save_tokenizer(path::AbstractString, tokenizer::ExperimentCharTokenizer)
    open(path, "w") do io
        JSON3.write(io, Dict("alphabet" => [string(character) for character in tokenizer.alphabet]))
    end
    return path
end

function load_tokenizer(path::AbstractString)::ExperimentCharTokenizer
    isfile(path) || throw(ArgumentError("tokenizer file does not exist: $(path)"))
    tokenizer_data = JSON3.read(read(path, String))
    hasproperty(tokenizer_data, :alphabet) || throw(ArgumentError("tokenizer file is missing alphabet"))
    alphabet = [only(String(character_string)) for character_string in tokenizer_data.alphabet]
    return ExperimentCharTokenizer(alphabet)
end

function build_lm_batches(
    texts::Vector{String},
    tokenizer::ExperimentCharTokenizer;
    context_length::Int,
    batch_size::Int,
    max_token_stream_length::Union{Nothing, Int} = nothing,
    document_separator::AbstractString = "\n",
)
    corpus = join(texts, document_separator)
    token_ids = KeemenaLM.Core.tokenizer_encode(tokenizer, corpus)
    if max_token_stream_length !== nothing
        max_token_stream_length > 1 ||
            throw(ArgumentError("max_token_stream_length must be > 1 when provided"))
        length(token_ids) >= max_token_stream_length ||
            throw(ArgumentError("token stream length $(length(token_ids)) is smaller than requested limit $(max_token_stream_length)"))
        token_ids = token_ids[1:max_token_stream_length]
    end
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

function sample_prompts(texts::Vector{String}; count::Int, prefix_characters::Int)
    prompts = String[]
    for text in texts
        isempty(strip(text)) && continue
        push!(prompts, first(text, min(prefix_characters, length(text))))
        length(prompts) == count && break
    end
    isempty(prompts) && throw(ArgumentError("unable to derive non-empty prompts from the test split"))
    return prompts
end

function generate_samples(model, tokenizer, prompts::Vector{String}; settings::ExperimentSettings)
    generation_config = GenerationConfig(
        max_new_tokens = settings.sample_generation_tokens,
        temperature = 0.0,
        seed = settings.generation_seed,
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
