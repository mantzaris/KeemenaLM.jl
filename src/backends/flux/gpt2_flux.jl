struct FluxLinear{WeightType<:AbstractMatrix{<:AbstractFloat}, BiasType<:AbstractVector{<:AbstractFloat}}
    weight::WeightType
    bias::BiasType
end

Flux.@layer FluxLinear

struct FluxLayerNorm{
    WeightType<:AbstractVector{<:AbstractFloat},
    BiasType<:AbstractVector{<:AbstractFloat},
    EpsilonType<:AbstractFloat,
}
    weight::WeightType
    bias::BiasType
    epsilon::EpsilonType
end

Flux.@layer FluxLayerNorm

struct FluxCausalSelfAttention{
    QProjType<:FluxLinear,
    KProjType<:FluxLinear,
    VProjType<:FluxLinear,
    OutProjType<:FluxLinear,
}
    q_proj::QProjType
    k_proj::KProjType
    v_proj::VProjType
    out_proj::OutProjType
    num_heads::Int
end

Flux.@layer FluxCausalSelfAttention

struct FluxTransformerBlock{
    AttentionNormType<:FluxLayerNorm,
    AttentionType<:FluxCausalSelfAttention,
    FfnNormType<:FluxLayerNorm,
    FfnInType<:FluxLinear,
    FfnOutType<:FluxLinear,
}
    attention_norm::AttentionNormType
    attention::AttentionType
    ffn_norm::FfnNormType
    ffn_in::FfnInType
    ffn_out::FfnOutType
end

Flux.@layer FluxTransformerBlock

"""
Flux GPT-2 style decoder-only LM for the Stage 4 training/inference path.
"""
struct FluxGPT2Model{
    TokenEmbeddingType<:AbstractMatrix{<:AbstractFloat},
    PositionEmbeddingType<:AbstractMatrix{<:AbstractFloat},
    BlockType<:FluxTransformerBlock,
    FinalLayerNormType<:FluxLayerNorm,
} <: AbstractCausalLM
    config::GPT2Config
    token_embedding::TokenEmbeddingType
    position_embedding::PositionEmbeddingType
    blocks::Vector{BlockType}
    final_layer_norm::FinalLayerNormType
end

Flux.@layer FluxGPT2Model

function build_gpt2_model(
    config::GPT2Config;
    seed::Int = 0,
    init_scale::Float32 = 0.02f0,
)
    validate(config)

    parameter_offset = Ref(seed)
    token_embedding = initialized_parameter(config.embedding_size, config.vocab_size, take_parameter_offset!(parameter_offset, config.embedding_size * config.vocab_size); scale = init_scale)
    position_embedding = initialized_parameter(config.embedding_size, config.context_length, take_parameter_offset!(parameter_offset, config.embedding_size * config.context_length); scale = init_scale)
    blocks = [
        FluxTransformerBlock(
            FluxLayerNorm(ones(Float32, config.embedding_size), zeros(Float32, config.embedding_size), 1.0f-5),
            FluxCausalSelfAttention(
                FluxLinear(
                    initialized_parameter(config.embedding_size, config.embedding_size, take_parameter_offset!(parameter_offset, config.embedding_size * config.embedding_size); scale = init_scale),
                    zeros(Float32, config.embedding_size),
                ),
                FluxLinear(
                    initialized_parameter(config.embedding_size, config.embedding_size, take_parameter_offset!(parameter_offset, config.embedding_size * config.embedding_size); scale = init_scale),
                    zeros(Float32, config.embedding_size),
                ),
                FluxLinear(
                    initialized_parameter(config.embedding_size, config.embedding_size, take_parameter_offset!(parameter_offset, config.embedding_size * config.embedding_size); scale = init_scale),
                    zeros(Float32, config.embedding_size),
                ),
                FluxLinear(
                    initialized_parameter(config.embedding_size, config.embedding_size, take_parameter_offset!(parameter_offset, config.embedding_size * config.embedding_size); scale = init_scale),
                    zeros(Float32, config.embedding_size),
                ),
                config.num_heads,
            ),
            FluxLayerNorm(ones(Float32, config.embedding_size), zeros(Float32, config.embedding_size), 1.0f-5),
            FluxLinear(
                initialized_parameter(config.ffn_hidden_size, config.embedding_size, take_parameter_offset!(parameter_offset, config.ffn_hidden_size * config.embedding_size); scale = init_scale),
                zeros(Float32, config.ffn_hidden_size),
            ),
            FluxLinear(
                initialized_parameter(config.embedding_size, config.ffn_hidden_size, take_parameter_offset!(parameter_offset, config.embedding_size * config.ffn_hidden_size); scale = init_scale),
                zeros(Float32, config.embedding_size),
            ),
        ) for _ in 1:config.num_layers
    ]
    final_layer_norm = FluxLayerNorm(ones(Float32, config.embedding_size), zeros(Float32, config.embedding_size), 1.0f-5)

    return FluxGPT2Model(config, token_embedding, position_embedding, blocks, final_layer_norm)
end

model_config(model::FluxGPT2Model) = model.config

function move_model_to_device(model::FluxGPT2Model; device::Symbol = :cpu)
    selected_device = resolve_flux_device(device)
    return selected_device === :gpu ? cuda_cu(model) : Flux.cpu(model)
end

function move_batch_to_device(batch; device::Symbol = :cpu)
    selected_device = resolve_flux_device(device)

    # Supported Stage 4 policy: token-id batches remain on CPU even when model/compute move to CUDA.
    if batch isa AbstractArray && eltype(batch) <: Integer
        return Flux.cpu(batch)
    end

    return selected_device === :gpu ? cuda_cu(batch) : Flux.cpu(batch)
end

const CUDA_PACKAGE_ID = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
const _CUDA_MODULE = Ref{Union{Nothing, Module}}(nothing)
const _CUDA_FUNCTIONAL = Ref{Union{Nothing, Bool}}(nothing)

function resolve_flux_device(device::Symbol)::Symbol
    if device === :cpu
        return :cpu
    elseif device === :gpu
        has_functional_cuda_gpu() || throw(ArgumentError("device=:gpu requested but no functional NVIDIA/CUDA backend is available"))
        return :gpu
    elseif device === :auto
        return has_functional_cuda_gpu() ? :gpu : :cpu
    else
        throw(ArgumentError("unsupported Flux device $(device); expected :cpu, :gpu, or :auto"))
    end
end

function has_functional_cuda_gpu()::Bool
    _CUDA_FUNCTIONAL[] !== nothing && return _CUDA_FUNCTIONAL[]::Bool

    cuda = cuda_module()
    if cuda === nothing
        _CUDA_FUNCTIONAL[] = false
        return false
    end

    functional = Base.invokelatest(getproperty(cuda, :functional))
    _CUDA_FUNCTIONAL[] = functional
    return functional
end

function cuda_module()
    _CUDA_MODULE[] !== nothing && return _CUDA_MODULE[]
    Base.find_package("CUDA") === nothing && return nothing

    Base.require(CUDA_PACKAGE_ID)
    module_object = get(Base.loaded_modules, CUDA_PACKAGE_ID, nothing)
    _CUDA_MODULE[] = module_object
    return module_object
end

function cuda_cu(data)
    cuda = cuda_module()
    cuda === nothing && throw(ArgumentError("CUDA.jl is not available in this environment"))
    return Base.invokelatest(getproperty(cuda, :cu), data)
end

function is_cuda_array(reference)
    reference isa Array && return false
    cuda = cuda_module()
    cuda === nothing && return false
    return reference isa getproperty(cuda, :CuArray)
end

function initialized_parameter(rows::Int, columns::Int, offset::Int; scale::Float32)::Matrix{Float32}
    total_values = rows * columns
    values = Vector{Float32}(undef, total_values)
    for index in 1:total_values
        values[index] = scale * sin(Float32(offset + index) * 0.37f0)
    end
    return reshape(values, rows, columns)
end

function take_parameter_offset!(parameter_offset::Base.RefValue{Int}, parameter_count::Int)::Int
    current_offset = parameter_offset[]
    parameter_offset[] += parameter_count
    return current_offset
end

function (layer::FluxLinear)(x::AbstractArray{<:Real, 3})
    input_size, sequence_length, batch_size = size(x)
    size(layer.weight, 2) == input_size ||
        throw(DimensionMismatch("FluxLinear expected input size $(size(layer.weight, 2)), got $(input_size)"))

    flattened_input = reshape(x, input_size, :)
    projected = layer.weight * flattened_input .+ layer.bias
    return reshape(projected, size(layer.weight, 1), sequence_length, batch_size)
end

function (layer_norm::FluxLayerNorm)(x::AbstractArray{<:Real, 3})
    feature_count = size(x, 1)
    mean_x = sum(x; dims = 1) ./ feature_count
    variance_x = sum((x .- mean_x) .^ 2; dims = 1) ./ feature_count
    normalized_x = (x .- mean_x) ./ sqrt.(variance_x .+ layer_norm.epsilon)
    return normalized_x .* reshape(layer_norm.weight, :, 1, 1) .+ reshape(layer_norm.bias, :, 1, 1)
end

function (attention::FluxCausalSelfAttention)(x::AbstractArray{<:Real, 3})
    embedding_size, sequence_length, batch_size = size(x)
    head_size = div(embedding_size, attention.num_heads)

    queries = reshape(attention.q_proj(x), head_size, attention.num_heads, sequence_length, batch_size)
    keys = reshape(attention.k_proj(x), head_size, attention.num_heads, sequence_length, batch_size)
    values = reshape(attention.v_proj(x), head_size, attention.num_heads, sequence_length, batch_size)

    mask = causal_mask_like(sequence_length, x)
    scale = inv(sqrt(Float32(head_size)))
    batch_outputs = [
        cat(
            [
                attention_head_output(
                    @view(queries[:, head_index, :, batch_index]),
                    @view(keys[:, head_index, :, batch_index]),
                    @view(values[:, head_index, :, batch_index]),
                    mask,
                    scale,
                ) for head_index in 1:attention.num_heads
            ]...;
            dims = 1,
        ) for batch_index in 1:batch_size
    ]
    merged_output = cat(batch_outputs...; dims = 3)
    return attention.out_proj(merged_output)
end

function attention_head_output(
    queries::AbstractMatrix{<:AbstractFloat},
    keys::AbstractMatrix{<:AbstractFloat},
    values::AbstractMatrix{<:AbstractFloat},
    mask::AbstractMatrix{Bool},
    scale::Float32,
)
    scores = permutedims(queries, (2, 1)) * keys .* scale
    masked_scores = ifelse.(mask, scores, eltype(scores)(-1.0f9))
    attention_weights = Flux.softmax(masked_scores; dims = 2)
    return values * permutedims(attention_weights, (2, 1))
end

move_like(data, reference) = is_cuda_array(reference) ? cuda_cu(data) : Flux.cpu(data)

function causal_mask_like(sequence_length::Int, reference)
    return Flux.Zygote.ignore() do
        move_like(causal_mask(sequence_length), reference)
    end
end

function (block::FluxTransformerBlock)(x::AbstractArray{<:Real, 3})
    attention_output = block.attention(block.attention_norm(x))
    residual_x = x .+ attention_output
    ffn_hidden = Flux.gelu.(block.ffn_in(block.ffn_norm(residual_x)))
    ffn_output = block.ffn_out(ffn_hidden)
    return residual_x .+ ffn_output
end

function lm_forward(
    model::FluxGPT2Model,
    input_token_ids::AbstractMatrix{<:Integer};
    cache = nothing,
    is_training::Bool = false,
)
    cache === nothing || throw(ArgumentError("Stage 1 Flux lm_forward does not support cache yet"))

    # Supported Stage 4 policy: token-id batches stay on CPU even when the model is moved to CUDA.
    input_token_ids = Flux.cpu(input_token_ids)

    sequence_length, batch_size = size(input_token_ids)
    config = model.config

    sequence_length > 0 || throw(ArgumentError("input_token_ids must have at least one time step"))
    sequence_length <= config.context_length ||
        throw(ArgumentError("input length $(sequence_length) exceeds context_length $(config.context_length)"))

    token_min = minimum(input_token_ids)
    token_max = maximum(input_token_ids)
    1 <= token_min <= token_max <= config.vocab_size ||
        throw(ArgumentError("token ids must stay within 1:vocab_size"))

    hidden_states = reshape(model.token_embedding[:, vec(input_token_ids)], config.embedding_size, sequence_length, batch_size) .+
        reshape(model.position_embedding[:, 1:sequence_length], config.embedding_size, sequence_length, 1)

    for block in model.blocks
        hidden_states = block(hidden_states)
    end
    hidden_states = model.final_layer_norm(hidden_states)

    logits = model.token_embedding' * reshape(hidden_states, config.embedding_size, :)
    logits = reshape(logits, config.vocab_size, sequence_length, batch_size)
    return logits, nothing
end
