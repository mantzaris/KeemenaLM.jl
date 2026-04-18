struct LuxLinear{WeightType<:AbstractMatrix{<:AbstractFloat}, BiasType<:AbstractVector{<:AbstractFloat}}
    weight::WeightType
    bias::BiasType
end

struct LuxLayerNorm{
    WeightType<:AbstractVector{<:AbstractFloat},
    BiasType<:AbstractVector{<:AbstractFloat},
    EpsilonType<:AbstractFloat,
}
    weight::WeightType
    bias::BiasType
    epsilon::EpsilonType
end

struct LuxCausalSelfAttention{
    QProjType<:LuxLinear,
    KProjType<:LuxLinear,
    VProjType<:LuxLinear,
    OutProjType<:LuxLinear,
}
    q_proj::QProjType
    k_proj::KProjType
    v_proj::VProjType
    out_proj::OutProjType
    num_heads::Int
end

struct LuxTransformerBlock{
    AttentionNormType<:LuxLayerNorm,
    AttentionType<:LuxCausalSelfAttention,
    FfnNormType<:LuxLayerNorm,
    FfnInType<:LuxLinear,
    FfnOutType<:LuxLinear,
}
    attention_norm::AttentionNormType
    attention::AttentionType
    ffn_norm::FfnNormType
    ffn_in::FfnInType
    ffn_out::FfnOutType
end

"""
Lux GPT-2 style decoder-only LM for the Stage 7 inference path.
"""
struct LuxGPT2Model{
    TokenEmbeddingType<:AbstractMatrix{<:AbstractFloat},
    PositionEmbeddingType<:AbstractMatrix{<:AbstractFloat},
    BlockType<:LuxTransformerBlock,
    FinalLayerNormType<:LuxLayerNorm,
} <: AbstractCausalLM
    config::GPT2Config
    token_embedding::TokenEmbeddingType
    position_embedding::PositionEmbeddingType
    blocks::Vector{BlockType}
    final_layer_norm::FinalLayerNormType
end

function build_gpt2_model(
    config::GPT2Config;
    seed::Int = 0,
    init_scale::Float32 = 0.02f0,
)
    validate(config)

    parameter_offset = Ref(seed)
    token_embedding = initialized_parameter(
        config.embedding_size,
        config.vocab_size,
        take_parameter_offset!(parameter_offset, config.embedding_size * config.vocab_size);
        scale = init_scale,
    )
    position_embedding = initialized_parameter(
        config.embedding_size,
        config.context_length,
        take_parameter_offset!(parameter_offset, config.embedding_size * config.context_length);
        scale = init_scale,
    )
    blocks = [
        LuxTransformerBlock(
            LuxLayerNorm(ones(Float32, config.embedding_size), zeros(Float32, config.embedding_size), 1.0f-5),
            LuxCausalSelfAttention(
                LuxLinear(
                    initialized_parameter(
                        config.embedding_size,
                        config.embedding_size,
                        take_parameter_offset!(parameter_offset, config.embedding_size * config.embedding_size);
                        scale = init_scale,
                    ),
                    zeros(Float32, config.embedding_size),
                ),
                LuxLinear(
                    initialized_parameter(
                        config.embedding_size,
                        config.embedding_size,
                        take_parameter_offset!(parameter_offset, config.embedding_size * config.embedding_size);
                        scale = init_scale,
                    ),
                    zeros(Float32, config.embedding_size),
                ),
                LuxLinear(
                    initialized_parameter(
                        config.embedding_size,
                        config.embedding_size,
                        take_parameter_offset!(parameter_offset, config.embedding_size * config.embedding_size);
                        scale = init_scale,
                    ),
                    zeros(Float32, config.embedding_size),
                ),
                LuxLinear(
                    initialized_parameter(
                        config.embedding_size,
                        config.embedding_size,
                        take_parameter_offset!(parameter_offset, config.embedding_size * config.embedding_size);
                        scale = init_scale,
                    ),
                    zeros(Float32, config.embedding_size),
                ),
                config.num_heads,
            ),
            LuxLayerNorm(ones(Float32, config.embedding_size), zeros(Float32, config.embedding_size), 1.0f-5),
            LuxLinear(
                initialized_parameter(
                    config.ffn_hidden_size,
                    config.embedding_size,
                    take_parameter_offset!(parameter_offset, config.ffn_hidden_size * config.embedding_size);
                    scale = init_scale,
                ),
                zeros(Float32, config.ffn_hidden_size),
            ),
            LuxLinear(
                initialized_parameter(
                    config.embedding_size,
                    config.ffn_hidden_size,
                    take_parameter_offset!(parameter_offset, config.embedding_size * config.ffn_hidden_size);
                    scale = init_scale,
                ),
                zeros(Float32, config.embedding_size),
            ),
        ) for _ in 1:config.num_layers
    ]
    final_layer_norm = LuxLayerNorm(ones(Float32, config.embedding_size), zeros(Float32, config.embedding_size), 1.0f-5)

    return LuxGPT2Model(config, token_embedding, position_embedding, blocks, final_layer_norm)
end

model_config(model::LuxGPT2Model) = model.config

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

function (layer::LuxLinear)(x::AbstractArray{<:Real, 3})
    input_size, sequence_length, batch_size = size(x)
    size(layer.weight, 2) == input_size ||
        throw(DimensionMismatch("LuxLinear expected input size $(size(layer.weight, 2)), got $(input_size)"))

    flattened_input = reshape(x, input_size, :)
    projected = layer.weight * flattened_input .+ layer.bias
    return reshape(projected, size(layer.weight, 1), sequence_length, batch_size)
end

function (layer_norm::LuxLayerNorm)(x::AbstractArray{<:Real, 3})
    feature_count = size(x, 1)
    mean_x = sum(x; dims = 1) ./ feature_count
    variance_x = sum((x .- mean_x) .^ 2; dims = 1) ./ feature_count
    normalized_x = (x .- mean_x) ./ sqrt.(variance_x .+ layer_norm.epsilon)
    return normalized_x .* reshape(layer_norm.weight, :, 1, 1) .+ reshape(layer_norm.bias, :, 1, 1)
end

function (attention::LuxCausalSelfAttention)(x::AbstractArray{<:Real, 3})
    embedding_size, sequence_length, batch_size = size(x)
    head_size = div(embedding_size, attention.num_heads)

    queries = reshape(attention.q_proj(x), head_size, attention.num_heads, sequence_length, batch_size)
    keys = reshape(attention.k_proj(x), head_size, attention.num_heads, sequence_length, batch_size)
    values = reshape(attention.v_proj(x), head_size, attention.num_heads, sequence_length, batch_size)

    mask = causal_mask(sequence_length)
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
    attention_weights = Lux.softmax(masked_scores; dims = 2)
    return values * permutedims(attention_weights, (2, 1))
end

function (block::LuxTransformerBlock)(x::AbstractArray{<:Real, 3})
    attention_output = block.attention(block.attention_norm(x))
    residual_x = x .+ attention_output
    ffn_hidden = Lux.gelu.(block.ffn_in(block.ffn_norm(residual_x)))
    ffn_output = block.ffn_out(ffn_hidden)
    return residual_x .+ ffn_output
end

function lm_forward(
    model::LuxGPT2Model,
    input_token_ids::AbstractMatrix{<:Integer};
    cache = nothing,
    is_training::Bool = false,
)
    cache === nothing || throw(ArgumentError("Stage 7 Lux lm_forward does not support cache yet"))

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
