struct FluxLinear
    weight::Matrix{Float32}
    bias::Vector{Float32}
end

Flux.@layer FluxLinear

struct FluxLayerNorm
    weight::Vector{Float32}
    bias::Vector{Float32}
    epsilon::Float32
end

Flux.@layer FluxLayerNorm

struct FluxCausalSelfAttention
    q_proj::FluxLinear
    k_proj::FluxLinear
    v_proj::FluxLinear
    out_proj::FluxLinear
    num_heads::Int
end

Flux.@layer FluxCausalSelfAttention

struct FluxTransformerBlock
    attention_norm::FluxLayerNorm
    attention::FluxCausalSelfAttention
    ffn_norm::FluxLayerNorm
    ffn_in::FluxLinear
    ffn_out::FluxLinear
end

Flux.@layer FluxTransformerBlock

"""
Flux GPT-2 style decoder-only LM for the Stage 1 CPU path.
"""
struct FluxGPT2Model <: AbstractCausalLM
    config::GPT2Config
    token_embedding::Matrix{Float32}
    position_embedding::Matrix{Float32}
    blocks::Vector{FluxTransformerBlock}
    final_layer_norm::FluxLayerNorm
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

    flattened_input = reshape(Float32.(x), input_size, :)
    projected = layer.weight * flattened_input
    projected .+= layer.bias
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

    attention_output = Array{Float32}(undef, head_size, attention.num_heads, sequence_length, batch_size)
    mask = causal_mask(sequence_length)
    scale = inv(sqrt(Float32(head_size)))

    for batch_index in 1:batch_size
        for head_index in 1:attention.num_heads
            for query_position in 1:sequence_length
                attention_scores = fill(-Inf32, sequence_length)
                query_vector = @view queries[:, head_index, query_position, batch_index]

                for key_position in 1:sequence_length
                    if mask[query_position, key_position]
                        key_vector = @view keys[:, head_index, key_position, batch_index]
                        attention_scores[key_position] = sum(query_vector .* key_vector) * scale
                    end
                end

                attention_weights = Flux.softmax(attention_scores)
                weighted_value = zeros(Float32, head_size)
                for value_position in 1:sequence_length
                    value_vector = @view values[:, head_index, value_position, batch_index]
                    weighted_value .+= attention_weights[value_position] .* value_vector
                end
                attention_output[:, head_index, query_position, batch_index] = weighted_value
            end
        end
    end

    merged_output = reshape(attention_output, embedding_size, sequence_length, batch_size)
    return attention.out_proj(merged_output)
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
    is_training && @warn("Stage 1 Flux lm_forward ignores dropout because only the CPU inference path is implemented")

    sequence_length, batch_size = size(input_token_ids)
    config = model.config

    sequence_length > 0 || throw(ArgumentError("input_token_ids must have at least one time step"))
    sequence_length <= config.context_length ||
        throw(ArgumentError("input length $(sequence_length) exceeds context_length $(config.context_length)"))

    for token_id in input_token_ids
        1 <= token_id <= config.vocab_size ||
            throw(ArgumentError("token id $(token_id) is outside 1:vocab_size"))
    end

    hidden_states = reshape(model.token_embedding[:, vec(input_token_ids)], config.embedding_size, sequence_length, batch_size)
    hidden_states .+= reshape(model.position_embedding[:, 1:sequence_length], config.embedding_size, sequence_length, 1)

    for block in model.blocks
        hidden_states = block(hidden_states)
    end
    hidden_states = model.final_layer_norm(hidden_states)

    logits = model.token_embedding' * reshape(hidden_states, config.embedding_size, :)
    logits = reshape(logits, config.vocab_size, sequence_length, batch_size)
    return logits, nothing
end
