function extract_weights(model::FluxGPT2Model)::Dict{String, Any}
    weights = Dict{String, Any}(
        "token_embedding" => copy(model.token_embedding),
        "position_embedding" => copy(model.position_embedding),
        "final_layer_norm.weight" => copy(model.final_layer_norm.weight),
        "final_layer_norm.bias" => copy(model.final_layer_norm.bias),
    )

    for (block_index, block) in enumerate(model.blocks)
        block_prefix = "blocks.$(block_index)"

        weights["$(block_prefix).attention_norm.weight"] = copy(block.attention_norm.weight)
        weights["$(block_prefix).attention_norm.bias"] = copy(block.attention_norm.bias)
        weights["$(block_prefix).attention.q_proj.weight"] = copy(block.attention.q_proj.weight)
        weights["$(block_prefix).attention.q_proj.bias"] = copy(block.attention.q_proj.bias)
        weights["$(block_prefix).attention.k_proj.weight"] = copy(block.attention.k_proj.weight)
        weights["$(block_prefix).attention.k_proj.bias"] = copy(block.attention.k_proj.bias)
        weights["$(block_prefix).attention.v_proj.weight"] = copy(block.attention.v_proj.weight)
        weights["$(block_prefix).attention.v_proj.bias"] = copy(block.attention.v_proj.bias)
        weights["$(block_prefix).attention.out_proj.weight"] = copy(block.attention.out_proj.weight)
        weights["$(block_prefix).attention.out_proj.bias"] = copy(block.attention.out_proj.bias)
        weights["$(block_prefix).ffn_norm.weight"] = copy(block.ffn_norm.weight)
        weights["$(block_prefix).ffn_norm.bias"] = copy(block.ffn_norm.bias)
        weights["$(block_prefix).ffn_in.weight"] = copy(block.ffn_in.weight)
        weights["$(block_prefix).ffn_in.bias"] = copy(block.ffn_in.bias)
        weights["$(block_prefix).ffn_out.weight"] = copy(block.ffn_out.weight)
        weights["$(block_prefix).ffn_out.bias"] = copy(block.ffn_out.bias)
    end

    return weights
end

function load_weights!(model::FluxGPT2Model, weights::Dict{String, Any})
    load_matrix!(model.token_embedding, weights, "token_embedding")
    load_matrix!(model.position_embedding, weights, "position_embedding")
    load_vector!(model.final_layer_norm.weight, weights, "final_layer_norm.weight")
    load_vector!(model.final_layer_norm.bias, weights, "final_layer_norm.bias")

    for (block_index, block) in enumerate(model.blocks)
        block_prefix = "blocks.$(block_index)"

        load_vector!(block.attention_norm.weight, weights, "$(block_prefix).attention_norm.weight")
        load_vector!(block.attention_norm.bias, weights, "$(block_prefix).attention_norm.bias")
        load_matrix!(block.attention.q_proj.weight, weights, "$(block_prefix).attention.q_proj.weight")
        load_vector!(block.attention.q_proj.bias, weights, "$(block_prefix).attention.q_proj.bias")
        load_matrix!(block.attention.k_proj.weight, weights, "$(block_prefix).attention.k_proj.weight")
        load_vector!(block.attention.k_proj.bias, weights, "$(block_prefix).attention.k_proj.bias")
        load_matrix!(block.attention.v_proj.weight, weights, "$(block_prefix).attention.v_proj.weight")
        load_vector!(block.attention.v_proj.bias, weights, "$(block_prefix).attention.v_proj.bias")
        load_matrix!(block.attention.out_proj.weight, weights, "$(block_prefix).attention.out_proj.weight")
        load_vector!(block.attention.out_proj.bias, weights, "$(block_prefix).attention.out_proj.bias")
        load_vector!(block.ffn_norm.weight, weights, "$(block_prefix).ffn_norm.weight")
        load_vector!(block.ffn_norm.bias, weights, "$(block_prefix).ffn_norm.bias")
        load_matrix!(block.ffn_in.weight, weights, "$(block_prefix).ffn_in.weight")
        load_vector!(block.ffn_in.bias, weights, "$(block_prefix).ffn_in.bias")
        load_matrix!(block.ffn_out.weight, weights, "$(block_prefix).ffn_out.weight")
        load_vector!(block.ffn_out.bias, weights, "$(block_prefix).ffn_out.bias")
    end

    return model
end

function load_matrix!(destination::Matrix{Float32}, weights::Dict{String, Any}, key::String)
    source = get_required_weight(weights, key)
    size(source) == size(destination) ||
        throw(ArgumentError("weight $(key) has shape $(size(source)), expected $(size(destination))"))
    destination .= Float32.(source)
    return destination
end

function load_vector!(destination::Vector{Float32}, weights::Dict{String, Any}, key::String)
    source = get_required_weight(weights, key)
    size(source) == size(destination) ||
        throw(ArgumentError("weight $(key) has shape $(size(source)), expected $(size(destination))"))
    destination .= Float32.(source)
    return destination
end

function get_required_weight(weights::Dict{String, Any}, key::String)
    haskey(weights, key) || throw(ArgumentError("weights dictionary is missing required key $(key)"))
    return weights[key]
end
