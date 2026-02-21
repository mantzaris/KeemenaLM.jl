"""
Framework-neutral base type for model configuration objects.
"""
abstract type AbstractModelConfig end

"""
Framework-neutral base type for decoder-only causal language models.
"""
abstract type AbstractCausalLM end

"""
Configuration for text generation settings.
"""
Base.@kwdef struct GenerationConfig
    max_new_tokens::Int = 64
    temperature::Float64 = 1.0
    top_k::Int = 0
    top_p::Float64 = 1.0
    seed::Union{Nothing, Int} = nothing
    eos_token_id::Union{Nothing, Int} = nothing
    stop_sequences::Vector{String} = String[]
end

function lm_forward(
    model::AbstractCausalLM,
    input_token_ids::AbstractMatrix{<:Integer};
    cache = nothing,
    is_training::Bool = false,
)
    error("TODO v0.1: lm_forward not implemented for model type $(typeof(model))")
end

function model_config(model::AbstractCausalLM)
    error("TODO v0.1: model_config not implemented for model type $(typeof(model))")
end

function extract_weights(model::AbstractCausalLM)::Dict{String, Any}
    error("TODO v0.1: extract_weights not implemented for model type $(typeof(model))")
end

function load_weights!(model::AbstractCausalLM, weights::Dict{String, Any})
    error("TODO v0.1: load_weights! not implemented for model type $(typeof(model))")
end

preprocess_text(preprocessing, text::AbstractString) = text

function tokenizer_encode(tokenizer, text::AbstractString)::Vector{Int}
    error("TODO v0.1: tokenizer_encode not implemented for tokenizer type $(typeof(tokenizer))")
end

function tokenizer_decode(tokenizer, token_ids::AbstractVector{<:Integer})::String
    error("TODO v0.1: tokenizer_decode not implemented for tokenizer type $(typeof(tokenizer))")
end
