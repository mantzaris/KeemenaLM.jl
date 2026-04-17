"""
Minimal GPT-2 style decoder configuration.
"""
Base.@kwdef struct GPT2Config <: AbstractModelConfig
    vocab_size::Int = 50_257
    context_length::Int = 1_024
    num_layers::Int = 12
    num_heads::Int = 12
    embedding_size::Int = 768
    ffn_hidden_size::Int = 3_072
    dropout_probability::Float64 = 0.1
    bos_token_id::Union{Nothing, Int} = nothing
    eos_token_id::Union{Nothing, Int} = nothing
end

StructTypes.StructType(::Type{GPT2Config}) = StructTypes.Struct()

"""
Validate a GPT-2 configuration and return it when valid.
"""
function validate(config::GPT2Config)
    config.vocab_size > 0 || throw(ArgumentError("vocab_size must be > 0"))
    config.context_length > 0 || throw(ArgumentError("context_length must be > 0"))
    config.num_layers > 0 || throw(ArgumentError("num_layers must be > 0"))
    config.num_heads > 0 || throw(ArgumentError("num_heads must be > 0"))
    config.embedding_size > 0 || throw(ArgumentError("embedding_size must be > 0"))
    config.ffn_hidden_size > 0 || throw(ArgumentError("ffn_hidden_size must be > 0"))
    isfinite(config.dropout_probability) || throw(ArgumentError("dropout_probability must be finite"))
    0.0 <= config.dropout_probability < 1.0 || throw(ArgumentError("dropout_probability must be in [0.0, 1.0)"))
    config.embedding_size % config.num_heads == 0 || throw(ArgumentError("embedding_size must be divisible by num_heads"))
    config.num_heads <= config.embedding_size || throw(ArgumentError("num_heads must be <= embedding_size"))

    if config.bos_token_id !== nothing
        1 <= config.bos_token_id <= config.vocab_size || throw(ArgumentError("bos_token_id must be in 1:vocab_size"))
    end

    if config.eos_token_id !== nothing
        1 <= config.eos_token_id <= config.vocab_size || throw(ArgumentError("eos_token_id must be in 1:vocab_size"))
    end

    return config
end
