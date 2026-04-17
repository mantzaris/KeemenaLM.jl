"""
Minimal training container shared by backend-specific training methods.
"""
mutable struct Trainer{ModelType<:AbstractCausalLM}
    model::ModelType
    optimizer::Any
    optimizer_state::Any
    backend::Symbol
    step::Int
    epoch::Int
    rng_state::Any
    metadata::Dict{String, Any}
end

function Trainer(
    model::ModelType;
    optimizer = nothing,
    optimizer_state = nothing,
    backend::Symbol = :unknown,
    step::Int = 0,
    epoch::Int = 0,
    rng_state = nothing,
    metadata::AbstractDict = Dict{String, Any}(),
) where {ModelType<:AbstractCausalLM}
    step >= 0 || throw(ArgumentError("step must be >= 0"))
    epoch >= 0 || throw(ArgumentError("epoch must be >= 0"))
    normalized_metadata = Dict{String, Any}(String(key) => value for (key, value) in pairs(metadata))
    return Trainer{ModelType}(model, optimizer, optimizer_state, backend, step, epoch, rng_state, normalized_metadata)
end

function train_step!(
    trainer::Trainer{<:AbstractCausalLM},
    input_token_ids::AbstractMatrix{<:Integer},
    target_token_ids::AbstractMatrix{<:Integer},
)
    error("TODO v0.1: train_step! not implemented for model type $(typeof(trainer.model))")
end
