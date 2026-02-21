"""
Minimal training container shared by backend-specific training methods.
"""
mutable struct Trainer{ModelType<:AbstractCausalLM}
    model::ModelType
    optimizer::Any
    backend_state::Any
    metadata::Dict{String, Any}
end

function Trainer(
    model::ModelType;
    optimizer = nothing,
    backend_state = nothing,
    metadata::Dict{String, Any} = Dict{String, Any}(),
) where {ModelType<:AbstractCausalLM}
    return Trainer{ModelType}(model, optimizer, backend_state, metadata)
end

function train_step!(
    trainer::Trainer{<:AbstractCausalLM},
    input_token_ids::AbstractMatrix{<:Integer},
    target_token_ids::AbstractMatrix{<:Integer},
)
    error("TODO v0.1: train_step! not implemented for model type $(typeof(trainer.model))")
end
