"""
Flux placeholder model for GPT-2 style decoder-only LM.
"""
struct FluxGPT2Model <: AbstractCausalLM
    config::GPT2Config
    parameters::Any
end

function build_gpt2_model(config::GPT2Config; keyword_arguments...)
    validate(config)
    return FluxGPT2Model(config, nothing)
end

model_config(model::FluxGPT2Model) = model.config

function lm_forward(
    model::FluxGPT2Model,
    input_token_ids::AbstractMatrix{<:Integer};
    cache = nothing,
    is_training::Bool = false,
)
    error("TODO v0.1: Flux lm_forward not implemented")
end
