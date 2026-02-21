"""
Lux placeholder model for GPT-2 style decoder-only LM.
"""
struct LuxGPT2Model <: AbstractCausalLM
    config::GPT2Config
    parameters::Any
    states::Any
end

function build_gpt2_model(config::GPT2Config; keyword_arguments...)
    validate(config)
    return LuxGPT2Model(config, nothing, nothing)
end

model_config(model::LuxGPT2Model) = model.config

function lm_forward(
    model::LuxGPT2Model,
    input_token_ids::AbstractMatrix{<:Integer};
    cache = nothing,
    is_training::Bool = false,
)
    error("TODO v0.1: Lux lm_forward not implemented")
end
