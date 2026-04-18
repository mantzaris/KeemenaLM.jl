module LuxBackend

using Lux

import ..Core: AbstractCausalLM, Bundle, GPT2Config, Trainer, causal_mask, validate
import ..Core: extract_weights, lm_forward, load_weights!, model_config, train_step!

include("gpt2_lux.jl")
include("weights_lux.jl")
include("train_lux.jl")

export LuxGPT2Model, build_gpt2_model, instantiate

function instantiate(config_or_bundle; keyword_arguments...)
    if config_or_bundle isa GPT2Config
        config = validate(config_or_bundle)
        return build_gpt2_model(config; keyword_arguments...)
    elseif config_or_bundle isa Bundle
        config = config_or_bundle.model_config
        config isa GPT2Config || error("Lux backend only supports GPT2Config in v0.1")
        model = build_gpt2_model(validate(config); keyword_arguments...)
        load_weights!(model, config_or_bundle.weights)
        return model
    else
        error("LuxBackend.instantiate expects GPT2Config or Bundle, got $(typeof(config_or_bundle))")
    end
end

end # module LuxBackend
