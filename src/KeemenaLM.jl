module KeemenaLM

include("core/Core.jl")
using .Core

include("backends/flux/FluxBackend.jl")
include("backends/lux/LuxBackend.jl")

export
    GPT2Config,
    validate,
    Bundle,
    BundleManifest,
    Checkpoint,
    available_models,
    download_model,
    resolve_bundle,
    download_model_artifact,
    resolve_model_artifact,
    resolve_tokenizer_bundle,
    save_bundle,
    load_bundle,
    load_model,
    save_checkpoint,
    load_checkpoint,
    GenerationConfig,
    generate,
    ChatSession,
    chat!,
    chat_repl,
    chatbot_behavior_cases,
    score_chatbot_behavior_completion,
    score_chatbot_behavior_suite,
    instantiate

function instantiate(config_or_bundle; backend::Symbol = :flux, keyword_arguments...)
    if backend === :flux
        return FluxBackend.instantiate(config_or_bundle; keyword_arguments...)
    elseif backend === :lux
        return LuxBackend.instantiate(config_or_bundle; keyword_arguments...)
    else
        error("Unknown backend: $(backend). Supported backends are :flux and :lux")
    end
end

end # module KeemenaLM
