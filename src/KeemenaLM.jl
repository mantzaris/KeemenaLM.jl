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
    save_bundle,
    load_bundle,
    GenerationConfig,
    generate,
    ChatSession,
    chat!,
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
