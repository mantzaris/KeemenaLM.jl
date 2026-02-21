module Core

using JLD2
using JSON3
using StructTypes

export
    AbstractModelConfig,
    AbstractCausalLM,
    GPT2Config,
    validate,
    GenerationConfig,
    lm_forward,
    model_config,
    extract_weights,
    load_weights!,
    preprocess_text,
    tokenizer_encode,
    tokenizer_decode,
    causal_mask,
    generate,
    ChatSession,
    chat!,
    BUNDLE_SCHEMA_VERSION,
    BundleManifest,
    Bundle,
    save_weights_jld2,
    load_weights_jld2,
    save_bundle,
    load_bundle,
    causal_lm_cross_entropy,
    Trainer,
    train_step!

include("types.jl")
include("configs/gpt2.jl")

include("model/masking.jl")

include("generation/sampling.jl")
include("generation/stopping.jl")
include("generation/generate.jl")
include("generation/chat.jl")

include("io/bundle_schema.jl")
include("io/weights_jld2.jl")
include("io/bundle_save.jl")
include("io/bundle_load.jl")

include("training/loss.jl")
include("training/trainer.jl")

end # module Core
