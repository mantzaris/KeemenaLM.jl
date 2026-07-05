module Core

using JLD2
using JSON3
using StructTypes
using ChainRulesCore

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
    chat_repl,
    chatbot_behavior_cases,
    score_chatbot_behavior_completion,
    score_chatbot_behavior_suite,
    BUNDLE_SCHEMA_VERSION,
    BundleManifest,
    Bundle,
    available_models,
    download_model,
    resolve_bundle,
    download_model_artifact,
    resolve_model_artifact,
    resolve_tokenizer_bundle,
    save_weights_jld2,
    load_weights_jld2,
    save_bundle,
    load_bundle,
    load_model,
    causal_lm_cross_entropy,
    Trainer,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManifest,
    Checkpoint,
    save_checkpoint,
    load_checkpoint,
    train_step!

include("types.jl")
include("configs/gpt2.jl")

include("model/masking.jl")

include("generation/sampling.jl")
include("generation/stopping.jl")
include("generation/generate.jl")
include("generation/chat.jl")
include("generation/behavior_gate.jl")

include("io/bundle_schema.jl")
include("io/model_sources.jl")
include("io/weights_jld2.jl")
include("io/bundle_save.jl")
include("io/bundle_load.jl")

include("training/loss.jl")
include("training/trainer.jl")
include("training/checkpoints.jl")

end # module Core
