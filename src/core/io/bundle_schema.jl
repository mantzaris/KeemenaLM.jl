const BUNDLE_SCHEMA_VERSION = 1

"""
Bundle manifest for model package metadata.
"""
Base.@kwdef struct BundleManifest
    schema_version::Int = BUNDLE_SCHEMA_VERSION
    architecture::String = "gpt2"
    model_config_file::String = "model_config.json"
    weights_file::String = "weights.jld2"
    weights_format::String = "jld2"
    tokenizer_path::Union{Nothing, String} = nothing
    preprocessing_path::Union{Nothing, String} = nothing
    metadata::Dict{String, Any} = Dict{String, Any}()
end

StructTypes.StructType(::Type{BundleManifest}) = StructTypes.Struct()

"""
In-memory bundle container used for save/load APIs.
"""
Base.@kwdef struct Bundle
    manifest::BundleManifest = BundleManifest()
    model_config::AbstractModelConfig
    weights::Dict{String, Any} = Dict{String, Any}()
    tokenizer::Any = nothing
    preprocessing::Any = nothing
end

function Bundle(model_config::AbstractModelConfig; keyword_arguments...)
    return Bundle(; model_config = model_config, keyword_arguments...)
end
