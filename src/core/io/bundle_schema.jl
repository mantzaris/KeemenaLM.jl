const BUNDLE_SCHEMA_VERSION = 1
const GPT2_PARAMETER_SCHEMA_V1 = "gpt2.v1"

"""
Bundle manifest for model package metadata.
"""
Base.@kwdef struct BundleManifest
    schema_version::Int = BUNDLE_SCHEMA_VERSION
    architecture::String = "gpt2"
    model_config_file::String = "model_config.json"
    weights_file::String = "weights.jld2"
    weights_format::String = "jld2"
    parameter_schema::String = GPT2_PARAMETER_SCHEMA_V1
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

function validate_bundle_manifest(manifest::BundleManifest)::BundleManifest
    manifest.schema_version == BUNDLE_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported bundle schema_version $(manifest.schema_version)"))
    manifest.architecture == "gpt2" ||
        throw(ArgumentError("unsupported architecture $(manifest.architecture)"))
    manifest.weights_format == "jld2" ||
        throw(ArgumentError("unsupported weights_format $(manifest.weights_format)"))
    manifest.parameter_schema == GPT2_PARAMETER_SCHEMA_V1 ||
        throw(ArgumentError("unsupported parameter_schema $(manifest.parameter_schema)"))
    validate_bundle_relative_path(manifest.model_config_file, "model_config_file")
    validate_bundle_relative_path(manifest.weights_file, "weights_file")
    manifest.tokenizer_path === nothing || validate_bundle_relative_path(manifest.tokenizer_path, "tokenizer_path")
    manifest.preprocessing_path === nothing || validate_bundle_relative_path(manifest.preprocessing_path, "preprocessing_path")
    return manifest
end

function validate_bundle_relative_path(path::AbstractString, field_name::AbstractString)::String
    isempty(path) && throw(ArgumentError("$(field_name) must be non-empty"))
    isabspath(path) && throw(ArgumentError("$(field_name) must be a relative bundle-local path"))

    normalized_path = normpath(path)
    normalized_path == "." && throw(ArgumentError("$(field_name) must not resolve to the bundle root"))
    startswith(normalized_path, "..") &&
        throw(ArgumentError("$(field_name) must stay within the bundle directory"))

    return normalized_path
end

function bundle_local_path(bundle_root::AbstractString, relative_path::AbstractString, field_name::AbstractString)::String
    normalized_relative_path = validate_bundle_relative_path(relative_path, field_name)
    return joinpath(bundle_root, normalized_relative_path)
end
