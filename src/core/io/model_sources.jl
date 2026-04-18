using Artifacts
using Pkg

Base.@kwdef struct OfficialModelSpec
    key::String
    artifact_name::String
    description::String
    install_note::String
    tokenizer_note::String
    architecture::String = "gpt2"
end

const OFFICIAL_MODEL_REGISTRY = Dict(
    "tiny-demo" => OfficialModelSpec(
        key = "tiny-demo",
        artifact_name = "keemenalm_tiny_demo",
        description = "Tiny GPT-2 demo bundle distributed via Julia Artifacts.",
        install_note = "Stage 6 uses local artifact registration. Run tools/build_public_model_artifact.jl to bind this model in artifacts/Artifacts.toml before resolving it by key.",
        tokenizer_note = "Tokenizer and preprocessing are still supplied explicitly; use the DemoCharTokenizer convention from the examples for this demo model.",
        architecture = "gpt2",
    ),
)

"""
Resolve a bundle source to a local directory path.
Stage 6 supports local directories and official artifact-backed model keys.
"""
function resolve_bundle(source; artifacts_toml::AbstractString = default_artifacts_toml())::String
    source_path = source isa AbstractString ? source : string(source)
    isdir(source_path) && return validate_bundle_directory(source_path)

    source_key = normalize_model_key(source)
    if source_key !== nothing && haskey(OFFICIAL_MODEL_REGISTRY, source_key)
        return resolve_official_model_bundle(source_key; artifacts_toml = artifacts_toml)
    end

    return validate_bundle_directory(source_path)
end

function available_models(; artifacts_toml::AbstractString = default_artifacts_toml())
    return [
        (
            key = specification.key,
            description = specification.description,
            artifact_name = specification.artifact_name,
            architecture = specification.architecture,
            installed = artifact_is_available(specification; artifacts_toml = artifacts_toml),
            install_note = specification.install_note,
            tokenizer_note = specification.tokenizer_note,
        ) for specification in sort!(collect(values(OFFICIAL_MODEL_REGISTRY)); by = specification -> specification.key)
    ]
end

function download_model(model_key; artifacts_toml::AbstractString = default_artifacts_toml())::String
    normalized_key = normalize_model_key(model_key)
    normalized_key === nothing && throw(ArgumentError("official model keys must be strings or symbols"))
    return resolve_official_model_bundle(normalized_key; artifacts_toml = artifacts_toml)
end

function load_model(source; backend::Symbol = :flux, artifacts_toml::AbstractString = default_artifacts_toml(), keyword_arguments...)
    bundle = load_bundle(source; artifacts_toml = artifacts_toml)
    return parentmodule(@__MODULE__).instantiate(bundle; backend = backend, keyword_arguments...)
end

function default_artifacts_toml()::String
    return joinpath(package_root(), "artifacts", "Artifacts.toml")
end

function package_root()::String
    return normpath(joinpath(@__DIR__, "..", "..", ".."))
end

function normalize_model_key(source)::Union{Nothing, String}
    if source isa Symbol
        return String(source)
    elseif source isa AbstractString
        return source
    else
        return nothing
    end
end

function official_model_spec(model_key::AbstractString)::OfficialModelSpec
    if !haskey(OFFICIAL_MODEL_REGISTRY, model_key)
        available_keys = join(sort!(collect(keys(OFFICIAL_MODEL_REGISTRY))), ", ")
        throw(ArgumentError("unknown official model key $(model_key); available keys: $(available_keys)"))
    end
    return OFFICIAL_MODEL_REGISTRY[model_key]
end

function resolve_official_model_bundle(model_key::AbstractString; artifacts_toml::AbstractString)::String
    specification = official_model_spec(model_key)
    isfile(artifacts_toml) ||
        throw(ArgumentError("official model registry file does not exist: $(artifacts_toml). Stage 6 supports locally registered official artifacts only; run tools/build_public_model_artifact.jl or provide a custom artifacts_toml path."))

    artifact_hash_value = Artifacts.artifact_hash(specification.artifact_name, artifacts_toml)
    artifact_hash_value === nothing &&
        throw(ArgumentError("official model $(model_key) is not bound in $(artifacts_toml). Stage 6 does not ship a remote fetch path here; run tools/build_public_model_artifact.jl for $(model_key) first."))

    Pkg.Artifacts.ensure_artifact_installed(specification.artifact_name, artifacts_toml)
    return validate_bundle_directory(Artifacts.artifact_path(artifact_hash_value))
end

function artifact_is_available(specification::OfficialModelSpec; artifacts_toml::AbstractString)::Bool
    isfile(artifacts_toml) || return false
    artifact_hash_value = Artifacts.artifact_hash(specification.artifact_name, artifacts_toml)
    artifact_hash_value === nothing && return false
    return isdir(Artifacts.artifact_path(artifact_hash_value))
end

function validate_bundle_directory(source_path::AbstractString)::String
    isdir(source_path) || throw(ArgumentError("bundle source is not a directory: $(source_path)"))

    bundle_manifest_path = joinpath(source_path, "bundle.json")
    isfile(bundle_manifest_path) ||
        throw(ArgumentError("bundle directory is missing bundle.json: $(source_path)"))

    return abspath(source_path)
end
