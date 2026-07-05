using Artifacts
using Pkg

Base.@kwdef struct OfficialModelSpec
    key::String
    artifact_name::String
    description::String
    install_note::String
    tokenizer_note::String
    architecture::String = "gpt2"
    bundle_subdir::String = ""
    tokenizer_bundle_subdir::String = ""
end

const OFFICIAL_MODEL_REGISTRY = Dict(
    "tiny-demo" => OfficialModelSpec(
        key = "tiny-demo",
        artifact_name = "keemenalm_tiny_demo",
        description = "Tiny GPT-2 demo bundle distributed via Julia Artifacts.",
        install_note = "Run tools/build_public_model_artifact.jl to bind this local demo model in artifacts/Artifacts.toml before resolving it by key.",
        tokenizer_note = "Tokenizer and preprocessing are supplied explicitly; use the DemoCharTokenizer convention from the examples for this demo model.",
        architecture = "gpt2",
    ),
    "tiny-chatbot-v9-broad-336m" => OfficialModelSpec(
        key = "tiny-chatbot-v9-broad-336m",
        artifact_name = "keemenalm_tiny_chatbot_v9_broad_336m",
        description = "Current v9 broad 336M scratch-chatbot research baseline with bundle, tokenizer bundle, and run metadata.",
        install_note = "Bind the release tarball in artifacts/Artifacts.toml with tools/package_tiny_chatbot_v9_release_artifact.sh after uploading it, then call download_model(\"tiny-chatbot-v9-broad-336m\").",
        tokenizer_note = "Use resolve_tokenizer_bundle(\"tiny-chatbot-v9-broad-336m\") with the tools/subword_real_text environment to load the matching KeemenaSubwords tokenizer bundle.",
        architecture = "gpt2",
        bundle_subdir = "bundle",
        tokenizer_bundle_subdir = "tokenizer_bundle",
    ),
)

"""
Resolve a bundle source to a local bundle directory path.

Sources can be:

- a bundle directory containing `bundle.json`
- a release-artifact root containing `bundle/bundle.json`
- an official model key from `available_models()`
"""
function resolve_bundle(source; artifacts_toml::AbstractString = default_artifacts_toml())::String
    source_path = source isa AbstractString ? source : string(source)
    if isdir(source_path)
        return resolve_local_bundle(source_path)
    end

    source_key = normalize_model_key(source)
    if source_key !== nothing && haskey(OFFICIAL_MODEL_REGISTRY, source_key)
        return resolve_official_model_bundle(source_key; artifacts_toml = artifacts_toml)
    end

    return validate_bundle_directory(source_path)
end

"""
Resolve a full model artifact root.

For the v9 chatbot artifact this root contains `bundle/`, `tokenizer_bundle/`,
and metadata. Bundle-only local directories are accepted for compatibility.
"""
function resolve_model_artifact(source; artifacts_toml::AbstractString = default_artifacts_toml())::String
    source_path = source isa AbstractString ? source : string(source)
    if isdir(source_path)
        if isfile(joinpath(source_path, "bundle.json")) || isfile(joinpath(source_path, "bundle", "bundle.json"))
            return abspath(source_path)
        end
        throw(ArgumentError("model artifact directory must contain bundle.json or bundle/bundle.json: $(source_path)"))
    end

    source_key = normalize_model_key(source)
    if source_key !== nothing && haskey(OFFICIAL_MODEL_REGISTRY, source_key)
        return resolve_official_model_root(source_key; artifacts_toml = artifacts_toml)
    end

    throw(ArgumentError("model artifact source is not a directory or official model key: $(source_path)"))
end

"""
Resolve the tokenizer bundle directory for a local artifact root or official key.
"""
function resolve_tokenizer_bundle(source; artifacts_toml::AbstractString = default_artifacts_toml())::String
    source_path = source isa AbstractString ? source : string(source)
    if isdir(source_path)
        direct_tokenizer = validate_tokenizer_bundle_directory(source_path; required = false)
        direct_tokenizer !== nothing && return direct_tokenizer

        nested_tokenizer = joinpath(source_path, "tokenizer_bundle")
        return validate_tokenizer_bundle_directory(nested_tokenizer)
    end

    source_key = normalize_model_key(source)
    if source_key !== nothing && haskey(OFFICIAL_MODEL_REGISTRY, source_key)
        specification = official_model_spec(source_key)
        isempty(specification.tokenizer_bundle_subdir) && throw(ArgumentError("official model $(source_key) does not define a tokenizer bundle. $(specification.tokenizer_note)"))
        artifact_root = resolve_official_model_root(source_key; artifacts_toml = artifacts_toml)
        return validate_tokenizer_bundle_directory(joinpath(artifact_root, specification.tokenizer_bundle_subdir))
    end

    throw(ArgumentError("tokenizer bundle source is not a directory or official model key: $(source_path)"))
end

function available_models(; artifacts_toml::AbstractString = default_artifacts_toml())
    return [
        (
            key = specification.key,
            description = specification.description,
            artifact_name = specification.artifact_name,
            architecture = specification.architecture,
            installed = artifact_is_available(specification; artifacts_toml = artifacts_toml),
            bundle_subdir = specification.bundle_subdir,
            tokenizer_bundle_subdir = specification.tokenizer_bundle_subdir,
            has_tokenizer_bundle = !isempty(specification.tokenizer_bundle_subdir),
            install_note = specification.install_note,
            tokenizer_note = specification.tokenizer_note,
        ) for specification in sort!(collect(values(OFFICIAL_MODEL_REGISTRY)); by = specification -> specification.key)
    ]
end

"""
Ensure an official model artifact is installed and return its bundle directory.
"""
function download_model(model_key; artifacts_toml::AbstractString = default_artifacts_toml())::String
    normalized_key = normalize_model_key(model_key)
    normalized_key === nothing && throw(ArgumentError("official model keys must be strings or symbols"))
    return resolve_official_model_bundle(normalized_key; artifacts_toml = artifacts_toml)
end

"""
Ensure an official model artifact is installed and return its artifact root.
"""
function download_model_artifact(model_key; artifacts_toml::AbstractString = default_artifacts_toml())::String
    normalized_key = normalize_model_key(model_key)
    normalized_key === nothing && throw(ArgumentError("official model keys must be strings or symbols"))
    return resolve_official_model_root(normalized_key; artifacts_toml = artifacts_toml)
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
    artifact_root = resolve_official_model_root(model_key; artifacts_toml = artifacts_toml)
    bundle_directory = isempty(specification.bundle_subdir) ? artifact_root : joinpath(artifact_root, specification.bundle_subdir)
    return validate_bundle_directory(bundle_directory)
end

function resolve_official_model_root(model_key::AbstractString; artifacts_toml::AbstractString)::String
    specification = official_model_spec(model_key)
    isfile(artifacts_toml) ||
        throw(ArgumentError("official model registry file does not exist: $(artifacts_toml). Use tools/package_tiny_chatbot_v9_release_artifact.sh to prepare a release artifact or tools/build_public_model_artifact.jl for tiny-demo."))

    artifact_hash_value = Artifacts.artifact_hash(specification.artifact_name, artifacts_toml)
    artifact_hash_value === nothing &&
        throw(ArgumentError("official model $(model_key) is not bound in $(artifacts_toml). $(specification.install_note)"))

    Pkg.Artifacts.ensure_artifact_installed(specification.artifact_name, artifacts_toml)
    artifact_root = Artifacts.artifact_path(artifact_hash_value)
    isdir(artifact_root) || throw(ArgumentError("official model artifact path does not exist after install: $(artifact_root)"))
    return artifact_root
end

function artifact_is_available(specification::OfficialModelSpec; artifacts_toml::AbstractString)::Bool
    isfile(artifacts_toml) || return false
    artifact_hash_value = Artifacts.artifact_hash(specification.artifact_name, artifacts_toml)
    artifact_hash_value === nothing && return false
    artifact_root = Artifacts.artifact_path(artifact_hash_value)
    isdir(artifact_root) || return false
    bundle_directory = isempty(specification.bundle_subdir) ? artifact_root : joinpath(artifact_root, specification.bundle_subdir)
    return isfile(joinpath(bundle_directory, "bundle.json"))
end

function resolve_local_bundle(source_path::AbstractString)::String
    if isfile(joinpath(source_path, "bundle.json"))
        return validate_bundle_directory(source_path)
    end

    nested_bundle_directory = joinpath(source_path, "bundle")
    if isfile(joinpath(nested_bundle_directory, "bundle.json"))
        return validate_bundle_directory(nested_bundle_directory)
    end

    return validate_bundle_directory(source_path)
end

function validate_bundle_directory(source_path::AbstractString)::String
    isdir(source_path) || throw(ArgumentError("bundle source is not a directory: $(source_path)"))

    bundle_manifest_path = joinpath(source_path, "bundle.json")
    isfile(bundle_manifest_path) ||
        throw(ArgumentError("bundle directory is missing bundle.json: $(source_path)"))

    return abspath(source_path)
end

function validate_tokenizer_bundle_directory(source_path::AbstractString; required::Bool = true)::Union{Nothing,String}
    if !isdir(source_path)
        required && throw(ArgumentError("tokenizer bundle source is not a directory: $(source_path)"))
        return nothing
    end

    tokenizer_json_path = joinpath(source_path, "tokenizer.json")
    manifest_path = joinpath(source_path, "keemena_training_manifest.json")
    if isfile(tokenizer_json_path) && isfile(manifest_path)
        return abspath(source_path)
    end

    required && throw(ArgumentError("tokenizer bundle is missing tokenizer.json or keemena_training_manifest.json: $(source_path)"))
    return nothing
end
