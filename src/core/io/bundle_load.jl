"""
Load a model bundle from a directory.
"""
function load_bundle(source; artifacts_toml::AbstractString = default_artifacts_toml())::Bundle
    directory_path = resolve_bundle(source; artifacts_toml = artifacts_toml)
    bundle_manifest_path = joinpath(directory_path, "bundle.json")

    manifest = open(bundle_manifest_path, "r") do io
        JSON3.read(read(io, String), BundleManifest)
    end
    validate_bundle_manifest(manifest)

    model_config_path = bundle_local_path(directory_path, manifest.model_config_file, "model_config_file")
    weights_path = bundle_local_path(directory_path, manifest.weights_file, "weights_file")
    isfile(model_config_path) || throw(ArgumentError("bundle is missing model config file: $(model_config_path)"))

    model_config = open(model_config_path, "r") do io
        JSON3.read(read(io, String), GPT2Config)
    end
    validate(model_config)
    weights = load_weights_jld2(weights_path)

    return Bundle(; manifest = manifest, model_config = model_config, weights = weights)
end
