"""
Save a model bundle to a directory.
"""
function save_bundle(directory_path::AbstractString, bundle::Bundle)
    manifest = validate_bundle_manifest(bundle.manifest)
    model_config = bundle.model_config
    model_config isa GPT2Config || throw(ArgumentError("Stage 2 save_bundle only supports GPT2Config"))

    mkpath(directory_path)

    bundle_manifest_path = joinpath(directory_path, "bundle.json")
    model_config_path = bundle_local_path(directory_path, manifest.model_config_file, "model_config_file")
    weights_path = bundle_local_path(directory_path, manifest.weights_file, "weights_file")

    mkpath(dirname(model_config_path))
    mkpath(dirname(weights_path))

    open(bundle_manifest_path, "w") do io
        JSON3.write(io, manifest)
    end
    open(model_config_path, "w") do io
        JSON3.write(io, model_config)
    end
    save_weights_jld2(weights_path, bundle.weights)

    return directory_path
end
