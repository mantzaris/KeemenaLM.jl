"""
Resolve a bundle source to a local directory path.
Stage 2 supports local directories only.
"""
function resolve_bundle(source)::String
    source_path = source isa AbstractString ? source : string(source)
    isdir(source_path) || throw(ArgumentError("bundle source is not a directory: $(source_path)"))

    bundle_manifest_path = joinpath(source_path, "bundle.json")
    isfile(bundle_manifest_path) ||
        throw(ArgumentError("bundle directory is missing bundle.json: $(source_path)"))

    return abspath(source_path)
end
