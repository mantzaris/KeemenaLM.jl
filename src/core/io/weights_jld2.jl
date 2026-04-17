"""
Persist backend-agnostic weight dictionaries in JLD2 format.
"""
function save_weights_jld2(path::AbstractString, weights::Dict{String, Any})
    mkpath(dirname(path))
    JLD2.jldsave(path; weights = weights)
    return path
end

"""
Load backend-agnostic weight dictionaries from JLD2 format.
"""
function load_weights_jld2(path::AbstractString)::Dict{String, Any}
    isfile(path) || throw(ArgumentError("weights file does not exist: $(path)"))
    loaded_data = JLD2.load(path)
    haskey(loaded_data, "weights") || throw(ArgumentError("weights file is missing the 'weights' entry: $(path)"))

    loaded_weights = loaded_data["weights"]
    loaded_weights isa AbstractDict || throw(ArgumentError("weights entry must be a dictionary"))

    return Dict{String, Any}(String(key) => value for (key, value) in pairs(loaded_weights))
end
