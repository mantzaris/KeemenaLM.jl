"""
Persist backend-agnostic weight dictionaries in JLD2 format.
"""
function save_weights_jld2(path::AbstractString, weights::Dict{String, Any})
    error("TODO v0.1: save_weights_jld2 not implemented")
end

"""
Load backend-agnostic weight dictionaries from JLD2 format.
"""
function load_weights_jld2(path::AbstractString)
    error("TODO v0.1: load_weights_jld2 not implemented")
end
