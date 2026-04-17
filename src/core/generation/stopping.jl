"""
Evaluate whether generation should stop based on token and text criteria.
"""
function should_stop_generation(
    generated_token_ids::AbstractVector{<:Integer},
    generated_text::AbstractString,
    generation_config::GenerationConfig,
)::Bool
    if generation_config.eos_token_id !== nothing &&
       !isempty(generated_token_ids) &&
       generated_token_ids[end] == generation_config.eos_token_id
        return true
    end

    for stop_sequence in generation_config.stop_sequences
        isempty(stop_sequence) && continue
        occursin(stop_sequence, generated_text) && return true
    end

    return false
end

function trim_stop_sequences(generated_text::AbstractString, generation_config::GenerationConfig)::String
    first_stop_index = nothing

    for stop_sequence in generation_config.stop_sequences
        isempty(stop_sequence) && continue
        stop_range = findfirst(stop_sequence, generated_text)
        stop_range === nothing && continue

        stop_start = first(stop_range)
        if first_stop_index === nothing || stop_start < first_stop_index
            first_stop_index = stop_start
        end
    end

    first_stop_index === nothing && return String(generated_text)
    first_stop_index == firstindex(generated_text) && return ""
    return String(generated_text[firstindex(generated_text):prevind(generated_text, first_stop_index)])
end
