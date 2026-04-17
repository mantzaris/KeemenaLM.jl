"""
Sample a token index from logits according to generation settings.
"""
mutable struct Stage1RNG
    state::UInt64
end

function Stage1RNG(seed::Integer)
    state = reinterpret(UInt64, Int64(seed))
    state == 0 && (state = 0x9e37_79b9_7f4a_7c15)
    return Stage1RNG(state)
end
Stage1RNG() = Stage1RNG(Int(time_ns()))

function advance!(rng::Stage1RNG)::UInt64
    state = rng.state
    state ⊻= state << 13
    state ⊻= state >> 7
    state ⊻= state << 17
    rng.state = state
    return state
end

function rand_uniform(rng::Stage1RNG)::Float64
    return Float64(advance!(rng) & 0x1f_ffff_ffff_ffff) / Float64(0x20_0000_0000_0000)
end

function sample_next_token(
    logits::AbstractVector{<:Real},
    generation_config::GenerationConfig;
    rng::Stage1RNG = Stage1RNG(),
)::Int
    isempty(logits) && throw(ArgumentError("logits must be non-empty"))
    generation_config.top_k >= 0 || throw(ArgumentError("top_k must be >= 0"))
    0.0 < generation_config.top_p <= 1.0 || throw(ArgumentError("top_p must be in (0.0, 1.0]"))
    generation_config.temperature >= 0.0 || throw(ArgumentError("temperature must be >= 0.0"))

    if generation_config.temperature == 0.0
        return findmax(logits)[2]
    end

    filtered_logits = Float64.(collect(logits)) ./ generation_config.temperature
    apply_top_k!(filtered_logits, generation_config.top_k)
    apply_top_p!(filtered_logits, generation_config.top_p)

    max_logit = maximum(filtered_logits)
    isfinite(max_logit) || throw(ArgumentError("sampling filter removed every candidate token"))

    probabilities = exp.(filtered_logits .- max_logit)
    probability_sum = sum(probabilities)
    isfinite(probability_sum) && probability_sum > 0.0 || throw(ArgumentError("invalid sampling probabilities"))
    probabilities ./= probability_sum

    sample_value = rand_uniform(rng)
    cumulative_probability = 0.0
    for token_index in eachindex(probabilities)
        cumulative_probability += probabilities[token_index]
        if sample_value <= cumulative_probability
            return token_index
        end
    end

    return lastindex(probabilities)
end

function apply_top_k!(logits::Vector{Float64}, top_k::Int)
    top_k == 0 && return logits
    keep_count = min(top_k, length(logits))
    keep_count == length(logits) && return logits

    sorted_indices = sortperm(logits; rev = true)
    for token_index in sorted_indices[(keep_count + 1):end]
        logits[token_index] = -Inf
    end
    return logits
end

function apply_top_p!(logits::Vector{Float64}, top_p::Float64)
    top_p >= 1.0 && return logits

    sorted_indices = sortperm(logits; rev = true)
    sorted_logits = logits[sorted_indices]
    max_logit = maximum(sorted_logits)
    probabilities = exp.(sorted_logits .- max_logit)
    probabilities ./= sum(probabilities)

    cumulative_probability = 0.0
    keep_count = 0
    for probability in probabilities
        keep_count += 1
        cumulative_probability += probability
        if cumulative_probability >= top_p
            break
        end
    end

    for token_index in sorted_indices[(keep_count + 1):end]
        logits[token_index] = -Inf
    end
    return logits
end
