"""
Compute causal language-model cross entropy loss.
"""
function causal_lm_cross_entropy(logits, targets)
    loss, _, _ = causal_lm_cross_entropy_with_cache(logits, targets)
    return loss
end

function causal_lm_cross_entropy_with_cache(logits, targets)
    ndims(logits) == 3 || throw(ArgumentError("logits must have shape (vocab_size, sequence_length, batch_size)"))
    ndims(targets) == 2 || throw(ArgumentError("targets must have shape (sequence_length, batch_size)"))

    vocab_size, sequence_length, batch_size = size(logits)
    size(targets) == (sequence_length, batch_size) ||
        throw(ArgumentError("targets shape must match the logits sequence and batch dimensions"))
    sequence_length > 0 || throw(ArgumentError("sequence_length must be > 0"))

    token_count = sequence_length * batch_size
    target_min = minimum(targets)
    target_max = maximum(targets)
    1 <= target_min <= target_max <= vocab_size ||
        throw(ArgumentError("target token ids must stay within 1:vocab_size"))

    flat_logits = reshape(logits, vocab_size, :)
    flat_targets = reshape(targets_like_logits(logits, targets), 1, :)
    max_logits = maximum(flat_logits; dims = 1)
    shifted_exp = exp.(flat_logits .- max_logits)
    probability_sums = sum(shifted_exp; dims = 1)
    probabilities = shifted_exp ./ probability_sums
    log_normalizers = max_logits .+ log.(probability_sums)
    log_probs = flat_logits .- log_normalizers

    class_ids = class_ids_like_logits(logits, vocab_size)
    target_mask = reshape(class_ids, :, 1) .== flat_targets
    selected_log_probs = sum(log_probs .* target_mask; dims = 1)
    loss = -sum(selected_log_probs) / token_count
    return loss, probabilities, target_mask
end

function targets_like_logits(logits, targets)
    device_targets = similar(logits, Int, size(targets))
    device_targets .= targets
    return device_targets
end

function class_ids_like_logits(logits, vocab_size::Int)
    class_ids = similar(logits, Int, vocab_size)
    class_ids .= 1:vocab_size
    return class_ids
end

function ChainRulesCore.rrule(::typeof(causal_lm_cross_entropy), logits, targets)
    loss, probabilities, target_mask = causal_lm_cross_entropy_with_cache(logits, targets)
    token_count = size(logits, 2) * size(logits, 3)

    function causal_lm_cross_entropy_pullback(loss_sensitivity)
        flat_gradient = (probabilities .- target_mask) .* (loss_sensitivity / token_count)
        return (
            ChainRulesCore.NoTangent(),
            reshape(flat_gradient, size(logits)),
            ChainRulesCore.NoTangent(),
        )
    end

    return loss, causal_lm_cross_entropy_pullback
end
