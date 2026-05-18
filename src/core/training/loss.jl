"""
Compute causal language-model cross entropy loss.
"""
function causal_lm_cross_entropy(logits, targets)
    loss, _, _, _ = causal_lm_cross_entropy_with_cache(logits, targets, nothing)
    return loss
end

function causal_lm_cross_entropy(logits, targets, loss_mask)
    loss, _, _, _ = causal_lm_cross_entropy_with_cache(logits, targets, loss_mask)
    return loss
end

function causal_lm_cross_entropy_with_cache(logits, targets, loss_mask)
    ndims(logits) == 3 || throw(ArgumentError("logits must have shape (vocab_size, sequence_length, batch_size)"))
    ndims(targets) == 2 || throw(ArgumentError("targets must have shape (sequence_length, batch_size)"))

    vocab_size, sequence_length, batch_size = size(logits)
    size(targets) == (sequence_length, batch_size) ||
        throw(ArgumentError("targets shape must match the logits sequence and batch dimensions"))
    if loss_mask !== nothing
        ndims(loss_mask) == 2 || throw(ArgumentError("loss_mask must have shape (sequence_length, batch_size)"))
        size(loss_mask) == (sequence_length, batch_size) ||
            throw(ArgumentError("loss_mask shape must match the logits sequence and batch dimensions"))
    end
    sequence_length > 0 || throw(ArgumentError("sequence_length must be > 0"))

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
    flat_loss_weights = reshape(loss_weights_like_logits(logits, loss_mask, sequence_length, batch_size), 1, :)
    weight_sum = sum(flat_loss_weights)
    weight_sum > 0 || throw(ArgumentError("loss_mask must include at least one target token"))
    loss = -sum(selected_log_probs .* flat_loss_weights) / weight_sum
    return loss, probabilities, target_mask, flat_loss_weights
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

function loss_weights_like_logits(logits, loss_mask, sequence_length::Int, batch_size::Int)
    loss_weights = similar(logits, eltype(logits), sequence_length, batch_size)
    if loss_mask === nothing
        loss_weights .= one(eltype(logits))
        return loss_weights
    end

    loss_weights .= loss_mask
    return loss_weights
end

function ChainRulesCore.rrule(::typeof(causal_lm_cross_entropy), logits, targets)
    loss, probabilities, target_mask, flat_loss_weights = causal_lm_cross_entropy_with_cache(logits, targets, nothing)
    weight_sum = sum(flat_loss_weights)

    function causal_lm_cross_entropy_pullback(loss_sensitivity)
        flat_gradient = (probabilities .- target_mask) .* flat_loss_weights .* (loss_sensitivity / weight_sum)
        return (
            ChainRulesCore.NoTangent(),
            reshape(flat_gradient, size(logits)),
            ChainRulesCore.NoTangent(),
        )
    end

    return loss, causal_lm_cross_entropy_pullback
end

function ChainRulesCore.rrule(::typeof(causal_lm_cross_entropy), logits, targets, loss_mask)
    loss, probabilities, target_mask, flat_loss_weights = causal_lm_cross_entropy_with_cache(logits, targets, loss_mask)
    weight_sum = sum(flat_loss_weights)

    function causal_lm_cross_entropy_pullback(loss_sensitivity)
        flat_gradient = (probabilities .- target_mask) .* flat_loss_weights .* (loss_sensitivity / weight_sum)
        return (
            ChainRulesCore.NoTangent(),
            reshape(flat_gradient, size(logits)),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
        )
    end

    return loss, causal_lm_cross_entropy_pullback
end
