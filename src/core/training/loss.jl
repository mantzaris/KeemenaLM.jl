"""
Compute causal language-model cross entropy loss.
"""
function causal_lm_cross_entropy(logits, targets)
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
    flat_targets = reshape(targets, 1, :)
    device_targets = similar(flat_logits, Int, size(flat_targets)...)
    device_targets .= flat_targets
    max_logits = maximum(flat_logits; dims = 1)
    log_normalizers = max_logits .+ log.(sum(exp.(flat_logits .- max_logits); dims = 1))
    log_probs = flat_logits .- log_normalizers

    class_ids = similar(flat_logits, Int, vocab_size, 1)
    class_ids .= reshape(collect(1:vocab_size), :, 1)
    # Use broadcasted equality rather than scalar indexing so the same path works on CPU and GPU arrays.
    target_mask = class_ids .== device_targets
    selected_log_probs = sum(log_probs .* target_mask; dims = 1)
    return -sum(selected_log_probs) / token_count
end
