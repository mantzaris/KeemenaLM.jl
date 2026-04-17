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

    total_loss = 0.0
    token_count = sequence_length * batch_size

    for batch_index in 1:batch_size
        for sequence_index in 1:sequence_length
            target_token_id = targets[sequence_index, batch_index]
            1 <= target_token_id <= vocab_size ||
                throw(ArgumentError("target token id $(target_token_id) is outside 1:vocab_size"))

            token_logits = @view logits[:, sequence_index, batch_index]
            max_logit = maximum(token_logits)
            log_normalizer = max_logit + log(sum(exp.(token_logits .- max_logit)))
            total_loss += log_normalizer - token_logits[target_token_id]
        end
    end

    return total_loss / token_count
end
