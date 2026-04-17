"""
Construct a lower-triangular causal attention mask.
"""
function causal_mask(sequence_length::Int)::Matrix{Bool}
    sequence_length >= 0 || error("sequence_length must be >= 0")
    return [column_index <= row_index for row_index in 1:sequence_length, column_index in 1:sequence_length]
end
