"""
Construct a lower-triangular causal attention mask.
"""
function causal_mask(sequence_length::Int)::BitMatrix
    sequence_length >= 0 || error("sequence_length must be >= 0")
    mask = falses(sequence_length, sequence_length)
    for row_index in 1:sequence_length
        @inbounds mask[row_index, 1:row_index] .= true
    end
    return mask
end
