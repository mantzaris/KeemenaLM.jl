using Test

struct LuxTestTokenizer
    token_to_id::Dict{Char, Int}
    id_to_token::Dict{Int, Char}
end

function LuxTestTokenizer(alphabet::AbstractString)
    characters = collect(alphabet)
    token_to_id = Dict(character => index for (index, character) in enumerate(characters))
    id_to_token = Dict(index => character for (index, character) in enumerate(characters))
    return LuxTestTokenizer(token_to_id, id_to_token)
end

KeemenaLM.Core.tokenizer_encode(tokenizer::LuxTestTokenizer, text::AbstractString) =
    [get(tokenizer.token_to_id, character, 1) for character in text]

KeemenaLM.Core.tokenizer_decode(tokenizer::LuxTestTokenizer, token_ids::AbstractVector{<:Integer}) =
    String([get(tokenizer.id_to_token, token_id, '?') for token_id in token_ids])

@testset "Lux bundle -> instantiate -> generate on CPU" begin
    tokenizer = LuxTestTokenizer(" abcdefgh")
    config = KeemenaLM.GPT2Config(
        vocab_size = length(tokenizer.token_to_id),
        context_length = 8,
        num_layers = 2,
        num_heads = 2,
        embedding_size = 8,
        ffn_hidden_size = 16,
    )
    flux_model = KeemenaLM.instantiate(config; backend = :flux, seed = 29)
    bundle = KeemenaLM.Bundle(model_config = config, weights = KeemenaLM.Core.extract_weights(flux_model))
    lux_model = KeemenaLM.instantiate(bundle; backend = :lux)

    generation_config = KeemenaLM.GenerationConfig(max_new_tokens = 4, temperature = 0.0, seed = 21)
    flux_output = KeemenaLM.generate(flux_model, tokenizer, nothing, "ab"; generation_config = generation_config)
    lux_output = KeemenaLM.generate(lux_model, tokenizer, nothing, "ab"; generation_config = generation_config)

    @test lux_model isa KeemenaLM.LuxBackend.LuxGPT2Model
    @test flux_output == lux_output
    @test startswith(lux_output, "ab")
    @test length(lux_output) == 6
end
