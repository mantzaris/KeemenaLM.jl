using Test

struct BundleTestTokenizer
    token_to_id::Dict{Char, Int}
    id_to_token::Dict{Int, Char}
end

function BundleTestTokenizer(alphabet::AbstractString)
    characters = collect(alphabet)
    token_to_id = Dict(character => index for (index, character) in enumerate(characters))
    id_to_token = Dict(index => character for (index, character) in enumerate(characters))
    return BundleTestTokenizer(token_to_id, id_to_token)
end

KeemenaLM.Core.tokenizer_encode(tokenizer::BundleTestTokenizer, text::AbstractString) =
    [get(tokenizer.token_to_id, character, 1) for character in text]

KeemenaLM.Core.tokenizer_decode(tokenizer::BundleTestTokenizer, token_ids::AbstractVector{<:Integer}) =
    String([get(tokenizer.id_to_token, token_id, '?') for token_id in token_ids])

@testset "Flux bundle roundtrip supports instantiate and generate" begin
    mktempdir() do temporary_directory
        tokenizer = BundleTestTokenizer(" abcdefgh")
        config = KeemenaLM.GPT2Config(
            vocab_size = length(tokenizer.token_to_id),
            context_length = 8,
            num_layers = 2,
            num_heads = 2,
            embedding_size = 8,
            ffn_hidden_size = 16,
        )
        model = KeemenaLM.instantiate(config; backend = :flux, seed = 21)
        bundle = KeemenaLM.Bundle(model_config = config, weights = KeemenaLM.Core.extract_weights(model))

        KeemenaLM.save_bundle(temporary_directory, bundle)
        loaded_bundle = KeemenaLM.load_bundle(temporary_directory)
        reloaded_model = KeemenaLM.instantiate(loaded_bundle; backend = :flux)

        generation_config = KeemenaLM.GenerationConfig(max_new_tokens = 4, temperature = 0.8, top_k = 4, seed = 777)
        original_output = KeemenaLM.generate(model, tokenizer, nothing, "ab"; generation_config = generation_config)
        reloaded_output = KeemenaLM.generate(reloaded_model, tokenizer, nothing, "ab"; generation_config = generation_config)

        @test loaded_bundle.manifest.parameter_schema == "gpt2.v1"
        @test sort(collect(keys(loaded_bundle.weights))) == sort(collect(keys(bundle.weights)))
        @test original_output == reloaded_output
        @test startswith(reloaded_output, "ab")
    end
end
