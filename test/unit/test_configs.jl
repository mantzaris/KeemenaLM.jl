using Test

@testset "GPT2Config validation" begin
    config = KeemenaLM.GPT2Config(
        vocab_size = 32,
        context_length = 16,
        num_layers = 2,
        num_heads = 4,
        embedding_size = 32,
        ffn_hidden_size = 64,
        eos_token_id = 2,
    )

    @test KeemenaLM.validate(config) === config

    invalid_config = KeemenaLM.GPT2Config(
        vocab_size = 32,
        context_length = 16,
        num_layers = 2,
        num_heads = 3,
        embedding_size = 32,
        ffn_hidden_size = 64,
    )

    @test_throws ArgumentError KeemenaLM.validate(invalid_config)
end
