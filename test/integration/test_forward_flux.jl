using Test

@testset "Flux forward pass on CPU" begin
    config = KeemenaLM.GPT2Config(
        vocab_size = 17,
        context_length = 8,
        num_layers = 2,
        num_heads = 2,
        embedding_size = 8,
        ffn_hidden_size = 16,
        eos_token_id = 2,
    )
    model = KeemenaLM.instantiate(config; backend = :flux, seed = 7)
    input_token_ids = reshape(Int[1, 2, 3, 4, 5, 6, 7, 8], 4, 2)

    logits, cache = KeemenaLM.Core.lm_forward(model, input_token_ids)
    loss = KeemenaLM.Core.causal_lm_cross_entropy(logits, input_token_ids)

    @test size(logits) == (config.vocab_size, size(input_token_ids, 1), size(input_token_ids, 2))
    @test cache === nothing
    @test all(isfinite, logits)
    @test isfinite(loss)
    @test loss > 0
end
