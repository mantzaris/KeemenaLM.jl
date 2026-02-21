using Test

@testset "integration placeholder: Flux forward" begin
    @test_broken begin
        config = KeemenaLM.GPT2Config(vocab_size = 32, context_length = 8, num_layers = 2, num_heads = 4, embedding_size = 32, ffn_hidden_size = 64)
        model = KeemenaLM.instantiate(config; backend = :flux)
        input_token_ids = fill(1, 4, 1)

        try
            KeemenaLM.Core.lm_forward(model, input_token_ids)
            true
        catch
            false
        end
    end
end
