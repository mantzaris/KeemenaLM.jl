using Test

@testset "integration placeholder: Lux generate" begin
    @test_broken begin
        config = KeemenaLM.GPT2Config(vocab_size = 32, context_length = 8, num_layers = 2, num_heads = 4, embedding_size = 32, ffn_hidden_size = 64)
        model = KeemenaLM.instantiate(config; backend = :lux)

        try
            KeemenaLM.generate(model, nothing, nothing, "hello")
            true
        catch
            false
        end
    end
end
