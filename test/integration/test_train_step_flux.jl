using Test

@testset "integration placeholder: Flux train_step!" begin
    @test_broken begin
        config = KeemenaLM.GPT2Config(vocab_size = 32, context_length = 8, num_layers = 2, num_heads = 4, embedding_size = 32, ffn_hidden_size = 64)
        model = KeemenaLM.instantiate(config; backend = :flux)
        trainer = KeemenaLM.Core.Trainer(model)

        input_token_ids = fill(1, 4, 1)
        target_token_ids = fill(2, 4, 1)

        try
            KeemenaLM.Core.train_step!(trainer, input_token_ids, target_token_ids)
            true
        catch
            false
        end
    end
end
