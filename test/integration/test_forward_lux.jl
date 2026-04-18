using Test

@testset "Lux forward pass on CPU" begin
    config = KeemenaLM.GPT2Config(
        vocab_size = 32,
        context_length = 8,
        num_layers = 2,
        num_heads = 4,
        embedding_size = 32,
        ffn_hidden_size = 64,
    )
    model = KeemenaLM.instantiate(config; backend = :lux, seed = 13)
    input_token_ids = reshape([1, 2, 3, 4, 5, 6, 7, 8], 4, 2)

    logits, cache = KeemenaLM.Core.lm_forward(model, input_token_ids)

    @test model isa KeemenaLM.LuxBackend.LuxGPT2Model
    @test size(logits) == (config.vocab_size, size(input_token_ids, 1), size(input_token_ids, 2))
    @test cache === nothing
    @test all(isfinite, logits)
end

@testset "Lux shared weight schema roundtrip across backends" begin
    config = KeemenaLM.GPT2Config(
        vocab_size = 24,
        context_length = 8,
        num_layers = 2,
        num_heads = 2,
        embedding_size = 8,
        ffn_hidden_size = 16,
    )
    lux_model = KeemenaLM.instantiate(config; backend = :lux, seed = 17)
    lux_weights = KeemenaLM.Core.extract_weights(lux_model)
    bundle = KeemenaLM.Bundle(model_config = config, weights = lux_weights)

    mktempdir() do temporary_directory
        KeemenaLM.save_bundle(temporary_directory, bundle)
        loaded_bundle = KeemenaLM.load_bundle(temporary_directory)
        reloaded_lux_model = KeemenaLM.instantiate(loaded_bundle; backend = :lux)
        flux_model = KeemenaLM.instantiate(loaded_bundle; backend = :flux)
        input_token_ids = reshape([1, 2, 3, 4], 4, 1)

        lux_logits, _ = KeemenaLM.Core.lm_forward(reloaded_lux_model, input_token_ids)
        flux_logits, _ = KeemenaLM.Core.lm_forward(flux_model, input_token_ids)

        @test loaded_bundle.manifest.parameter_schema == "gpt2.v1"
        @test sort!(collect(keys(lux_weights))) == sort!(collect(keys(loaded_bundle.weights)))
        @test size(lux_logits) == size(flux_logits)
        @test lux_logits ≈ flux_logits atol = 1.0f-5
    end
end
