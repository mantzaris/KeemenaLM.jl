using Test

@testset "sample_next_token" begin
    logits = Float32[0.1, 0.2, 3.0, 2.5]

    greedy_config = KeemenaLM.Core.GenerationConfig(temperature = 0.0)
    @test KeemenaLM.Core.sample_next_token(logits, greedy_config) == 3

    top_k_config = KeemenaLM.Core.GenerationConfig(temperature = 1.0, top_k = 1, seed = 11)
    @test KeemenaLM.Core.sample_next_token(logits, top_k_config; rng = KeemenaLM.Core.Stage1RNG(11)) == 3

    nucleus_logits = Float32[4.0, 1.0, -2.0]
    top_p_config = KeemenaLM.Core.GenerationConfig(temperature = 1.0, top_p = 0.8)
    @test KeemenaLM.Core.sample_next_token(nucleus_logits, top_p_config; rng = KeemenaLM.Core.Stage1RNG(22)) == 1

    seeded_config = KeemenaLM.Core.GenerationConfig(temperature = 1.0, top_k = 3)
    first_draw = KeemenaLM.Core.sample_next_token(logits, seeded_config; rng = KeemenaLM.Core.Stage1RNG(1234))
    second_draw = KeemenaLM.Core.sample_next_token(logits, seeded_config; rng = KeemenaLM.Core.Stage1RNG(1234))
    @test first_draw == second_draw

    @test_throws ArgumentError KeemenaLM.Core.sample_next_token(Float32[], seeded_config)
    @test_throws ArgumentError KeemenaLM.Core.sample_next_token(logits, KeemenaLM.Core.GenerationConfig(top_k = -1))
    @test_throws ArgumentError KeemenaLM.Core.sample_next_token(logits, KeemenaLM.Core.GenerationConfig(top_p = 0.0))
end
