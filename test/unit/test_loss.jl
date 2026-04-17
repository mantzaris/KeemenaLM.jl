using Test

@testset "causal_lm_cross_entropy" begin
    logits = reshape(
        Float32[
            log(0.8), log(0.2),
            log(0.1), log(0.9),
        ],
        2,
        2,
        1,
    )
    targets = reshape(Int[1, 2], 2, 1)

    loss = KeemenaLM.Core.causal_lm_cross_entropy(logits, targets)
    expected = (-log(0.8) - log(0.9)) / 2

    @test isapprox(loss, expected; atol = 1.0e-6)
    @test_throws ArgumentError KeemenaLM.Core.causal_lm_cross_entropy(zeros(Float32, 2, 2), targets)
    @test_throws ArgumentError KeemenaLM.Core.causal_lm_cross_entropy(logits, zeros(Int, 3, 1))
end
