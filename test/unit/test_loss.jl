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

    masked_loss = KeemenaLM.Core.causal_lm_cross_entropy(logits, targets, Float32[1; 0;;])
    @test isapprox(masked_loss, -log(0.8); atol = 1.0e-6)

    second_only_loss = KeemenaLM.Core.causal_lm_cross_entropy(logits, targets, Float32[0; 1;;])
    @test isapprox(second_only_loss, -log(0.9); atol = 1.0e-6)

    @test_throws ArgumentError KeemenaLM.Core.causal_lm_cross_entropy(logits, targets, zeros(Float32, 2, 1))
    @test_throws ArgumentError KeemenaLM.Core.causal_lm_cross_entropy(logits, targets, ones(Float32, 3, 1))
    @test_throws ArgumentError KeemenaLM.Core.causal_lm_cross_entropy(zeros(Float32, 2, 2), targets)
    @test_throws ArgumentError KeemenaLM.Core.causal_lm_cross_entropy(logits, zeros(Int, 3, 1))
end
