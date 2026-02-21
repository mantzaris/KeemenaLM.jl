using Test

@testset "causal_mask" begin
    mask = KeemenaLM.Core.causal_mask(4)

    @test size(mask) == (4, 4)
    @test mask[1, 1]
    @test !mask[1, 2]
    @test mask[2, 1]
    @test mask[2, 2]
    @test !mask[2, 3]
    @test mask[4, 1]
    @test mask[4, 4]

    @test size(KeemenaLM.Core.causal_mask(0)) == (0, 0)
    @test_throws ErrorException KeemenaLM.Core.causal_mask(-1)
end
