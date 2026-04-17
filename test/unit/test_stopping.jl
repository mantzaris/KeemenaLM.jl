using Test

@testset "should_stop_generation" begin
    eos_config = KeemenaLM.Core.GenerationConfig(eos_token_id = 3)
    @test KeemenaLM.Core.should_stop_generation([1, 2, 3], "abc", eos_config)
    @test !KeemenaLM.Core.should_stop_generation([1, 2], "ab", eos_config)

    stop_sequence_config = KeemenaLM.Core.GenerationConfig(stop_sequences = ["<END>", "###"])
    @test KeemenaLM.Core.should_stop_generation([1, 2], "hello<END>", stop_sequence_config)
    @test !KeemenaLM.Core.should_stop_generation([1, 2], "hello", stop_sequence_config)

    trimmed = KeemenaLM.Core.trim_stop_sequences("reply<END>ignored", stop_sequence_config)
    @test trimmed == "reply"
end
