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

@testset "chat stop defaults include explicit chat markers" begin
    config = KeemenaLM.Core.chat_generation_config(KeemenaLM.GenerationConfig(stop_sequences = ["<CUSTOM>"]))

    @test "<CUSTOM>" in config.stop_sequences
    @test "<END_ASSISTANT>" in config.stop_sequences
    @test "<CHAT_END>" in config.stop_sequences
    @test "\nUser:" in config.stop_sequences
    @test "\nAssistant:" in config.stop_sequences
end
