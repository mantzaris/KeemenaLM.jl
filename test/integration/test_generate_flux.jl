using Test

struct MockGenerateModel <: KeemenaLM.Core.AbstractCausalLM
    config::KeemenaLM.GPT2Config
    next_token_id::Int
end

KeemenaLM.Core.model_config(model::MockGenerateModel) = model.config

function KeemenaLM.Core.lm_forward(
    model::MockGenerateModel,
    input_token_ids::AbstractMatrix{<:Integer};
    cache = nothing,
    is_training::Bool = false,
)
    sequence_length, batch_size = size(input_token_ids)
    logits = fill(-1.0f6, model.config.vocab_size, sequence_length, batch_size)
    logits[model.next_token_id, end, 1] = 0.0f0
    return logits, nothing
end

struct TestCharTokenizer
    token_to_id::Dict{Char, Int}
    id_to_token::Dict{Int, Char}
end

function TestCharTokenizer(alphabet::AbstractString)
    characters = collect(alphabet)
    token_to_id = Dict(character => index for (index, character) in enumerate(characters))
    id_to_token = Dict(index => character for (index, character) in enumerate(characters))
    return TestCharTokenizer(token_to_id, id_to_token)
end

KeemenaLM.Core.tokenizer_encode(tokenizer::TestCharTokenizer, text::AbstractString) =
    [get(tokenizer.token_to_id, character, 1) for character in text]

KeemenaLM.Core.tokenizer_decode(tokenizer::TestCharTokenizer, token_ids::AbstractVector{<:Integer}) =
    String([get(tokenizer.id_to_token, token_id, '?') for token_id in token_ids])

@testset "Flux generate on CPU is deterministic with a fixed seed" begin
    tokenizer = TestCharTokenizer(" abcdefgh")
    config = KeemenaLM.GPT2Config(
        vocab_size = length(tokenizer.token_to_id),
        context_length = 8,
        num_layers = 2,
        num_heads = 2,
        embedding_size = 8,
        ffn_hidden_size = 16,
    )
    model = KeemenaLM.instantiate(config; backend = :flux, seed = 9)
    generation_config = KeemenaLM.GenerationConfig(max_new_tokens = 4, temperature = 0.8, top_k = 4, seed = 123)

    first_output = KeemenaLM.generate(model, tokenizer, nothing, "ab"; generation_config = generation_config)
    second_output = KeemenaLM.generate(model, tokenizer, nothing, "ab"; generation_config = generation_config)
    changed_seed_output = KeemenaLM.generate(
        model,
        tokenizer,
        nothing,
        "ab";
        generation_config = KeemenaLM.GenerationConfig(max_new_tokens = 4, temperature = 0.8, top_k = 4, seed = 124),
    )

    @test first_output == second_output
    @test startswith(first_output, "ab")
    @test length(first_output) == 6
    @test first_output != changed_seed_output
end

@testset "Flux generate respects stop sequences" begin
    tokenizer = TestCharTokenizer(" abcdefgh")
    config = KeemenaLM.GPT2Config(
        vocab_size = length(tokenizer.token_to_id),
        context_length = 8,
        num_layers = 1,
        num_heads = 2,
        embedding_size = 8,
        ffn_hidden_size = 16,
    )
    model = KeemenaLM.instantiate(config; backend = :flux, seed = 3)
    generation_config = KeemenaLM.GenerationConfig(max_new_tokens = 6, temperature = 0.0, stop_sequences = [" "])

    output = KeemenaLM.generate(model, tokenizer, nothing, "ab"; generation_config = generation_config)

    @test !occursin(" ", output[3:end])
    @test startswith(output, "ab")
    @test length(output) >= 2
end

@testset "generate omits EOS token text from returned output" begin
    tokenizer = TestCharTokenizer(" ab")
    eos_token_id = tokenizer.token_to_id['b']
    model = MockGenerateModel(
        KeemenaLM.GPT2Config(
            vocab_size = length(tokenizer.token_to_id),
            context_length = 8,
            num_layers = 1,
            num_heads = 1,
            embedding_size = 4,
            ffn_hidden_size = 8,
            eos_token_id = eos_token_id,
        ),
        eos_token_id,
    )

    output = KeemenaLM.generate(
        model,
        tokenizer,
        nothing,
        "a";
        generation_config = KeemenaLM.GenerationConfig(max_new_tokens = 3, temperature = 0.0, eos_token_id = eos_token_id),
    )

    @test output == "a"
end
