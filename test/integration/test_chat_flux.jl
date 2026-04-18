using Test

struct ChatTestTokenizer
    token_to_id::Dict{Char, Int}
    id_to_token::Dict{Int, Char}
end

function ChatTestTokenizer(alphabet::AbstractString)
    characters = collect(alphabet)
    token_to_id = Dict(character => index for (index, character) in enumerate(characters))
    id_to_token = Dict(index => character for (index, character) in enumerate(characters))
    return ChatTestTokenizer(token_to_id, id_to_token)
end

KeemenaLM.Core.tokenizer_encode(tokenizer::ChatTestTokenizer, text::AbstractString) =
    [get(tokenizer.token_to_id, character, 1) for character in text]

KeemenaLM.Core.tokenizer_decode(tokenizer::ChatTestTokenizer, token_ids::AbstractVector{<:Integer}) =
    String([get(tokenizer.id_to_token, token_id, '?') for token_id in token_ids])

struct HistoryAwareChatModel <: KeemenaLM.Core.AbstractCausalLM
    config::KeemenaLM.GPT2Config
    first_token_id::Int
    later_token_id::Int
    trigger_token_id::Int
end

KeemenaLM.Core.model_config(model::HistoryAwareChatModel) = model.config

function KeemenaLM.Core.lm_forward(
    model::HistoryAwareChatModel,
    input_token_ids::AbstractMatrix{<:Integer};
    cache = nothing,
    is_training::Bool = false,
)
    sequence_length, batch_size = size(input_token_ids)
    logits = fill(-1.0f6, model.config.vocab_size, sequence_length, batch_size)
    next_token_id = any(input_token_ids .== model.trigger_token_id) ? model.later_token_id : model.first_token_id
    logits[next_token_id, end, 1] = 0.0f0
    return logits, nothing
end

@testset "Flux bundle load -> chat session -> one chat turn" begin
    tokenizer = ChatTestTokenizer(" abcdefghijklmnopqrstuvwxyz.,!?")
    config = KeemenaLM.GPT2Config(
        vocab_size = length(tokenizer.token_to_id),
        context_length = 64,
        num_layers = 2,
        num_heads = 2,
        embedding_size = 8,
        ffn_hidden_size = 16,
    )
    model = KeemenaLM.instantiate(config; backend = :flux, seed = 41)
    bundle = KeemenaLM.Bundle(model_config = config, weights = KeemenaLM.Core.extract_weights(model))

    mktempdir() do temporary_directory
        KeemenaLM.save_bundle(temporary_directory, bundle)

        loaded_bundle = KeemenaLM.load_bundle(temporary_directory)
        reloaded_model = KeemenaLM.instantiate(loaded_bundle; backend = :flux)
        session = KeemenaLM.ChatSession(
            reloaded_model,
            tokenizer,
            nothing;
            system_prompt = "You are a tiny demo assistant.",
            generation_config = KeemenaLM.GenerationConfig(max_new_tokens = 4, temperature = 0.0, seed = 7),
        )

        reply = KeemenaLM.chat!(session, "hello")

        @test reply isa String
        @test session.message_history == [
            (role = "user", content = "hello"),
            (role = "assistant", content = reply),
        ]
    end
end

@testset "multi-turn chat preserves history and prompt assembly" begin
    tokenizer = ChatTestTokenizer(" xyhi?:")
    config = KeemenaLM.GPT2Config(
        vocab_size = length(tokenizer.token_to_id),
        context_length = 64,
        num_layers = 1,
        num_heads = 1,
        embedding_size = 4,
        ffn_hidden_size = 8,
    )
    model = HistoryAwareChatModel(
        config,
        tokenizer.token_to_id['x'],
        tokenizer.token_to_id['y'],
        tokenizer.token_to_id['x'],
    )
    session = KeemenaLM.ChatSession(
        model,
        tokenizer,
        nothing;
        system_prompt = "Stay terse.",
        generation_config = KeemenaLM.GenerationConfig(max_new_tokens = 1, temperature = 0.0),
    )

    first_reply = KeemenaLM.chat!(session, "hi")
    second_prompt = KeemenaLM.Core.render_chat_prompt(session; pending_user_text = "hi?")
    second_reply = KeemenaLM.chat!(session, "hi?")

    @test first_reply == "x"
    @test second_reply == "y"
    @test occursin("System: Stay terse.", second_prompt)
    @test occursin("User: hi", second_prompt)
    @test occursin("Assistant: x", second_prompt)
    @test session.message_history == [
        (role = "user", content = "hi"),
        (role = "assistant", content = "x"),
        (role = "user", content = "hi?"),
        (role = "assistant", content = "y"),
    ]
end

@testset "chat_repl is a thin loop around chat!" begin
    tokenizer = ChatTestTokenizer(" xyhi")
    config = KeemenaLM.GPT2Config(
        vocab_size = length(tokenizer.token_to_id),
        context_length = 64,
        num_layers = 1,
        num_heads = 1,
        embedding_size = 4,
        ffn_hidden_size = 8,
    )
    model = HistoryAwareChatModel(
        config,
        tokenizer.token_to_id['x'],
        tokenizer.token_to_id['y'],
        tokenizer.token_to_id['x'],
    )
    session = KeemenaLM.ChatSession(
        model,
        tokenizer,
        nothing;
        generation_config = KeemenaLM.GenerationConfig(max_new_tokens = 1, temperature = 0.0),
    )

    input = IOBuffer("hi\n/exit\n")
    output = IOBuffer()
    returned_session = KeemenaLM.chat_repl(
        session;
        input = input,
        output = output,
        user_prompt = "you> ",
        assistant_prompt = "bot> ",
    )

    output_text = String(take!(output))
    @test returned_session === session
    @test occursin("you> ", output_text)
    @test occursin("bot> x", output_text)
    @test session.message_history == [
        (role = "user", content = "hi"),
        (role = "assistant", content = "x"),
    ]
end
