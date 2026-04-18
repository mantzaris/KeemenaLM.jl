#!/usr/bin/env julia
using KeemenaLM

struct DemoCharTokenizer
    token_to_id::Dict{Char, Int}
    id_to_token::Dict{Int, Char}
end

function DemoCharTokenizer(alphabet::AbstractString)
    characters = collect(alphabet)
    token_to_id = Dict(character => index for (index, character) in enumerate(characters))
    id_to_token = Dict(index => character for (index, character) in enumerate(characters))
    return DemoCharTokenizer(token_to_id, id_to_token)
end

KeemenaLM.Core.tokenizer_encode(tokenizer::DemoCharTokenizer, text::AbstractString) =
    [get(tokenizer.token_to_id, character, 1) for character in text]

KeemenaLM.Core.tokenizer_decode(tokenizer::DemoCharTokenizer, token_ids::AbstractVector{<:Integer}) =
    String([get(tokenizer.id_to_token, token_id, '?') for token_id in token_ids])

length(ARGS) == 1 || error("usage: julia --project=. examples/chat_repl.jl <bundle_dir>")

bundle = load_bundle(ARGS[1])
model = instantiate(bundle; backend = :flux)
tokenizer = DemoCharTokenizer(" abcdefghijklmnopqrstuvwxyz.,!?")

session = ChatSession(
    model,
    tokenizer,
    nothing;
    system_prompt = "You are a tiny demo assistant.",
    generation_config = GenerationConfig(max_new_tokens = 48, temperature = 0.0),
)

println("Loaded bundle from $(ARGS[1])")
println("Tokenizer and preprocessing are still supplied explicitly in Stage 5.")
println("Type /exit or /quit to leave the REPL.")
chat_repl(session)
