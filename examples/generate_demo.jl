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

length(ARGS) == 1 || error("usage: julia --project=. examples/generate_demo.jl <bundle_dir_or_model_key>")

println("Available official models:")
for model_info in available_models()
    println("  ", model_info.key, " - ", model_info.description)
    println("    install note: ", model_info.install_note)
    println("    tokenizer note: ", model_info.tokenizer_note)
end

tokenizer = DemoCharTokenizer(" abcdefghijklmnopqrstuvwxyz.,!?")
model = load_model(ARGS[1]; backend = :flux)
generation_config = GenerationConfig(max_new_tokens = 24, temperature = 0.0, seed = 7)
output = generate(model, tokenizer, nothing, "hello"; generation_config = generation_config)

println("prompt> hello")
println("output> ", output)
