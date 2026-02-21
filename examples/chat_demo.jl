#!/usr/bin/env julia
using KeemenaLM

# Scaffold-only chat entry point.
config = GPT2Config(vocab_size = 256, context_length = 64, num_layers = 2, num_heads = 4, embedding_size = 64, ffn_hidden_size = 128)
model = instantiate(config; backend = :flux)

# TODO v0.1: initialize real tokenizer/preprocessing and call ChatSession/chat!.
println("Chat scaffold ready: call ChatSession(...) and chat!(...) after generation is implemented.")
println("Model type: $(typeof(model))")
