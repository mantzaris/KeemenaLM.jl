#!/usr/bin/env julia
using KeemenaLM

# Scaffold-only generation entry point.
config = GPT2Config(vocab_size = 256, context_length = 64, num_layers = 2, num_heads = 4, embedding_size = 64, ffn_hidden_size = 128)
model = instantiate(config; backend = :flux)

# TODO v0.1: replace placeholders with real tokenizer/preprocessing and generation.
println("Generation scaffold ready: call generate(...) after backend forward is implemented.")
println("Model type: $(typeof(model))")
