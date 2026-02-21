#!/usr/bin/env julia
using KeemenaLM

# Scaffold-only example for intended Flux training workflow.
config = GPT2Config(vocab_size = 256, context_length = 64, num_layers = 2, num_heads = 4, embedding_size = 64, ffn_hidden_size = 128)
model = instantiate(config; backend = :flux)

# TODO v0.1: build tokenizer/preprocessing adapters and training loop.
println("Flux training scaffold ready: $(typeof(model))")
