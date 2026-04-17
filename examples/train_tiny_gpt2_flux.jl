#!/usr/bin/env julia
using KeemenaLM
using Flux

config = GPT2Config(
    vocab_size = 32,
    context_length = 8,
    num_layers = 2,
    num_heads = 2,
    embedding_size = 16,
    ffn_hidden_size = 32,
)
device = :auto
model = KeemenaLM.FluxBackend.move_model_to_device(instantiate(config; backend = :flux, seed = 42); device = device)
trainer = KeemenaLM.Core.Trainer(
    model;
    optimizer = Flux.Descent(0.01),
    backend = :flux,
    metadata = Dict("example" => "train_tiny_gpt2_flux"),
)

# Supported Stage 4 policy: integer token-id batches stay on CPU even when model/compute move to CUDA.
input_token_ids = reshape(Int[1, 2, 3, 4, 1, 2, 3, 4], 4, 2)
target_token_ids = reshape(Int[2, 3, 4, 5, 2, 3, 4, 5], 4, 2)

for step_index in 1:5
    result = KeemenaLM.Core.train_step!(trainer, input_token_ids, target_token_ids)
    println("step=$(step_index) loss=$(result.loss)")
end

checkpoint_path = joinpath(pwd(), "train_tiny_gpt2_flux_checkpoint.jld2")
KeemenaLM.save_checkpoint(checkpoint_path, trainer, model; example = "stage4")

bundle_path = joinpath(pwd(), "train_tiny_gpt2_flux_bundle")
bundle = KeemenaLM.Bundle(model_config = config, weights = KeemenaLM.Core.extract_weights(model))
KeemenaLM.save_bundle(bundle_path, bundle)

println("Checkpoint saved to $(checkpoint_path)")
println("Bundle saved to $(bundle_path)")
