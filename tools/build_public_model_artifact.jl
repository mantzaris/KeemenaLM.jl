#!/usr/bin/env julia
using Pkg
using Pkg.Artifacts
using KeemenaLM

function copy_directory_contents(source_directory::AbstractString, destination_directory::AbstractString)
    for entry in readdir(source_directory)
        source_path = joinpath(source_directory, entry)
        destination_path = joinpath(destination_directory, entry)
        cp(source_path, destination_path; force = true)
    end
end

model_key = isempty(ARGS) ? "tiny-demo" : ARGS[1]
model_key == "tiny-demo" || error("tools/build_public_model_artifact.jl only builds the tiny-demo toy artifact. Use tools/package_tiny_chatbot_v9_release_artifact.sh for the v9 chatbot artifact.")
specification = KeemenaLM.Core.official_model_spec(model_key)
artifacts_toml = KeemenaLM.Core.default_artifacts_toml()

alphabet = " abcdefghijklmnopqrstuvwxyz.,!?"
config = KeemenaLM.GPT2Config(
    vocab_size = length(alphabet),
    context_length = 64,
    num_layers = 2,
    num_heads = 2,
    embedding_size = 8,
    ffn_hidden_size = 16,
)
model = KeemenaLM.instantiate(config; backend = :flux, seed = 2026)
bundle = KeemenaLM.Bundle(model_config = config, weights = KeemenaLM.Core.extract_weights(model))

mktempdir() do temporary_directory
    bundle_directory = joinpath(temporary_directory, "bundle")
    KeemenaLM.save_bundle(bundle_directory, bundle)

    artifact_hash_value = Pkg.Artifacts.create_artifact() do artifact_directory
        copy_directory_contents(bundle_directory, artifact_directory)
    end

    mkpath(dirname(artifacts_toml))
    Pkg.Artifacts.bind_artifact!(artifacts_toml, specification.artifact_name, artifact_hash_value; force = true)

    println("Registered local official model: ", specification.key)
    println("Artifact name: ", specification.artifact_name)
    println("Artifacts.toml: ", artifacts_toml)
    println("Artifact path: ", Pkg.Artifacts.artifact_path(artifact_hash_value))
    println("Install note: ", specification.install_note)
    println("Tokenizer note: ", specification.tokenizer_note)
end
