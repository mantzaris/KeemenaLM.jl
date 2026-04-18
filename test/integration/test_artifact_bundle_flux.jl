using Test
using Pkg
using Pkg.Artifacts

struct ArtifactTestTokenizer
    token_to_id::Dict{Char, Int}
    id_to_token::Dict{Int, Char}
end

function ArtifactTestTokenizer(alphabet::AbstractString)
    characters = collect(alphabet)
    token_to_id = Dict(character => index for (index, character) in enumerate(characters))
    id_to_token = Dict(index => character for (index, character) in enumerate(characters))
    return ArtifactTestTokenizer(token_to_id, id_to_token)
end

KeemenaLM.Core.tokenizer_encode(tokenizer::ArtifactTestTokenizer, text::AbstractString) =
    [get(tokenizer.token_to_id, character, 1) for character in text]

KeemenaLM.Core.tokenizer_decode(tokenizer::ArtifactTestTokenizer, token_ids::AbstractVector{<:Integer}) =
    String([get(tokenizer.id_to_token, token_id, '?') for token_id in token_ids])

function bind_test_official_model(bundle_directory::AbstractString, artifacts_toml::AbstractString)
    artifact_hash_value = Pkg.Artifacts.create_artifact() do artifact_directory
        for entry in readdir(bundle_directory)
            cp(joinpath(bundle_directory, entry), joinpath(artifact_directory, entry); force = true)
        end
    end

    specification = KeemenaLM.Core.official_model_spec("tiny-demo")
    Pkg.Artifacts.bind_artifact!(artifacts_toml, specification.artifact_name, artifact_hash_value; force = true)
    return artifact_hash_value
end

@testset "artifact-backed official model resolution works end-to-end" begin
    mktempdir() do temporary_directory
        artifacts_toml = joinpath(temporary_directory, "Artifacts.toml")
        tokenizer = ArtifactTestTokenizer(" abcdefghijklmnopqrstuvwxyz.,!?")
        config = KeemenaLM.GPT2Config(
            vocab_size = length(tokenizer.token_to_id),
            context_length = 64,
            num_layers = 2,
            num_heads = 2,
            embedding_size = 8,
            ffn_hidden_size = 16,
        )
        original_model = KeemenaLM.instantiate(config; backend = :flux, seed = 52)
        bundle = KeemenaLM.Bundle(model_config = config, weights = KeemenaLM.Core.extract_weights(original_model))
        bundle_directory = joinpath(temporary_directory, "bundle")
        KeemenaLM.save_bundle(bundle_directory, bundle)

        @test_throws ArgumentError KeemenaLM.download_model("tiny-demo"; artifacts_toml = artifacts_toml)
        bind_test_official_model(bundle_directory, artifacts_toml)

        local_resolved_path = KeemenaLM.resolve_bundle(bundle_directory; artifacts_toml = artifacts_toml)
        official_models = KeemenaLM.available_models(; artifacts_toml = artifacts_toml)
        artifact_resolved_path = KeemenaLM.resolve_bundle("tiny-demo"; artifacts_toml = artifacts_toml)
        downloaded_path = KeemenaLM.download_model("tiny-demo"; artifacts_toml = artifacts_toml)
        loaded_bundle = KeemenaLM.load_bundle("tiny-demo"; artifacts_toml = artifacts_toml)
        loaded_model = KeemenaLM.load_model("tiny-demo"; backend = :flux, artifacts_toml = artifacts_toml)

        generation_config = KeemenaLM.GenerationConfig(max_new_tokens = 4, temperature = 0.8, top_k = 4, seed = 1234)
        generated_text = KeemenaLM.generate(loaded_model, tokenizer, nothing, "ab"; generation_config = generation_config)
        session = KeemenaLM.ChatSession(
            loaded_model,
            tokenizer,
            nothing;
            system_prompt = "You are a tiny official demo assistant.",
            generation_config = KeemenaLM.GenerationConfig(max_new_tokens = 4, temperature = 0.0, seed = 9),
        )
        chat_reply = KeemenaLM.chat!(session, "hi")

        @test local_resolved_path == abspath(bundle_directory)
        @test any(model_info -> model_info.key == "tiny-demo", official_models)
        @test any(model_info -> model_info.key == "tiny-demo" && model_info.installed, official_models)
        @test any(model_info -> model_info.key == "tiny-demo" && occursin("local artifact registration", lowercase(model_info.install_note)), official_models)
        @test artifact_resolved_path == downloaded_path
        @test isfile(joinpath(artifact_resolved_path, "bundle.json"))
        @test loaded_bundle.manifest.parameter_schema == "gpt2.v1"
        @test typeof(loaded_model) == typeof(original_model)
        @test startswith(generated_text, "ab")
        @test chat_reply isa String
        @test session.message_history == [
            (role = "user", content = "hi"),
            (role = "assistant", content = chat_reply),
        ]
    end
end

@testset "local bundle directory takes precedence over official model key collisions" begin
    mktempdir() do temporary_directory
        artifacts_toml = joinpath(temporary_directory, "Artifacts.toml")
        tokenizer = ArtifactTestTokenizer(" abcdefghijklmnopqrstuvwxyz.,!?")
        config = KeemenaLM.GPT2Config(
            vocab_size = length(tokenizer.token_to_id),
            context_length = 64,
            num_layers = 2,
            num_heads = 2,
            embedding_size = 8,
            ffn_hidden_size = 16,
        )
        model = KeemenaLM.instantiate(config; backend = :flux, seed = 61)
        bundle = KeemenaLM.Bundle(model_config = config, weights = KeemenaLM.Core.extract_weights(model))

        artifact_bundle_directory = joinpath(temporary_directory, "artifact_bundle")
        KeemenaLM.save_bundle(artifact_bundle_directory, bundle)
        bind_test_official_model(artifact_bundle_directory, artifacts_toml)

        collision_bundle_directory = joinpath(temporary_directory, "tiny-demo")
        KeemenaLM.save_bundle(collision_bundle_directory, bundle)

        cd(temporary_directory) do
            @test KeemenaLM.resolve_bundle("tiny-demo"; artifacts_toml = artifacts_toml) == abspath(collision_bundle_directory)
        end

        rm(collision_bundle_directory; recursive = true)
        @test KeemenaLM.resolve_bundle("tiny-demo"; artifacts_toml = artifacts_toml) != abspath(collision_bundle_directory)
        @test isfile(joinpath(KeemenaLM.resolve_bundle("tiny-demo"; artifacts_toml = artifacts_toml), "bundle.json"))
    end
end
