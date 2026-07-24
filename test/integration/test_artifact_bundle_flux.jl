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

function copy_directory_contents(source_directory::AbstractString, destination_directory::AbstractString)
    for entry in readdir(source_directory)
        cp(joinpath(source_directory, entry), joinpath(destination_directory, entry); force = true)
    end
end

function bind_test_official_model(source_directory::AbstractString, artifacts_toml::AbstractString, model_key::AbstractString)
    artifact_hash_value = Pkg.Artifacts.create_artifact() do artifact_directory
        copy_directory_contents(source_directory, artifact_directory)
    end

    specification = KeemenaLM.Core.official_model_spec(model_key)
    Pkg.Artifacts.bind_artifact!(artifacts_toml, specification.artifact_name, artifact_hash_value; force = true)
    return artifact_hash_value
end

function write_dummy_tokenizer_bundle(tokenizer_bundle_directory::AbstractString)
    mkpath(tokenizer_bundle_directory)
    write(joinpath(tokenizer_bundle_directory, "tokenizer.json"), "{}")
    write(joinpath(tokenizer_bundle_directory, "keemena_training_manifest.json"), "{}")
    return tokenizer_bundle_directory
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
        bind_test_official_model(bundle_directory, artifacts_toml, "tiny-demo")

        local_resolved_path = KeemenaLM.resolve_bundle(bundle_directory; artifacts_toml = artifacts_toml)
        official_models = KeemenaLM.available_models(; artifacts_toml = artifacts_toml)
        artifact_resolved_path = KeemenaLM.resolve_bundle("tiny-demo"; artifacts_toml = artifacts_toml)
        downloaded_path = KeemenaLM.download_model("tiny-demo"; artifacts_toml = artifacts_toml)
        downloaded_artifact_root = KeemenaLM.download_model_artifact("tiny-demo"; artifacts_toml = artifacts_toml)
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
        @test any(model_info -> model_info.key == "tiny-demo" && occursin("local demo", lowercase(model_info.install_note)), official_models)
        @test artifact_resolved_path == downloaded_path
        @test downloaded_artifact_root == downloaded_path
        @test isfile(joinpath(artifact_resolved_path, "bundle.json"))
        @test_throws ArgumentError KeemenaLM.resolve_tokenizer_bundle("tiny-demo"; artifacts_toml = artifacts_toml)
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

@testset "v9 artifact root resolves bundle and tokenizer sidecar" begin
    mktempdir() do temporary_directory
        artifacts_toml = joinpath(temporary_directory, "Artifacts.toml")
        config = KeemenaLM.GPT2Config(vocab_size = 32, context_length = 16, num_layers = 1, num_heads = 2, embedding_size = 8, ffn_hidden_size = 16)
        model = KeemenaLM.instantiate(config; backend = :flux, seed = 88)
        bundle = KeemenaLM.Bundle(model_config = config, weights = KeemenaLM.Core.extract_weights(model))

        artifact_root = joinpath(temporary_directory, "v9_artifact_root")
        bundle_directory = joinpath(artifact_root, "bundle")
        tokenizer_bundle_directory = joinpath(artifact_root, "tokenizer_bundle")
        KeemenaLM.save_bundle(bundle_directory, bundle)
        write_dummy_tokenizer_bundle(tokenizer_bundle_directory)
        mkpath(joinpath(artifact_root, "metadata"))
        write(joinpath(artifact_root, "metadata", "metrics.json"), "{}")

        bind_test_official_model(artifact_root, artifacts_toml, "tiny-chatbot-v9-broad-336m")

        model_info = only(filter(info -> info.key == "tiny-chatbot-v9-broad-336m", KeemenaLM.available_models(; artifacts_toml = artifacts_toml)))
        resolved_root = KeemenaLM.resolve_model_artifact("tiny-chatbot-v9-broad-336m"; artifacts_toml = artifacts_toml)
        downloaded_root = KeemenaLM.download_model_artifact("tiny-chatbot-v9-broad-336m"; artifacts_toml = artifacts_toml)
        resolved_bundle = KeemenaLM.resolve_bundle("tiny-chatbot-v9-broad-336m"; artifacts_toml = artifacts_toml)
        downloaded_bundle = KeemenaLM.download_model("tiny-chatbot-v9-broad-336m"; artifacts_toml = artifacts_toml)
        resolved_tokenizer = KeemenaLM.resolve_tokenizer_bundle("tiny-chatbot-v9-broad-336m"; artifacts_toml = artifacts_toml)

        @test model_info.installed
        @test model_info.has_tokenizer_bundle
        @test model_info.bundle_subdir == "bundle"
        @test model_info.tokenizer_bundle_subdir == "tokenizer_bundle"
        @test resolved_root == downloaded_root
        @test resolved_bundle == downloaded_bundle
        @test dirname(resolved_bundle) == resolved_root
        @test dirname(resolved_tokenizer) == resolved_root
        @test isfile(joinpath(resolved_bundle, "bundle.json"))
        @test isfile(joinpath(resolved_tokenizer, "tokenizer.json"))
        @test KeemenaLM.resolve_bundle(artifact_root; artifacts_toml = artifacts_toml) == abspath(bundle_directory)
        @test KeemenaLM.resolve_model_artifact(artifact_root; artifacts_toml = artifacts_toml) == abspath(artifact_root)
        @test KeemenaLM.resolve_tokenizer_bundle(artifact_root; artifacts_toml = artifacts_toml) == abspath(tokenizer_bundle_directory)
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
        bind_test_official_model(artifact_bundle_directory, artifacts_toml, "tiny-demo")

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
