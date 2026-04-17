using Test

@testset "bundle IO helpers" begin
    mktempdir() do temporary_directory
        weights = Dict{String, Any}(
            "matrix" => Float32[1 2; 3 4],
            "vector" => Float32[5, 6],
        )
        weights_path = joinpath(temporary_directory, "weights.jld2")

        saved_path = KeemenaLM.Core.save_weights_jld2(weights_path, weights)
        loaded_weights = KeemenaLM.Core.load_weights_jld2(saved_path)

        @test saved_path == weights_path
        @test loaded_weights["matrix"] == weights["matrix"]
        @test loaded_weights["vector"] == weights["vector"]
    end
end

@testset "resolve_bundle local directory" begin
    mktempdir() do temporary_directory
        bundle_directory = joinpath(temporary_directory, "bundle")
        mkpath(bundle_directory)
        write(joinpath(bundle_directory, "bundle.json"), "{}")

        resolved_path = KeemenaLM.resolve_bundle(bundle_directory)

        @test resolved_path == abspath(bundle_directory)
        @test_throws ArgumentError KeemenaLM.resolve_bundle(joinpath(temporary_directory, "missing_bundle"))
    end
end

@testset "bundle save/load roundtrip with dummy weights" begin
    mktempdir() do temporary_directory
        config = KeemenaLM.GPT2Config(vocab_size = 16, context_length = 8, num_layers = 1, num_heads = 2, embedding_size = 8, ffn_hidden_size = 16)
        weights = Dict{String, Any}("dummy_weight" => Float32[1, 2, 3])
        bundle = KeemenaLM.Bundle(model_config = config, weights = weights)

        KeemenaLM.save_bundle(temporary_directory, bundle)
        loaded_bundle = KeemenaLM.load_bundle(temporary_directory)

        @test loaded_bundle.manifest.architecture == "gpt2"
        @test loaded_bundle.manifest.parameter_schema == "gpt2.v1"
        @test loaded_bundle.model_config.vocab_size == config.vocab_size
        @test loaded_bundle.model_config.context_length == config.context_length
        @test loaded_bundle.model_config.num_layers == config.num_layers
        @test loaded_bundle.model_config.num_heads == config.num_heads
        @test loaded_bundle.model_config.embedding_size == config.embedding_size
        @test loaded_bundle.model_config.ffn_hidden_size == config.ffn_hidden_size
        @test loaded_bundle.weights["dummy_weight"] == weights["dummy_weight"]
        @test isfile(joinpath(temporary_directory, "bundle.json"))
        @test isfile(joinpath(temporary_directory, "model_config.json"))
        @test isfile(joinpath(temporary_directory, "weights.jld2"))
    end
end

@testset "bundle save/load rejects unsafe manifest paths" begin
    mktempdir() do temporary_directory
        config = KeemenaLM.GPT2Config(vocab_size = 16, context_length = 8, num_layers = 1, num_heads = 2, embedding_size = 8, ffn_hidden_size = 16)
        weights = Dict{String, Any}("dummy_weight" => Float32[1, 2, 3])

        absolute_bundle = KeemenaLM.Bundle(
            manifest = KeemenaLM.BundleManifest(model_config_file = "/tmp/model_config.json"),
            model_config = config,
            weights = weights,
        )
        traversal_bundle = KeemenaLM.Bundle(
            manifest = KeemenaLM.BundleManifest(weights_file = "../weights.jld2"),
            model_config = config,
            weights = weights,
        )

        @test_throws ArgumentError KeemenaLM.save_bundle(temporary_directory, absolute_bundle)
        @test_throws ArgumentError KeemenaLM.save_bundle(temporary_directory, traversal_bundle)
    end
end

@testset "bundle save/load supports nested relative manifest paths" begin
    mktempdir() do temporary_directory
        config = KeemenaLM.GPT2Config(vocab_size = 16, context_length = 8, num_layers = 1, num_heads = 2, embedding_size = 8, ffn_hidden_size = 16)
        weights = Dict{String, Any}("dummy_weight" => Float32[1, 2, 3])
        manifest = KeemenaLM.BundleManifest(
            model_config_file = "configs/model_config.json",
            weights_file = "weights/portable/weights.jld2",
            tokenizer_path = "tokenizer/assets",
            preprocessing_path = "preprocessing/assets",
        )
        bundle = KeemenaLM.Bundle(manifest = manifest, model_config = config, weights = weights)

        KeemenaLM.save_bundle(temporary_directory, bundle)
        loaded_bundle = KeemenaLM.load_bundle(temporary_directory)

        @test isfile(joinpath(temporary_directory, "configs", "model_config.json"))
        @test isfile(joinpath(temporary_directory, "weights", "portable", "weights.jld2"))
        @test loaded_bundle.manifest.model_config_file == "configs/model_config.json"
        @test loaded_bundle.manifest.weights_file == "weights/portable/weights.jld2"
        @test loaded_bundle.weights["dummy_weight"] == weights["dummy_weight"]
    end
end
