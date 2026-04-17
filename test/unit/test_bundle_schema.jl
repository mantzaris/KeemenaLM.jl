using Test

@testset "bundle schema constructors" begin
    manifest = KeemenaLM.BundleManifest()

    @test manifest.schema_version == KeemenaLM.Core.BUNDLE_SCHEMA_VERSION
    @test manifest.architecture == "gpt2"
    @test manifest.model_config_file == "model_config.json"
    @test manifest.weights_file == "weights.jld2"
    @test manifest.weights_format == "jld2"
    @test manifest.parameter_schema == "gpt2.v1"
    @test isempty(manifest.metadata)

    config = KeemenaLM.GPT2Config(vocab_size = 32, context_length = 16)
    weights = Dict{String, Any}("dummy_weight" => ones(Float32, 2, 2))

    bundle = KeemenaLM.Bundle(model_config = config, weights = weights)

    @test bundle.model_config === config
    @test haskey(bundle.weights, "dummy_weight")
    @test bundle.manifest.weights_format == "jld2"
    @test KeemenaLM.Core.validate_bundle_manifest(manifest) === manifest
end

@testset "bundle manifest path safety" begin
    @test_throws ArgumentError KeemenaLM.Core.validate_bundle_manifest(
        KeemenaLM.BundleManifest(model_config_file = "/tmp/absolute.json"),
    )
    @test_throws ArgumentError KeemenaLM.Core.validate_bundle_manifest(
        KeemenaLM.BundleManifest(weights_file = "../escape.jld2"),
    )
    @test_throws ArgumentError KeemenaLM.Core.validate_bundle_manifest(
        KeemenaLM.BundleManifest(tokenizer_path = "../../tokenizer"),
    )

    nested_manifest = KeemenaLM.BundleManifest(
        model_config_file = "configs/model_config.json",
        weights_file = "weights/model/weights.jld2",
        tokenizer_path = "assets/tokenizer",
        preprocessing_path = "assets/preprocessing",
    )
    @test KeemenaLM.Core.validate_bundle_manifest(nested_manifest) === nested_manifest
end
