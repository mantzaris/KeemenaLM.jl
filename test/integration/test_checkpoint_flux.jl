using Flux
using Test

struct CheckpointTestTokenizer
    token_to_id::Dict{Char, Int}
    id_to_token::Dict{Int, Char}
end

function CheckpointTestTokenizer(alphabet::AbstractString)
    characters = collect(alphabet)
    token_to_id = Dict(character => index for (index, character) in enumerate(characters))
    id_to_token = Dict(index => character for (index, character) in enumerate(characters))
    return CheckpointTestTokenizer(token_to_id, id_to_token)
end

KeemenaLM.Core.tokenizer_encode(tokenizer::CheckpointTestTokenizer, text::AbstractString) =
    [get(tokenizer.token_to_id, character, 1) for character in text]

KeemenaLM.Core.tokenizer_decode(tokenizer::CheckpointTestTokenizer, token_ids::AbstractVector{<:Integer}) =
    String([get(tokenizer.id_to_token, token_id, '?') for token_id in token_ids])

@testset "Flux checkpoint roundtrip" begin
    mktempdir() do temporary_directory
        tokenizer = CheckpointTestTokenizer(" abcdefgh")
        config = KeemenaLM.GPT2Config(
            vocab_size = length(tokenizer.token_to_id),
            context_length = 8,
            num_layers = 2,
            num_heads = 2,
            embedding_size = 8,
            ffn_hidden_size = 16,
        )
        model = KeemenaLM.instantiate(config; backend = :flux, seed = 31)
        optimizer = Flux.Descent(0.01)
        optimizer_state = Flux.setup(optimizer, model)
        trainer = KeemenaLM.Core.Trainer(
            model;
            optimizer = optimizer,
            optimizer_state = optimizer_state,
            backend = :flux,
            step = 7,
            epoch = 2,
            rng_state = Dict("seed" => 1234),
            metadata = Dict("note" => "stage3"),
        )

        checkpoint_path = joinpath(temporary_directory, "flux_checkpoint.jld2")
        saved_path = KeemenaLM.save_checkpoint(checkpoint_path, trainer, model; run = "integration")
        checkpoint = KeemenaLM.load_checkpoint(saved_path)

        reloaded_model = KeemenaLM.instantiate(checkpoint.model_config; backend = checkpoint.manifest.backend)
        KeemenaLM.Core.load_weights!(reloaded_model, checkpoint.weights)
        reloaded_trainer = KeemenaLM.Core.Trainer(
            reloaded_model;
            optimizer = checkpoint.optimizer,
            optimizer_state = checkpoint.optimizer_state,
            backend = checkpoint.manifest.backend,
            step = checkpoint.step,
            epoch = checkpoint.epoch,
            rng_state = checkpoint.rng_state,
            metadata = checkpoint.metadata,
        )

        generation_config = KeemenaLM.GenerationConfig(max_new_tokens = 4, temperature = 0.8, top_k = 4, seed = 555)
        original_output = KeemenaLM.generate(model, tokenizer, nothing, "ab"; generation_config = generation_config)
        reloaded_output = KeemenaLM.generate(reloaded_model, tokenizer, nothing, "ab"; generation_config = generation_config)
        original_weights = KeemenaLM.Core.extract_weights(model)
        reloaded_weights = KeemenaLM.Core.extract_weights(reloaded_model)

        @test saved_path == checkpoint_path
        @test checkpoint.manifest.backend == :flux
        @test checkpoint.manifest.architecture == "gpt2"
        @test checkpoint.manifest.parameter_schema == "gpt2.v1"
        @test checkpoint.step == trainer.step
        @test checkpoint.epoch == trainer.epoch
        @test checkpoint.rng_state == trainer.rng_state
        @test checkpoint.metadata["note"] == "stage3"
        @test checkpoint.metadata["run"] == "integration"
        @test checkpoint.optimizer isa Flux.Descent
        @test checkpoint.optimizer.eta == optimizer.eta
        @test typeof(checkpoint.optimizer_state) == typeof(optimizer_state)
        @test sort(collect(keys(original_weights))) == sort(collect(keys(reloaded_weights)))
        @test all(original_weights[key] == reloaded_weights[key] for key in keys(original_weights))
        @test reloaded_trainer.step == trainer.step
        @test reloaded_trainer.epoch == trainer.epoch
        @test reloaded_trainer.backend == :flux
        @test original_output == reloaded_output
    end
end
