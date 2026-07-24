using Flux
using Test

@testset "Flux device helper semantics" begin
    @test Base.get_extension(KeemenaLM, :KeemenaLMMetalExt) === nothing
    @test !KeemenaLM.FluxBackend.has_functional_metal_gpu()
    @test KeemenaLM.FluxBackend.resolve_flux_device(:cpu) === :cpu

    config = KeemenaLM.GPT2Config(
        vocab_size = 8,
        context_length = 4,
        num_layers = 1,
        num_heads = 2,
        embedding_size = 4,
        ffn_hidden_size = 8,
    )
    model = KeemenaLM.instantiate(config; backend = :flux, seed = 5)
    batch = reshape(Int[1, 2, 3, 4], 4, 1)
    float_batch = Float32[1 2; 3 4]

    cpu_model = KeemenaLM.FluxBackend.move_model_to_device(model; device = :cpu)
    cpu_batch = KeemenaLM.FluxBackend.move_batch_to_device(batch; device = :cpu)
    cpu_float_batch = KeemenaLM.FluxBackend.move_batch_to_device(float_batch; device = :cpu)
    auto_model = KeemenaLM.FluxBackend.move_model_to_device(model; device = :auto)
    auto_batch = KeemenaLM.FluxBackend.move_batch_to_device(batch; device = :auto)
    auto_float_batch = KeemenaLM.FluxBackend.move_batch_to_device(float_batch; device = :auto)

    @test cpu_model.token_embedding isa Matrix{Float32}
    @test cpu_batch isa Matrix{Int}
    @test cpu_float_batch isa Matrix{Float32}
    @test auto_batch isa Matrix{Int}
    @test Base.get_extension(KeemenaLM, :KeemenaLMMetalExt) === nothing

    metal_error = try
        KeemenaLM.FluxBackend.move_model_to_device(model; device = :metal)
        nothing
    catch exception
        exception
    end
    @test metal_error isa ArgumentError
    if metal_error isa ArgumentError
        error_message = sprint(showerror, metal_error)
        @test occursin("device=:metal", error_message)
        @test occursin("install Metal.jl", error_message)
        @test occursin("using Metal", error_message)
    end

    if KeemenaLM.FluxBackend.has_functional_cuda_gpu()
        @test !(auto_model.token_embedding isa Matrix{Float32})
        @test !(auto_float_batch isa Matrix{Float32})
        gpu_token_batch = KeemenaLM.FluxBackend.move_batch_to_device(batch; device = :gpu)
        @test gpu_token_batch isa Matrix{Int}
    else
        @test auto_model.token_embedding isa Matrix{Float32}
        @test auto_float_batch isa Matrix{Float32}
        @test_throws ArgumentError KeemenaLM.FluxBackend.move_batch_to_device(batch; device = :gpu)
        @test_throws ArgumentError KeemenaLM.FluxBackend.move_batch_to_device(float_batch; device = :gpu)
    end
end

@testset "Flux train_step! updates parameters and stays finite on a toy run" begin
    config = KeemenaLM.GPT2Config(
        vocab_size = 16,
        context_length = 8,
        num_layers = 1,
        num_heads = 2,
        embedding_size = 8,
        ffn_hidden_size = 16,
    )
    model = KeemenaLM.instantiate(config; backend = :flux, seed = 19)
    trainer = KeemenaLM.Core.Trainer(
        model;
        optimizer = Flux.Descent(0.01),
        backend = :flux,
        metadata = Dict("test" => "stage4"),
    )

    input_token_ids = reshape(Int[1, 2, 3, 4, 1, 2, 3, 4], 4, 2)
    target_token_ids = reshape(Int[2, 3, 4, 5, 2, 3, 4, 5], 4, 2)

    original_token_embedding = copy(model.token_embedding)
    losses = Float64[]
    for _ in 1:3
        result = KeemenaLM.Core.train_step!(trainer, input_token_ids, target_token_ids)
        push!(losses, result.loss)
    end

    @test trainer.step == 3
    @test all(isfinite, losses)
    @test all(loss -> loss > 0, losses)
    @test original_token_embedding != model.token_embedding
    @test trainer.optimizer_state !== nothing
end

@testset "Flux checkpoint can be saved after training" begin
    mktempdir() do temporary_directory
        config = KeemenaLM.GPT2Config(
            vocab_size = 16,
            context_length = 8,
            num_layers = 1,
            num_heads = 2,
            embedding_size = 8,
            ffn_hidden_size = 16,
        )
        model = KeemenaLM.instantiate(config; backend = :flux, seed = 23)
        trainer = KeemenaLM.Core.Trainer(model; optimizer = Flux.Descent(0.01), backend = :flux)

        input_token_ids = reshape(Int[1, 2, 3, 4], 4, 1)
        target_token_ids = reshape(Int[2, 3, 4, 5], 4, 1)
        KeemenaLM.Core.train_step!(trainer, input_token_ids, target_token_ids)

        checkpoint_path = joinpath(temporary_directory, "post_training_checkpoint.jld2")
        KeemenaLM.save_checkpoint(checkpoint_path, trainer, model; source = "train_step_test")
        checkpoint = KeemenaLM.load_checkpoint(checkpoint_path)

        @test isfile(checkpoint_path)
        @test checkpoint.step == trainer.step
        @test checkpoint.metadata["source"] == "train_step_test"
        @test checkpoint.optimizer_state !== nothing
    end
end
