using Test
using KeemenaLM

@testset "KeemenaLM.jl" begin
    include("unit/test_masking.jl")
    include("unit/test_configs.jl")
    include("unit/test_bundle_schema.jl")
    include("unit/test_bundle_io.jl")
    include("unit/test_sampling.jl")
    include("unit/test_stopping.jl")
    include("unit/test_loss.jl")

    include("integration/test_forward_flux.jl")
    include("integration/test_bundle_flux.jl")
    include("integration/test_checkpoint_flux.jl")
    include("integration/test_forward_lux.jl")
    include("integration/test_generate_flux.jl")
    include("integration/test_generate_lux.jl")
    include("integration/test_train_step_flux.jl")
    include("integration/test_train_step_lux.jl")
end
