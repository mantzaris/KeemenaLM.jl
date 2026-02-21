using Test
using KeemenaLM

@testset "KeemenaLM.jl" begin
    include("unit/test_masking.jl")
    include("unit/test_configs.jl")
    include("unit/test_bundle_schema.jl")

    include("integration/test_forward_flux.jl")
    include("integration/test_forward_lux.jl")
    include("integration/test_generate_flux.jl")
    include("integration/test_generate_lux.jl")
    include("integration/test_train_step_flux.jl")
    include("integration/test_train_step_lux.jl")
end
