using InterpretableModels
using Test

@testset "InterpretableModels.jl" begin
    include("test_utils.jl")
    #include("test_binarize_features.jl")
    include("test_slim_scoring.jl")
    include("test_lagged_factor_linear_itrm.jl")
end
