using InterpretableModels
using Test

@testset "InterpretableModels.jl" begin
    include("test_utils.jl")
    include("test_slim_scoring.jl")
    include("test_lagged_factor_linear_itrm.jl")
    include("test_moving_average_linear_itrm.jl")
    include("test_lasso_scoring_system.jl")
end
