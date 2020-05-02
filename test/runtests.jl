using InterpretableModels
using Test

@testset "InterpretableModels.jl" begin
    include("test_utils.jl")
    include("test_slim_scoring.jl")
end
