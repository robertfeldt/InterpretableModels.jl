module InterpretableModels
using CSV, BlackBoxOptim, DataFrames

export SlimScoringSystem, bboptimize_slim_scoring_system

include("utils.jl")
include("slim_scoring.jl")

end # module
