module InterpretableModels
using CSV, BlackBoxOptim, DataFrames, Statistics, StatsBase
using Lasso, MLDataUtils, ROC

export SLIMScoringSystem, bboptimize_slim_scoring_system,
    isclassificationmodel, isregressionmodel, istimeseriesmodel, horizon

export LaggedFactorLITRM, MovingAverageLITRM, predict, fit!

export LassoLIMOptimizer, LIMScoringSystem

# Core and utils
include("utils.jl")
include("loss_functions.jl")
include(joinpath("models", "types.jl"))
include(joinpath("optimizers", "types.jl"))

# Optimizers
include(joinpath("optimizers", "blackboxoptim.jl"))

# Default optimizer is a black-box, heuristic one since they can fit all
# models. You can use a more model-specific optimizer to (maybe) get
# better performance.
DefaultOptimizer(m) = BlackBoxOptimizer(m)

# Models
include(joinpath("models", "slim_scoring.jl"))
include(joinpath("models", "abstract_linear_itrm.jl"))
include(joinpath("models", "lagged_factor_linear_itrm.jl"))
include(joinpath("models", "moving_average_linear_itrm.jl"))
include(joinpath("models", "lasso_simple_scoring.jl"))

# Optimizers

end # module
