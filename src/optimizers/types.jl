abstract type InterpretableModelOptimizer end

abstract type HeuristicOptimizer <: InterpretableModelOptimizer end

fit!(m::InterpretableModel, o::InterpretableModelOptimizer) = notdef((m, o), :fit!)
