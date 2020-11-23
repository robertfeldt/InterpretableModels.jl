using InterpretableModels: zerooneconvert, optimize

@testset "LIMScoringSystem via Lasso" begin

N, P = 1_000, 500
realcoefs = zeros(Int, P)
realcoefs[1] = 4
realcoefs[42] = -2
realcoefs[244] = -8
X = rand(N, P) .< 0.5
yraw = X * realcoefs
y = zerooneconvert.(yraw) # class is 0 or 1
models = optimize(LIMScoringSystem, LassoLIMOptimizer(), X, y)

@test models isa AbstractVector
@test models[1] isa LIMScoringSystem

end