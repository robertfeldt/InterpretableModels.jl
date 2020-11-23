"""
    Linear Integer Model Scoring System

A LIM scoring system for classification which can present itself
and its predictions nicely.

TODO: Merge this with the existing SLIM scoring systems, for now
I just copied in the LASSO-specific code
"""
struct LIMScoringSystem{S} <: InterpretableClassificationModel
    maxscore::Int               # Max score per rule
    scores::AbstractVector{Int} # Actual scores per rule
    treshold::Int
    colnames::Vector{S}
    classfeaturename::S
end

LIMScoringSystem(maxscore::Int, scores::AbstractVector{Int}, treshold::Int) =
    LIMScoringSystem{String}(maxscore, scores, treshold, String["F$i" for i in 1:length(scores)], "<<unknown>>")

zerooneconvert(v::Number) = (v < 0) ? 0 : 1

function classify(ss::LIMScoringSystem, X::AbstractMatrix{N}) where {N<:Number}
    zerooneconvert.(ss.treshold .+ X * ss.scores)
end

StatsBase.predict(m::LIMScoringSystem, X::AbstractMatrix) = classify(m, X)

# Calculate integer score in range from -L to L, given beta value and absolute max beta.
rescale(beta, absmaxbeta, L) = round(Int, L*beta/absmaxbeta)

function rescale_lasso_coefficients(betas::AbstractVector{Float64}, beta0 = 0; L::Int = 10)
    amb = maximum(abs.(betas))
    rescaledbetas = rescale.(betas, amb, L)
    return rescaledbetas, rescale(beta0, amb, L)
end

# A Lasso-based Linear Integer Model optimizer:
struct LassoLIMOptimizer <: InterpretableModelOptimizer
    withintercept::Bool
    maxnummodels::Int
    maxscore::Int
    LassoLIMOptimizer(; maxscore::Int = 10, withintercept = true, maxnummodels = 30) =
        new(withintercept, maxnummodels, maxscore)
end

function optimize(::Type{<:LIMScoringSystem}, o::LassoLIMOptimizer,
    X::AbstractMatrix, y::AbstractVector)
    lp = fit(LassoPath, X, y, Binomial(), LogitLink(); intercept = o.withintercept)
    nmodels = min(o.maxnummodels, size(lp.coefs, 2))
    wi = o.withintercept
    coefs = unique(map(i ->
        rescale_lasso_coefficients(lp.coefs[:, i], (wi ? lp.b0[i] : 0); L = o.maxscore),
        2:nmodels))
    LIMScoringSystem[LIMScoringSystem(o.maxscore, c...) for c in coefs]
end
