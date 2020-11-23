# Very simple Scoring systems created similar to the select-regress-round technique in:
#  Jung et al, "Simple Rules for Complex Decisions" https://arxiv.org/pdf/1702.04690.pdf
# However, we do LassoPath-round to get multiple scoring systems from a single run of
# Lasso and we use the Lasso path on all features rather than forward regression to
# select features.
using Lasso

abstract type LearntModel end
abstract type Classifier <: LearntModel end

struct LinearIntegerModel <: Classifier
    L::Int                  # Max beta value used when creating the scoring system
    β::AbstractVector{Int}
    treshold::Int # If sum of betas is higher that treshold we predict the class
end
marginconvert(v::Number) = (v < 0) ? -1 : 1
zerooneconvert(v::Number) = (v < 0) ? 0 : 1
extremascores(β::AbstractVector{N}) where {N<:Number} =
    (sum(filter(v -> v < 0, β)), sum(filter(v -> v > 0, β)))
function classify(ss::LinearIntegerModel, X::AbstractMatrix{N}) where {N<:Number}
    zerooneconvert.(ss.treshold .+ X * ss.β)
end

using StatsBase
StatsBase.predict(m::LinearIntegerModel, X::AbstractMatrix) = classify(m, X)

# Calculate integer score in range from -L to L, given beta value and absolute max beta.
rescale(beta, absmaxbeta, L) = round(Int, L*beta/absmaxbeta)

function rescale_lasso_coefficients(betas::AbstractVector{Float64}, beta0 = 0; L::Int = 10)
    amb = maximum(abs.(betas))
    rsbetas = rescale.(betas, amb, L)
    return rsbetas, rescale(beta0, amb, L)
end

# Returns a set of zero or more models that have been optimized given
# a dataset.
abstract type ModelOptimizer end

# A Lasso-based Linear Integer Model optimizer:
struct LassoLIMOptimizer <: ModelOptimizer
    withintercept::Bool
    maxnummodels::Int
    L::Int
    LassoLIMOptimizer(; L::Int = 10, withintercept = true, maxnummodels = 30) =
        new(withintercept, maxnummodels, L)
end

function optimize(::Type{LinearIntegerModel}, o::LassoLIMOptimizer,
    X::AbstractMatrix, y::AbstractVector)
    lp = fit(LassoPath, X, y, Binomial(), LogitLink(); intercept = o.withintercept)
    nmodels = min(o.maxnummodels, size(lp.coefs, 2))
    wi = o.withintercept
    coefs = unique(map(i ->
        rescale_lasso_coefficients(lp.coefs[:, i], (wi ? lp.b0[i] : 0); L = o.L),
        2:nmodels))
    LinearIntegerModel[LinearIntegerModel(o.L, c...) for c in coefs]
end

using ROC

function rankbyauc(models::AbstractVector{M}, X, y) where {M<:LearntModel}
    preds = map(m -> predict(m, X), models)
    rocs = map(pr -> roc(pr, y), preds)
    aucs = map(AUC, rocs)
    p = sortperm(aucs, rev=true)
    return models[p], aucs[p], preds[p]
end

using SparseArrays
N, P = 10_000, 5_000
realcoefs = zeros(Int, P)
realcoefs[1] = 4
realcoefs[42] = -2
realcoefs[244] = -8
@time sparsecoefs = sparsevec(realcoefs)
X = rand(N, P) .< 0.5

@time yraw = X * realcoefs
#@time yrawsparse = X * sparsecoefs
using MLDataUtils
@time y = zerooneconvert.(yraw)
models = optimize(LinearIntegerModel, LassoLIMOptimizer(), X, y)

ms, aucs, preds = rankbyauc(models, X, y)

# Let's try SLOPE and see if similar result. Assumes RCall available and will install
# SLOPE R package which requires a functioning compiler build chain.
using RCall
function rlibrary(libname)
    @rput libname
    R"require(libname, character.only = TRUE)"
end
rlibrary("SLOPE")

struct SlopeLIMOptimizer <: ModelOptimizer
    withintercept::Bool
    maxnummodels::Int
    L::Int
    SlopeLIMOptimizer(; L::Int = 5, withintercept = true, maxnummodels = 30) =
        new(withintercept, maxnummodels, L)
end

function optimize(::Type{LinearIntegerModel}, o::SlopeLIMOptimizer,
    X::AbstractMatrix, y::AbstractVector)

    @rput X
    @rput y
    wintercept = o.withintercept
    @rput wintercept
    R"slopefit <- SLOPE(X, y, family='binomial', intercept=wintercept)"
    @rget slopefit

    nmodels = min(o.maxnummodels, size(lp.coefs, 2))
    wi = o.withintercept
    coefs = unique(map(i ->
        rescale_lasso_coefficients(lp.coefs[:, i], (wi ? lp.b0[i] : 0); L = o.L),
        2:nmodels))
    LinearIntegerModel[LinearIntegerModel(o.L, c...) for c in coefs]
end
