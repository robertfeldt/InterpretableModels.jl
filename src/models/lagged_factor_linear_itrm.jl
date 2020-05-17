# (coeff, featurenum, lag) for each factor and possibly an intercept.
"""
A lagged factor linear regression model for time series prediction. Each factor has
a coefficient, uses a given feature of the data and with a given lag back
in time. By setting a minlag we can ensure that the model can be used that
many time steps in advance, since no factor with less than `minlag` can be
used. The `maxlag` specifies how far back in time we are allowed to look,
since very large lag values can limit the amount of prediction that can be done
in short time series.
"""
mutable struct LaggedFactorLITRM <: AbstractLinearITRM
    nfactors::Int
    nfeatures::Int
    minlag::Int
    maxlag::Int
    maxcoef::Float64
    mincoef::Float64
    hasintercept::Bool
    floorcoefs::Bool
    ndigits::Int # When printing
    params::Union{Nothing,Vector{Float64}}
    featurenames::Vector{String}
    LaggedFactorLITRM(nfactors::Int, nfeatures::Int;
        minlag::Int = 3, maxlag::Int = 10,
        maxcoef = 1e3, mincoef = -maxcoef,
        hasintercept = true, floorcoefs = false, ndigits = 2,
        params = nothing,
        featurenames = String[]) = begin
        new(nfactors, nfeatures, minlag, maxlag, maxcoef, mincoef,
            hasintercept, floorcoefs, ndigits, params, featurenames)
    end
end

function isvalidparams(m::LaggedFactorLITRM, ps::Vector{Float64})
    lagvals = lags(m, ps)
    minl, maxl = extrema(lagvals)
    (minlag(m) <= minl) && (maxl <= maxlag(m))
end

actualminlag(m::LaggedFactorLITRM, ps::Vector{Float64} = params(m)) =
    minimum(lags(m, ps))

actualmaxlag(m::LaggedFactorLITRM, ps::Vector{Float64} = params(m)) =
    maximum(lags(m, ps))

lag(m::LaggedFactorLITRM, fi::Int, ps::Vector{Float64} = params(m)) =
    floor(Int, ps[fi*3])
lag(m::LaggedFactorLITRM, lagval::Float64) = floor(Int, lagval)
lags(m::LaggedFactorLITRM, ps::Vector{Float64} = params(m)) =
    Int[lag(m, ps[fi*3]) for fi in 1:nfactors(m)]

horizon(m::LaggedFactorLITRM) = actualminlag(m)

const AlmostRoundUpTo1 = 0.9999999

function parambounds(m::LaggedFactorLITRM)
    bounds = Tuple{Float64, Float64}[]
    for _ in 1:nfactors(m)
        push!(bounds, (m.mincoef, m.maxcoef))
        # Since we will floor(Int, v) the following we add almost 1.0 to:
        push!(bounds, (1.0, float(m.nfeatures) + AlmostRoundUpTo1))
        push!(bounds, (float(m.minlag), float(m.maxlag) + AlmostRoundUpTo1))
    end
    hasintercept(m) && push!(bounds, (m.mincoef, m.maxcoef))
    bounds
end

function predictforrow(m::LaggedFactorLITRM, X::AbstractMatrix{N}, row::Int,
    ps::Vector{Float64} = params(m)) where {N<:Number}
    prediction = hasintercept(m) ? ps[end] : 0.0
    for fi in 1:nfactors(m)
        coef, featureidx, lag = getfactor(m, fi, ps)
        prediction += (coef * X[row-lag, featureidx])
    end
    return prediction
end

function getfactor(m::LaggedFactorLITRM, fi::Int, ps::Vector{Float64} = params(m))
    i = 1 + (fi-1)*3
    c, f, l = ps[i:(i+2)]
    coef = m.floorcoefs ? floor(Int, c) : c
    return coef, floor(Int, f), floor(Int, l)
end

modelname(m::LaggedFactorLITRM) =
    "LaggedFactor LITRM (Linear Interpretable Timeseries Regression Model)"

function writefactor(io::IO, m::LaggedFactorLITRM, fi::Int)
    c, f, l = getfactor(m, fi)
    coef = r(m, c)
    if coef != 0
        op = (coef > 0.0) ? "+" : "-"
        write(io, " $op $(abs(coef)) * $(featurename(m, f))[T-$(l)]")
    end
end
