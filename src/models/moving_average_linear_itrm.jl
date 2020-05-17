# (coeff, featurenum, lag, windowlen) for each factor and possibly an intercept.
"""
A moving average linear regression model for time series prediction. This is
similar to but more general than a lagged factor model since the factors can
be moving averages over a range of lags rather than simply a single lag.
In theory, this could "smoothen out" noise in the dependence from one feature
to another while still allowing non-moving average (lagged factors) by setting
the length of the moving window to 1. The `minlag` and `maxlag` limits
how far back in time we can look.
"""
mutable struct MovingAverageLITRM <: AbstractLinearITRM
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
    MovingAverageLITRM(nfactors::Int, nfeatures::Int;
        minlag::Int = 3, maxlag::Int = 10,
        maxcoef = 1e3, mincoef = -maxcoef,
        hasintercept = true, floorcoefs = false, ndigits = 2,
        params = nothing,
        featurenames = String[]) = begin
        new(nfactors, nfeatures, minlag, maxlag, maxcoef, mincoef,
            hasintercept, floorcoefs, ndigits, params, featurenames)
    end
end

function parambounds(m::MovingAverageLITRM)
    bounds = Tuple{Float64, Float64}[]
    for _ in 1:nfactors(m)
        # Order of params: coef, feature, startlag, windowlen
        push!(bounds, (m.mincoef, m.maxcoef))
        # Since we will floor(Int, v) the following we add almost 1.0 to:
        push!(bounds, (1.0, float(m.nfeatures) + AlmostRoundUpTo1))
        push!(bounds, (float(m.minlag), float(m.maxlag) + AlmostRoundUpTo1))
        # Window length 1.0 means it is just a lagged factor with no
        # moving average window.
        push!(bounds, (1.0, float(m.maxlag) - float(m.minlag) + 1 + AlmostRoundUpTo1))
    end
    hasintercept(m) && push!(bounds, (m.mincoef, m.maxcoef))
    bounds
end

function isvalidparams(m::MovingAverageLITRM, ps::Vector{Float64})
    spans = windowspans(m, ps)
    (minlag(m) <= minimum(map(first, spans))) &&
    (maximum(map(last, spans)) <= maxlag(m)) &&
    all(t -> 1 <= t[2] <= (maxlag(m) - minlag(m) + 1),  spans)
end

windowspans(m::MovingAverageLITRM, ps::Vector{Float64} = params(m)) =
    Tuple{Int, Int, Int}[getfactor(m, fi, ps)[[3,4,5]] for fi in 1:nfactors(m)]

startofwindows(m::MovingAverageLITRM, ps::Vector{Float64} = params(m)) =
    map(first, windowspans(m, ps))

windowlengths(m::MovingAverageLITRM, ps::Vector{Float64} = params(m)) =
    map(t -> t[2], windowspans(m, ps))

endofwindows(m::MovingAverageLITRM, ps::Vector{Float64} = params(m)) =
    map(last, windowspans(m, ps))

# If we have params the smallest (since they are positive in the params vector)
# lag value is our horizon. If we don't have params we use the
actualminlag(m::MovingAverageLITRM, ps::Vector{Float64} = params(m)) =
    hasparams(m) ? minimum(startofwindows(m, ps)) : minlag(m)

actualmaxlag(m::MovingAverageLITRM, ps::Vector{Float64} = params(m)) =
        hasparams(m) ? maximum(endofwindows(m, ps)) : maxlag(m)

horizon(m::MovingAverageLITRM) = actualminlag(m)

function predictforrow(m::MovingAverageLITRM, X::AbstractMatrix{N}, row::Int,
    ps::Vector{Float64} = params(m)) where {N<:Number}
    prediction = hasintercept(m) ? ps[end] : 0.0
    for fi in 1:nfactors(m)
        coef, featureidx, winstart, winlen, wend = getfactor(m, fi, ps)
        #@show (row, winstart, winlen, wend, actualmaxlag(m), maxlag(m), ps)
        if winlen > 1
            windowavg = mean(X[(row-wend):(row-winstart), featureidx])
        else
            windowavg = X[row-winstart, featureidx]
        end
        prediction += (coef * windowavg)
    end
    return prediction
end

function windowfromparams(m::MovingAverageLITRM, ws::N1, wl::N2) where {N1<:Number,N2<:Number}
    wstart = floor(Int, ws)
    wend = min(maxlag(m), wstart + floor(Int, wl) - 1)
    wlen = wend - wstart + 1
    return wstart, wlen, wend
end

function getfactor(m::MovingAverageLITRM, fi::Int, ps::Vector{Float64} = params(m))
    i = 1 + (fi-1)*4
    c, f, ws, wl = ps[i:(i+3)]
    coef = m.floorcoefs ? floor(Int, c) : c
    return coef, floor(Int, f), windowfromparams(m, ws, wl)...
end

modelname(m::MovingAverageLITRM) =
    "MovingAverage factor LITRM (Linear Interpretable Timeseries Regression Model)"

function writefactor(io::IO, m::MovingAverageLITRM, fi::Int)
    c, f, ws, wlen, wend = getfactor(m, fi)
    coef = r(m, c)
    if coef != 0
        op = (coef > 0.0) ? "+" : "-"
        write(io, " $op $(abs(coef)) * ")
        f = "$(featurename(m, f))[T-$(ws)]"
        s = (wlen > 1) ? "MAvg($(f), $(wlen))" : f
        write(io, s)
    end
end
