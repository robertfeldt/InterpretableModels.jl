abstract type AbstractLinearITRM <: InterpretableTimeseriesRegressionModel end

hasintercept(m::AbstractLinearITRM) = m.hasintercept
getintercept(m::AbstractLinearITRM) = hasintercept(m) ? m.params[end] : 0.0
ndigits(m::AbstractLinearITRM) = m.ndigits
nfactors(m::AbstractLinearITRM) = m.nfactors
nfeatures(m::AbstractLinearITRM) = m.nfeatures
maxlag(m::AbstractLinearITRM) = m.maxlag
minlag(m::AbstractLinearITRM) = m.minlag
mincoef(m::AbstractLinearITRM) = m.mincoef
maxcoef(m::AbstractLinearITRM) = m.maxcoef
r(m::AbstractLinearITRM, v::Float64) = round(v, digits = ndigits(m))
hasparams(m::AbstractLinearITRM) = !isnothing(params(m))
isvalidparams(m::AbstractLinearITRM, ps::Vector{Float64}) = true
setparams!(m::AbstractLinearITRM, ps::Vector{Float64}) = isvalidparams(m, ps) && (m.params = ps)
params(m::AbstractLinearITRM) = m.params
setfeaturenames!(m::AbstractLinearITRM, fns::Vector{String}) = (m.featurenames = fns)
featurename(m::AbstractLinearITRM, fi::Int) =
    (length(m.featurenames) > 0) ? m.featurenames[fi] : "feature_$(fi)"

function StatsBase.predict(m::AbstractLinearITRM, X::AbstractMatrix{N},
    ps::Vector{Float64} = params(m)) where {N<:Number}
    preds = Array{Float64}(undef, size(X, 1)-maxlag(m))
    StatsBase.predict!(m, preds, X, ps)
end

function StatsBase.predict!(m::AbstractLinearITRM, preds::Vector{Float64},
    X::AbstractMatrix{N}, ps::Vector{Float64} = params(m)) where {N<:Number}
    previ = 0
    for row in (maxlag(m)+1):size(X, 1)
        preds[(previ += 1)] = predictforrow(m, X, row, ps)
    end
    preds # Note that due to lag this will be shorter than num rows of X
end

function Base.show(io::IO, m::AbstractLinearITRM)
    writedescription(io, m)
    write(io, "  model: ")
    writemodel(io, m)
end

function writemodel(io::IO, m::AbstractLinearITRM)
    if !hasparams(m)
        write(io, "<no params set yet>")
        return
    end
    hasintercept(m) && write(io, string(r(m, getintercept(m))))
    for fi in 1:nfactors(m)
        writefactor(io, m, fi)
    end
end

modelname(m::AbstractLinearITRM) = "Linear ITRM (Interpretable Timeseries Regression Model)"

function writedescription(io::IO, m::AbstractLinearITRM)
    write(io, modelname(m) * "\n")
    if mincoef(m) < 0.0 && maxcoef(m) > 0.0
        write(io, "  with positive and negative coefficients")
    elseif maxcoef(m) > 0.0
        write(io, "  with positive coefficients only")
    elseif mincoef(m) < 0.0
        write(io, "  with negative coefficients only")
    end
    write(io, " (allowed range: [$(mincoef(m)), $(maxcoef(m))])\n")
    write(io, "  num factors  = $(nfactors(m))\n")
    write(io, "  num features = $(nfeatures(m))\n")
    write(io, "  lag range    = $(minlag(m))-$(maxlag(m))\n")
end

modelasstring(m::AbstractLinearITRM, df::DataFrame, ps::Vector{Float64} = m.params) =
    modelasstring(m, ps, names(df))

function modelasstring(m::AbstractLinearITRM, params::Vector{Float64} = m.params,
    featurenames::AbstractVector = String[])
    setparams!(m, params)
    length(featurenames) > 0 && setfeaturenames!(m, map(string, featurenames))
    iob = IOBuffer()
    writemodel(iob, m)
    String(take!(iob))
end

function calcfitness(m::AbstractLinearITRM, X::AbstractMatrix{N}, y::AbstractVector{N},
    ps::Vector{Float64} = m.params, lossfn::Function = l2loss) where {N<:Number}
    yhat = predict(m, X, ps)
    lossfn(yhat, align(yhat, y))
end

align(yhat, y) = y[(end-length(yhat)+1):end]

defaultoptimizer(m::AbstractLinearITRM) =
    InterpretableModels.DefaultOptimizer(m)

function predictandcalcloss(m::AbstractLinearITRM, X::AbstractMatrix{Float64},
            y::AbstractVector{Float64}, ps::Vector{Float64}, lossfn::Function)
    yhat = predict(m, X, ps)
    return yhat, lossfn(yhat, align(yhat, y))
end

fit!(m::AbstractLinearITRM, X::AbstractMatrix{Float64}, y::AbstractVector{Float64}) =
    fit!(m, defaultoptimizer(m), X, y)
