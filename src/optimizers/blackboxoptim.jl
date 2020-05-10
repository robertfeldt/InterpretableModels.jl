mutable struct BlackBoxOptimizer <: HeuristicOptimizer
    lossfn
    optparams::AbstractDict
    optres
    BlackBoxOptimizer(lossfn::Function; kws...) = new(lossfn, kws, nothing)
end

BlackBoxOptimizer(m::InterpretableModel; lossfn = l2loss, kws...) =
    BlackBoxOptimizer(lossfn; kws...)

const DefaultBBOParams = Dict(
    :PopulationSize => 500,
    :MaxTime => 15.0,
)

function getp(key, d1, opt, dbackup = DefaultBBOParams)
    haskey(d1, key) && return d1[key]
    haskey(opt.optparams, key) && return opt.optparams[key]
    haskey(dbackup, key) && return dbackup[key]
    error("No value for parameter $key found!")
end

function fit!(m::InterpretableModel, o::BlackBoxOptimizer,
                X::AbstractMatrix{Float64}, y::AbstractVector{Float64}; kws...)

    @assert size(X, 1) == length(y)

    fitnessfn(ps::Vector{Float64}) = begin
        yhat, lossvalue = predictandcalcloss(m, X, y, ps, o.lossfn)
        lossvalue # Loss value is the fitness so should be minimixed
    end

    o.optres = bboptimize(fitnessfn;
                SearchSpace = parambounds(m),
                PopulationSize = getp(:PopulationSize, kws, o),
                MaxTime = getp(:MaxTime, kws, o))

    setparams!(m, best_candidate(o.optres))
    m
end
