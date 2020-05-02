"""
    slim_scoring_loss01_norm0(λ, X, y, C0)

SLIM (Supersparse Linear Integer Models) loss function that includes
the 0-1 loss and the norm-0 loss parts but not the norm-1 part (which is used 
in the original paper). To include also the latter, use `slim_scoring_loss`.
"""
function slim_scoring_loss01_norm0(λ::Vector{N0}, 
    X::AbstractMatrix{N1}, y::AbstractVector{N2}, C0::Number) where {N0<:Number,N1<:Number,N2<:Number}
    ι = round.(Int, λ)
    loss01 = sum((y .* (X * ι)) .<= 0)/length(y)
    norm0 = sum(ι[2:end] .!= 0) # No loss for lambda[1] which is the intercept
    loss01 + C0*norm0
end

abstract type InterpretableModel end

struct SlimScoringSystem <: InterpretableModel
    λ::Vector{Int}
    L::Int
    nonzeroindices::Vector{Int}
    trainaccuracy::Float64
    SlimScoringSystem(λ::Vector{N}, L::Int, trainaccuracy::Float64 = -100.0) where {N<:Number} = begin
        @assert 0.0 <= trainaccuracy <= 1.0
        new(floor.(Int, λ), L, nonzeroindices(λ), trainaccuracy)
    end
end
function SlimScoringSystem(λ::Vector{N}, L::Int, 
    X::AbstractMatrix{N1}, y::AbstractVector{N2}) where {N<:Number,N1<:Number,N2<:Number}
    SlimScoringSystem(λ, L, sign_accuracy(λ, X, y))
end
numrules(s::SlimScoringSystem) = nnzeroasint(s.λ) - 1

function bboptimize_slim_scoring_system(X::AbstractMatrix{N1}, y::AbstractVector{N2}, 
    C0::Real; 
    L::Int = 5, 
    MaxTime = 60.0, 
    PopSize = 50) where {N1<:Number,N2<:Number}

    C0 = float(C0)
    fitfn(λ::Vector{Float64}) = slim_scoring_loss01_norm0(λ, X, y, C0)

    P = size(X, 2)

    r = bboptimize(fitfn;
            SearchRange = bbo_floor_int_searchrange(L),
            NumDimensions = P,
            MaxTime = MaxTime, 
            PopulationSize = PopSize)

    return SlimScoringSystem(best_candidate(r), L, X, y)
end

function bboptimize_slim_scoring_system(df::DataFrame, classcolumn::Int, C0::Real; kws...) where {N1<:Number,N2<:Number}
    y = as_sign_labels(df[:, classcolumn])
    N, P = size(df)
    idxs = collect(1:P)
    deleteat!(idxs, classcolumn)
    X = hcat(ones(Float64, N), Matrix{Float64}(df[:, idxs]))
    bboptimize_slim_scoring_system(X, y, C0; kws...)
end
