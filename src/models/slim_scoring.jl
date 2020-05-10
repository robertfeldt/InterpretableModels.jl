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

abstract type AbstractSLIMScoringSystem <: InterpretableModel end

"""
    BareSLIMScoringSystem

The "bare" SLIM Scoring System for classification represents only the
scoring system itself but does NOT now about the names of features
and the class to be predicted and thus cannot print itself nicely.
Use a SLIMScoringSystem to get a nicer interface and API.
"""
struct BareSLIMScoringSystem <: AbstractSLIMScoringSystem
    λ::Vector{Int}
    L::Int
    nonzeroindices::Vector{Int}
    trainaccuracy::Float64
    BareSLIMScoringSystem(λ::Vector{N}, L::Int, trainaccuracy::Float64 = -Inf) where {N<:Number} = begin
        @assert 0.0 <= trainaccuracy <= 1.0
        new(floor.(Int, λ), L, nonzeroindices(λ), trainaccuracy)
    end
end
function BareSLIMScoringSystem(λ::Vector{N}, L::Int,
    X::AbstractMatrix{N1}, y::AbstractVector{N2}) where {N<:Number,N1<:Number,N2<:Number}
    BareSLIMScoringSystem(λ, L, sign_accuracy(λ, X, y))
end
numrules(s::BareSLIMScoringSystem) = nnzeroasint(s.λ) - 1
trainaccuracy(s::BareSLIMScoringSystem) =
    (0.0 <= s.trainaccuracy <= 1.0) ? s.trainaccuracy : nothing

"""
    SLIMScoringSystem

A SLIM scoring system for classification which can present itself
and its predictions nicely.
"""
struct SLIMScoringSystem{S} <: AbstractSLIMScoringSystem
    s::BareSLIMScoringSystem
    colnames::Vector{S}
    classfeaturename::S
    classcolidx::Int
end
function SLIMScoringSystem(s::BareSLIMScoringSystem, df::DataFrame,
    classcolumnname::Union{Symbol,Int,S}) where {S<:AbstractString}
    cname, cidx = column_name_and_index(df, classcolumnname)
    ns = names(df)
    deleteat!(ns, cidx)
    SLIMScoringSystem(s, ns, cname, cidx)
end

function Base.show(io::IO, s::SLIMScoringSystem)
end

function column_name_and_index(df::DataFrame, idx::Int)
    @assert 1 <= idx <= size(df, 2)
    return names(df)[idx], idx
end
function column_name_and_index(df::DataFrame, colname::Symbol)
    ns = names(df)
    @assert in(colname, ns)
    colname, findfirst(n -> n == colname, ns)
end
column_name_and_index(df::DataFrame, colname::AbstractString) =
    column_name_and_index(df, Symbol(string(colname)))

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

    return BareSLIMScoringSystem(best_candidate(r), L, X, y)
end

function bboptimize_slim_scoring_system(df::DataFrame, classcolumn::Int, C0::Real; kws...) where {N1<:Number,N2<:Number}
    y = as_sign_labels(df[:, classcolumn])
    N, P = size(df)
    idxs = collect(1:P)
    deleteat!(idxs, classcolumn)
    X = hcat(ones(Float64, N), Matrix{Float64}(df[:, idxs]))
    bare_ss = bboptimize_slim_scoring_system(X, y, C0; kws...)
    return SLIMScoringSystem(bare_ss, df, classcolumn)
end
