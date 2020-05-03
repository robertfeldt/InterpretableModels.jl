"""
    bbo_floor_int_searchrange(L::Int)

Generate a BlackBoxOptim SearchRange for optimizing integers on a
float scale when we are going to use floor to get back an Int.
"""
bbo_floor_int_searchrange(L::Int) = (0.499999999-L, L+0.499999999)

@inline floortoints(λ::Vector{N}) where {N<:Number} = floor.(Int, λ)
@inline floortoints(λ::Vector{Int}) = λ

"""
    nnzeroasint(λ::Vector{N})

Count number of nonzero elements of a vector when floored to integers.
"""
nnzeroasint(λ::Vector{N}) where {N<:Number} = sum(floortoints(λ) .!= 0)

"""
    nonzeroindices(λ::Vector{N}, sorted = false)

Return the indices of the nonzero elements of a vector when they are floored to integers.
"""
function nonzeroindices(λ::Vector{N}, sorted = true) where {N<:Number}
    idxs = findall(v -> v != 0, λ)
    return sorted ? idxs[sortperm(λ[idxs], rev=true)] : idxs
end

"""
    sign_predict(λ::Vector, X)

Predict class, indicated with +1 or -1, i.e. sign-based, by multiplying
a vector `λ` interpreted as integer coefficients and a design matrix `X`.
"""
sign_predict(λ::Vector{Int}, X::AbstractMatrix) = Int[(v<0 ? -1 : 1) for v in (X * λ)]
sign_predict(λ::Vector{N}, X::AbstractMatrix) where {N<:Number} = 
    sign_predict(floortoints(λ), X)

"""
    sign_accuracy(yhat, y)

Accuracy with which yhat predicts the (signed) class labels in y.
"""
@inline function sign_accuracy(yhat::Vector{N1}, y::Vector{N2}) where {N1<:Number,N2<:Number}
    @assert length(yhat) == length(y)
    sum((yhat .* y) .> 0)/float(length(yhat))
end

"""
    sign_accuracy(λ, X, y)

Accuracy with which a coefficient vector `λ` predicts `y` given design matrix `X`.
"""
sign_accuracy(λ::Vector{N}, X::AbstractMatrix{N1}, y::Vector{N2}) where {N<:Number,N1<:Number,N2<:Number} =
    sign_accuracy(sign_predict(λ, X), y)

"""
    sign_misclassification_rate(yhat, y)

Misclassification rate for the prediction yhat of the (signed) class labels in y.
"""
@inline sign_misclassification_rate(yhat, y) = 1.0 - sign_accuracy(yhat, y)

"""
    aspct(v::Float64)

Convert to a percentage value.
"""
@inline aspct(v::Float64) = round(100.0*v, digits=3)

as_sign_labels(y::Vector{N}) where {N<:Number} = Int[(c>0 ? 1 : -1) for c in y]
function as_sign_labels(y::AbstractVector)
    uniquevalues = sort(unique(y))
    @assert 1 <= length(uniquevalues) <= 2
    Int[(c==uniquevalues[1] ? 1 : -1) for c in y]
end

function uptobutexcluding(P::Int, excl::AbstractVector)
    setdiff(collect(1:P), excl)
end
