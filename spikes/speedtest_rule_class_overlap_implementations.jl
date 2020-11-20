# For MurTree and other sparse decision rules/sets/trees implementations
# we want very fast ways of checking for which data instances they "overlap" (or not).
# We test a few different ways below:
#  1. BitArray
#  2. Set{Int}
#  3. SparseVector{Int32,UInt8}

Ns = [100, 1000, 10000, 100000] # Number of training data instances

using Random, StatsBase, SparseArrays

# We use Set{Int} as the basis and create the others from that.
abstract type AbstractIndicatorSet end

struct IndicatorSet <: AbstractIndicatorSet
    n::Int
    set::Set{Int} # Maybe use UInt32 here instead sincec rarely more than 2^32 data instances?
    IndicatorSet(n::Int, pcttrue = 0.5) = new(n, Set{Int}(shuffle(1:n)[1:max(1, round(Int, pcttrue*n))]))
end
Base.length(iset::IndicatorSet) = iset.n
Base.in(i::Integer, iset::IndicatorSet) = in(i, iset.set)
Base.eachindex(iset::IndicatorSet) = 1:iset.n
SparseArrays.nnz(iset::IndicatorSet) = length(iset.set)

function make_bitarray(iset::IndicatorSet)
    a = BitArray(undef, length(iset))
    for i in eachindex(iset)
        a[i] = in(i, iset) ? true : false
    end
    a
end

import SparseArrays.sparsevec
function sparsevec(iset::IndicatorSet)
    sparsevec(collect(iset.set), ones(UInt8, nnz(iset)))
end

iset = IndicatorSet(100, 0.4)
iba = make_bitarray(iset)
ispa = sparsevec(iset)

# Count intersection of sets
function countintersection(iset1::IndicatorSet, iset2::IndicatorSet)
    n = 0
    for i in iset1.set
        in(i, iset2.set) && (n += 1)
    end
    n
end

function countintersection(iba1::BitArray, iba2::BitArray)
    sum(iba1 .& iba2)
end

function countintersection(iba1::SparseVector{UInt8,Int64}, iba2::SparseVector{UInt8,Int64})
    n = 0
    for i in iba1.nzind
        in(i, iba2.nzind) && (n += 1)
    end
    n
end

iset2 = IndicatorSet(100, 0.4)
iba2 = make_bitarray(iset2)
ispa2 = sparsevec(iset2)
@btime countintersection($iset, $iset2)
@btime countintersection($iba, $iba2)
@btime countintersection($ispa, $ispa2)

# Now let's do speedtests
using BenchmarkTools

Times = Dict{String,Dict{Any,Float64}}()
Times["Set"] = Dict{Any,Float64}()
Times["BitArray"] = Dict{Any,Float64}()
Times["SparseVec"] = Dict{Any,Float64}()
for n in Ns
    for pct = [0.20, 0.50, 0.80]
        iset1 = IndicatorSet(n, pct)
        iset2 = IndicatorSet(n, pct)
        Times["Set"][(n, pct)] = @belapsed countintersection($iset1, $iset2)

        iba1 = make_bitarray(iset1)
        iba2 = make_bitarray(iset2)
        Times["BitArray"][(n, pct)] = @belapsed countintersection($iba1, $iba2)

        ispa1 = sparsevec(iset1)
        ispa2 = sparsevec(iset2)
        Times["SparseVec"][(n, pct)] = @belapsed countintersection($ispa1, $ispa2)

        println(".")
    end
end

# BitArray is by far the fastest. Probably it is much faster also for calculating
# the full overlap matrix (with TP, TN, FP, FN), at least if we save the number
# of ones.

struct IndicatorBitArray <: AbstractIndicatorSet
    n::Int
    numones::Int
    numzeros::Int
    ba::BitArray
    IndicatorBitArray(iset::IndicatorSet) = begin
        n = length(iset)
        numones = length(iset.set)
        new(n, numones, n-numones, make_bitarray(iset))
    end
end
Base.length(iba::IndicatorBitArray) = iba.n
SparseArrays.nnz(iba::IndicatorBitArray) = iba.numones

function countintersection(iba1::IndicatorBitArray, iba2::IndicatorBitArray)
    sum(iba1.ba .& iba2.ba)
end

function countoverlaps(iba1::IndicatorBitArray, iba2::IndicatorBitArray)
    Num = length(iba1)
    TP = countintersection(iba1, iba2)
    FN = nnz(iba1) - TP
    FP = nnz(iba2) - TP
    TN = Num - TP - FN - FP
    return TP, FN, TN, FP
end

iset2 = IndicatorSet(100000, 0.1)

iba1 = IndicatorBitArray(iset1)
iba2 = IndicatorBitArray(iset2)

sum(countoverlaps(iba1, iba2))
@btime countoverlaps($iba1, $iba2) # circa 2.45 microsec on my MBPro 2015 for N=100,000
@btime countintersection($iba1, $iba2) # circa 2.158 microsec