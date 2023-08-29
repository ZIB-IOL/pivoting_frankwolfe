module PivotingFrankWolfe

using FrankWolfe
using LinearAlgebra
using SparseArrays
using BasicLU

export pivoting_away_frank_wolfe

include("matrix_operations.jl")
include("algorithm.jl")

end
