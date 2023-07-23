module CardinalityGuaranteedFrankWolfe

using FrankWolfe
using LinearAlgebra
using SparseArrays
using BasicLU

export cardinality_guaranteed_away_frank_wolfe

include("matrix_operations.jl")
include("algorithm.jl")

end
