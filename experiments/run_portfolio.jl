using MAT
using LinearAlgebra
using FrankWolfe
using CardinalityGuaranteedFrankWolfe
using Random
using JSON

portfolio_dir = readdir(joinpath(@__DIR__, "portfolio/"), join=true)
raw_data = map(portfolio_dir) do f
    W = MAT.matread(f)["W"]
    # increasing size for larger problems
    W = vcat(W, W)
    W = hcat(W, W)
    W = hcat(W, W)
end

function build_objective_gradient(W)
    (n, p) = size(W)
    invpn = 1 / n / p
    function f(x)
        s = 0.0
        for t in 1:p
            s -= invpn * log(dot(x, @view(W[:,t])))
        end
        return s + 10
    end
    function ∇f(storage, x)
        storage .= 0
        for t in 1:p
            temp_rev = dot(x, @view(W[:,t]))
            @. storage -= @view(W[:,t]) / temp_rev
        end
        storage .*= invpn
    end
    (f, ∇f)
end

const f0, grad!0 = build_objective_gradient(raw_data[1])
const lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)

const x0_prec = FrankWolfe.compute_extreme_point(lmo, ones(size(raw_data[1], 1)))

# precompiling everything
CardinalityGuaranteedFrankWolfe.cardinality_guaranteed_away_frank_wolfe(f0, grad!0, lmo, x0_prec, verbose=false, lazy=true, max_iteration=3, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set([]))
FrankWolfe.away_frank_wolfe(f0, grad!0, lmo, x0_prec, verbose=false, lazy=true, trajectory=true, max_iteration=3, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set([]))
FrankWolfe.blended_pairwise_conditional_gradient(f0, grad!0, lmo, x0_prec, verbose=false, lazy=true, max_iteration=3, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set([]))

const max_iteration = 10000

for (i, df) in enumerate(raw_data)
    f, grad! = build_objective_gradient(df)
    res_file = joinpath(@__DIR__, "results", "portfolio_$(i).json")
    if isfile(res_file)
        continue
    end
    touch(res_file)
    @info "Running portfolio $i"
    x0 = FrankWolfe.compute_extreme_point(lmo, ones(size(df, 1)))
    res_cgafw = []
    CardinalityGuaranteedFrankWolfe.cardinality_guaranteed_away_frank_wolfe(f, grad!, lmo, x0, line_search=FrankWolfe.MonotonicStepSize(), verbose=true, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_cgafw), full_solve=true)
    res_afw = []
    FrankWolfe.away_frank_wolfe(f, grad!, lmo, x0, verbose=true, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_afw))
    res_bpcg = []
    FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, x0, verbose=true, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_bpcg))
    all_results = Dict(
        "cg_afw" => res_cgafw,
        "afw" => res_afw,
        "bpcg" => res_bpcg,
    )
    open(res_file, "w") do f
        write(f, JSON.json(all_results, 4))
    end
end
