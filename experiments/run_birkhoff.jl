using LinearAlgebra
using FrankWolfe
using JSON
using Random
using PivotingFrankWolfe

# portfolio_dir = readdir(joinpath(@__DIR__, "portfolio/"), join=true)
# push!(portfolio_dir, portfolio_dir[1])

const size_dims = [30^2, 30^2, 30^2, 40^2, 40^2, 50^2]
Random.seed!(42)
raw_data = map(size_dims) do n
    W = randn(n, n)
    W' * W
end

function build_objective(W, mu)
    n = size(W, 1)
    wn = norm(W)
    w_1 = 200 * randn(n)
    W = W + mu * I
    f(x) = 1 / n^2 * (1/2 * dot(x, W, x) + dot(x, w_1)) + wn
    function grad!(storage, x)
        mul!(storage, W, x)
        storage .+= w_1
        storage ./= n^2
    end
    (f, grad!)
end

const lmo = FrankWolfe.BirkhoffPolytopeLMO()

const max_iteration = 10000

for (i, df) in enumerate(raw_data)
    n = size(df, 1)
    for mu in (0.0, 0.5)
        f, grad! = build_objective(df, mu)
        res_file = joinpath(@__DIR__, "results_birkhoff_$(i)_$(n).json")
        if isfile(res_file)
            continue
        end
        touch(res_file)
        x0 = FrankWolfe.compute_extreme_point(lmo, ones(size(df, 1)))
        all_results = Dict()
        for lazy in (true, )
            @info "Running Birkhoff portfolio $i $(size(df)) $mu $lazy"
            res_pafw = []
            PivotingFrankWolfe.pivoting_away_frank_wolfe(f, grad!, lmo, x0, verbose=false, lazy=lazy, line_search=FrankWolfe.MonotonicStepSize(), max_iteration=max_iteration, callback=PivotingFrankWolfe.make_trajectory_with_active_set(res_pafw), full_solve=true)
            res_afw = []
            FrankWolfe.away_frank_wolfe(f, grad!, lmo, x0, verbose=false, lazy=lazy, max_iteration=max_iteration, line_search=FrankWolfe.MonotonicStepSize(), callback=PivotingFrankWolfe.make_trajectory_with_active_set(res_afw))
            res_bpcg = []
            FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, x0, verbose=false, lazy=lazy, max_iteration=max_iteration, line_search=FrankWolfe.MonotonicStepSize(), callback=PivotingFrankWolfe.make_trajectory_with_active_set(res_bpcg))
            res_pbpcg = []
            PivotingFrankWolfe.pivoting_pairwise_frank_wolfe(f, grad!, lmo, x0, verbose=false, lazy=lazy, max_iteration=max_iteration, line_search=FrankWolfe.MonotonicStepSize(), callback=PivotingFrankWolfe.make_trajectory_with_active_set(res_pbpcg), full_solve=true)
            for (k, v) in [
                "pafw" => res_pafw,
                "afw" => res_afw,
                "bpcg" => res_bpcg,
                "pbpcg" => res_pbpcg,
            ]
                if lazy
                    all_results[k] = v
                else
                    all_results[k * "_nlazy"] = v
                end
            end
        end
        open(res_file, "w") do f
            write(f, JSON.json(all_results, 4))
        end
    end
end
