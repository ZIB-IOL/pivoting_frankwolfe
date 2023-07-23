using CardinalityGuaranteedFrankWolfe
using FrankWolfe
using DelimitedFiles
using LinearAlgebra
using JSON
using Random

Random.seed!(42)

df = DelimitedFiles.readdlm(joinpath(@__DIR__, "GISETTE/gisette_valid.data"))
labels = open(joinpath(@__DIR__, "GISETTE/gisette_valid.labels")) do f
    parse.(Int, readlines(f))
end

function build_objective_gradient(df, labels)
    ℓ(u) = log(exp(u/2) + exp(-u/2))
    dℓ(u) = -1/2 + inv(1 + exp(-u))
    n, p = size(df)
    invnp = inv(n) * inv(p)
    function f(x)
        loss_term = 0.0
        for i in 1:n
            dtemp = dot(@view(df[i,:]), x)
            loss_term += invnp * (ℓ(dtemp) - labels[i] * dtemp / 2)
        end
        return loss_term
    end
    function grad!(storage, x)
        storage .= 0
        for i in 1:n
            a_i = @view(df[i,:])
            dtemp = dot(a_i, x)
            scalar = invnp * (dℓ(dtemp) - labels[i] / 2)
            @inbounds for j in 1:p
                storage[j] += scalar * a_i[j]
            end
        end
    end
    (f, grad!)
end


# precompiling everything
const f0, grad!0 = build_objective_gradient(df, labels)
const lmo0 = FrankWolfe.LpNormLMO{1}(1.0)
const x0_prec = FrankWolfe.compute_extreme_point(lmo0, randn(size(df, 2)))

CardinalityGuaranteedFrankWolfe.cardinality_guaranteed_away_frank_wolfe(f0, grad!0, lmo0, x0_prec, verbose=false, lazy=true, max_iteration=3, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set([]))
FrankWolfe.away_frank_wolfe(f0, grad!0, lmo0, x0_prec, verbose=false, lazy=true, trajectory=true, max_iteration=3, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set([]))
FrankWolfe.blended_pairwise_conditional_gradient(f0, grad!0, lmo0, x0_prec, verbose=false, lazy=true, max_iteration=3, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set([]))

# similar for K-sparse polytope
const lmo_k0 = FrankWolfe.KSparseLMO(3, 3.0)
const xk0_prec = FrankWolfe.compute_extreme_point(lmo_k0, randn(size(df, 2)))

CardinalityGuaranteedFrankWolfe.cardinality_guaranteed_away_frank_wolfe(f0, grad!0, lmo_k0, xk0_prec, verbose=false, lazy=true, max_iteration=3, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set([]))
FrankWolfe.away_frank_wolfe(f0, grad!0, lmo_k0, xk0_prec, verbose=false, lazy=true, trajectory=true, max_iteration=3, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set([]))
FrankWolfe.blended_pairwise_conditional_gradient(f0, grad!0, lmo_k0, xk0_prec, verbose=false, lazy=true, max_iteration=3, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set([]))

const max_iteration = 5000
const NO_KSPARSE = false

for tau in (20.0, 30.0, 35.0, 40.0, 60.0, 70.0)
    f, grad! = build_objective_gradient(df, labels)
    for K in (10, 3)
        # other f, grad! for K-sparse, conditioning is too bad otherwise
        f, grad! = build_objective_gradient(df / 50, labels)
        if NO_KSPARSE
            break
        end
        res_file = joinpath(@__DIR__, "results", "logreg_ksparse_$(Int(tau))_$K.json")
        if isfile(res_file)
            continue
        end
        # creating the file to avoid another job running on the same instance
        touch(res_file)
        @info "Running Ksparse $tau $K"
        lmo_k = FrankWolfe.KSparseLMO(K, 0.5 * tau / K)
        xk0 = FrankWolfe.compute_extreme_point(lmo_k, randn(size(df, 2)))
        res_cgafw = []
        CardinalityGuaranteedFrankWolfe.cardinality_guaranteed_away_frank_wolfe(f, grad!, lmo_k, xk0, verbose=true, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_cgafw), gradient=collect(xk0), full_solve=true)
        res_afw = []
        FrankWolfe.away_frank_wolfe(f, grad!, lmo_k, xk0, verbose=false, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_afw), gradient=collect(xk0))
        res_bpcg = []
        FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo_k, xk0, verbose=false, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_bpcg), gradient=collect(xk0))
        all_results = Dict(
            "tau" => tau,
            "K" => K,
            "cg_afw" => res_cgafw,
            "afw" => res_afw,
            "bpcg" => res_bpcg,
        )
        open(res_file, "w") do f
            write(f, JSON.json(all_results, 4))
        end
        res_file = joinpath(@__DIR__, "results", "logreg_$(Int(tau)).json")
        if isfile(res_file)
            continue
        end
        touch(res_file)
        @info "Running L1 logreg $tau"
        lmo = FrankWolfe.LpNormLMO{1}(tau)
        x0 = FrankWolfe.compute_extreme_point(lmo, randn(size(df, 2)))
        res_cgafw = []
        CardinalityGuaranteedFrankWolfe.cardinality_guaranteed_away_frank_wolfe(f, grad!, lmo, x0, verbose=true, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_cgafw), full_solve=true)
        res_afw = []
        FrankWolfe.away_frank_wolfe(f, grad!, lmo, x0, verbose=true, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_afw))
        res_bpcg = []
        FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, x0, verbose=true, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_bpcg))
        all_results = Dict(
            "tau" => tau,
            "cg_afw" => res_cgafw,
            "afw" => res_afw,
            "bpcg" => res_bpcg,
        )
        open(res_file, "w") do f
            write(f, JSON.json(all_results, 4))
        end
    end
end
