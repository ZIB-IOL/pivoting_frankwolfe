using CardinalityGuaranteedFrankWolfe
using FrankWolfe
using LinearAlgebra
using JSON
using Random

Random.seed!(42)

function build_objective_gradient(y, A)
    buffer = similar(y)
    AtA = A' * A
    invnp = 1 / prod(size(A))
    function f(x)
        mul!(buffer, A, x)
        s = 0.0
        for i in eachindex(y)
            s += invnp * (y[i] - buffer[i])^2
        end
        return s
    end
    function grad!(storage, x)
        # storage = 2/n A'A x
        mul!(storage, AtA, x, 2/n, 0)
        # storage += -2/n A' y
        mul!(storage, A', y, -2/n, 1)
        storage .*= invnp
        return storage
    end
    (f, grad!)
end

n = 6000
m = 7000

const A = randn(m, n)
const x_true = randn(n)
const frac_zero = 0.7
for i in eachindex(x_true)
    if rand() <= 0.7
        x_true[i] = 0
    end
end
const tau = norm(x_true, 1)
const y = A * x_true + 1 * randn(m)

const f, grad! = build_objective_gradient(y, A)
const lmo0 = FrankWolfe.LpNormLMO{1}(norm(x_true, 1) / 100)

const x0 = FrankWolfe.compute_extreme_point(lmo0, randn(n))
const x0_ill = FrankWolfe.compute_extreme_point(lmo0, randn(2n))

const f1, grad1! = build_objective_gradient(y, hcat(A, A))

# precompiling everything
for (f0, grad0!, x0, xk0_prec) in ((f, grad!, x0, xk0_prec), (f1, grad1!, x0_ill, xk0_prec_ill))
    @info "standard"
    @info "$(size(xk0_prec))"
    CardinalityGuaranteedFrankWolfe.cardinality_guaranteed_away_frank_wolfe(f0, grad0!, lmo0, x0, verbose=false, lazy=true, max_iteration=3, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set([]))
    FrankWolfe.away_frank_wolfe(f0, grad0!, lmo0, x0, verbose=false, lazy=true, trajectory=true, max_iteration=3, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set([]))
    FrankWolfe.blended_pairwise_conditional_gradient(f0, grad0!, lmo0, x0, verbose=false, lazy=true, max_iteration=3, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set([]))
    # similar for K-sparse polytope
    # 
end


const max_iteration = 5000
const ILL_CONDITIONED_VARIANT = true

for tau_frac in (5, 20, 80, 100, 120, 150)
    res_file = joinpath(@__DIR__, "results", "signalrecovery_$(tau_frac)_$(m)_$(n).json")
    if isfile(res_file)
        continue
    end
    touch(res_file)
    @info "Running L1 signal recovery $tau_frac"
    lmo = FrankWolfe.LpNormLMO{1}(tau / tau_frac)
    res_cgafw = []
    CardinalityGuaranteedFrankWolfe.cardinality_guaranteed_away_frank_wolfe(f, grad!, lmo, x0, verbose=false, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_cgafw), full_solve=true)
    res_afw = []
    FrankWolfe.away_frank_wolfe(f, grad!, lmo, x0, verbose=false, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_afw))
    res_bpcg = []
    FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, x0, verbose=false, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_bpcg))
    all_results = Dict(
        "tau_frac" => tau_frac,
        "tau" => tau,
        "cg_afw" => res_cgafw,
        "afw" => res_afw,
        "bpcg" => res_bpcg,
    )
    open(res_file, "w") do f
        write(f, JSON.json(all_results, 4))
    end
end

if ILL_CONDITIONED_VARIANT
    for tau_frac in (5, 20, 80, 100, 120, 150)
        res_file = joinpath(@__DIR__, "results", "signalrecovery_illcond_$(tau_frac)_$(m)_$(n).json")
        if isfile(res_file)
            continue
        end
        touch(res_file)
        @info "Running L1 ill-conditioned signal recovery $tau_frac"
        lmo = FrankWolfe.LpNormLMO{1}(tau / tau_frac)
        res_cgafw = []
        CardinalityGuaranteedFrankWolfe.cardinality_guaranteed_away_frank_wolfe(f1, grad1!, lmo, x0_ill, verbose=false, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_cgafw), full_solve=true)
        res_afw = []
        FrankWolfe.away_frank_wolfe(f1, grad1!, lmo, x0_ill, verbose=false, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_afw))
        res_bpcg = []
        FrankWolfe.blended_pairwise_conditional_gradient(f1, grad1!, lmo, x0_ill, verbose=false, lazy=true, max_iteration=max_iteration, callback=CardinalityGuaranteedFrankWolfe.make_trajectory_with_active_set(res_bpcg))
        all_results = Dict(
            "tau_frac" => tau_frac,
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
