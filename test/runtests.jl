using PivotingFrankWolfe
using Test
using FrankWolfe
using LinearAlgebra

@testset "Basic end-to-end test" begin
    n = 10_000
    lmo = FrankWolfe.ProbabilitySimplexOracle(3.0)
    f(x) = 1/2 * norm(x)^2
    grad!(storage, x) = storage .= x
    x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n))
    res = PivotingFrankWolfe.pivoting_away_frank_wolfe(f, grad!, lmo, x0, verbose=true, lazy=true)
    res_away = FrankWolfe.away_frank_wolfe(f, grad!, lmo, x0, verbose=true, lazy=true)
    res_pairwise = FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, x0, verbose=true, lazy=true)
end


@testset "End-to-end k-sparse test" begin
    n = 10_000
    lmo = FrankWolfe.KSparseLMO(60, 1.0)
    # 1/2 ||x - 1||^2
    f(x) = 1/2 * norm(x)^2 - sum(x) + n/2
    function grad!(storage, x)
        storage .= x
        storage .-= 1
    end
    M = PivotingFrankWolfe.construct_initial_matrix_and_lambda(x0)[1]
    x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n))
    res = PivotingFrankWolfe.pivoting_away_frank_wolfe(f, grad!, lmo, x0, verbose=true, lazy=true, line_search=FrankWolfe.Adaptive(verbose=false), max_iteration=3000)
    res_away = FrankWolfe.away_frank_wolfe(f, grad!, lmo, x0, verbose=true, lazy=true)
    res_pairwise = FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, x0, verbose=true, lazy=true)
end
