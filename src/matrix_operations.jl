
"""
Construct the initial basis matrix of the algorithm.

[̃v₀; Dₙ] ∈ R^(n+2,n+2) with ̃v₀ = (v₀, 0, 1)
"""
function construct_initial_matrix_and_lambda(v0::AbstractVector{T}) where {T}
    n = length(v0)
    M0 = spzeros(T, n + 2, n + 2)
    for idx in eachindex(v0)
        M0[idx,1] = v0[idx]
    end
    M0[n+1,1] = 0
    M0[n+2,1] = 1
    for idx in 1:n
        M0[idx,idx+1] = 1
    end
    for col in 2:n+2
        M0[n+1,col] = 1
        M0[n+2,col] = 1
    end
    λ0 = zeros(T, n+2)
    λ0[1] = 1
    return M0, λ0
end

"""
Matrix pruning algorithm. Returns number of full LU factorizations from scratch.
"""
function matrix_pruning!(M, λ, F; maxerrortol=1e-8)
    n = size(M, 1) - 2
    k = 0
    for i in 1:n
        if M[n+1,i] != 0.0
            k = i
            break
        end
    end
    nrefactor = 0
    
    for i in eachindex(λ)
        if M[n+1,i] == 0.0 && λ[i] <= 0.0
            M[:,i] = M[:,i] + M[:,k]
            # update factorization
            nrefactor += update_factorized_object(F, M[:,i], i, M, maxerrortol=maxerrortol)
        end
    end
    return nrefactor
end

"""
When a column `colidx` of the matrix M changes to `newcol`, update the corresponding factorization.
Returns whether a refactor was operated
"""
function update_factorized_object(F, newcol::SparseVector, colidx::Int, M; maxerrortol=1e-8)
    lhs = BasicLU.solve_for_update(F, newcol, getsol=true)
    piv = lhs[colidx]
    piverr = try
        BasicLU.solve_for_update(F, colidx)
        BasicLU.update(F, piv)
    catch e
        @debug "Factorization error\n$e"
        @debug "Maximum matrix entry $(maximum(M))"
        Inf
    end
    if piverr > maxerrortol
        # refactor from scratch
        try
            BasicLU.factorize(F, M)
        catch e
            @warn "Second error \n$e"
            @warn "det $(det(M))"
            println(newcol)
            error()
        end
        return true
    end
    return false
end

"""
    away_update!(M, k, λ, η, F)

Adapts the weights and factorization during an away step.
"""
function away_update!(M, k, λ, η, F)
    @assert 0 < λ[k] < 1 "$(λ[k])"
    λ .*= (1 + η)
    λ[k] -= η
    nrefactor = matrix_pruning!(M, λ, F)
    return nrefactor
end

"""
    fw_update!(M, F, v, λ, η, k_v = 0)

Carathéodory-enforcing update

If k_v > 0, it corresponds to the index of v in M, and the v vertex argument will be ignored. 
Otherwise, v must be introduced in the matrix at a new index.
F is the LU factorization object..
"""
function fw_update!(M, F, v, λ, η, k_v = 0; full_solve=false)
    vs = standardize(v)
    for i in eachindex(λ)
        if λ[i] > 0
            @assert M[end-1,i] == 0.0 "$i $(λ[i]) $(M[end-1,i])"
        else
            @assert M[end-1,i] != 0.0 "$i $(λ[i]) $(M[end-1,i])"
        end
    end
    if k_v ≠ 0
        k = k_v
        r = similar(λ)
    else # k_v == 0
        # r = -M⁻¹ vs
        r = if full_solve
            -M \ collect(vs)
        else
            -BasicLU.solve(F, vs, 'N')
        end
        if norm(M * r + vs) > 1e-9
            @error("Sparse linear system could not be solved")
            @show "full $full_solve"
            @show det(M)
            @show (norm(M * r + vs))
            @show norm(M *  (M \ collect(vs)) - vs)
            r2 = -M\collect(vs)
            @show norm(r-r2)
            error(norm(M * r + vs))
        end
        # θ, k = findmin(r[i] < 0 ? -λ[i] / r[i] : Inf for i in 1:n)
        θ = Inf
        k = 0
        @inbounds for i in eachindex(r)
            if r[i] < -1e-10
                val = -λ[i] / r[i]
                if val < θ
                    θ = val
                    k = i
                end
            end
        end
        # @show [(i, -λ[i] / r[i]) for i in eachindex(λ) if r[i] < 0]
        # @show [(i, -λ[i] / r[i], r[i]) for i in eachindex(λ) if r[i] < -1e-10]
        # @show k
        # @show θ
        # @show sparse(r)
        # @show sparse(λ)
        @debug begin
            for j in 1:size(M, 2)
                @assert abs(dot(M[:,j], vs) / norm(vs) / norm(M[:,j])) < 0.999 "$(dot(M[:,j], vs)) $(vs) $(M[:,j])"
            end
        end
        @view(M[:,k]) .= 0
        nz_indices = SparseArrays.nonzeroinds(vs)
        @inbounds for idx in nz_indices
            M[idx, k] = vs[idx]
        end
    
        # λ <- λ + r θ
        @inbounds for idx in eachindex(r)
            if r[idx] ≉ 0
                λ[idx] += r[idx] * θ
            end
        end
        λ[k] = θ
    end
    λ .*= (1 - η)
    λ[k] += η
    simplex_proj_condat!(r, λ)
    λ .= r
    for i in eachindex(λ)
        if M[end-1,i] != 0.0 && λ[i] > 0
            λ[i] = 0
        end
    end
    s = sum(λ)
    if abs(s - 1) >= 1e-9
        @warn "non simplex $(abs(s - 1))"
        λ ./= s
    end
    return k, matrix_pruning!(M, λ, F)
end

function standardize(v::Vector)
    v = copy(v)
    push!(v, 0)
    push!(v, 1)
end

standardize(v) = standardize(Base.copymutable(v))
function standardize(v::SparseVector)
    v2 = spzeros(length(v) + 2)
    copyto!(v2, v)
    v2[end-1] = 0
    v2[end] = 1
    return v2
end

function standardize(v::FrankWolfe.ScaledHotVector)
    res = spzeros(length(v) + 2)
    res[v.val_idx] = v.active_val
    res[end] = 1
    return res
end

# Taken from ProximalOperators.jl
# MIT licensed
# stores the result in-place in y
function simplex_proj_condat!(y, x, tau=1.0)
    # Implements algorithm proposed in:
    # Condat, L. "Fast projection onto the simplex and the l1 ball",
    # Mathematical Programming, 158:575–585, 2016.
    R = eltype(x)
    v = [x[1]]
    v_tilde = R[]
    rho = x[1] - tau
    N = length(x)
    for k in 2:N
        if x[k] > rho
            rho += (x[k] - rho) / (length(v) + 1)
            if rho > x[k] - tau
                push!(v, x[k])
            else
                append!(v_tilde, v)
                v = [x[k]]
                rho = x[k] - tau
            end
        end
    end
    for z in v_tilde
        if z > rho
            push!(v, z)
            rho += (z - rho) / length(v)
        end
    end
    v_changed = true
    while v_changed
        v_changed = false
        k = 1
        while k <= length(v)
            z = v[k]
            if z <= rho
                deleteat!(v, k)
                v_changed = true
                rho += (rho - z) / length(v)
            else
                k = k + 1
            end
        end
    end
    for i in eachindex(y)
        y[i] = max(x[i] - rho, 0)
    end
end
