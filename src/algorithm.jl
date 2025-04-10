using FrankWolfe: fast_dot

"""
    pivoting_pairwise_frank_wolfe(f, grad!, lmo, x0; kwargs...)

"""
function pivoting_pairwise_frank_wolfe(
    f,
    grad!,
    lmo,
    x0;
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Adaptive(),
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode::FrankWolfe.MemoryEmphasis=FrankWolfe.InplaceEmphasis(),
    gradient=nothing,
    callback=nothing,
    traj_data=[],
    timeout=Inf,
    lazy=false,
    linesearch_workspace=nothing,
    lazy_tolerance=2.0,
    extra_vertex_storage=nothing,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    full_solve=true,
)
    # add the first vertex to active set from initialization
    active_set = FrankWolfe.ActiveSet([(1.0, x0)])

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %14i %10i\n"
    headers = ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "#ActiveSet", "Nfactors")
    function format_state(state, active_set, nfactors, args...)
        rep = (
            FrankWolfe.st[Symbol(state.tt)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
            length(active_set),
            nfactors,
        )
        return rep
    end

    if trajectory
        callback = FrankWolfe.make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = FrankWolfe.make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    t = 0
    FrankWolfe.compute_active_set_iterate!(active_set)
    x = FrankWolfe.get_active_set_iterate(active_set)
    primal = convert(eltype(x), Inf)
    tt = FrankWolfe.regular
    time_start = time_ns()

    d = similar(x)

    if verbose
        println("\nPivoting Pairwise Frank-Wolfe Algorithm.")
        NumType = eltype(x)
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $NumType",
        )
        grad_type = typeof(gradient)
        println("GRADIENTTYPE: $grad_type LAZY: $lazy lazy_tolerance: $lazy_tolerance")
        if memory_mode isa FrankWolfe.InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
    end

    # likely not needed anymore as now the iterates are provided directly via the active set
    if gradient === nothing
        gradient = similar(x)
    end

    grad!(gradient, x)
    v = FrankWolfe.compute_extreme_point(lmo, gradient)
    # if !lazy, phi is maintained as the global dual gap
    phi = max(0, fast_dot(x, gradient) - fast_dot(v, gradient))
    local_gap = zero(phi)
    gamma = one(phi)

    if linesearch_workspace === nothing
        linesearch_workspace = FrankWolfe.build_linesearch_workspace(line_search, x, gradient)
    end

    @assert length(active_set) == 1 
    M, λ = construct_initial_matrix_and_lambda(active_set.atoms[1])
    F = BasicLU.LUFactor(size(M, 1))
    BasicLU.factorize(F, M)
    # index i-th contains the index of the i-th vertex of the active set in the matrix M columns 
    lookup_indices = [1]
    nfactors = 1

    while t <= max_iteration && phi >= max(epsilon, eps(epsilon))

        # managing time limit
        time_at_loop = time_ns()
        if t == 0
            time_start = time_at_loop
        end
        # time is measured at beginning of loop for consistency throughout all algorithms
        tot_time = (time_at_loop - time_start) / 1e9

        if timeout < Inf
            if tot_time ≥ timeout
                if verbose
                    @info "Time limit reached"
                end
                break
            end
        end

        #####################
        t += 1

        # compute current iterate from active set
        x = FrankWolfe.get_active_set_iterate(active_set)
        primal = f(x)
        if t > 1
            grad!(gradient, x)
        end

        _, v_local, v_local_loc, _, a_lambda, a, a_loc, _, _ =
            FrankWolfe.active_set_argminmax(active_set, gradient)

        dot_forward_vertex = FrankWolfe.fast_dot(gradient, v_local)
        dot_away_vertex = FrankWolfe.fast_dot(gradient, a)
        local_gap = dot_away_vertex - dot_forward_vertex
        if !lazy
            if t > 1
                v = FrankWolfe.compute_extreme_point(lmo, gradient)
                dual_gap = FrankWolfe.fast_dot(gradient, x) - FrankWolfe.fast_dot(gradient, v)
                phi = dual_gap
            end
        end
        if local_gap ≥ phi / lazy_tolerance
            d = FrankWolfe.muladd_memory_mode(memory_mode, d, a, v_local)
            vertex_taken = v_local
            gamma_max = a_lambda
            gamma = FrankWolfe.perform_line_search(
                line_search,
                t,
                f,
                grad!,
                gradient,
                x,
                d,
                gamma_max,
                linesearch_workspace,
                memory_mode,
            )
            gamma = min(gamma_max, gamma)
            @assert length(active_set) == length(lookup_indices)
            k = lookup_indices[a_loc]
            nfactors += away_update!(M, k, λ, gamma / (1 - gamma), F)
            # don't remove zeros on the first run to keep the indices in the active set identical
            # as to not invalidate v_local_loc
            _update_activeset_weights!(active_set, lookup_indices, λ, remove_zeros=false)
            index_in_matrix = lookup_indices[v_local_loc]
            index_in_matrix, nrefactor = fw_update!(M, F, vertex_taken, λ, gamma, index_in_matrix, full_solve=full_solve)
            nfactors += nrefactor
            _update_activeset_weights!(active_set, lookup_indices, λ)
            tt = FrankWolfe.pairwise
            if callback !== nothing
                state = FrankWolfe.CallbackState(
                    t,
                    primal,
                    primal - phi,
                    phi,
                    tot_time,
                    x,
                    vertex_taken,
                    d,
                    gamma,
                    f,
                    grad!,
                    lmo,
                    gradient,
                    tt,
                )
                if callback(state, active_set, nfactors) === false
                    break
                end
            end
        else # add to active set
            if lazy # otherwise, v computed above already
                # optionally try to use the storage
                if use_extra_vertex_storage
                    lazy_threshold = FrankWolfe.fast_dot(gradient, x) - phi / lazy_tolerance
                    (found_better_vertex, new_forward_vertex) =
                        FrankWolfe.storage_find_argmin_vertex(extra_vertex_storage, gradient, lazy_threshold)
                    if found_better_vertex
                        if verbose
                            @debug("Found acceptable lazy vertex in storage")
                        end
                        v = new_forward_vertex
                        tt = FrankWolfe.lazylazy
                    else
                        v = FrankWolfe.compute_extreme_point(lmo, gradient)
                        tt = FrankWolfe.regular
                    end
                else
                    # for t == 1, v is already computed before first iteration
                    if t > 1
                        v = FrankWolfe.compute_extreme_point(lmo, gradient)
                    end
                    tt = FrankWolfe.regular
                end
            end
            vertex_taken = v
            dual_gap = FrankWolfe.fast_dot(gradient, x) - FrankWolfe.fast_dot(gradient, v)
            # if we are about to exit, compute dual_gap with the cleaned-up x
            if dual_gap ≤ epsilon
                FrankWolfe.active_set_renormalize!(active_set)
                FrankWolfe.active_set_cleanup!(active_set)
                FrankWolfe.compute_active_set_iterate!(active_set)
                x = FrankWolfe.get_active_set_iterate(active_set)
                grad!(gradient, x)
                dual_gap = FrankWolfe.fast_dot(gradient, x) - FrankWolfe.fast_dot(gradient, v)
            end
            if !lazy || dual_gap ≥ phi / lazy_tolerance
                d = FrankWolfe.muladd_memory_mode(memory_mode, d, x, v)

                gamma = FrankWolfe.perform_line_search(
                    line_search,
                    t,
                    f,
                    grad!,
                    gradient,
                    x,
                    d,
                    one(eltype(x)),
                    linesearch_workspace,
                    memory_mode,
                )

                if callback !== nothing
                    state = FrankWolfe.CallbackState(
                        t,
                        primal,
                        primal - phi,
                        phi,
                        tot_time,
                        x,
                        vertex_taken,
                        d,
                        gamma,
                        f,
                        grad!,
                        lmo,
                        gradient,
                        tt,
                    )
                    if callback(state, active_set, nfactors) === false
                        break
                    end
                    index_in_activeset = FrankWolfe.find_atom(active_set, vertex_taken)
                    index_in_matrix = if index_in_activeset >= 1
                        lookup_indices[index_in_activeset]
                    else
                        0
                    end
                    is_new_vertex = index_in_matrix == 0
                    index_in_matrix, nrefactor = fw_update!(M, F, vertex_taken, λ, gamma, index_in_matrix, full_solve=full_solve)
                    nfactors += nrefactor
                    if is_new_vertex
                        FrankWolfe.active_set_update!(active_set, gamma, vertex_taken, false, nothing)
                        index_in_activeset = length(active_set)
                        push!(lookup_indices, index_in_matrix)
                        @assert length(lookup_indices) == length(active_set)
                    else
                        FrankWolfe.active_set_update!(active_set, gamma, vertex_taken, false, index_in_activeset)
                    end
                    _update_activeset_weights!(active_set, lookup_indices, λ)
                end
            else # dual step
                # set to computed dual_gap for consistency between the lazy and non-lazy run.
                # that is ok as we scale with the K = 2.0 default anyways
                # we only update the dual gap if the step was regular (not lazy from discarded set)
                if tt != FrankWolfe.lazylazy
                    phi = dual_gap
                    @debug begin
                        @assert tt == FrankWolfe.regular
                        v2 = FrankWolfe.compute_extreme_point(lmo, gradient)
                        g = dot(gradient, x-v2)
                        if abs(g - dual_gap) > 100 * sqrt(eps())
                            error("dual gap estimation error $g $dual_gap")
                        end
                    end
                else
                    @info "useless step"
                end
                tt = FrankWolfe.dualstep
                if callback !== nothing
                    state = FrankWolfe.CallbackState(
                        t,
                        primal,
                        primal - phi,
                        phi,
                        tot_time,
                        x,
                        vertex_taken,
                        nothing,
                        gamma,
                        f,
                        grad!,
                        lmo,
                        gradient,
                        tt,
                    )
                    if callback(state, active_set, nfactors) === false
                        break
                    end
                end
            end
        end
        primal = f(x)
    end

    # recompute everything once more for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # do also cleanup of active_set due to many operations on the same set

    if verbose
        FrankWolfe.compute_active_set_iterate!(active_set)
        x = FrankWolfe.get_active_set_iterate(active_set)
        grad!(gradient, x)
        v = FrankWolfe.compute_extreme_point(lmo, gradient)
        primal = f(x)
        phi = FrankWolfe.fast_dot(x, gradient) - FrankWolfe.fast_dot(v, gradient)
        tt = FrankWolfe.last
        tot_time = (time_ns() - time_start) / 1e9
        if callback !== nothing
            state = FrankWolfe.CallbackState(
                t,
                primal,
                primal - phi,
                phi,
                tot_time,
                x,
                v,
                nothing,
                gamma,
                f,
                grad!,
                lmo,
                gradient,
                tt,
            )
            callback(state, active_set, nfactors)
        end
    end
    FrankWolfe.active_set_renormalize!(active_set)
    FrankWolfe.active_set_cleanup!(active_set)
    FrankWolfe.compute_active_set_iterate!(active_set)
    x = FrankWolfe.get_active_set_iterate(active_set)
    grad!(gradient, x)
    v = FrankWolfe.compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = FrankWolfe.fast_dot(x, gradient) - FrankWolfe.fast_dot(v, gradient)
    tt = FrankWolfe.pp
    tot_time = (time_ns() - time_start) / 1e9
    if callback !== nothing
        state = FrankWolfe.CallbackState(
            t,
            primal,
            primal - dual_gap,
            phi,
            tot_time,
            x,
            v,
            nothing,
            gamma,
            f,
            grad!,
            lmo,
            gradient,
            tt,
        )
        callback(state, active_set, nfactors)
    end

    return x, v, primal, dual_gap, traj_data, active_set
end

function pivoting_away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0;
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Adaptive(),
    lazy_tolerance=2.0,
    epsilon=1e-7,
    away_steps=true,
    lazy=false,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode::FrankWolfe.MemoryEmphasis=FrankWolfe.InplaceEmphasis(),
    gradient=nothing,
    callback=nothing,
    traj_data=[],
    timeout=Inf,
    linesearch_workspace=nothing,
    recompute_last_vertex=true,
    full_solve=true,
)
    # add the first vertex to active set from initialization
    active_set = FrankWolfe.ActiveSet([(1.0, x0)])

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %14i %10i\n"
    headers = ("Type", "Iter", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "#ActiveSet", "Nfactors")
    function format_state(state, active_set, nfactors)
        rep = (
            FrankWolfe.st[Symbol(state.tt)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
            length(active_set),
            nfactors,
        )
        return rep
    end

    if isempty(active_set)
        throw(ArgumentError("Empty active set"))
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    x = FrankWolfe.get_active_set_iterate(active_set)
    tt = FrankWolfe.regular

    if trajectory
        callback = FrankWolfe.make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = FrankWolfe.make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    time_start = time_ns()

    d = similar(x)

    if verbose
        println("\nPivoting Away-step Frank-Wolfe Algorithm.")
        NumType = eltype(x)
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $NumType",
        )
        grad_type = typeof(gradient)
        println(
            "GRADIENTTYPE: $grad_type LAZY: $lazy lazy_tolerance: $lazy_tolerance AWAYSTEPS: $away_steps",
        )
        println("Linear Minimization Oracle: $(typeof(lmo))")
        println("Full linear solve: $(full_solve)")
    end

    # likely not needed anymore as now the iterates are provided directly via the active set
    if gradient === nothing
        gradient = similar(x)
    end

    x = FrankWolfe.get_active_set_iterate(active_set)
    primal = f(x)
    v = active_set.atoms[1]
    phi_value = convert(eltype(x), Inf)
    gamma = one(phi_value)
    
    if length(active_set) != 1
        error("Active set must be singleton for pivoting initialization")
    end
    M, λ = construct_initial_matrix_and_lambda(v)
    F = BasicLU.LUFactor(size(M, 1))
    BasicLU.factorize(F, M)

    lookup_indices = [1]
    nfactors = 1

    if linesearch_workspace === nothing
        linesearch_workspace = FrankWolfe.build_linesearch_workspace(line_search, x, gradient)
    end

    while t <= max_iteration && phi_value >= max(eps(float(typeof(phi_value))), epsilon)
        time_at_loop = time_ns()
        if t == 0
            time_start = time_at_loop
        end
        # time is measured at beginning of loop for consistency throughout all algorithms
        tot_time = (time_at_loop - time_start) / 1e9

        if timeout < Inf
            if tot_time ≥ timeout
                if verbose
                    @info "Time limit reached"
                end
                break
            end
        end

        #####################
        t += 1

        # compute current iterate from active set
        x = FrankWolfe.get_active_set_iterate(active_set)
        grad!(gradient, x)

        if away_steps
            if lazy
                d, vertex, index_in_activeset, gamma_max, phi_value, away_step_taken, fw_step_taken, tt =
                    FrankWolfe.lazy_afw_step(
                        x,
                        gradient,
                        lmo,
                        active_set,
                        phi_value,
                        epsilon,
                        d;
                        lazy_tolerance=lazy_tolerance,
                        memory_mode=memory_mode,
                    )
            else
                d, vertex, index_in_activeset, gamma_max, phi_value, away_step_taken, fw_step_taken, tt =
                    FrankWolfe.afw_step(x, gradient, lmo, active_set, epsilon, d, memory_mode=memory_mode)
            end
        else
            error("Pure FW not supported at the moment")
            d, vertex, index_in_activeset, gamma_max, phi_value, away_step_taken, fw_step_taken, tt =
                FrankWolfe.fw_step(x, gradient, lmo, d, memory_mode=memory_mode)
        end

        if fw_step_taken || away_step_taken
            gamma = FrankWolfe.perform_line_search(
                line_search,
                t,
                f,
                grad!,
                gradient,
                x,
                d,
                gamma_max,
                linesearch_workspace,
                memory_mode,
            )
            gamma = min(gamma_max, gamma)
            if index_in_activeset <= 0
                index_in_activeset = FrankWolfe.find_atom(active_set, vertex)
            else
                @assert active_set.atoms[index_in_activeset] == vertex
            end
            index_in_matrix = if index_in_activeset > 0
                lookup_indices[index_in_activeset]
            else
                0
            end
            if away_step_taken
                nfactors += away_update!(M, index_in_matrix, λ, gamma, F)
                _update_activeset_weights!(active_set, lookup_indices, λ)
                @assert FrankWolfe.active_set_validate(active_set)
            elseif fw_step_taken
                is_new_vertex = index_in_matrix == 0
                index_in_matrix, nrefactor = fw_update!(M, F, vertex, λ, gamma, index_in_matrix, full_solve=full_solve)
                nfactors += nrefactor
                if is_new_vertex
                    FrankWolfe.active_set_update!(active_set, gamma, vertex, false, nothing)
                    index_in_activeset = length(active_set)
                    push!(lookup_indices, index_in_matrix)
                else
                    FrankWolfe.active_set_update!(active_set, gamma, vertex, false, index_in_activeset)
                end
                _update_activeset_weights!(active_set, lookup_indices, λ)
            end
        end

        if callback !== nothing
            state = FrankWolfe.CallbackState(
                t,
                primal,
                primal - dual_gap,
                dual_gap,
                tot_time,
                x,
                vertex,
                d,
                gamma,
                f,
                grad!,
                lmo,
                gradient,
                tt,
            )
            if callback(state, active_set, nfactors) === false
                break
            end
        end
        primal = f(x)
        dual_gap = phi_value
    end

    # recompute everything once more for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # do also cleanup of active_set due to many operations on the same set

    x = FrankWolfe.get_active_set_iterate(active_set)
    grad!(gradient, x)
    v = FrankWolfe.compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    tt = FrankWolfe.last
    tot_time = (time_ns() - time_start) / 1e9
    if callback !== nothing
        state = FrankWolfe.CallbackState(
            t,
            primal,
            primal - dual_gap,
            dual_gap,
            tot_time,
            x,
            v,
            nothing,
            gamma,
            f,
            grad!,
            lmo,
            gradient,
            tt,
        )
        callback(state, active_set, nfactors)
    end

    FrankWolfe.active_set_renormalize!(active_set)
    FrankWolfe.active_set_cleanup!(active_set)
    x = FrankWolfe.get_active_set_iterate(active_set)
    grad!(gradient, x)
    if recompute_last_vertex
        v = FrankWolfe.compute_extreme_point(lmo, gradient)
        primal = f(x)
        dual_gap = FrankWolfe.fast_dot(x, gradient) - FrankWolfe.fast_dot(v, gradient)
    end
    tt = FrankWolfe.pp
    tot_time = (time_ns() - time_start) / 1e9
    if callback !== nothing
        state = FrankWolfe.CallbackState(
            t,
            primal,
            primal - dual_gap,
            dual_gap,
            tot_time,
            x,
            v,
            nothing,
            gamma,
            f,
            grad!,
            lmo,
            gradient,
            tt,
        )
        callback(state, active_set, nfactors)
    end

    return x, v, primal, dual_gap, traj_data, active_set
end

"""
Transfers the weights from λ to the active set, updates the lookup_indices correspondingly
"""
function _update_activeset_weights!(active_set, lookup_indices, λ; zero_threshold=1e-12, weight_simplex_tol=1e-8, remove_zeros=true)
    @assert abs(sum(λ) - 1) < weight_simplex_tol "$(sum(λ))"
    # new weights in λ, update active set
    # while loop keeps track of deleted vertices
    idx_as = 1
    active_set.x .= 0
    while idx_as <= length(active_set)
        new_weight = λ[lookup_indices[idx_as]]
        if new_weight >= zero_threshold
            active_set.weights[idx_as] = new_weight
            _add!(active_set.x, new_weight, active_set.atoms[idx_as])
            idx_as += 1
        elseif remove_zeros
            deleteat!(active_set, idx_as)
            deleteat!(lookup_indices, idx_as)
        else
            idx_as += 1
        end
    end
    FrankWolfe.active_set_renormalize!(active_set)
    FrankWolfe.compute_active_set_iterate!(active_set)
end

function _add!(x, λ, v)
    @. x += λ * v
end

function _add!(x, λ, v::SparseArrays.SparseVector)
    nzvals = SparseArrays.nonzeros(v)
    nzinds = SparseArrays.nonzeroinds(v)
    @inbounds for idx in eachindex(nzvals)
        x[nzinds[idx]] += λ * nzvals[idx]
    end
end

function _add!(x, λ, v::FrankWolfe.ScaledHotVector)
    x[v.val_idx] += λ * v.active_val
end

function make_trajectory_with_active_set(traj_vector)
    function callback(state, args...)
        if state.tt !== last || state.tt !== pp
            res = (FrankWolfe.callback_state(state)..., length(args[1]))
            push!(traj_vector, res)
        end
        return true
    end
end
