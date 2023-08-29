using Plots
using FrankWolfe
using JSON
using LaTeXStrings

include(joinpath(@__DIR__, "plot_functions.jl"))

all_results = filter!(s -> endswith(s, "json"), readdir(joinpath(@__DIR__, "results/"), join=true))

const plot_dir = joinpath(@__DIR__, "plots/")
if !isdir(plot_dir)
    mkdir(plot_dir)
end

const alg_names = ["afw", "bpcg", "pafw", "pbpcg"]
label_dict = Dict("bpcg" => L"\texttt{BPFW}", "afw" => L"\texttt{AFW}", "pafw" => L"\texttt{P-AFW}", "pbpcg" => L"\texttt{P-BPFW}")
for (k,v) in label_dict
    label_dict[k * "_nlazy"] = v
end

const l_styles = [:solid, :dashdot, :dashdotdot, :dot]
colors = ["purple", "green", "DeepSkyBlue", "brown"]

const alg_names_nlazy = [s * "_nlazy" for s in alg_names]

const alg_names_bpcg = [s for s in vcat(alg_names, alg_names_nlazy) if occursin("bpcg", s)]
sort!(alg_names_bpcg, by=s -> count('p', s))
const alg_names_afw = [s for s in vcat(alg_names, alg_names_nlazy) if occursin("afw", s)]
sort!(alg_names_afw, by=s -> count('p', s))

const alg_names_afw_lazy = ["afw", "pafw"]

d_colors = distinguishable_colors(8, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
colors_dict = Dict(alg => d_colors[i] for (i, alg) in enumerate(vcat(alg_names, alg_names_nlazy)))

for res_file in all_results
    base_name = split(basename(res_file), '.')[1]
    file_string = open(res_file) do f
        read(f, String)
    end
    @info base_name
    if isempty(file_string)
        @info "empty file"
        continue
    end
    res_dict = JSON.parse(file_string)
    for (alg_names, lazy_string) in [(alg_names, ""), (alg_names_nlazy, "_nlazy"), (alg_names_bpcg, "_bpcg"), ((alg_names_afw, "_afw")), (alg_names_afw_lazy, "afw_lazy")]
        if occursin("birkhoff", base_name) && any(s -> occursin("nlazy", s), alg_names)
            continue
        end
        trajectories = [res_dict[alg_name] for alg_name in alg_names]
        for idx in eachindex(trajectories)
            if length(trajectories[idx]) > 800
                remove_each = length(trajectories[idx]) ÷ 200
                @info remove_each
                trajectories[idx] = trajectories[idx][1:remove_each:end]
            end
        end
        label_names = map(alg_names) do alg_name
            if !occursin("nlazy", alg_name) # && lazy_string ∈ ("_bpcg", "_afw")
                L"\texttt{L}-" * label_dict[alg_name]
            else
                label_dict[alg_name]
            end
        end
        colors = [colors_dict[alg_name] for alg_name in alg_names]
        p_traj, p_traj_primal, p_traj_dual = plot_trajectories(trajectories, label_names, legend_position=:topright, lstyle=l_styles, width=1.2, colors=colors)
        savefig(p_traj, joinpath(plot_dir, base_name * lazy_string * "_trajectory.pdf"))
        savefig(p_traj_primal, joinpath(plot_dir, base_name * lazy_string * "_primal_trajectory.pdf"))
        savefig(p_traj_dual, joinpath(plot_dir, base_name * lazy_string * "_dual_trajectory.pdf"))
        p_sparsity, p_sparsity_dual = plot_sparsity(trajectories, label_names, legend_position=:topright, lstyle=l_styles, width=1.2, colors=colors)
        savefig(p_sparsity, joinpath(plot_dir, base_name * lazy_string * "_sparsity.pdf"))
        savefig(p_sparsity_dual, joinpath(plot_dir, base_name * lazy_string * "_sparsity_dual.pdf"))
    end
end
