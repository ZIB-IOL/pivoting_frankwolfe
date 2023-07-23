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

const alg_names = ["bpcg", "afw", "cg_afw"]
label_dict = Dict("bpcg" => L"\texttt{BPFW}", "afw" => L"\texttt{AFW}", "cg_afw" => L"\texttt{CGM-AFW}")
const l_styles = [:solid, :dash, :dot]

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
    trajectories = [res_dict[alg_name] for alg_name in alg_names]
    label_names = [label_dict[alg_name] for alg_name in alg_names]
    p_traj = plot_trajectories(trajectories, label_names, legend_position=:topright, lstyle=l_styles, width=1.8)
    savefig(p_traj, joinpath(plot_dir, base_name * "_trajectory.pdf"))
    p_sparsity = plot_sparsity(trajectories, label_names, legend_position=:topright, lstyle=l_styles, width=1.8)
    savefig(p_sparsity, joinpath(plot_dir, base_name * "_sparsity.pdf"))
end
