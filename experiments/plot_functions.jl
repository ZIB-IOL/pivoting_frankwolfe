using Plots
using LaTeXStrings

function plot_sparsity(data, label; filename=nothing, xscalelog=false, legend_position=:topright, lstyle=fill(:solid, length(label)),width=1.5,shift=true)
    # theme(:dark)
    # theme(:vibrant)
    gr()

    x = []
    y = []
    ps = nothing
    ds = nothing
    offset = 2
    xscale = xscalelog ? :log : :identity
    for i in eachindex(data)
        trajectory = data[i]
        x = [trajectory[j][6] for j in offset:length(trajectory)]
        y = [trajectory[j][2] for j in offset:length(trajectory)]
        if shift
            y .-= (minimum(y) - 1e-5)
        end
        if i == 1
            ps = plot(
                x,
                y,
                label=label[i],
                xaxis=xscale,
                yaxis=:log,
                ylabel=L"f(\textbf{x})",
                legend=legend_position,
                linestyle=lstyle[i],
                yguidefontsize=12,
                xguidefontsize=12,
                legendfontsize=8,
                width=width,
            )
        else
            plot!(x, y, label=label[i], linestyle=lstyle[i], width=width)
        end
    end
    for i in eachindex(data)
        trajectory = data[i]
        x = [trajectory[j][6] for j in offset:length(trajectory)]
        y = [trajectory[j][4] for j in offset:length(trajectory)]
        if i == 1
            ds = plot(
                x,
                y,
                label=label[i],
                legend=false,
                linestyle=lstyle[i],
                xaxis=xscale,
                yaxis=:log,
                ylabel=L"\textrm{Dual\ gap}",
                yguidefontsize=12,
                xguidefontsize=12,
                width=width,
            )
        else
            plot!(x, y, label=label[i], linestyle=lstyle[i], width=width)
        end
    end
    
    fp = plot(ps, ds, layout=(1, 2)) # layout = @layout([A{0.01h}; [B C; D E]]))
    xlabel!(fp, L"\textrm{Active\ set\ cardinality}")
    plot!(size=(600, 270))
    if filename !== nothing
        savefig(fp, filename)
    end
    return fp
end


function plot_trajectories(
    data,
    label;
    filename=nothing,
    xscalelog=false,
    legend_position=:topright,
    lstyle=fill(:solid, length(data)),
    width=1.5,
    shift=true,
)
    # theme(:dark)
    # theme(:vibrant)
    Plots.gr()

    x = []
    y = []
    pit = nothing
    pti = nothing
    dit = nothing
    dti = nothing
    offset = 2
    xscale = xscalelog ? :log : :identity
    for i in 1:length(data)
        trajectory = data[i]
        x = [trajectory[j][1] for j in offset:length(trajectory)]
        y = [trajectory[j][2] for j in offset:length(trajectory)]
        if shift
            y .-= (minimum(y) - 1e-5)
        end
        if i == 1
            pit = plot(
                x,
                y,
                label=label[i],
                xaxis=xscale,
                yaxis=:log,
                ylabel=L"f(\textbf{x})",
                legend=legend_position,
                yguidefontsize=12,
                xguidefontsize=12,
                legendfontsize=8,
                width=width,
                linestyle=lstyle[i],
            )
        else
            plot!(x, y, label=label[i], width=width, linestyle=lstyle[i])
        end
    end
    for i in 1:length(data)
        trajectory = data[i]
        x = [trajectory[j][5] for j in offset:length(trajectory)]
        y = [trajectory[j][2] for j in offset:length(trajectory)]
        if shift
            y .-= (minimum(y) - 1e-5)
        end
        if i == 1
            pti = plot(
                x,
                y,
                label=label[i],
                legend=false,
                xaxis=xscale,
                yaxis=:log,
                yguidefontsize=12,
                xguidefontsize=12,
                width=width,
                linestyle=lstyle[i],
            )
        else
            plot!(x, y, label=label[i], width=width, linestyle=lstyle[i])
        end
    end
    for i in 1:length(data)
        trajectory = data[i]
        x = [trajectory[j][1] for j in offset:length(trajectory)]
        y = [trajectory[j][4] for j in offset:length(trajectory)]
        if i == 1
            dit = plot(
                x,
                y,
                label=label[i],
                legend=false,
                xaxis=xscale,
                yaxis=:log,
                ylabel=L"\textrm{Dual\ gap}",
                xlabel=L"\textrm{Iterations}",
                yguidefontsize=12,
                xguidefontsize=12,
                width=width,
                linestyle=lstyle[i],
            )
        else
            plot!(x, y, label=label[i], width=width, linestyle=lstyle[i])
        end
    end
    for i in 1:length(data)
        trajectory = data[i]
        x = [trajectory[j][5] for j in offset:length(trajectory)]
        y = [trajectory[j][4] for j in offset:length(trajectory)]
        if i == 1
            dti = plot(
                x,
                y,
                label=label[i],
                legend=false,
                xaxis=xscale,
                yaxis=:log,
                xlabel=L"\textrm{Time}",
                yguidefontsize=12,
                xguidefontsize=12,
                width=width,
                linestyle=lstyle[i],
            )
        else
            plot!(x, y, label=label[i], width=width, linestyle=lstyle[i])
        end
    end
    fp = plot(pit, pti, dit, dti, layout=(2, 2)) # layout = @layout([A{0.01h}; [B C; D E]]))
    plot!(size=(600, 400))
    if filename !== nothing
        savefig(fp, filename)
    end
    return fp
end
