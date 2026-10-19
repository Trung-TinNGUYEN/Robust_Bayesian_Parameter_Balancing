using CairoMakie
using Statistics
using Distributions
using DataFrames
using KernelDensity
using LaTeXStrings


include(joinpath(@__DIR__, "plot_font_constants.jl"))

# Interval utilities

"""
    ci_interval_from_samples(x, level)

Compute the equal-tailed credible interval from sorted samples at the given nominal level.

# Arguments
- `x`: Vector of samples (will be sorted internally).
- `level`: Nominal coverage level, e.g. `0.95`.

# Returns
A 2-tuple `(lo, hi)` of `Float64` interval endpoints.
"""
function ci_interval_from_samples(x::AbstractVector{<:Real}, level::Real)
    xs = sort(Float64.(x))
    m = length(xs)
    a = 1.0 - float(level)
    lo = xs[clamp(Int(ceil(a / 2 * m)), 1, m)]
    hi = xs[clamp(Int(ceil((1.0 - a / 2) * m)), 1, m)]
    return lo, hi
end

"""
    hdi_interval_from_samples(x, level)

Compute the Highest Density Interval (HDI) from sorted samples at the given nominal level.

Finds the shortest contiguous interval that contains at least `floor(level * m)` of the
`m` sorted samples. Returns `(NaN, NaN)` if `x` is empty.

# Arguments
- `x`: Vector of samples (will be sorted internally).
- `level`: Nominal coverage level, e.g. `0.95`.

# Returns
A 2-tuple `(lo, hi)` of `Float64` interval endpoints.
"""
function hdi_interval_from_samples(x::AbstractVector{<:Real}, level::Real)
    xs = sort(Float64.(x))
    m = length(xs)
    m == 0 && return NaN, NaN
    k = clamp(Int(floor(level * m)), 1, m)
    best_lo = 1
    best_w = Inf
    @inbounds for i in 1:(m - k + 1)
        w = xs[i + k - 1] - xs[i]
        if w < best_w
            best_w = w
            best_lo = i
        end
    end
    return xs[best_lo], xs[best_lo + k - 1]
end

"""
    binom_accept_band(n, p0; alpha)

Compute a two-sided binomial acceptance band around nominal proportion `p0` for a sample
of size `n`.

# Arguments
- `n`: Sample size (returns `(NaN, NaN)` if `n <= 0`).
- `p0`: Nominal proportion.

# Keywords
- `alpha`: Significance level for the band (default `0.05`).

# Returns
A 2-tuple `(lo, hi)` of proportion-scale bounds.
"""
function binom_accept_band(n::Int, p0::Float64; alpha::Float64 = 0.05)
    n <= 0 && return (NaN, NaN)
    d = Binomial(n, p0)
    return (quantile(d, alpha / 2) / n, quantile(d, 1 - alpha / 2) / n)
end

# Problematic parameter identification

"""
    identify_problematic_params(widths_mat; kwargs...) -> BitVector

Flag parameters (columns of `widths_mat`) whose 95 % posterior interval width is
unusually large.  A parameter is flagged **if, for any fitted model**, its width
exceeds the `width_quantile`-th quantile of that model's width distribution
(default: the top 10 %, i.e. `width_quantile = 0.90`).

Parameters flagged by this criterion are highlighted in **red bold** on heatmap
x-axis labels in the paper figures.  Caption text should read, for example:
"Red labels mark parameters whose interval width falls in the top 10 % of all
widths for at least one fitted model."
"""

function identify_problematic_params(
    widths_mat::Matrix{Float64};
    coverage_mat::Union{Matrix{Float64}, Nothing} = nothing,
    nominal_level::Float64 = 0.95,
    width_quantile::Float64 = 0.90,
    coverage_alpha::Float64 = 0.05,
    n_folds::Int = 0
)
    d = size(widths_mat, 2)
    n_models = size(widths_mat, 1)
    flags = falses(d)

    for k in 1:n_models
        w_k = widths_mat[k, :]
        w_finite = filter(isfinite, w_k)
        isempty(w_finite) && continue
        thresh = quantile(w_finite, width_quantile)
        for j in 1:d
            w_k[j] > thresh && (flags[j] = true)
        end
    end

    if coverage_mat !== nothing && n_folds > 0
        p0 = nominal_level
        z_crit = quantile(Normal(), 1.0 - coverage_alpha / 2)
        se = sqrt(p0 * (1.0 - p0) / n_folds)
        for j in 1:d
            abs(coverage_mat[j, 1] - p0) > z_crit * se && (flags[j] = true)
        end
    end
    return flags
end

# Model label and style helpers

"""
    _short_label(res)

Return a compact display label for a CV result, e.g. `"Gaussian RE"` or `"Student-t Fixed"`.

Combines the likelihood family name (`Gaussian`, `Student-t`, or `Skew-N`) with a
random-effects suffix (`RE` if `res.has_random_effects`, otherwise `Fixed`).
"""
function _short_label(res)
    m = res.model == :gaussian ? "Gaussian" :
        res.model == :t ? "Student-t" :
        res.model == :skew ? "Skew-N" : titlecase(string(res.model))
    re = res.has_random_effects ? "RE" : "Fixed"
    return "$m $re"
end

_short_label(res, ::Symbol) = _short_label(res)

"""
    ModelStyle

Holds the visual encoding (color, marker, linestyle, and sizes) for one model configuration.

Fixed-effects models use a square marker and dashed line; random-effects models
use a circle marker and solid line. Color is determined by likelihood family:
Gaussian maps to steelblue, Student-t maps to firebrick, and skew-normal maps
to forestgreen.
"""
# Visual encoding per team spec:
#   Fixed effects:  square marker, dashed line
#   Random effects: circle marker, solid line
#   Colour by likelihood family: gaussian=steelblue, t=firebrick, skew=forestgreen
struct ModelStyle
    color::Any
    marker::Symbol
    linestyle::Symbol
    linewidth::Float64
    markersize::Int
    strokewidth::Float64
    strokecolor::Any
    label::String
end

"""
    _model_style(res)

Return a `ModelStyle` for the given CV result, encoding color by likelihood family and
marker/linestyle by whether random effects are present.
"""
function _model_style(res)
    lbl  = _short_label(res)
    m    = res.model
    is_re = res.has_random_effects
    c = m == :gaussian ? :steelblue :
        m == :t        ? :firebrick :
        m == :skew     ? :forestgreen : :gray50

    # Fixed = square (:rect) + dashed line; RE = circle + solid line (consistent across all families)
    if is_re
        ModelStyle(c, :circle, :solid, 2.5, 12, 0.0, c, lbl)
    else
        ModelStyle((c, 0.75), :rect, :dash, 1.8, 10, 1.5, c, lbl)
    end
end

"""
    _plot_styled_curve!(ax, xvals, yvals, sty)

Plot a lines-plus-scatter curve onto `ax` using the color, linestyle, marker, and size
attributes stored in `sty`.
"""
function _plot_styled_curve!(ax, xvals, yvals, sty::ModelStyle)
    lines!(ax, xvals, yvals; color = sty.color, linewidth = sty.linewidth, linestyle = sty.linestyle)
    scatter!(ax, xvals, yvals; color = sty.color, marker = sty.marker, markersize = sty.markersize,
             strokewidth = sty.strokewidth, strokecolor = sty.strokecolor)
end

"""
    _legend_element(sty)

Return a two-element vector `[LineElement, MarkerElement]` so the legend entry shows both
the line style and the marker shape encoded in `sty`.
"""
# Returns a [LineElement, MarkerElement] pair so the legend shows both line style and marker shape.
function _legend_element(sty::ModelStyle)
    [
        LineElement(color = sty.color, linestyle = sty.linestyle, linewidth = sty.linewidth),
        MarkerElement(color = sty.color, marker = sty.marker, markersize = sty.markersize,
                      strokewidth = sty.strokewidth, strokecolor = sty.strokecolor)
    ]
end

# Ridgeline plot

"""
    plot_ridgeline_comparison(results, metric; title, trim_quantile, outpath)

Produce a ridgeline (stacked density) plot of `metric` across all models in `results`.

Each model is shown as a kernel density estimate offset along the y-axis, colored by
likelihood family and shaded by whether random effects are present. Non-PIT metrics are
trimmed to `trim_quantile` before plotting. If provided, saves the figure to `outpath`.
"""
function plot_ridgeline_comparison(results::Dict{Symbol, CVResult}, metric::Symbol = :pit;
    title::String = "Distribution Comparison",
    trim_quantile::Float64 = 0.99,
    outpath::Union{String, Nothing} = nothing
)
    sorted_keys = sort(collect(keys(results)))
    n_models = length(sorted_keys)
    data_map = Dict()
    max_val = -Inf

    for key in sorted_keys
        res = results[key]
        vals = filter(isfinite, Float64.(getfield(res, metric)))
        if metric != :pit
            q = quantile(vals, trim_quantile)
            vals = vals[vals .<= q]
        end
        data_map[key] = vals
        max_val = max(max_val, maximum(vals))
    end
    metric == :pit && (max_val = 1.0)

    fig = Figure(size = (800, 600), fontsize = FONT_BASE)
    ax = Axis(fig[1, 1],
        xlabel = string(metric), ylabel = "Model",
        xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL,
        yticks = (1:n_models, [_short_label(results[k]) for k in sorted_keys]),
        title = title, titlesize = FONT_TITLE,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL,
        xgridvisible = false, ygridvisible = false,
        leftspinevisible = false, topspinevisible = false, rightspinevisible = false)

    colors = Dict(:gaussian => :blue, :t => :red, :skew => :green)
    for (i, key) in enumerate(sorted_keys)
        res = results[key]
        density!(ax, data_map[key],
            offset = i, color = (colors[res.model], res.has_random_effects ? 0.6 : 0.3),
            strokewidth = 1.5, strokecolor = colors[res.model],
            bandwidth = metric == :pit ? 0.05 : 0.2, direction = :x)
    end
    xlims!(ax, 0, max_val * 1.05)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

# Multilevel coverage

"""
    plot_multilevel_comparison(results; outpath, title)

Plot empirical vs. nominal coverage across multiple coverage levels for each model.

Draws a diagonal reference line and one styled curve per model. If provided, saves the
figure to `outpath`.
"""
function plot_multilevel_comparison(results::Dict{Symbol, CVResult};
    outpath::Union{String, Nothing} = nothing,
    title::String = "Multi-level Coverage Comparison"
)
    fig = Figure(size = (800, 600), fontsize = FONT_BASE)
    ax = Axis(fig[1, 1],
        xlabel = "Nominal level", ylabel = "Empirical coverage",
        xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL,
        title = title, titlesize = FONT_TITLE,
        xticks = 0:0.1:1, yticks = 0:0.1:1,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL)

    lines!(ax, [0, 1], [0, 1], linestyle = :dash, color = :black, linewidth = 1)

    seen = Set{String}()
    leg_e = Any[LineElement(color = :black, linestyle = :dash, linewidth = 1)]
    leg_l = ["Ideal"]

    for key in sort(collect(keys(results)))
        res = results[key]
        sty = _model_style(res)
        _plot_styled_curve!(ax, res.nominal_levels, vec(mean(res.pred_coverage, dims = 1)), sty)
        if !(sty.label in seen)
            push!(seen, sty.label)
            push!(leg_e, _legend_element(sty))
            push!(leg_l, sty.label)
        end
    end

    xlims!(ax, 0, 1.02); ylims!(ax, 0, 1.02)
    Legend(fig[1, 2], leg_e, leg_l, framevisible = true, labelsize = FONT_LEGEND,
           tellwidth = true, tellheight = false, padding = (6, 6, 4, 4), rowgap = 2)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

# PIT / CRPS / WIS comparison plots

"""
    plot_pit_comparison(results; outpath)

Plot overlaid PIT density estimates for each model in `results`, with a dashed reference
line at density 1 (uniform). If provided, saves the figure to `outpath`.
"""
function plot_pit_comparison(results::Dict{Symbol, CVResult}; outpath::Union{String, Nothing} = nothing)
    fig = Figure(size = (700, 500), fontsize = FONT_BASE)
    ax = Axis(fig[1, 1], xlabel = "PIT", ylabel = "Density", title = "PIT Distribution",
        xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL, titlesize = FONT_TITLE,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL)
    hlines!(ax, [1.0], linestyle = :dash, color = :black)
    colors = Dict(:gaussian => :blue, :t => :red, :skew => :green)
    for key in sort(collect(keys(results)))
        res = results[key]
        density!(ax, res.pit, color = (colors[res.model], res.has_random_effects ? 0.6 : 0.3),
                strokewidth = 2, strokecolor = colors[res.model], label = _short_label(res))
    end
    axislegend(ax, position = :rt, nbanks = 2, labelsize = FONT_LEGEND)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

"""
    plot_crps_comparison(results; trim_quantile, outpath)

Plot overlaid CRPS density estimates for each model in `results`, trimmed to
`trim_quantile` to suppress long tails. If provided, saves the figure to `outpath`.
"""
function plot_crps_comparison(results::Dict{Symbol, CVResult};
    trim_quantile::Float64 = 0.99, outpath::Union{String, Nothing} = nothing)
    max_crps = maximum([quantile(filter(isfinite, res.crps), trim_quantile) for res in values(results)])
    fig = Figure(size = (700, 500), fontsize = FONT_BASE)
    ax = Axis(fig[1, 1], xlabel = "CRPS", ylabel = "Density", title = "CRPS Distribution",
        xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL, titlesize = FONT_TITLE,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL)
    colors = Dict(:gaussian => :blue, :t => :red, :skew => :green)
    for key in sort(collect(keys(results)))
        res = results[key]
        crps_clean = filter(isfinite, res.crps)
        crps_trim = crps_clean[crps_clean .<= max_crps]
        density!(ax, crps_trim, color = (colors[res.model], res.has_random_effects ? 0.6 : 0.3),
                strokewidth = 2, strokecolor = colors[res.model], label = _short_label(res))
    end
    xlims!(ax, 0, max_crps * 1.1)
    axislegend(ax, position = :rt, nbanks = 2, labelsize = FONT_LEGEND)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

"""
    plot_wis_comparison(results; trim_quantile, outpath)

Plot overlaid WIS density estimates for each model in `results`, trimmed to
`trim_quantile` to suppress long tails. If provided, saves the figure to `outpath`.
"""
function plot_wis_comparison(results::Dict{Symbol, CVResult};
    trim_quantile::Float64 = 0.99, outpath::Union{String, Nothing} = nothing)
    max_wis = maximum([quantile(filter(isfinite, res.wis), trim_quantile) for res in values(results)])
    fig = Figure(size = (700, 500), fontsize = FONT_BASE)
    ax = Axis(fig[1, 1], xlabel = "WIS", ylabel = "Density", title = "WIS Distribution",
        xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL, titlesize = FONT_TITLE,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL)
    colors = Dict(:gaussian => :blue, :t => :red, :skew => :green)
    for key in sort(collect(keys(results)))
        res = results[key]
        wis_trim = res.wis[res.wis .<= max_wis]
        density!(ax, wis_trim, color = (colors[res.model], res.has_random_effects ? 0.6 : 0.3),
                strokewidth = 2, strokecolor = colors[res.model], label = _short_label(res))
    end
    xlims!(ax, 0, max_wis * 1.1)
    axislegend(ax, position = :rt, nbanks = 2, labelsize = FONT_LEGEND)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

# Group coverage comparison

"""
    plot_group_coverage_comparison(results; selected_keys, level, group_col, alpha_band, outpath)

Plot per-group empirical coverage at a single nominal `level`, with groups sorted by size
and a binomial acceptance band shown as a shaded region.

Each model is shown as a scatter series. Requires `obs_metadata` to be present in each
`CVResult`. If provided, saves the figure to `outpath`.
"""
function plot_group_coverage_comparison(results::Dict{Symbol, CVResult};
    selected_keys::Union{Vector{Symbol}, Nothing} = nothing,
    level::Float64 = 0.95, group_col::Symbol = :qrow,
    alpha_band::Float64 = 0.05, outpath::Union{String, Nothing} = nothing
)
    all_keys = sort(collect(keys(results)))
    ktp = isnothing(selected_keys) ? all_keys : intersect(selected_keys, all_keys)
    isempty(ktp) && error("No matching keys")
    first_res = results[ktp[1]]
    df_first = first_res.obs_metadata
    isnothing(df_first) && error("Metadata required")

    actual_col = if hasproperty(df_first, group_col)
        group_col
    elseif hasproperty(df_first, :qidx)
        :qidx
    elseif hasproperty(df_first, :Q_row)
        :Q_row
    elseif hasproperty(df_first, :param_idx)
        :param_idx
    else
        error("Grouping column missing")
    end

    idx_lev = findfirst(approx(level), first_res.nominal_levels)
    ugroups = unique(df_first[!, actual_col])
    G = length(ugroups)
    g_n = [count(df_first[!, actual_col] .== g) for g in ugroups]
    ord = sortperm(g_n)
    g_n_sorted = g_n[ord]
    g_lo = [binom_accept_band(n, level; alpha = alpha_band)[1] for n in g_n_sorted]
    g_hi = [binom_accept_band(n, level; alpha = alpha_band)[2] for n in g_n_sorted]

    fig = Figure(size = (1000, 500), fontsize = FONT_BASE)
    ax = Axis(fig[1, 1],
        xlabel = "Groups (sorted by size)", ylabel = "Coverage",
        xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL,
        title = "Group Coverage @ $(Int(round(100 * level)))%",
        titlesize = FONT_TITLE,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL)

    x = 1:G
    band!(ax, x, g_lo, g_hi, color = (:gray, 0.2), label = "Binomial Band")
    hlines!(ax, [level], linestyle = :dash, color = :black, label = "Nominal")

    base_colors = Dict(:gaussian => :blue, :t => :red, :skew => :green)
    markers_map = Dict(true => :circle, false => :rect)
    for key in ktp
        res = results[key]
        raw_cov = res.pred_coverage[:, idx_lev]
        df_curr = res.obs_metadata
        g_cov = [mean(raw_cov[df_curr[!, actual_col] .== g]) for g in ugroups]
        c_base = base_colors[res.model]
        c_final = res.has_random_effects ? c_base : (c_base, 0.5)
        scatter!(ax, x, g_cov[ord],
            color = c_final, marker = markers_map[res.has_random_effects],
            markersize = res.has_random_effects ? 10 : 12,
            strokewidth = 1, strokecolor = c_base, label = _short_label(res))
    end
    ylims!(ax, 0.0, 1.05); xlims!(ax, 0, G + 1)
    axislegend(ax, position = :rb, labelsize = FONT_LEGEND)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

# Coverage-width heatmap with problematic parameter identification

"""
    plot_coverage_width_heatmap(results; level, outpath)

Display a heatmap of posterior interval widths (models × parameters) at the given nominal
`level`, with parameter labels colored red and bold for parameters flagged by
`identify_problematic_params`. If provided, saves the figure to `outpath`.
"""
function plot_coverage_width_heatmap(results::Dict{Symbol, CVResult};
    level::Float64 = 0.95, outpath::Union{String, Nothing} = nothing)
    first_res = first(values(results))
    param_names = first_res.param_names
    d = length(param_names)
    idx_lev = findfirst(x -> isapprox(x, level, atol = 1e-5), first_res.nominal_levels)

    sorted_keys = sort(collect(keys(results)))
    widths_mat = zeros(length(sorted_keys), d)
    for (i, key) in enumerate(sorted_keys)
        widths_mat[i, :] = results[key].interval_width[:, idx_lev]
    end

    cov_mat = nothing; n_folds_cov = 0
    for key in sorted_keys
        res = results[key]
        if res.coverage_CI !== nothing && any(res.coverage_CI .> 0)
            ci = findfirst(x -> isapprox(x, level, atol = 1e-5), res.nominal_levels)
            if ci !== nothing
                cov_mat = reshape(res.coverage_CI[:, ci], :, 1)
                n_folds_cov = res.n_folds
                break
            end
        end
    end

    is_prob = identify_problematic_params(widths_mat;
        coverage_mat = cov_mat, nominal_level = level, n_folds = n_folds_cov)
    model_labels = [_short_label(results[k]) for k in sorted_keys]

    fig = Figure(size = (1400, 700), fontsize = FONT_BASE, figure_padding = (10, 10, 10, 120))
    ax = Axis(fig[1, 1],
        xlabel = "Parameters", ylabel = "Models",
        xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL,
        title = "Interval Width @ $(Int(100 * level))%",
        titlesize = FONT_TITLE,
        xticks = (1:d, param_names), yticks = (1:length(sorted_keys), model_labels),
        xticklabelrotation = pi / 3, xticklabelsize = FONT_HEATMAP_TXT,
        yticklabelsize = FONT_TICKLABEL)
    hm = heatmap!(ax, 1:d, 1:length(sorted_keys), widths_mat', colormap = :viridis)
    Colorbar(fig[1, 2], hm, label = "Width", labelsize = FONT_XLABEL)

    prob_idxs = findall(is_prob)
    !isempty(prob_idxs) && scatter!(ax, Float64.(prob_idxs), fill(0.35, length(prob_idxs)),
        marker = :dtriangle, markersize = 8, color = :red)

    ax.xticklabelsvisible = false
    for (j, name) in enumerate(param_names)
        c = is_prob[j] ? :red : :black
        fw = is_prob[j] ? :bold : :regular
        text!(ax, j, 0.0; text = name, rotation = pi / 2, align = (:right, :center),
              fontsize = FONT_HEATMAP_TXT, color = c, font = fw)
    end
    ylims!(ax, -2.5, length(sorted_keys) + 0.5)
    xlims!(ax, 0.5, d + 0.5)

    !isnothing(outpath) && save(outpath, fig)
    return fig
end

# BRENDA combined panel

"""
    plot_brenda_panel(results; level, trim_quantile, outpath)

Produce a 2×3 combined figure with coverage, PIT, CRPS, WIS, and a width heatmap for all
models in `results`.

Panel layout: (row 1) multi-level coverage, PIT density, CRPS density;
(row 2) WIS density, interval-width heatmap spanning two columns.
If provided, saves the figure to `outpath`.
"""
function plot_brenda_panel(results::Dict{Symbol, CVResult};
    level::Float64 = 0.95, trim_quantile::Float64 = 0.99,
    outpath::Union{String, Nothing} = nothing)

    fig = Figure(size = (1600, 900), fontsize = FONT_BASE)
    Label(fig[0, 1:3], "BRENDA LOSO - Model Comparison", fontsize = FONT_SUPERTITLE, font = :bold)

    colors = Dict(:gaussian => :steelblue, :t => :firebrick, :skew => :forestgreen)
    markers_map = Dict(true => :circle, false => :utriangle)
    sorted_keys = sort(collect(keys(results)))

    # Coverage
    ax1 = Axis(fig[1, 1], xlabel = "Nominal", ylabel = "Empirical", title = "Coverage",
        xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL, titlesize = FONT_TITLE,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL)
    lines!(ax1, [0, 1], [0, 1], linestyle = :dash, color = :black)
    for key in sorted_keys
        res = results[key]
        emp = vec(mean(res.pred_coverage, dims = 1))
        is_re = res.has_random_effects
        c_line = is_re ? colors[res.model] : (colors[res.model], 0.65)
        ls = is_re ? :solid : :dash
        lw = is_re ? 2.0 : 1.2
        lines!(ax1, res.nominal_levels, emp, color = c_line, linestyle = ls, linewidth = lw)
        scatter!(ax1, res.nominal_levels, emp, marker = markers_map[is_re],
                color = c_line, markersize = is_re ? 10 : 8, label = _short_label(res))
    end
    xlims!(ax1, 0, 1); ylims!(ax1, 0, 1)
    axislegend(ax1, position = :rb, nbanks = 2, merge = true, unique = true, labelsize = FONT_LEGEND - 2)

    # PIT
    ax2 = Axis(fig[1, 2], xlabel = "PIT", ylabel = "Density", title = "PIT",
        xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL, titlesize = FONT_TITLE,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL)
    hlines!(ax2, [1.0], linestyle = :dash, color = :black)
    for key in sorted_keys
        res = results[key]
        density!(ax2, res.pit, color = (colors[res.model], res.has_random_effects ? 0.6 : 0.3),
                strokewidth = 2, strokecolor = colors[res.model])
    end

    # CRPS
    max_crps = maximum([quantile(filter(isfinite, res.crps), trim_quantile) for res in values(results)])
    ax3 = Axis(fig[1, 3], xlabel = "CRPS", ylabel = "Density", title = "CRPS",
        xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL, titlesize = FONT_TITLE,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL)
    for key in sorted_keys
        res = results[key]
        crps_clean = filter(isfinite, res.crps)
        density!(ax3, crps_clean[crps_clean .<= max_crps],
                color = (colors[res.model], res.has_random_effects ? 0.6 : 0.3),
                strokewidth = 2, strokecolor = colors[res.model])
    end
    xlims!(ax3, 0, max_crps * 1.1)

    # WIS
    max_wis = maximum([quantile(filter(isfinite, res.wis), trim_quantile) for res in values(results)])
    ax4 = Axis(fig[2, 1], xlabel = "WIS", ylabel = "Density", title = "WIS",
        xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL, titlesize = FONT_TITLE,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL)
    for key in sorted_keys
        res = results[key]
        density!(ax4, res.wis[res.wis .<= max_wis],
                color = (colors[res.model], res.has_random_effects ? 0.6 : 0.3),
                strokewidth = 2, strokecolor = colors[res.model])
    end
    xlims!(ax4, 0, max_wis * 1.1)

    # Width heatmap
    first_res = first(values(results))
    d_param = length(first_res.param_names)
    idx_lev = findfirst(x -> isapprox(x, level, atol = 1e-5), first_res.nominal_levels)
    widths_mat = zeros(length(sorted_keys), d_param)
    for (i, key) in enumerate(sorted_keys)
        widths_mat[i, :] = results[key].interval_width[:, idx_lev]
    end
    is_prob = identify_problematic_params(widths_mat)
    model_labels = [_short_label(results[k]) for k in sorted_keys]

    ax5 = Axis(fig[2, 2:3],
        xlabel = "Parameters", ylabel = "Models",
        xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL,
        title = "Width @ $(Int(100 * level))%", titlesize = FONT_TITLE,
        xticks = (1:d_param, first_res.param_names),
        yticks = (1:length(sorted_keys), model_labels),
        xticklabelrotation = pi / 3, xticklabelsize = FONT_HEATMAP_TXT,
        yticklabelsize = FONT_TICKLABEL)
    hm = heatmap!(ax5, 1:d_param, 1:length(sorted_keys), widths_mat', colormap = :viridis)
    Colorbar(fig[2, 4], hm, label = "Width", labelsize = FONT_XLABEL)

    prob_js = findall(is_prob)
    !isempty(prob_js) && scatter!(ax5, Float64.(prob_js), fill(0.35, length(prob_js)),
        marker = :dtriangle, markersize = 7, color = :red)

    ax5.xticklabelsvisible = false
    for (j, name) in enumerate(first_res.param_names)
        c = is_prob[j] ? :red : :black
        fw = is_prob[j] ? :bold : :regular
        text!(ax5, j, 0.0; text = name, rotation = pi / 2, align = (:right, :center),
              fontsize = FONT_HEATMAP_TXT, color = c, font = fw)
    end
    ylims!(ax5, -2.5, length(sorted_keys) + 0.5)
    xlims!(ax5, 0.5, d_param + 0.5)

    !isnothing(outpath) && save(outpath, fig)
    return fig
end

# Orchestrators

"""
    plot_all_comparisons_synthetic(results; outdir)

Save the full suite of standard comparison plots for a synthetic-data experiment to `outdir`.

Calls `plot_multilevel_comparison`, three `plot_ridgeline_comparison` variants,
`plot_coverage_width_heatmap`, `plot_brenda_panel`, and `plot_group_coverage_comparison`,
writing numbered PDF files.
"""
function plot_all_comparisons_synthetic(results::Dict{Symbol, CVResult}; outdir::String)
    mkpath(outdir)
    plot_multilevel_comparison(results; outpath = joinpath(outdir, "01_multilevel.pdf"))
    plot_ridgeline_comparison(results, :pit; title = "PIT Distribution", outpath = joinpath(outdir, "02_pit.pdf"))
    plot_ridgeline_comparison(results, :crps; title = "CRPS Distribution", outpath = joinpath(outdir, "03_crps.pdf"))
    plot_ridgeline_comparison(results, :wis; title = "WIS Distribution", outpath = joinpath(outdir, "04_wis.pdf"))
    plot_coverage_width_heatmap(results; outpath = joinpath(outdir, "05_width_heatmap.pdf"))
    plot_brenda_panel(results; outpath = joinpath(outdir, "06_combined_panel.pdf"))
    plot_group_coverage_comparison(results; outpath = joinpath(outdir, "07_group_coverage.pdf"))
    println("Synthetic plots saved to $outdir")
end

"""
    plot_all_comparisons_brenda(results; outdir)

Save the full suite of standard comparison plots for a BRENDA LOSO experiment to `outdir`.

Identical plot sequence to `plot_all_comparisons_synthetic`; separated so the printed
progress message reflects the correct experiment name.
"""
function plot_all_comparisons_brenda(results::Dict{Symbol, CVResult}; outdir::String)
    mkpath(outdir)
    plot_multilevel_comparison(results; outpath = joinpath(outdir, "01_multilevel.pdf"))
    plot_ridgeline_comparison(results, :pit; title = "PIT Distribution", outpath = joinpath(outdir, "02_pit.pdf"))
    plot_ridgeline_comparison(results, :crps; title = "CRPS Distribution", outpath = joinpath(outdir, "03_crps.pdf"))
    plot_ridgeline_comparison(results, :wis; title = "WIS Distribution", outpath = joinpath(outdir, "04_wis.pdf"))
    plot_coverage_width_heatmap(results; outpath = joinpath(outdir, "05_width_heatmap.pdf"))
    plot_brenda_panel(results; outpath = joinpath(outdir, "06_combined_panel.pdf"))
    plot_group_coverage_comparison(results; outpath = joinpath(outdir, "07_group_coverage.pdf"))
    println("BRENDA plots saved to $outdir")
end

function plot_forest_comparisons(results::Dict{Symbol, CVResult};
    outdir::String, y_database::Union{Vector{Float64}, Nothing} = nothing,
    y_db_std::Union{Vector{Float64}, Nothing} = nothing,
    kinetic_divisor::Float64 = log(10.0))
    @warn "plot_forest_comparisons(CVResult): use ForestPayload version in plot_paper.jl instead. Skipping."
    return nothing
end

"""
    plot_sweep_params(ladder_nu, ladder_alpha; outdir)

Save coverage-vs-degrees-of-freedom and coverage-vs-alpha plots to `outdir` by calling
`plot_ladder_sweep` on each of the two `LadderResult` objects.
"""
function plot_sweep_params(ladder_nu::LadderResult, ladder_alpha::LadderResult; outdir::String)
    mkpath(outdir)
    plot_ladder_sweep(ladder_nu; outpath = joinpath(outdir, "coverage_vs_df.pdf"))
    plot_ladder_sweep(ladder_alpha; outpath = joinpath(outdir, "coverage_vs_alpha.pdf"))
    println("Sweep plots saved to $outdir")
end

# Ladder sweep plotting

"""
    plot_ladder_sweep(ladder; nominal_lev, outpath)

Plot empirical coverage (CI and HDI) as a function of the swept hyperparameter stored in
`ladder`, at nominal level `nominal_lev`.

Draws a dashed reference line at the nominal level with dotted plus/minus 1.96 SE bands.
If provided, saves the figure to `outpath`.
"""
function plot_ladder_sweep(ladder::LadderResult; nominal_lev::Float64 = 0.95,
    outpath::Union{String, Nothing} = nothing)
    param = ladder.sweep_param
    x = ladder.sweep_values
    idx_lev = findfirst(approx(nominal_lev), ladder.results[1].nominal_levels)
    gauss_ci = [mean(r.coverage_CI[:, idx_lev]) for r in ladder.results]
    gauss_hdi = [mean(r.coverage_HDI[:, idx_lev]) for r in ladder.results]

    fig = Figure(size = (800, 600), fontsize = FONT_BASE)
    ax = Axis(fig[1, 1], xlabel = String(param), ylabel = "Coverage",
              xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL,
              title = "Coverage vs $(param) @ $(Int(nominal_lev * 100))%",
              titlesize = FONT_TITLE,
              xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL)

    n_eff = 30 * 10
    se = 1.96 * sqrt(nominal_lev * (1 - nominal_lev) / n_eff)
    hlines!(ax, [nominal_lev], color = :gray, linestyle = :dash, label = "Nominal")
    hlines!(ax, [nominal_lev - se, nominal_lev + se], color = :gray, linestyle = :dot)

    lines!(ax, x, gauss_ci, color = :blue, linewidth = 2, label = "CI")
    scatter!(ax, x, gauss_ci, color = :blue, markersize = 12)
    lines!(ax, x, gauss_hdi, color = :blue, linestyle = :dash, linewidth = 2, label = "HDI")
    scatter!(ax, x, gauss_hdi, color = :blue, marker = :circle, markersize = 8)

    axislegend(ax, position = :rb, labelsize = FONT_LEGEND)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end
