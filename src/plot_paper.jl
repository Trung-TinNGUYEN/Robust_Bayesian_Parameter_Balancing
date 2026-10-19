# Paper-ready plot functions. Include after plot_utils.jl.

using LaTeXStrings

"""
    _build_legend(results, sorted_keys) -> (legend_elements, legend_labels)

Build legend entries from a dict of `CVResult` objects, with "Ideal" as the
first entry (dashed black line).  Deduplicates entries by model label.
"""
function _build_legend(results, sorted_keys)
    # "Ideal" entry: dashed black line only (no marker needed for the diagonal reference)
    leg_e = Any[LineElement(color = :black, linestyle = :dash, linewidth = 1)]
    leg_l = ["Ideal"]
    seen = Set{String}()
    for key in sorted_keys
        sty = _model_style(results[key])
        if !(sty.label in seen)
            push!(seen, sty.label)
            push!(leg_e, _legend_element(sty))  # returns [LineElement, MarkerElement]
            push!(leg_l, sty.label)
        end
    end
    return leg_e, leg_l
end

"""
    plot_parameter_coverage_comparison(results; use_hdi, outpath, title) -> Figure

Multi-level coverage plot for latent-space parameters (not predictive coverage).
Plots empirical vs nominal coverage across all nominal levels for each model.
If `use_hdi` is true, uses HDI intervals; otherwise uses equal-tailed intervals.
If `outpath` is provided, saves the figure there.
"""
function plot_parameter_coverage_comparison(results::Dict{Symbol, CVResult};
    use_hdi::Bool = false, outpath::Union{String, Nothing} = nothing,
    title::String = "Parameter Coverage")
    fig = Figure(size = (800, 600), fontsize = FONT_BASE)
    ax = Axis(fig[1, 1], xlabel = "Nominal level", ylabel = "Empirical coverage",
        xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL,
        title = title, titlesize = FONT_TITLE,
        xticks = 0:0.1:1, yticks = 0:0.1:1,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL)
    lines!(ax, [0, 1], [0, 1], linestyle = :dash, color = :black, linewidth = 1)
    sk = sort(collect(keys(results)))
    for key in sk
        res = results[key]
        cov_mat = use_hdi ? res.coverage_HDI : res.coverage_CI
        size(cov_mat, 2) == length(res.nominal_levels) || continue
        all(iszero, cov_mat) && continue
        sty = _model_style(res)
        _plot_styled_curve!(ax, res.nominal_levels, vec(mean(cov_mat, dims = 1)), sty)
    end
    xlims!(ax, 0, 1.02); ylims!(ax, 0, 1.02)
    leg_e, leg_l = _build_legend(results, sk)
    Legend(fig[1, 2], leg_e, leg_l, framevisible = true, labelsize = FONT_LEGEND,
        tellwidth = true, tellheight = false, padding = (6, 6, 4, 4), rowgap = 2)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

"""
    plot_multilevel_comparison(results; outpath, title) -> Figure

Multi-level predictive coverage plot: empirical coverage vs nominal level for all
models, with the ideal diagonal as a dashed reference line.  Paper-ready style
(BW-friendly markers and line styles).  If `outpath` is provided, saves to PDF.
"""
function plot_multilevel_comparison(results::Dict{Symbol, CVResult};
    outpath::Union{String, Nothing} = nothing, title::String = "Predictive Coverage")
    fig = Figure(size = (800, 600), fontsize = FONT_BASE)
    ax = Axis(fig[1, 1], xlabel = "Nominal level", ylabel = "Empirical coverage",
        xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL,
        title = title, titlesize = FONT_TITLE,
        xticks = 0:0.1:1, yticks = 0:0.1:1,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL)
    lines!(ax, [0, 1], [0, 1], linestyle = :dash, color = :black, linewidth = 1)
    sk = sort(collect(keys(results)))
    for key in sk
        res = results[key]; sty = _model_style(res)
        _plot_styled_curve!(ax, res.nominal_levels, vec(mean(res.pred_coverage, dims = 1)), sty)
    end
    xlims!(ax, 0, 1.02); ylims!(ax, 0, 1.02)
    leg_e, leg_l = _build_legend(results, sk)
    Legend(fig[1, 2], leg_e, leg_l, framevisible = true, labelsize = FONT_LEGEND,
        tellwidth = true, tellheight = false, padding = (6, 6, 4, 4), rowgap = 2)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

"""
    plot_group_coverage_comparison(results; selected_keys, level, min_group_size,
        alpha_band, outpath, custom_colors) -> Figure

Per-group coverage plot at a single nominal `level`.  Groups are sorted by size
along the x-axis; a binomial acceptance band is shown in gray.  Groups smaller
than `min_group_size` are excluded.  Optional `custom_colors` overrides the
default family-based color for specific model keys.
"""
function plot_group_coverage_comparison(results::Dict{Symbol, CVResult};
    selected_keys::Union{Vector{Symbol}, Nothing} = nothing,
    level::Float64 = 0.95, min_group_size::Int = 3,
    alpha_band::Float64 = 0.05, outpath::Union{String, Nothing} = nothing,
    # Optional per-key color overrides distinguish models from the same family.
    custom_colors::Union{Dict{Symbol,<:Any}, Nothing} = nothing)

    all_keys = sort(collect(keys(results)))
    ktp = isnothing(selected_keys) ? all_keys : intersect(selected_keys, all_keys)
    isempty(ktp) && error("No matching keys")
    first_res = results[ktp[1]]; df_first = first_res.obs_metadata
    isnothing(df_first) && error("Metadata required")
    actual_col = hasproperty(df_first, :group) ? :group :
                 hasproperty(df_first, :qrow)  ? :qrow  : error("No grouping col")
    idx_lev = findfirst(x -> isapprox(x, level), first_res.nominal_levels)
    ugroups_all = unique(df_first[!, actual_col])
    g_sizes  = Dict(g => count(df_first[!, actual_col] .== g) for g in ugroups_all)
    ugroups  = [g for g in ugroups_all if g_sizes[g] >= min_group_size]
    G        = length(ugroups)
    g_n      = [g_sizes[g] for g in ugroups]
    ord      = sortperm(g_n); g_n_sorted = g_n[ord]
    g_lo     = [binom_accept_band(n, level; alpha = alpha_band)[1] for n in g_n_sorted]
    g_hi     = [binom_accept_band(n, level; alpha = alpha_band)[2] for n in g_n_sorted]

    fig = Figure(size = (900, 500), fontsize = FONT_BASE)
    ax  = Axis(fig[1, 1],
        xlabel        = "Groups (sorted by size)", ylabel = "Coverage",
        xlabelsize    = FONT_XLABEL, ylabelsize = FONT_YLABEL,
        title         = "Group Coverage @ $(Int(round(100 * level)))%",
        titlesize     = FONT_TITLE,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL)

    x = 1:G
    band!(ax, x, g_lo, g_hi, color = (:gray, 0.15))
    hlines!(ax, [level], linestyle = :dash, color = :black, linewidth = 1)

    seen = Set{String}(); leg_e = Any[]; leg_l = String[]
    for (ki, key) in enumerate(ktp)
        res     = results[key]
        raw_cov = res.pred_coverage[:, idx_lev]
        df_curr = res.obs_metadata
        g_cov   = [mean(raw_cov[df_curr[!, actual_col] .== g]) for g in ugroups]

        # Resolve style; apply colour override when supplied
        sty     = _model_style(res)
        c_plot  = (!isnothing(custom_colors) && haskey(custom_colors, key)) ?
                  custom_colors[key] : sty.color
        c_strk  = (!isnothing(custom_colors) && haskey(custom_colors, key)) ?
                  custom_colors[key] : sty.strokecolor

        jit = (ki - (length(ktp) + 1) / 2) * 0.12
        scatter!(ax, Float64.(collect(x)) .+ jit, g_cov[ord],
            color       = c_plot,
            marker      = sty.marker,
            markersize  = sty.markersize,
            strokewidth = sty.strokewidth,
            strokecolor = c_strk)

        if !(sty.label in seen)
            push!(seen, sty.label)
            # If a colour override is active, rebuild the legend element to match
            leg_sty = (!isnothing(custom_colors) && haskey(custom_colors, key)) ?
                ModelStyle(c_plot, sty.marker, sty.linestyle, sty.linewidth,
                           sty.markersize, sty.strokewidth, c_strk, sty.label) :
                sty
            push!(leg_e, _legend_element(leg_sty))
            push!(leg_l, sty.label)
        end
    end

    ylims!(ax, 0.0, 1.05); xlims!(ax, 0, G + 1)
    Legend(fig[1, 2], leg_e, leg_l,
        framevisible = true, labelsize = FONT_LEGEND,
        tellwidth = true, rowgap = 1)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

"""
    plot_gaussian_group_coverage(results; level, outpath) -> Figure

Convenience wrapper: calls `plot_group_coverage_comparison` with only the
Gaussian Fixed and RE models, using distinct colors (steelblue / firebrick).
"""
function plot_gaussian_group_coverage(results::Dict{Symbol, CVResult};
    level::Float64 = 0.95, outpath::Union{String, Nothing} = nothing)
    plot_group_coverage_comparison(results;
        selected_keys = [:gaussian_Fix, :gaussian_RE],
        level         = level,
        custom_colors = Dict(
            :gaussian_Fix => :steelblue,
            :gaussian_RE  => :firebrick),
        outpath       = outpath)
end

"""
    plot_group_coverage_paired(results; pairs, level, min_group_size, outpath) -> Figure

Scatter plot comparing per-group coverage at `level` between two models for each
supplied pair `(key1, key2)`.  One panel per pair; diagonal is the identity line.
"""
function plot_group_coverage_paired(results::Dict{Symbol, CVResult};
    pairs::Vector{Tuple{Symbol, Symbol}}, level::Float64 = 0.95,
    min_group_size::Int = 3, outpath::Union{String, Nothing} = nothing)
    n_pairs = length(pairs)
    fig = Figure(size = (min(500 * n_pairs, 1400), 420), fontsize = FONT_BASE)
    for (pi, (k1, k2)) in enumerate(pairs)
        (haskey(results, k1) && haskey(results, k2)) || continue
        r1, r2 = results[k1], results[k2]; df = r1.obs_metadata; isnothing(df) && continue
        actual_col = hasproperty(df, :group) ? :group : :qrow
        idx_lev = findfirst(x -> isapprox(x, level), r1.nominal_levels)
        isnothing(idx_lev) && continue
        ugroups = unique(df[!, actual_col])
        g_sizes = Dict(g => count(df[!, actual_col] .== g) for g in ugroups)
        ugroups = filter(g -> g_sizes[g] >= min_group_size, ugroups)
        G = length(ugroups); G == 0 && continue
        cov1 = [mean(r1.pred_coverage[df[!, actual_col] .== g, idx_lev]) for g in ugroups]
        cov2 = [mean(r2.pred_coverage[df[!, actual_col] .== g, idx_lev]) for g in ugroups]
        ax = Axis(fig[1, pi],
            xlabel = "$(_short_label(r1)) coverage", ylabel = "$(_short_label(r2)) coverage",
            xlabelsize = FONT_XLABEL, ylabelsize = FONT_YLABEL,
            title = "$(_short_label(r1)) vs $(_short_label(r2))", titlesize = FONT_TITLE,
            xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL,
            aspect = DataAspect())
        lines!(ax, [0, 1], [0, 1], color = (:gray, 0.5), linestyle = :dash)
        scatter!(ax, cov1, cov2, markersize = 5, color = (:black, 0.45),
            strokewidth = 0.5, strokecolor = :black)
        xlims!(ax, -0.02, 1.05); ylims!(ax, -0.02, 1.05)
    end
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

"""
    plot_coverage_width_heatmap(results; level, outpath) -> Figure

Heatmap of posterior interval widths at `level` across models (rows) and parameters
(columns).  Parameters in the top 10% of widths for any model are labelled in red bold.
"""
function plot_coverage_width_heatmap(results::Dict{Symbol, CVResult};
    level::Float64 = 0.95, outpath::Union{String, Nothing} = nothing)
    first_res = first(values(results)); param_names = first_res.param_names; d = length(param_names)
    idx_lev = findfirst(x -> isapprox(x, level, atol = 1e-5), first_res.nominal_levels)
    sorted_keys = sort(collect(keys(results))); n_models = length(sorted_keys)
    widths_mat = zeros(n_models, d)
    for (i, key) in enumerate(sorted_keys)
        widths_mat[i, :] = results[key].interval_width[:, idx_lev]
    end
    all(iszero, widths_mat) && @warn "plot_coverage_width_heatmap: all widths zero"
    is_prob = identify_problematic_params(widths_mat)
    model_labels = [_short_label(results[k]) for k in sorted_keys]
    fig = Figure(size = (1400, 700), fontsize = FONT_BASE, figure_padding = (10, 10, 10, 10))
    ax = Axis(fig[1, 1], ylabel = "Models", ylabelsize = FONT_YLABEL,
        title = "Interval Width @ $(Int(100 * level))%", titlesize = FONT_TITLE,
        yticks = (1:n_models, model_labels), yticklabelsize = FONT_TICKLABEL,
        xticklabelsvisible = false, xticksvisible = false)
    hm = heatmap!(ax, 1:d, 1:n_models, widths_mat', colormap = :viridis)
    Colorbar(fig[1, 2], hm, label = "Width", labelsize = FONT_XLABEL)
    prob_idxs = findall(is_prob)
    !isempty(prob_idxs) && scatter!(ax, Float64.(prob_idxs), fill(0.35, length(prob_idxs)),
        marker = :dtriangle, markersize = 8, color = :red)
    for (j, name) in enumerate(param_names)
        c = is_prob[j] ? :red : :black; fw = is_prob[j] ? :bold : :regular
        text!(ax, j, 0.0; text = name, rotation = pi / 2, align = (:right, :center),
              fontsize = FONT_HEATMAP_TXT, color = c, font = fw)
    end
    ylims!(ax, -2.5, n_models + 0.5); xlims!(ax, 0.5, d + 0.5)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

"""
    plot_scoring_summary(results; level, outpath, title) -> Figure

Bar-chart summary of mean CRPS, mean WIS, and empirical coverage at `level`
for each model.  Useful for a compact one-figure comparison.
"""
function plot_scoring_summary(results::Dict{Symbol, CVResult};
    level::Float64 = 0.95, outpath::Union{String, Nothing} = nothing,
    title::String = "Scoring Summary")
    sorted_keys = sort(collect(keys(results))); n = length(sorted_keys)
    labels = [_short_label(results[k]) for k in sorted_keys]
    crps_vals = [mean(filter(isfinite, results[k].crps)) for k in sorted_keys]
    wis_vals = [mean(filter(isfinite, results[k].wis)) for k in sorted_keys]
    cov_vals = [let idx = findfirst(x -> isapprox(x, level), results[k].nominal_levels);
                    idx !== nothing ? mean(results[k].pred_coverage[:, idx]) : NaN end
                for k in sorted_keys]
    cs = [_model_style(results[k]).color for k in sorted_keys]
    fig = Figure(size = (1000, 400), fontsize = FONT_BASE)
    Label(fig[0, 1:3], title, fontsize = FONT_TITLE, font = :bold)
    ax1 = Axis(fig[1, 1], ylabel = "CRPS", ylabelsize = FONT_YLABEL,
        xticks = (1:n, labels), xticklabelrotation = pi / 4,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL,
        title = L"CRPS ($\downarrow$ better)", titlesize = FONT_TITLE)
    barplot!(ax1, 1:n, crps_vals, color = cs, strokewidth = 1, strokecolor = :black)
    ax2 = Axis(fig[1, 2], ylabel = "WIS", ylabelsize = FONT_YLABEL,
        xticks = (1:n, labels), xticklabelrotation = pi / 4,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL,
        title = L"WIS ($\downarrow$ better)", titlesize = FONT_TITLE)
    barplot!(ax2, 1:n, wis_vals, color = cs, strokewidth = 1, strokecolor = :black)
    ax3 = Axis(fig[1, 3], ylabel = "Coverage", ylabelsize = FONT_YLABEL,
        xticks = (1:n, labels), xticklabelrotation = pi / 4,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL,
        title = "Coverage @ $(Int(100 * level))%", titlesize = FONT_TITLE)
    barplot!(ax3, 1:n, cov_vals, color = cs, strokewidth = 1, strokecolor = :black)
    hlines!(ax3, [level], linestyle = :dash, color = :black, linewidth = 1)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

"""
    plot_brenda_combined_panel(results; level, outpath, pathway) -> Figure

Two-row combined panel for the paper: row 1 is a full-width interval-width
heatmap with red-labelled problematic parameters; row 2 shows CRPS, WIS, and
coverage bar charts.  The multilevel coverage curve is produced separately.
"""
function plot_brenda_combined_panel(results::Dict{Symbol, CVResult};
    level::Float64 = 0.95, outpath::Union{String, Nothing} = nothing,
    pathway::String = "TCA")
    sorted_keys = sort(collect(keys(results))); n_models = length(sorted_keys)
    first_res = results[sorted_keys[1]]
    param_names = first_res.param_names; d_param = length(param_names)
    idx_lev = findfirst(x -> isapprox(x, level, atol = 1e-5), first_res.nominal_levels)

    # Layout: row 0 = title; row 1 = full-width heatmap; row 2 = three bar charts.
    # The multilevel coverage curve is intentionally omitted here (it is a standalone figure
    # in the paper).  The heatmap is given the full column span so the parameter labels
    # are as readable as possible.
    fig = Figure(size = (1800, 950), fontsize = FONT_BASE)
    Label(fig[0, 1:3], "$pathway BRENDA LOSO - Model Comparison",
        fontsize = FONT_SUPERTITLE, font = :bold)

    # Row 1: interval-width heatmap.
    widths_mat = zeros(n_models, d_param)
    for (i, key) in enumerate(sorted_keys)
        widths_mat[i, :] = results[key].interval_width[:, idx_lev]
    end
    is_prob = identify_problematic_params(widths_mat)
    model_labels = [_short_label(results[k]) for k in sorted_keys]

    ax_hm = Axis(fig[1, 1:3], ylabel = "Models", ylabelsize = FONT_YLABEL,
        title = "Interval Width @ $(Int(100 * level))%  " *
                "(red labels: top 10% width in ≥1 model)",
        titlesize = FONT_TITLE,
        yticks = (1:n_models, model_labels), yticklabelsize = FONT_TICKLABEL,
        xticklabelsvisible = false, xticksvisible = false)
    hm = heatmap!(ax_hm, 1:d_param, 1:n_models, widths_mat', colormap = :viridis)
    Colorbar(fig[1, 4], hm, label = "Width", labelsize = FONT_XLABEL)

    prob_idxs = findall(is_prob)
    !isempty(prob_idxs) && scatter!(ax_hm, Float64.(prob_idxs),
        fill(0.35, length(prob_idxs)), marker = :dtriangle, markersize = 8, color = :red)

    for (j, name) in enumerate(param_names)
        c = is_prob[j] ? :red : :black; fw = is_prob[j] ? :bold : :regular
        text!(ax_hm, j, 0.0; text = name, rotation = pi / 2, align = (:right, :center),
            fontsize = FONT_HEATMAP_TXT - 1, color = c, font = fw)
    end
    ylims!(ax_hm, -2.5, n_models + 0.5); xlims!(ax_hm, 0.5, d_param + 0.5)

    # give the heatmap row more height than the bar row
    rowsize!(fig.layout, 1, Relative(0.62))

    # Row 2: CRPS, WIS, and 95% coverage summaries.
    crps_vals = [mean(filter(isfinite, results[k].crps)) for k in sorted_keys]
    wis_vals  = [mean(filter(isfinite, results[k].wis))  for k in sorted_keys]
    cov_vals  = [let i2 = findfirst(x -> isapprox(x, level), results[k].nominal_levels);
                     i2 !== nothing ? mean(results[k].pred_coverage[:, i2]) : NaN end
                 for k in sorted_keys]
    cs = [_model_style(results[k]).color for k in sorted_keys]

    ax_crps = Axis(fig[2, 1], ylabel = "CRPS", ylabelsize = FONT_YLABEL,
        xticks = (1:n_models, model_labels), xticklabelrotation = pi / 4,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL,
        title = L"CRPS ($\downarrow$ better)", titlesize = FONT_TITLE)
    barplot!(ax_crps, 1:n_models, crps_vals, color = cs, strokewidth = 1, strokecolor = :black)

    ax_wis = Axis(fig[2, 2], ylabel = "WIS", ylabelsize = FONT_YLABEL,
        xticks = (1:n_models, model_labels), xticklabelrotation = pi / 4,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL,
        title = L"WIS ($\downarrow$ better)", titlesize = FONT_TITLE)
    barplot!(ax_wis, 1:n_models, wis_vals, color = cs, strokewidth = 1, strokecolor = :black)

    ax_cov = Axis(fig[2, 3], ylabel = "Coverage", ylabelsize = FONT_YLABEL,
        xticks = (1:n_models, model_labels), xticklabelrotation = pi / 4,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL,
        title = "Coverage @ $(Int(100 * level))%", titlesize = FONT_TITLE)
    barplot!(ax_cov, 1:n_models, cov_vals, color = cs, strokewidth = 1, strokecolor = :black)
    hlines!(ax_cov, [level], linestyle = :dash, color = :black, linewidth = 1)

    !isnothing(outpath) && save(outpath, fig)
    return fig
end

"""
    ForestStyle

Visual encoding for a single model series in a forest plot: marker shape, color,
stroke, fill, line width, and line style.
"""
struct ForestStyle
    marker      :: Symbol
    color       :: Any
    markersize  :: Int
    strokewidth :: Float64
    strokecolor :: Any
    fill        :: Any
    linewidth   :: Float64
    linestyle   :: Any
end

const _FOREST_STYLES = Dict{Symbol, ForestStyle}(
    :database     => ForestStyle(:circle,    Makie.wong_colors()[1], 9,  0.5, :black,    Makie.wong_colors()[1], 2.0, :solid),
    :gaussian_Fix => ForestStyle(:utriangle, :gray45,               9,  1.5, :gray45,   :white,                1.5, :dash),
    :gaussian_RE  => ForestStyle(:circle,    :firebrick,            9,  0.5, :black,    :firebrick,            1.8, :solid),
    :t_Fix        => ForestStyle(:diamond,   :gray45,               10, 1.5, :gray45,   :white,                1.5, :dash),
    :t_RE         => ForestStyle(:diamond,   :forestgreen,          10, 0.5, :black,    :forestgreen,          1.8, :solid),
    :skew_Fix     => ForestStyle(:rect,      :gray45,               9,  1.5, :gray45,   :white,                1.5, :dash),
    :skew_RE      => ForestStyle(:star5,     :darkorchid3,          10, 0.5, :black,    :darkorchid3,          1.8, :solid),
)

"""
    _get_forest_style(key) -> ForestStyle

Look up the `ForestStyle` for a model key, falling back to pattern matching on
the key string if no exact entry exists in `_FOREST_STYLES`.
"""
function _get_forest_style(key::Symbol)
    haskey(_FOREST_STYLES, key) && return _FOREST_STYLES[key]
    s = string(key)
    startswith(s, "gaussian") && endswith(s, "Fix")  && return _FOREST_STYLES[:gaussian_Fix]
    startswith(s, "gaussian") && endswith(s, "RE")   && return _FOREST_STYLES[:gaussian_RE]
    startswith(s, "t")        && endswith(s, "Fix")  && return _FOREST_STYLES[:t_Fix]
    startswith(s, "t")        && endswith(s, "RE")   && return _FOREST_STYLES[:t_RE]
    startswith(s, "skew")     && endswith(s, "Fix")  && return _FOREST_STYLES[:skew_Fix]
    startswith(s, "skew")     && endswith(s, "RE")   && return _FOREST_STYLES[:skew_RE]
    return ForestStyle(:xcross, :gray30, 9, 1.0, :gray30, :gray30, 1.5, :solid)
end

_forest_legend_element(sty::ForestStyle) = MarkerElement(
    marker = sty.marker, color = sty.fill, markersize = sty.markersize,
    strokewidth = sty.strokewidth, strokecolor = sty.strokecolor)

"""
    make_forest_plot(forests; model_keys, title, outpath, kinetic_divisor) -> Figure

Render a three-panel forest plot (standard chemical potentials, kcat, KM) from a
dict of `ForestPayload` objects.  Each row shows the database reference and one
interval per model key.  kcat and KM values are divided by `kinetic_divisor`
(default `log(10)`) to convert internal log-e scale to log10 for display.
"""
function make_forest_plot(forests::Dict{Symbol, ForestPayload};
    model_keys::Vector{Symbol} = sort(collect(keys(forests))),
    title::String = "Model Comparison",
    outpath::Union{String, Nothing} = nothing,
    kinetic_divisor::Float64 = log(10.0))

    isempty(model_keys) && error("make_forest_plot: model_keys is empty")
    first_fp = forests[first(model_keys)]; obs_names = first_fp.obs_names; n_obs = length(obs_names)
    for k in model_keys
        length(forests[k].model_mean) == n_obs || error("make_forest_plot: length mismatch for $k")
    end

    mu0_idx  = findall(n -> startswith(n, "mu0:") || startswith(n, "\u03bc0:"), obs_names)
    kcat_idx = findall(n -> startswith(n, "kcat:"), obs_names)
    KM_idx   = findall(n -> startswith(n, "KM:"), obs_names)
    isempty(mu0_idx) && isempty(kcat_idx) && isempty(KM_idx) &&
        error("make_forest_plot: no mu0:/kcat:/KM: entries")

    to_display(name, x) = (startswith(name, "kcat:") || startswith(name, "KM:")) ? x / kinetic_divisor : x
    function short_name(nm::String)
        startswith(nm, "\u03bc0:") && return nm[nextind(nm, 0, 4):end]
        startswith(nm, "mu0:")     && return nm[5:end]
        return replace(nm, r"^(kcat|KM):" => "")
    end
    function build_panel(indices::Vector{Int}; drop = (_::String -> false))
        rows = NamedTuple[]
        for idx in indices
            nm = obs_names[idx]; drop(nm) && continue
            push!(rows, (name = nm, short = short_name(nm),
                db_mean = to_display(nm, first_fp.db_mean[idx]),
                db_lo = to_display(nm, first_fp.db_lo[idx]),
                db_hi = to_display(nm, first_fp.db_hi[idx]),
                m_mean = [to_display(nm, forests[k].model_mean[idx]) for k in model_keys],
                m_lo   = [to_display(nm, forests[k].model_lo[idx])   for k in model_keys],
                m_hi   = [to_display(nm, forests[k].model_hi[idx])   for k in model_keys]))
        end
        df = DataFrame(rows); nrow(df) > 0 && sort!(df, :name, rev = true); df
    end

    df_mu0 = build_panel(mu0_idx); df_kcat = build_panel(kcat_idx)
    df_KM = build_panel(KM_idx; drop = nm -> occursin("h2o", lowercase(nm)))
    n_models = length(model_keys); n_series = 1 + n_models; db_sty = _FOREST_STYLES[:database]
    legend_labels = ["Database"]; legend_elements = [_forest_legend_element(db_sty)]
    for key in model_keys
        sty = _get_forest_style(key)
        # Change here from Fix to Fixed
        push!(legend_labels, replace(string(key), "_" => " ", "Fix" => "Fixed"))
        push!(legend_elements, _forest_legend_element(sty))
    end
    half_span = min(PAPER_FOREST_WITHIN_ROW_SPAN, 0.10 * n_series)
    y_offsets = n_series == 1 ? [0.0] : collect(range(-half_span, half_span, length = n_series))

    row_count = max(nrow(df_mu0) + nrow(df_kcat) + 4, nrow(df_KM) + 2)
    fig_h = clamp(row_count * PAPER_FOREST_ROW_PIXELS,
        PAPER_FOREST_MIN_HEIGHT, PAPER_FOREST_MAX_HEIGHT)
    fig = Figure(size = (1200, fig_h), fontsize = FONT_BASE)
    Label(fig[0, :], title, fontsize = FONT_SUPERTITLE, font = :bold)

    function draw_panel(ax, df::DataFrame, centered::Bool)
        n = nrow(df); n == 0 && return
        for (i, row) in enumerate(eachrow(df))
            ybase = 1.0 + (i - 1) * PAPER_FOREST_ROW_GAP
            db_x = centered ? 0.0 : row.db_mean
            lo_hw = db_x - (centered ? (row.db_lo - row.db_mean) : row.db_lo)
            hi_hw = (centered ? (row.db_hi - row.db_mean) : row.db_hi) - db_x
            ypos = ybase + y_offsets[1]
            errorbars!(ax, [db_x], [ypos], [lo_hw], [hi_hw],
                direction = :x, color = db_sty.color, linewidth = db_sty.linewidth, whiskerwidth = 0)
            scatter!(ax, [db_x], [ypos], color = db_sty.fill, marker = db_sty.marker,
                markersize = db_sty.markersize, strokewidth = db_sty.strokewidth, strokecolor = db_sty.strokecolor)
        end
        for (m, key) in enumerate(model_keys)
            sty = _get_forest_style(key)
            for (i, row) in enumerate(eachrow(df))
                ybase = 1.0 + (i - 1) * PAPER_FOREST_ROW_GAP
                val = centered ? (row.m_mean[m] - row.db_mean) : row.m_mean[m]
                lo_v = centered ? (row.m_lo[m] - row.db_mean) : row.m_lo[m]
                hi_v = centered ? (row.m_hi[m] - row.db_mean) : row.m_hi[m]
                (isfinite(val) && isfinite(lo_v) && isfinite(hi_v)) || continue
                ypos = ybase + y_offsets[1 + m]
                errorbars!(ax, [val], [ypos], [val - lo_v], [hi_v - val],
                    direction = :x, color = sty.color, linewidth = sty.linewidth,
                    whiskerwidth = 0, linestyle = sty.linestyle)
                scatter!(ax, [val], [ypos], color = sty.fill, marker = sty.marker,
                    markersize = sty.markersize, strokewidth = sty.strokewidth, strokecolor = sty.strokecolor)
            end
        end
        ytick_positions = [1.0 + (i - 1) * PAPER_FOREST_ROW_GAP for i in 1:n]
        ax.yticks = (ytick_positions, df.short); ax.yreversed = true
        ax.yticklabelsize = FONT_FOREST_YTICK; ax.xticklabelsize = FONT_TICKLABEL
        ylims!(ax, 1.0 - PAPER_FOREST_ROW_GAP, 1.0 + n * PAPER_FOREST_ROW_GAP)
        hidespines!(ax, :t, :r)
    end

    ax1 = Axis(fig[1, 1], xlabel = "Deviation from Database [kJ/mol]",
        xlabelsize = FONT_XLABEL, xgridvisible = true)
    draw_panel(ax1, df_mu0, true)
    Label(fig[1, 1, Top()], "(a) Standard chemical potentials", halign = :center, fontsize = FONT_TITLE)

    ax2 = Axis(fig[2, 1], xlabel = L"$\log_{10}(k_{\mathrm{cat}})$ [s$^{-1}$]",
        xlabelsize = FONT_XLABEL, xgridvisible = true)
    draw_panel(ax2, df_kcat, false)
    Label(fig[2, 1, Top()], "(b) kcat values", halign = :center, fontsize = FONT_TITLE)

    ax3 = Axis(fig[1:2, 2], xlabel = L"$\log_{10}(K_M)$ [M]",
        xlabelsize = FONT_XLABEL, xgridvisible = true)
    draw_panel(ax3, df_KM, false)
    Label(fig[1, 2, Top()], "(c) KM values", halign = :center, fontsize = FONT_TITLE)

    colgap!(fig.layout, 60)
    Legend(fig[3, 1:2], legend_elements, legend_labels,
        orientation = :horizontal, tellheight = true, tellwidth = false,
        framevisible = true, padding = (8, 8, 4, 4), colgap = 14, nbanks = 1, labelsize = FONT_LEGEND)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

"""
    plot_forest_comparisons(forests; outdir, kinetic_divisor)

Generate and save BRENDA forest plots to `outdir`: one plot for Fixed-only models,
one for RE-only models, and one per likelihood family (Gaussian, Student-t, Skew-N)
showing Fixed vs RE side by side.
"""
function plot_forest_comparisons(forests::Dict{Symbol, ForestPayload};
    outdir::String, kinetic_divisor::Float64 = log(10.0))
    mkpath(outdir); isempty(forests) && return
    is_re(k) = endswith(string(k), "_RE")
    ff = Dict(k => v for (k, v) in forests if !is_re(k))
    fr = Dict(k => v for (k, v) in forests if is_re(k))
    !isempty(ff) && make_forest_plot(ff; title = "Model Comparison (Fixed Effects)",
        outpath = joinpath(outdir, "01_models_fixed.pdf"), kinetic_divisor = kinetic_divisor)
    for (mdl, fn, ttl) in [("gaussian","02_gauss_fixed_vs_re.pdf","Gaussian: Fixed vs RE (Heterogeneous Data)"),
            ("t","03_t_fixed_vs_re.pdf","Student-t: Fixed vs RE (Heterogeneous Data)"),
            ("skew","04_skew_fixed_vs_re.pdf","Skew-N: Fixed vs RE (Heterogeneous Data)")]
        sub = Dict(k => v for (k, v) in forests if startswith(string(k), mdl))
        isempty(sub) && continue
        make_forest_plot(sub; title = ttl, outpath = joinpath(outdir, fn), kinetic_divisor = kinetic_divisor)
    end
    !isempty(fr) && make_forest_plot(fr; title = "Model Comparison (Random Effects)",
        outpath = joinpath(outdir, "05_models_re.pdf"), kinetic_divisor = kinetic_divisor)
end

"""
    plot_synthetic_forest_comparisons(forests; outdir, kinetic_divisor)

Generate and save synthetic-data forest plots to `outdir`: all models together,
Gaussian Fixed vs all RE models, and per-family Fixed vs RE comparisons.
"""
function plot_synthetic_forest_comparisons(forests::Dict{Symbol, ForestPayload};
    outdir::String, kinetic_divisor::Float64 = log(10.0))
    mkpath(outdir); isempty(forests) && return
    is_re(k) = endswith(string(k), "_RE"); is_fix(k) = endswith(string(k), "_Fix")
    ak = sort(collect(keys(forests))); rk = sort(filter(is_re, ak)); fk = sort(filter(is_fix, ak))
    length(ak) >= 2 && make_forest_plot(forests; model_keys = ak,
        title = "All Models vs Database", outpath = joinpath(outdir, "01_all_models.pdf"), kinetic_divisor = kinetic_divisor)
    gf = filter(k -> string(k) == "gaussian_Fix", ak)
    if !isempty(gf) && !isempty(rk)
        k5 = vcat(gf, rk); make_forest_plot(Dict(k => forests[k] for k in k5); model_keys = k5,
            title = "RE Models + Gaussian Baseline", outpath = joinpath(outdir, "02_re_plus_gauss_baseline.pdf"),
            kinetic_divisor = kinetic_divisor)
    end
    !isempty(rk) && make_forest_plot(Dict(k => forests[k] for k in rk); model_keys = rk,
        title = "Model Comparison (Random Effects)", outpath = joinpath(outdir, "03_re_models.pdf"), kinetic_divisor = kinetic_divisor)
    !isempty(fk) && make_forest_plot(Dict(k => forests[k] for k in fk); model_keys = fk,
        title = "Model Comparison (Fixed Effects)", outpath = joinpath(outdir, "04_fixed_models.pdf"), kinetic_divisor = kinetic_divisor)
    for (mdl, fn, ttl) in [("gaussian","05_gauss_fixed_vs_re.pdf","Gaussian: Fixed vs RE (Heterogeneous Data)"),
            ("t","06_t_fixed_vs_re.pdf","Student-t: Fixed vs RE (Heterogeneous Data)"),
            ("skew","07_skew_fixed_vs_re.pdf","Skew-N: Fixed vs RE (Heterogeneous Data)")]
        sub = Dict(k => v for (k, v) in forests if startswith(string(k), mdl))
        isempty(sub) && continue
        make_forest_plot(sub; title = ttl, outpath = joinpath(outdir, fn), kinetic_divisor = kinetic_divisor)
    end
end

function plot_paper_brenda(results::Dict{Symbol, CVResult};
    outdir::String, pathway::String = "TCA", kinetic_divisor::Float64 = log(10.0))
    mkpath(outdir)
    plot_multilevel_comparison(results; outpath = joinpath(outdir, "01_multilevel_coverage.pdf"),
        title = "Predictive Coverage - $pathway BRENDA LOSO")
    plot_coverage_width_heatmap(results; outpath = joinpath(outdir, "02_width_heatmap.pdf"))
    plot_gaussian_group_coverage(results; level = 0.95,
        outpath = joinpath(outdir, "03_gaussian_group_coverage.pdf"))
    frp = Tuple{Symbol, Symbol}[]
    for m in [:gaussian, :t, :skew]
        fk, rk = Symbol("$(m)_Fix"), Symbol("$(m)_RE")
        haskey(results, fk) && haskey(results, rk) && push!(frp, (fk, rk))
    end
    !isempty(frp) && plot_group_coverage_paired(results; pairs = frp,
        outpath = joinpath(outdir, "03b_group_coverage_fix_vs_re.pdf"))
    rp = Tuple{Symbol, Symbol}[]; rks = [:gaussian_RE, :t_RE, :skew_RE]
    for i in 1:length(rks), j in (i+1):length(rks)
        haskey(results, rks[i]) && haskey(results, rks[j]) && push!(rp, (rks[i], rks[j]))
    end
    !isempty(rp) && plot_group_coverage_paired(results; pairs = rp,
        outpath = joinpath(outdir, "03c_group_coverage_re_families.pdf"))
    plot_scoring_summary(results; outpath = joinpath(outdir, "04_scoring_summary.pdf"),
        title = "$pathway - Scoring Summary")
    plot_pit_comparison(results; outpath = joinpath(outdir, "05_pit.pdf"))
end

function plot_paper_synthetic(results::Dict{Symbol, CVResult};
    outdir::String, pathway::String = "TCA", kinetic_divisor::Float64 = log(10.0))
    mkpath(outdir)
    plot_multilevel_comparison(results; outpath = joinpath(outdir, "01_multilevel_coverage.pdf"),
        title = "Parameter Coverage - $pathway Synthetic")
    plot_coverage_width_heatmap(results; outpath = joinpath(outdir, "02_width_heatmap.pdf"))
    plot_scoring_summary(results; outpath = joinpath(outdir, "03_scoring_summary.pdf"),
        title = "$pathway Synthetic - Scoring Summary")
    first_res = first(values(results))
    if first_res.obs_metadata !== nothing
        frp = Tuple{Symbol, Symbol}[]
        for m in [:gaussian, :t, :skew]
            fk, rk = Symbol("$(m)_Fix"), Symbol("$(m)_RE")
            haskey(results, fk) && haskey(results, rk) && push!(frp, (fk, rk))
        end
        !isempty(frp) && plot_group_coverage_paired(results; pairs = frp,
            outpath = joinpath(outdir, "04_group_coverage_fix_vs_re.pdf"))
    end
end

# Coverage sweep plots

"""
    CoverageSweepResult

Stores paired Gaussian and robust-model coverage across a sweep of one
distributional parameter, with source heterogeneity removed from the data
generating process.

# Fields
- sweep_param    :nu or :alpha
- sweep_values   grid of the swept parameter
- gauss_ci / gauss_hdi  mean 95% coverage for the Gaussian fixed-effect fit
- robust_ci / robust_hdi same for the Student-t or Skew-N fixed-effect fit
- robust_model   :t or :skew
- n_reps         number of simulation replicates used
- pathway        "tca" or "glycolysis"
- coverage_target :predictive, :parameter, or :auto

For predictive sweeps, the `*_ci` vectors store mean predictive coverage and
the `*_hdi` vectors are retained for compatibility with the parameter-coverage
layout.
"""
struct CoverageSweepResult
    sweep_param  :: Symbol
    sweep_values :: Vector{Float64}
    gauss_ci     :: Vector{Float64}
    gauss_hdi    :: Vector{Float64}
    robust_ci    :: Vector{Float64}
    robust_hdi   :: Vector{Float64}
    robust_model :: Symbol
    n_reps       :: Int
    pathway      :: String
    coverage_target :: Symbol
end

# Per-curve styles for fixed-effect distribution sweeps.
struct SweepCurveStyle
    color      :: Any
    marker     :: Symbol
    linestyle  :: Any
    linewidth  :: Float64
    markersize :: Int
    label      :: String
end

const _SWEEP_STYLES = Dict{Symbol, SweepCurveStyle}(
    :gauss_ci   => SweepCurveStyle(:steelblue,   :circle,  :solid, 2.2, 9,  "Gaussian CI"),
    :gauss_hdi  => SweepCurveStyle(:steelblue,   :circle,  :dash,  1.5, 7,  "Gaussian HDI"),
    :t_ci       => SweepCurveStyle(:darkorange,  :diamond, :solid, 2.2, 9,  "Student-t CI"),
    :t_hdi      => SweepCurveStyle(:darkorange,  :diamond, :dash,  1.5, 7,  "Student-t HDI"),
    :skew_ci    => SweepCurveStyle(:darkorange,  :star5,   :solid, 2.2, 10, "Skew-N CI"),
    :skew_hdi   => SweepCurveStyle(:darkorange,  :star5,   :dash,  1.5, 7,  "Skew-N HDI"),
)

function _sweep_legend_element(s::SweepCurveStyle)
    [LineElement(color=s.color, linestyle=s.linestyle, linewidth=s.linewidth),
     MarkerElement(color=s.color, marker=s.marker, markersize=s.markersize,
                   strokewidth=0.0, strokecolor=s.color)]
end

function _draw_sweep_curve!(ax, x, y, s::SweepCurveStyle)
    finite_idx = findall(isfinite, y)
    isempty(finite_idx) && return

    start_idx = finite_idx[1]
    prev_idx = start_idx
    for idx in Iterators.drop(finite_idx, 1)
        if idx != prev_idx + 1
            _draw_sweep_segment!(ax, x, y, start_idx, prev_idx, s)
            start_idx = idx
        end
        prev_idx = idx
    end
    _draw_sweep_segment!(ax, x, y, start_idx, prev_idx, s)

    scatter!(ax, x[finite_idx], y[finite_idx];
        color=s.color, marker=s.marker, markersize=s.markersize,
        strokewidth=0.0, strokecolor=s.color)
end

function _draw_sweep_segment!(ax, x, y, i::Int, j::Int, s::SweepCurveStyle)
    i == j && return
    lines!(ax, x[i:j], y[i:j];
        color=s.color, linewidth=s.linewidth, linestyle=s.linestyle)
end

"""
    plot_coverage_sweep(sweep_res; n_eff, outpath)

Plot coverage from a `CoverageSweepResult`.

`n_eff` is the effective number of independent replicates used for the
binomial reference band. It defaults to `sweep_res.n_reps`, a conservative
choice, and can be overridden when a better effective sample size is known.
"""
function plot_coverage_sweep(
    sweep_res  :: CoverageSweepResult;
    nominal_lev :: Float64 = 0.95,
    n_eff       :: Union{Int, Nothing} = nothing,
    outpath     :: Union{String, Nothing} = nothing
) :: Figure

    is_nu = sweep_res.sweep_param == :nu
    xlabel_str = is_nu ? "t degrees of freedom (log scale)" : L"Skewness $\alpha$"
    title_str  = is_nu ?
        "Coverage vs df (t-noise, $(sweep_res.pathway))" :
        "Coverage vs skew ($(sweep_res.pathway))"
    xscale_val = is_nu ? log10 : identity

    _n_eff = isnothing(n_eff) ? sweep_res.n_reps : n_eff
    se = 1.96 * sqrt(nominal_lev * (1.0 - nominal_lev) / _n_eff)

    robust_model = sweep_res.robust_model
    ci_key  = robust_model == :t ? :t_ci   : :skew_ci
    hdi_key = robust_model == :t ? :t_hdi  : :skew_hdi
    is_predictive = sweep_res.coverage_target == :predictive

    x = sweep_res.sweep_values

    fig = Figure(size = (900, 500), fontsize = FONT_BASE)
    ax  = Axis(fig[1, 1],
        xlabel        = xlabel_str, ylabel = "Coverage",
        xlabelsize    = FONT_XLABEL, ylabelsize = FONT_YLABEL,
        title         = title_str,  titlesize  = FONT_TITLE,
        xticklabelsize = FONT_TICKLABEL, yticklabelsize = FONT_TICKLABEL,
        xscale        = xscale_val,
        yticks        = 0.80:0.05:1.00)

    # Reference lines
    hlines!(ax, [nominal_lev], linestyle = :dash, color = :gray50,
            linewidth = 1.0, label = "nominal $(nominal_lev)")
    hlines!(ax, [nominal_lev - se, nominal_lev + se], linestyle = :dot,
            color = :gray70, linewidth = 0.8)

    curve_check = is_predictive ?
        [("Gaussian", sweep_res.gauss_ci),
         ("$(sweep_res.robust_model)", sweep_res.robust_ci)] :
        [("Gaussian CI", sweep_res.gauss_ci),
         ("Gaussian HDI", sweep_res.gauss_hdi),
         ("$(sweep_res.robust_model) CI", sweep_res.robust_ci),
         ("$(sweep_res.robust_model) HDI", sweep_res.robust_hdi)]

    for (label, vals) in curve_check
        all(x -> !isfinite(x), vals) &&
            @warn "Coverage sweep curve has no finite values" label=label pathway=sweep_res.pathway parameter=sweep_res.sweep_param
    end

    _draw_sweep_curve!(ax, x, sweep_res.gauss_ci,  _SWEEP_STYLES[:gauss_ci])
    _draw_sweep_curve!(ax, x, sweep_res.robust_ci,  _SWEEP_STYLES[ci_key])
    if !is_predictive
        _draw_sweep_curve!(ax, x, sweep_res.gauss_hdi, _SWEEP_STYLES[:gauss_hdi])
        _draw_sweep_curve!(ax, x, sweep_res.robust_hdi, _SWEEP_STYLES[hdi_key])
    end

    all_vals = is_predictive ?
        filter(isfinite, vcat(sweep_res.gauss_ci, sweep_res.robust_ci)) :
        filter(isfinite, vcat(
            sweep_res.gauss_ci, sweep_res.gauss_hdi,
            sweep_res.robust_ci, sweep_res.robust_hdi))
    if !isempty(all_vals)
        ylo = max(0.0, min(0.79, minimum(all_vals) - 0.02))
        yhi = max(1.01, maximum(all_vals) + 0.02)
        ylims!(ax, ylo, yhi)
    end

    robust_label = replace(_SWEEP_STYLES[ci_key].label, " CI" => "")

    leg_e = Any[
        LineElement(color = :gray50, linestyle = :dash, linewidth = 1.0),
        LineElement(color = :gray70, linestyle = :dot, linewidth = 0.8),
        _sweep_legend_element(_SWEEP_STYLES[:gauss_ci]),
        _sweep_legend_element(_SWEEP_STYLES[ci_key]),
    ]
    leg_l = Any[
        "nominal $(nominal_lev)",
        L"binom $\pm$1.96 SE",
        is_predictive ? "Gaussian" : "Gaussian CI",
        is_predictive ? robust_label : _SWEEP_STYLES[ci_key].label,
    ]
    if !is_predictive
        push!(leg_e, _sweep_legend_element(_SWEEP_STYLES[:gauss_hdi]))
        push!(leg_e, _sweep_legend_element(_SWEEP_STYLES[hdi_key]))
        push!(leg_l, "Gaussian HDI")
        push!(leg_l, _SWEEP_STYLES[hdi_key].label)
    end
    Legend(fig[1, 2], leg_e, leg_l, framevisible = true,
           labelsize = FONT_LEGEND, tellwidth = true,
           padding = (6, 6, 4, 4), rowgap = 2)

    !isnothing(outpath) && save(outpath, fig)
    return fig
end

"""
    plot_point_accuracy_sweep(csv_path; outpath)

Plot point-estimation summaries saved by `run_coverage_sweep_experiment`.
The panels show mean absolute bias, mean RMSE, Mahalanobis MSE, and mean
95% interval width for the Gaussian fixed-effect fit and the matched robust
fixed-effect fit.
"""
function plot_point_accuracy_sweep(
    csv_path::String;
    outpath::Union{String,Nothing} = nothing
) :: Figure
    df = CSV.read(csv_path, DataFrame)
    required = [
        :sweep_val, :robust_model, :pathway,
        :gauss_mean_abs_bias, :robust_mean_abs_bias,
        :gauss_mean_rmse, :robust_mean_rmse,
        :gauss_mahal_mse, :robust_mahal_mse,
        :gauss_width95, :robust_width95
    ]
    missing_cols = setdiff(required, Symbol.(names(df)))
    isempty(missing_cols) ||
        error("plot_point_accuracy_sweep: missing columns $(missing_cols) in $csv_path")

    sweep_param = (:sweep_param in Symbol.(names(df))) ? Symbol(df.sweep_param[1]) :
        (occursin("alpha", lowercase(basename(csv_path))) ? :alpha : :nu)
    is_nu = sweep_param == :nu
    x = Vector{Float64}(df.sweep_val)
    xscale_val = is_nu ? log10 : identity
    xlabel_str = is_nu ? "t degrees of freedom (log scale)" : L"Skewness $\alpha$"
    robust_model = Symbol(df.robust_model[1])
    robust_label = robust_model == :t ? "Student-t" : "Skew-N"
    pathway = String(df.pathway[1])

    panels = [
        (:gauss_mean_abs_bias, :robust_mean_abs_bias, "Mean absolute bias"),
        (:gauss_mean_rmse, :robust_mean_rmse, "Mean RMSE"),
        (:gauss_mahal_mse, :robust_mahal_mse, "Mahalanobis MSE"),
        (:gauss_width95, :robust_width95, "Mean 95% interval width")
    ]

    fig = Figure(size = (780, 520), fontsize = FONT_BASE)
    axes = Axis[]
    for (j, (gcol, rcol, ylabel_str)) in enumerate(panels)
        row = j <= 2 ? 1 : 2
        col = isodd(j) ? 1 : 2
        ax = Axis(fig[row, col],
            xlabel = row == 2 ? xlabel_str : "",
            ylabel = ylabel_str,
            title = j == 1 ? "Point accuracy ($(pathway))" : ylabel_str,
            xscale = xscale_val,
            xlabelsize = FONT_XLABEL,
            ylabelsize = FONT_YLABEL,
            titlesize = FONT_TITLE,
            xticklabelsize = FONT_TICKLABEL,
            yticklabelsize = FONT_TICKLABEL)
        push!(axes, ax)
        lines!(ax, x, Vector{Float64}(df[!, gcol]), color = :steelblue,
            linewidth = 2.2)
        scatter!(ax, x, Vector{Float64}(df[!, gcol]), color = :steelblue,
            marker = :circle, markersize = 8)
        lines!(ax, x, Vector{Float64}(df[!, rcol]), color = :darkorange,
            linewidth = 2.2)
        scatter!(ax, x, Vector{Float64}(df[!, rcol]), color = :darkorange,
            marker = robust_model == :t ? :diamond : :star5, markersize = 9)
    end

    Legend(fig[1:2, 3],
        [
            [LineElement(color = :steelblue, linewidth = 2.2),
             MarkerElement(color = :steelblue, marker = :circle, markersize = 8)],
            [LineElement(color = :darkorange, linewidth = 2.2),
             MarkerElement(color = :darkorange,
                 marker = robust_model == :t ? :diamond : :star5, markersize = 9)]
        ],
        ["Gaussian", robust_label],
        framevisible = true,
        labelsize = FONT_LEGEND,
        tellwidth = true,
        padding = (6, 6, 4, 4),
        rowgap = 2)

    !isnothing(outpath) && save(outpath, fig)
    return fig
end
