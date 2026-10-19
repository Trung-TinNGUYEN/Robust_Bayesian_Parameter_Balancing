# make_revision_figures.jl
#
# Generates the new main-text figures for the revision from the CSV outputs of
#   run_revision_misspecification.jl
#   run_revision_realdata.jl
#   run_revision_downstream.jl
#
# Font sizes are set well above the defaults used in the previous submission, in
# response to the reviewer comment that figure fonts were too small.
#
# Output: figures/figure/
#   Fig2_residual_diagnostics.pdf
#   Fig3_misspecification.pdf
#   Fig4_calibration_curves.pdf
#   Fig5_loso_calibration.pdf
#   Fig8_downstream.pdf
#
# Environment controls:
#   PB_FIG_ONLY                 comma-separated subset of Fig2,Fig3,Fig4,Fig5,Fig7
#                               to draw (default all five)
#   PB_FIG_DOWNSTREAM_SUFFIX    suffix of the run_revision_downstream.jl outputs
#                               that Fig 7 reads (default empty; _priorheld for the
#                               propagation of the shorter manuscript, where it is
#                               Fig 5)
#   PB_FIG_OUTDIR               output folder (default the folder above)

import Pkg
Pkg.activate(dirname(@__DIR__))

using CairoMakie
using CSV
using DataFrames
using JLD2
using LinearAlgebra
using Printf
using Statistics

const OUTDIR = get(ENV, "PB_FIG_OUTDIR", joinpath(dirname(@__DIR__), "figures", "figure"))
const FIG_ONLY = strip.(split(get(ENV, "PB_FIG_ONLY", "Fig2,Fig3,Fig4,Fig5,Fig7"), ","))
const DOWNSTREAM_SUFFIX = get(ENV, "PB_FIG_DOWNSTREAM_SUFFIX", "")
mkpath(OUTDIR)

# Typography sized for print, not for the plotting canvas: a 1500-unit figure is
# placed at 1.0\textwidth = 379.4 pt, i.e. 0.253 printed pt per unit, so 32 units is
# 8.1 pt, the required minimum, and 36 units is 9.1 pt.
const FS_LABEL = 34
const FS_TICK = 32
const FS_TITLE = 36
const FS_LEGEND = 32
const LW = 3.0
const MS = 13

const THEME = Theme(
    fontsize = FS_LABEL,
    Axis = (xlabelsize = FS_LABEL, ylabelsize = FS_LABEL,
            xticklabelsize = FS_TICK, yticklabelsize = FS_TICK,
            titlesize = FS_TITLE, xgridvisible = true, ygridvisible = true),
    Legend = (labelsize = FS_LEGEND, titlesize = FS_LEGEND, patchsize = (26, 18)),
    Label = (fontsize = FS_TITLE,),
)
set_theme!(THEME)

const COLS = Dict(
    "gaussian"    => RGBf(0.20, 0.29, 0.55),
    "t_oracle"    => RGBf(0.79, 0.35, 0.09),
    "t_est"       => RGBf(0.93, 0.62, 0.20),
    "skew_oracle" => RGBf(0.12, 0.47, 0.31),
    "skew_est"    => RGBf(0.33, 0.71, 0.45),
)
const LABELS = Dict(
    "gaussian"    => "Gaussian",
    "t_oracle"    => "Student-t (ν known)",
    "t_est"       => "Student-t (ν estimated)",
    "skew_oracle" => "Skew-normal (α known)",
    "skew_est"    => "Skew-normal (α estimated)",
)

_read(path) = isfile(path) ? DataFrame(CSV.File(path)) : nothing

function _panel_label!(fig, row, col, text)
    Label(fig[row, col, TopLeft()], text, fontsize = FS_TITLE + 3, font = :bold,
          padding = (0, 12, 16, 0), halign = :right)
end

# ---------------------------------------------------------------------------
# Fig 2: residual diagnostics on the curated data
# ---------------------------------------------------------------------------

_skew(x) = (m = mean(x); s = std(x, corrected = false); s <= 0 ? 0.0 : mean(((x .- m) ./ s) .^ 3))

function residuals_for(pathway::Symbol)
    root = joinpath(dirname(@__DIR__), "output", string(pathway))
    b = load(joinpath(root, "data", "brenda_inputs_$(pathway).jld2"))
    p = load(joinpath(root, "data", "posterior_$(pathway).jld2"))
    y = Vector{Float64}(b["y"]); s2 = Vector{Float64}(b["σ2"])
    A = Matrix{Float64}(b["Arows"]); g = Vector{String}(b["grp"])
    kinds = Vector{String}(getproperty(b["meta"], :kind))
    q0 = vec(Array(p["q_prior"])); C0 = Matrix(Array(p["C_prior"]))
    C0inv = inv(Symmetric(C0))
    P = C0inv + A' * (A ./ s2)
    qhat = Symmetric(P) \ (A' * (y ./ s2) + C0inv * q0)
    r = (y .- A * qhat) ./ sqrt.(s2)
    return r, g, kinds
end

function figure2()
    # Legends sit in their own rows so that no legend box covers a curve or a bar,
    # and each standard deviation is printed where the data leave the panel empty.
    fig = Figure(size = (1500, 1330))
    c_km = RGBf(0.79, 0.35, 0.09); c_kc = RGBf(0.12, 0.47, 0.31)
    for (col, pw, name) in [(1, :glycolysis, "Glycolysis"), (2, :tca, "TCA cycle")]
        r, g, kinds = residuals_for(pw)

        ax = Axis(fig[1, col], title = name,
                  xlabel = "standardised residual", ylabel = "density")
        hist!(ax, clamp.(r, -12, 12), bins = 70, normalization = :pdf,
              color = (:steelblue, 0.55), strokewidth = 0.5, strokecolor = :white)
        xs = range(-12, 12, length = 400)
        lines!(ax, xs, (1 / sqrt(2pi)) .* exp.(-0.5 .* xs .^ 2), color = :black,
               linewidth = LW)
        xlims!(ax, -12, 12)
        text!(ax, 0.02, 0.96; text = @sprintf("SD = %.2f", std(r)), space = :relative,
              align = (:left, :top), fontsize = FS_LEGEND)

        ax2 = Axis(fig[3, col], xlabel = "standardised residual", ylabel = "density")
        els = Any[]; labs = Any[]
        for (k, c) in zip(["KM", "kcat"], [c_km, c_kc])
            v = r[kinds .== k]
            isempty(v) && continue
            density!(ax2, clamp.(v, -12, 12), color = (c, 0.35), strokecolor = c,
                     strokewidth = LW)
            push!(els, PolyElement(color = (c, 0.35), strokecolor = c, strokewidth = LW))
            push!(labs, k == "kcat" ?
                  rich("k", subscript("cat"), @sprintf("  (skew %.2f)", _skew(v))) :
                  rich("K", subscript("M"), @sprintf("  (skew %.2f)", _skew(v))))
        end
        xlims!(ax2, -12, 12)
        Legend(fig[4, col], els, labs; orientation = :horizontal, framevisible = true,
               tellwidth = false, tellheight = true, labelsize = FS_LEGEND)

        ax3 = Axis(fig[5, col], xlabel = "mean standardised residual per organism",
                   ylabel = "number of organisms")
        src = unique(g); ns = [count(==(s), g) for s in src]
        mus = [mean(r[g .== s]) for s in src]
        keep = ns .>= 5
        hist!(ax3, mus[keep], bins = 24, color = (:darkorange, 0.6),
              strokewidth = 0.5, strokecolor = :white)
        vlines!(ax3, [0.0], color = :black, linewidth = LW, linestyle = :dash)
        text!(ax3, 0.98, 0.96; text = @sprintf("SD = %.2f", std(mus[keep])),
              space = :relative, align = (:right, :top), fontsize = FS_LEGEND)
    end
    Legend(fig[2, 1:2],
           [PolyElement(color = (:steelblue, 0.55)), LineElement(color = :black, linewidth = LW)],
           ["standardised residuals", "N(0,1) implied by the half-decade convention"];
           orientation = :horizontal, framevisible = true, tellwidth = false,
           tellheight = true, labelsize = FS_LEGEND)
    _panel_label!(fig, 1, 1, "A"); _panel_label!(fig, 1, 2, "B")
    _panel_label!(fig, 3, 1, "C"); _panel_label!(fig, 3, 2, "D")
    _panel_label!(fig, 5, 1, "E"); _panel_label!(fig, 5, 2, "F")
    colgap!(fig.layout, 1, 40)
    save(joinpath(OUTDIR, "Fig2_residual_diagnostics.pdf"), fig)
    println("wrote Fig2_residual_diagnostics.pdf")
end

# ---------------------------------------------------------------------------
# Fig 3: coverage against the data-generating shape parameter
# ---------------------------------------------------------------------------

"""
    _shared_legend!(fig, row, entries)

Build a legend covering every fitted model, rather than only those present in one
panel.  `entries` is a vector of keys into `COLS` and `LABELS`.
"""
function _shared_legend!(fig, row, entries; nbanks = 2)
    els = [[LineElement(color = COLS[k], linewidth = LW),
            MarkerElement(color = COLS[k], marker = :circle, markersize = MS)]
           for k in entries]
    Legend(fig[row, 1:2], els, [LABELS[k] for k in entries],
           orientation = :horizontal, nbanks = nbanks, framevisible = true,
           labelsize = FS_LEGEND)
end

function figure3(; design = "record", setting = "scale", n_full = 100)
    fig = Figure(size = (1500, 1150))
    panels = [(1, 1, :glycolysis, "t", "Glycolysis, Student-t data"),
              (1, 2, :tca, "t", "TCA cycle, Student-t data"),
              (2, 1, :glycolysis, "skew", "Glycolysis, skew-normal data"),
              (2, 2, :tca, "skew", "TCA cycle, skew-normal data")]
    for (row, col, pw, dgp, title) in panels
        df = _read(joinpath(dirname(@__DIR__), "output", string(pw), "results", "revision",
                            "misspec_grid_$(pw).csv"))
        df === nothing && continue
        sub = filter(r -> r.design == design && r.setting == setting && r.dgp == dgp, df)
        isempty(sub) && continue
        xlab = dgp == "t" ? "degrees of freedom ν" : "skewness α"
        # Evenly spaced tick labels: the sweep has more points than can be labelled
        # without crowding, and every point still carries a marker.
        tv = dgp == "t" ? [2.5, 5.0, 10.0, 20.0, 50.0] : [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        ax = Axis(fig[row, col], title = title, xlabel = xlab,
                  ylabel = "coverage (95%)",
                  xscale = dgp == "t" ? log10 : identity,
                  xticks = (tv, [string(isinteger(v) ? Int(v) : v) for v in tv]))
        for fitname in unique(sub.fit)
            s = sort(filter(r -> r.fit == fitname, sub), :sweep_val)
            # The cross-family skew-normal fit to heavy-tailed data loses most replicates
            # to sampler failures at small nu (S8 Table), so its curve is drawn only where
            # at least 85 of the replicates are usable.
            dgp == "t" && fitname == "skew_est" && (s = filter(r -> r.n_reps >= 85, s))
            lines!(ax, s.sweep_val, s.cov95, color = COLS[fitname], linewidth = LW,
                   label = LABELS[fitname])
            # Open markers flag cells in which sampler failures reduced the number
            # of usable replicates; those coverage estimates are unreliable.
            full = s.n_reps .>= n_full
            scatter!(ax, s.sweep_val[full], s.cov95[full], color = COLS[fitname],
                     markersize = MS)
            if any(.!full)
                scatter!(ax, s.sweep_val[.!full], s.cov95[.!full], color = :white,
                         strokecolor = COLS[fitname], strokewidth = 2.5,
                         marker = :diamond, markersize = MS + 3)
            end
        end
        hlines!(ax, [0.95], color = :black, linestyle = :dash, linewidth = 2)
        ylims!(ax, 0.0, 1.03)
    end
    _shared_legend!(fig, 3, ["gaussian", "t_oracle", "t_est", "skew_oracle", "skew_est"];
                    nbanks = 2)
    colgap!(fig.layout, 1, 40)
    for (r, c, l) in [(1, 1, "A"), (1, 2, "B"), (2, 1, "C"), (2, 2, "D")]
        _panel_label!(fig, r, c, l)
    end
    save(joinpath(OUTDIR, "Fig3_misspecification.pdf"), fig)
    println("wrote Fig3_misspecification.pdf")
end

# ---------------------------------------------------------------------------
# Fig 4: nominal against empirical coverage across levels
# ---------------------------------------------------------------------------

function figure4(; design = "record", setting = "scale", nu_pt = 3.0, alpha_pt = 4.0)
    fig = Figure(size = (1500, 1000))
    panels = [(1, 1, :glycolysis, "t", nu_pt, "Glycolysis, Student-t data (ν = $(nu_pt))"),
              (1, 2, :tca, "t", nu_pt, "TCA cycle, Student-t data (ν = $(nu_pt))"),
              (2, 1, :glycolysis, "skew", alpha_pt, "Glycolysis, skew-normal data (α = $(alpha_pt))"),
              (2, 2, :tca, "skew", alpha_pt, "TCA cycle, skew-normal data (α = $(alpha_pt))")]
    for (row, col, pw, dgp, val, title) in panels
        df = _read(joinpath(dirname(@__DIR__), "output", string(pw), "results", "revision",
                            "misspec_levels_$(pw).csv"))
        df === nothing && continue
        sub = filter(r -> r.design == design && r.setting == setting && r.dgp == dgp &&
                          isapprox(r.sweep_val, val; atol = 1e-6), df)
        isempty(sub) && continue
        ax = Axis(fig[row, col], title = title, xlabel = "nominal level",
                  ylabel = "empirical coverage")
        lines!(ax, [0, 1], [0, 1], color = :black, linestyle = :dash, linewidth = 2)
        for fitname in unique(sub.fit)
            s = sort(filter(r -> r.fit == fitname, sub), :nominal)
            lines!(ax, s.nominal, s.empirical, color = COLS[fitname], linewidth = LW,
                   label = LABELS[fitname])
            scatter!(ax, s.nominal, s.empirical, color = COLS[fitname], markersize = MS - 4)
        end
        limits!(ax, 0, 1, 0, 1)
    end
    _shared_legend!(fig, 3, ["gaussian", "t_oracle", "t_est", "skew_oracle", "skew_est"];
                    nbanks = 2)
    for (r, c, l) in [(1, 1, "A"), (1, 2, "B"), (2, 1, "C"), (2, 2, "D")]
        _panel_label!(fig, r, c, l)
    end
    save(joinpath(OUTDIR, "Fig4_calibration_curves.pdf"), fig)
    println("wrote Fig4_calibration_curves.pdf")
end

# ---------------------------------------------------------------------------
# Fig 5: real-data leave-one-source-out calibration
# ---------------------------------------------------------------------------

# The skew-normal cell is deliberately absent: on real data the estimated shape is
# driven to the boundary of the family and the fits return non-finite predictive
# scores, so those results are withdrawn rather than plotted.  See S1 Text.
const CELL_STYLE = Dict(
    "gaussian_Fix"           => (RGBf(0.20, 0.29, 0.55), :solid,  "Gaussian, fixed effects"),
    "gaussian_RE_org"        => (RGBf(0.35, 0.55, 0.85), :solid,  "Gaussian, organism effects"),
    "gaussian_RE_orgtype"    => (RGBf(0.55, 0.72, 0.95), :dash,   "Gaussian, organism × type effects"),
    "t_fixed50_RE_org"       => (RGBf(0.79, 0.35, 0.09), :dot,    "Student-t (ν = 50), organism effects"),
    "t_moment_Fix"           => (RGBf(0.60, 0.25, 0.10), :solid,  "Student-t (ν estimated), fixed effects"),
    "t_moment_RE_org"        => (RGBf(0.93, 0.62, 0.20), :solid,  "Student-t (ν estimated), organism effects"),
    "t_moment_RE_orgtype"    => (RGBf(0.95, 0.78, 0.35), :dash,   "Student-t (ν estimated), organism × type"),
)

function figure5()
    fig = Figure(size = (1500, 820), figure_padding = (16, 44, 16, 16))
    axes_list = Axis[]
    for (col, pw, name) in [(1, :glycolysis, "Glycolysis"), (2, :tca, "TCA cycle")]
        df = _read(joinpath(dirname(@__DIR__), "output", string(pw), "results", "revision",
                            "realdata_levels_$(pw).csv"))
        df === nothing && continue
        ax = Axis(fig[1, col], title = name, xlabel = "nominal level",
                  ylabel = "empirical coverage")
        push!(axes_list, ax)
        lines!(ax, [0, 1], [0, 1], color = :black, linestyle = :dash, linewidth = 2)
        for cell in unique(df.cell)
            # The three Gaussian models only, as the S1 Text caption says.  The file holds
            # every cell after the full run of run_revision_realdata.jl, and the Student-t
            # results are in the supporting tables.
            (haskey(CELL_STYLE, cell) && startswith(cell, "gaussian")) || continue
            c, ls, lab = CELL_STYLE[cell]
            s = sort(filter(r -> r.cell == cell, df), :nominal)
            lines!(ax, s.nominal, s.empirical, color = c, linestyle = ls,
                   linewidth = LW, label = lab)
        end
        limits!(ax, 0, 1, 0, 1)
    end
    if !isempty(axes_list)
        Legend(fig[2, 1:2], axes_list[1], orientation = :horizontal, nbanks = 3,
               framevisible = true)
    end
    colgap!(fig.layout, 1, 50)
    _panel_label!(fig, 1, 1, "A"); _panel_label!(fig, 1, 2, "B")
    save(joinpath(OUTDIR, "Fig5_loso_calibration.pdf"), fig)
    println("wrote Fig5_loso_calibration.pdf")
end

# ---------------------------------------------------------------------------
# Fig 7: downstream propagation
# ---------------------------------------------------------------------------

function figure7()
    # One column per pathway and one row per quantity.  Reaction identifiers are not
    # printed: the bars keep the same reaction order in rows A to C, S37 and S38
    # Tables give every value, and dropping the rotated labels leaves each panel
    # tall enough for its ticks at the printed type size.
    fig = Figure(size = (1500, 1580))   # fits one page with its caption
    cells_order = ["gaussian_Fix", "gaussian_RE_org", "gaussian_RE_orgtype",
                   "t_moment_RE_org", "t_moment_RE_orgtype"]
    short = Dict("gaussian_Fix" => "Gaussian, fixed effects",
                 "gaussian_RE_org" => "Gaussian, organism",
                 "gaussian_RE_orgtype" => "Gaussian, organism × type",
                 "t_moment_RE_org" => "Student-t, organism",
                 "t_moment_RE_orgtype" => "Student-t, organism × type")
    seen = String[]
    barax(r, col, title, ylab; bottom = false) = Axis(fig[r, col]; title = title,
        ylabel = ylab, xlabel = bottom ? "reaction" : "",
        xticksvisible = false, xticklabelsvisible = false, xgridvisible = false,
        yticks = WilkinsonTicks(5))

    for (col, pw, name) in [(1, :glycolysis, "Glycolysis"), (2, :tca, "TCA cycle")]
        base = joinpath(dirname(@__DIR__), "output", string(pw), "results", "revision")
        rx = _read(joinpath(base, "downstream_reactions_$(pw)$(DOWNSTREAM_SUFFIX).csv"))
        el = _read(joinpath(base, "downstream_elasticity_$(pw)$(DOWNSTREAM_SUFFIX).csv"))
        rx === nothing && continue
        Label(fig[0, col], name; fontsize = FS_TITLE + 2, font = :bold, tellwidth = false)

        pop = filter(r -> r.target == "population", rx)
        present = [c for c in cells_order if c in unique(pop.cell)]
        append!(seen, present)
        rxns = unique(pop.reaction)
        nb = length(present)

        ax = barax(1, col, "Rate, consensus parameters", "width, log₁₀ v")
        for (k, cell) in enumerate(present)
            s = filter(r -> r.cell == cell, pop)
            idx = [findfirst(==(r), rxns) for r in s.reaction]
            off = (k - (nb + 1) / 2) * (0.8 / nb)
            barplot!(ax, idx .+ off, s.logv_ci95_width, width = 0.8 / nb,
                     color = CELL_STYLE[cell][1])
        end

        newo = filter(r -> r.target == "new_organism", rx)
        ax2 = barax(2, col, "Rate, new organism", "width, log₁₀ v")
        present2 = [c for c in cells_order if c in unique(newo.cell)]
        append!(seen, present2)
        nb2 = max(length(present2), 1)
        # The fixed-effects model cannot produce this quantity; its consensus
        # interval is drawn as a reference line to make the gap visible.
        fixref = filter(r -> r.cell == "gaussian_Fix", pop)
        if !isempty(fixref)
            hlines!(ax2, [median(fixref.logv_ci95_width)], color = :black,
                    linestyle = :dash, linewidth = 2.5)
        end
        for (k, cell) in enumerate(present2)
            s = filter(r -> r.cell == cell, newo)
            idx = [findfirst(==(r), rxns) for r in s.reaction]
            off = (k - (nb2 + 1) / 2) * (0.8 / nb2)
            barplot!(ax2, idx .+ off, s.logv_ci95_width, width = 0.8 / nb2,
                     color = CELL_STYLE[cell][1])
        end

        ax3 = barax(3, col, "Equilibrium constant",
                    rich("width, log₁₀ K", subscript("eq")); bottom = true)
        for (k, cell) in enumerate(present)
            s = filter(r -> r.cell == cell, pop)
            idx = [findfirst(==(r), rxns) for r in s.reaction]
            off = (k - (nb + 1) / 2) * (0.8 / nb)
            barplot!(ax3, idx .+ off, s.logKeq_ci95_width, width = 0.8 / nb,
                     color = CELL_STYLE[cell][1])
        end

        if el !== nothing && !isempty(el)
            ax4 = Axis(fig[4, col]; title = "Substrate elasticity",
                       xlabel = "95% width, ∂log v / ∂log c", ylabel = "density",
                       yticks = WilkinsonTicks(5))
            for cell in present
                v = filter(r -> r.cell == cell && r.target == "population" &&
                                isfinite(r.elasticity_ci95_width), el)
                isempty(v) && continue
                c = CELL_STYLE[cell][1]
                density!(ax4, v.elasticity_ci95_width, color = (c, 0.25),
                         strokecolor = c, strokewidth = LW)
            end
        end
    end
    order = [c for c in cells_order if c in seen]
    els = Any[PolyElement(color = CELL_STYLE[c][1]) for c in order]
    labs = Any[short[c] for c in order]
    push!(els, LineElement(color = :black, linestyle = :dash, linewidth = 2.5))
    push!(labs, "fixed effects, consensus median")
    Legend(fig[5, 1:2], els, labs; orientation = :horizontal, nbanks = 3,
           framevisible = true, tellwidth = false, tellheight = true)
    for (r, c, l) in [(1, 1, "A"), (1, 2, "B"), (2, 1, "C"), (2, 2, "D"),
                      (3, 1, "E"), (3, 2, "F"), (4, 1, "G"), (4, 2, "H")]
        _panel_label!(fig, r, c, l)
    end
    colgap!(fig.layout, 1, 50)
    save(joinpath(OUTDIR, "Fig8_downstream.pdf"), fig)
    println("wrote Fig8_downstream.pdf")
end

for (name, f) in [("Fig2", figure2), ("Fig3", figure3), ("Fig4", figure4),
                  ("Fig5", figure5), ("Fig7", figure7)]
    name in FIG_ONLY || continue
    try
        f()
    catch err
        @warn "$name failed" exception = (err, catch_backtrace())
    end
end
println("Figures written to $OUTDIR")
