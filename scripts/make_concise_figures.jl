# make_concise_figures.jl
#
# The two re-rendered figures of the concise manuscript.
# Plotting only: every value is read from a published supporting table, nothing is
# fitted or recomputed.
#
#   Fig 5  Fig5_conditional_repair.pdf
#          Leave-one-source-out coverage of nominal 95% intervals within each
#          enzyme-by-quantity-type stratum with at least 20 held-out records, for the
#          four pathways, under three constructions (S17 Table method codes):
#            convention  half-decade convention
#            global      fit-free replication record scale
#            tau_rel     per-type between-organism variance with the released
#                        fallback rule, on the replication scale
#
#   Fig 6  Fig6_calibration_curves.pdf
#          Record-averaged empirical against nominal coverage at the 21 nominal
#          levels of S18 Table, for glycolysis and the TCA cycle, under
#            convention   half-decade convention
#            global_cell  pre-specified default (iterated within-cell scale)
#            scale_mix2m  mean-matched two-class record population
#          Row 2 plots the same values as empirical minus nominal coverage.
#          Fixed-effects and organism-intercept curves are not drawn: no supporting
#          table holds their coverage at every nominal level (S10 Table has the 95%
#          level only, and S18 Table covers later constructions).
#
# Inputs:  supporting_tables/S17_Table.csv, S18_Table.csv
# Outputs: figures/figure/Fig5_conditional_repair.pdf
#          figures/figure/Fig6_calibration_curves.pdf
#          figures/figures_for_upload/Fig{5,6}.{png,tif}
#          (1575 px wide, 300 dpi, RGB; the TIFF is LZW-compressed), made from the
#          PDF with pdftoppm and sips as for the other upload files.
#
# Run: julia --project=. scripts/make_concise_figures.jl

import Pkg
Pkg.activate(dirname(@__DIR__))

using CairoMakie, CSV, DataFrames, Printf, Statistics

# PB_FIG_PKGDIR sets the folder whose figure/ and figures_for_upload/ receive the
# figures, PB_FIG_TABLEDIR the folder of the two tables, and PB_FIG_STRATA_TABLE and
# PB_FIG_LEVELS_TABLE their file names (defaults the concise package and its tables).
const PKGDIR = get(ENV, "PB_FIG_PKGDIR", joinpath(dirname(@__DIR__), "figures"))
const TABLEDIR = get(ENV, "PB_FIG_TABLEDIR", joinpath(dirname(@__DIR__), "supporting_tables"))
const STRATA_TABLE = get(ENV, "PB_FIG_STRATA_TABLE", "S17_Table.csv")
const LEVELS_TABLE = get(ENV, "PB_FIG_LEVELS_TABLE", "S18_Table.csv")
const FIGDIR = joinpath(PKGDIR, "figure")
const UPDIR = joinpath(PKGDIR, "figures_for_upload")
mkpath(FIGDIR); mkpath(UPDIR)

# Typography sized for print, as in make_revision_figures.jl: a 1500-unit canvas
# at the 5.25 in upload width (1575 px at 300 dpi) prints 0.253 pt per unit, so 32
# units is 8.1 pt, the required minimum; in the manuscript (6.5 in text width) it is
# larger still.
const FS_LABEL = 34
const FS_TICK = 32
const FS_TITLE = 36
const FS_LEGEND = 32
const LW = 3.0
const MS = 17

set_theme!(Theme(
    fontsize = FS_LABEL,
    Axis = (xlabelsize = FS_LABEL, ylabelsize = FS_LABEL,
            xticklabelsize = FS_TICK, yticklabelsize = FS_TICK,
            titlesize = FS_TITLE, xgridvisible = true, ygridvisible = true),
    Legend = (labelsize = FS_LEGEND, titlesize = FS_LEGEND, patchsize = (30, 18)),
    Label = (fontsize = FS_TITLE,),
))

# One colour per construction across the concise manuscript. Blue and orange are
# those of the organism-by-type hierarchy under the convention and of the estimated
# scale in Fig 4 (make_fig8_recalibration.jl); purple marks the per-type
# between-organism variance and the two-class population. Every construction also
# has its own marker, so identity never rests on colour alone.
const C_CONV = RGBf(0.20, 0.29, 0.55)   # #334A8C
const C_SCALE = RGBf(0.79, 0.35, 0.09)  # #C95917
const C_TYPE = RGBf(0.64, 0.40, 0.72)   # #A366B8

const PATHWAYS = [("glycolysis", "Glycolysis"), ("tca", "TCA cycle"),
                  ("pentose_phosphate", "Pentose phosphate pathway"),
                  ("shikimate", "Shikimate pathway")]

function _panel_label!(fig, row, col, text)
    Label(fig[row, col, TopLeft()], text, fontsize = FS_TITLE + 3, font = :bold,
          padding = (0, 12, 16, 0), halign = :right)
end

# "5.3.1.9|kcat" -> 5.3.1.9 k_cat, with the quantity type set as in the text.
function _stratum_label(s::AbstractString)
    ec, kind = split(s, "|")
    kind == "kcat" && return rich("$(ec)  k", subscript("cat"))
    kind == "KM" && return rich("$(ec)  K", subscript("M"))
    return "$(ec)  $(kind)"
end

"""
    export_upload(pdfpath, stem)

Rasterise a figure PDF to the upload formats: a 1575 px wide RGB PNG and an
LZW-compressed TIFF, both tagged 300 dpi (5.25 in at 300 dpi).
"""
function export_upload(pdfpath, stem)
    png = joinpath(UPDIR, "$(stem).png")
    tif = joinpath(UPDIR, "$(stem).tif")
    base = joinpath(UPDIR, stem)
    run(`pdftoppm -png -r 300 -scale-to-x 1575 -scale-to-y -1 -singlefile $pdfpath $base`)
    run(pipeline(`sips -s dpiWidth 300 -s dpiHeight 300 $png`, devnull))
    run(pipeline(`sips -s format tiff -s formatOptions lzw -s dpiWidth 300 -s dpiHeight 300 $png --out $tif`, devnull))
    println("wrote figures_for_upload/$(stem).png and $(stem).tif")
end

# ---------------------------------------------------------------------------
# Fig 5: conditional coverage and its repair
# ---------------------------------------------------------------------------

const FIG5_METHODS = [
    ("convention", C_CONV, :rect, MS, "half-decade convention"),
    ("global", C_SCALE, :diamond, MS + 3, "fit-free replication scale"),
    ("tau_rel", C_TYPE, :utriangle, MS + 2, "per-type between-organism variance"),
]

function figure5()
    s28 = CSV.read(joinpath(TABLEDIR, STRATA_TABLE), DataFrame;
                   types = Dict(:stratum => String))
    s28 = s28[(s28.level .== "enzyme") .& (s28.n_rec .>= 20), :]

    # Glycolysis and the TCA cycle carry the most strata, so each takes the full
    # width; the two smaller pathways share the third row. Tick labels are rotated
    # and the legend has its own row, so no text overprints a label or a marker.
    # At 1.0\textwidth of the manuscript (6.5 in, text height 8.86 in) the canvas
    # prints 7.4 in tall, leaving room for a caption of about 90 words.
    fig = Figure(size = (1500, 1700))
    slots = Dict("glycolysis" => (1, 1:2, "A"), "tca" => (2, 1:2, "B"),
                 "pentose_phosphate" => (3, 1, "C"), "shikimate" => (3, 2, "D"))
    dx = 0.24
    for (tag, name) in PATHWAYS
        sub = s28[s28.pathway .== tag, :]
        conv = sub[sub.method .== "convention", :]
        conv = conv[sortperm(collect(zip(conv.coverage, conv.stratum))), :]
        strata = conv.stratum
        cov = Dict(m => [sub[(sub.method .== m) .& (sub.stratum .== s), :coverage][1]
                         for s in strata] for (m, _) in FIG5_METHODS)
        r, cols, letter = slots[tag]
        ax = Axis(fig[r, cols];
            title = name,
            ylabel = "coverage (95%)",
            xticks = (1:length(strata), [_stratum_label(s) for s in strata]),
            xticklabelrotation = pi / 2,
            xgridvisible = false,
            yticks = 0.5:0.1:1.0)
        hlines!(ax, [0.95]; color = :black, linestyle = :dash, linewidth = 2.5)
        for i in eachindex(strata)
            lines!(ax, [i - dx, i, i + dx],
                   [cov["convention"][i], cov["global"][i], cov["tau_rel"][i]];
                   color = RGBAf(0, 0, 0, 0.25), linewidth = 2)
        end
        for (k, (m, c, mk, ms, _)) in enumerate(FIG5_METHODS)
            scatter!(ax, (1:length(strata)) .+ (k - 2) * dx, cov[m];
                     color = c, marker = mk, markersize = ms,
                     strokecolor = :white, strokewidth = 1)
        end
        xlims!(ax, 0.4, length(strata) + 0.6)
        ylims!(ax, 0.45, 1.05)   # headroom so markers at 1.0 clear the frame
        _panel_label!(fig, r, first(cols), letter)
        @printf("%-18s strata %2d  worst: convention %.3f  global %.3f  tau_rel %.3f\n",
                tag, length(strata), minimum(cov["convention"]),
                minimum(cov["global"]), minimum(cov["tau_rel"]))
    end
    Legend(fig[4, 1:2],
        Any[[MarkerElement(color = c, marker = mk, markersize = ms + 2)
             for (_, c, mk, ms, _) in FIG5_METHODS]...,
            LineElement(color = :black, linestyle = :dash, linewidth = 2.5)],
        [[lab for (_, _, _, _, lab) in FIG5_METHODS]..., "nominal 0.95"];
        orientation = :horizontal, nbanks = 2, framevisible = true,
        tellwidth = false, tellheight = true)
    colgap!(fig.layout, 1, 70)
    rowgap!(fig.layout, 3, 20)
    out = joinpath(FIGDIR, "Fig5_conditional_repair.pdf")
    save(out, fig)
    println("wrote figure/Fig5_conditional_repair.pdf")
    export_upload(out, "Fig5")
end

# ---------------------------------------------------------------------------
# Fig 6: calibration across all nominal levels
# ---------------------------------------------------------------------------

# All curves are solid; dashed lines are kept for the reference (exact calibration).
const FIG6_METHODS = [
    ("convention", C_CONV, :rect, :solid, "half-decade convention"),
    ("global_cell", C_SCALE, :diamond, :solid, "pre-specified default (single scale)"),
    ("scale_mix2m", C_TYPE, :utriangle, :solid, "two-class record population"),
]

function figure6()
    s29 = CSV.read(joinpath(TABLEDIR, LEVELS_TABLE), DataFrame)
    fig = Figure(size = (1500, 1500))
    dev_lo, dev_hi = Inf, -Inf
    devaxes = Axis[]
    for (col, (tag, name)) in enumerate(PATHWAYS[1:2])
        ax = Axis(fig[1, col]; title = name, xlabel = "nominal level",
                  ylabel = "empirical coverage",
                  xticks = 0:0.2:1, yticks = 0:0.2:1)
        axd = Axis(fig[2, col]; xlabel = "nominal level",
                   ylabel = "empirical − nominal", xticks = 0:0.2:1)
        push!(devaxes, axd)
        lines!(ax, [0, 1], [0, 1]; color = :black, linestyle = :dash, linewidth = 2.5)
        hlines!(axd, [0.0]; color = :black, linestyle = :dash, linewidth = 2.5)
        for (m, c, mk, ls, _) in FIG6_METHODS
            s = sort(s29[(s29.pathway .== tag) .& (s29.method .== m), :], :nominal)
            @assert nrow(s) == 21 "expected 21 nominal levels for $tag / $m"
            d = s.empirical .- s.nominal
            dev_lo = min(dev_lo, minimum(d)); dev_hi = max(dev_hi, maximum(d))
            lines!(ax, s.nominal, s.empirical; color = c, linestyle = ls, linewidth = LW)
            scatter!(ax, s.nominal, s.empirical; color = c, marker = mk,
                     markersize = MS - 4)
            lines!(axd, s.nominal, d; color = c, linestyle = ls, linewidth = LW)
            scatter!(axd, s.nominal, d; color = c, marker = mk, markersize = MS - 2)
            @printf("%-10s %-12s MAD %.3f  at 0.50 %.3f  at 0.95 %.3f\n", tag, m,
                    mean(abs.(d)), s.empirical[s.nominal .== 0.5][1],
                    s.empirical[s.nominal .== 0.95][1])
        end
        limits!(ax, -0.02, 1.02, -0.02, 1.02)
    end
    # Both deviation panels share one y range so the pathways compare directly.
    tlo = floor(dev_lo / 0.05) * 0.05
    thi = ceil(dev_hi / 0.05) * 0.05
    for axd in devaxes
        limits!(axd, -0.02, 1.02, tlo - 0.01, thi + 0.01)
        axd.yticks = collect(tlo:0.05:thi + 1e-9)
    end
    linkyaxes!(devaxes...)
    Legend(fig[3, 1:2],
        Any[[[LineElement(color = c, linestyle = ls, linewidth = LW),
              MarkerElement(color = c, marker = mk, markersize = MS)]
             for (_, c, mk, ls, _) in FIG6_METHODS]...,
            LineElement(color = :black, linestyle = :dash, linewidth = 2.5)],
        [[lab for (_, _, _, _, lab) in FIG6_METHODS]..., "exact calibration"];
        orientation = :horizontal, nbanks = 2, framevisible = true,
        tellwidth = false, tellheight = true)
    rowsize!(fig.layout, 1, Relative(0.47))
    colgap!(fig.layout, 1, 70)
    for (r, c, l) in [(1, 1, "A"), (1, 2, "B"), (2, 1, "C"), (2, 2, "D")]
        _panel_label!(fig, r, c, l)
    end
    out = joinpath(FIGDIR, "Fig6_calibration_curves.pdf")
    save(out, fig)
    println("wrote figure/Fig6_calibration_curves.pdf")
    export_upload(out, "Fig6")
end

figure5()
figure6()
println("Figures written to $FIGDIR and $UPDIR")
