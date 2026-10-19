# make_fig1_overview.jl
#
# Fig 1 for the revision: a workflow schematic of
# the pipeline, from curated records to a checked posterior and its downstream
# use.
#
# Replaces the earlier hand-drawn overview, which opposed a "Bayesian
# Parameter Balancing Framework" to "Classical Parameter Balancing
# (Limitations)".  Reviewer 1 objected to exactly that framing, on the correct
# ground that classical parameter balancing is itself a Bayesian procedure, and
# the revised text says so; the figure now uses the same neutral terminology as
# the text, and names what is actually added rather than what is allegedly
# missing elsewhere.
#
# Output: figures/figure/Fig1_overview.png and figures_for_upload/Fig1.{png,pdf,tif}

import Pkg
Pkg.activate(dirname(@__DIR__))

using CairoMakie, CSV, DataFrames, Printf

# PB_FIG1_CONCISE=1 writes the concise package's version (no conformal box line).
# PB_FIG_PKGDIR sets the package folder that receives figure/ and figures_for_upload/.
const CONCISE = get(ENV, "PB_FIG1_CONCISE", "0") == "1"
const SUB = get(ENV, "PB_FIG_PKGDIR", joinpath(dirname(@__DIR__), "figures"))
const OUTDIR = joinpath(SUB, "figure")
const UPDIR = joinpath(SUB, "figures_for_upload")
mkpath(OUTDIR); mkpath(UPDIR)

# 1680-unit canvas at 1.0\textwidth = 379.4 pt, i.e. 0.226 printed pt per unit, so
# 36 units is 8.1 pt, the required minimum.
const FS_T = 42     # box title
const FS_B = 36     # box body
const FS_S = 36     # stage label

const C_DATA = RGBf(0.20, 0.29, 0.55)
const C_MODEL = RGBf(0.12, 0.47, 0.31)
const C_CHECK = RGBf(0.79, 0.35, 0.09)
const C_USE = RGBf(0.42, 0.24, 0.51)

function box!(ax, x, y, w, h, col, title, lines; titlecol = col)
    poly!(ax, Point2f[(x, y), (x + w, y), (x + w, y + h), (x, y + h)];
          color = RGBAf(col.r, col.g, col.b, 0.10), strokecolor = col,
          strokewidth = 2.5)
    text!(ax, x + 0.5w, y + h - 0.24; text = title, align = (:center, :top),
          fontsize = FS_T, font = :bold, color = titlecol)
    for (i, l) in enumerate(lines)
        text!(ax, x + 0.5w, y + h - 0.82 - 0.44 * (i - 1); text = l,
              align = (:center, :top), fontsize = FS_B, color = :black)
    end
end

arrow!(ax, x1, y1, x2, y2; col = RGBf(0.35, 0.35, 0.35)) =
    arrows!(ax, [x1], [y1], [x2 - x1], [y2 - y1]; color = col, linewidth = 3.5,
            arrowsize = 20)

fig = Figure(size = (1680, 1650), backgroundcolor = :white)
ax = Axis(fig[1, 1]; aspect = DataAspect())
hidedecorations!(ax); hidespines!(ax)
xlims!(ax, 0, 16.8); ylims!(ax, 0, 16.2)

# Boxes sit on a two-column grid so that each is wide enough for 8 pt body text
# (at most about 34 characters a line), and every arrow runs through a gap between
# boxes, never across text.
const XL, XR, BW = 0.3, 8.9, 7.6          # column origins and box width
const R1, R2, R3, RH = 12.9, 9.0, 5.1, 3.0  # row origins and box height

box!(ax, XL, R1, BW, RH, C_DATA, "Curated records",
     [rich("BRENDA k", subscript("cat"), " and K", subscript("M"), " values,"),
      "each with its source organism",
      "eQuilibrator formation",
      "energies and covariance"])
box!(ax, XR, R1, BW, RH, C_MODEL, "Independent basis",
     ["Dependency matrix Q maps a", "basis q to every quantity",
      "Haldane relations hold", "exactly for every draw"])
box!(ax, XR, R2, BW, RH, C_MODEL, "Record-level model",
     ["One likelihood term per record", "Organism-by-type offsets",
      "Record error scale estimated", "from replication"])
box!(ax, XL, R2, BW, RH, C_MODEL, "Joint posterior on q",
     ["Collapsed sampler", "Sparse precision scales", "to genome size",
      "All derived quantities follow"])
box!(ax, XL, R3, BW, RH, C_CHECK, "Uncertainty checked",
     ["Leave-one-source-out coverage", "Marginal and within-stratum",
      (CONCISE ? ["Averaged per record", "and per organism"] :
                 ["Conformal option for a", "finite-sample certificate"])...])
box!(ax, XR, R3, BW, RH, C_USE, "Downstream use",
     ["Convenience rate law", "Rates, equilibrium constants", "and elasticities",
      "Consensus vs new organism"])
# The summary box names the data it speaks for (a co-author read "the half-decade
# convention understates record dispersion about 2.5-fold" as a general claim).
# Every coverage number is the record-averaged leave-one-source-out coverage of nominal
# 95% intervals on the four pathways, as in Results.  The assumed record uncertainty is
# given as a number (the first author asked on 9 October 2026 what "half-decade
# uncertainty" means), and it is named again where the Student-t result is stated,
# because the sentence in between introduces an estimated scale.  The caption says
# that it is assigned to every kinetic record, and the model component is named as in
# the box above it ("Organism-by-type offsets").  No colon or semicolon, as in the
# manuscript.  The two coverage ranges are read from the leave-one-source-out summaries
# (standard parameter balancing from realdata_summary, the pre-specified default from
# sigmacal_summary), so that the figure follows the outputs.
function coverage_range(stem, cell)
    v = Float64[]
    for pw in ("glycolysis", "tca", "pentose_phosphate", "shikimate")
        df = CSV.read(joinpath(dirname(@__DIR__), "output", pw, "results", "revision",
                               "$(stem)_$(pw).csv"), DataFrame)
        push!(v, only(df.Cov95[df.cell .== cell]))
    end
    return @sprintf("%.2f to %.2f", minimum(v), maximum(v))
end
const COV_SPB = CONCISE ? coverage_range("realdata_summary", "gaussian_Fix") : ""
const COV_DEFAULT = CONCISE ? coverage_range("sigmacal_summary", "gaussian_sigmacal_RE_orgtype") : ""
const SUMMARY_TITLE = CONCISE ? "What the checks establish on the four curated pathways" :
                                "What the checks establish"
const SUMMARY_LINES = CONCISE ?
    [rich("Standard parameter balancing with an assumed log", subscript("10"),
          " standard deviation of 0.5"),
     "covers only $(COV_SPB) of held-out records at a nominal level of 0.95.",
     "Organism-by-type offsets and a record error scale estimated from replication",
     "raise this coverage to $(COV_DEFAULT) at unchanged point accuracy.",
     rich("With these offsets at the assumed log", subscript("10"),
          " standard deviation of 0.5, a Student-t"),
     "likelihood trades coverage for sharpness and a skew-normal likelihood fails."] :
    ["The half-decade convention understates record dispersion",
     "about 2.5-fold, so intervals built on it are overconfident.",
     "Organism structure and an estimated error scale restore",
     "held-out coverage; robust likelihoods trade it for sharpness.",
     "In the pooled records turnover numbers vary more between",
     "organisms than Michaelis constants; stratifying repairs them."]
box!(ax, XL, 0.3, 16.2, 3.9, RGBf(0.3, 0.3, 0.3), SUMMARY_TITLE, SUMMARY_LINES)

xm = XL + BW                                             # right edge of the left column
arrow!(ax, xm, R1 + RH / 2, XR, R1 + RH / 2)             # records -> basis
arrow!(ax, XR + BW / 2, R1, XR + BW / 2, R2 + RH)        # basis -> model
arrow!(ax, XR, R2 + RH / 2, xm, R2 + RH / 2)             # model -> posterior
arrow!(ax, XL + BW / 2, R2, XL + BW / 2, R3 + RH)        # posterior -> checks
arrow!(ax, xm, R2, XR, R3 + RH)                          # posterior -> downstream

save(joinpath(OUTDIR, "Fig1_overview.png"), fig; px_per_unit = 300 / 96)
save(joinpath(UPDIR, "Fig1.png"), fig; px_per_unit = 300 / 96)
save(joinpath(UPDIR, "Fig1.pdf"), fig)
# Upload TIFF by the route of make_concise_figures.jl (rasterise the PDF at 1575 px wide,
# then LZW TIFF at 300 dpi), which gives an RGB file with no alpha channel.
let base = joinpath(UPDIR, "Fig1_from_pdf"), tif = joinpath(UPDIR, "Fig1.tif")
    run(`pdftoppm -png -r 300 -scale-to-x 1575 -scale-to-y -1 -singlefile $(joinpath(UPDIR, "Fig1.pdf")) $base`)
    run(pipeline(`sips -s format tiff -s formatOptions lzw -s dpiWidth 300 -s dpiHeight 300 $(base * ".png") --out $tif`, devnull))
    rm(base * ".png")
end
println("wrote Fig1_overview.png and figures_for_upload/Fig1.{png,pdf,tif}")
