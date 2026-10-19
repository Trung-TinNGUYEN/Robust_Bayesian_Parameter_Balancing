# make_fig8_recalibration.jl
#
# Fig 7 for the revision: the four-pathway
# recalibration story, previously visible only in the supporting tables.
#
# Panel A: leave-one-source-out coverage of nominal 95% intervals per pathway
#   for (i) the fixed-effects model, (ii) the organism-by-type hierarchy under
#   the half-decade convention, (iii) the same hierarchy with the record-level
#   error scale estimated on training folds, per record and per held-out
#   organism.
# Panel B: the marginal residual dispersion factor and the estimated
#   within-cell error scale (standard-deviation units) per pathway.
#
# Inputs: output/<pathway>/results/revision/{realdata_summary,sigmacal_summary,
# realdata_residuals}_<pathway>.csv for the four pathways.
# Output: figures/figure/Fig7_recalibration.pdf and figures_for_upload/Fig5.{pdf,png}

import Pkg
Pkg.activate(dirname(@__DIR__))

using CairoMakie, CSV, DataFrames, Printf, Statistics

# PB_FIG_PKGDIR sets the folder whose figure/ and figures_for_upload/ receive the figure.
const SUB = get(ENV, "PB_FIG_PKGDIR", joinpath(dirname(@__DIR__), "figures"))
const OUTDIR = joinpath(SUB, "figure")
const UPDIR = joinpath(SUB, "figures_for_upload")
mkpath(OUTDIR); mkpath(UPDIR)

# Print sizes: 1500 units at 379.4 pt = 0.253 pt per unit; 32 units = 8.1 pt.
const FS_LABEL = 34
const FS_TICK = 32
const FS_TITLE = 36
const FS_LEGEND = 32
const MS = 16

set_theme!(Theme(
    fontsize = FS_LABEL,
    Axis = (xlabelsize = FS_LABEL, ylabelsize = FS_LABEL,
            xticklabelsize = FS_TICK, yticklabelsize = FS_TICK,
            titlesize = FS_TITLE, xgridvisible = true, ygridvisible = true),
    Legend = (labelsize = FS_LEGEND, titlesize = FS_LEGEND, patchsize = (26, 18)),
    Label = (fontsize = FS_TITLE,),
))

const PATHWAYS = [("glycolysis", "Glycolysis"), ("tca", "TCA cycle"),
                  ("pentose_phosphate", "Pentose\nphosphate"), ("shikimate", "Shikimate")]

getrow(df, cell) = df[findfirst(==(cell), df.cell), :]

cov_fix = Float64[]; cov_raw = Float64[]; cov_cal = Float64[]; cov_org = Float64[]
disp = Float64[]; chat_sd = Float64[]
for (tag, _) in PATHWAYS
    rd = CSV.read(joinpath(dirname(@__DIR__), "output", tag, "results", "revision",
                           "realdata_summary_$(tag).csv"), DataFrame)
    sc = CSV.read(joinpath(dirname(@__DIR__), "output", tag, "results", "revision",
                           "sigmacal_summary_$(tag).csv"), DataFrame)
    rs = CSV.read(joinpath(dirname(@__DIR__), "output", tag, "results", "revision",
                           "realdata_residuals_$(tag).csv"), DataFrame)
    push!(cov_fix, getrow(rd, "gaussian_Fix").Cov95)
    push!(cov_raw, getrow(rd, "gaussian_RE_orgtype").Cov95)
    push!(cov_cal, getrow(sc, "gaussian_sigmacal_RE_orgtype").Cov95)
    cf = CSV.read(joinpath(dirname(@__DIR__), "output", tag, "results", "revision",
                           "conformal_summary_$(tag).csv"), DataFrame)
    push!(cov_org, cf[findfirst(==("gaussian"), cf.method), :Cov95_org])
    push!(disp, rs[findfirst(==("all"), rs.subset), :sd])
    push!(chat_sd, sqrt(getrow(sc, "gaussian_sigmacal_RE_orgtype").c_all))
end

@printf("coverage fix: %s\n", join(round.(cov_fix, digits = 3), " "))
@printf("coverage raw: %s\n", join(round.(cov_raw, digits = 3), " "))
@printf("coverage cal: %s\n", join(round.(cov_cal, digits = 3), " "))
@printf("coverage org: %s\n", join(round.(cov_org, digits = 3), " "))
@printf("dispersion:   %s\n", join(round.(disp, digits = 2), " "))
@printf("chat (sd):    %s\n", join(round.(chat_sd, digits = 2), " "))

# Two panels stacked, so that each has the full width for its axis label and its
# four pathway tick labels at the printed type size (8 pt at 5.25 inches).
fig = Figure(size = (1500, 1500))

ax1 = Axis(fig[1, 1];
    ylabel = "Coverage (95%)",
    xticks = (1:4, [lab for (_, lab) in PATHWAYS]),
    yticks = 0.5:0.1:1.0,
    title = "Leave-one-source-out calibration")
ylims!(ax1, 0.45, 1.0)
hlines!(ax1, [0.95]; color = :black, linestyle = :dash, linewidth = 2.5)
x = 1:4
c_fix = RGBf(0.55, 0.55, 0.55)
c_raw = RGBf(0.20, 0.29, 0.55)
c_cal = RGBf(0.79, 0.35, 0.09)
for (i, xi) in enumerate(x)
    lines!(ax1, [xi, xi], [cov_fix[i], cov_org[i]]; color = RGBAf(0, 0, 0, 0.25),
           linewidth = 2)
end
s1 = scatter!(ax1, x .- 0.12, cov_fix; color = c_fix, markersize = MS, marker = :rect)
s2 = scatter!(ax1, x, cov_raw; color = c_raw, markersize = MS, marker = :circle)
s3 = scatter!(ax1, x .+ 0.12, cov_cal; color = c_cal, markersize = MS + 2, marker = :diamond)
s4 = scatter!(ax1, x .+ 0.24, cov_org; color = c_cal, markersize = MS + 2,
              marker = :diamond, strokecolor = c_cal, strokewidth = 2.5,
              alpha = 0.0)
s4 = scatter!(ax1, x .+ 0.24, cov_org; color = :white, markersize = MS + 2,
              marker = :diamond, strokecolor = c_cal, strokewidth = 2.5)
leg1 = Legend(fig[2, 1],
    Any[s1, s2, s3, s4, LineElement(color = :black, linestyle = :dash, linewidth = 2.5)],
    ["fixed effects, convention",
     "organism-by-type, convention",
     "+ estimated scale, per record",
     "+ estimated scale, per organism",
     "nominal 0.95"];
    orientation = :horizontal, nbanks = 3, framevisible = true,
    labelsize = FS_LEGEND, patchsize = (26, 18), padding = (6, 6, 4, 4),
    tellheight = true, tellwidth = false)

ax2 = Axis(fig[3, 1];
    ylabel = "SD factor on convention",
    xticks = (1:4, [lab for (_, lab) in PATHWAYS]),
    title = "Estimated scale tracks dispersion")
ylims!(ax2, 0.8, 4.2)
hlines!(ax2, [1.0]; color = :black, linestyle = :dash, linewidth = 2.5)
for (i, xi) in enumerate(x)
    lines!(ax2, [xi, xi], [chat_sd[i], disp[i]]; color = RGBAf(0, 0, 0, 0.25),
           linewidth = 2)
end
p1 = scatter!(ax2, x .- 0.08, disp; color = RGBf(0.12, 0.47, 0.31),
              markersize = MS, marker = :circle)
p2 = scatter!(ax2, x .+ 0.08, chat_sd; color = c_cal, markersize = MS + 2,
              marker = :diamond)
leg2 = Legend(fig[4, 1],
    Any[p1, p2, LineElement(color = :black, linestyle = :dash, linewidth = 2.5)],
    ["marginal residual dispersion",
     "estimated record-error scale",
     "half-decade convention"];
    orientation = :horizontal, nbanks = 2, framevisible = true,
    labelsize = FS_LEGEND, patchsize = (26, 18), padding = (6, 6, 4, 4),
    tellheight = true, tellwidth = false)

Label(fig[1, 1, TopLeft()], "A"; fontsize = FS_TITLE, font = :bold,
      padding = (0, 26, 14, 0), halign = :right)
Label(fig[3, 1, TopLeft()], "B"; fontsize = FS_TITLE, font = :bold,
      padding = (0, 26, 14, 0), halign = :right)
rowgap!(fig.layout, 1, 8)
rowgap!(fig.layout, 2, 34)
rowgap!(fig.layout, 3, 8)

save(joinpath(OUTDIR, "Fig7_recalibration.pdf"), fig)
save(joinpath(UPDIR, "Fig5.pdf"), fig)
save(joinpath(UPDIR, "Fig5.png"), fig; px_per_unit = 300 / 96)
println("wrote Fig7_recalibration.pdf and figures_for_upload/Fig5.{pdf,png}")
