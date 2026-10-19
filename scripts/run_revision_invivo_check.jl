# run_revision_invivo_check.jl
#
# External in vivo check of the E. coli hold-out predictions, added for the
# revision.
#
# Purpose.  run_revision_ecoli_holdout.jl refits two models with every E. coli
# record removed and draws joint predictions for every forward turnover number.
# Davidi et al. (2016, PNAS 113:3401) estimated in vivo maximal apparent
# turnover numbers (kmax) for E. coli from pFBA fluxes and proteomics, which
# involves no BRENDA record at all.  This driver asks how those kmax values sit
# in the four predictive distributions of the hold-out:
#
#   fe_org  fixed effects, organism level (the consensus a'q)
#   fe_rec  fixed effects, record level (consensus plus the half-decade record error)
#   pd_org  pre-specified default, organism level (a'q plus a new E. coli offset)
#   pd_rec  pre-specified default, record level (plus the estimated record error)
#
# kmax is a lower bound on kcat and typically about two-fold below it, so a
# kmax above the upper 97.5% limit is the informative miss; a kmax below the
# lower limit is expected more often than 2.5% of the time.  A sensitivity
# version shifts log kmax up by log(1.9) as a rough kcat proxy: the median
# kcat/kmax ratio is 1.82 over the 132 pairs of Davidi's Dataset S1 and 1.93
# over the 14 matched rows here with both values.
#
# Subsets of the matched forward rows (data/external/davidi2016_kmax_matched.csv):
#   direct_1s        kmax given directly in 1/s (per chain), on rows that carry
#                    BRENDA kcat records (15 rows)
#   all_matched      every matched row that carries BRENDA kcat records,
#                    including kmax converted from specific activity (29 rows)
#   all_matched_incl_norecord  all_matched plus the four matched rows without any
#                    record, predicted from the full Q alone (TCA r_0280, r_0831,
#                    r_0505; pentose phosphate r_0091), supplementary (33 rows)
#
# Per model, subset and pathway (and pooled over pathways) the summary gives n;
# the number and fraction of kmax values inside the 95% interval with an exact
# (Clopper-Pearson) binomial 95% CI; the numbers above the upper and below the
# lower limit; the median interval width in decades; the mean log predictive
# density of log kmax (nats per ln unit) from a Gaussian fitted to the draws and
# from a Gaussian kernel density (Silverman bandwidth); the mean CRPS (ln units)
# from the draws; and the mean and median PIT of log kmax.  The Gaussian fit is
# the primary log score: the kernel bandwidth is a small fraction of the draw
# spread, so the kernel density has much thinner tails and lpd_kde falls far
# below lpd_gauss for a target many standard deviations out (the narrow
# fixed-effects consensus in particular).
#
# Outputs, written under output/invivo/:
#   invivo_check_rows.csv     one row per matched kcat row, model and shift
#   invivo_check_summary.csv  one row per shift, subset, model and pathway
#
# Environment controls:
#   PB_INVIVO_KMAX_TABLE  kmax table (default data/external/davidi2016_kmax_matched.csv)
#   PB_INVIVO_RATIO       kcat/kmax ratio of the sensitivity shift (default 1.9)
#   PB_REV_SUFFIX         suffix of the hold-out draw files to read (default none;
#                         _k12 for the hold-out run that also removes E. coli K-12)
#   PB_INVIVO_OUT_SUFFIX  suffix for the output names (default PB_REV_SUFFIX)
#
# Run (seconds; after run_revision_ecoli_holdout.jl):
#   julia --project=. scripts/run_revision_invivo_check.jl

import Pkg
Pkg.activate(dirname(@__DIR__))

using Dates, Printf, Statistics
using CSV, DataFrames, Distributions, JLD2

const PATHWAYS = [:glycolysis, :tca, :pentose_phosphate, :shikimate]
const KMAX_TABLE = get(ENV, "PB_INVIVO_KMAX_TABLE",
                       joinpath(dirname(@__DIR__), "data", "external", "davidi2016_kmax_matched.csv"))
const RATIO = parse(Float64, get(ENV, "PB_INVIVO_RATIO", "1.9"))
const HOLDOUT_SUFFIX = get(ENV, "PB_REV_SUFFIX", "")
const OUT_SUFFIX = get(ENV, "PB_INVIVO_OUT_SUFFIX", HOLDOUT_SUFFIX)
const OUTDIR = joinpath(dirname(@__DIR__), "output", "invivo")
const MODELS = [("fe_org", "fixed effects, organism level (consensus)"),
                ("fe_rec", "fixed effects, record level"),
                ("pd_org", "pre-specified default, organism level"),
                ("pd_rec", "pre-specified default, record level")]
const SHIFTS = [("kmax", 0.0), ("kmax_x$(RATIO)", log(RATIO))]
const SUBSETS = ["direct_1s", "all_matched", "all_matched_incl_norecord"]

# crps_from_samples of prediction_utils.jl, copied so that the driver needs no
# sampler code.
function crps_from_samples(x::Vector{Float64}, y::Float64)
    m = length(x)
    xs = sort(x)
    term1 = mean(abs.(xs .- y))
    s = 0.0
    @inbounds for i in 1:m
        s += (2i - m - 1) * xs[i]
    end
    term2 = (2.0 / (m * m)) * s
    return term1 - 0.5 * term2
end

"Log density at z of a Gaussian kernel density of x (Silverman's bandwidth)."
function kde_logpdf(x::Vector{Float64}, z::Float64)
    n = length(x)
    s = std(x)
    iqr = quantile(x, 0.75) - quantile(x, 0.25)
    h = 0.9 * (iqr > 0 ? min(s, iqr / 1.34) : s) * n^(-1 / 5)
    v = [-0.5 * ((z - xi) / h)^2 for xi in x]
    m = maximum(v)
    return m + log(sum(exp.(v .- m))) - log(n) - log(h) - 0.5 * log(2pi)
end

"Mid-rank PIT of z under the draws x."
pit_midrank(x, z) = (count(<(z), x) + 0.5 * count(==(z), x)) / length(x)

"Exact (Clopper-Pearson) binomial confidence interval."
function clopper_pearson(k::Int, n::Int; level::Float64 = 0.95)
    n == 0 && return (NaN, NaN)
    a = (1 - level) / 2
    lo = k == 0 ? 0.0 : quantile(Beta(k, n - k + 1), a)
    hi = k == n ? 1.0 : quantile(Beta(k + 1, n - k), 1 - a)
    return (lo, hi)
end

println("Revision in vivo check (Davidi et al. 2016 kmax against the E. coli hold-out)")
println("  kmax table   = $(KMAX_TABLE)")
println("  hold-out     = ecoli_holdout_draws_<pathway>$(HOLDOUT_SUFFIX).jld2")
println("  shift        = log($(RATIO)) = $(round(log(RATIO), digits = 4))")
println("  started      = $(now())")
flush(stdout)

kt = CSV.read(KMAX_TABLE, DataFrame; stringtype = String)
kt.kmax_1s = [ismissing(v) ? missing : Float64(v) for v in kt.kmax_1s]
kt.has_kcat_records = [v isa Bool ? v : lowercase(string(v)) == "true" for v in kt.has_kcat_records]
km = kt[.!ismissing.(kt.kmax_1s) .& (kt.direction .== "forward"), :]
@printf("  matched forward rows: %d (with records %d, direct 1/s with records %d)\n",
        nrow(km), count(km.has_kcat_records),
        count(km.has_kcat_records .& (km.kmax_source .== "direct_1s_per_chain")))

rows = NamedTuple[]
for pw in PATHWAYS
    f = joinpath(dirname(@__DIR__), "output", string(pw), "results", "revision",
                 "ecoli_holdout_draws_$(pw)$(HOLDOUT_SUFFIX).jld2")
    D = load(f)
    qcol = Dict(q => j for (j, q) in enumerate(D["Q_rows"]))
    sub = km[km.pathway .== string(pw), :]
    for r in eachrow(sub)
        j = get(qcol, r.Q_row, 0)
        j == 0 && error("$(pw) Q row $(r.Q_row) has no hold-out draws")
        D["reaction_ids"][j] == r.reaction_id && D["directions"][j] == "forward" ||
            error("$(pw) Q row $(r.Q_row): draw file says $(D["reaction_ids"][j]) $(D["directions"][j])")
        direct = r.kmax_source == "direct_1s_per_chain"
        for (sname, shift) in SHIFTS, (mname, _) in MODELS
            x = Vector{Float64}(D[mname][:, j])
            z = log(r.kmax_1s) + shift
            lo, med, hi = quantile(x, [0.025, 0.5, 0.975])
            mu, sd = mean(x), std(x)
            push!(rows, (shift = sname, pathway = string(pw), Q_row = r.Q_row,
                reaction_id = r.reaction_id, reaction_name = r.reaction_name,
                davidi_reaction_id = r.davidi_reaction_id, match_quality = r.match_quality,
                kmax_1s = r.kmax_1s, kmax_source = r.kmax_source,
                has_kcat_records = r.has_kcat_records,
                in_direct_1s = direct && r.has_kcat_records,
                in_all_matched = r.has_kcat_records,
                in_all_matched_incl_norecord = true,
                model = mname, log_target = z,
                pred_lo = lo, pred_median = med, pred_hi = hi,
                pred_mean = mu, pred_sd = sd,
                inside95 = lo <= z <= hi, above95 = z > hi, below95 = z < lo,
                width95_decades = (hi - lo) / log(10),
                log10_target_over_median = (z - med) / log(10),
                lpd_gauss = logpdf(Normal(mu, sd), z),
                lpd_kde = kde_logpdf(x, z),
                crps = crps_from_samples(x, z),
                pit = pit_midrank(x, z)))
        end
    end
end
rdf = DataFrame(rows)

srows = NamedTuple[]
for (sname, _) in SHIFTS, subset in SUBSETS, (mname, mlabel) in MODELS,
    pw in vcat(string.(PATHWAYS), "pooled")
    m = (rdf.shift .== sname) .& (rdf.model .== mname) .& rdf[!, Symbol("in_", subset)]
    pw == "pooled" || (m .&= rdf.pathway .== pw)
    d = rdf[m, :]
    n = nrow(d)
    n == 0 && continue
    k = count(d.inside95)
    ci = clopper_pearson(k, n)
    push!(srows, (shift = sname, subset = subset, model = mname, model_label = mlabel,
        pathway = pw, n = n, n_inside95 = k, frac_inside95 = k / n,
        frac_inside95_ci_lo = ci[1], frac_inside95_ci_hi = ci[2],
        n_above95 = count(d.above95), n_below95 = count(d.below95),
        median_width95_decades = median(d.width95_decades),
        median_log10_target_over_median = median(d.log10_target_over_median),
        mean_lpd_gauss = mean(d.lpd_gauss), median_lpd_gauss = median(d.lpd_gauss),
        mean_lpd_kde = mean(d.lpd_kde),
        mean_crps = mean(d.crps), mean_pit = mean(d.pit), median_pit = median(d.pit),
        above95_rows = join(d.reaction_id[d.above95], ";")))
end
sdf = DataFrame(srows)

mkpath(OUTDIR)
CSV.write(joinpath(OUTDIR, "invivo_check_rows$(OUT_SUFFIX).csv"), rdf)
CSV.write(joinpath(OUTDIR, "invivo_check_summary$(OUT_SUFFIX).csv"), sdf)
@printf("  wrote %d row-level and %d summary rows to %s\n", nrow(rdf), nrow(sdf), OUTDIR)

for (sname, _) in SHIFTS, subset in SUBSETS
    t = sdf[(sdf.shift .== sname) .& (sdf.subset .== subset), :]
    println("\n== target $(sname), subset $(subset)")
    show(stdout, select(t, :model, :pathway, :n, :n_inside95, :frac_inside95,
                        :frac_inside95_ci_lo, :frac_inside95_ci_hi, :n_above95, :n_below95,
                        :median_width95_decades, :median_log10_target_over_median,
                        :mean_lpd_gauss, :mean_lpd_kde, :mean_crps, :mean_pit, :median_pit);
         allrows = true, allcols = true)
    println()
end
println("\nRows above the upper 95% limit (target kmax, all_matched_incl_norecord)")
show(stdout, select(rdf[(rdf.shift .== "kmax") .& rdf.above95, :], :model, :pathway,
                    :reaction_id, :davidi_reaction_id, :kmax_1s, :kmax_source,
                    :has_kcat_records, :log_target, :pred_lo, :pred_hi);
     allrows = true, allcols = true)
println()
println("Done at $(now())")
