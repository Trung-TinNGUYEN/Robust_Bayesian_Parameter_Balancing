# run_revision_inference.jl
#
# Uncertainty quantification for the headline real-data summaries, added for the
# revision.
#
# The leave-one-source-out summaries reported in the manuscript are averages over
# individual records, but records are clustered within source organisms and are
# not independent: a single organism can contribute hundreds of records, and the
# organism effect is exactly the structure the paper is about.  Reporting a naive
# binomial standard error would therefore understate the uncertainty
# substantially.  Two cluster-aware procedures are used instead.
#
#   1. A nonparametric bootstrap that resamples whole ORGANISMS with replacement
#      and recomputes the record-averaged statistic, giving a percentile interval
#      for coverage, CRPS and predictive RMSE.
#
#   2. A paired Wilcoxon signed-rank test across organisms comparing two model
#      cells on the same held-out organisms, with the Hodges--Lehmann estimate of
#      the median paired difference.  The inferential unit is the organism, so
#      the test respects the clustering.
#
# Inputs are the per-record vectors written by run_revision_realdata.jl to
#   output/<pathway>/results/revision/realdata_percell_<pathway>.jld2
#
# Outputs, written alongside them:
#   inference_bootstrap_<pathway>.csv   bootstrap intervals per model cell
#   inference_paired_<pathway>.csv      paired organism-level comparisons
#
# Environment controls:
#   PB_REV_PATHWAYS  comma-separated subset of {glycolysis,tca} (default both)
#   PB_REV_NBOOT     bootstrap replicates (default 10000)

import Pkg
Pkg.activate(dirname(@__DIR__))

using CSV
using DataFrames
using JLD2
using Printf
using Random
using Statistics
using HypothesisTests

const REV_PATHWAYS = Symbol.(strip.(split(get(ENV, "PB_REV_PATHWAYS", "glycolysis,tca"), ",")))
const NBOOT = parse(Int, get(ENV, "PB_REV_NBOOT", "10000"))
const SEED = 20260805

"""
    cluster_bootstrap(values, groups, nboot, rng; transform) -> (estimate, lo, hi, se)

Percentile bootstrap for `transform(mean(values))`, resampling whole groups with
replacement.  Records within a group are kept together, so the interval reflects
between-group as well as within-group variability.  The `transform` argument
allows nonlinear summaries such as the root mean squared error, for which
`values` are the squared errors and `transform` is `sqrt`; applying the transform
to each bootstrap replicate rather than to the average of replicates is what
makes the resulting percentile interval correct.
"""
function cluster_bootstrap(values::Vector{Float64}, groups::Vector{String},
                           nboot::Int, rng::AbstractRNG;
                           transform::Function = identity)
    levels = unique(groups)
    idx_by_level = Dict(g => findall(==(g), groups) for g in levels)
    G = length(levels)
    stats = Vector{Float64}(undef, nboot)
    buf = Float64[]
    for b in 1:nboot
        empty!(buf)
        for _ in 1:G
            append!(buf, @view values[idx_by_level[levels[rand(rng, 1:G)]]])
        end
        stats[b] = transform(mean(buf))
    end
    return (transform(mean(values)), quantile(stats, 0.025), quantile(stats, 0.975),
            std(stats))
end

"""
    organism_means(values, groups) -> (levels, means)

Aggregate a per-record statistic to one value per organism.  All paired tests
operate on these aggregates, so that each organism contributes once.
"""
function organism_means(values::Vector{Float64}, groups::Vector{String})
    levels = sort(unique(groups))
    return levels, [mean(values[groups .== g]) for g in levels]
end

"""
    hodges_lehmann(d) -> Float64

Median of the Walsh averages of `d`, the location estimator associated with the
Wilcoxon signed-rank test.
"""
function hodges_lehmann(d::Vector{Float64})
    n = length(d)
    n == 0 && return NaN
    walsh = Float64[]
    sizehint!(walsh, n * (n + 1) ÷ 2)
    @inbounds for i in 1:n, j in i:n
        push!(walsh, (d[i] + d[j]) / 2)
    end
    return median(walsh)
end

function run_pathway(pathway::Symbol)
    dir = joinpath(dirname(@__DIR__), "output", string(pathway), "results", "revision")
    src = joinpath(dir, "realdata_percell_$(pathway).jld2")
    isfile(src) || (@warn "missing $src; skipping"; return nothing)
    stored = load(src)
    percell = stored["percell"]
    cells = sort(collect(keys(percell)))
    println("\n=== $(pathway): $(length(cells)) model cells, $(NBOOT) bootstrap replicates")

    boot = DataFrame(pathway = String[], cell = String[], statistic = String[],
                     estimate = Float64[], ci_lo = Float64[], ci_hi = Float64[],
                     boot_se = Float64[], n_records = Int[], n_organisms = Int[])

    for cell in cells
        d = percell[cell]
        org = Vector{String}(d.org)
        rng = MersenneTwister(SEED)
        for (name, vals, tf) in [("coverage95", Float64.(d.covered95), identity),
                                 ("CRPS", Vector{Float64}(d.crps), identity),
                                 ("pred_RMSE", Vector{Float64}(d.pred_error) .^ 2, sqrt)]
            est, lo, hi, se = cluster_bootstrap(vals, org, NBOOT, rng; transform = tf)
            push!(boot, (string(pathway), cell, name, est, lo, hi, se,
                         length(vals), length(unique(org))))
            @printf("  %-22s %-11s %.4f  [%.4f, %.4f]  (cluster SE %.4f)\n",
                    cell, name, est, lo, hi, se)
        end
    end

    paired = DataFrame(pathway = String[], statistic = String[], cell_a = String[],
                       cell_b = String[], n_organisms = Int[],
                       median_diff_HL = Float64[], p_value = Float64[],
                       mean_a = Float64[], mean_b = Float64[])

    # Every ordered pair is compared, so the table supports whichever contrast a
    # reader considers primary.
    println("\n  Paired Wilcoxon signed-rank tests across organisms")
    for (name, get) in [("coverage95", d -> Float64.(d.covered95)),
                        ("CRPS", d -> Vector{Float64}(d.crps))]
        for i in 1:length(cells), j in (i + 1):length(cells)
            a, b = cells[i], cells[j]
            la, ma = organism_means(get(percell[a]), Vector{String}(percell[a].org))
            lb, mb = organism_means(get(percell[b]), Vector{String}(percell[b].org))
            la == lb || error("organism sets differ between $a and $b")
            diff = mb .- ma
            nz = diff[diff .!= 0]
            p = isempty(nz) ? 1.0 : pvalue(SignedRankTest(nz))
            push!(paired, (string(pathway), name, a, b, length(diff),
                           hodges_lehmann(diff), p, mean(ma), mean(mb)))
            @printf("    %-10s %-22s vs %-22s  HL=%+.4f  p=%.3g\n",
                    name, a, b, hodges_lehmann(diff), p)
        end
    end

    CSV.write(joinpath(dir, "inference_bootstrap_$(pathway).csv"), boot)
    CSV.write(joinpath(dir, "inference_paired_$(pathway).csv"), paired)
    println("  wrote inference_bootstrap_$(pathway).csv and inference_paired_$(pathway).csv")
    return nothing
end

for pw in REV_PATHWAYS
    run_pathway(pw)
end
println("\ndone")
