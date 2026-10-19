# run_revision_hetero_inference.jl
#
# Cluster-aware uncertainty for the error-scale constructions, added for the
# revision.
#
# The error-scale comparison of run_revision_hetero.jl reports its coverages as
# point estimates.  Records are clustered within organisms, and every
# construction is evaluated on the same records under common random numbers, so
# the honest uncertainty for a contrast between two constructions is the
# ORGANISM-LEVEL uncertainty of a paired difference.  This driver applies the
# same two procedures used for the headline summaries in
# run_revision_inference.jl:
#
#   1. a nonparametric bootstrap resampling whole organisms with replacement,
#      giving percentile intervals for each construction's coverage and, run on
#      the per-organism PAIRED differences, for each contrast;
#   2. a paired Wilcoxon signed-rank test across organisms with the
#      Hodges--Lehmann estimate of the median paired difference.
#
# Contrasts reported, chosen to match the claims in the manuscript:
#   global      - global_cell   the fit-free scale against the iterated one;
#   tau_eb      - global        the per-type between-organism variance;
#   type        - global        the per-type record scale (the harmful split);
#   global_cell - convention    recalibration against the half-decade convention;
#   scale_mix   - global        the record scale as a random effect (far tail);
#   scale_mix13, scale_mix20    the same at shape floors 1.3 and 2.0;
#   scale_lnmix - global        a log-normal scale population, which needs no floor;
#   tau_post    - global        the between-organism variance integrated over
#                               its posterior instead of plugged in.
# Each contrast is also evaluated on the turnover-number records alone, where
# the per-type variance claim lives, and at the nominal 95% and 99% levels.
#
# Multiplicity: every (contrast, level, subset) is tested on four pathways,
# and that quadruple is the family for a Holm adjustment (p_holm).  The number
# of organisms whose coverage actually differs between the two constructions
# (n_nonzero) is reported with every test, because the signed-rank statistic
# acts on those organisms only.
#
# Inputs:  output/<pathway>/results/revision/hetero_percell_<pathway>.jld2
# Outputs: output/<pathway>/results/revision/hetero_inference_<pathway>.csv
#
# Environment controls:
#   PB_REV_PATHWAYS  subset of {glycolysis,tca,pentose_phosphate,shikimate}
#   PB_REV_NBOOT     bootstrap replicates (default 10000)

import Pkg
Pkg.activate(dirname(@__DIR__))

using CSV, DataFrames, JLD2, Printf, Random, Statistics
using HypothesisTests

const PATHWAYS = Symbol.(strip.(split(get(ENV, "PB_REV_PATHWAYS",
    "glycolysis,tca,pentose_phosphate,shikimate"), ",")))
const NBOOT = parse(Int, get(ENV, "PB_REV_NBOOT", "10000"))
const SEED = 20260901
const CONTRASTS = [("global", "global_cell"), ("tau_eb", "global"),
                   ("type", "global"), ("global_cell", "convention"),
                   ("scale_mix", "global"), ("scale_mix13", "global"),
                   ("scale_mix20", "global"), ("scale_lnmix", "global"),
                   ("scale_mixml", "global"), ("scale_mix2", "global"),
                   ("scale_mix2m", "global"), ("tau_post", "global"),
                   ("joint2m", "global"), ("joint2m", "scale_mix2m"), ("joint2m", "tau_eb"),
                   ("tau_rel", "global"), ("joint2m_rel", "global"), ("joint2m_rel", "scale_mix2m")]
const SINGLES = ["convention", "global_cell", "global", "tau_eb", "scale_mix",
                 "scale_lnmix", "scale_mixml", "scale_mix2", "scale_mix2m", "joint2m", "tau_post",
                 "tau_rel", "joint2m_rel"]

function cluster_bootstrap(values::Vector{Float64}, groups::Vector{String},
                           nboot::Int, rng::AbstractRNG)
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
        stats[b] = mean(buf)
    end
    (mean(values), quantile(stats, 0.025), quantile(stats, 0.975), std(stats))
end

organism_means(values, groups) = (lv = sort(unique(groups));
    (lv, [mean(values[groups .== g]) for g in lv]))

function hodges_lehmann(d::Vector{Float64})
    n = length(d); n == 0 && return NaN
    walsh = Float64[]
    sizehint!(walsh, n * (n + 1) ÷ 2)
    @inbounds for i in 1:n, j in i:n
        push!(walsh, (d[i] + d[j]) / 2)
    end
    median(walsh)
end

"Bootstrap over organisms of the mean per-organism difference."
function paired_bootstrap(d::Vector{Float64}, nboot::Int, rng::AbstractRNG)
    G = length(d)
    stats = [mean(d[rand(rng, 1:G, G)]) for _ in 1:nboot]
    (mean(d), quantile(stats, 0.025), quantile(stats, 0.975), std(stats))
end

function run_pathway(pathway::Symbol)
    dir = joinpath(dirname(@__DIR__), "output", string(pathway), "results", "revision")
    dat = load(joinpath(dir, "hetero_percell_$(pathway).jld2"))
    org = Vector{String}(dat["org"]); kind = Vector{String}(dat["kind"])
    rng = MersenneTwister(SEED)
    rows = NamedTuple[]

    for (level, cov) in [("95", Dict(m => Float64.(v) for (m, v) in dat["cov95"])),
                         ("99", Dict(m => Float64.(v) for (m, v) in dat["cov99"]))],
        subset in ["all", "kcat"]
        keep = subset == "all" ? trues(length(org)) : (kind .== "kcat")
        o = org[keep]
        # absolute coverage with organism-cluster bootstrap
        for m in SINGLES
            est, lo, hi, se = cluster_bootstrap(cov[m][keep], o, NBOOT, rng)
            _, om = organism_means(cov[m][keep], o)
            push!(rows, (pathway = string(pathway), level = level, subset = subset,
                quantity = "coverage", method = m, reference = "",
                estimate = est, ci_lo = lo, ci_hi = hi, boot_se = se,
                org_estimate = mean(om), org_ci_lo = NaN, org_ci_hi = NaN,
                hl_diff = NaN, p_wilcoxon = NaN, p_holm = NaN, n_nonzero = 0,
                n_organisms = length(unique(o))))
        end
        # paired contrasts on the same records, aggregated per organism
        for (m, ref) in CONTRASTS
            (haskey(cov, m) && haskey(cov, ref)) || continue
            lv, dm = organism_means(cov[m][keep] .- cov[ref][keep], o)
            est, lo, hi, se = paired_bootstrap(dm, NBOOT, rng)
            nz = filter(!iszero, dm)
            p = isempty(nz) ? 1.0 : pvalue(SignedRankTest(nz))
            # record-weighted paired difference with cluster bootstrap
            estr, lor, hir, ser = cluster_bootstrap(cov[m][keep] .- cov[ref][keep],
                                                    o, NBOOT, rng)
            push!(rows, (pathway = string(pathway), level = level, subset = subset,
                quantity = "delta_coverage", method = m, reference = ref,
                estimate = estr, ci_lo = lor, ci_hi = hir, boot_se = ser,
                org_estimate = est, org_ci_lo = lo, org_ci_hi = hi,
                hl_diff = hodges_lehmann(dm), p_wilcoxon = p, p_holm = NaN,
                n_nonzero = length(nz),
                n_organisms = length(lv)))
            @printf("%-18s L%s %-5s %-11s - %-11s  drec=%+.4f [%+.4f,%+.4f]  dorg=%+.4f  HL=%+.4f  p=%.2g\n",
                    string(pathway), level, subset, m, ref, estr, lor, hir, est,
                    hodges_lehmann(dm), p)
        end
    end
    DataFrame(rows)
end

"Holm step-down adjustment of a vector of p-values."
function holm(ps::Vector{Float64})
    m = length(ps); ord = sortperm(ps); adj = fill(NaN, m); running = 0.0
    for (k, i) in enumerate(ord)
        running = max(running, (m - k + 1) * ps[i])
        adj[i] = min(running, 1.0)
    end
    adj
end

println("Cluster-aware inference for the error-scale constructions")
println("  bootstrap replicates = $NBOOT")
dfs = Dict{Symbol,DataFrame}()
for p in PATHWAYS
    dfs[p] = run_pathway(p)
end
# Holm within each (contrast, level, subset) family across the pathways run
all = vcat(values(dfs)...)
isc = all.quantity .== "delta_coverage"
fam = [(all.method[i], all.reference[i], all.level[i], all.subset[i]) for i in 1:nrow(all)]
for f in unique(fam[isc])
    ii = findall(i -> isc[i] && fam[i] == f, 1:nrow(all))
    all.p_holm[ii] = holm(Float64.(all.p_wilcoxon[ii]))
end
for p in PATHWAYS
    dir = joinpath(dirname(@__DIR__), "output", string(p), "results", "revision")
    CSV.write(joinpath(dir, "hetero_inference_$(p).csv"), all[all.pathway .== string(p), :])
end
println("Done.")
