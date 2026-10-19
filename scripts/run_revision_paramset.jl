# run_revision_paramset.jl
#
# The released balanced parameter set, added for the revision.
#
# Limitation addressed: the submission reported calibration diagnostics but
# never shipped the artifact the method exists to produce.  The paper's
# recommendation is that a balanced parameter set be released with uncertainty
# that has been checked rather than assumed; this driver produces exactly that
# for the four pathways, in the units a modeller uses.
#
# The model is the one the paper recommends, fitted to ALL records of a
# pathway: organism-by-quantity-type hierarchy, record error scale from the
# fit-free replication estimator, and a between-organism variance stratified by
# quantity type (Paule-Mandel marginal estimator on the fitted cell residuals).
#
# Two intervals are reported for every quantity, because they answer different
# questions and the paper's central point is that the difference matters:
#
#   consensus      the parameter common to all sources, which is what a
#                  modeller wants when pooling literature values;
#   new organism   the predictive interval for an organism not among the
#                  curated sources, which is the usual modelling situation and
#                  is wider by the between-organism standard deviation of that
#                  quantity type.
#
# A third interval answers the question a modeller of a CURATED organism asks:
#
#   organism-conditional   the posterior for a named organism that does have
#                          records, conditioning on that organism's fitted
#                          offset for the quantity type (its organism-by-type
#                          cell).  Reported for every organism with at least
#                          MIN_ORG_REC records in the pathway, and always for
#                          S. cerevisiae, E. coli and H. sapiens when present,
#                          because the pathways are yeast pathways populated
#                          with cross-organism records and the consensus is
#                          not a yeast value.
#
# Output: output/paramset/balanced_parameters.csv  and one CSV per pathway;
#         output/paramset/balanced_parameters_by_organism.csv for the
#         organism-conditional posteriors.
#
# Environment controls added for the EM-iteration sensitivity run (S1 Text):
#   PB_REV_OUTER_MULT  multiplies the number of outer EM iterations of the fit
#                      (default 1, which is the released 3, or 4 on the TCA cycle)
#   PB_REV_SUFFIX      appended to the output directory name (default empty, which
#                      writes the released set). A run with PB_REV_OUTER_MULT
#                      other than 1 must set it, so that it cannot overwrite the
#                      released set

import Pkg
Pkg.activate(dirname(@__DIR__))

using Dates, LinearAlgebra, Printf, Random, Statistics
using ArviZ, BlockArrays, CairoMakie, Colors, CSV, DataFrames, Distributions
using JLD2, JSON, KernelDensity, LaTeXStrings, MCMCChains, ProgressMeter, SpecialFunctions

include(joinpath(dirname(@__DIR__), "src", "metabolic_reconstruction.jl"))
include(joinpath(dirname(@__DIR__), "src", "parameter_balancing.jl"))
include(joinpath(dirname(@__DIR__), "src", "makie_functions.jl"))
include(joinpath(dirname(@__DIR__), "src", "parameter_balancing_core.jl"))
include(joinpath(dirname(@__DIR__), "src", "prediction_utils.jl"))
include(joinpath(dirname(@__DIR__), "src", "hierarchy_utils.jl"))
include(joinpath(dirname(@__DIR__), "src", "experiment_runners.jl"))
include(joinpath(dirname(@__DIR__), "src", "mutant_flags.jl"))

const SEED = 20260826
const PATHWAYS = [:glycolysis, :tca, :pentose_phosphate, :shikimate]
const C_MIN, C_MAX = 0.25, 40.0
const DF_MIN = 4
const NDRAW = 4000
const MIN_ORG_REC = 20
const NAMED_ORGS = ["saccharomyces cerevisiae", "escherichia coli", "homo sapiens"]

# ---- estimators, identical to run_revision_hetero.jl ----------------------
function replication_stats(y, sigma2, repkey, strat, idx)
    bygroup = Dict{String,Vector{Int}}()
    for i in idx
        push!(get!(bygroup, repkey[i], Int[]), i)
    end
    out = Dict{String,Tuple{Float64,Float64,Int}}()
    for (_, ig) in bygroup
        ng = length(ig); ng >= 2 || continue
        ybar = mean(y[i] for i in ig)
        for i in ig
            s = strat[i]
            S, D, df = get(out, s, (0.0, 0.0, 0))
            out[s] = (S + (y[i] - ybar)^2, D + sigma2[i] * (1.0 - 1.0 / ng), df + 1)
        end
    end
    Dict(s => (v[1], v[2], Float64(v[3])) for (s, v) in out)
end
raw_scale(S, D) = D > 0 ? clamp(S / D, C_MIN, C_MAX) : NaN

# ---- units ---------------------------------------------------------------
"Natural units and label for a quantity type, given a log-scale value."
function to_natural(kind, v)
    kind == "kcat" ? (exp(v), "1/s") : (exp(v) * 1000, "mM")
end

function run_pathway(pathway::Symbol)
    root = joinpath(dirname(@__DIR__), "output", string(pathway))
    brenda = load(joinpath(root, "data", "brenda_inputs_$(pathway).jld2"))
    post = load(joinpath(root, "data", "posterior_$(pathway).jld2"))
    y = Vector{Float64}(brenda["y"]); sigma2 = Vector{Float64}(brenda["σ2"])
    Arows = Matrix{Float64}(brenda["Arows"]); org = Vector{String}(brenda["grp"])
    meta = brenda["meta"]
    kinds = Vector{String}(getproperty(meta, :kind))
    ecs = string.(getproperty(meta, :ec_code))
    subs = string.(getproperty(meta, :substrate))
    qrows = Vector{Int}(getproperty(meta, :Q_row))
    q0 = vec(Array(post["q_prior"])); C0 = Matrix(Array(post["C_prior"]))
    eb = EBSpec(q0 = q0, C0 = C0, κ0 = parse(Float64, get(ENV, "PB_EB_KAPPA0", "1e-9")))
    d = size(Arows, 2)
    A0, b0 = _prepare_information_pair(Arows, d, Matrix{Float64}(brenda["A0"]),
                                       Vector{Float64}(brenda["b0"]);
                                       deduplicate_thermo_block = true)
    org_type = [string(org[i], "|", kinds[i]) for i in eachindex(org)]
    repkey = [string(org[i], "|", qrows[i]) for i in eachindex(org)]
    # Enzyme form of every record, so that each released row says how many of the
    # records behind it are site-directed mutants (S1 Text, enzyme-form section).
    mm = mutant_mask(dirname(@__DIR__), CSVNAME_OF[pathway], y, org, kinds, ecs, subs)
    if haskey(ENV, "PB_REV_EXCLUDE_MUTANTS")
        # Wild-type variant of the released set: drop the records the flag marks as
        # site-directed mutants before the fit, as run_revision_hetero.jl does for the
        # enzyme-form sensitivity run.  Written to output/paramset_wildtype.
        keep = findall(.!mm.mutant)
        @printf("  enzyme-form filter: %d of %d records are mutant-flagged, keeping %d\n",
                count(mm.mutant), length(y), length(keep))
        y = y[keep]; sigma2 = sigma2[keep]; Arows = Arows[keep, :]; org = org[keep]
        kinds = kinds[keep]; ecs = ecs[keep]; subs = subs[keep]; qrows = qrows[keep]
        org_type = org_type[keep]; repkey = repkey[keep]
        mm = (mutant = mm.mutant[keep], wildtype = mm.wildtype[keep],
              comments = mm.comments[keep])
    end
    idx = collect(eachindex(y))

    # fit-free record scale, pooled (the recommended default)
    gstat = replication_stats(y, sigma2, repkey, fill("all", length(y)), idx)
    cg = haskey(gstat, "all") ? raw_scale(gstat["all"][1], gstat["all"][2]) : 1.0
    isfinite(cg) || (cg = 1.0)
    ctr = fill(cg, length(y))

    slow = pathway in (:tca,)
    spec = FoldFitSpec(model = :gaussian, hierarchical = true,
        n_outer = (slow ? 4 : 3) * parse(Int, get(ENV, "PB_REV_OUTER_MULT", "1")),
        n_mcmc = slow ? 400 : 300,
        burnin = slow ? 120 : 80, thin = 1, n_keep_min = 200,
        posterior_refresh = 1, adaptive_stop = false, rtol_outer = 1e-3,
        min_outer = 2, tau_update = :ig, tau_pc_rate = 0.0,
        ν_init = 50.0, α_init = 0.0, update_nu = false, update_alpha = false,
        refine_shape = false, skew_location = :mean)
    rng = MersenneTwister(SEED)
    hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
    fit = fit_fold_mcem(rng, A0, b0, Arows, y, ctr .* sigma2, org_type, eb, hs, spec;
                        method = :collapsed)
    τ2_pool = fit.hs.τ²
    τ2_raw = tau_pm_by_kind(y, sigma2, ctr, Arows, fit, org_type, kinds, idx, τ2_pool)
    # The per-type marginal estimator is used only where it agrees with the
    # pooled MCEM variance to within a factor of five.  On the shikimate
    # pathway both types fall to a small fraction of the pooled value, which is
    # the estimator losing the between-organism signal against a record scale of
    # 11 to 13 rather than a type difference, and a floor value there would
    # release unmeasured-organism intervals barely wider than the consensus.
    τ2_of = Dict{String,Float64}(); τ2_src = Dict{String,String}()
    for k in unique(kinds)
        t = get(τ2_raw, k, NaN)
        # Same rule as tau2_by_type_with_fallback in hierarchy_utils.jl, written out here
        # so that each type that falls back is reported as the set is built.
        if isfinite(t) && 0.2 * τ2_pool <= t <= 5.0 * τ2_pool
            τ2_of[k] = t; τ2_src[k] = "per_type"
        else
            τ2_of[k] = τ2_pool; τ2_src[k] = "pooled_fallback"
            isfinite(t) && @printf("    %s %s: per-type tau2 %.3f is outside [0.2, 5] x pooled %.3f, using the pooled value\n",
                                   string(pathway), k, t, τ2_pool)
        end
    end

    # one row per distinct curated quantity
    rows = NamedTuple[]
    seen = Dict{Int,Int}()
    for i in eachindex(y)
        haskey(seen, qrows[i]) && continue
        seen[qrows[i]] = i
    end
    S = size(fit.qsamps, 2)
    for (qr, i) in sort(collect(seen), by = first)
        a = view(Arows, i, :)
        mu_s = [dot(a, view(fit.qsamps, :, s)) for s in 1:S]
        k = kinds[i]
        τ2 = get(τ2_of, k, τ2_pool)
        cons = quantile(mu_s, [0.025, 0.5, 0.975])
        rr = MersenneTwister(SEED + qr)
        newo = sort!([mu_s[rand(rr, 1:S)] + sqrt(τ2) * randn(rr) for _ in 1:NDRAW])
        no = quantile(newo, [0.025, 0.5, 0.975]; sorted = true)
        nrec = count(==(qr), qrows)
        norg = length(unique(org[qrows .== qr]))
        u = to_natural(k, 0.0)[2]
        push!(rows, (pathway = string(pathway), kind = k, ec_code = ecs[i],
            substrate = subs[i], Q_row = qr, n_records = nrec, n_organisms = norg,
            n_mutant_records = count(j -> qrows[j] == qr && mm.mutant[j], eachindex(y)),
            n_wildtype_labelled = count(j -> qrows[j] == qr && mm.wildtype[j], eachindex(y)),
            units = u,
            consensus_lo = to_natural(k, cons[1])[1],
            consensus_median = to_natural(k, cons[2])[1],
            consensus_hi = to_natural(k, cons[3])[1],
            neworganism_lo = to_natural(k, no[1])[1],
            neworganism_median = to_natural(k, no[2])[1],
            neworganism_hi = to_natural(k, no[3])[1],
            log_consensus_median = cons[2], log_consensus_sd = std(mu_s),
            tau2_type = τ2, tau2_source = get(τ2_src, k, "pooled_fallback"),
            tau2_pooled = τ2_pool, tau2_per_type_raw = get(τ2_raw, k, NaN),
            c_record_scale = cg))
    end
    df = DataFrame(rows)
    # kinetic quantities of the pathway with and without records
    nkin = try
        blocks = getfield(post["μ_cov"], :blocks)
        size(post["Q_obs"], 1) - size(blocks[1], 1)
    catch
        -1
    end
    @printf("%-18s %3d quantities, c=%.2f, tau2 kcat/KM = %.2f/%.2f, median new-organism/consensus width ratio %.1f, kinetic rows in the pathway %d\n",
            string(pathway), nrow(df), cg,
            get(τ2_of, "kcat", NaN), get(τ2_of, "KM", NaN),
            median(log.(df.neworganism_hi ./ df.neworganism_lo) ./
                   log.(df.consensus_hi ./ df.consensus_lo)), nkin)

    # organism-conditional posteriors
    orgrows = NamedTuple[]
    bs = fit.bsamps
    lv = fit.grp_levels
    orgcount = Dict{String,Int}()
    for o in org
        orgcount[o] = get(orgcount, o, 0) + 1
    end
    named = [o for o in sort(collect(keys(orgcount)))
             if orgcount[o] >= MIN_ORG_REC || o in NAMED_ORGS]
    for o in named, (qr, i) in sort(collect(seen), by = first)
        k = kinds[i]
        cell = string(o, "|", k)
        gi = findfirst(==(cell), lv)
        gi === nothing && continue          # organism has no records of this type
        a = view(Arows, i, :)
        mu_s = [dot(a, view(fit.qsamps, :, s)) for s in 1:S]
        bo = vec(bs[gi, :])
        S2 = min(length(mu_s), length(bo))
        dr = sort!(mu_s[1:S2] .+ bo[1:S2])
        qs = quantile(dr, [0.025, 0.5, 0.975]; sorted = true)
        ndir = count(j -> qrows[j] == qr && org[j] == o, eachindex(y))
        ntype = count(j -> kinds[j] == k && org[j] == o, eachindex(y))
        push!(orgrows, (pathway = string(pathway), organism = o, kind = k,
            ec_code = ecs[i], substrate = subs[i], Q_row = qr,
            n_records_organism_quantity = ndir, n_records_organism_type = ntype,
            n_mutant_records_organism_quantity =
                count(j -> qrows[j] == qr && org[j] == o && mm.mutant[j], eachindex(y)),
            organism_own_median = (jj = findall(j -> qrows[j] == qr && org[j] == o, eachindex(y));
                                   isempty(jj) ? NaN : to_natural(k, median(y[jj]))[1]),
            units = to_natural(k, 0.0)[2],
            organism_lo = to_natural(k, qs[1])[1], organism_median = to_natural(k, qs[2])[1],
            organism_hi = to_natural(k, qs[3])[1],
            log_offset_median = median(bo), log_offset_sd = std(bo),
            consensus_median = to_natural(k, median(mu_s))[1]))
    end
    # The joint posterior of the basis vector q.  Every derived quantity of the
    # pathway, including the rows that carry no record and the thermodynamic
    # block, is a fixed linear function of q, so releasing the posterior mean and
    # covariance (and the draws themselves) lets a modeller reconstruct any
    # quantity, and any function of several of them, with the Haldane relations
    # holding exactly.  The tabulated rows above are the curated quantities only.
    qm = vec(mean(fit.qsamps, dims = 2))
    qc = cov(fit.qsamps, dims = 2)
    CSV.write(joinpath(outdir, "basis_posterior_mean_$(pathway).csv"),
              DataFrame(index = 1:length(qm), q_mean = qm, q_sd = sqrt.(diag(qc))))
    CSV.write(joinpath(outdir, "basis_posterior_cov_$(pathway).csv"),
              DataFrame(qc, :auto))
    CSV.write(joinpath(outdir, "basis_posterior_draws_$(pathway).csv"),
              DataFrame(permutedims(fit.qsamps), :auto))
    # The map from basis coordinates to biochemical quantities: one row per curated
    # derived quantity, giving its row of Q, so that a user can form that quantity's
    # posterior from the basis draws as a dot product and any function of several
    # quantities with the Haldane relations holding exactly.
    seenq = sort(collect(seen), by = first)
    dm = DataFrame(hcat([Arows[i, :] for (_, i) in seenq]...) |> permutedims, :auto)
    insertcols!(dm, 1, :Q_row => [qr for (qr, _) in seenq],
                :kind => [kinds[i] for (_, i) in seenq],
                :ec_code => [ecs[i] for (_, i) in seenq],
                :substrate => [subs[i] for (_, i) in seenq])
    CSV.write(joinpath(outdir, "basis_design_$(pathway).csv"), dm)
    @printf("    basis posterior: D = %d, %d draws (mean, covariance and draws written)\n",
            length(qm), size(fit.qsamps, 2))

    odf = DataFrame(orgrows)
    for o in intersect(NAMED_ORGS, named)
        sub = odf[odf.organism .== o, :]
        nrow(sub) == 0 && continue
        @printf("    %-26s %3d quantities (%3d with direct records), median log offset kcat/KM = %s\n",
                o, nrow(sub), count(>(0), sub.n_records_organism_quantity),
                join([@sprintf("%+.2f", median(sub.log_offset_median[sub.kind .== kk]))
                      for kk in ("kcat", "KM") if any(sub.kind .== kk)], "/"))
    end
    (df, odf)
end

parse(Int, get(ENV, "PB_REV_OUTER_MULT", "1")) == 1 || !isempty(get(ENV, "PB_REV_SUFFIX", "")) ||
    error("PB_REV_OUTER_MULT is not 1: set PB_REV_SUFFIX as well, so that the published outputs are kept")
println("Released balanced parameter set with checked uncertainty")
println("  started = $(now())")
outdir = joinpath(dirname(@__DIR__), "output",
                 (haskey(ENV, "PB_REV_EXCLUDE_MUTANTS") ? "paramset_wildtype" : "paramset") *
                 get(ENV, "PB_REV_SUFFIX", ""))
mkpath(outdir)
allrows = DataFrame(); allorg = DataFrame()
for p in PATHWAYS
    df, odf = run_pathway(p)
    CSV.write(joinpath(outdir, "balanced_parameters_$(p).csv"), df)
    CSV.write(joinpath(outdir, "balanced_parameters_by_organism_$(p).csv"), odf)
    global allrows = isempty(allrows) ? df : vcat(allrows, df)
    global allorg = isempty(allorg) ? odf : vcat(allorg, odf)
end
CSV.write(joinpath(outdir, "balanced_parameters.csv"), allrows)
CSV.write(joinpath(outdir, "balanced_parameters_by_organism.csv"), allorg)
@printf("wrote %d quantities across %d pathways, and %d organism-conditional rows\n",
        nrow(allrows), length(PATHWAYS), nrow(allorg))
println("Done at $(now())")
