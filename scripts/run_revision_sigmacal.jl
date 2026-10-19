# run_revision_sigmacal.jl
#
# Empirical recalibration of the record-level error convention, added for the
# revision.
#
# Limitation addressed:
#   BRENDA reports no per-record uncertainties, so parameter balancing assigns
#   every record the same half-decade log-scale standard deviation
#   sigma = 0.5 log 10.  The residual diagnostics show this convention
#   understates the observed dispersion by a factor of about two and a half,
#   and the manuscript identifies it as the proximate reason predictive
#   coverage stays below nominal in every fitted model.  The obvious question,
#   previously not tested, is whether an error scale estimated from the data
#   restores nominal coverage without any new curation.
#
# Design.  Organism-level leave-one-source-out, as in every other real-data
# analysis.  For the recalibrated cells, a multiplicative error scale c is
# estimated on the training fold only, by a within-cell moment iteration.
# Records in the same organism-by-type cell share the same random effect
# exactly, so with r_i = y_i - a_i' qhat the marginal consensus residual,
# the within-cell dispersion identifies c free of the between-cell variance:
#
#   E[ sum_g sum_{i in g} (r_i - rbar_g)^2 ]  =  c * sum_g sum_{i in g} sigma2_i (1 - 1/n_g),
#
# up to a second-order term from the uncertainty in qhat, which is common to
# the records of a cell and largely cancels in the contrast.  The update
# neither subtracts the fitted tau^2 nor conditions on the fitted random
# effect, so it inherits neither's bias: conditional residuals understate c
# because a cell's effect is partly fit to its own records, and marginal
# residuals require an estimate of tau^2 whose error is amplified by 1/sigma2.
# Cells with a single record contribute nothing.  The iteration refits the
# model under c_k because qhat and tau2 respond to the reweighting; c itself
# stabilises after the first pass.  The `_pt` cell estimates one scale per
# quantity type (kcat, KM) with the same update restricted to the cells of
# that type.  Held-out records are then predicted with variance c * sigma2
# (per-type where applicable).  Everything is training-fold only, so the
# cross-validation remains leakage-free.
#
# Outputs, written under output/<pathway>/results/revision/:
#   sigmacal_summary_<pathway>.csv    one row per model cell
#   sigmacal_coverage_<pathway>.csv   predictive coverage at every nominal level
#
# Environment controls:
#   PB_REV_PATHWAYS  comma-separated subset of {glycolysis,tca} (default both)
#   PB_REV_SIGMACAL_ITERS  empirical-Bayes iterations (default 4)

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
include(joinpath(dirname(@__DIR__), "src", "experiment_runners.jl"))

const REV_LEVELS = vcat(0.01, collect(0.05:0.05:0.95), 0.99)
const REV_SEED = 20260805
const REV_PATHWAYS = Symbol.(strip.(split(get(ENV, "PB_REV_PATHWAYS", "glycolysis,tca"), ",")))
const SIGMACAL_ITERS = parse(Int, get(ENV, "PB_REV_SIGMACAL_ITERS", "4"))
const C_MIN, C_MAX = 0.25, 25.0

"""
    posterior_q_moments(fit, Arows_tr) -> (m, v)

Posterior mean and variance of the marginal linear predictor `a_i' q` (basis
parameters only, no random effect) for every training record, from the kept
posterior samples.
"""
function posterior_q_moments(fit::FoldFit, Arows_tr::AbstractMatrix)
    M = Arows_tr * fit.qsamps                       # n_tr x S
    return vec(mean(M, dims = 2)), vec(var(M, dims = 2))
end

"""
    c_update_within(y, m, sigma2, cells, idx) -> Float64

Within-cell moment estimator of the error scale on the record subset `idx`:
`sum_g sum_{i in g}(r_i - rbar_g)^2 / sum_g sum_{i in g} sigma2_i (1 - 1/n_g)`
with `r_i = y_i - m_i`, clamped to `[C_MIN, C_MAX]`.  `m` is the posterior mean
of `a_i' q`, `cells` the random-effect cell of every record, and `sigma2` the
ORIGINAL (unscaled) conventional variances.  Cells with one record in `idx`
contribute nothing.
"""
function c_update_within(y, m, sigma2, cells, idx)
    bygroup = Dict{String,Vector{Int}}()
    for i in idx
        push!(get!(bygroup, cells[i], Int[]), i)
    end
    num = 0.0; den = 0.0
    for (_, ig) in bygroup
        ng = length(ig)
        ng >= 2 || continue
        rbar = mean(y[i] - m[i] for i in ig)
        for i in ig
            num += (y[i] - m[i] - rbar)^2
            den += sigma2[i] * (1.0 - 1.0 / ng)
        end
    end
    den > 0 || return 1.0
    return clamp(num / den, C_MIN, C_MAX)
end

function run_pathway(pathway::Symbol)
    root = joinpath(dirname(@__DIR__), "output", string(pathway))
    results_dir = joinpath(root, "results", "revision"); mkpath(results_dir)
    brenda = load(joinpath(root, "data", "brenda_inputs_$(pathway).jld2"))
    post = load(joinpath(root, "data", "posterior_$(pathway).jld2"))
    y = Vector{Float64}(brenda["y"]); sigma2 = Vector{Float64}(brenda["σ2"])
    Arows = Matrix{Float64}(brenda["Arows"]); org = Vector{String}(brenda["grp"])
    kinds = Vector{String}(getproperty(brenda["meta"], :kind))
    q0 = vec(Array(post["q_prior"])); C0 = Matrix(Array(post["C_prior"]))
    eb = EBSpec(q0 = q0, C0 = C0); d = size(Arows, 2)
    A0, b0 = _prepare_information_pair(Arows, d, Matrix{Float64}(brenda["A0"]),
                                       Vector{Float64}(brenda["b0"]);
                                       deduplicate_thermo_block = true)
    org_type = [string(org[i], "|", kinds[i]) for i in eachindex(org)]

    is_tca = pathway == :tca
    spec = FoldFitSpec(model = :gaussian, hierarchical = true,
        n_outer = is_tca ? 4 : 3, n_mcmc = is_tca ? 250 : 150,
        burnin = is_tca ? 80 : 40, thin = 1,
        n_keep_min = 60, n_pred_draws = 400, b_draws_ll = 50,
        posterior_refresh = 1, adaptive_stop = false, rtol_outer = 1e-3,
        min_outer = 2, tau_update = :ig, tau_pc_rate = 0.0,
        ν_init = 50.0, α_init = 0.0,
        update_nu = false, update_alpha = false, refine_shape = false,
        shape_opt_tol = 1e-3, skew_location = :mean,
        nu_penalty = 0.0, nu_target = 50.0, alpha_penalty = 0.0)

    # scope: :none (baseline, c = 1), :global (one scale), :pertype (kcat, KM)
    cells = [("gaussian_RE_orgtype",             :none),
             ("gaussian_sigmacal_RE_orgtype",    :global),
             ("gaussian_sigmacal_pt_RE_orgtype", :pertype)]

    summary = DataFrame(pathway = String[], cell = String[],
                        CRPS = Float64[], CRPS_median = Float64[], Cov95 = Float64[],
                        Width95 = Float64[], PredRMSE = Float64[], tau2 = Float64[],
                        c_all = Float64[], c_kcat = Float64[], c_KM = Float64[],
                        n_folds = Int[])
    covdf = DataFrame(pathway = String[], cell = String[], nominal = Float64[],
                      empirical = Float64[])
    idx95 = findfirst(x -> isapprox(x, 0.95; atol = 1e-8), REV_LEVELS)
    folds = sort!(unique(org)); nf = length(folds); n = length(y); K = length(REV_LEVELS)

    println("\n=== $(pathway): N=$n organisms=$nf, EB iterations=$SIGMACAL_ITERS")

    for (label, scope) in cells
        print("  -> $label ... "); flush(stdout); t0 = time()
        crps = Vector{Float64}(undef, n); perr = Vector{Float64}(undef, n)
        covered = Matrix{Bool}(undef, n, K); widths = Matrix{Float64}(undef, n, K)
        tau2s = Vector{Float64}(undef, nf)
        c_all = fill(NaN, nf); c_kc = fill(NaN, nf); c_km = fill(NaN, nf)
        seeds = _pb_thread_seeds(MersenneTwister(REV_SEED), nf)

        Threads.@threads :dynamic for f in 1:nf
            rng_f = _pb_rng_from_seed(seeds[f])
            te = findall(==(folds[f]), org); tr = findall(!=(folds[f]), org)
            Ar_tr = Arows[tr, :]
            kinds_tr = kinds[tr]

            # Per-record error-scale multiplier on the training fold.
            cvec = ones(length(tr))
            fit = nothing
            hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
            fit = fit_fold_mcem(rng_f, A0, b0, Ar_tr, y[tr], cvec .* sigma2[tr],
                                org_type[tr], eb, hs, spec; method = :collapsed)
            cf = Dict("kcat" => 1.0, "KM" => 1.0); cg = 1.0
            cells_tr = org_type[tr]
            if scope != :none
                for _ in 1:SIGMACAL_ITERS
                    m, _ = posterior_q_moments(fit, Ar_tr)
                    ytr = y[tr]; s2tr = sigma2[tr]
                    if scope == :global
                        cg = c_update_within(ytr, m, s2tr, cells_tr, eachindex(tr))
                        cvec .= cg
                    else
                        for k in ("kcat", "KM")
                            idxk = findall(==(k), kinds_tr)
                            isempty(idxk) && continue
                            cf[k] = c_update_within(ytr, m, s2tr, cells_tr, idxk)
                        end
                        cvec .= [cf[k] for k in kinds_tr]
                    end
                    hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
                    fit = fit_fold_mcem(rng_f, A0, b0, Ar_tr, y[tr], cvec .* sigma2[tr],
                                        org_type[tr], eb, hs, spec; method = :collapsed)
                end
            end
            tau2s[f] = fit.hs.τ²
            if scope == :global
                c_all[f] = cg
            elseif scope == :pertype
                c_kc[f] = cf["kcat"]; c_km[f] = cf["KM"]
            end

            for i in te
                ci = scope == :none ? 1.0 : (scope == :global ? cg : cf[kinds[i]])
                p = predict_row_metrics_mc_hier(rng_f, view(Arows, i, :), y[i],
                                                ci * sigma2[i], org_type[i], fit, spec;
                                                cover_levels = REV_LEVELS)
                crps[i] = p.crps; perr[i] = p.pred_error
                covered[i, :] = p.covered; widths[i, :] = p.widths
            end
        end

        ok = isfinite.(crps)
        _fmean(vv) = (fv = filter(isfinite, vv); isempty(fv) ? NaN : mean(fv))
        push!(summary, (string(pathway), label,
                        mean(crps[ok]), median(crps[ok]), mean(covered[:, idx95]),
                        mean(widths[ok, idx95]), sqrt(mean(perr[ok] .^ 2)),
                        mean(filter(isfinite, tau2s)),
                        _fmean(c_all), _fmean(c_kc), _fmean(c_km), nf))
        for (k, lv) in enumerate(REV_LEVELS)
            push!(covdf, (string(pathway), label, lv, mean(covered[:, k])))
        end
        @printf("CRPS=%.4f (med %.4f) Cov95=%.4f tau2=%.3f c=%.2f c_kcat=%.2f c_KM=%.2f  [%.1f min]\n",
                mean(crps[ok]), median(crps[ok]), mean(covered[:, idx95]),
                mean(filter(isfinite, tau2s)),
                _fmean(c_all), _fmean(c_kc), _fmean(c_km), (time() - t0) / 60)
        flush(stdout)
    end

    CSV.write(joinpath(results_dir, "sigmacal_summary_$(pathway).csv"), summary)
    CSV.write(joinpath(results_dir, "sigmacal_coverage_$(pathway).csv"), covdf)
    println("\n  $(pathway) sigma-recalibration summary")
    show(stdout, summary; allrows = true, allcols = true); println()
    return nothing
end

println("Revision sigma-recalibration analysis")
println("  threads = $(Threads.nthreads()), EB iterations = $SIGMACAL_ITERS")
println("  started = $(now())")
for pw in REV_PATHWAYS
    run_pathway(pw)
end
println("Done at $(now())")
