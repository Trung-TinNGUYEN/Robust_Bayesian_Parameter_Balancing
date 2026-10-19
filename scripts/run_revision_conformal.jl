# run_revision_conformal.jl
#
# Closing the residual coverage gap, added for the revision.
#
# The recalibrated hierarchy lands at 0.91-0.92 against a nominal 0.95, and the
# residual gap cannot be closed by symmetric parametric tails: a
# variance-matched Student-t moves the 99%+ quantiles but barely the 95% one.
# This driver therefore evaluates, on top of the recalibrated organism-by-type
# hierarchy and under the same leave-one-source-out protocol, three predictive
# constructions that differ ONLY in how the interval is formed:
#
#   gaussian      the parametric Gaussian predictive (the sigmacal reference);
#   t_re          Student-t organism effects: the unseen-organism draw b* uses
#                 a variance-matched t whose degrees of freedom are estimated
#                 from the kurtosis of the training cells' fitted effects
#                 (heavy tails in the hierarchy, not the likelihood);
#   conformal     organism-level quantile calibration computed from the SAME
#                 fit that is being calibrated: the standardised
#                 unseen-organism prediction errors of the training organisms,
#                 weighted so every organism counts equally, supply empirical
#                 quantiles at every nominal level.  This variant is reported
#                 as a diagnostic: because the fit saw the scoring organisms'
#                 records, their scores are shrunk and the intervals inherit
#                 an in-sample optimism.
#   conformal_split  the proper organism-level split-conformal construction:
#                 the training organisms are split in half, the recalibrated
#                 hierarchy is fitted on one half only, the scores come from
#                 the other half, which the fit never saw, and the held-out
#                 organism is predicted from the half-fit with the calibrated
#                 quantiles.  Under exchangeability of organisms the marginal
#                 coverage of this construction is finite-sample correct; the
#                 price is a fit on half the organisms.
#
# Everything is estimated on the training fold: the error scale c (within-cell
# iteration, as in run_revision_sigmacal.jl), the degrees of freedom nu_b, and
# the conformal quantiles.  The cross-validation remains leakage-free.
#
# Outputs, written under output/<pathway>/results/revision/:
#   conformal_summary_<pathway>.csv    one row per predictive construction
#   conformal_coverage_<pathway>.csv   coverage at every nominal level
#
# Environment controls:
#   PB_REV_PATHWAYS  subset of {glycolysis,tca,pentose_phosphate,shikimate}
#   PB_REV_SIGMACAL_ITERS  EB iterations for the error scale (default 4)

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
const REV_PATHWAYS = Symbol.(strip.(split(get(ENV, "PB_REV_PATHWAYS",
    "glycolysis,tca,pentose_phosphate,shikimate"), ",")))
const SIGMACAL_ITERS = parse(Int, get(ENV, "PB_REV_SIGMACAL_ITERS", "4"))
const C_MIN, C_MAX = 0.25, 25.0
const N_PRED = 800

_exkurt(x) = (m = mean(x); s = std(x, corrected = false); s <= 0 ? 0.0 : mean(((x .- m) ./ s) .^ 4) - 3.0)
nu_b_from_kurtosis(k) = k <= 0.05 ? 50.0 : clamp(4.0 + 6.0 / k, 4.5, 50.0)

function posterior_q_moments(fit::FoldFit, Arows_tr::AbstractMatrix)
    M = Arows_tr * fit.qsamps
    return vec(mean(M, dims = 2)), vec(var(M, dims = 2))
end

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

"""
    org_weighted_quantile(z, orgs, p) -> Float64

Empirical quantile of `z` in which every organism carries equal total weight,
so organisms with many records do not dominate the calibration.
"""
function weighted_quantile(z::Vector{Float64}, w::Vector{Float64}, p::Float64,
                           G_eff::Float64)
    # Finite-sample conformal correction: the weighted analogue of taking the
    # ceil((n+1)p)-th order statistic, with n replaced by the effective number
    # of calibration units.
    padj = p >= 0.5 ? min(1.0, p * (G_eff + 1) / G_eff) :
                      max(0.0, 1.0 - (1.0 - p) * (G_eff + 1) / G_eff)
    idx = sortperm(z)
    tot = sum(w)
    acc = 0.0
    for i in idx
        acc += w[i] / tot
        acc >= padj && return z[i]
    end
    return z[idx[end]]
end

function calib_weights(orgs::Vector{String}, mode::Symbol)
    counts = Dict{String,Int}()
    for o in orgs
        counts[o] = get(counts, o, 0) + 1
    end
    G = length(counts)
    if mode == :organism
        return [1.0 / (G * counts[orgs[i]]) for i in eachindex(orgs)], Float64(G)
    else
        n = length(orgs)
        # Effective units for dependent records within organisms: organisms,
        # weighted by their record share.
        return fill(1.0 / n, n), Float64(G)
    end
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

    slow = pathway in (:tca,)
    spec = FoldFitSpec(model = :gaussian, hierarchical = true,
        n_outer = slow ? 4 : 3, n_mcmc = slow ? 250 : 150,
        burnin = slow ? 80 : 40, thin = 1,
        n_keep_min = 60, n_pred_draws = 400, b_draws_ll = 50,
        posterior_refresh = 1, adaptive_stop = false, rtol_outer = 1e-3,
        min_outer = 2, tau_update = :ig, tau_pc_rate = 0.0,
        ν_init = 50.0, α_init = 0.0,
        update_nu = false, update_alpha = false, refine_shape = false,
        shape_opt_tol = 1e-3, skew_location = :mean,
        nu_penalty = 0.0, nu_target = 50.0, alpha_penalty = 0.0)

    methods = ["gaussian", "t_re", "conformal", "conformal_split", "conformal_split_rw"]
    folds = sort!(unique(org)); nf = length(folds); n = length(y); K = length(REV_LEVELS)
    idx95 = findfirst(x -> isapprox(x, 0.95; atol = 1e-8), REV_LEVELS)

    covered = Dict(m => Matrix{Bool}(undef, n, K) for m in methods)
    width95 = Dict(m => Vector{Float64}(undef, n) for m in methods)
    nubs = fill(NaN, nf)
    seeds = _pb_thread_seeds(MersenneTwister(REV_SEED), nf)

    println("\n=== $(pathway): N=$n organisms=$nf")

    Threads.@threads :dynamic for f in 1:nf
        rng_f = _pb_rng_from_seed(seeds[f])
        te = findall(==(folds[f]), org); tr = findall(!=(folds[f]), org)
        Ar_tr = Arows[tr, :]
        cells_tr = org_type[tr]

        # Recalibrated fit, as in run_revision_sigmacal.jl.
        cg = 1.0
        hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
        fit = fit_fold_mcem(rng_f, A0, b0, Ar_tr, y[tr], cg .* sigma2[tr],
                            org_type[tr], eb, hs, spec; method = :collapsed)
        for _ in 1:SIGMACAL_ITERS
            m, _ = posterior_q_moments(fit, Ar_tr)
            cg = c_update_within(y[tr], m, sigma2[tr], cells_tr, eachindex(tr))
            hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
            fit = fit_fold_mcem(rng_f, A0, b0, Ar_tr, y[tr], cg .* sigma2[tr],
                                org_type[tr], eb, hs, spec; method = :collapsed)
        end
        τ2 = fit.hs.τ²

        # Degrees of freedom for the t organism effect: kurtosis of the fitted
        # cell effects with at least three records (fewer are shrinkage-dominated).
        counts = Dict{String,Int}()
        for c in cells_tr
            counts[c] = get(counts, c, 0) + 1
        end
        bmeans = Float64[]
        for (i, c) in enumerate(fit.grp_levels)
            counts[c] >= 3 && push!(bmeans, mean(view(fit.bsamps, i, :)))
        end
        νb = length(bmeans) >= 10 ? nu_b_from_kurtosis(_exkurt(bmeans)) : 50.0
        nubs[f] = νb

        # Conformal calibration scores: every training record standardised
        # under the unseen-organism predictive (fitted cell effect excluded).
        mq, vq = posterior_q_moments(fit, Ar_tr)
        z = [(y[tr][j] - mq[j]) / sqrt(vq[j] + τ2 + cg * sigma2[tr][j])
             for j in eachindex(tr)]
        orgs_tr = org[tr]
        w_ow, Gow = calib_weights(orgs_tr, :organism)
        lo = Vector{Float64}(undef, K); hi = Vector{Float64}(undef, K)
        for (k, lv) in enumerate(REV_LEVELS)
            lo[k] = weighted_quantile(z, w_ow, (1 - lv) / 2, Gow)
            hi[k] = weighted_quantile(z, w_ow, (1 + lv) / 2, Gow)
        end

        # Proper split: fit on half the training organisms, calibrate on the
        # other half, which the fit never saw.
        orgs_pool = sort!(unique(org[tr]))
        perm = randperm(rng_f, length(orgs_pool))
        half = Set(orgs_pool[perm[1:div(length(orgs_pool), 2)]])
        trA = [j for j in tr if org[j] in half]
        trB = [j for j in tr if !(org[j] in half)]
        ArA = Arows[trA, :]
        cA = 1.0
        hsA = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
        fitA = fit_fold_mcem(rng_f, A0, b0, ArA, y[trA], cA .* sigma2[trA],
                             org_type[trA], eb, hsA, spec; method = :collapsed)
        for _ in 1:SIGMACAL_ITERS
            mA, _ = posterior_q_moments(fitA, ArA)
            cA = c_update_within(y[trA], mA, sigma2[trA], org_type[trA], eachindex(trA))
            hsA = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
            fitA = fit_fold_mcem(rng_f, A0, b0, ArA, y[trA], cA .* sigma2[trA],
                                 org_type[trA], eb, hsA, spec; method = :collapsed)
        end
        τ2A = fitA.hs.τ²
        MB = Arows[trB, :] * fitA.qsamps
        mB = vec(mean(MB, dims = 2)); vB = vec(var(MB, dims = 2))
        zB = [(y[trB[j]] - mB[j]) / sqrt(vB[j] + τ2A + cA * sigma2[trB[j]])
              for j in eachindex(trB)]
        orgsB = org[trB]
        wB_ow, GB = calib_weights(orgsB, :organism)
        wB_rw, _ = calib_weights(orgsB, :record)
        loS = Vector{Float64}(undef, K); hiS = Vector{Float64}(undef, K)
        loR = Vector{Float64}(undef, K); hiR = Vector{Float64}(undef, K)
        for (k, lv) in enumerate(REV_LEVELS)
            loS[k] = weighted_quantile(zB, wB_ow, (1 - lv) / 2, GB)
            hiS[k] = weighted_quantile(zB, wB_ow, (1 + lv) / 2, GB)
            loR[k] = weighted_quantile(zB, wB_rw, (1 - lv) / 2, GB)
            hiR[k] = weighted_quantile(zB, wB_rw, (1 + lv) / 2, GB)
        end
        SA = size(fitA.qsamps, 2)

        # Held-out predictions.
        S = size(fit.qsamps, 2)
        tscale = sqrt((νb - 2) / νb)
        for i in te
            a = view(Arows, i, :)
            mu_s = [dot(a, view(fit.qsamps, :, s)) for s in 1:S]
            m_i = mean(mu_s); v_i = var(mu_s)
            sd_par = sqrt(v_i + τ2 + cg * sigma2[i])
            # Monte Carlo predictive draws for the two parametric methods.
            dg = Vector{Float64}(undef, N_PRED)
            dt = Vector{Float64}(undef, N_PRED)
            for r in 1:N_PRED
                s = rand(rng_f, 1:S)
                eps = sqrt(cg * sigma2[i]) * randn(rng_f)
                bg = sqrt(τ2) * randn(rng_f)
                bt = sqrt(τ2) * tscale * rand(rng_f, TDist(νb))
                dg[r] = mu_s[s] + bg + eps
                dt[r] = mu_s[s] + bt + eps
            end
            sort!(dg); sort!(dt)
            muA = [dot(a, view(fitA.qsamps, :, s)) for s in 1:SA]
            mA_i = mean(muA)
            sdA_i = sqrt(var(muA) + τ2A + cA * sigma2[i])
            for (k, lv) in enumerate(REV_LEVELS)
                a1 = (1 - lv) / 2; a2 = (1 + lv) / 2
                qgl = quantile(dg, a1; sorted = true); qgu = quantile(dg, a2; sorted = true)
                qtl = quantile(dt, a1; sorted = true); qtu = quantile(dt, a2; sorted = true)
                ccl = m_i + lo[k] * sd_par; ccu = m_i + hi[k] * sd_par
                csl = mA_i + loS[k] * sdA_i; csu = mA_i + hiS[k] * sdA_i
                crl = mA_i + loR[k] * sdA_i; cru = mA_i + hiR[k] * sdA_i
                covered["gaussian"][i, k] = qgl <= y[i] <= qgu
                covered["t_re"][i, k] = qtl <= y[i] <= qtu
                covered["conformal"][i, k] = ccl <= y[i] <= ccu
                covered["conformal_split"][i, k] = csl <= y[i] <= csu
                covered["conformal_split_rw"][i, k] = crl <= y[i] <= cru
                if k == idx95
                    width95["gaussian"][i] = qgu - qgl
                    width95["t_re"][i] = qtu - qtl
                    width95["conformal"][i] = ccu - ccl
                    width95["conformal_split"][i] = csu - csl
                    width95["conformal_split_rw"][i] = cru - crl
                end
            end
        end
    end

    summary = DataFrame(pathway = String[], method = String[], Cov95 = Float64[],
                        Cov95_org = Float64[],
                        Width95 = Float64[], nu_b = Float64[], n_folds = Int[])
    covdf = DataFrame(pathway = String[], method = String[], nominal = Float64[],
                      empirical = Float64[])
    orgset = sort!(unique(org))
    byorg = Dict(o => findall(==(o), org) for o in orgset)
    for m in methods
        cov_org = mean(mean(covered[m][byorg[o], idx95]) for o in orgset)
        push!(summary, (string(pathway), m, mean(covered[m][:, idx95]), cov_org,
                        mean(width95[m]), m == "t_re" ? mean(filter(isfinite, nubs)) : NaN,
                        nf))
        for (k, lv) in enumerate(REV_LEVELS)
            push!(covdf, (string(pathway), m, lv, mean(covered[m][:, k])))
        end
        @printf("  %-18s Cov95=%.4f Cov95org=%.4f Width95=%.2f%s\n", m,
                mean(covered[m][:, idx95]), cov_org,
                mean(width95[m]),
                m == "t_re" ? @sprintf(" nu_b=%.1f", mean(filter(isfinite, nubs))) : "")
    end

    CSV.write(joinpath(results_dir, "conformal_summary_$(pathway).csv"), summary)
    CSV.write(joinpath(results_dir, "conformal_coverage_$(pathway).csv"), covdf)
    return nothing
end

println("Revision quantile-calibration analysis")
println("  threads = $(Threads.nthreads())")
println("  started = $(now())")
for pw in REV_PATHWAYS
    run_pathway(pw)
end
println("Done at $(now())")
