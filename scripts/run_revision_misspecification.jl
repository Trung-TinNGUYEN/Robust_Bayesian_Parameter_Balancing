# run_revision_misspecification.jl
#
# Cross-family misspecification and unknown-shape experiments added for the
# revision.
#
# Reviewer requests addressed here:
#   (i)   "What happens if the skewed model is applied to data generated from
#          the t-distribution and vice versa?"
#   (ii)  "Knowledge about the degrees of freedom of the t-distribution and the
#          skewness parameter should be removed to assess a practically relevant
#          scenario."
#   (iii) "It would be useful to show the agreement between nominal and
#          empirical coverage for several different alpha-levels."
#
# ---------------------------------------------------------------------------
# Two generator settings
# ---------------------------------------------------------------------------
# scale  : the curated standard deviation sigma_n is the SCALE of the Student-t
#          generator, so heavy tails inflate the true residual variance to
#          sigma_n^2 nu/(nu-2) and leave the residual mean at zero.  The
#          skew-normal generator keeps the variance sigma_n^2 (its scale is
#          omega = sigma_n / sqrt(1 - 2 delta^2/pi), with
#          delta = alpha / sqrt(1 + alpha^2), under both settings) and keeps its
#          canonical location at zero, so the residual mean moves to
#          omega*delta*sqrt(2/pi).  These are the generators of the originally
#          submitted sweeps.  The Gaussian working likelihood is then
#          misspecified in shape and in one moment, the variance under the
#          Student-t generator and the mean under the skew-normal generator.
#
# moment : the residual is rescaled and recentred so that it has mean zero and
#          variance exactly sigma_n^2 for every sweep value.  Only the SHAPE of
#          the residual distribution changes along the sweep.  This isolates
#          pure shape misspecification and is the stricter test.
#
# ---------------------------------------------------------------------------
# Two observation designs
# ---------------------------------------------------------------------------
# summary : one observation per curated quantity, as in the originally
#           submitted sweeps.  For glycolysis this gives 64 observations for 60
#           basis parameters, so there are only four residual degrees of freedom
#           and the robustness parameters are effectively unidentifiable.
#
# record  : `records_per_quantity` replicate records per curated quantity, which
#           mirrors the raw BRENDA design in which each derived quantity is
#           reported many times.  Here the shape parameter can genuinely be
#           learned, so "estimate the shape rather than assume it" is a
#           meaningful comparison.
#
# ---------------------------------------------------------------------------
# Fitted models
# ---------------------------------------------------------------------------
#   gaussian     Gaussian working likelihood (classical parameter balancing)
#   t_oracle     Student-t, nu fixed at the data-generating value
#   t_est        Student-t, nu estimated from the data (no oracle knowledge)
#   skew_oracle  Skew-normal, alpha fixed at the data-generating value
#   skew_est     Skew-normal, alpha estimated from the data (no oracle knowledge)
#
# Under the Student-t generator, `skew_est` is the cross-family entry; under the
# skew-normal generator, `t_est` is the cross-family entry.
#
# Shape estimation uses a two-stage plug-in scheme.  Stage 1 runs one MCEM outer
# pass at an initial shape and takes the resulting M-step shape update.  Stage 2
# refits with the shape held at that estimate.  Both stages keep the
# empirical-Bayes prior scale fixed at its initialised value, exactly as in the
# oracle fits, so the only difference between `*_oracle` and `*_est` is whether
# the shape parameter was known.  (Re-estimating the prior scale inside every
# synthetic replicate shrinks the prior and under-covers even at the Gaussian
# null point, which would confound the comparison.)
#
# Outputs, written under output/<pathway>/results/revision/:
#   misspec_grid_<pathway>.csv     one row per (setting, dgp, sweep value, fit)
#   misspec_levels_<pathway>.csv   the same, expanded over all nominal levels
#
# Environment controls:
#   PB_REV_NREPS     replicates per cell (default 100)
#   PB_REV_PATHWAYS  comma-separated subset of {glycolysis,tca} (default both)
#   PB_REV_SETTINGS  comma-separated subset of {scale,moment} (default both)
#   PB_REV_DESIGNS   comma-separated subset of {summary,record} (default both)
#   PB_REV_RECORDS   replicate records per curated quantity in the record design
#                    (default 20)

import Pkg
Pkg.activate(dirname(@__DIR__))

using Dates
using LinearAlgebra
using Printf
using Random
using Statistics

using ArviZ
using BlockArrays
using CairoMakie
using Colors
using CSV
using DataFrames
using Distributions
using JLD2
using JSON
using KernelDensity
using LaTeXStrings
using MCMCChains
using ProgressMeter
using SpecialFunctions

include(joinpath(dirname(@__DIR__), "src", "metabolic_reconstruction.jl"))
include(joinpath(dirname(@__DIR__), "src", "parameter_balancing.jl"))
include(joinpath(dirname(@__DIR__), "src", "makie_functions.jl"))
include(joinpath(dirname(@__DIR__), "src", "parameter_balancing_core.jl"))
include(joinpath(dirname(@__DIR__), "src", "prediction_utils.jl"))
include(joinpath(dirname(@__DIR__), "src", "experiment_runners.jl"))

const REV_LEVELS = vcat(0.01, collect(0.05:0.05:0.95), 0.99)
const REV_SEED = 20260805
const REV_NREPS = parse(Int, get(ENV, "PB_REV_NREPS", "100"))
const REV_PATHWAYS = Symbol.(strip.(split(get(ENV, "PB_REV_PATHWAYS", "glycolysis,tca"), ",")))
const REV_SETTINGS = Symbol.(strip.(split(get(ENV, "PB_REV_SETTINGS", "scale,moment"), ",")))
const REV_DESIGNS = Symbol.(strip.(split(get(ENV, "PB_REV_DESIGNS", "summary,record"), ",")))
const REV_RECORDS = parse(Int, get(ENV, "PB_REV_RECORDS", "20"))
const REV_SHAPE_ITERS = parse(Int, get(ENV, "PB_REV_SHAPE_ITERS", "2"))

# Sweep points. PB_REV_NU and PB_REV_ALPHA override them (comma-separated) so that
# points can be added without recomputing the rest: every point is simulated from
# the same master seed, so a point's results do not depend on which others are run.
_rev_values(key, default) = haskey(ENV, key) ?
    [parse(Float64, strip(x)) for x in split(ENV[key], ",") if !isempty(strip(x))] : default
const NU_VALUES = _rev_values("PB_REV_NU", [2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0])
const ALPHA_VALUES = _rev_values("PB_REV_ALPHA", [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
const REV_SUFFIX = get(ENV, "PB_REV_SUFFIX", "")

"""
    simulate_residuals(rng, mu, sigma2, n_reps; model, nu, alpha, setting)

Draw `n_reps` synthetic observation vectors.  With `setting = :scale` the curated
`sigma_n` is the scale of the Student-t generator (variance `sigma_n^2 nu/(nu-2)`), and
the skew-normal generator keeps the variance `sigma_n^2` with its canonical location at
zero (mean `omega*delta*sqrt(2/pi)`); with `setting = :moment` the residual is rescaled
and recentred to have mean zero and variance `sigma_n^2`.
"""
function simulate_residuals(
    rng::AbstractRNG,
    mu::Vector{Float64},
    sigma2::Vector{Float64},
    n_reps::Int;
    model::Symbol,
    nu::Float64,
    alpha::Float64,
    setting::Symbol,
)
    m = length(mu)
    sd = sqrt.(max.(sigma2, eps(Float64)))
    X = zeros(Float64, m, n_reps)

    if model == :gaussian
        @inbounds for r in 1:n_reps, i in 1:m
            X[i, r] = mu[i] + sd[i] * randn(rng)
        end

    elseif model == :t
        nu > 2.0 || error("Student-t generator requires nu > 2 for finite variance")
        scale = setting == :moment ? sqrt((nu - 2.0) / nu) : 1.0
        @inbounds for r in 1:n_reps, i in 1:m
            X[i, r] = mu[i] + sd[i] * scale * rand(rng, TDist(nu))
        end

    elseif model == :skew
        delta = alpha / sqrt(1.0 + alpha^2)
        m1 = sqrt(2.0 / pi)
        denom = 1.0 - (2.0 / pi) * delta^2
        denom > 0.0 || error("invalid skew-normal variance factor at alpha=$alpha")
        omega = sd ./ sqrt(denom)
        # :moment recentres the residual mean at zero; :scale leaves the
        # canonical location at zero, so the residual mean is omega*delta*m1.
        xi = setting == :moment ? (.-omega .* delta .* m1) : zeros(Float64, m)
        @inbounds for r in 1:n_reps, i in 1:m
            X[i, r] = mu[i] + rand(rng, SkewNormal(xi[i], omega[i], alpha))
        end
    else
        error("simulate_residuals: unknown model $model")
    end
    return X
end

_sample_skewness(x) = (m = mean(x); s = std(x, corrected = false);
                       s <= 0 ? 0.0 : mean(((x .- m) ./ s) .^ 3))
_sample_exkurtosis(x) = (m = mean(x); s = std(x, corrected = false);
                         s <= 0 ? 0.0 : mean(((x .- m) ./ s) .^ 4) - 3.0)

"""
    alpha_from_skewness(g1) -> alpha

Invert the skew-normal moment map
`g1 = (4 - pi)/2 * (delta sqrt(2/pi))^3 / (1 - 2 delta^2/pi)^{3/2}`
for the shape parameter.  The attainable range is |g1| < 0.9953; values outside
it are clamped.  Used only to initialise the MCEM shape update, which has
alpha = 0 as a fixed point and therefore cannot leave a symmetric start.
"""
function alpha_from_skewness(g1::Float64)
    s = sign(g1)
    a = min(abs(g1), 0.985)
    a < 1e-4 && return 0.0
    f(delta) = ((4.0 - pi) / 2.0) * (delta * sqrt(2.0 / pi))^3 /
               (1.0 - (2.0 / pi) * delta^2)^1.5
    lo, hi = 0.0, 0.999999
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        f(mid) < a ? (lo = mid) : (hi = mid)
    end
    delta = 0.5 * (lo + hi)
    delta = clamp(delta, 0.0, 0.999)
    return s * delta / sqrt(1.0 - delta^2)
end

"""
    nu_from_kurtosis(k) -> nu

Invert the Student-t excess-kurtosis relation `k = 6/(nu - 4)` for `nu > 4`,
clamped to the range used by the shape grid.  Heavier-than-t tails (very large
`k`) map to the lower clamp.
"""
function nu_from_kurtosis(k::Float64)
    k <= 0.05 && return 50.0
    return clamp(4.0 + 6.0 / k, 2.5, 50.0)
end

"""
    moment_shape_init(fit_model, resid_std) -> (nu0, alpha0)

Method-of-moments starting values for the robustness parameters, computed from
standardised residuals of a preliminary Gaussian fit.
"""
function moment_shape_init(fit_model::Symbol, resid_std::Vector{Float64})
    if fit_model == :t
        return nu_from_kurtosis(_sample_exkurtosis(resid_std)), 0.0
    elseif fit_model == :skew
        return 50.0, alpha_from_skewness(_sample_skewness(resid_std))
    else
        return 50.0, 0.0
    end
end

"""
    rev_spec(model, pathway; nu_init, alpha_init, estimate_shape, skew_location)

Build a one-outer-pass `FoldFitSpec`.  `estimate_shape` only switches the M-step
shape update on; the prior scale is never re-estimated because `n_outer = 1`.
"""
function rev_spec(model::Symbol, pathway::Symbol;
                  nu_init::Float64 = 8.0,
                  alpha_init::Float64 = 0.0,
                  estimate_shape::Bool = false,
                  skew_location::Symbol = :mean)
    is_tca = pathway == :tca
    n_mc, thin, keep =
        model == :gaussian ? (400, 2, 150) :
        model == :t        ? (600, 2, 200) :
                             ((is_tca ? 1600 : 2000), 4, 250)
    return FoldFitSpec(
        model = model, hierarchical = false,
        n_outer = 1, n_mcmc = n_mc, burnin = div(n_mc, 4),
        thin = thin, n_keep_min = keep, n_pred_draws = 800, b_draws_ll = 50,
        posterior_refresh = 1, adaptive_stop = false, rtol_outer = 5e-4,
        min_outer = 1, tau_update = :ig, tau_pc_rate = 0.0,
        ν_init = model == :t ? max(nu_init, 2.05) : 50.0,
        α_init = model == :skew ? alpha_init : 0.0,
        update_nu = estimate_shape && model == :t,
        update_alpha = estimate_shape && model == :skew,
        refine_shape = estimate_shape,
        shape_opt_tol = 1e-3,
        skew_location = skew_location,
        nu_penalty = 0.0, nu_target = max(nu_init, 2.05), alpha_penalty = 0.0)
end

"""
    revision_recovery(posterior_file; ...) -> NamedTuple

Replicated parameter-recovery experiment.  The variance handed to the fitted
likelihood is always the curated `sigma_n^2`, so no data-generating information
leaks into the fit.  When `estimate_shape` is true the shape parameter is
estimated separately within every replicate by the two-stage plug-in scheme.
"""
function revision_recovery(
    posterior_file::String;
    pathway::Symbol,
    n_reps::Int,
    seed::Int,
    data_model::Symbol,
    nu_data::Float64,
    alpha_data::Float64,
    setting::Symbol,
    fit_model::Symbol,
    estimate_shape::Bool,
    shape_fixed::Float64,
    skew_location::Symbol,
    records_per_quantity::Int = 1,
    levels::Vector{Float64} = REV_LEVELS,
)
    rng = MersenneTwister(seed)
    q_true, Q_obs, Sigma_obs, q_prior, C_prior = _load_synth_jld2(posterior_file)
    _, d = size(Q_obs)
    sigma2_0 = collect(max.(diag(Symmetric(Matrix{Float64}(Sigma_obs))), eps(Float64)))

    # Replicate the curated design `records_per_quantity` times.  Each replicate
    # row observes the same derived quantity with an independent measurement
    # error, mirroring the raw BRENDA design.
    Arows = repeat(Matrix(Q_obs), records_per_quantity, 1)
    sigma2 = repeat(sigma2_0, records_per_quantity)
    mu_model = Arows * q_true
    m = length(mu_model)

    X = simulate_residuals(rng, mu_model, sigma2, n_reps;
                           model = data_model, nu = nu_data, alpha = alpha_data,
                           setting = setting)

    grp = ["rep" for _ in 1:m]
    eb = EBSpec(q0 = q_prior, C0 = C_prior)
    A0 = zeros(Float64, d, d)
    b0 = zeros(Float64, d)

    spec_fixed = rev_spec(fit_model, pathway;
                          nu_init = shape_fixed, alpha_init = shape_fixed,
                          estimate_shape = false, skew_location = skew_location)
    spec_gauss_pre = rev_spec(:gaussian, pathway)

    K = length(levels)
    cov_rep = Array{Float64}(undef, d, K, n_reps)
    width_rep = Array{Float64}(undef, d, K, n_reps)
    err_rep = Matrix{Float64}(undef, d, n_reps)
    err2_rep = Matrix{Float64}(undef, d, n_reps)
    nu_hat = Vector{Float64}(undef, n_reps)
    alpha_hat = Vector{Float64}(undef, n_reps)

    # Each replicate is fitted with a freshly initialised hyper-state, so the
    # empirical-Bayes prior scale is identical in every cell.
    fresh_hs() = init_hyper(eb; λ0 = spec_fixed.λ_init, τ²0 = spec_fixed.τ²_init,
                            π0 = spec_fixed.π_init)

    function run_rep!(r::Int)
        y = X[:, r]
        nu_use, alpha_use = shape_fixed, shape_fixed

        if estimate_shape
            # Step 1: preliminary Gaussian fit supplies standardised residuals.
            pre = fit_fold_mcem(MersenneTwister(seed + 1000 * r), A0, b0, Arows, y,
                                sigma2, grp, eb, fresh_hs(), spec_gauss_pre;
                                method = :collapsed)
            q_pre = vec(mean(pre.qsamps, dims = 2))
            resid_std = (y .- Arows * q_pre) ./ sqrt.(sigma2)

            # Step 2: method-of-moments start, then MCEM shape refinement.  The
            # MCEM shape update alone cannot escape alpha = 0, so the moment
            # start is what makes the skew-normal shape estimable at all.
            nu_use, alpha_use = moment_shape_init(fit_model, resid_std)
            for it in 1:REV_SHAPE_ITERS
                spec_it = rev_spec(fit_model, pathway; nu_init = nu_use,
                                   alpha_init = alpha_use, estimate_shape = true,
                                   skew_location = skew_location)
                f_it = fit_fold_mcem(MersenneTwister(seed + 1000 * r + 13 * it), A0, b0,
                                     Arows, y, sigma2, grp, eb, fresh_hs(), spec_it;
                                     method = :collapsed)
                nu_use, alpha_use = f_it.ν, f_it.α
            end
        end

        # Final fit with the shape held fixed, so oracle and estimated cells
        # differ only in the value of the shape parameter.
        spec_final = rev_spec(fit_model, pathway; nu_init = nu_use,
                              alpha_init = alpha_use, estimate_shape = false,
                              skew_location = skew_location)
        fit = fit_fold_mcem(MersenneTwister(seed + 1000 * r + 7), A0, b0, Arows, y,
                            sigma2, grp, eb, fresh_hs(), spec_final; method = :collapsed)
        nu_hat[r] = fit_model == :t ? nu_use : NaN
        alpha_hat[r] = fit_model == :skew ? alpha_use : NaN

        qhat = vec(mean(fit.qsamps, dims = 2))
        err = qhat .- q_true
        err_rep[:, r] .= err
        err2_rep[:, r] .= err .^ 2
        c_CI, _ = compute_coverage_matrices(fit.qsamps, q_true; levels = levels)
        w_CI, _ = compute_parameter_intervals(fit.qsamps; levels = levels)
        cov_rep[:, :, r] .= c_CI
        width_rep[:, :, r] .= w_CI
    end

    if _pb_should_parallel(n_reps, "PB_PARALLEL_SWEEPS", nothing)
        Threads.@threads :dynamic for r in 1:n_reps
            run_rep!(r)
        end
    else
        for r in 1:n_reps
            run_rep!(r)
        end
    end

    # A replicate is dropped if its sampler produced any non-finite summary.
    ok = [all(isfinite, view(err_rep, :, r)) && all(isfinite, view(cov_rep, :, :, r)) &&
          all(isfinite, view(width_rep, :, :, r)) for r in 1:n_reps]
    n_ok = count(ok)
    n_ok > 0 || error("revision_recovery: all $n_reps replicates failed")
    n_ok < n_reps && @warn "revision_recovery: dropped $(n_reps - n_ok) of $n_reps replicates with non-finite summaries" fit_model data_model nu_data alpha_data

    bias = vec(sum(view(err_rep, :, ok), dims = 2)) ./ n_ok
    rmse = sqrt.(vec(sum(view(err2_rep, :, ok), dims = 2)) ./ n_ok)
    coverage = dropdims(sum(view(cov_rep, :, :, ok), dims = 3), dims = 3) ./ n_ok
    width = dropdims(sum(view(width_rep, :, :, ok), dims = 3), dims = 3) ./ n_ok
    nu_ok = filter(isfinite, nu_hat[ok])
    al_ok = filter(isfinite, alpha_hat[ok])

    return (
        coverage_by_level = vec(mean(coverage, dims = 1)),
        width_by_level = vec(mean(width, dims = 1)),
        mean_abs_bias = mean(abs.(bias)),
        mean_rmse = mean(rmse),
        nu_hat = isempty(nu_ok) ? NaN : mean(nu_ok),
        nu_hat_sd = length(nu_ok) > 1 ? std(nu_ok) : NaN,
        alpha_hat = isempty(al_ok) ? NaN : mean(al_ok),
        alpha_hat_sd = length(al_ok) > 1 ? std(al_ok) : NaN,
        levels = levels, d = d, n_reps = n_ok,
    )
end

function run_pathway(pathway::Symbol)
    posterior_file = joinpath(dirname(@__DIR__), "output", string(pathway), "data",
                              "posterior_$(pathway).jld2")
    isfile(posterior_file) || error("posterior file not found: $posterior_file")
    results_dir = joinpath(dirname(@__DIR__), "output", string(pathway), "results", "revision")
    mkpath(results_dir)

    grid_rows = DataFrame(pathway = String[], design = String[], records = Int[],
                          setting = String[], dgp = String[],
                          sweep_param = String[], sweep_val = Float64[],
                          fit = String[], shape_known = Bool[],
                          cov95 = Float64[], width95 = Float64[],
                          mean_abs_bias = Float64[], mean_rmse = Float64[],
                          nu_hat = Float64[], nu_hat_sd = Float64[],
                          alpha_hat = Float64[], alpha_hat_sd = Float64[],
                          n_reps = Int[])
    level_rows = DataFrame(pathway = String[], design = String[], setting = String[],
                           dgp = String[], sweep_param = String[], sweep_val = Float64[],
                           fit = String[], nominal = Float64[], empirical = Float64[],
                           width = Float64[])

    idx95 = findfirst(x -> isapprox(x, 0.95; atol = 1e-8), REV_LEVELS)

    function record!(design, nrec, setting, dgp, sweep_param, val, label, known, res)
        push!(grid_rows, (string(pathway), string(design), nrec, string(setting),
                          string(dgp), string(sweep_param), val, label, known,
                          res.coverage_by_level[idx95], res.width_by_level[idx95],
                          res.mean_abs_bias, res.mean_rmse,
                          res.nu_hat, res.nu_hat_sd, res.alpha_hat, res.alpha_hat_sd,
                          res.n_reps))
        for (k, lv) in enumerate(REV_LEVELS)
            push!(level_rows, (string(pathway), string(design), string(setting),
                               string(dgp), string(sweep_param), val, label,
                               lv, res.coverage_by_level[k], res.width_by_level[k]))
        end
        @printf("      %-12s cov95=%.3f  width95=%.2f  rmse=%.3f  nu_hat=%.2f  alpha_hat=%.2f\n",
                label, res.coverage_by_level[idx95], res.width_by_level[idx95],
                res.mean_rmse, res.nu_hat, res.alpha_hat)
        flush(stdout)
    end

    for design in REV_DESIGNS, setting in REV_SETTINGS
        nrec = design == :record ? REV_RECORDS : 1
        # The skew-normal generator in the `scale` setting keeps the canonical
        # location at zero, which is the parameterisation used by the originally
        # submitted alpha sweep; the fitted skew-normal model uses the matching
        # location convention.
        skew_loc = setting == :scale ? :location : :mean

        println("\n[$(pathway) | $(design) | $(setting)] Student-t generator")
        for nu in NU_VALUES
            @printf("  nu_data = %.2f\n", nu); flush(stdout)
            for (label, known, fit_model, est, shape) in [
                    ("gaussian", true,  :gaussian, false, 50.0),
                    ("t_oracle", true,  :t,        false, max(nu, 2.05)),
                    ("t_est",    false, :t,        true,  8.0),
                    ("skew_est", false, :skew,     true,  0.0)]
                res = revision_recovery(posterior_file; pathway = pathway,
                    n_reps = REV_NREPS, seed = REV_SEED,
                    data_model = :t, nu_data = nu, alpha_data = 0.0,
                    setting = setting, fit_model = fit_model,
                    estimate_shape = est, shape_fixed = shape,
                    skew_location = skew_loc, records_per_quantity = nrec)
                record!(design, nrec, setting, :t, :nu, nu, label, known, res)
            end
        end

        println("\n[$(pathway) | $(design) | $(setting)] Skew-normal generator")
        for al in ALPHA_VALUES
            @printf("  alpha_data = %.2f\n", al); flush(stdout)
            for (label, known, fit_model, est, shape) in [
                    ("gaussian",    true,  :gaussian, false, 50.0),
                    ("skew_oracle", true,  :skew,     false, al),
                    ("skew_est",    false, :skew,     true,  0.0),
                    ("t_est",       false, :t,        true,  8.0)]
                res = revision_recovery(posterior_file; pathway = pathway,
                    n_reps = REV_NREPS, seed = REV_SEED,
                    data_model = :skew, nu_data = 50.0, alpha_data = al,
                    setting = setting, fit_model = fit_model,
                    estimate_shape = est, shape_fixed = shape,
                    skew_location = skew_loc, records_per_quantity = nrec)
                record!(design, nrec, setting, :skew, :alpha, al, label, known, res)
            end
        end
    end

    CSV.write(joinpath(results_dir, "misspec_grid_$(pathway)$(REV_SUFFIX).csv"), grid_rows)
    CSV.write(joinpath(results_dir, "misspec_levels_$(pathway)$(REV_SUFFIX).csv"), level_rows)
    println("  wrote $(joinpath(results_dir, "misspec_grid_$(pathway)$(REV_SUFFIX).csv"))")
    return nothing
end

println("Revision misspecification study")
println("  threads    = $(Threads.nthreads())")
println("  replicates = $REV_NREPS")
println("  pathways   = $(REV_PATHWAYS)")
println("  settings   = $(REV_SETTINGS)")
println("  designs    = $(REV_DESIGNS) (records per quantity = $REV_RECORDS)")
println("  nu values  = $(NU_VALUES)")
println("  alpha values = $(ALPHA_VALUES)")
println("  started    = $(now())")
flush(stdout)

for pw in REV_PATHWAYS
    t0 = time()
    run_pathway(pw)
    @printf("[%s] elapsed %.1f min\n", pw, (time() - t0) / 60)
    flush(stdout)
end

println("Done at $(now())")
