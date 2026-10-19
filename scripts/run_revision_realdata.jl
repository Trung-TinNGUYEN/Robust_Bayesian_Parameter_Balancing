# run_revision_realdata.jl
#
# Real-data re-analysis added for the revision.
#
# Reviewer requests addressed here:
#   * "The source-level random-intercept model assumes that all quantities
#      reported by a source share the same additive shift on the transformed
#      scale.  This appears difficult to justify for heterogeneous quantities
#      such as K_M and k_cat."
#        -> a source-by-quantity-type random intercept is fitted and compared
#           with the pooled source intercept of the original submission.
#   * "Knowledge about the degrees of freedom ... should be removed."
#        -> the Student-t degrees of freedom and the skew-normal shape are
#           initialised from the method of moments on training-fold residuals
#           and then refined by MCEM, instead of being fixed a priori.
#
# Two facts about the curated data motivate these changes and are recorded here
# as diagnostics:
#   1. BRENDA reports no per-record standard errors, so every record carries the
#      same assumed log-scale standard deviation sigma = 0.5 log 10.  The
#      standardised residuals of a Gaussian fit have standard deviation well
#      above one, so this convention understates the real dispersion.
#   2. The residual skewness has opposite signs for k_cat and K_M records, so a
#      single pooled skewness parameter averages to approximately zero.  This is
#      why the skew-normal model of the original submission collapsed onto the
#      Gaussian model, and why the shape must be stratified by quantity type.
#
# Folds are always organism-level leave-one-source-out: every record of the
# held-out organism is excluded from training, so a held-out source is genuinely
# unseen and its random effect is drawn from the fitted between-source
# distribution.  The random-effect grouping is a separate vector, which is what
# allows a source-by-type effect to be evaluated under organism-level folds.
#
# Outputs, written under output/<pathway>/results/revision/:
#   realdata_summary_<pathway>.csv    one row per model cell
#   realdata_levels_<pathway>.csv     predictive coverage at every nominal level
#   realdata_residuals_<pathway>.csv  residual diagnostics by quantity type
#
# Environment controls:
#   PB_REV_PATHWAYS  comma-separated subset of {glycolysis,tca} (default both)
#   PB_REV_CELLS     comma-separated subset of the model-cell labels below

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
const REV_PATHWAYS = Symbol.(strip.(split(get(ENV, "PB_REV_PATHWAYS", "glycolysis,tca"), ",")))
# Output suffix, so that a subset run does not overwrite the full tables.
const REV_SUFFIX = get(ENV, "PB_REV_SUFFIX", "")
# Multiplier of the number of outer EM iterations of every cell (default 1, the
# published setting), for the EM-iteration sensitivity run (S1 Text).
const OUTER_MULT = parse(Int, get(ENV, "PB_REV_OUTER_MULT", "1"))
OUTER_MULT == 1 || !isempty(REV_SUFFIX) || error("PB_REV_OUTER_MULT is not 1: set PB_REV_SUFFIX as well, so that the published outputs are kept")

_sample_skewness(x) = (m = mean(x); s = std(x, corrected = false);
                       s <= 0 ? 0.0 : mean(((x .- m) ./ s) .^ 3))
_sample_exkurtosis(x) = (m = mean(x); s = std(x, corrected = false);
                         s <= 0 ? 0.0 : mean(((x .- m) ./ s) .^ 4) - 3.0)

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
    delta = clamp(0.5 * (lo + hi), 0.0, 0.999)
    return s * delta / sqrt(1.0 - delta^2)
end

nu_from_kurtosis(k::Float64) = k <= 0.05 ? 50.0 : clamp(4.0 + 6.0 / k, 2.5, 50.0)

"""
    ridge_qhat(Arows, y, sigma2, q0, C0) -> qhat

Closed-form Gaussian parameter-balancing solution.  Used only to obtain
training-fold residuals for the method-of-moments shape initialisation, so it
adds negligible cost and uses no held-out information.
"""
function ridge_qhat(Arows::AbstractMatrix, y::Vector{Float64}, sigma2::Vector{Float64},
                    q0::Vector{Float64}, C0::AbstractMatrix)
    C0inv = inv(Symmetric(Matrix{Float64}(C0)))
    P = C0inv + Arows' * (Arows ./ sigma2)
    rhs = Arows' * (y ./ sigma2) + C0inv * q0
    return Symmetric(P) \ rhs
end

"""
    loso_by_key(rng, A0, b0, Arows, y, sigma2, re_group, fold_key, eb, make_spec_fn; ...)

Leave-one-source-out cross-validation in which the fold partition (`fold_key`)
and the random-effect grouping (`re_group`) are supplied separately.  This is
what permits a source-by-quantity-type random intercept to be evaluated under
organism-level folds.  `make_spec_fn(train_idx)` returns the `FoldFitSpec` for a
fold and may depend on training-fold data only.
"""
function loso_by_key(rng::AbstractRNG, A0, b0, Arows::Matrix{Float64},
                     y::Vector{Float64}, sigma2::Vector{Float64},
                     re_group::Vector{String}, fold_key::Vector{String},
                     eb::EBSpec, make_spec_fn::Function;
                     cover_levels::Vector{Float64} = REV_LEVELS,
                     method::Symbol = :collapsed)
    folds = sort!(unique(fold_key))
    n_folds = length(folds)
    n = length(y)
    K = length(cover_levels)

    crps = Vector{Float64}(undef, n)
    wis = Vector{Float64}(undef, n)
    loglik = Vector{Float64}(undef, n)
    pit = Vector{Float64}(undef, n)
    pred_err = Vector{Float64}(undef, n)
    covered = Matrix{Bool}(undef, n, K)
    widths = Matrix{Float64}(undef, n, K)
    fold_tau2 = Vector{Float64}(undef, n_folds)
    fold_nu = Vector{Float64}(undef, n_folds)
    fold_alpha = Vector{Float64}(undef, n_folds)

    seeds = _pb_thread_seeds(rng, n_folds)

    function run_fold!(f::Int, rng_f::AbstractRNG)
        test_idx = findall(==(folds[f]), fold_key)
        train_idx = findall(!=(folds[f]), fold_key)
        spec = make_spec_fn(train_idx)
        hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
        fit = fit_fold_mcem(rng_f, A0, b0, Arows[train_idx, :], y[train_idx],
                            sigma2[train_idx], re_group[train_idx], eb, hs, spec;
                            method = method)
        fold_tau2[f] = fit.hs.τ²
        fold_nu[f] = fit.ν
        fold_alpha[f] = fit.α
        for i in test_idx
            p = predict_row_metrics_mc_hier(rng_f, view(Arows, i, :), y[i], sigma2[i],
                                            re_group[i], fit, spec;
                                            cover_levels = cover_levels)
            crps[i] = p.crps; wis[i] = p.wis; loglik[i] = p.loglik; pit[i] = p.pit
            pred_err[i] = p.pred_error
            covered[i, :] = p.covered
            widths[i, :] = p.widths
        end
    end

    if _pb_should_parallel(n_folds, "PB_PARALLEL_CV", nothing)
        Threads.@threads :dynamic for f in 1:n_folds
            run_fold!(f, _pb_rng_from_seed(seeds[f]))
        end
    else
        for f in 1:n_folds
            run_fold!(f, _pb_rng_from_seed(seeds[f]))
        end
    end

    return (crps = crps, wis = wis, loglik = loglik, pit = pit, pred_error = pred_err,
            covered = covered, widths = widths, n_folds = n_folds,
            tau2 = mean(filter(isfinite, fold_tau2)),
            nu = mean(filter(isfinite, fold_nu)),
            alpha = mean(filter(isfinite, fold_alpha)))
end

"""
    cell_spec(model, hierarchical, pathway; shape_mode)

Model specification for one cell.  `shape_mode = :fixed` reproduces the original
submission (nu = 50 for the Student-t, alpha initialised at zero); `:moment`
initialises the shape from training-fold residual moments before the MCEM
refinement, which is what makes the skew-normal shape identifiable.
"""
function cell_spec(model::Symbol, hierarchical::Bool, pathway::Symbol;
                   nu_init::Float64 = 50.0, alpha_init::Float64 = 0.0,
                   estimate::Bool = false)
    is_tca = pathway == :tca
    n_out = OUTER_MULT * (model == :gaussian ? (is_tca ? 4 : 3) : (is_tca ? 5 : 4))
    n_mc = is_tca ? 250 : 150
    burn = is_tca ? 80 : 40
    return FoldFitSpec(
        model = model, hierarchical = hierarchical,
        n_outer = n_out, n_mcmc = n_mc, burnin = burn,
        thin = 1, n_keep_min = 60, n_pred_draws = 400, b_draws_ll = 50,
        posterior_refresh = 1, adaptive_stop = true, rtol_outer = 1e-3, min_outer = 2,
        tau_update = :ig, tau_pc_rate = 0.0,
        ν_init = model == :t ? max(nu_init, 2.05) : 50.0,
        α_init = model == :skew ? alpha_init : 0.0,
        update_nu = estimate && model == :t,
        update_alpha = estimate && model == :skew,
        refine_shape = estimate, shape_opt_tol = 1e-3,
        skew_location = :mean,
        nu_penalty = 0.0, nu_target = max(nu_init, 2.05), alpha_penalty = 0.0)
end

function residual_diagnostics(Arows, y, sigma2, q0, C0, grp, kinds)
    qhat = ridge_qhat(Arows, y, sigma2, q0, C0)
    r = (y .- Arows * qhat) ./ sqrt.(sigma2)
    rows = DataFrame(subset = String[], n = Int[], mean = Float64[], sd = Float64[],
                     skewness = Float64[], excess_kurtosis = Float64[],
                     frac_abs_gt3 = Float64[], moment_nu = Float64[],
                     moment_alpha = Float64[])
    function add!(label, idx)
        v = r[idx]
        push!(rows, (label, length(v), mean(v), std(v), _sample_skewness(v),
                     _sample_exkurtosis(v), mean(abs.(v) .> 3),
                     nu_from_kurtosis(_sample_exkurtosis(v)),
                     alpha_from_skewness(_sample_skewness(v))))
    end
    add!("all", 1:length(r))
    for k in sort(unique(kinds))
        add!(k, findall(==(k), kinds))
    end
    src = unique(grp)
    mus = [mean(r[grp .== s]) for s in src]
    ns = [count(==(s), grp) for s in src]
    push!(rows, ("between_source_means", length(mus), mean(mus), std(mus),
                 _sample_skewness(mus), _sample_exkurtosis(mus), NaN, NaN, NaN))
    keep = ns .>= 5
    push!(rows, ("between_source_means_n_ge_5", count(keep), mean(mus[keep]),
                 std(mus[keep]), _sample_skewness(mus[keep]),
                 _sample_exkurtosis(mus[keep]), NaN, NaN, NaN))
    return rows, r
end

function run_pathway(pathway::Symbol)
    root = joinpath(dirname(@__DIR__), "output", string(pathway))
    brenda = load(joinpath(root, "data", "brenda_inputs_$(pathway).jld2"))
    post = load(joinpath(root, "data", "posterior_$(pathway).jld2"))
    results_dir = joinpath(root, "results", "revision")
    mkpath(results_dir)

    y = Vector{Float64}(brenda["y"])
    sigma2 = Vector{Float64}(brenda["σ2"])
    Arows = Matrix{Float64}(brenda["Arows"])
    org = Vector{String}(brenda["grp"])
    meta = brenda["meta"]
    kinds = Vector{String}(getproperty(meta, :kind))
    q0 = vec(Array(post["q_prior"]))
    C0 = Matrix(Array(post["C_prior"]))
    eb = EBSpec(q0 = q0, C0 = C0)
    d = size(Arows, 2)

    # `A0`/`b0` are zeroed when the design contains identity-like rows, matching
    # the deduplication rule used by the original driver.
    A0, b0 = _prepare_information_pair(Arows, d,
                                       Matrix{Float64}(brenda["A0"]),
                                       Vector{Float64}(brenda["b0"]);
                                       deduplicate_thermo_block = true)

    org_type = [string(org[i], "|", kinds[i]) for i in eachindex(org)]

    println("\n=== $(pathway): N=$(length(y)) d=$d organisms=$(length(unique(org))) " *
            "organism-by-type cells=$(length(unique(org_type)))")
    diag_rows, _ = residual_diagnostics(Arows, y, sigma2, q0, C0, org, kinds)
    CSV.write(joinpath(results_dir, "realdata_residuals_$(pathway)$(REV_SUFFIX).csv"), diag_rows)
    show(stdout, diag_rows; allrows = true, allcols = true); println()

    # Model cells.  `re` selects the random-effect grouping; folds are always
    # organism-level.
    cells = [
        ("gaussian_Fix",        :gaussian, false, :none,     :fixed),
        ("gaussian_RE_org",     :gaussian, true,  :org,      :fixed),
        ("gaussian_RE_orgtype", :gaussian, true,  :org_type, :fixed),
        ("t_fixed50_RE_org",    :t,        true,  :org,      :fixed),
        ("t_moment_Fix",        :t,        false, :none,     :moment),
        ("t_moment_RE_org",     :t,        true,  :org,      :moment),
        ("t_moment_RE_orgtype", :t,        true,  :org_type, :moment),
        ("skew_moment_RE_orgtype", :skew,  true,  :org_type, :moment),
    ]
    wanted = get(ENV, "PB_REV_CELLS", "")
    if !isempty(wanted)
        keepset = Set(strip.(split(wanted, ",")))
        cells = [c for c in cells if c[1] in keepset]
    end

    summary = DataFrame(pathway = String[], cell = String[], model = String[],
                        re_grouping = String[], shape_mode = String[],
                        CRPS = Float64[], WIS = Float64[], LPD = Float64[],
                        Cov95 = Float64[], Width95 = Float64[],
                        PredRMSE = Float64[], PredBias = Float64[],
                        tau2 = Float64[], nu = Float64[], alpha = Float64[],
                        n_folds = Int[])
    levels_df = DataFrame(pathway = String[], cell = String[], nominal = Float64[],
                          empirical = Float64[], width = Float64[])
    idx95 = findfirst(x -> isapprox(x, 0.95; atol = 1e-8), REV_LEVELS)
    percell = Dict{String,NamedTuple}()

    for (label, model, hier, re_kind, shape_mode) in cells
        re_group = re_kind == :org_type ? org_type : org
        println("  -> $label"); flush(stdout)
        t0 = time()

        make_spec_fn = function (train_idx)
            if shape_mode == :moment
                qh = ridge_qhat(view(Arows, train_idx, :), y[train_idx],
                                sigma2[train_idx], q0, C0)
                r = (y[train_idx] .- Arows[train_idx, :] * qh) ./ sqrt.(sigma2[train_idx])
                nu0 = nu_from_kurtosis(_sample_exkurtosis(r))
                al0 = alpha_from_skewness(_sample_skewness(r))
                return cell_spec(model, hier, pathway; nu_init = nu0,
                                 alpha_init = al0, estimate = true)
            else
                return cell_spec(model, hier, pathway; nu_init = 50.0,
                                 alpha_init = 0.0, estimate = false)
            end
        end

        res = loso_by_key(MersenneTwister(REV_SEED), A0, b0, Arows, y, sigma2,
                          re_group, org, eb, make_spec_fn)

        # Per-record output is retained so that uncertainty in the summary
        # statistics can be assessed by resampling whole organisms, which is the
        # correct unit given that records are clustered within organism.
        percell[label] = (covered95 = Vector{Bool}(res.covered[:, idx95]),
                          crps = res.crps, wis = res.wis, loglik = res.loglik,
                          pred_error = res.pred_error, org = org)

        cov95 = mean(res.covered[:, idx95])
        w95 = mean(res.widths[:, idx95])
        push!(summary, (string(pathway), label, string(model),
                        re_kind == :none ? "none" : string(re_kind), string(shape_mode),
                        mean(res.crps), mean(res.wis), mean(res.loglik),
                        cov95, w95,
                        sqrt(mean(res.pred_error .^ 2)), mean(res.pred_error),
                        res.tau2, res.nu, res.alpha, res.n_folds))
        for (k, lv) in enumerate(REV_LEVELS)
            push!(levels_df, (string(pathway), label, lv,
                              mean(res.covered[:, k]), mean(res.widths[:, k])))
        end
        @printf("     CRPS=%.4f  Cov95=%.4f  Width95=%.3f  PredRMSE=%.4f  tau2=%.3f  nu=%.2f  alpha=%.3f  [%.1f min]\n",
                mean(res.crps), cov95, w95, sqrt(mean(res.pred_error .^ 2)),
                res.tau2, res.nu, res.alpha, (time() - t0) / 60)
        flush(stdout)
    end

    CSV.write(joinpath(results_dir, "realdata_summary_$(pathway)$(REV_SUFFIX).csv"), summary)
    CSV.write(joinpath(results_dir, "realdata_levels_$(pathway)$(REV_SUFFIX).csv"), levels_df)
    JLD2.jldsave(joinpath(results_dir, "realdata_percell_$(pathway)$(REV_SUFFIX).jld2");
                 percell = percell, org = org)
    println("\n  $(pathway) summary")
    show(stdout, summary; allrows = true, allcols = true); println()
    return nothing
end

println("Revision real-data re-analysis")
println("  threads  = $(Threads.nthreads())")
println("  pathways = $(REV_PATHWAYS)")
println("  started  = $(now())")
flush(stdout)

for pw in REV_PATHWAYS
    t0 = time()
    run_pathway(pw)
    @printf("[%s] elapsed %.1f min\n", pw, (time() - t0) / 60)
    flush(stdout)
end
println("Done at $(now())")
