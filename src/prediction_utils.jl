using Random
using Distributions
using Statistics
using DataFrames

# Prediction functions

"""
    logmeanexp(v)

Compute the log of the mean of exponentials of `v` in a numerically stable way.

Uses the identity log(mean(exp.(v))) = max(v) + log(mean(exp.(v .- max(v)))),
which avoids overflow or underflow when entries of `v` differ by large amounts.

# Arguments
- `v`: Vector of log-domain values.

# Returns
A scalar `Float64`.
"""
@inline function logmeanexp(v::Vector{Float64})
    m = maximum(v)
    return m + log(mean(exp.(v .- m)))
end

"""
    logpdf_t_locscale(y, μ, s, ν)

Log-pdf of the Student-t distribution with location `μ`, scale `s`, and degrees of freedom `ν`.

Evaluates the standardized TDist log-pdf at `(y - μ) / s` and subtracts `log(s)` for the
Jacobian correction.

# Arguments
- `y`: Observed value.
- `μ`: Location parameter.
- `s`: Scale parameter (must be positive).
- `ν`: Degrees of freedom.

# Returns
A scalar `Float64` log-density.
"""
@inline function logpdf_t_locscale(y::Float64, μ::Float64, s::Float64, ν::Float64)
    return logpdf(TDist(ν), (y - μ)/s) - log(s)
end

"""
    sample_skewnormal_meanvar(rng, μ, σ2, α)

Draw one sample from a skew-normal distribution reparametrized to have mean `μ` and
variance `σ2` at shape parameter `α`.

Internally converts `(μ, σ2, α)` to the canonical location `ξ` and scale `ω` using
`skew_omega2_from_meas`, then generates a draw via the stochastic-representation
``z = δ |u| + \\sqrt{1 - δ^2} v`` where ``u, v \\sim N(0,1)``.

# Arguments
- `rng`: Random number generator.
- `μ`: Target mean.
- `σ2`: Target variance.
- `α`: Skewness shape parameter.

# Returns
A 3-tuple `(y, ξ, ω)`: the sampled value, the canonical location, and the canonical scale
used to generate it.
"""
function sample_skewnormal_meanvar(rng::AbstractRNG, μ::Float64, σ2::Float64, α::Float64)
    ω2 = skew_omega2_from_meas(σ2, α)
    ω = sqrt(ω2)
    δ = α / sqrt(1 + α^2)
    ξ = μ - ω * δ * sqrt(2/pi)
    u = randn(rng)
    v = randn(rng)
    z = δ * abs(u) + sqrt(1 - δ^2) * v
    return ξ + ω * z, ξ, ω
end

"""
    sample_skewnormal_locationvar(rng, ξ, σ2, α)

Draw one sample from a skew-normal distribution whose canonical location is
`ξ` and whose marginal variance is `σ2` at shape `α`.
"""
function sample_skewnormal_locationvar(rng::AbstractRNG, ξ::Float64, σ2::Float64, α::Float64)
    ω2 = skew_omega2_from_meas(σ2, α)
    ω = sqrt(ω2)
    δ = α / sqrt(1 + α^2)
    u = randn(rng)
    v = randn(rng)
    z = δ * abs(u) + sqrt(1 - δ^2) * v
    return ξ + ω * z, ξ, ω
end

"""
    predict_row_metrics_mc_hier(rng, arow, y_obs, σ2_obs, grp_obs, fit, spec; cover_levels)

Compute predictive calibration metrics for a single held-out observation using Monte Carlo
draws from the posterior.

Samples `spec.n_pred_draws` columns from `fit.qsamps`, adds a random-effects draw when
`spec.hierarchical` is true (using the group's posterior samples if the group is in
`fit.grp_levels`, otherwise drawing from the estimated random-effects variance), then
generates a predictive draw and log-density according to `spec.model` (`:gaussian`, `:t`,
or `:skew`).

# Arguments
- `rng`: Random number generator.
- `arow`: Design-matrix row for the held-out observation.
- `y_obs`: Held-out response value.
- `σ2_obs`: Known observational variance for the held-out observation.
- `grp_obs`: Group identifier string for the held-out observation.
- `fit`: Fitted fold object containing posterior samples.
- `spec`: Specification controlling model family, hierarchy, and number of draws.

# Keywords
- `cover_levels`: Nominal coverage levels for interval scoring (default `[0.5, 0.8, 0.9, 0.95]`).
- `ν_obs`, `α_obs`: Optional record-specific skew-t shape parameters for the
  held-out observation, overriding `fit.ν` and `fit.α`. Used when the fit was
  produced with record-indexed shapes (per quantity type), so the predictive
  distribution of a held-out record carries the shape of its own quantity type.
  Only honoured for `spec.model == :skewt`.

# Returns
A named tuple `(pit, crps, wis, loglik, covered, widths, y_samples)`.
"""
function predict_row_metrics_mc_hier(
    rng::AbstractRNG,
    arow::AbstractVector{<:Real},
    y_obs::Float64,
    σ2_obs::Float64,
    grp_obs::String,
    fit::FoldFit,
    spec::FoldFitSpec;
    cover_levels::Vector{Float64} = [0.5,0.8,0.9,0.95],
    ν_obs::Union{Nothing,Float64} = nothing,
    α_obs::Union{Nothing,Float64} = nothing,
    ν_draws::Union{Nothing,AbstractVector{Float64}} = nothing,
    α_draws::Union{Nothing,AbstractVector{Float64}} = nothing
)
    S = size(fit.qsamps, 2)
    mdraw = spec.n_pred_draws
    idxs = rand(rng, 1:S, mdraw)

    y_samples = Vector{Float64}(undef, mdraw)
    logpdfs = Vector{Float64}(undef, mdraw)

    use_b = (fit.bsamps !== nothing) && spec.hierarchical
    gmap = Dict{String,Int}()
    if use_b
        for (i,g) in enumerate(fit.grp_levels)
            gmap[g] = i
        end
    end
    is_known_group = use_b && haskey(gmap, grp_obs)
    gidx = is_known_group ? gmap[grp_obs] : 0

    for j in 1:mdraw
        s = idxs[j]
        q = view(fit.qsamps, :, s)

        μ = dot(arow, q)
        bdraw = 0.0

        if spec.hierarchical
            if is_known_group
                bdraw = fit.bsamps[gidx, s]
            else
                bdraw = randn(rng) * sqrt(max(fit.hs.τ², 0.0))
            end
        end
        μ += bdraw

        if spec.model == :gaussian
            y_samples[j] = μ + randn(rng) * sqrt(σ2_obs)
            logpdfs[j] = logpdf(Normal(μ, sqrt(σ2_obs)), y_obs)
        elseif spec.model == :t
            ν = fit.ν
            s2 = t_scale2_from_meas(σ2_obs, ν)
            sscale = sqrt(s2)
            y_samples[j] = μ + rand(rng, TDist(ν)) * sscale
            logpdfs[j] = logpdf_t_locscale(y_obs, μ, sscale, ν)
        elseif spec.model == :skew
            α = fit.α
            if spec.skew_location == :mean
                y_samples[j], ξ, ω = sample_skewnormal_meanvar(rng, μ, σ2_obs, α)
            elseif spec.skew_location == :location
                y_samples[j], ξ, ω = sample_skewnormal_locationvar(rng, μ, σ2_obs, α)
            else
                error("skew_location must be :mean or :location")
            end
            logpdfs[j] = logpdf_skewnormal(y_obs, ξ, ω, α)
        elseif spec.model == :skewt
            # Per-draw shapes (fully Bayesian shape estimation) take precedence,
            # aligned with the posterior draw `s`; then a record-specific
            # override; then the shared fitted shape.
            ν = ν_draws !== nothing ? ν_draws[s] : (ν_obs === nothing ? fit.ν : ν_obs)
            α = α_draws !== nothing ? α_draws[s] : (α_obs === nothing ? fit.α : α_obs)
            y_samples[j], ξ, ω = sample_skewt_meanvar(rng, μ, σ2_obs, ν, α)
            logpdfs[j] = logpdf_skewt(y_obs, ξ, ω, ν, α)
        else
            error("Unknown model: $(spec.model)")
        end
    end

    ll = logmeanexp(logpdfs)

    pit = pit_from_samples(rng, y_samples, y_obs)
    crps = crps_from_samples(y_samples, y_obs)
    wis  = wis_from_samples(y_samples, y_obs; cover_levels=cover_levels)
    covered, widths = coverage_widths_from_samples(y_samples, y_obs; cover_levels=cover_levels)
    pred_mean = mean(y_samples)
    pred_sd = std(y_samples)

    return (
        pit=pit,
        crps=crps,
        wis=wis,
        loglik=ll,
        covered=covered,
        widths=widths,
        pred_mean=pred_mean,
        pred_sd=pred_sd,
        pred_error=y_obs - pred_mean,
        y_samples=y_samples
    )
end

# Calibration metrics

"""
    crps_from_samples(x, y)

Compute the Continuous Ranked Probability Score (CRPS) for a single observation `y` given
sorted predictive samples `x`, using the energy-score formula.

The energy-score identity used is CRPS = E|X - y| - 0.5 E|X - X'|, computed in O(m log m)
time after sorting.

# Arguments
- `x`: Vector of predictive samples (need not be pre-sorted).
- `y`: Scalar observation.

# Returns
A non-negative `Float64` CRPS value; lower is better.
"""
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

"""
    wis_from_samples(x, y; cover_levels)

Compute the Weighted Interval Score (WIS) for a single observation `y` from predictive
samples `x` at multiple nominal coverage levels.

The WIS combines the absolute median error with interval scores at each level in
`cover_levels`, weighted by the corresponding alpha/2 values and normalized by
`length(cover_levels) + 0.5`.

# Arguments
- `x`: Vector of predictive samples.
- `y`: Scalar observation.

# Keywords
- `cover_levels`: Nominal interval levels (default `[0.5, 0.8, 0.9, 0.95]`).

# Returns
A non-negative `Float64` WIS value; lower is better.
"""
function wis_from_samples(x::Vector{Float64}, y::Float64; cover_levels=[0.5,0.8,0.9,0.95])
    xs = sort(x)
    m = length(xs)

    med = xs[clamp(Int(ceil(0.5 * m)), 1, m)]
    abs_med = abs(y - med)

    K = length(cover_levels)
    num = 0.5 * abs_med
    den = K + 0.5

    for p in cover_levels
        α = 1.0 - p
        loq = α / 2
        hiq = 1.0 - α / 2
        lo = xs[clamp(Int(ceil(loq * m)), 1, m)]
        hi = xs[clamp(Int(ceil(hiq * m)), 1, m)]

        iscore = (hi - lo)
        if y < lo
            iscore += (2 / α) * (lo - y)
        elseif y > hi
            iscore += (2 / α) * (y - hi)
        end

        w = α / 2
        num += w * iscore
    end

    return num / den
end

"""
    pit_from_samples(rng, x, y)

Compute a randomized Probability Integral Transform (PIT) value for observation `y`
given predictive samples `x`.

Randomization is applied at the boundary cases (observation below all samples or above
all samples) and when `y` falls between two consecutive samples, following the standard
discrete-PIT construction that yields uniform PIT values under a correctly specified model.

# Arguments
- `rng`: Random number generator used for the randomization step.
- `x`: Vector of predictive samples (need not be pre-sorted).
- `y`: Scalar observation.

# Returns
A `Float64` in `(0, 1)`.
"""
function pit_from_samples(rng::AbstractRNG, x::Vector{Float64}, y::Float64)
    xs = sort(x)
    m = length(xs)
    j_below = searchsortedlast(xs, y)

    if j_below == 0
        return rand(rng) * (1.0 / m)
    elseif j_below == m
        return 1.0 - rand(rng) * (1.0 / m)
    else
        return (j_below + rand(rng)) / m
    end
end

"""
    coverage_widths_from_samples(x, y; cover_levels)

Return empirical interval coverage indicators and widths for each nominal level.

For each level `p` in `cover_levels`, extracts the equal-tailed empirical interval
`[lo, hi]` from the sorted samples and checks whether `y` falls inside.

# Arguments
- `x`: Vector of predictive samples.
- `y`: Scalar observation.

# Keywords
- `cover_levels`: Nominal coverage levels (default `[0.5, 0.8, 0.9, 0.95]`).

# Returns
A 2-tuple `(covered, widths)` where `covered` is a `Vector{Bool}` and `widths` is a
`Vector{Float64}`, both of length `length(cover_levels)`.
"""
function coverage_widths_from_samples(x::Vector{Float64}, y::Float64; cover_levels=[0.5,0.8,0.9,0.95])
    xs = sort(x)
    m = length(xs)
    K = length(cover_levels)

    covered = Vector{Bool}(undef, K)
    widths = Vector{Float64}(undef, K)

    for (k, p) in enumerate(cover_levels)
        α = 1.0 - p
        loq = α / 2
        hiq = 1.0 - α / 2

        lo_idx = clamp(Int(ceil(loq * m)), 1, m)
        hi_idx = clamp(Int(ceil(hiq * m)), 1, m)

        lo = xs[lo_idx]
        hi = xs[hi_idx]

        covered[k] = (lo <= y <= hi)
        widths[k] = hi - lo
    end

    return covered, widths
end

# Parameter recovery metrics

function _equal_tailed_interval(sorted_samples::AbstractVector{Float64}, level::Float64)
    S = length(sorted_samples)
    0.0 < level < 1.0 || error("coverage level must be in (0, 1), got $level")

    tail = 1.0 - level
    lo_idx = Int(ceil(tail / 2 * S))
    hi_idx = Int(ceil((1 - tail / 2) * S))

    lo_idx = clamp(lo_idx, 1, S)
    hi_idx = clamp(hi_idx, 1, S)

    return sorted_samples[lo_idx], sorted_samples[hi_idx]
end

function _empirical_hdi_interval(sorted_samples::AbstractVector{Float64}, level::Float64)
    S = length(sorted_samples)
    0.0 < level < 1.0 || error("coverage level must be in (0, 1), got $level")

    n_in = clamp(Int(ceil(level * S)), 1, S)
    n_in == S && return sorted_samples[1], sorted_samples[end]

    best_i = 1
    best_width = Inf
    last_start = S - n_in + 1
    @inbounds for i in 1:last_start
        j = i + n_in - 1
        width = sorted_samples[j] - sorted_samples[i]
        if width < best_width
            best_width = width
            best_i = i
        end
    end

    return sorted_samples[best_i], sorted_samples[best_i + n_in - 1]
end

"""
    compute_parameter_intervals(qsamps; levels)

Compute equal-tailed CI and empirical HDI widths for each latent parameter from
an MCMC sample matrix. The HDI is the shortest interval among sorted posterior
draws containing at least the requested empirical mass.

# Arguments
- `qsamps`: `d × S` matrix of posterior samples; rows are parameters, columns are draws.

# Keywords
- `levels`: Nominal coverage levels (default `[0.5, 0.8, 0.9, 0.95]`).

# Returns
A 2-tuple `(CI_mat, HDI_mat)`, each a `d × K` matrix of interval widths.
"""
function compute_parameter_intervals(qsamps::Matrix{Float64}; levels::Vector{Float64}=[0.5,0.8,0.9,0.95])
    d, S = size(qsamps)
    K = length(levels)

    CI_mat = Matrix{Float64}(undef, d, K)
    HDI_mat = Matrix{Float64}(undef, d, K)

    for i in 1:d
        qi = sort(qsamps[i, :])
        for (k, p) in enumerate(levels)
            ci_lo, ci_hi = _equal_tailed_interval(qi, p)
            hdi_lo, hdi_hi = _empirical_hdi_interval(qi, p)

            CI_mat[i, k] = ci_hi - ci_lo
            HDI_mat[i, k] = hdi_hi - hdi_lo
        end
    end

    return CI_mat, HDI_mat
end

"""
    compute_coverage_matrices(qsamps, q_true; levels)

Check whether each true parameter value falls within its posterior equal-tailed
interval and empirical HDI at each nominal level.

Returns indicator matrices (1.0 = covered, 0.0 = not covered) for both interval
definitions.

# Arguments
- `qsamps`: `d × S` matrix of posterior samples.
- `q_true`: Length-`d` vector of true parameter values.

# Keywords
- `levels`: Nominal coverage levels (default `[0.5, 0.8, 0.9, 0.95]`).

# Returns
A 2-tuple `(coverage_CI, coverage_HDI)`, each a `d × K` matrix of `Float64` indicators.
"""
function compute_coverage_matrices(qsamps::Matrix{Float64}, q_true::Vector{Float64}; levels::Vector{Float64}=[0.5,0.8,0.9,0.95])
    d, S = size(qsamps)
    K = length(levels)

    coverage_CI = Matrix{Float64}(undef, d, K)
    coverage_HDI = Matrix{Float64}(undef, d, K)

    for i in 1:d
        qi = sort(qsamps[i, :])
        true_val = q_true[i]

        for (k, p) in enumerate(levels)
            ci_lo, ci_hi = _equal_tailed_interval(qi, p)
            hdi_lo, hdi_hi = _empirical_hdi_interval(qi, p)

            coverage_CI[i, k] = (ci_lo <= true_val <= ci_hi) ? 1.0 : 0.0
            coverage_HDI[i, k] = (hdi_lo <= true_val <= hdi_hi) ? 1.0 : 0.0
        end
    end

    return coverage_CI, coverage_HDI
end

# Cross-validation helpers

"""
    leave_one_out_cv(rng, A0, b0, Arows, y, σ2, grp, eb, spec; cover_levels)

Leave-one-observation-out cross-validation loop.

For each observation `i`, fits the model on all remaining observations via
`fit_fold_mcem` and evaluates predictive metrics on observation `i` using
`predict_row_metrics_mc_hier`.

# Arguments
- `rng`: Random number generator.
- `A0`: Prior design matrix.
- `b0`: Prior mean vector.
- `Arows`: Full design matrix (`n × p`).
- `y`: Response vector (length `n`).
- `σ2`: Observational variance vector (length `n`).
- `grp`: Group label vector (length `n`).
- `eb`: Empirical Bayes hyperparameter specification.
- `spec`: Fold-fit specification (model family, hierarchy, draw count, etc.).

# Keywords
- `cover_levels`: Nominal coverage levels (default `[0.5, 0.8, 0.9, 0.95]`).

# Returns
A named tuple `(pit, crps, wis, loglik, pred_coverage, pred_widths)`, each entry an
`n`-length vector or `n × K` matrix.
"""
function leave_one_out_cv(
    rng::AbstractRNG,
    A0::AbstractMatrix,
    b0::AbstractVector,
    Arows::AbstractMatrix,
    y::Vector{Float64},
    σ2::Vector{Float64},
    grp::Vector{String},
    eb::EBSpec,
    spec::FoldFitSpec;
    cover_levels::Vector{Float64} = [0.5,0.8,0.9,0.95],
    method::Symbol = :collapsed,
    store_fits::Bool = false,
    parallel::Union{Nothing,Bool} = nothing
)
    n = length(y)

    pit_vec = Vector{Float64}(undef, n)
    crps_vec = Vector{Float64}(undef, n)
    wis_vec = Vector{Float64}(undef, n)
    loglik_vec = Vector{Float64}(undef, n)
    pred_mean_vec = Vector{Float64}(undef, n)
    pred_sd_vec = Vector{Float64}(undef, n)
    pred_error_vec = Vector{Float64}(undef, n)

    K = length(cover_levels)
    pred_coverage = Matrix{Bool}(undef, n, K)
    pred_widths = Matrix{Float64}(undef, n, K)

    fold_rows = Vector{NamedTuple}(undef, n)
    fold_fit_slots = Vector{Union{Nothing,FoldFit}}(undef, n)
    fill!(fold_fit_slots, nothing)

    run_parallel = _pb_should_parallel(n, "PB_PARALLEL_CV", parallel)
    fold_seeds = _pb_thread_seeds(rng, n)
    if run_parallel
        @info "leave_one_out_cv: running $n folds across $(Threads.nthreads()) Julia threads"
    end

    function _run_fold!(i::Int, rng_fold::AbstractRNG)
        train_idx = [j for j in 1:n if j != i]

        y_train = y[train_idx]
        σ2_train = σ2[train_idx]
        Arows_train = Arows[train_idx, :]
        grp_train = grp[train_idx]

        hs_init = init_hyper(eb; λ0=spec.λ_init, τ²0=spec.τ²_init, π0=spec.π_init)

        fit = fit_fold_mcem(rng_fold, A0, b0, Arows_train, y_train, σ2_train, grp_train, eb, hs_init, spec;
            method=method)

        fold_key = string(i)
        fold_rows[i] = (
            fold = i,
            held_out_group = fold_key,
            n_train = length(train_idx),
            n_test = 1,
            λ = fit.hs.λ,
            τ² = fit.hs.τ²,
            π = fit.π,
            ν = fit.ν,
            α = fit.α
        )
        store_fits && (fold_fit_slots[i] = fit)

        pred = predict_row_metrics_mc_hier(
            rng_fold, view(Arows, i, :), y[i], σ2[i], grp[i], fit, spec;
            cover_levels=cover_levels
        )

        pit_vec[i] = pred.pit
        crps_vec[i] = pred.crps
        wis_vec[i] = pred.wis
        loglik_vec[i] = pred.loglik
        pred_coverage[i, :] = pred.covered
        pred_widths[i, :] = pred.widths
        pred_mean_vec[i] = pred.pred_mean
        pred_sd_vec[i] = pred.pred_sd
        pred_error_vec[i] = pred.pred_error
    end

    if run_parallel
        Threads.@threads :dynamic for i in 1:n
            _run_fold!(i, _pb_rng_from_seed(fold_seeds[i]))
        end
    else
        for i in 1:n
            _run_fold!(i, _pb_rng_from_seed(fold_seeds[i]))
        end
    end

    fold_info = DataFrame(fold_rows)
    fold_fits = Dict{String,FoldFit}()
    if store_fits
        for i in 1:n
            fit = fold_fit_slots[i]
            fit === nothing || (fold_fits[string(i)] = fit)
        end
    end

    return (pit=pit_vec, crps=crps_vec, wis=wis_vec, loglik=loglik_vec,
            pred_coverage=pred_coverage, pred_widths=pred_widths,
            pred_mean=pred_mean_vec, pred_sd=pred_sd_vec, pred_error=pred_error_vec,
            fold_info=fold_info, fold_fits=fold_fits)
end

"""
    leave_one_source_out_cv(rng, A0, b0, Arows, y, σ2, grp, eb, spec; cover_levels, method)

Leave-one-source-out cross-validation loop.

Groups observations by `grp` and, for each unique group, fits the model on all other
groups and evaluates predictions on the held-out group. In addition to the standard
per-observation metrics, returns a `fold_info` DataFrame with per-fold hyperparameter
estimates and a `fold_fits` dictionary mapping each held-out group name to its fitted
`FoldFit` object.

# Arguments
- `rng`: Random number generator.
- `A0`: Prior design matrix.
- `b0`: Prior mean vector.
- `Arows`: Full design matrix (`n × p`).
- `y`: Response vector (length `n`).
- `σ2`: Observational variance vector (length `n`).
- `grp`: Group label vector (length `n`); each unique value defines one fold.
- `eb`: Empirical Bayes hyperparameter specification.
- `spec`: Fold-fit specification.

# Keywords
- `cover_levels`: Nominal coverage levels (default `[0.5, 0.8, 0.9, 0.95]`).
- `method`: Fitting method passed to `fit_fold_mcem` (default `:collapsed`).

# Returns
A named tuple `(pit, crps, wis, loglik, pred_coverage, pred_widths, fold_info, fold_fits)`.
"""
function leave_one_source_out_cv(
    rng::AbstractRNG,
    A0::AbstractMatrix,
    b0::AbstractVector,
    Arows::AbstractMatrix,
    y::Vector{Float64},
    σ2::Vector{Float64},
    grp::Vector{String},
    eb::EBSpec,
    spec::FoldFitSpec;
    cover_levels::Vector{Float64} = [0.5,0.8,0.9,0.95],
    method::Symbol = :collapsed,
    store_fits::Bool = true,
    parallel::Union{Nothing,Bool} = nothing
)
    unique_groups = sort!(unique(grp))
    n_folds = length(unique_groups)
    n = length(y)

    pit_vec = Vector{Float64}(undef, n)
    crps_vec = Vector{Float64}(undef, n)
    wis_vec = Vector{Float64}(undef, n)
    loglik_vec = Vector{Float64}(undef, n)
    pred_mean_vec = Vector{Float64}(undef, n)
    pred_sd_vec = Vector{Float64}(undef, n)
    pred_error_vec = Vector{Float64}(undef, n)

    K = length(cover_levels)
    pred_coverage = Matrix{Bool}(undef, n, K)
    pred_widths = Matrix{Float64}(undef, n, K)

    fold_rows = Vector{NamedTuple}(undef, n_folds)
    fold_fit_slots = Vector{Union{Nothing,FoldFit}}(undef, n_folds)
    fill!(fold_fit_slots, nothing)

    run_parallel = _pb_should_parallel(n_folds, "PB_PARALLEL_CV", parallel)
    fold_seeds = _pb_thread_seeds(rng, n_folds)
    if run_parallel
        @info "leave_one_source_out_cv: running $n_folds folds across $(Threads.nthreads()) Julia threads"
    end

    function _run_source_fold!(fold_idx::Int, rng_fold::AbstractRNG)
        test_group = unique_groups[fold_idx]
        test_idx = findall(x -> x == test_group, grp)
        train_idx = findall(x -> x != test_group, grp)

        y_train = y[train_idx]
        σ2_train = σ2[train_idx]
        Arows_train = Arows[train_idx, :]
        grp_train = grp[train_idx]

        hs_init = init_hyper(eb; λ0=spec.λ_init, τ²0=spec.τ²_init, π0=spec.π_init)

        fit = fit_fold_mcem(rng_fold, A0, b0, Arows_train, y_train, σ2_train, grp_train, eb, hs_init, spec;
            method=method)

        fold_rows[fold_idx] = (
            fold = fold_idx,
            held_out_group = test_group,
            n_train = length(train_idx),
            n_test = length(test_idx),
            λ = fit.hs.λ,
            τ² = fit.hs.τ²,
            π = fit.π,
            ν = fit.ν,
            α = fit.α
        )
        store_fits && (fold_fit_slots[fold_idx] = fit)

        for i in test_idx
            pred = predict_row_metrics_mc_hier(
                rng_fold, view(Arows, i, :), y[i], σ2[i], grp[i], fit, spec;
                cover_levels=cover_levels
            )

            pit_vec[i] = pred.pit
            crps_vec[i] = pred.crps
            wis_vec[i] = pred.wis
            loglik_vec[i] = pred.loglik
            pred_coverage[i, :] = pred.covered
            pred_widths[i, :] = pred.widths
            pred_mean_vec[i] = pred.pred_mean
            pred_sd_vec[i] = pred.pred_sd
            pred_error_vec[i] = pred.pred_error
        end
    end

    if run_parallel
        Threads.@threads :dynamic for fold_idx in 1:n_folds
            _run_source_fold!(fold_idx, _pb_rng_from_seed(fold_seeds[fold_idx]))
        end
    else
        for fold_idx in 1:n_folds
            _run_source_fold!(fold_idx, _pb_rng_from_seed(fold_seeds[fold_idx]))
        end
    end

    fold_info = DataFrame(fold_rows)
    fold_fits = Dict{String,FoldFit}()
    if store_fits
        for fold_idx in 1:n_folds
            fit = fold_fit_slots[fold_idx]
            fit === nothing || (fold_fits[unique_groups[fold_idx]] = fit)
        end
    end

    return (pit=pit_vec, crps=crps_vec, wis=wis_vec, loglik=loglik_vec,
            pred_coverage=pred_coverage, pred_widths=pred_widths, fold_info=fold_info,
            fold_fits=fold_fits,
            pred_mean=pred_mean_vec, pred_sd=pred_sd_vec, pred_error=pred_error_vec)
end
