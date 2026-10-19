using Random
using Distributions
using DataFrames
using LinearAlgebra
using Statistics
using CSV
using JLD2
using Dates
using BlockDiagonals

# Local helpers

"""
    _load_synth_jld2(path) -> (q_true, Q_obs, Sigma_obs, q_prior, C_prior)

Load a JLD2 file produced by the parameter-balancing pipeline and return the
posterior mean `q_post` as the synthetic ground truth, along with the observation
matrix, observation covariance, and prior.
"""
function _load_synth_jld2(path::String)
    obj = load(path)
    q_true = vec(Array(obj["q_post"]))
    Q_obs = Matrix(Array(obj["Q_obs"]))
    Σ_obs = Symmetric(Matrix(Array(obj["μ_cov"])))
    q_prior = vec(Array(obj["q_prior"]))
    C_prior = Symmetric(Matrix(Array(obj["C_prior"])))
    return q_true, Q_obs, Σ_obs, q_prior, C_prior
end

function _mean_or_nothing(x)
    isempty(x) && return nothing
    return mean(x)
end

function _extract_fold_shape(fold_info::DataFrame, col::Symbol)
    if !(col in propertynames(fold_info)) || nrow(fold_info) == 0
        return nothing
    end
    vals = collect(skipmissing(fold_info[!, col]))
    isempty(vals) && return nothing
    return mean(Float64.(vals))
end

function _default_sampler_method()
    raw = lowercase(strip(get(ENV, "PB_INFERENCE_METHOD", "collapsed")))
    if raw in ("structured_vi", "vi", "structured-vi", "laplace")
        return :structured_vi
    elseif raw == "original"
        return :original
    elseif raw == "collapsed"
        return :collapsed
    else
        @warn "Unknown PB_INFERENCE_METHOD=$raw; using collapsed sampler"
        return :collapsed
    end
end

"""
    make_primary_foldfit_spec(; model, hierarchical, ...) -> FoldFitSpec

Build the `FoldFitSpec` used for all BRENDA and synthetic LOSO experiments reported
in the paper.  Defaults are tuned for production runs: longer chains for non-Gaussian
models, PC prior for tau when hierarchical, and adaptive outer stopping.

# Keywords
- `model`: `:gaussian`, `:t`, or `:skew`.
- `hierarchical`: whether to fit random source effects.
- `n_outer`: number of outer MCEM iterations (default 6 for Gaussian, 8 otherwise).
- `n_mcmc`: posterior samples per outer iteration.
- `burnin`: samples discarded at the start of each MCMC block.
- `thin`: thinning factor.
- Other keywords mirror the `FoldFitSpec` fields directly.
"""
function make_primary_foldfit_spec(;
    model::Symbol = :gaussian,
    hierarchical::Bool = true,
    n_outer::Int = (model == :gaussian ? 6 : 8),
    n_mcmc::Int = (model == :gaussian ? 150 : 200),
    burnin::Int = (model == :gaussian ? 50 : 75),
    thin::Int = 2,
    n_keep_min::Int = 75,
    n_pred_draws::Int = 400,
    posterior_refresh::Int = 1,
    adaptive_stop::Bool = true,
    rtol_outer::Float64 = 1e-3,
    min_outer::Int = 3,
    tau_update::Symbol = (hierarchical ? :pc : :ig),
    tau_pc_rate::Float64 = (hierarchical ? 0.05 : 0.0),
    ν_init::Float64 = 7.0,
    α_init::Float64 = 0.0,
    skew_location::Symbol = :mean,
    update_nu::Bool = (model == :t),
    update_alpha::Bool = (model == :skew),
    refine_shape::Bool = (model != :gaussian),
    nu_penalty::Float64 = 0.0,
    nu_target::Float64 = 10.0,
    alpha_penalty::Float64 = 0.0
)
    return FoldFitSpec(
        model = model,
        hierarchical = hierarchical,
        n_outer = n_outer,
        n_mcmc = n_mcmc,
        burnin = burnin,
        thin = thin,
        n_keep_min = n_keep_min,
        n_pred_draws = n_pred_draws,
        posterior_refresh = posterior_refresh,
        adaptive_stop = adaptive_stop,
        rtol_outer = rtol_outer,
        min_outer = min_outer,
        tau_update = tau_update,
        tau_pc_rate = tau_pc_rate,
        ν_init = ν_init,
        α_init = α_init,
        skew_location = skew_location,
        update_nu = update_nu,
        update_alpha = update_alpha,
        refine_shape = refine_shape,
        nu_penalty = nu_penalty,
        nu_target = nu_target,
        alpha_penalty = alpha_penalty
    )
end

"""
    make_timing_foldfit_spec(; model, hierarchical, ...) -> FoldFitSpec

Build a lightweight `FoldFitSpec` for computational scaling benchmarks.
Uses shorter chains and disables adaptive stopping to isolate per-iteration cost.
"""
function make_timing_foldfit_spec(;
    model::Symbol = :gaussian,
    hierarchical::Bool = true,
    n_outer::Int = 5,
    n_mcmc::Int = 100,
    burnin::Int = 25,
    thin::Int = 1,
    posterior_refresh::Int = 1
)
    return FoldFitSpec(
        model = model,
        hierarchical = hierarchical,
        n_outer = n_outer,
        n_mcmc = n_mcmc,
        burnin = burnin,
        thin = thin,
        n_keep_min = max(25, (n_mcmc - burnin) ÷ max(thin, 1)),
        n_pred_draws = 200,
        posterior_refresh = posterior_refresh,
        adaptive_stop = false,
        tau_update = hierarchical ? :pc : :ig,
        tau_pc_rate = hierarchical ? 0.05 : 0.0,
        ν_init = 7.0,
        α_init = 0.0,
        skew_location = :mean,
        update_nu = (model == :t),
        update_alpha = (model == :skew),
        refine_shape = (model != :gaussian)
    )
end


# Observable-space helpers

"""
    make_information_pair(Q_block, mu_block, Sigma_block) -> (A0, b0)

Convert a block of thermodynamic observations into the canonical information pair
`(A0, b0) = (Q' Sigma^{-1} Q, Q' Sigma^{-1} mu)` used as input to the Gaussian
posterior update.  `Sigma_block` is the eQuilibrator covariance matrix.
"""
function make_information_pair(
    Q_block::AbstractMatrix,
    μ_block::AbstractVector,
    Σ_block::AbstractMatrix
)
    Qm = Matrix{Float64}(Q_block)
    μv = Vector{Float64}(μ_block)
    Σm = Symmetric(Matrix{Float64}(Σ_block))

    size(Qm, 1) == length(μv) || error("make_information_pair: dimension mismatch")
    size(Σm, 1) == length(μv) && size(Σm, 2) == length(μv) ||
        error("make_information_pair: covariance dimension mismatch")

    ZQ = Σm \ Qm
    Zμ = Σm \ μv

    A0 = transpose(Qm) * ZQ
    b0 = transpose(Qm) * Zμ

    return Matrix(Symmetric(A0)), Vector(b0)
end

"""
    build_database_reference_from_blocks(; mu0_mean, mu0_cov, KM_mu, KM_cov, kcat_mu, kcat_cov)
        -> (db_mean, db_std)

Concatenate eQuilibrator/BRENDA database means and standard deviations into flat
vectors in the order `[mu0; KM; kcat]` for use as forest-plot reference values.
"""
function build_database_reference_from_blocks(;
    μ0_mean::AbstractVector,
    μ0_cov::AbstractMatrix,
    KM_mu::AbstractVector,
    KM_cov::AbstractMatrix,
    kcat_mu::AbstractVector,
    kcat_cov::AbstractMatrix
)
    μ0_std   = sqrt.(max.(diag(Symmetric(Matrix{Float64}(μ0_cov))), 0.0))
    KM_std   = sqrt.(max.(diag(Symmetric(Matrix{Float64}(KM_cov))), 0.0))
    kcat_std = sqrt.(max.(diag(Symmetric(Matrix{Float64}(kcat_cov))), 0.0))

    db_mean = vcat(Vector{Float64}(μ0_mean), Vector{Float64}(KM_mu), Vector{Float64}(kcat_mu))
    db_std  = vcat(μ0_std, KM_std, kcat_std)

    return db_mean, db_std
end

function _count_identity_like_rows(Arows::AbstractMatrix; atol::Float64 = 1e-10)
    n = size(Arows, 1)
    cnt = 0
    for i in 1:n
        row = vec(Arows[i, :])
        nz = findall(x -> abs(x) > atol, row)
        if length(nz) == 1 && isapprox(abs(row[nz[1]]), 1.0; atol = atol)
            cnt += 1
        end
    end
    return cnt
end

function _prepare_information_pair(
    Arows::AbstractMatrix,
    d::Int,
    A0::Union{Nothing,AbstractMatrix},
    b0::Union{Nothing,AbstractVector};
    deduplicate_thermo_block::Bool = true,
    atol::Float64 = 1e-10
)
    _A0 = isnothing(A0) ? zeros(Float64, d, d) : Matrix{Float64}(A0)
    _b0 = isnothing(b0) ? zeros(Float64, d)    : Vector{Float64}(b0)

    if deduplicate_thermo_block && (!isnothing(A0) || !isnothing(b0))
        n_id = _count_identity_like_rows(Arows; atol = atol)
        if n_id > 0
            @warn "Arows contains $n_id identity-like rows; possible μ0 duplication detected. Using A0 = 0 and b0 = 0 to avoid double counting."
            _A0 .= 0.0
            _b0 .= 0.0
        end
    end

    return _A0, _b0
end

"""
    _observable_summaries_from_qsamps(qsamps, obs_map) -> (mean, std)

Project posterior samples `qsamps` (d x S) through `obs_map` (n_obs x d) and
return the marginal posterior mean and standard deviation in observable space.
"""
function _observable_summaries_from_qsamps(
    qsamps::AbstractMatrix,
    obs_map::AbstractMatrix
)
    size(obs_map, 2) == size(qsamps, 1) ||
        error("_observable_summaries_from_qsamps: dimension mismatch")
    Z = Matrix{Float64}(obs_map) * Matrix{Float64}(qsamps)
    return vec(mean(Z, dims=2)), vec(std(Z, dims=2))
end

"""
    observable_posterior_summary(qsamps, obs_map) -> (mean, lo, hi)

Project posterior samples `qsamps` (d x S) through `obs_map` (n_obs x d) and
return the pointwise posterior mean and 2.5th/97.5th empirical quantiles in
observable space.  Uses empirical quantiles rather than a normal approximation,
so it correctly reflects the full joint posterior covariance.
"""
function observable_posterior_summary(
    qsamps::AbstractMatrix,
    obs_map::AbstractMatrix
)
    size(obs_map, 2) == size(qsamps, 1) ||
        error("observable_posterior_summary: dimension mismatch")
    Z = Matrix{Float64}(obs_map) * Matrix{Float64}(qsamps)   # n_obs × n_draws
    n = size(Z, 1)
    z_mean = vec(mean(Z, dims=2))
    z_lo   = [quantile(view(Z, i, :), 0.025) for i in 1:n]
    z_hi   = [quantile(view(Z, i, :), 0.975) for i in 1:n]
    return (mean=z_mean, lo=collect(z_lo), hi=collect(z_hi))
end

"""
    ForestPayload

Observable-space summaries for forest plots.  Decoupled from `CVResult` so
the plotting layer has no access to latent-space quantities.

# Fields
- `obs_names`: parameter names in observable space.
- `db_mean`, `db_lo`, `db_hi`: database reference mean and 95% CI.
- `model_mean`, `model_lo`, `model_hi`: posterior mean and 95% CI from the fitted model.
"""
struct ForestPayload
    obs_names  :: Vector{String}
    db_mean    :: Vector{Float64}
    db_lo      :: Vector{Float64}
    db_hi      :: Vector{Float64}
    model_mean :: Vector{Float64}
    model_lo   :: Vector{Float64}
    model_hi   :: Vector{Float64}
end

# Closed-form forest payload. This bypasses MCMC for display.
# Uses the same observable-space mapping as `extract_result_comparison`:
#   q_post, C_post = infer_parameters(Q_obs, μ_obs, Σ_obs, q_prior, C_prior)
#   μ_model = Q_plot * q_post
#   x_cov   = Q_plot * C_post * Q_plot'

"""
    compute_closedform_forest_payload(
        Arows, y, σ2, q_prior, C_prior, Q_plot,
        obs_names, db_mean, db_std;
        τ²=0.0, A0=nothing, b0=nothing
    ) -> ForestPayload

Compute forest plot summaries via the exact closed-form Gaussian posterior,
including the thermodynamic information pair (A0, b0).

Posterior precision:
    P = C_prior^{-1} + A0 + Arows' diag(sigma2 + tau2)^{-1} Arows
Posterior mean:
    q_post = P^{-1} (b0 + Arows' diag(sigma2 + tau2)^{-1} y + C_prior^{-1} q_prior)

tau2 inflates only the BRENDA observation noise (sigma2). The thermodynamic
block (A0, b0) enters without tau2 inflation because eQuilibrator constraints
are not subject to inter-study heterogeneity.

For Fixed models: tau2 = 0.
For RE models: tau2 = estimated tau2 from MCMC (adds source variance to obs noise).
"""
function compute_closedform_forest_payload(
    Arows     :: AbstractMatrix{<:Real},
    y         :: AbstractVector{<:Real},
    σ2        :: AbstractVector{<:Real},
    q_prior   :: AbstractVector{<:Real},
    C_prior   :: AbstractMatrix{<:Real},
    Q_plot    :: AbstractMatrix{<:Real},
    obs_names :: Vector{String},
    db_mean   :: Vector{Float64},
    db_std    :: Vector{Float64};
    τ²        :: Float64 = 0.0,
    A0        :: Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    b0        :: Union{Nothing, AbstractVector{<:Real}} = nothing
)
    n, d = size(Arows)
    n_obs = size(Q_plot, 1)

    length(obs_names) == n_obs ||
        error("compute_closedform_forest_payload: obs_names length ($(length(obs_names))) != Q_plot rows ($n_obs)")
    length(db_mean) == n_obs ||
        error("compute_closedform_forest_payload: db_mean length ($(length(db_mean))) != Q_plot rows ($n_obs)")
    length(db_std) == n_obs ||
        error("compute_closedform_forest_payload: db_std length ($(length(db_std))) != Q_plot rows ($n_obs)")

    A = Matrix{Float64}(Arows)
    yv = Vector{Float64}(y)
    q0 = Vector{Float64}(q_prior)
    C0 = Symmetric(Matrix{Float64}(C_prior))
    Qp = Matrix{Float64}(Q_plot)

    # Effective BRENDA observation covariance: sigma2_i + tau2.
    # tau2 captures source heterogeneity and inflates only the BRENDA noise.
    Cx = Diagonal(Vector{Float64}(σ2) .+ τ²)

    # Closed-form Gaussian posterior
    C0_inv = inv(C0)
    AtCxinv = A' / Cx              # d × n

    # Build posterior precision: P = C0^{-1} + A0 + A' Cx^{-1} A.
    P    = Symmetric(C0_inv + AtCxinv * A)
    info = AtCxinv * yv + C0_inv * q0

    # Include thermodynamic information pair (eQuilibrator constraints)
    if !isnothing(A0) && !isnothing(b0)
        _A0 = Matrix{Float64}(A0)
        _b0 = Vector{Float64}(b0)
        size(_A0) == (d, d) ||
            error("compute_closedform_forest_payload: A0 size $(size(_A0)) != ($d,$d)")
        length(_b0) == d ||
            error("compute_closedform_forest_payload: b0 length $(length(_b0)) != $d")
        P    = Symmetric(Matrix(P) + _A0)
        info = info + _b0
    end

    C_post = Symmetric(inv(P))
    q_post = C_post * info

    # Map to observable space.
    model_mean = vec(Qp * q_post)
    model_cov  = Symmetric(Qp * C_post * Qp')
    model_std  = sqrt.(max.(diag(model_cov), 0.0))

    return ForestPayload(
        obs_names,
        db_mean,
        db_mean .- 1.96 .* db_std,
        db_mean .+ 1.96 .* db_std,
        model_mean,
        model_mean .- 1.96 .* model_std,
        model_mean .+ 1.96 .* model_std
    )
end

"""
    compute_synthetic_closedform_forest(
        jld2_path, Q_plot, obs_names, n_reps, seed;
        data_model, data_has_re, τ²_data, τ²_fit, ...
    ) -> ForestPayload

Full synthetic pipeline: generate data, then compute closed-form forest payload.
"""
function compute_synthetic_closedform_forest(
    jld2_path :: String,
    Q_plot    :: AbstractMatrix{<:Real},
    obs_names :: Vector{String};
    n_reps    :: Int = 50,
    seed      :: Int = 1,
    data_model :: Symbol = :gaussian,
    data_has_re :: Bool = true,
    τ²_data   :: Float64 = 1.0,
    ν_data    :: Float64 = 7.0,
    α_data    :: Float64 = 4.0,
    τ²_fit    :: Float64 = 0.0,   # 0 for Fixed, estimated τ² for RE
    A0        :: Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    b0        :: Union{Nothing, AbstractVector{<:Real}} = nothing,
    deduplicate_thermo_block :: Bool = true
)
    rng = MersenneTwister(seed)
    q_true, Q_obs, Σ_obs, q_prior, C_prior = _load_synth_jld2(jld2_path)
    μ_true = vec(Matrix{Float64}(Q_obs) * Vector{Float64}(q_true))
    m, d = size(Q_obs)

    # Generate synthetic data
    X, _, _ = _simulate_replicates(
        rng, μ_true, Matrix(Σ_obs), n_reps;
        model = data_model, ν = ν_data, α = α_data,
        has_random_effects = data_has_re, τ² = τ²_data,
        sparsity = 0.1, n_rogue = 5
    )

    # Flatten to Arows/y/σ2 (same as run_synthetic_loso_experiment)
    σ2_vec = max.(diag(Symmetric(Matrix(Σ_obs))), eps(Float64))
    n_est = count(!isnan, X)
    Arows = zeros(Float64, n_est, d)
    yv = zeros(Float64, n_est)
    σ2v = zeros(Float64, n_est)
    idx = 0
    for r in 1:n_reps, i in 1:m
        val = X[i, r]
        isnan(val) && continue
        idx += 1
        Arows[idx, :] .= view(Q_obs, i, :)
        yv[idx] = val
        σ2v[idx] = σ2_vec[i]
    end
    if idx < n_est
        Arows = Arows[1:idx, :]; resize!(yv, idx); resize!(σ2v, idx)
    end
    _A0, _b0 = _prepare_information_pair(
        Matrix{Float64}(Arows), d, A0, b0;
        deduplicate_thermo_block = deduplicate_thermo_block
    )

    # Database reference: ground truth observables with OBSERVATION uncertainty
    db_mean = vec(Matrix{Float64}(Q_plot) * Vector{Float64}(q_true))
    db_sd   = sqrt.(max.(diag(Symmetric(Matrix{Float64}(Σ_obs))), eps(Float64)))

    return compute_closedform_forest_payload(
        Arows, yv, σ2v, Vector{Float64}(q_prior), Matrix{Float64}(C_prior),
        Matrix{Float64}(Q_plot), obs_names, db_mean, db_sd;
        τ² = τ²_fit, A0 = _A0, b0 = _b0
    )
end

function _synthetic_database_reference(
    q_true::AbstractVector,
    Q_obs::AbstractMatrix,
    Σ_obs::AbstractMatrix
)
    db_mean = vec(Matrix(Q_obs) * Vector(q_true))
    db_std  = sqrt.(max.(diag(Symmetric(Matrix{Float64}(Σ_obs))), 0.0))
    return db_mean, db_std
end

# Data simulation
# Inference code uses per-observation variances σ2_i and does not model
# the full cross-observation covariance Σ_obs. Therefore the synthetic study
# should generate observation noise consistent with that fitted model.
#
# The default skew generator is mean-zero and variance-matched coordinatewise:
# if eps_i ~ SN(xi_i, omega_i, alpha), choose
#   delta = alpha / sqrt(1 + alpha^2),
#   omega_i^2 = sigma_i^2 / (1 - 2*delta^2/pi),
#   xi_i = - omega_i * delta * sqrt(2/pi),
# so that E[eps_i] = 0 and Var(eps_i) = sigma_i^2.
# For location-parameterized sweeps, xi_i is set to zero for the residual
# draw; the outer linear predictor supplies the canonical location.
"""
    _simulate_replicates(rng, mu_model, Sigma_obs, n_reps; model, nu, alpha,
        has_random_effects, tau2, sparsity, n_rogue, rogue_scale, skew_location)
        -> (X, b, rogue_indices)

Generate `n_reps` replicates of synthetic observations from a specified likelihood
family with optional random source effects and rogue observations.

Each replicate draws noise from the diagonal of `Sigma_obs` only; the full
off-diagonal covariance is not used.  Random effects are iid `N(0, tau2)` scalars
added to all observables within a replicate.  Returns the `m x n_reps` observation
matrix `X` (NaN for missing), the vector of per-replicate random effects `b`, and
the indices of any rogue observations that were injected.
"""
function _simulate_replicates(
    rng::AbstractRNG,
    μ_model::Vector{Float64},
    Σ_obs::Matrix{Float64},
    n_reps::Int;
    model::Symbol = :gaussian,
    ν::Float64 = 50.0,
    α::Float64 = 0.0,
    has_random_effects::Bool = true,
    τ²::Float64 = 1.0,
    sparsity::Float64 = 0.0,
    n_rogue::Int = 5,
    rogue_scale::Float64 = 10.0,
    skew_location::Symbol = :mean
)
    m = length(μ_model)
    @assert size(Σ_obs, 1) == m && size(Σ_obs, 2) == m "Σ_obs must be m×m"

    # Use only marginal variances because the fitted model uses σ2_i only.
    σ2_vec = max.(diag(Symmetric(Σ_obs)), eps(Float64))
    σ_vec = sqrt.(σ2_vec)

    X = zeros(Float64, m, n_reps)
    b = has_random_effects ? rand(rng, Normal(0.0, sqrt(τ²)), n_reps) : zeros(Float64, n_reps)

    if model == :gaussian
        for r in 1:n_reps
            ε = randn(rng, m) .* σ_vec
            X[:, r] .= μ_model .+ b[r] .+ ε
        end

    elseif model == :t
        @assert ν > 2.0 "For finite variance Student-t generation, require ν > 2."
        scale_vec = sqrt.(((ν - 2.0) / ν) .* σ2_vec)
        for r in 1:n_reps
            ε = Vector{Float64}(undef, m)
            @inbounds for i in 1:m
                ε[i] = scale_vec[i] * rand(rng, TDist(ν))
            end
            X[:, r] .= μ_model .+ b[r] .+ ε
        end

    elseif model == :skew
        skew_location in (:mean, :location) ||
            error("skew_location must be :mean or :location")
        δ = α / sqrt(1.0 + α^2)
        denom = 1.0 - (2.0 / π) * δ^2
        @assert denom > 0.0 "Invalid skew parameter: variance correction must be positive."

        ω_vec = sqrt.(σ2_vec ./ denom)
        ξ_vec = skew_location == :mean ?
            .-ω_vec .* δ .* sqrt(2.0 / π) :
            zeros(Float64, m)

        for r in 1:n_reps
            ε = Vector{Float64}(undef, m)
            @inbounds for i in 1:m
                ε[i] = rand(rng, SkewNormal(ξ_vec[i], ω_vec[i], α))
            end
            X[:, r] .= μ_model .+ b[r] .+ ε
        end
    else
        error("Unknown model: $model")
    end

    rogue_indices = Int[]
    if n_rogue > 0
        rogue_indices = sort(randperm(rng, m)[1:min(m, n_rogue)])
        for i in rogue_indices
            bad_reps = rand(rng, 1:n_reps, min(n_reps, 3))
            shift_val = rogue_scale * (rand(rng) > 0.5 ? 1.0 : -1.0)
            X[i, bad_reps] .+= shift_val
        end
    end

    if sparsity > 0.0
        mask = rand(rng, m, n_reps) .< sparsity
        for i in 1:m
            if all(mask[i, :])
                mask[i, rand(rng, 1:n_reps)] = false
            end
        end
        for r in 1:n_reps, i in 1:m
            if mask[i, r]
                X[i, r] = NaN
            end
        end
    end

    return X, b, rogue_indices
end

"""
    _simulate_sweep_replicates(rng, μ_model, Σ_obs, n_reps; model, ν, α, skew_location)

Generates synthetic data for the coverage-sweep experiments using
sweep-specific robust-family parameterizations. The Gaussian working likelihood
with variance σ_i² is correctly specified at the null point (large ν or
α = 0) and incorrectly specified away from it.

Theoretical justification:
    - Student-t:  ε_i = σ_i · T_i with T_i ~ t_ν.
                  Mean 0, variance σ_i² · ν/(ν−2).
                  Reduces to N(0, σ_i²) as ν becomes large.
    - Skew-N:     ε_i = SN(ξ_i, ω_i, α), with
                  ω_i² = σ_i² / (1 - 2δ²/π). With
                  `skew_location = :mean`, ξ_i centers the residual mean at
                  zero; with `:location`, ξ_i = 0.
                  Reduces to N(0, σ_i²) at α = 0.

Both null points yield the Gaussian DGP used by the Gaussian working
likelihood. Away from the null, the sweep changes the residual distribution
seen by the fitted model.

Rogue contamination and sparsity are disabled so the sweep isolates shape.
"""
function _simulate_sweep_replicates(
    rng::AbstractRNG,
    μ_model::Vector{Float64},
    Σ_obs::Matrix{Float64},
    n_reps::Int;
    model::Symbol = :gaussian,
    ν::Float64    = 50.0,
    α::Float64    = 0.0,
    skew_location::Symbol = :mean,
)
    m       = length(μ_model)
    σ2_vec  = max.(diag(Symmetric(Σ_obs)), eps(Float64))
    σ_vec   = sqrt.(σ2_vec)
    X       = zeros(Float64, m, n_reps)

    if model == :gaussian
        @inbounds for r in 1:n_reps, i in 1:m
            X[i, r] = μ_model[i] + σ_vec[i] * randn(rng)
        end

    elseif model == :t
        @assert ν > 2.0 "Student-t generator requires ν > 2 for finite variance."
        @inbounds for r in 1:n_reps, i in 1:m
            X[i, r] = μ_model[i] + σ_vec[i] * rand(rng, TDist(ν))
        end

    elseif model == :skew
        skew_location in (:mean, :location) ||
            error("skew_location must be :mean or :location")
        δ  = α / sqrt(1.0 + α^2)
        mh = sqrt(2.0 / π)
        denom = 1.0 - (2.0 / π) * δ^2
        denom > 0.0 ||
            error("invalid skew-normal variance factor at α=$α")
        ω_vec = σ_vec ./ sqrt(denom)
        ξ_vec = skew_location == :mean ?
            .- ω_vec .* δ .* mh :
            zeros(Float64, m)
        @inbounds for r in 1:n_reps, i in 1:m
            X[i, r] = μ_model[i] + rand(rng, SkewNormal(ξ_vec[i], ω_vec[i], α))
        end
    else
        error("_simulate_sweep_replicates: unknown model $model")
    end

    return X
end

"""
    _sweep_fit_variance(sigma2_obs, fit_model, data_model, nu_data, alpha_data)

Translate sweep-specific data scale into the variance convention used by the
existing likelihood code. Only the Student-t canonical-scale sweep needs this
conversion; the skew-normal sweep uses the same variance-matched convention as
the fitted skew-normal likelihood.
"""
function _sweep_fit_variance(
    sigma2_obs::Vector{Float64},
    fit_model::Symbol,
    data_model::Symbol,
    nu_data::Float64,
    alpha_data::Float64,
)
    sigma2_fit = copy(sigma2_obs)

    if fit_model == :t && data_model == :t
        nu_data > 2.0 ||
            error("Student-t sweep requires nu_data > 2, got $nu_data")
        sigma2_fit .*= nu_data / (nu_data - 2.0)
    end

    return sigma2_fit
end

function _posterior_covariance_from_samples(qsamps::Matrix{Float64})
    d, S = size(qsamps)
    S > 1 || return Symmetric(zeros(Float64, d, d))
    qbar = vec(mean(qsamps, dims = 2))
    centered = qsamps .- qbar
    return Symmetric((centered * centered') ./ (S - 1))
end

function _mahalanobis_error(err::Vector{Float64}, C::Symmetric{Float64,Matrix{Float64}})
    d = length(err)
    scale = max(tr(C) / max(d, 1), eps(Float64))
    C_reg = Symmetric(Matrix(C) + (1e-10 * scale + eps(Float64)) * I)
    F = safe_cholesky(C_reg; allow_ridge = true)
    return dot(err, F \ err)
end

"""
    run_synthetic_loso_experiment(jld2_path; pathway, n_reps, seed, data_model,
        fit_model, fit_has_re, data_has_re, tau2_data, nu_data, alpha_data,
        nominal_levels, spec, param_names, method, A0, b0, obs_map, obs_names,
        obs_database_mean, obs_database_std, deduplicate_thermo_block) -> CVResult

Run a complete leave-one-source-out cross-validation experiment on synthetic data.

Loads the parameter-balancing posterior from `jld2_path`, generates `n_reps`
synthetic replicates under `data_model`, then fits `fit_model` with or without
random effects.  Returns a `CVResult` containing predictive scores (PIT, CRPS,
WIS), coverage matrices, posterior summaries, and observable-space diagnostics.
For skew-normal data, `skew_location = :mean` uses the mean-parameterized
generator; `:location` uses the canonical skew-normal location as the linear
predictor.
"""
function run_synthetic_loso_experiment(
    jld2_path::String;
    pathway::String = "glycolysis",
    n_reps::Int = 50,
    seed::Int = 1,
    data_model::Symbol = :gaussian,
    fit_model::Symbol = :gaussian,
    fit_has_re::Bool = true,
    data_has_re::Bool = true,
    τ²_data::Float64 = 1.0,
    ν_data::Float64 = 7.0,
    α_data::Float64 = 4.0,
    nominal_levels::Vector{Float64} = [0.5, 0.8, 0.9, 0.95],
    spec::Union{FoldFitSpec,Nothing} = nothing,
    param_names::Union{Vector{String},Nothing} = nothing,
    method::Symbol = _default_sampler_method(),
    A0::Union{Nothing,AbstractMatrix} = nothing,
    b0::Union{Nothing,AbstractVector} = nothing,
    obs_map::Union{Nothing,AbstractMatrix} = nothing,
    obs_names::Union{Vector{String},Nothing} = nothing,
    obs_database_mean::Union{Nothing,AbstractVector} = nothing,
    obs_database_std::Union{Nothing,AbstractVector} = nothing,
    deduplicate_thermo_block::Bool = true,
    sparsity::Float64 = 0.1,
    n_rogue::Int = 5,
    rogue_scale::Float64 = 10.0,
    skew_location::Symbol = :mean
) :: CVResult
    rng = MersenneTwister(seed)

    q_true, Q_obs, Σ_obs, q_prior, C_prior = _load_synth_jld2(jld2_path)
    μ_model = Q_obs * q_true
    m, d = size(Q_obs)

    X, b_true, rogue_idxs = _simulate_replicates(
        rng, μ_model, Matrix(Σ_obs), n_reps;
        model = data_model,
        ν = ν_data,
        α = α_data,
        has_random_effects = data_has_re,
        τ² = τ²_data,
        sparsity = sparsity,
        n_rogue = n_rogue,
        rogue_scale = rogue_scale,
        skew_location = skew_location
    )

    n_est = count(!isnan, X)
    Arows = zeros(Float64, n_est, d)
    y = zeros(Float64, n_est)
    σ2 = zeros(Float64, n_est)
    grp = Vector{String}(undef, n_est)
    meta_qrow = zeros(Int, n_est)

    idx = 0
    σ2_vec = diag(Σ_obs)

    for r in 1:n_reps
        gr = "rep_" * string(r)
        for i in 1:m
            val = X[i, r]
            if !isnan(val)
                idx += 1
                Arows[idx, :] .= view(Q_obs, i, :)
                y[idx] = val
                σ2[idx] = σ2_vec[i]
                grp[idx] = gr
                meta_qrow[idx] = i
            end
        end
    end

    if idx < n_est
        Arows = Arows[1:idx, :]
        resize!(y, idx)
        resize!(σ2, idx)
        resize!(grp, idx)
        resize!(meta_qrow, idx)
    end

    eb = EBSpec(q0 = q_prior, C0 = C_prior)

    if spec === nothing
        spec = make_primary_foldfit_spec(
            model = fit_model,
            hierarchical = fit_has_re
        )
    end

    _A0, _b0 = _prepare_information_pair(
        Arows, d, A0, b0;
        deduplicate_thermo_block = deduplicate_thermo_block
    )

    cv_result = leave_one_source_out_cv(
        rng, _A0, _b0, Arows, y, σ2, grp, eb, spec;
        cover_levels = nominal_levels,
        method = method,
        store_fits = false
    )

    hs_full = init_hyper(
        eb;
        λ0 = spec.λ_init,
        τ²0 = spec.τ²_init,
        π0 = spec.π_init
    )
    fit_all = fit_fold_mcem(
        rng, _A0, _b0, Arows, y, σ2, grp, eb, hs_full, spec;
        method = method
    )

    q_post_mean = vec(mean(fit_all.qsamps, dims = 2))
    q_post_std  = vec(std(fit_all.qsamps, dims = 2))

    bias = q_post_mean .- q_true
    rmse = sqrt.(bias.^2 .+ q_post_std.^2)

    interval_width, _ = compute_parameter_intervals(fit_all.qsamps; levels = nominal_levels)
    coverage_CI, coverage_HDI = compute_coverage_matrices(fit_all.qsamps, q_true; levels = nominal_levels)

    # Observable-space summaries for forest plots
    obs_map_eff = isnothing(obs_map) ? Matrix(Q_obs) : Matrix{Float64}(obs_map)
    size(obs_map_eff, 2) == d || error("run_synthetic_loso_experiment: obs_map dimension mismatch")

    obs_post_mean, obs_post_std = _observable_summaries_from_qsamps(fit_all.qsamps, obs_map_eff)

    if isnothing(obs_database_mean) || isnothing(obs_database_std)
        db_mean_eff, db_std_eff = _synthetic_database_reference(q_true, Q_obs, Σ_obs)
    else
        db_mean_eff = Vector{Float64}(obs_database_mean)
        db_std_eff  = Vector{Float64}(obs_database_std)
    end

    n_obsdim = size(obs_map_eff, 1)
    length(obs_post_mean) == n_obsdim || error("run_synthetic_loso_experiment: observable mean length mismatch")
    length(db_mean_eff)   == n_obsdim || error("run_synthetic_loso_experiment: database mean length mismatch")
    length(db_std_eff)    == n_obsdim || error("run_synthetic_loso_experiment: database std length mismatch")

    obs_names_eff = isnothing(obs_names) ? ["obs_$i" for i in 1:n_obsdim] : obs_names
    length(obs_names_eff) == n_obsdim || error("run_synthetic_loso_experiment: obs_names length mismatch")

    obs_metadata = DataFrame(
        obs_id = 1:length(y),
        qrow = meta_qrow,
        group = grp
    )

    ν_fit = fit_model == :t ? _extract_fold_shape(cv_result.fold_info, :ν) : nothing
    α_fit = fit_model == :skew ? _extract_fold_shape(cv_result.fold_info, :α) : nothing

    return CVResult(
        experiment_type = :synthetic,
        pathway = pathway,
        model = fit_model,
        has_random_effects = fit_has_re,
        n_replicates = n_reps,
        n_folds = length(unique(grp)),
        seed = seed,
        ν = ν_fit,
        α = α_fit,
        λ_final = cv_result.fold_info.λ[end],
        τ²_final = cv_result.fold_info.τ²[end],
        param_names = param_names === nothing ? ["q$i" for i in 1:d] : param_names,
        q_true = q_true,
        q_post_mean = q_post_mean,
        q_post_std = q_post_std,
        nominal_levels = nominal_levels,
        coverage_CI = coverage_CI,
        coverage_HDI = coverage_HDI,
        interval_width = interval_width,
        bias = bias,
        rmse = rmse,
        pit = cv_result.pit,
        crps = cv_result.crps,
        wis = cv_result.wis,
        pred_coverage = cv_result.pred_coverage,
        pred_mean = cv_result.pred_mean,
        pred_sd = cv_result.pred_sd,
        pred_error = cv_result.pred_error,
        obs_metadata = obs_metadata,
        fold_info = cv_result.fold_info,
        obs_names = obs_names_eff,
        obs_post_mean = obs_post_mean,
        obs_post_std = obs_post_std,
        obs_database_mean = db_mean_eff,
        obs_database_std = db_std_eff
    )
end

"""
    run_sweep_recovery(jld2_path; kwargs...) -> NamedTuple

Run a sweep-specific parameter-recovery simulation. Each replicate is an
independent synthetic data set with one observation per observable. The function
fits the requested model to each replicate and averages the parameter-coverage
indicators across replicates.

This is intentionally different from leave-one-source-out CV: coverage-vs-shape
figures need empirical recovery over repeated synthetic data sets, not one
coverage indicator from a single pooled data set.
"""
function run_sweep_recovery(
    jld2_path :: String;
    n_reps    :: Int    = 200,
    seed      :: Int    = 1,
    data_model :: Symbol = :gaussian,
    fit_model  :: Symbol = :gaussian,
    ν_data     :: Float64 = 50.0,
    α_data     :: Float64 = 0.0,
    nominal_levels :: Vector{Float64} = [0.95],
    spec       :: Union{FoldFitSpec,Nothing} = nothing,
    method     :: Symbol = _default_sampler_method(),
    A0         :: Union{Nothing,AbstractMatrix} = nothing,
    b0         :: Union{Nothing,AbstractVector} = nothing,
    deduplicate_thermo_block :: Bool = true,
    skew_location::Symbol = :mean,
    parallel::Union{Nothing,Bool} = nothing,
)
    rng = MersenneTwister(seed)
    q_true, Q_obs, Σ_obs, q_prior, C_prior = _load_synth_jld2(jld2_path)
    μ_model = Q_obs * q_true
    m, d    = size(Q_obs)
    σ2_vec  = collect(max.(diag(Symmetric(Matrix{Float64}(Σ_obs))), eps(Float64)))

    # Canonical sweep generator.
    X = _simulate_sweep_replicates(
        rng, μ_model, Matrix(Σ_obs), n_reps;
        model = data_model, ν = ν_data, α = α_data,
        skew_location = skew_location,
    )

    Arows = Matrix(Q_obs)
    σ2    = _sweep_fit_variance(σ2_vec, fit_model, data_model, ν_data, α_data)
    grp   = ["rep" for _ in 1:m]

    eb   = EBSpec(q0 = q_prior, C0 = C_prior)
    spec = isnothing(spec) ?
           make_primary_foldfit_spec(model = fit_model, hierarchical = false) :
           spec
    _A0, _b0 = _prepare_information_pair(
        Arows, d, A0, b0;
        deduplicate_thermo_block = deduplicate_thermo_block,
    )

    K = length(nominal_levels)
    cov_CI_rep  = Array{Float64}(undef, d, K, n_reps)
    cov_HDI_rep = Array{Float64}(undef, d, K, n_reps)
    width_CI_rep  = Array{Float64}(undef, d, K, n_reps)
    width_HDI_rep = Array{Float64}(undef, d, K, n_reps)
    err_rep = Matrix{Float64}(undef, d, n_reps)
    err2_rep = Matrix{Float64}(undef, d, n_reps)
    post_sd_rep = Matrix{Float64}(undef, d, n_reps)
    mahal_rep = Vector{Float64}(undef, n_reps)

    run_parallel = _pb_should_parallel(n_reps, "PB_PARALLEL_SWEEPS", parallel)
    if run_parallel
        @info "run_sweep_recovery: running $n_reps synthetic replicates across $(Threads.nthreads()) Julia threads"
    end

    function _run_replicate!(r::Int)
        rng_r = MersenneTwister(seed + r)
        hs    = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
        fit   = fit_fold_mcem(rng_r, _A0, _b0, Arows, X[:, r], σ2, grp, eb, hs, spec;
                              method = method)
        qhat = vec(mean(fit.qsamps, dims = 2))
        err = qhat .- q_true
        err_rep[:, r] .= err
        err2_rep[:, r] .= err .^ 2
        post_sd_rep[:, r] .= vec(std(fit.qsamps, dims = 2))
        mahal_rep[r] = _mahalanobis_error(err, _posterior_covariance_from_samples(fit.qsamps))

        c_CI, c_HDI = compute_coverage_matrices(fit.qsamps, q_true; levels = nominal_levels)
        w_CI, w_HDI = compute_parameter_intervals(fit.qsamps; levels = nominal_levels)
        cov_CI_rep[:, :, r] .= c_CI
        cov_HDI_rep[:, :, r] .= c_HDI
        width_CI_rep[:, :, r] .= w_CI
        width_HDI_rep[:, :, r] .= w_HDI
    end

    if run_parallel
        Threads.@threads :dynamic for r in 1:n_reps
            _run_replicate!(r)
        end
    else
        for r in 1:n_reps
            _run_replicate!(r)
        end
    end

    bias = vec(sum(err_rep, dims = 2)) ./ n_reps
    rmse = sqrt.(vec(sum(err2_rep, dims = 2)) ./ n_reps)
    coverage_CI = dropdims(sum(cov_CI_rep, dims = 3), dims = 3) ./ n_reps
    coverage_HDI = dropdims(sum(cov_HDI_rep, dims = 3), dims = 3) ./ n_reps
    interval_width_CI = dropdims(sum(width_CI_rep, dims = 3), dims = 3) ./ n_reps
    interval_width_HDI = dropdims(sum(width_HDI_rep, dims = 3), dims = 3) ./ n_reps

    return (
        coverage_CI    = coverage_CI,
        coverage_HDI   = coverage_HDI,
        interval_width_CI = interval_width_CI,
        interval_width_HDI = interval_width_HDI,
        bias            = bias,
        rmse            = rmse,
        mean_abs_bias   = mean(abs.(bias)),
        mean_rmse       = mean(rmse),
        mean_post_sd    = mean(vec(sum(post_sd_rep, dims = 2)) ./ n_reps),
        mahal_mse       = mean(mahal_rep),
        nominal_levels = nominal_levels,
        d              = d,
        n_reps         = n_reps,
    )
end

"""
    run_synthetic_loso_experiment_with_forest(jld2_path; forest_obs_map,
        forest_obs_names, kwargs...) -> (CVResult, ForestPayload)

Like `run_synthetic_loso_experiment`, but also returns a `ForestPayload` with
observable-space posterior quantiles derived from a full-data fit.  Use this
variant when forest plots are needed in addition to CV scores.
"""
function run_synthetic_loso_experiment_with_forest(
    jld2_path::String;
    forest_obs_map   :: AbstractMatrix,
    forest_obs_names :: Vector{String},
    kwargs...
) :: Tuple{CVResult, ForestPayload}

    res = run_synthetic_loso_experiment(jld2_path; kwargs...)

    q_true, Q_obs, Σ_obs, _, _ = _load_synth_jld2(jld2_path)
    μ_true  = vec(Matrix{Float64}(Q_obs) * Vector{Float64}(q_true))
    db_sd   = sqrt.(max.(diag(Symmetric(Matrix{Float64}(Σ_obs))), eps(Float64)))

    size(forest_obs_map, 1) == length(μ_true) ||
        error("run_synthetic_loso_experiment_with_forest: " *
              "forest_obs_map row count ($(size(forest_obs_map,1))) " *
              "must equal synthetic observable count ($(length(μ_true)))")
    length(forest_obs_names) == length(μ_true) ||
        error("run_synthetic_loso_experiment_with_forest: " *
              "forest_obs_names length mismatch")

    # Re-run full-data fit to get draws for observable-space quantiles.
    # (Duplicates fit_fold_mcem; acceptable since this path is only called
    #  from run_synthetic_experiment, not from the sweep loops.)
    rng = MersenneTwister(res.seed)
    _, Q_obs_m, Σ_obs_m, q_prior, C_prior = _load_synth_jld2(jld2_path)
    m, d = size(Q_obs_m)
    μ_model = vec(Matrix{Float64}(Q_obs_m) * Vector{Float64}(q_true))

    X, _, _ = _simulate_replicates(
        rng, μ_model, Matrix(Σ_obs_m), res.n_replicates;
        model = res.model,
        has_random_effects = res.has_random_effects,
        τ² = res.τ²_final > 0.0 ? res.τ²_final : 1.0,
        sparsity = 0.1,
        n_rogue = 5
    )
    n_est = count(!isnan, X)
    Arows_f = zeros(Float64, n_est, d)
    y_f = zeros(Float64, n_est); σ2_f = zeros(Float64, n_est)
    grp_f = Vector{String}(undef, n_est)
    idx = 0; σ2_vec = diag(Σ_obs_m)
    for r in 1:res.n_replicates
        gr = "rep_" * string(r)
        for i in 1:m
            val = X[i, r]
            isnan(val) && continue
            idx += 1
            Arows_f[idx, :] .= view(Q_obs_m, i, :)
            y_f[idx] = val; σ2_f[idx] = σ2_vec[i]; grp_f[idx] = gr
        end
    end
    if idx < n_est
        Arows_f = Arows_f[1:idx,:]; resize!(y_f, idx)
        resize!(σ2_f, idx); resize!(grp_f, idx)
    end

    eb_f = EBSpec(q0=q_prior, C0=C_prior)
    spec_f = make_primary_foldfit_spec(model=res.model, hierarchical=res.has_random_effects)
    hs_f = init_hyper(eb_f; λ0=spec_f.λ_init, τ²0=spec_f.τ²_init, π0=spec_f.π_init)

    # Propagate caller's A0/b0 through _prepare_information_pair (respects dedup guard).
    _A0_raw = get(kwargs, :A0, nothing)
    _b0_raw = get(kwargs, :b0, nothing)
    _dedup  = get(kwargs, :deduplicate_thermo_block, true)
    _A0_f, _b0_f = _prepare_information_pair(
        Arows_f, d, _A0_raw, _b0_raw;
        deduplicate_thermo_block = _dedup
    )

    fit_f = fit_fold_mcem(rng, _A0_f, _b0_f, Arows_f, y_f, σ2_f, grp_f,
                          eb_f, hs_f, spec_f; method=_default_sampler_method())

    obs_sum = observable_posterior_summary(fit_f.qsamps, Matrix{Float64}(forest_obs_map))

    forest = ForestPayload(
        forest_obs_names,
        vec(μ_true),
        vec(μ_true .- 1.96 .* db_sd),
        vec(μ_true .+ 1.96 .* db_sd),
        obs_sum.mean,
        obs_sum.lo,
        obs_sum.hi
    )

    return res, forest
end

"""
    run_ladder_experiment(jld2_path; sweep_param, sweep_values, has_random_effects,
        n_reps, seed, pathway, nominal_levels, method) -> LadderResult

Sweep a single hyperparameter (`sweep_param = :nu` or `:alpha`) over `sweep_values`,
running a full synthetic LOSO experiment at each value.  Returns a `LadderResult`
whose `results` field is indexed by the sweep values, used to produce coverage-vs-df
and coverage-vs-alpha diagnostic plots.
"""
function run_ladder_experiment(
    jld2_path::String;
    sweep_param::Symbol = :ν,
    sweep_values::Vector{Float64} = [3.0, 5.0, 7.0, 10.0, 15.0],
    has_random_effects::Bool = true,
    n_reps::Int = 20,
    seed::Int = 1,
    pathway::String = "glycolysis",
    nominal_levels::Vector{Float64} = [0.5, 0.8, 0.9, 0.95],
    method::Symbol = _default_sampler_method()
) :: LadderResult

    results = CVResult[]

    for val in sweep_values
        if sweep_param == :ν
            spec = make_primary_foldfit_spec(
                model = :t,
                hierarchical = has_random_effects,
                ν_init = max(val, 2.1)
            )
            res = run_synthetic_loso_experiment(
                jld2_path;
                pathway = pathway,
                n_reps = n_reps,
                seed = seed,
                data_model = :t,
                fit_model = :t,
                data_has_re = has_random_effects,
                fit_has_re = has_random_effects,
                ν_data = val,
                τ²_data = has_random_effects ? 1.0 : 0.0,
                nominal_levels = nominal_levels,
                spec = spec,
                method = method
            )
        elseif sweep_param == :α
            spec = make_primary_foldfit_spec(
                model = :skew,
                hierarchical = has_random_effects,
                α_init = 0.0
            )
            res = run_synthetic_loso_experiment(
                jld2_path;
                pathway = pathway,
                n_reps = n_reps,
                seed = seed,
                data_model = :skew,
                fit_model = :skew,
                data_has_re = has_random_effects,
                fit_has_re = has_random_effects,
                α_data = val,
                τ²_data = has_random_effects ? 1.0 : 0.0,
                nominal_levels = nominal_levels,
                spec = spec,
                method = method
            )
        else
            error("Unknown sweep_param: $sweep_param")
        end

        push!(results, res)
    end

    return LadderResult(
        sweep_param = sweep_param,
        sweep_values = sweep_values,
        has_random_effects = has_random_effects,
        results = results
    )
end

# BRENDA helpers

"""
    canonicalize(s) -> String

Normalize a bibliographic source string: lowercase, strip leading/trailing whitespace,
collapse internal whitespace, and remove zero-width Unicode characters.
"""
canonicalize(s::AbstractString) = begin
    t = lowercase(strip(s))
    t = replace(t, r"\s+" => " ")
    t = replace(t, r"[\u200B-\u200D\uFEFF]" => "")
    return t
end

function _auto_source_cols(df::DataFrame)
    cands = Symbol[
        :source, :Source,
        :reference, :Reference,
        :citation, :Citation,
        :pubmed, :PubMed, :pmid, :PMID,
        :doi, :DOI,
        :paper, :Paper,
        :organism, :Organism, :species, :Species
    ]
    cols = Symbol[]
    for c in cands
        if c in propertynames(df)
            push!(cols, c)
            if c in (:reference, :Reference, :pubmed, :PubMed, :pmid, :PMID, :doi, :DOI, :source, :Source)
                return [c]
            end
        end
    end
    return isempty(cols) ? Symbol[] : [first(cols)]
end

function _make_group_id(df::DataFrame, group_cols::Vector{Symbol})
    n = nrow(df)
    if isempty(group_cols)
        return fill("ALL", n)
    end
    gid = Vector{String}(undef, n)
    for i in 1:n
        parts = String[]
        for c in group_cols
            v = df[i, c]
            push!(parts, ismissing(v) ? "NA" : canonicalize(string(v)))
        end
        gid[i] = join(parts, " | ")
    end
    return gid
end

"""
    make_information_pair(Q0, y0, Sigma0) -> (A0, b0)

Compute the thermodynamic information pair `(A0, b0) = (Q0' Sigma0^{-1} Q0, Q0' Sigma0^{-1} y0)`
from eQuilibrator observations `y0` with covariance `Sigma0` and design matrix `Q0`.
"""
function make_information_pair(
    Q0::AbstractMatrix, y0::AbstractVector, Σ0::AbstractMatrix)
    n = size(Q0, 1)
    length(y0) == n || error("make_information_pair: y0 length mismatch")
    size(Σ0, 1) == n && size(Σ0, 2) == n || error("make_information_pair: Σ0 must be n×n")
    Σs = Symmetric(Matrix{Float64}(Σ0))
    F0 = cholesky(Σs)
    Σinv_Q0 = F0 \ Matrix{Float64}(Q0)
    Σinv_y0 = F0 \ Vector{Float64}(y0)
    A0 = Matrix{Float64}(Q0' * Σinv_Q0)
    b0 = Vector{Float64}(Q0' * Σinv_y0)
    return A0, b0
end

"""
    build_database_reference_from_blocks(d; Q_mu0, mu0_mean, mu0_cov, Q_KM, KM_mu, KM_cov,
        Q_kcat, kcat_mu, kcat_cov, atol) -> (db_mean, db_std)

Compute a precision-weighted database reference in the latent parameter space of
dimension `d` from three observation blocks (thermodynamic, KM, kcat).  Only
identity-like rows of each Q block (one nonzero entry) contribute; mixed rows
are skipped.  Returns `(NaN, NaN)` for latent dimensions with no contributing
observation.
"""
function build_database_reference_from_blocks(
    d::Int;
    Q_μ0::AbstractMatrix, μ0_mean::AbstractVector, μ0_cov::AbstractMatrix,
    Q_KM::AbstractMatrix, KM_mu::AbstractVector, KM_cov::AbstractMatrix,
    Q_kcat::AbstractMatrix, kcat_mu::AbstractVector, kcat_cov::AbstractMatrix,
    atol::Float64 = 1e-10)

    mean_num = zeros(Float64, d)
    prec_sum = zeros(Float64, d)

    function accumulate_block!(Qb, mb, Cb)
        vb = diag(Symmetric(Matrix{Float64}(Cb)))
        nrow = size(Qb, 1)
        for r in 1:nrow
            nz = findall(j -> abs(Qb[r, j]) > atol, 1:d)
            length(nz) == 1 || continue
            j = nz[1]; a = Qb[r, j]
            abs(a) > atol || continue
            μj = mb[r] / a; vj = vb[r] / (a^2)
            isfinite(μj) && isfinite(vj) && vj > 0 || continue
            w = 1.0 / vj
            mean_num[j] += w * μj
            prec_sum[j] += w
        end
    end

    accumulate_block!(Q_μ0, μ0_mean, μ0_cov)
    accumulate_block!(Q_KM, KM_mu, KM_cov)
    accumulate_block!(Q_kcat, kcat_mu, kcat_cov)

    db_mean = fill(NaN, d); db_std = fill(NaN, d)
    for j in 1:d
        if prec_sum[j] > 0.0
            db_mean[j] = mean_num[j] / prec_sum[j]
            db_std[j] = sqrt(1.0 / prec_sum[j])
        end
    end
    return db_mean, db_std
end

"""
    run_brenda_loso_experiment(y, sigma2, Arows, grp; pathway, spec, eb, seed,
        param_names, nominal_levels, method, A0, b0, obs_map, obs_names,
        obs_database_mean, obs_database_std, ...) -> CVResult

Run a leave-one-source-out cross-validation experiment on real BRENDA data.

`y`, `sigma2`, `Arows`, and `grp` are the pre-processed observations, per-observation
variances, design-matrix rows, and source group labels respectively.  Returns a
`CVResult` with the same structure as `run_synthetic_loso_experiment`.
"""
function run_brenda_loso_experiment(
    y::Vector{Float64},
    σ2::Vector{Float64},
    Arows::Matrix{Float64},
    grp::Vector{String};
    pathway::String = "glycolysis",
    spec::FoldFitSpec = make_primary_foldfit_spec(),
    eb::EBSpec,
    seed::Int = 1,
    param_names::Union{Vector{String},Nothing} = nothing,
    nominal_levels::Vector{Float64} = [0.5, 0.8, 0.9, 0.95],
    method::Symbol = _default_sampler_method(),
    A0::Union{Nothing,AbstractMatrix} = nothing,
    b0::Union{Nothing,AbstractVector} = nothing,
    obs_map::Union{Nothing,AbstractMatrix} = nothing,
    obs_names::Union{Vector{String},Nothing} = nothing,
    obs_database_mean::Union{Nothing,AbstractVector} = nothing,
    obs_database_std::Union{Nothing,AbstractVector} = nothing,
    deduplicate_thermo_block::Bool = true
) :: CVResult
    rng = MersenneTwister(seed)
    d = size(Arows, 2)

    _A0, _b0 = _prepare_information_pair(
        Arows, d, A0, b0;
        deduplicate_thermo_block = deduplicate_thermo_block
    )

    cv_result = leave_one_source_out_cv(
        rng, _A0, _b0, Arows, y, σ2, grp, eb, spec;
        cover_levels = nominal_levels,
        method = method,
        store_fits = false
    )

    hs_full = init_hyper(
        eb;
        λ0 = spec.λ_init,
        τ²0 = spec.τ²_init,
        π0 = spec.π_init
    )
    fit_all = fit_fold_mcem(
        rng, _A0, _b0, Arows, y, σ2, grp, eb, hs_full, spec;
        method = method
    )

    q_post_mean = vec(mean(fit_all.qsamps, dims = 2))
    q_post_std  = vec(std(fit_all.qsamps, dims = 2))
    interval_width, _ = compute_parameter_intervals(fit_all.qsamps; levels = nominal_levels)

    coverage_CI = zeros(Float64, d, length(nominal_levels))
    coverage_HDI = zeros(Float64, d, length(nominal_levels))

    # Observable-space summaries for forest plots
    obs_post_mean = nothing
    obs_post_std = nothing
    obs_names_eff = obs_names
    db_mean_eff = obs_database_mean
    db_std_eff = obs_database_std

    if !isnothing(obs_map)
        obs_map_eff = Matrix{Float64}(obs_map)
        size(obs_map_eff, 2) == d || error("run_brenda_loso_experiment: obs_map dimension mismatch")

        obs_post_mean, obs_post_std = _observable_summaries_from_qsamps(fit_all.qsamps, obs_map_eff)

        n_obsdim = size(obs_map_eff, 1)
        if isnothing(obs_names_eff)
            obs_names_eff = ["obs_$i" for i in 1:n_obsdim]
        end
        length(obs_names_eff) == n_obsdim || error("run_brenda_loso_experiment: obs_names length mismatch")

        if !isnothing(db_mean_eff)
            length(db_mean_eff) == n_obsdim || error("run_brenda_loso_experiment: database mean length mismatch")
        end
        if !isnothing(db_std_eff)
            length(db_std_eff) == n_obsdim || error("run_brenda_loso_experiment: database std length mismatch")
        end
    end

    obs_metadata = DataFrame(
        obs_id = 1:length(y),
        group = grp,
        y_obs = y,
        σ2_obs = σ2
    )

    return CVResult(
        experiment_type = :brenda_loso,
        pathway = pathway,
        model = spec.model,
        has_random_effects = spec.hierarchical,
        n_replicates = nothing,
        n_folds = length(unique(grp)),
        seed = seed,
        ν = spec.model == :t ? _extract_fold_shape(cv_result.fold_info, :ν) : nothing,
        α = spec.model == :skew ? _extract_fold_shape(cv_result.fold_info, :α) : nothing,
        λ_final = cv_result.fold_info.λ[end],
        τ²_final = cv_result.fold_info.τ²[end],
        param_names = param_names === nothing ? ["q$i" for i in 1:d] : param_names,
        q_true = nothing,
        q_post_mean = q_post_mean,
        q_post_std = q_post_std,
        nominal_levels = nominal_levels,
        coverage_CI = coverage_CI,
        coverage_HDI = coverage_HDI,
        interval_width = interval_width,
        bias = nothing,
        rmse = nothing,
        pit = cv_result.pit,
        crps = cv_result.crps,
        wis = cv_result.wis,
        pred_coverage = cv_result.pred_coverage,
        pred_mean = cv_result.pred_mean,
        pred_sd = cv_result.pred_sd,
        pred_error = cv_result.pred_error,
        obs_metadata = obs_metadata,
        fold_info = cv_result.fold_info,
        obs_names = obs_names_eff,
        obs_post_mean = obs_post_mean,
        obs_post_std = obs_post_std,
        obs_database_mean = isnothing(db_mean_eff) ? nothing : Vector{Float64}(db_mean_eff),
        obs_database_std = isnothing(db_std_eff) ? nothing : Vector{Float64}(db_std_eff)
    )
end

"""
    run_brenda_loo_experiment(y, sigma2, Arows, grp; pathway, spec, eb, seed,
        param_names, nominal_levels, method, A0, b0, obs_map, obs_names,
        obs_database_mean, obs_database_std, ...) -> CVResult

Run a leave-one-record-out cross-validation experiment on real BRENDA data.

This complements `run_brenda_loso_experiment`: each raw record is held out
individually, while the source labels `grp` are retained for the random-effects
predictive rule. If the held-out record's source is represented in the training
set, prediction conditions on that fitted source effect; otherwise it integrates
over a new source effect, as in LOSO.
"""
function run_brenda_loo_experiment(
    y::Vector{Float64},
    σ2::Vector{Float64},
    Arows::Matrix{Float64},
    grp::Vector{String};
    pathway::String = "glycolysis",
    spec::FoldFitSpec = make_primary_foldfit_spec(),
    eb::EBSpec,
    seed::Int = 1,
    param_names::Union{Vector{String},Nothing} = nothing,
    nominal_levels::Vector{Float64} = [0.5, 0.8, 0.9, 0.95],
    method::Symbol = _default_sampler_method(),
    A0::Union{Nothing,AbstractMatrix} = nothing,
    b0::Union{Nothing,AbstractVector} = nothing,
    obs_map::Union{Nothing,AbstractMatrix} = nothing,
    obs_names::Union{Vector{String},Nothing} = nothing,
    obs_database_mean::Union{Nothing,AbstractVector} = nothing,
    obs_database_std::Union{Nothing,AbstractVector} = nothing,
    deduplicate_thermo_block::Bool = true
) :: CVResult
    rng = MersenneTwister(seed)
    d = size(Arows, 2)

    _A0, _b0 = _prepare_information_pair(
        Arows, d, A0, b0;
        deduplicate_thermo_block = deduplicate_thermo_block
    )

    cv_result = leave_one_out_cv(
        rng, _A0, _b0, Arows, y, σ2, grp, eb, spec;
        cover_levels = nominal_levels,
        method = method
    )

    hs_full = init_hyper(
        eb;
        λ0 = spec.λ_init,
        τ²0 = spec.τ²_init,
        π0 = spec.π_init
    )
    fit_all = fit_fold_mcem(
        rng, _A0, _b0, Arows, y, σ2, grp, eb, hs_full, spec;
        method = method
    )

    q_post_mean = vec(mean(fit_all.qsamps, dims = 2))
    q_post_std  = vec(std(fit_all.qsamps, dims = 2))
    interval_width, _ = compute_parameter_intervals(fit_all.qsamps; levels = nominal_levels)

    coverage_CI = zeros(Float64, d, length(nominal_levels))
    coverage_HDI = zeros(Float64, d, length(nominal_levels))

    obs_post_mean = nothing
    obs_post_std = nothing
    obs_names_eff = obs_names
    db_mean_eff = obs_database_mean
    db_std_eff = obs_database_std

    if !isnothing(obs_map)
        obs_map_eff = Matrix{Float64}(obs_map)
        size(obs_map_eff, 2) == d || error("run_brenda_loo_experiment: obs_map dimension mismatch")

        obs_post_mean, obs_post_std = _observable_summaries_from_qsamps(fit_all.qsamps, obs_map_eff)

        n_obsdim = size(obs_map_eff, 1)
        if isnothing(obs_names_eff)
            obs_names_eff = ["obs_$i" for i in 1:n_obsdim]
        end
        length(obs_names_eff) == n_obsdim || error("run_brenda_loo_experiment: obs_names length mismatch")

        if !isnothing(db_mean_eff)
            length(db_mean_eff) == n_obsdim || error("run_brenda_loo_experiment: database mean length mismatch")
        end
        if !isnothing(db_std_eff)
            length(db_std_eff) == n_obsdim || error("run_brenda_loo_experiment: database std length mismatch")
        end
    end

    obs_metadata = DataFrame(
        obs_id = 1:length(y),
        record_id = 1:length(y),
        group = grp,
        y_obs = y,
        σ2_obs = σ2
    )

    return CVResult(
        experiment_type = :brenda_loo,
        pathway = pathway,
        model = spec.model,
        has_random_effects = spec.hierarchical,
        n_replicates = nothing,
        n_folds = length(y),
        seed = seed,
        ν = spec.model == :t ? _extract_fold_shape(cv_result.fold_info, :ν) : nothing,
        α = spec.model == :skew ? _extract_fold_shape(cv_result.fold_info, :α) : nothing,
        λ_final = cv_result.fold_info.λ[end],
        τ²_final = cv_result.fold_info.τ²[end],
        param_names = param_names === nothing ? ["q$i" for i in 1:d] : param_names,
        q_true = nothing,
        q_post_mean = q_post_mean,
        q_post_std = q_post_std,
        nominal_levels = nominal_levels,
        coverage_CI = coverage_CI,
        coverage_HDI = coverage_HDI,
        interval_width = interval_width,
        bias = nothing,
        rmse = nothing,
        pit = cv_result.pit,
        crps = cv_result.crps,
        wis = cv_result.wis,
        pred_coverage = cv_result.pred_coverage,
        pred_mean = cv_result.pred_mean,
        pred_sd = cv_result.pred_sd,
        pred_error = cv_result.pred_error,
        obs_metadata = obs_metadata,
        fold_info = cv_result.fold_info,
        obs_names = obs_names_eff,
        obs_post_mean = obs_post_mean,
        obs_post_std = obs_post_std,
        obs_database_mean = isnothing(db_mean_eff) ? nothing : Vector{Float64}(db_mean_eff),
        obs_database_std = isnothing(db_std_eff) ? nothing : Vector{Float64}(db_std_eff)
    )
end

# Variant of run_brenda_loso_experiment that also returns ForestPayload.
function run_brenda_loso_experiment_with_forest(
    y    :: Vector{Float64},
    σ2   :: Vector{Float64},
    Arows:: Matrix{Float64},
    grp  :: Vector{String};
    forest_obs_map    :: AbstractMatrix,
    forest_obs_names  :: Vector{String},
    forest_db_mean    :: Vector{Float64},
    forest_db_std     :: Vector{Float64},
    kwargs...
) :: Tuple{CVResult, ForestPayload}

    res = run_brenda_loso_experiment(y, σ2, Arows, grp; kwargs...)

    d = size(Arows, 2)
    n_fobs = size(forest_obs_map, 1)
    length(forest_obs_names) == n_fobs ||
        error("run_brenda_loso_experiment_with_forest: obs_names length mismatch")
    length(forest_db_mean) == n_fobs ||
        error("run_brenda_loso_experiment_with_forest: db_mean length mismatch")
    length(forest_db_std) == n_fobs ||
        error("run_brenda_loso_experiment_with_forest: db_std length mismatch")

    # Re-run full-data fit to recover draws for observable quantiles.
    rng = MersenneTwister(res.seed)
    eb_f = EBSpec(q0 = get(kwargs, :eb, nothing) === nothing ?
                        error("pass eb= kwarg") :
                        kwargs[:eb].q0,
                  C0 = kwargs[:eb].C0)

    spec_kw = get(kwargs, :spec, make_primary_foldfit_spec())
    hs_f = init_hyper(eb_f; λ0=spec_kw.λ_init, τ²0=spec_kw.τ²_init, π0=spec_kw.π_init)

    # Propagate caller's A0/b0 through _prepare_information_pair (respects dedup guard).
    _A0_raw = get(kwargs, :A0, nothing)
    _b0_raw = get(kwargs, :b0, nothing)
    _dedup  = get(kwargs, :deduplicate_thermo_block, true)
    _A0_f, _b0_f = _prepare_information_pair(
        Arows, d, _A0_raw, _b0_raw;
        deduplicate_thermo_block = _dedup
    )

    fit_f = fit_fold_mcem(rng, _A0_f, _b0_f, Arows, y, σ2, grp,
                          eb_f, hs_f, spec_kw; method=_default_sampler_method())

    obs_sum = observable_posterior_summary(fit_f.qsamps, Matrix{Float64}(forest_obs_map))

    forest = ForestPayload(
        forest_obs_names,
        forest_db_mean,
        forest_db_mean .- 1.96 .* forest_db_std,
        forest_db_mean .+ 1.96 .* forest_db_std,
        obs_sum.mean,
        obs_sum.lo,
        obs_sum.hi
    )
    return res, forest
end

# Parameter-name helper

function make_param_names(;
    μ0_map, KM_map, kcatfmap, kcatrmap,
    bigg_ids, bigg_reaction_ids,
    array_species, array_reactions, d
)
    resolve_spec(key) = begin
        sid = key isa Integer ? array_species[key] : key
        get(bigg_ids, sid, sid)
    end

    resolve_rxn(key) = begin
        rid = key isa Integer ? array_reactions[key] : key
        get(bigg_reaction_ids, rid, rid)
    end

    index_to_name = Dict{Int, String}()

    for (species_key, idx) in μ0_map
        if idx <= d
            index_to_name[idx] = "μ0:$(resolve_spec(species_key))"
        end
    end

    for ((rxn_key, species_key), idx) in KM_map
        if idx <= d
            index_to_name[idx] = "KM:$(resolve_rxn(rxn_key)):$(resolve_spec(species_key))"
        end
    end

    for (rxn_key, idx) in kcatfmap
        if idx <= d
            index_to_name[idx] = "kcat:$(resolve_rxn(rxn_key)):fwd"
        end
    end

    for (rxn_key, idx) in kcatrmap
        if idx <= d
            index_to_name[idx] = "kcat:$(resolve_rxn(rxn_key)):rev"
        end
    end

    return [get(index_to_name, i, "Param_$i") for i in 1:d]
end

# Observable-space name construction. This is block-aware and does not rely on Q sparsity.
# Q_kcat and Q_KM rows have MULTIPLE nonzero columns (they are linear
# combinations of latent parameters). Attempting to infer names from Q
# sparsity always fails for these blocks. Instead, we construct names
# from the same iteration order used by KM_obs() and kcat_obs() in
# parameter_balancing.jl.
#
# Call signature mirrors the observation extraction: pass the same
# rxns, maps, and ID dicts used to build Q_μ0, Q_KM, Q_kcat.

"""
    make_observable_names_from_obs(; μ0_names, KM_names, kcat_names)

Simplest form: caller provides pre-built name vectors for each block.
Returns vcat(μ0_names, KM_names, kcat_names).
"""
function make_observable_names_from_obs(;
    μ0_names  :: Vector{String},
    KM_names  :: Vector{String},
    kcat_names:: Vector{String}
)
    return vcat(μ0_names, KM_names, kcat_names)
end

"""
    build_obs_block_names(rxns, ...) -> (μ0_names, KM_names, kcat_names)

Reconstruct observable names in the same order as
`chemical_potential_obs`, `KM_obs`, `kcat_obs` from parameter_balancing.jl.

`metabolite_ids` = BRENDA names (for database lookup matching KM_obs).
`display_metabolite_ids` = bigg names (for human-readable labels).
"""
function build_obs_block_names(
    rxns,
    metabolites_obs       :: Vector{String},     # species display names (bigg)
    dict_KM_medians       :: Dict,               # same dict used in KM_obs
    dict_kcat_medians     :: Dict,               # same dict used in kcat_obs
    reaction_ids          :: Dict,               # ec_codes
    metabolite_ids        :: Dict,               # BRENDA metabolite IDs (for lookup)
    forward_metabolites   :: Dict,
    reverse_metabolites   :: Dict,
    bigg_reaction_ids     :: Dict;               # for display names
    display_metabolite_ids :: Union{Dict,Nothing} = nothing,  # bigg IDs for display
    extract_name_fn       :: Function = identity
)
    # If no separate display mapping, use the lookup IDs
    disp_met = isnothing(display_metabolite_ids) ? metabolite_ids : display_metabolite_ids

    # μ0 names: one per observed metabolite, in order
    μ0_names = ["μ0:$m" for m in metabolites_obs]

    # KM names: same iteration order as KM_obs
    KM_names = String[]
    for r in rxns
        rid = reaction_ids[r[1]]
        rxn_display = get(bigg_reaction_ids, r[1], r[1])
        substrates = vcat(r[2].substrates, r[2].products)
        for s in substrates
            sname = extract_name_fn(s)
            met_lookup = metabolite_ids[sname]
            met_display = get(disp_met, sname, met_lookup)
            k = (rid, met_lookup)
            if haskey(dict_KM_medians, k)
                push!(KM_names, "KM:$(rxn_display):$(met_display)")
            end
        end
    end

    # kcat names: same iteration order as kcat_obs
    kcat_names = String[]
    for r in rxns
        rid = reaction_ids[r[1]]
        rxn_display = get(bigg_reaction_ids, r[1], r[1])
        reactant = forward_metabolites[r[1]]
        if haskey(dict_kcat_medians, (rid, reactant))
            push!(kcat_names, "kcat:$(rxn_display):fwd")
        end
        product = reverse_metabolites[r[1]]
        if haskey(dict_kcat_medians, (rid, product))
            push!(kcat_names, "kcat:$(rxn_display):rev")
        end
    end

    return μ0_names, KM_names, kcat_names
end

# Keep old function as deprecated fallback
function make_observable_names(;
    Q_μ0   :: AbstractMatrix,
    Q_KM   :: AbstractMatrix,
    Q_kcat :: AbstractMatrix,
    param_names :: Vector{String},
    atol   :: Float64 = 1e-10
)
    @warn "make_observable_names: Using Q-sparsity fallback. " *
          "kcat/KM rows with multiple nonzeros will get generic 'combo_row' names. " *
          "Use build_obs_block_names() + make_observable_names_from_obs() instead."
    d = length(param_names)

    function row_names(Qblock::AbstractMatrix, fallback_prefix::String)
        nms = String[]
        for r in 1:size(Qblock, 1)
            nz = findall(j -> abs(Qblock[r, j]) > atol, 1:d)
            if length(nz) == 1
                push!(nms, param_names[nz[1]])
            else
                push!(nms, fallback_prefix * "combo_row$(r)")
            end
        end
        return nms
    end

    return vcat(
        row_names(Matrix{Float64}(Q_μ0),   "μ0:"),
        row_names(Matrix{Float64}(Q_KM),   "KM:"),
        row_names(Matrix{Float64}(Q_kcat), "kcat:")
    )
end

# Export

function result_to_tables(res::CVResult)
    params = DataFrame(
        param_id = 1:length(res.param_names),
        param_name = res.param_names,
        q_post_mean = res.q_post_mean,
        q_post_std = res.q_post_std
    )

    if res.q_true !== nothing
        params.q_true = res.q_true
        params.bias = res.bias
        params.rmse = res.rmse
    end

    for (i, level) in enumerate(res.nominal_levels)
        params[!, Symbol("width_$(Int(round(level * 100)))")] = res.interval_width[:, i]
        if res.q_true !== nothing
            params[!, Symbol("coverage_$(Int(round(level * 100)))")] = res.coverage_CI[:, i]
        end
    end

    obs = copy(res.obs_metadata)
    obs.pit = res.pit
    obs.crps = res.crps
    obs.wis = res.wis
    if res.pred_mean !== nothing
        obs.pred_mean = res.pred_mean
    end
    if res.pred_sd !== nothing
        obs.pred_sd = res.pred_sd
    end
    if res.pred_error !== nothing
        obs.pred_error = res.pred_error
        obs.abs_pred_error = abs.(res.pred_error)
        obs.sq_pred_error = res.pred_error .^ 2
    end

    for (i, level) in enumerate(res.nominal_levels)
        obs[!, Symbol("covered_$(Int(round(level * 100)))")] = res.pred_coverage[:, i]
    end

    fold = res.fold_info

    summary = DataFrame(
        experiment_type = [String(res.experiment_type)],
        pathway = [res.pathway],
        model = [String(res.model)],
        has_random_effects = [res.has_random_effects],
        n_folds = [res.n_folds],
        λ_final = [res.λ_final],
        τ²_final = [res.τ²_final],
        ν = [res.ν],
        α = [res.α]
    )

    return (params = params, obs = obs, fold = fold, summary = summary)
end

function save_results(res::CVResult, outdir::String; run_id::String = "")
    tables = result_to_tables(res)
    isdir(outdir) || mkpath(outdir)

    rid = run_id == "" ? Dates.format(now(), "yyyymmdd_HHMMSS") : run_id

    CSV.write(joinpath(outdir, "params_" * rid * ".csv"), tables.params)
    CSV.write(joinpath(outdir, "obs_" * rid * ".csv"), tables.obs)
    CSV.write(joinpath(outdir, "fold_" * rid * ".csv"), tables.fold)
    CSV.write(joinpath(outdir, "summary_" * rid * ".csv"), tables.summary)

    return rid
end

# Experiment 1

function experiment_1_synthetic_data(; method::Symbol = _default_sampler_method())
    println("\n" * "="^80)
    println("EXPERIMENT 1: Synthetic Data - Parameter Recovery")
    println("="^80)

    results = Dict{Symbol, CVResult}()

    models = [:gaussian, :t, :skew]
    re_options = [false, true]
    pnames = @isdefined(PARAM_NAMES_TCA) ? PARAM_NAMES_TCA :
             (@isdefined(PARAM_NAMES_GLYC) ? PARAM_NAMES_GLYC : nothing)

    for model in models
        for has_re in re_options
            key = Symbol("$(model)_$(has_re ? "RE" : "Fix")")
            println("\nRunning: $model ($(has_re ? "Random Effects" : "Fixed Effects"))")

            spec = make_primary_foldfit_spec(
                model = model,
                hierarchical = has_re
            )

            results[key] = run_synthetic_loso_experiment(
                POSTERIOR_FILE;
                pathway = PATHWAY,
                n_reps = N_REPS_SYN,
                seed = SEED,
                data_model = model,
                fit_model = model,
                data_has_re = has_re,
                fit_has_re = has_re,
                τ²_data = has_re ? TAU2_DATA : 0.0,
                ν_data = NU_DATA,
                α_data = ALPHA_DATA,
                nominal_levels = NOMINAL_LEVELS,
                spec = spec,
                param_names = pnames,
                method = method
            )
        end
    end

    outfile = joinpath(RESULTS_DIR, "exp1_synthetic_data.jld2")
    JLD2.jldsave(outfile; results...)
    println("\nResults saved to $outfile")

    println("\nGenerating plots...")
    plot_all_comparisons_synthetic(results; outdir = PLOTS_SYNTHETIC)
    plot_forest_comparisons(results; outdir = PLOTS_FOREST)

    return results
end

# Experiment 2

function experiment_2_ladder_sweeps(; method::Symbol = _default_sampler_method())
    println("\n" * "="^80)
    println("EXPERIMENT 2: Ladder Sweeps - Robustness Analysis")
    println("="^80)

    DF_GRID = [2.1, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    ALPHA_GRID = [-10.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 10.0]

    println("\nSweeping ν (t-distribution df) with random effects")
    sweep_nu_re = run_ladder_experiment(
        POSTERIOR_FILE;
        sweep_param = :ν,
        sweep_values = DF_GRID,
        has_random_effects = true,
        n_reps = N_REPS_SYN,
        seed = SEED,
        pathway = PATHWAY,
        nominal_levels = NOMINAL_LEVELS,
        method = method
    )

    println("\nSweeping ν (t-distribution df) without random effects")
    sweep_nu_fix = run_ladder_experiment(
        POSTERIOR_FILE;
        sweep_param = :ν,
        sweep_values = DF_GRID,
        has_random_effects = false,
        n_reps = N_REPS_SYN,
        seed = SEED,
        pathway = PATHWAY,
        nominal_levels = NOMINAL_LEVELS,
        method = method
    )

    println("\nSweeping α (skew-normal shape) with random effects")
    sweep_alpha_re = run_ladder_experiment(
        POSTERIOR_FILE;
        sweep_param = :α,
        sweep_values = ALPHA_GRID,
        has_random_effects = true,
        n_reps = N_REPS_SYN,
        seed = SEED,
        pathway = PATHWAY,
        nominal_levels = NOMINAL_LEVELS,
        method = method
    )

    println("\nSweeping α (skew-normal shape) without random effects")
    sweep_alpha_fix = run_ladder_experiment(
        POSTERIOR_FILE;
        sweep_param = :α,
        sweep_values = ALPHA_GRID,
        has_random_effects = false,
        n_reps = N_REPS_SYN,
        seed = SEED,
        pathway = PATHWAY,
        nominal_levels = NOMINAL_LEVELS,
        method = method
    )

    JLD2.jldsave(joinpath(RESULTS_DIR, "exp2_sweep_nu_RE.jld2"); res = sweep_nu_re)
    JLD2.jldsave(joinpath(RESULTS_DIR, "exp2_sweep_nu_Fix.jld2"); res = sweep_nu_fix)
    JLD2.jldsave(joinpath(RESULTS_DIR, "exp2_sweep_alpha_RE.jld2"); res = sweep_alpha_re)
    JLD2.jldsave(joinpath(RESULTS_DIR, "exp2_sweep_alpha_Fix.jld2"); res = sweep_alpha_fix)

    println("\nLadder sweep results saved")

    println("\nGenerating plots...")
    plot_sweep_params(sweep_nu_re, sweep_alpha_re; outdir = PLOTS_SWEEP)

    return (
        nu_re = sweep_nu_re,
        nu_fix = sweep_nu_fix,
        alpha_re = sweep_alpha_re,
        alpha_fix = sweep_alpha_fix
    )
end

# Experiment 3

function experiment_3_interval_comparison(; method::Symbol = _default_sampler_method())
    println("\n" * "="^80)
    println("EXPERIMENT 3: Interval Comparison - Coverage & Width Analysis")
    println("="^80)

    results = Dict{Symbol, CVResult}()
    pnames = @isdefined(PARAM_NAMES_TCA) ? PARAM_NAMES_TCA :
             (@isdefined(PARAM_NAMES_GLYC) ? PARAM_NAMES_GLYC : nothing)

    configs = [
        (:gaussian, true, :gaussian, true),
        (:t, true, :gaussian, true),
        (:skew, true, :gaussian, true),
        (:gaussian, true, :t, true),
        (:t, true, :t, true),
        (:skew, true, :t, true),
        (:gaussian, true, :skew, true),
        (:t, true, :skew, true),
        (:skew, true, :skew, true)
    ]

    for (data_model, data_re, fit_model, fit_re) in configs
        key = Symbol("$(data_model)_data_$(data_re ? "RE" : "Fix")_$(fit_model)_fit_$(fit_re ? "RE" : "Fix")")
        println("\nData: $data_model ($(data_re ? "RE" : "Fix")) | Fit: $fit_model ($(fit_re ? "RE" : "Fix"))")

        spec = make_primary_foldfit_spec(
            model = fit_model,
            hierarchical = fit_re
        )

        results[key] = run_synthetic_loso_experiment(
            POSTERIOR_FILE;
            pathway = PATHWAY,
            n_reps = N_REPS_SYN,
            seed = SEED,
            data_model = data_model,
            fit_model = fit_model,
            data_has_re = data_re,
            fit_has_re = fit_re,
            τ²_data = data_re ? TAU2_DATA : 0.0,
            ν_data = NU_DATA,
            α_data = ALPHA_DATA,
            nominal_levels = NOMINAL_LEVELS,
            spec = spec,
            param_names = pnames,
            method = method
        )
    end

    outfile = joinpath(RESULTS_DIR, "exp3_interval_comparison.jld2")
    JLD2.jldsave(outfile; results...)
    println("\nResults saved to $outfile")

    println("\nGenerating plots...")
    plot_all_comparisons_synthetic(results; outdir = PLOTS_SYNTHETIC)
    plot_forest_comparisons(results; outdir = PLOTS_FOREST)

    return results
end

# Experiment 4

function experiment_4_brenda_loso(
    y::Vector{Float64},
    σ2::Vector{Float64},
    Arows::Matrix{Float64},
    grp::Vector{String},
    eb::EBSpec;
    method::Symbol = _default_sampler_method()
)
    println("\n" * "="^80)
    println("EXPERIMENT 4: BRENDA LOSO - Real Data Cross-Validation")
    println("="^80)

    results = Dict{Symbol, CVResult}()

    models = [:gaussian, :t, :skew]
    re_options = [false, true]
    pnames = @isdefined(PARAM_NAMES_TCA) ? PARAM_NAMES_TCA :
             (@isdefined(PARAM_NAMES_GLYC) ? PARAM_NAMES_GLYC : nothing)

    for model in models
        for has_re in re_options
            key = Symbol("$(model)_$(has_re ? "RE" : "Fix")")
            println("\nRunning: $model ($(has_re ? "Random Effects" : "Fixed Effects"))")

            spec = make_primary_foldfit_spec(
                model = model,
                hierarchical = has_re
            )

            results[key] = run_brenda_loso_experiment(
                y, σ2, Arows, grp;
                pathway = PATHWAY,
                spec = spec,
                eb = eb,
                seed = SEED,
                param_names = pnames,
                nominal_levels = NOMINAL_LEVELS,
                method = method
            )
        end
    end

    outfile = joinpath(RESULTS_DIR, "exp4_brenda_loso.jld2")
    JLD2.jldsave(outfile; results...)
    println("\nResults saved to $outfile")

    println("\nGenerating plots...")
    plot_all_comparisons_brenda(results; outdir = PLOTS_BRENDA)

    return results
end

# Experiment 5

function experiment_5_timing_benchmarks(
    y::Vector{Float64},
    σ2::Vector{Float64},
    Arows::Matrix{Float64},
    grp::Vector{String},
    eb::EBSpec;
    method::Symbol = _default_sampler_method()
)
    println("\n" * "="^80)
    println("EXPERIMENT 5: Timing Benchmarks - Computational Cost")
    println("="^80)

    n_outer_grid = [2, 5, 10, 15]
    n_mcmc_grid = [20, 50, 100, 150]
    fixed_outer = 3
    fixed_mcmc = 30

    results = DataFrame(
        pathway = String[],
        model = Symbol[],
        hierarchical = Bool[],
        sweep_type = Symbol[],
        n_outer = Int[],
        n_mcmc = Int[],
        time_sec = Float64[],
        n_folds = Int[]
    )

    models = [:gaussian, :t, :skew]

    for model in models
        for hier in [false, true]
            println("\nModel: $model, Hierarchical: $hier")

            println("  Sweeping n_outer (fixed n_mcmc=$fixed_mcmc)")
            for n_out in n_outer_grid
                spec = make_timing_foldfit_spec(
                    model = model,
                    hierarchical = hier,
                    n_outer = n_out,
                    n_mcmc = fixed_mcmc,
                    burnin = min(10, max(0, fixed_mcmc - 5)),
                    thin = 2
                )

                t_start = time()
                run_brenda_loso_experiment(
                    y, σ2, Arows, grp;
                    pathway = PATHWAY,
                    spec = spec,
                    eb = eb,
                    seed = SEED,
                    method = method
                )
                t_elap = time() - t_start

                push!(results, (PATHWAY, model, hier, :outer, n_out, fixed_mcmc, t_elap, length(unique(grp))))
                print("    n_outer=$n_out: $(round(t_elap, digits=1))s | ")
            end
            println()

            println("  Sweeping n_mcmc (fixed n_outer=$fixed_outer)")
            for n_mc in n_mcmc_grid
                spec = make_timing_foldfit_spec(
                    model = model,
                    hierarchical = hier,
                    n_outer = fixed_outer,
                    n_mcmc = n_mc,
                    burnin = min(10, max(0, n_mc - 5)),
                    thin = 2
                )

                t_start = time()
                run_brenda_loso_experiment(
                    y, σ2, Arows, grp;
                    pathway = PATHWAY,
                    spec = spec,
                    eb = eb,
                    seed = SEED,
                    method = method
                )
                t_elap = time() - t_start

                push!(results, (PATHWAY, model, hier, :mcmc, fixed_outer, n_mc, t_elap, length(unique(grp))))
                print("    n_mcmc=$n_mc: $(round(t_elap, digits=1))s | ")
            end
            println()
        end
    end

    outfile = joinpath(RESULTS_DIR, "exp5_timing_benchmarks.csv")
    CSV.write(outfile, results)
    println("\nTiming results saved to $outfile")

    return results
end

# Main execution

function run_all_tca_experiments(;
    y::Union{Nothing,Vector{Float64}} = nothing,
    σ2::Union{Nothing,Vector{Float64}} = nothing,
    Arows::Union{Nothing,Matrix{Float64}} = nothing,
    grp::Union{Nothing,Vector{String}} = nothing,
    eb::Union{Nothing,EBSpec} = nothing,
    run_real::Bool = false,
    run_timing::Bool = false,
    method::Symbol = _default_sampler_method()
)
    println("\n" * "="^80)
    println("TCA CYCLE - COMPLETE EXPERIMENTS")
    println("="^80)
    println("Output directory: $OUTPUT_ROOT")
    println("Seed: $SEED")
    println("Number of replicates (synthetic): $N_REPS_SYN")
    println("Primary sampler method: $method")
    println("="^80)

    exp1_results = experiment_1_synthetic_data(method = method)
    exp2_results = experiment_2_ladder_sweeps(method = method)
    exp3_results = experiment_3_interval_comparison(method = method)

    exp4_results = nothing
    exp5_results = nothing

    if run_real
        isnothing(y) && error("run_all_tca_experiments: y is required when run_real=true")
        isnothing(σ2) && error("run_all_tca_experiments: σ2 is required when run_real=true")
        isnothing(Arows) && error("run_all_tca_experiments: Arows is required when run_real=true")
        isnothing(grp) && error("run_all_tca_experiments: grp is required when run_real=true")
        isnothing(eb) && error("run_all_tca_experiments: eb is required when run_real=true")

        exp4_results = experiment_4_brenda_loso(y, σ2, Arows, grp, eb; method = method)

        if run_timing
            exp5_results = experiment_5_timing_benchmarks(y, σ2, Arows, grp, eb; method = method)
        end
    end

    println("\n" * "="^80)
    println("COMPLETE - All TCA cycle experiments finished")
    println("="^80)

    return (
        exp1 = exp1_results,
        exp2 = exp2_results,
        exp3 = exp3_results,
        exp4 = exp4_results,
        exp5 = exp5_results
    )
end

function run_all_glycolysis_experiments(;
    y::Union{Nothing,Vector{Float64}} = nothing,
    σ2::Union{Nothing,Vector{Float64}} = nothing,
    Arows::Union{Nothing,Matrix{Float64}} = nothing,
    grp::Union{Nothing,Vector{String}} = nothing,
    eb::Union{Nothing,EBSpec} = nothing,
    run_real::Bool = false,
    run_timing::Bool = false,
    method::Symbol = _default_sampler_method()
)
    println("\n" * "="^80)
    println("GLYCOLYSIS PATHWAY - COMPLETE EXPERIMENTS")
    println("="^80)
    println("Output directory: $OUTPUT_ROOT")
    println("Seed: $SEED")
    println("Number of replicates (synthetic): $N_REPS_SYN")
    println("Primary sampler method: $method")
    println("="^80)

    exp1_results = experiment_1_synthetic_data(method = method)
    exp2_results = experiment_2_ladder_sweeps(method = method)
    exp3_results = experiment_3_interval_comparison(method = method)

    exp4_results = nothing
    exp5_results = nothing

    if run_real
        isnothing(y) && error("run_all_glycolysis_experiments: y is required when run_real=true")
        isnothing(σ2) && error("run_all_glycolysis_experiments: σ2 is required when run_real=true")
        isnothing(Arows) && error("run_all_glycolysis_experiments: Arows is required when run_real=true")
        isnothing(grp) && error("run_all_glycolysis_experiments: grp is required when run_real=true")
        isnothing(eb) && error("run_all_glycolysis_experiments: eb is required when run_real=true")

        exp4_results = experiment_4_brenda_loso(y, σ2, Arows, grp, eb; method = method)

        if run_timing
            exp5_results = experiment_5_timing_benchmarks(y, σ2, Arows, grp, eb; method = method)
        end
    end

    println("\n" * "="^80)
    println("COMPLETE - All glycolysis experiments finished")
    println("="^80)

    return (
        exp1 = exp1_results,
        exp2 = exp2_results,
        exp3 = exp3_results,
        exp4 = exp4_results,
        exp5 = exp5_results
    )
end
