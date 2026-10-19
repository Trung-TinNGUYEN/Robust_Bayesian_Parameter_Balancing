using LinearAlgebra
using Distributions
using Random
using Statistics
using Printf
using DataFrames
using SpecialFunctions: digamma, logabsgamma

# Shared parallel-execution helpers.  The expensive experiment loops are
# deterministic under threading because each independent unit receives a seed
# drawn before work is scheduled across threads.
function _pb_env_bool(name::String, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    return raw in ("1", "true", "yes", "y", "on")
end

function _pb_should_parallel(n_units::Integer, env_name::String, requested::Union{Nothing,Bool})
    n_units > 1 || return false
    Threads.nthreads() > 1 || return false
    if requested !== nothing
        return requested
    end
    return _pb_env_bool("PB_PARALLEL", true) && _pb_env_bool(env_name, true)
end

function _pb_thread_seeds(rng::AbstractRNG, n_units::Integer)
    return rand(rng, UInt32, n_units)
end

function _pb_rng_from_seed(seed::UInt32)
    return MersenneTwister(Int(seed))
end

"""
    CVResult

Result container for a single cross-validation experiment.

Holds predictive metrics (PIT, CRPS, WIS, coverage), final empirical-Bayes
hyperparameters (`λ_final`, `τ²_final`, `ν`, `α`), marginal posterior
summaries for the log-kinetic parameters (`q_post_mean`, `q_post_std`), and
fold-level metadata and observable-space forest-plot quantities.
"""
Base.@kwdef struct CVResult
    experiment_type::Symbol
    pathway::String
    model::Symbol
    has_random_effects::Bool
    n_replicates::Union{Int,Nothing}
    n_folds::Int
    seed::Int

    ν::Union{Float64,Nothing} = nothing
    α::Union{Float64,Nothing} = nothing

    λ_final::Float64
    τ²_final::Float64


    param_names::Vector{String}
    q_true::Union{Vector{Float64},Nothing} = nothing
    q_post_mean::Vector{Float64}
    q_post_std::Vector{Float64}

    nominal_levels::Vector{Float64}
    coverage_CI::Matrix{Float64}
    coverage_HDI::Matrix{Float64}
    interval_width::Matrix{Float64}

    bias::Union{Vector{Float64},Nothing} = nothing
    rmse::Union{Vector{Float64},Nothing} = nothing

    pit::Vector{Float64}
    crps::Vector{Float64}
    wis::Vector{Float64}
    pred_coverage::Matrix{Bool}
    pred_mean::Union{Vector{Float64},Nothing} = nothing
    pred_sd::Union{Vector{Float64},Nothing} = nothing
    pred_error::Union{Vector{Float64},Nothing} = nothing

    obs_metadata::Union{DataFrame,Nothing} = nothing
    fold_info::DataFrame

    # Observable-space summaries for forest plots
    obs_names::Union{Vector{String},Nothing} = nothing
    obs_post_mean::Union{Vector{Float64},Nothing} = nothing
    obs_post_std::Union{Vector{Float64},Nothing} = nothing
    obs_database_mean::Union{Vector{Float64},Nothing} = nothing
    obs_database_std::Union{Vector{Float64},Nothing} = nothing
end


"""
    LadderResult

Result of a hyperparameter sweep (e.g., varying degrees of freedom or skewness).

Holds the swept parameter name, the grid of values, and a `CVResult` for each
grid point.
"""
Base.@kwdef struct LadderResult
    sweep_param::Symbol
    sweep_values::Vector{Float64}
    has_random_effects::Bool
    results::Vector{CVResult}
end

"""
    EBSpec

Empirical Bayes prior specification.

Contains the prior mean `q0` and covariance `C0` for the log-kinetic parameters,
the concentration parameter `κ0` for the hierarchical mean, and shape/rate
parameters (`aλ`, `bλ`, `aτ`, `bτ`) for the inverse-gamma priors on `λ` and
`τ²`, plus Beta prior parameters (`aπ`, `bπ`) for the contamination weight.
"""
Base.@kwdef struct EBSpec
    q0::Vector{Float64}
    C0::Matrix{Float64}
    κ0::Float64 = 50.0
    aλ::Float64 = 1e-2
    bλ::Float64 = 1e-2
    aτ::Float64 = 1e-2
    bτ::Float64 = 1e-2
    aπ::Float64 = 1.0
    bπ::Float64 = 9.0
end

"""
    HyperState

Mutable container for the current MCEM hyperparameter state.

Fields `q_prior`, `λ`, `τ²`, and `π` are updated in-place at each outer
EM iteration.
"""
mutable struct HyperState
    q_prior::Vector{Float64}
    λ::Float64
    τ²::Float64
    π::Float64
end

"""
    FoldFit

Result of fitting one MCEM fold.

Holds the kept posterior samples (`qsamps`, `bsamps`), the sorted group labels
(`grp_levels`), a `τ²` trace (`τ²samps`), the contamination weight (`π`), and
the estimated shape parameters (`ν` for Student-t, `α` for skew-normal), plus
the final `HyperState`.
"""
struct FoldFit
    qsamps::Matrix{Float64}
    bsamps::Union{Matrix{Float64},Nothing}
    grp_levels::Vector{String}
    τ²samps::Vector{Float64}
    π::Float64
    ν::Float64
    α::Float64
    hs::HyperState
    # Kept posterior draws of the per-group skew-t shapes when
    # `update_shapes_mh` is on: group => 2 x S matrix of (ν, α) draws aligned
    # with the columns of `qsamps`.  `nothing` otherwise.
    shape_draws::Union{Nothing,Dict{String,Matrix{Float64}}}
end

# Backward-compatible constructor for the pre-shape-draws field order.
FoldFit(qsamps, bsamps, grp_levels, τ²samps, π, ν, α, hs) =
    FoldFit(qsamps, bsamps, grp_levels, τ²samps, π, ν, α, hs, nothing)

"""
    FoldFitSpec

Configuration for one MCEM fold fit.

Specifies the likelihood model (`:gaussian`, `:t`, `:skew`), whether to include
group-level random effects (`hierarchical`), contamination settings (`contam`,
`δ_shift`, `κ_scale`), sampler counts (`n_outer`, `n_mcmc`, `burnin`, `thin`),
shape-parameter grids and optimization flags (`ν_grid`, `α_grid`,
`update_nu`, `update_alpha`, `refine_shape`), and the `τ²` prior choice
(`tau_update`: `:ig` for inverse-gamma or `:pc` for penalized-complexity).
For skew-normal fits, `skew_location = :mean` treats the linear predictor as
the distribution mean; `:location` treats it as the canonical skew-normal
location parameter.
"""
Base.@kwdef struct FoldFitSpec
    model::Symbol = :gaussian
    hierarchical::Bool = true
    contam::Bool = false

    δ_shift::Float64 = 2.0
    κ_scale::Float64 = 10.0

    allow_ridge::Bool = true

    n_outer::Int = 5
    n_mcmc::Int = 100
    burnin::Int = 25
    thin::Int = 1
    n_keep_min::Int = 75

    λ_init::Float64 = 1.0
    τ²_init::Float64 = 1.0
    π_init::Float64 = 0.05

    n_pred_draws::Int = 200
    b_draws_ll::Int = 50

    ν_grid::Vector{Float64} = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0]
    α_grid::Vector{Float64} = [-10.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 10.0]

    ν_init::Float64 = 5.0
    α_init::Float64 = 0.0
    skew_location::Symbol = :mean

    update_nu::Bool = true
    update_alpha::Bool = true
    refine_shape::Bool = true
    shape_opt_tol::Float64 = 1e-4

    # Optional penalization on shape updates
    nu_penalty::Float64 = 0.0
    nu_target::Float64 = 10.0
    alpha_penalty::Float64 = 0.0

    # τ update choice for hierarchical models
    tau_update::Symbol = :ig      # :ig or :pc
    tau_pc_rate::Float64 = 0.0    # rate lambda_pc in p(tau) = lambda_pc exp(-lambda_pc tau)

    # Exact augmented Gibbs corresponds to posterior_refresh = 1
    posterior_refresh::Int = 1

    # Adaptive outer stopping
    adaptive_stop::Bool = true
    rtol_outer::Float64 = 1e-3
    min_outer::Int = 3

    # Fully Bayesian skew-t shape estimation (Metropolis within Gibbs, one
    # (ν, α) pair per shape group).  The proposal is a symmetric random walk on
    # (log(ν - 2), α); the prior is Gaussian on the same coordinates.  The
    # acceptance ratio uses the marginal skew-t likelihood with the
    # augmentation latents integrated out, and the latents are redrawn from
    # their full conditional immediately afterwards, so the pair
    # (shapes, latents) is a valid blocked update.
    update_shapes_mh::Bool = false
    shape_mh_sd_lognu::Float64 = 0.15
    shape_mh_sd_alpha::Float64 = 0.30
    shape_prior_lognu_mean::Float64 = log(6.0)   # prior mean of log(ν - 2)
    shape_prior_lognu_sd::Float64 = 1.0
    shape_prior_alpha_sd::Float64 = 5.0
end

struct ParamRecResult
    label::String
    q_mean::Vector{Float64}
    q_std::Vector{Float64}
    y_mean::Vector{Float64}
    y_std::Vector{Float64}
end

struct PredResult
    label::String
    y_mean::Vector{Float64}
    y_std::Vector{Float64}
end

struct SweepResult
    param_name::Symbol
    values::Vector{Float64}
    metrics::Dict{Float64, Dict{Symbol, Tuple{Float64, Float64}}}
end

# Initialization and Helpers

"""
    init_hyper(eb::EBSpec; λ0=1.0, τ²0=1.0, π0=0.05) -> HyperState

Initialize a `HyperState` from an `EBSpec`, copying `eb.q0` as the starting
prior mean and using the supplied scalar starting values for `λ`, `τ²`, and `π`.
"""
function init_hyper(eb::EBSpec; λ0::Float64=1.0, τ²0::Float64=1.0, π0::Float64=0.05)
    return HyperState(copy(eb.q0), λ0, τ²0, π0)
end

const _RIDGE = 1e-9

"""
    make_C0_inv_mul(C0; allow_ridge=true) -> Function

Return a closure `f(X)` that computes `C0⁻¹ * X` via a cached Cholesky
factorization. If `C0` is near-singular and `allow_ridge` is true, a small
diagonal ridge is added before factorizing.
"""
function make_C0_inv_mul(C0::AbstractMatrix{<:Real}; allow_ridge::Bool=true)
    C0s = Symmetric(Matrix{Float64}(C0))
    F = nothing
    try
        F = cholesky(C0s)
    catch
        allow_ridge || error("C0 is not SPD and ridge not allowed")
        d = size(C0s, 1)
        ρ = 1e-12 * tr(C0s) / d
        F = cholesky(Symmetric(Matrix(C0s) + ρ*I))
    end
    return X -> (F \ (F' \ X))
end

"""
    safe_cholesky(A; allow_ridge=false) -> Cholesky

Cholesky factorization of the symmetric matrix `A` with progressive diagonal
ridge fallback. If `A` is not strictly positive definite and `allow_ridge` is
true, increasingly large perturbations (from `1e-12` to `1e-2` times the mean
diagonal) are tried in sequence until the factorization succeeds.
"""
function safe_cholesky(A::Symmetric{Float64,Matrix{Float64}}; allow_ridge::Bool=false)
    try
        return cholesky(A)
    catch
        allow_ridge || error("Matrix not SPD and ridge not allowed")
        d = size(A, 1)
        base = max(tr(A) / d, 1e-6)
        # Progressive ridge: try increasingly large perturbations
        for scale in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
            ρ = scale * base
            try
                return cholesky(Symmetric(Matrix(A) + ρ * I))
            catch
                continue
            end
        end
        error("safe_cholesky: failed even with large ridge (base trace/d = $base)")
    end
end

function _all_present_finite(x)
    for v in x
        if ismissing(v) || !isfinite(v)
            return false
        end
    end
    return true
end

function _all_positive_finite(x)
    for v in x
        if ismissing(v) || !isfinite(v) || !(v > 0)
            return false
        end
    end
    return true
end

function _validate_fold_fit_inputs(
    caller::AbstractString,
    A0::AbstractMatrix,
    b0::AbstractVector,
    Arows::AbstractMatrix,
    y::AbstractVector,
    σ2::AbstractVector,
    grp::AbstractVector{String},
    eb::EBSpec,
    hs_init::HyperState,
    spec::FoldFitSpec
)
    N, D = size(Arows)
    N > 0 || error("$caller: Arows must contain at least one observation")
    D > 0 || error("$caller: Arows must contain at least one basis quantity")

    size(A0) == (D, D) || error("$caller: A0 must be D x D; got $(size(A0)) with D=$D")
    length(b0) == D || error("$caller: b0 must have length D=$D; got $(length(b0))")
    length(y) == N || error("$caller: y must have length N=$N; got $(length(y))")
    length(σ2) == N || error("$caller: σ2 must have length N=$N; got $(length(σ2))")
    length(grp) == N || error("$caller: grp must have length N=$N; got $(length(grp))")
    length(eb.q0) == D || error("$caller: eb.q0 must have length D=$D; got $(length(eb.q0))")
    size(eb.C0) == (D, D) || error("$caller: eb.C0 must be D x D; got $(size(eb.C0))")
    length(hs_init.q_prior) == D ||
        error("$caller: hs_init.q_prior must have length D=$D; got $(length(hs_init.q_prior))")

    _all_present_finite(Arows) || error("$caller: Arows contains missing or non-finite values")
    _all_present_finite(A0) || error("$caller: A0 contains missing or non-finite values")
    _all_present_finite(b0) || error("$caller: b0 contains missing or non-finite values")
    _all_present_finite(y) || error("$caller: y contains missing or non-finite values")
    _all_positive_finite(σ2) || error("$caller: σ2 must contain positive finite variances")
    _all_present_finite(eb.q0) || error("$caller: eb.q0 contains missing or non-finite values")
    _all_present_finite(eb.C0) || error("$caller: eb.C0 contains missing or non-finite values")
    _all_present_finite(hs_init.q_prior) ||
        error("$caller: hs_init.q_prior contains missing or non-finite values")

    spec.model in (:gaussian, :t, :skew, :skewt) ||
        error("$caller: unknown likelihood model $(spec.model)")
    spec.skew_location in (:mean, :location) ||
        error("$caller: skew_location must be :mean or :location")
    spec.n_outer >= 1 || error("$caller: n_outer must be positive")
    spec.n_mcmc >= 1 || error("$caller: n_mcmc must be positive")
    0 <= spec.burnin < spec.n_mcmc ||
        error("$caller: burnin must satisfy 0 <= burnin < n_mcmc")
    spec.thin >= 1 || error("$caller: thin must be positive")
    spec.n_keep_min >= 1 || error("$caller: n_keep_min must be positive")
    hs_init.λ > 0 && isfinite(hs_init.λ) || error("$caller: initial λ must be positive and finite")
    if spec.hierarchical
        hs_init.τ² > 0 && isfinite(hs_init.τ²) ||
            error("$caller: initial τ² must be positive and finite for hierarchical fits")
    else
        isfinite(hs_init.τ²) || error("$caller: initial τ² must be finite")
    end
    0 <= hs_init.π <= 1 && isfinite(hs_init.π) ||
        error("$caller: initial π must lie in [0, 1]")
    return nothing
end

# Robust likelihood helpers

"""
    logsumexp2(a, b) -> Float64

Numerically stable evaluation of `log(exp(a) + exp(b))`.
"""
@inline function logsumexp2(a::Float64, b::Float64)
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

"""
    t_scale2_from_meas(σ2, ν) -> Float64

Convert a measured variance `σ2` to the scale² of the Student-t distribution
under the parametrization where `Var(y) = σ2`, i.e. `scale² = (ν-2)/ν · σ2`.
"""
@inline function t_scale2_from_meas(σ2::Float64, ν::Float64)
    return (ν - 2.0) / ν * σ2
end

"""
    skew_omega2_from_meas(σ2, α) -> Float64

Convert a measured variance `σ2` to the ω² parameter of the skew-normal
distribution under the mean-zero reparametrization, where
`ω² = σ2 / (1 - (2/π) δ²)` and `δ = α / √(1 + α²)`.
"""
@inline function skew_omega2_from_meas(σ2::Float64, α::Float64)
    δ = α / sqrt(1 + α^2)
    denom = (1 - (2/π) * δ^2)
    return σ2 / denom
end

# Numerical guards for the skew-t Gibbs updates.  The scale mixture weight is
# bounded away from zero and infinity, and the half-normal latent is capped at a
# large multiple of its conditional standard deviation.
const W_MIN_SKEWT = 1e-8
const W_MAX_SKEWT = 1e8
const U_SD_CAP_SKEWT = 12.0

"""
    skewt_b_nu(ν) -> Float64

The constant `b_ν = sqrt(ν/π) Γ((ν-1)/2) / Γ(ν/2)`, which is `E[W^{-1/2}] E[|T|]`
for `W ~ Gamma(ν/2, ν/2)` and standard normal `T`.  It sets the mean of the
skew-t: `E[Z] = ξ + ω δ b_ν`.  Finite for `ν > 1`.
"""
@inline function skewt_b_nu(ν::Float64)
    ν > 1.0 || return 0.0
    return sqrt(ν / π) * exp(logabsgamma((ν - 1) / 2)[1] - logabsgamma(ν / 2)[1])
end

"""
    skewt_omega2_from_meas(σ2, ν, α) -> Float64

Convert a measured variance `σ2` into the scale parameter `ω²` of the skew-t
under the mean-zero, variance-matched reparametrisation.  With
`Z = ξ + ω U₀ / sqrt(W)`, `U₀ ~ SN(0,1,α)` and `W ~ Gamma(ν/2, ν/2)`,

    Var(Z) = ω² [ ν/(ν-2) − δ² b_ν² ],    δ = α / sqrt(1+α²),

so matching `Var(Z) = σ2` gives the expression below.  Requires `ν > 2`.  The
bracket is positive for all admissible `(ν, α)` because `δ² b_ν² < ν/(ν−2)`, but
it is floored defensively so that a numerically degenerate shape cannot produce a
negative scale.
"""
@inline function skewt_omega2_from_meas(σ2::Float64, ν::Float64, α::Float64)
    ν > 2.0 || error("skew-t requires ν > 2 for finite variance, got ν = $ν")
    δ = α / sqrt(1 + α^2)
    b = skewt_b_nu(ν)
    denom = ν / (ν - 2.0) - δ^2 * b^2
    return σ2 / max(denom, 1e-8)
end

"""
    logpdf_skewt(y, ξ, ω, ν, α) -> Float64

Log density of the Azzalini--Capitanio skew-t,
`f(y) = (2/ω) t_ν(z) T_{ν+1}( α z sqrt((ν+1)/(ν+z²)) )` with `z = (y-ξ)/ω`.
"""
@inline function logpdf_skewt(y::Float64, ξ::Float64, ω::Float64,
                              ν::Float64, α::Float64)
    z = (y - ξ) / ω
    lp = log(2.0) - log(ω) + logpdf(TDist(ν), z)
    arg = α * z * sqrt((ν + 1.0) / (ν + z^2))
    return lp + logcdf(TDist(ν + 1.0), arg)
end

"""
    sample_skewt_meanvar(rng, μ, σ2, ν, α) -> (y, ξ, ω)

Draw one variate from the skew-t with mean `μ` and variance `σ2`, using the
representation `Z = ξ + ω (δ|T₀| + sqrt(1-δ²) T₁) / sqrt(W)`.
"""
@inline function sample_skewt_meanvar(rng::AbstractRNG, μ::Float64, σ2::Float64,
                                      ν::Float64, α::Float64)
    ω2 = skewt_omega2_from_meas(σ2, ν, α)
    ω = sqrt(ω2)
    δ = α / sqrt(1 + α^2)
    ξ = μ - ω * δ * skewt_b_nu(ν)
    w = rand(rng, Gamma(ν / 2, 2.0 / ν))
    u0 = δ * abs(randn(rng)) + sqrt(1 - δ^2) * randn(rng)
    return (ξ + ω * u0 / sqrt(w), ξ, ω)
end

"""
    logpdf_skewnormal(y, ξ, ω, α) -> Float64

Log-pdf of the skew-normal distribution SN(ξ, ω, α) evaluated at `y`.
Uses the standard representation: `log(2/ω) + logpdf(N(0,1), z) + logcdf(N(0,1), αz)`
where `z = (y - ξ) / ω`.
"""
@inline function logpdf_skewnormal(y::Float64, ξ::Float64, ω::Float64, α::Float64)
    z = (y - ξ) / ω
    return log(2/ω) + logpdf(Normal(), z) + logcdf(Normal(), α * z)
end

"""
    apply_contamination(y, σ2, z, δ_shift, κ_scale) -> (y1, σ21)

Return contamination-adjusted observations and variances. For each index `i`
where `z[i] == 1`, the observation is shifted by `-δ_shift` and its variance
is inflated by `κ_scale`; clean observations are returned unchanged.
"""
function apply_contamination(y::Vector{Float64}, σ2::Vector{Float64}, z::Vector{Int}, δ_shift::Float64, κ_scale::Float64)
    n = length(y)
    y1 = similar(y)
    σ21 = similar(σ2)
    @inbounds for i in 1:n
        if z[i] == 1
            y1[i] = y[i] - δ_shift
            σ21[i] = κ_scale * σ2[i]
        else
            y1[i] = y[i]
            σ21[i] = σ2[i]
        end
    end
    return y1, σ21
end

"""
    select_t_df(r; grid) -> Float64

Pick the degrees-of-freedom value from `grid` that maximises the Student-t
log-likelihood on the standardised residuals `r`.
"""
function select_t_df(r::Vector{Float64}; grid::Vector{Float64}=[2.1,3.0,5.0,7.0,10.0,15.0,30.0,50.0])
    best_ν, best_ll = grid[1], -Inf
    for ν in grid
        ll = sum(logpdf(TDist(ν), r))
        if ll > best_ll
            best_ll, best_ν = ll, ν
        end
    end
    return best_ν
end

"""
    select_skew_alpha(r; grid) -> Float64

Pick the skewness parameter `α` from `grid` that maximises the standardised
skew-normal log-likelihood on residuals `r`.
"""
function select_skew_alpha(r::Vector{Float64}; grid::Vector{Float64}=[-10.0,-6.0,-4.0,-2.0,0.0,2.0,4.0,6.0,10.0])
    best_α, best_ll = 0.0, -Inf
    for α in grid
        ll = 0.0
        @inbounds for x in r
            ll += log(2) + logpdf(Normal(), x) + logcdf(Normal(), α * x)
        end
        if ll > best_ll
            best_ll, best_α = ll, α
        end
    end
    return best_α
end

"""
    Latents

Mutable container for the current latent variable state in the augmented Gibbs sampler.

- `w`: scale-mixture weights used by the Student-t representation (``w_i \\sim \\text{Gamma}``).
- `tpos`: truncated-normal draws for the skew-normal representation.
- `z`: binary contamination indicators (0 = clean, 1 = contaminated).
"""
mutable struct Latents
    w::Vector{Float64}
    tpos::Vector{Float64}
    z::Vector{Int}
end

# Non-hierarchical posterior

"""
    posterior_q_gaussian(A0, b0, Arows, y_eff, var_eff, eb, hs; allow_ridge) -> (μ, F)

Compute the Gaussian posterior for `q` on the non-hierarchical path.

Assembles the precision matrix `Prec = (1/λ) C0⁻¹ + A0 + AᵀWA` and right-hand
side from the weighted observations, then returns the posterior mean `μ` and
the Cholesky factor `F` of the precision matrix.

# Arguments
- `A0`, `b0`: additional prior precision and mean contribution (e.g. from regularisation).
- `Arows`: `n × d` design matrix.
- `y_eff`, `var_eff`: effective observations and variances after latent-variable augmentation.
- `eb`: empirical Bayes prior specification.
- `hs`: current hyperparameter state.

# Returns
`(μ, F)` where `μ` is the posterior mean and `F` is the Cholesky of the precision.
"""
function posterior_q_gaussian(
    A0::AbstractMatrix,
    b0::AbstractVector,
    Arows::AbstractMatrix,
    y_eff::AbstractVector,
    var_eff::AbstractVector,
    eb::EBSpec,
    hs::HyperState;
    allow_ridge::Bool=true
)
    n, d = size(Arows)

    C0_inv_mul = make_C0_inv_mul(eb.C0; allow_ridge=allow_ridge)
    inv_C0 = C0_inv_mul(Matrix{Float64}(I, d, d))

    inv_C_q = (1/hs.λ) .* inv_C0
    inv_Cq = (1/hs.λ) .* C0_inv_mul(hs.q_prior)

    w = 1.0 ./ var_eff

    Prec = zeros(Float64, d, d)
    rhs  = zeros(Float64, d)

    @inbounds for i in 1:n
        wi = w[i]
        ai = view(Arows, i, :)
        Prec .+= wi .* (ai * ai')
        rhs  .+= wi .* y_eff[i] .* ai
    end

    Prec .= Prec .+ inv_C_q .+ A0
    rhs  .= rhs .+ inv_Cq .+ b0

    F = safe_cholesky(Symmetric(Prec); allow_ridge=allow_ridge)
    μ = F \ rhs
    return μ, F
end

# Collapsed posterior: marginalize b via Schur complement
#
# Mathematical derivation.
#
# Joint precision of (q, b):
#
#   P = [ P_qq + Pi_q,   P_qb        ]
#       [ P_bq,           D_bb        ]
#
# where:
#   P_qq[j,k]     = sum_i w_i * a_ij * a_ik          (data precision for q)
#   P_qb[j,g]     = sum_{i: g_i=g} w_i * a_ij        (cross-precision q-b)
#   D_bb[g,g]     = 1/tau2 + sum_{i: g_i=g} w_i      (diagonal b precision)
#   Pi_q          = (1/lambda) C0^{-1} + A0          (prior precision for q)
#
# Joint RHS:
#   r_q = (1/lambda) C0^{-1} q_prior + b0 + sum_i w_i y_i a_i
#   r_b[g] = sum_{i: g_i=g} w_i y_i
#
# Schur complement for exact marginalization over b:
#
#   P_q_marg = (P_qq + Pi_q) - P_qb * D_bb^{-1} * P_bq   (d x d matrix)
#   r_q_marg = r_q - P_qb * D_bb^{-1} * r_b
#
# Since D_bb is diagonal, D_bb^{-1} costs O(G), and the Schur product
# P_qb * D_bb^{-1} * P_bq costs O(d^2 * G).
#
# Conditional recovery of b given q:
#
#   b_g | q ~ Normal(mu_{b_g|q}, sigma2_{b_g|q})
#   sigma2_{b_g|q} = 1 / D_bb[g,g]
#   mu_{b_g|q} = sigma2_{b_g|q} * (r_b[g] - P_bq[g,:] * q)
#
# This is G independent scalar Normal draws.
#
# Rao-Blackwell identity for the tau2 update:
#
#   E[||b||^2 | q] = sum_g (mu_{b_g|q}^2 + sigma2_{b_g|q})
#
# By the law of total variance:
#   Var(E[||b||^2 | q]) <= Var(||b||^2)
#
# so averaging E[||b||^2 | q] over q-samples has lower variance
# than averaging ||b_s||^2 over (q_s, b_s) samples.

"""
    CollapsedPosterior

Stores all quantities needed for:
  1. Sampling q from the d-dimensional marginal posterior  (via Cholesky F_q)
  2. Recovering b|q via G independent scalar Normal draws  (via D_bb_inv, P_bq, r_b)
  3. Rao-Blackwellized hyperparameter updates              (via D_bb_inv)
"""
struct CollapsedPosterior
    μ_q::Vector{Float64}          # d: marginal posterior mean of q
    F_q::Cholesky                 # d x d: Cholesky of marginal precision
    P_bq::Matrix{Float64}         # G x d: cross-precision (transposed for row-access)
    D_bb_inv::Vector{Float64}     # G: 1/D_bb[g,g] = conditional variance of b_g|q
    r_b::Vector{Float64}          # G: RHS for b block
    gmap::Dict{String,Int}        # group name to index
    gid::Vector{Int}              # length n: group index for observation
    groups::Vector{String}        # sorted group names
end

# Precomputed invariants for collapsed Gibbs 
struct CollapsedInvariants
    inv_C0::Matrix{Float64}       # d x d: C0^{-1}
    groups::Vector{String}        # sorted group names
    gmap::Dict{String,Int}        # group to index
    gid::Vector{Int}              # length n: per-obs group index
    row_nz::Vector{Vector{Int}}   # length n: nonzero column indices per Arows row
    G::Int
    d::Int
    n::Int
end

# Precomputed invariants for non-hierarchical Gaussian path 
struct GaussianInvariants
    inv_C0::Matrix{Float64}       # d x d: C0^{-1}
    row_nz::Vector{Vector{Int}}   # length n: nonzero column indices per Arows row
    d::Int
    n::Int
end

"""
    _compute_row_supports(Arows) -> Vector{Vector{Int}}

For each row of Arows, return the sorted indices of nonzero entries.
Enables sparse-aware precision assembly: O(sum_i s_i^2) instead of O(n d^2).
"""
function _compute_row_supports(Arows::AbstractMatrix; tol::Float64=0.0)
    n, d = size(Arows)
    row_nz = Vector{Vector{Int}}(undef, n)
    @inbounds for i in 1:n
        nz = Int[]
        for j in 1:d
            if abs(Arows[i, j]) > tol
                push!(nz, j)
            end
        end
        row_nz[i] = nz
    end
    return row_nz
end

"""
    precompute_collapsed_invariants(Arows, grp, eb; allow_ridge)

Quantities independent of (y_eff, var_eff, hs.λ, hs.τ²).
Call once per fold. Cost: O(d^3) + O(nd) for sparsity scan.
"""
function precompute_collapsed_invariants(
    Arows::AbstractMatrix,
    grp::AbstractVector{<:AbstractString},
    eb::EBSpec;
    allow_ridge::Bool = true
)
    n, d = size(Arows)
    groups = sort!(unique(String.(grp)))
    G = length(groups)
    gmap = Dict{String,Int}(g => i for (i, g) in enumerate(groups))
    gid = [gmap[String(g)] for g in grp]
    C0_inv_mul = make_C0_inv_mul(eb.C0; allow_ridge=allow_ridge)
    inv_C0 = C0_inv_mul(Matrix{Float64}(I, d, d))
    row_nz = _compute_row_supports(Arows)
    return CollapsedInvariants(inv_C0, groups, gmap, gid, row_nz, G, d, n)
end

"""
    precompute_gaussian_invariants(Arows, eb; allow_ridge)

Non-hierarchical analogue. Call once per fold.
"""
function precompute_gaussian_invariants(
    Arows::AbstractMatrix,
    eb::EBSpec;
    allow_ridge::Bool = true
)
    n, d = size(Arows)
    C0_inv_mul = make_C0_inv_mul(eb.C0; allow_ridge=allow_ridge)
    inv_C0 = C0_inv_mul(Matrix{Float64}(I, d, d))
    row_nz = _compute_row_supports(Arows)
    return GaussianInvariants(inv_C0, row_nz, d, n)
end

"""
    posterior_q_collapsed_fast(...)

Sparse-aware, zero-allocation (except Cholesky) refresh.
Assembly cost: O(sum_i s_i^2) + O(d^2 G) Schur + O(d^3) Cholesky.
"""
function posterior_q_collapsed_fast(
    A0::AbstractMatrix,
    b0::AbstractVector,
    Arows::AbstractMatrix,
    y_eff::AbstractVector,
    var_eff::AbstractVector,
    ci::CollapsedInvariants,
    eb::EBSpec,
    hs::HyperState;
    allow_ridge::Bool = true,
    P_qq::Matrix{Float64},
    P_qb::Matrix{Float64},
    D_bb_diag::Vector{Float64},
    r_b::Vector{Float64},
    r_q_data::Vector{Float64},
    P_q_marg::Matrix{Float64},
    r_q_marg::Vector{Float64},
    D_bb_inv::Vector{Float64},
    P_bq::Matrix{Float64}
)
    d, G, n = ci.d, ci.G, ci.n
    gid = ci.gid
    row_nz = ci.row_nz
    inv_λ = 1.0 / hs.λ
    inv_τ² = hs.τ² > 0 ? 1.0 / hs.τ² : 1e12

    fill!(P_qq, 0.0)
    fill!(P_qb, 0.0)
    fill!(D_bb_diag, inv_τ²)
    fill!(r_b, 0.0)
    fill!(r_q_data, 0.0)

    # Sparse assembly: only touch nonzero entries per row
    @inbounds for i in 1:n
        wi = 1.0 / var_eff[i]
        nz = row_nz[i]
        g_i = gid[i]
        wi_yi = wi * y_eff[i]

        for ki in eachindex(nz)
            k = nz[ki]
            aik = Arows[i, k]
            wiak = wi * aik
            # Upper triangle of P_qq (symmetric fill)
            for ji in ki:length(nz)
                j = nz[ji]
                v = wiak * Arows[i, j]
                P_qq[j, k] += v
                if j != k
                    P_qq[k, j] += v
                end
            end
            r_q_data[k] += wi_yi * aik
            P_qb[k, g_i] += wiak
        end

        D_bb_diag[g_i] += wi
        r_b[g_i] += wi_yi
    end

    # D_bb_inv (preallocated, no allocation)
    @inbounds for g in 1:G
        D_bb_inv[g] = 1.0 / D_bb_diag[g]
    end

    # Marginal precision = P_qq + (1/lambda) C0^{-1} + A0 - Schur.
    @inbounds for k in 1:d, j in 1:d
        P_q_marg[j, k] = P_qq[j, k] + inv_λ * ci.inv_C0[j, k] + A0[j, k]
    end

    @inbounds for g in 1:G
        inv_d = D_bb_inv[g]
        for k in 1:d
            pqbk = P_qb[k, g]
            pqbk == 0.0 && continue
            for j in k:d
                v = inv_d * pqbk * P_qb[j, g]
                P_q_marg[j, k] -= v
                if j != k; P_q_marg[k, j] -= v; end
            end
        end
    end

    @inbounds for j in 1:d, k in j+1:d
        avg = 0.5 * (P_q_marg[j, k] + P_q_marg[k, j])
        P_q_marg[j, k] = avg
        P_q_marg[k, j] = avg
    end

    # Marginal RHS
    @inbounds for j in 1:d
        s = b0[j]
        for k in 1:d
            s += inv_λ * ci.inv_C0[j, k] * hs.q_prior[k]
        end
        r_q_marg[j] = r_q_data[j] + s
    end
    @inbounds for g in 1:G
        factor = D_bb_inv[g] * r_b[g]
        for j in 1:d
            r_q_marg[j] -= P_qb[j, g] * factor
        end
    end

    F_q = safe_cholesky(Symmetric(P_q_marg); allow_ridge=allow_ridge)
    μ_q = F_q \ r_q_marg

    # P_bq (preallocated, no allocation)
    @inbounds for g in 1:G, j in 1:d
        P_bq[g, j] = P_qb[j, g]
    end

    return CollapsedPosterior(μ_q, F_q, P_bq, D_bb_inv, r_b, ci.gmap, ci.gid, ci.groups)
end

"""
    posterior_q_gaussian_fast(...)

Non-hierarchical fast path. Sparse-aware, reuses precomputed inv_C0.
"""
function posterior_q_gaussian_fast(
    A0::AbstractMatrix,
    b0::AbstractVector,
    Arows::AbstractMatrix,
    y_eff::AbstractVector,
    var_eff::AbstractVector,
    gi::GaussianInvariants,
    eb::EBSpec,
    hs::HyperState;
    allow_ridge::Bool = true,
    Prec::Matrix{Float64},
    rhs::Vector{Float64}
)
    d, n = gi.d, gi.n
    row_nz = gi.row_nz
    inv_λ = 1.0 / hs.λ

    # Prior precision + A0
    @inbounds for k in 1:d, j in 1:d
        Prec[j, k] = inv_λ * gi.inv_C0[j, k] + A0[j, k]
    end

    # Prior RHS
    @inbounds for j in 1:d
        s = b0[j]
        for k in 1:d
            s += inv_λ * gi.inv_C0[j, k] * hs.q_prior[k]
        end
        rhs[j] = s
    end

    # Sparse data assembly
    @inbounds for i in 1:n
        wi = 1.0 / var_eff[i]
        nz = row_nz[i]
        wi_yi = wi * y_eff[i]

        for ki in eachindex(nz)
            k = nz[ki]
            aik = Arows[i, k]
            wiak = wi * aik
            for ji in ki:length(nz)
                j = nz[ji]
                v = wiak * Arows[i, j]
                Prec[j, k] += v
                if j != k; Prec[k, j] += v; end
            end
            rhs[k] += wi_yi * aik
        end
    end

    F = safe_cholesky(Symmetric(Prec); allow_ridge=allow_ridge)
    μ = F \ rhs
    return μ, F
end

# Collapsed posterior using explicit dense matrices.
function posterior_q_collapsed(
    A0::AbstractMatrix,
    b0::AbstractVector,
    Arows::AbstractMatrix,
    y_eff::AbstractVector,
    var_eff::AbstractVector,
    grp::AbstractVector{<:AbstractString},
    eb::EBSpec,
    hs::HyperState;
    allow_ridge::Bool = true
)
    n, d = size(Arows)
    groups = sort!(unique(String.(grp)))
    G = length(groups)
    gmap = Dict{String,Int}(g => i for (i, g) in enumerate(groups))
    gid = [gmap[String(g)] for g in grp]
    C0_inv_mul = make_C0_inv_mul(eb.C0; allow_ridge=allow_ridge)
    inv_C0 = C0_inv_mul(Matrix{Float64}(I, d, d))
    Π_q = (1 / hs.λ) .* inv_C0 .+ A0
    inv_Cq_rhs = (1 / hs.λ) .* C0_inv_mul(hs.q_prior) .+ b0
    inv_τ² = hs.τ² > 0 ? 1.0 / hs.τ² : 1e12
    w = 1.0 ./ var_eff
    P_qq = zeros(Float64, d, d)
    r_q_data = zeros(Float64, d)
    P_qb = zeros(Float64, d, G)
    D_bb_diag = fill(inv_τ², G)
    r_b = zeros(Float64, G)
    @inbounds for i in 1:n
        wi = w[i]; ai = view(Arows, i, :); g_i = gid[i]
        for k in 1:d
            wiak = wi * ai[k]
            for j in k:d
                v = wiak * ai[j]; P_qq[j, k] += v
                if j != k; P_qq[k, j] += v; end
            end
        end
        for j in 1:d
            r_q_data[j] += wi * y_eff[i] * ai[j]
            P_qb[j, g_i] += wi * ai[j]
        end
        D_bb_diag[g_i] += wi; r_b[g_i] += wi * y_eff[i]
    end
    D_bb_inv = 1.0 ./ D_bb_diag
    P_q_marg = P_qq .+ Π_q
    @inbounds for g in 1:G
        inv_d = D_bb_inv[g]
        for k in 1:d; pqbk = P_qb[k, g]
            for j in k:d; v = inv_d * pqbk * P_qb[j, g]
                P_q_marg[j, k] -= v; if j != k; P_q_marg[k, j] -= v; end
            end
        end
    end
    @inbounds for j in 1:d, k in j+1:d
        avg = 0.5 * (P_q_marg[j, k] + P_q_marg[k, j])
        P_q_marg[j, k] = avg; P_q_marg[k, j] = avg
    end
    r_q_marg = r_q_data .+ inv_Cq_rhs
    @inbounds for g in 1:G
        factor = D_bb_inv[g] * r_b[g]
        for j in 1:d; r_q_marg[j] -= P_qb[j, g] * factor; end
    end
    F_q = safe_cholesky(Symmetric(P_q_marg); allow_ridge=allow_ridge)
    μ_q = F_q \ r_q_marg
    P_bq = Matrix(P_qb')
    return CollapsedPosterior(μ_q, F_q, P_bq, D_bb_inv, r_b, gmap, gid, groups)
end

# Collapsed sampling

"""
    sample_q_collapsed(rng, cp::CollapsedPosterior) -> q::Vector{Float64}

Draw q from the collapsed Gaussian posterior using the d-dimensional Cholesky.
Cost: O(d^2) per sample.
"""
@inline function sample_q_collapsed(rng::AbstractRNG, cp::CollapsedPosterior)
    d = length(cp.μ_q)
    z = randn(rng, d)
    # With P = U' * U, solving U * x = z gives covariance P^{-1}.
    return cp.μ_q .+ (cp.F_q.U \ z)
end

"""
    sample_b_conditional!(b, rng, q, cp::CollapsedPosterior)

Draw b_g | q ~ N(mu_{b_g|q}, sigma2_{b_g|q}) for g = 1, ..., G independently.

  mu_{b_g|q}    = D_bb_inv[g] * (r_b[g] - P_bq[g,:] * q)
  sigma2_{b_g|q} = D_bb_inv[g]

Cost: O(dG) per sample (G dot products of dimension d).
"""
function sample_b_conditional!(
    b::Vector{Float64},
    rng::AbstractRNG,
    q::Vector{Float64},
    cp::CollapsedPosterior
)
    G = length(cp.D_bb_inv)
    d = length(q)
    @inbounds for g in 1:G
        # r_b[g] - P_bq[g,:] * q
        residual = cp.r_b[g]
        @inbounds for j in 1:d
            residual -= cp.P_bq[g, j] * q[j]
        end
        σ²_bg = cp.D_bb_inv[g]
        μ_bg = σ²_bg * residual
        b[g] = μ_bg + sqrt(σ²_bg) * randn(rng)
    end
    return b
end

"""
    conditional_b_moments(q, cp::CollapsedPosterior) -> (μ_b, σ²_b)

Compute E[b|q] and Var[b|q] analytically for Rao-Blackwellization.
Returns vectors of length G.
"""
function conditional_b_moments(q::Vector{Float64}, cp::CollapsedPosterior)
    G = length(cp.D_bb_inv)
    d = length(q)
    μ_b = Vector{Float64}(undef, G)
    @inbounds for g in 1:G
        residual = cp.r_b[g]
        @inbounds for j in 1:d
            residual -= cp.P_bq[g, j] * q[j]
        end
        μ_b[g] = cp.D_bb_inv[g] * residual
    end
    return μ_b, cp.D_bb_inv  # σ²_b = D_bb_inv (constant, shared across q samples)
end

# Joint Gaussian sampler.
"""
    sample_from_precision(rng, μ, F) -> Vector{Float64}

Draw one sample from N(μ, F⁻¹) where `F` is the Cholesky factor of the
precision matrix. Solves `F.U * x = z` for a standard normal vector `z`,
so the result has covariance `(FᵀF)⁻¹ = F⁻¹`. Cost: O(d²).
"""
@inline function sample_from_precision(rng::AbstractRNG, μ::Vector{Float64}, F::Cholesky)
    z = randn(rng, length(μ))
    return μ .+ (F.U \ z)
end

# Joint posterior for backward compatibility

function posterior_qb_gaussian(
    A0::AbstractMatrix,
    b0::AbstractVector,
    Arows::AbstractMatrix,
    y_eff::AbstractVector,
    var_eff::AbstractVector,
    grp::AbstractVector{<:AbstractString},
    eb::EBSpec,
    hs::HyperState;
    hierarchical::Bool = true,
    allow_ridge::Bool = true
)
    if !hierarchical
        μq, Fq = posterior_q_gaussian(A0, b0, Arows, y_eff, var_eff, eb, hs; allow_ridge=allow_ridge)
        return μq, nothing, Fq, Dict{String,Int}(), String[]
    end

    n, d = size(Arows)

    groups = sort!(unique(String.(grp)))
    G = length(groups)
    gmap = Dict{String,Int}(g => i for (i, g) in enumerate(groups))
    gid = [gmap[String(g)] for g in grp]

    C0_inv_mul = make_C0_inv_mul(eb.C0; allow_ridge=allow_ridge)
    inv_C0 = C0_inv_mul(Matrix{Float64}(I, d, d))

    inv_C_q = (1/hs.λ) .* inv_C0
    inv_Cq = (1/hs.λ) .* C0_inv_mul(hs.q_prior)

    τ²_use = hs.τ²
    inv_τ² = τ²_use > 0 ? 1.0 / τ²_use : 1e12

    w = 1.0 ./ var_eff

    Prec_qq = zeros(Float64, d, d)
    bb_q = zeros(Float64, d)
    Prec_qb = zeros(Float64, d, G)
    Prec_bb_diag = zeros(Float64, G)
    bb_b = zeros(Float64, G)

    @inbounds for i in 1:n
        wi = w[i]
        ai = view(Arows, i, :)
        Prec_qq .+= wi .* (ai * ai')
        bb_q .+= wi .* y_eff[i] .* ai

        g_i = gid[i]
        Prec_bb_diag[g_i] += wi
        bb_b[g_i] += wi * y_eff[i]
        Prec_qb[:, g_i] .+= wi .* ai
    end

    Prec = zeros(Float64, d + G, d + G)
    Prec[1:d, 1:d] .= inv_C_q .+ A0 .+ Prec_qq
    Prec[1:d, d+1:end] .= Prec_qb
    Prec[d+1:end, 1:d] .= Prec_qb'
    @inbounds for g in 1:G
        Prec[d+g, d+g] = inv_τ² + Prec_bb_diag[g]
    end

    rhs = vcat(inv_Cq .+ b0 .+ bb_q, bb_b)

    F = safe_cholesky(Symmetric(Prec); allow_ridge=allow_ridge)
    μ = F \ rhs

    return μ[1:d], μ[d+1:end], F, gmap, groups
end

# Rao-Blackwellized hyperparameter updates
#
# For lambda: use q-samples only because lambda depends on q.
#
# For tau2: Rao-Blackwellization averages E[||b||^2 | q_s] over q-samples.
#
# Rao-Blackwell identity:
#   For any function h(q, b) with E[h] = theta,
#   Var(E[h(q,b)|q]) <= Var(h(q,b))
#   with equality iff h is measurable w.r.t. q alone.
#
# Applied here with h(q,b) = ||b||^2 and analytic conditional moments.

"""
    update_hyperparameters_rb!(hs, qsamps, cp, eb, C0_inv_mul)

Rao-Blackwellized hyperparameter update using collapsed posterior quantities.

For τ²: computes E[||b||^2 | q_s] analytically for each q-sample, avoiding
explicit b-sampling entirely in the M-step.
"""
function update_hyperparameters_rb!(
    hs::HyperState,
    qsamps::AbstractMatrix,
    cp::CollapsedPosterior,
    eb::EBSpec,
    C0_inv_mul::Function
)
    d, S = size(qsamps)
    G = length(cp.D_bb_inv)

    # Update q_prior.
    μ_bar = vec(mean(qsamps, dims=2))
    inv_λ = 1.0 / hs.λ
    inv_κ = 1.0 / eb.κ0
    hs.q_prior .= (inv_λ .* μ_bar .+ inv_κ .* eb.q0) ./ (inv_λ + inv_κ)

    # Update λ.
    D = 0.0
    tmp = zeros(Float64, d)
    @inbounds for j in 1:S
        tmp .= view(qsamps, :, j) .- hs.q_prior
        D += dot(tmp, C0_inv_mul(tmp))
    end
    D /= S
    hs.λ = (eb.bλ + 0.5 * D) / (eb.aλ + 1 + 0.5 * d)

    # Update tau2 via Rao-Blackwellization.
    # E[||b||^2 | q_s] = sum_g (mu_{b_g|q_s}^2 + sigma2_{b_g})
    # where sigma2_{b_g} = D_bb_inv[g] is constant across s.

    Sbb_rb = 0.0
    sum_var = sum(cp.D_bb_inv)   # sum_g sigma2_{b_g}, constant across samples

    @inbounds for s in 1:S
        q_s = view(qsamps, :, s)
        # Compute E[||b||^2 | q_s] = sum_g mu_{b_g|q_s}^2 + sum_var.
        sum_mu2 = 0.0
        @inbounds for g in 1:G
            residual = cp.r_b[g]
            @inbounds for j in 1:d
                residual -= cp.P_bq[g, j] * q_s[j]
            end
            μ_bg = cp.D_bb_inv[g] * residual
            sum_mu2 += μ_bg * μ_bg
        end
        Sbb_rb += sum_mu2 + sum_var
    end
    Sbb_rb /= S

    hs.τ² = (eb.bτ + 0.5 * Sbb_rb) / (eb.aτ + 1 + 0.5 * G)

    return nothing
end

"""
    update_hyperparameters!(hs, qsamps, bsamps, eb, C0_inv_mul)

Monte Carlo M-step for `q_prior`, `λ`, and `τ²` used by the non-collapsed
(joint `(q, b)`) sampling path. Updates `τ²` directly from the sampled
random-effect norms `||b_s||²` rather than via Rao-Blackwellisation.
"""
function update_hyperparameters!(
    hs::HyperState,
    qsamps::AbstractMatrix,
    bsamps::Union{AbstractMatrix,Nothing},
    eb::EBSpec,
    C0_inv_mul::Function
)
    d, S = size(qsamps)

    μ_bar = vec(mean(qsamps, dims=2))
    inv_λ = 1.0 / hs.λ
    inv_κ = 1.0 / eb.κ0
    hs.q_prior .= (inv_λ .* μ_bar .+ inv_κ .* eb.q0) ./ (inv_λ + inv_κ)

    D = 0.0
    tmp = zeros(Float64, d)
    @inbounds for j in 1:S
        tmp .= view(qsamps, :, j) .- hs.q_prior
        D += dot(tmp, C0_inv_mul(tmp))
    end
    D /= S
    hs.λ = (eb.bλ + 0.5 * D) / (eb.aλ + 1 + 0.5 * d)

    if bsamps !== nothing
        G = size(bsamps, 1)
        Sbb = 0.0
        @inbounds for j in 1:S
            bj = view(bsamps, :, j)
            Sbb += sum(abs2, bj)
        end
        Sbb /= S
        hs.τ² = (eb.bτ + 0.5 * Sbb) / (eb.aτ + 1 + 0.5 * G)
    end
    return nothing
end

# Adaptive convergence monitoring
#
# Monitor convergence of hyperparameters across outer iterations.
# The outer EM loop can terminate early when the relative change in all
# hyperparameters falls below rtol.
#
# JUSTIFICATION:
# The MCEM outer loop is a stochastic approximation to the EM fixed point.
# Once relative changes are below tolerance, additional iterations
# contribute diminishing returns since the marginal likelihood surface
# is typically concave near the mode (Dempster et al., 1977).

"""
    check_convergence(hs_old, hs_new, rtol) -> Bool

Returns true if all hyperparameters have converged.
"""
function check_convergence(
    λ_old::Float64, τ²_old::Float64, π_old::Float64,
    λ_new::Float64, τ²_new::Float64, π_new::Float64,
    rtol::Float64
)
    Δλ = abs(λ_new - λ_old) / max(abs(λ_old), 1e-10)
    Δτ = abs(τ²_new - τ²_old) / max(abs(τ²_old), 1e-10)
    Δπ = abs(π_new - π_old) / max(abs(π_old), 1e-10)
    return max(Δλ, Δτ, Δπ) < rtol
end

# Helper functions

"""
    update_latents!(lat, y1, μlin, σ21, model, ν, α, rng; skew_location=:mean)

Update the augmented latent variables in `lat` given the current linear
predictor `μlin` and effective observations `y1`, `σ21`.

- `:t` model: resamples scale-mixture weights `lat.w` from their Gamma
  full-conditional given the current residuals.
- `:skew` model: resamples truncated-normal auxiliary variables `lat.tpos`
  from their truncated-normal full conditional. The skew-normal linear
  predictor can be either the mean (`skew_location = :mean`) or the canonical
  location (`:location`).
- `:gaussian` model: no-op (latents are unused).
"""
function update_latents!(lat, y1, μlin, σ21, model, ν, α, rng; skew_location::Symbol = :mean)
    n = length(y1)
    if model == :t
        @inbounds for i in 1:n
            ri = y1[i] - μlin[i]
            s2 = t_scale2_from_meas(σ21[i], ν)
            shape = (ν + 1.0)/2.0
            scale = 2.0 / (ν + (ri^2)/s2)
            lat.w[i] = rand(rng, Gamma(shape, scale))
        end
    elseif model == :skew
        skew_location in (:mean, :location) ||
            error("skew_location must be :mean or :location")
        δs = α / sqrt(1 + α^2)
        mHN = sqrt(2/pi)
        @inbounds for i in 1:n
            ω2 = skew_omega2_from_meas(σ21[i], α)
            Δ  = sqrt(ω2) * δs
            Γ  = ω2 * (1 - δs^2)
            vpost = 1.0 / (1.0 + (Δ^2)/Γ)
            center_shift = skew_location == :mean ? Δ * mHN : 0.0
            mpost = vpost * (Δ / Γ) * (y1[i] - μlin[i] + center_shift)
            u0 = cdf(Normal(mpost, sqrt(vpost)), 0.0)
            lat.tpos[i] = quantile(Normal(mpost, sqrt(vpost)), u0 + (1-u0)*rand(rng))
        end
    elseif model == :skewt
        # Skew-t: a scale mixture of skew-normals.  Writing r = y - mu and
        # A = r + Δ b_ν, the augmented model is
        #     A = Δ ũ + sqrt(Γ/w) e,   ũ | w ~ HalfNormal(0, 1/w),
        #     w ~ Gamma(ν/2, ν/2),
        # which keeps both full conditionals conjugate:
        #     ũ | w, r ~ TN+( vpost (Δ/Γ) A , vpost / w ),
        #     w | ũ, r ~ Gamma( ν/2 + 1, [ν + ũ² + (A - Δũ)²/Γ] / 2 ).
        # Scaling the half-normal by 1/sqrt(w), rather than the response, is what
        # makes the w-conditional conjugate.
        δs = α / sqrt(1 + α^2)
        b = skewt_b_nu(ν)
        @inbounds for i in 1:n
            ω2 = skewt_omega2_from_meas(σ21[i], ν, α)
            Δ  = sqrt(ω2) * δs
            Γ  = ω2 * (1 - δs^2)
            A  = y1[i] - μlin[i] + Δ * b
            vpost = 1.0 / (1.0 + (Δ^2) / Γ)
            mpost = vpost * (Δ / Γ) * A

            # The two conditionals are coupled through w, and a badly scaled
            # starting point (large residuals before the first q update) can send
            # w towards zero and u towards infinity.  Both draws are therefore
            # clamped to generous but finite ranges; the bounds are far outside
            # the support the chain occupies once it has equilibrated, so they do
            # not bias the stationary distribution in practice.
            wi = clamp(lat.w[i], W_MIN_SKEWT, W_MAX_SKEWT)
            sd_u = sqrt(vpost / wi)
            u0 = cdf(Normal(mpost, sd_u), 0.0)
            ut = quantile(Normal(mpost, sd_u), u0 + (1 - u0) * rand(rng))
            ut = isfinite(ut) ? clamp(ut, 0.0, mpost + U_SD_CAP_SKEWT * sd_u) :
                                max(mpost, 0.0)
            lat.tpos[i] = ut

            rate = (ν + ut^2 + (A - Δ * ut)^2 / Γ) / 2
            rate = clamp(rate, 1e-12, 1e12)
            lat.w[i] = clamp(rand(rng, Gamma(ν / 2 + 1.0, 1.0 / rate)),
                             W_MIN_SKEWT, W_MAX_SKEWT)
        end
    end
end

"""
    update_latents!(lat, y1, μlin, σ21, model, ν::AbstractVector, α::AbstractVector, rng;
                    skew_location=:mean)

Record-indexed shape variant of [`update_latents!`](@ref): `ν[i]` and `α[i]` are
the skew-t shape parameters of record `i`, so that different quantity types can
carry different asymmetry and tail weight. The conditional updates are identical
to the shared-shape skew-t case, applied record by record; each record consumes
one uniform and one Gamma draw in the same order as the shared-shape path, so a
constant shape vector reproduces the shared-shape chain draw for draw.
Only `model == :skewt` is supported.
"""
function update_latents!(lat, y1, μlin, σ21, model, ν::AbstractVector{Float64},
                         α::AbstractVector{Float64}, rng; skew_location::Symbol = :mean)
    model == :skewt ||
        error("update_latents!: record-indexed shapes are implemented for :skewt only")
    n = length(y1)
    @inbounds for i in 1:n
        νi = ν[i]; αi = α[i]
        δs = αi / sqrt(1 + αi^2)
        b  = skewt_b_nu(νi)
        ω2 = skewt_omega2_from_meas(σ21[i], νi, αi)
        Δ  = sqrt(ω2) * δs
        Γ  = ω2 * (1 - δs^2)
        A  = y1[i] - μlin[i] + Δ * b
        vpost = 1.0 / (1.0 + (Δ^2) / Γ)
        mpost = vpost * (Δ / Γ) * A

        wi = clamp(lat.w[i], W_MIN_SKEWT, W_MAX_SKEWT)
        sd_u = sqrt(vpost / wi)
        u0 = cdf(Normal(mpost, sd_u), 0.0)
        ut = quantile(Normal(mpost, sd_u), u0 + (1 - u0) * rand(rng))
        ut = isfinite(ut) ? clamp(ut, 0.0, mpost + U_SD_CAP_SKEWT * sd_u) :
                            max(mpost, 0.0)
        lat.tpos[i] = ut

        rate = (νi + ut^2 + (A - Δ * ut)^2 / Γ) / 2
        rate = clamp(rate, 1e-12, 1e12)
        lat.w[i] = clamp(rand(rng, Gamma(νi / 2 + 1.0, 1.0 / rate)),
                         W_MIN_SKEWT, W_MAX_SKEWT)
    end
end

"""
    _skewt_marginal_loglik(y1, μlin, σ21, idx, ν, α) -> Float64

Marginal (latents-integrated) skew-t log likelihood of the records in `idx`
under the mean-parameterised, variance-matched skew-t with shape `(ν, α)`.
"""
function _skewt_marginal_loglik(y1, μlin, σ21, idx, ν::Float64, α::Float64)
    δ = α / sqrt(1 + α^2)
    b = skewt_b_nu(ν)
    ll = 0.0
    @inbounds for i in idx
        ω = sqrt(skewt_omega2_from_meas(σ21[i], ν, α))
        ξ = μlin[i] - ω * δ * b
        ll += logpdf_skewt(y1[i], ξ, ω, ν, α)
    end
    return ll
end

"""
    update_shapes_mh!(shapes, νa, αa, idx_by_group, y1, μlin, σ21, spec, rng;
                      accept) -> nothing

One Metropolis sweep over the per-group skew-t shape pairs.  For each group a
joint symmetric random-walk proposal is made on `(log(ν - 2), α)` and accepted
against the marginal skew-t likelihood of that group's records plus the
Gaussian prior on the same coordinates.  On acceptance the record-indexed
shape vectors `νa`, `αa` are updated in place.  `accept` accumulates
`[accepted, proposed]` counts per group.
"""
function update_shapes_mh!(shapes::Dict{String,NTuple{2,Float64}},
                           νa::Vector{Float64}, αa::Vector{Float64},
                           idx_by_group::Dict{String,Vector{Int}},
                           y1, μlin, σ21, spec::FoldFitSpec, rng::AbstractRNG;
                           accept::Union{Nothing,Dict{String,Vector{Int}}} = nothing)
    for (t, idx) in idx_by_group
        ν0, α0 = shapes[t]
        z0 = log(ν0 - 2.0)
        l0 = _skewt_marginal_loglik(y1, μlin, σ21, idx, ν0, α0)
        lp0 = logpdf(Normal(spec.shape_prior_lognu_mean, spec.shape_prior_lognu_sd), z0) +
              logpdf(Normal(0.0, spec.shape_prior_alpha_sd), α0)

        z1 = z0 + spec.shape_mh_sd_lognu * randn(rng)
        α1 = α0 + spec.shape_mh_sd_alpha * randn(rng)
        ν1 = 2.0 + exp(z1)
        l1 = _skewt_marginal_loglik(y1, μlin, σ21, idx, ν1, α1)
        lp1 = logpdf(Normal(spec.shape_prior_lognu_mean, spec.shape_prior_lognu_sd), z1) +
              logpdf(Normal(0.0, spec.shape_prior_alpha_sd), α1)

        ok = isfinite(l1) && log(rand(rng)) < (l1 + lp1) - (l0 + lp0)
        if ok
            shapes[t] = (ν1, α1)
            @inbounds for i in idx
                νa[i] = ν1
                αa[i] = α1
            end
        end
        if accept !== nothing
            a = get!(accept, t, [0, 0])
            a[1] += ok ? 1 : 0
            a[2] += 1
        end
    end
    return nothing
end

function update_contamination!(lat, y, μlin, σ2, hs, spec, ν, α, rng)
    spec.model == :skewt &&
        error("contamination mixture is not derived for the skew-t")
    n = length(y)
    @inbounds for i in 1:n
        yi = y[i]
        μi = μlin[i]
        logp0 = log(1 - hs.π)
        logp1 = log(hs.π)

        if spec.model == :gaussian
            logp0 += logpdf(Normal(μi, sqrt(σ2[i])), yi)
            logp1 += logpdf(Normal(μi + spec.δ_shift, sqrt(spec.κ_scale * σ2[i])), yi)
        elseif spec.model == :t
            s20 = t_scale2_from_meas(σ2[i], ν)
            s21 = t_scale2_from_meas(spec.κ_scale * σ2[i], ν)
            logp0 += logpdf(TDist(ν), (yi - μi) / sqrt(s20)) - 0.5*log(s20)
            logp1 += logpdf(TDist(ν), (yi - (μi + spec.δ_shift)) / sqrt(s21)) - 0.5*log(s21)
        elseif spec.model == :skew
            ω0 = sqrt(skew_omega2_from_meas(σ2[i], α))
            ω1 = sqrt(skew_omega2_from_meas(spec.κ_scale * σ2[i], α))
            if spec.skew_location == :mean
                δs = α / sqrt(1 + α^2)
                mshift0 = ω0 * δs * sqrt(2/pi)
                mshift1 = ω1 * δs * sqrt(2/pi)
                ξ0 = μi - mshift0
                ξ1 = (μi + spec.δ_shift) - mshift1
            elseif spec.skew_location == :location
                ξ0 = μi
                ξ1 = μi + spec.δ_shift
            else
                error("skew_location must be :mean or :location")
            end
            logp0 += logpdf_skewnormal(yi, ξ0, ω0, α)
            logp1 += logpdf_skewnormal(yi, ξ1, ω1, α)
        end

        m = max(logp0, logp1)
        p1 = exp(logp1 - m) / (exp(logp0 - m) + exp(logp1 - m))
        lat.z[i] = Int(rand(rng, Bernoulli(p1)))
    end
end

# Small numerical helpers

@inline _loggamma(x::Float64) = first(logabsgamma(x))

function argmax_golden(f::Function, a::Float64, b::Float64;
                       tol::Float64 = 1e-4, maxiter::Int = 200)
    a < b || error("argmax_golden: require a < b")

    ϕ = (sqrt(5.0) - 1.0) / 2.0
    c = b - ϕ * (b - a)
    d = a + ϕ * (b - a)
    fc = f(c)
    fd = f(d)

    for _ in 1:maxiter
        if abs(b - a) <= tol * (1.0 + abs(c) + abs(d))
            break
        end
        if fc > fd
            b = d
            d = c
            fd = fc
            c = b - ϕ * (b - a)
            fc = f(c)
        else
            a = c
            c = d
            fc = fd
            d = a + ϕ * (b - a)
            fd = f(d)
        end
    end

    return fc > fd ? c : d
end

# Effective observation refresh from current latent state

"""
    compute_y1_sigma21!(y1, σ21, y, σ2, lat, spec)

Fill `y1` and `σ21` with contamination-adjusted observations and variances.
When `spec.contam` is false this is a plain copy; otherwise observations flagged
by `lat.z` are shifted and their variance inflated according to `spec.δ_shift`
and `spec.κ_scale`.
"""
function compute_y1_sigma21!(y1::Vector{Float64}, σ21::Vector{Float64},
                            y::Vector{Float64}, σ2::Vector{Float64},
                            lat::Latents, spec::FoldFitSpec)
    n = length(y)
    if !spec.contam
        @inbounds for i in 1:n
            y1[i]  = y[i]
            σ21[i] = σ2[i]
        end
        return nothing
    end
    @inbounds for i in 1:n
        if lat.z[i] == 1
            y1[i]  = y[i] - spec.δ_shift
            σ21[i] = spec.κ_scale * σ2[i]
        else
            y1[i]  = y[i]
            σ21[i] = σ2[i]
        end
    end
    return nothing
end

"""
    compute_var_eff!(var_eff, σ21, lat, spec, ν, α)

Fill `var_eff` with the effective per-observation variance used in the
Gaussian precision assembly, accounting for the current model:
- `:gaussian`: `var_eff[i] = σ21[i]`.
- `:t`: `var_eff[i] = scale²(σ21[i], ν) / w[i]` (scale-mixture weight divides variance).
- `:skew`: `var_eff[i] = ω²(σ21[i], α) · (1 − δ²)` (conditional variance of skew-normal).
"""
function compute_var_eff!(var_eff::Vector{Float64}, σ21::Vector{Float64},
                          lat::Latents, spec::FoldFitSpec, ν::Float64, α::Float64)
    n = length(σ21)
    if spec.model == :gaussian
        @inbounds for i in 1:n
            var_eff[i] = σ21[i]
        end
    elseif spec.model == :t
        @inbounds for i in 1:n
            var_eff[i] = t_scale2_from_meas(σ21[i], ν) / lat.w[i]
        end
    elseif spec.model == :skew
        δs = α / sqrt(1 + α^2)
        @inbounds for i in 1:n
            ω2 = skew_omega2_from_meas(σ21[i], α)
            var_eff[i] = ω2 * (1.0 - δs^2)
        end
    elseif spec.model == :skewt
        # Conditional variance given both latents: Γ / w.
        δs = α / sqrt(1 + α^2)
        @inbounds for i in 1:n
            ω2 = skewt_omega2_from_meas(σ21[i], ν, α)
            var_eff[i] = ω2 * (1.0 - δs^2) / max(lat.w[i], 1e-12)
        end
    else
        error("compute_var_eff!: unknown model $(spec.model)")
    end
    return nothing
end

"""
    compute_var_eff!(var_eff, σ21, lat, spec, ν::AbstractVector, α::AbstractVector)

Record-indexed shape variant of [`compute_var_eff!`](@ref) for the skew-t model:
the effective variance of record `i` uses its own shape pair `(ν[i], α[i])`.
"""
function compute_var_eff!(var_eff::Vector{Float64}, σ21::Vector{Float64},
                          lat::Latents, spec::FoldFitSpec,
                          ν::AbstractVector{Float64}, α::AbstractVector{Float64})
    spec.model == :skewt ||
        error("compute_var_eff!: record-indexed shapes are implemented for :skewt only")
    n = length(σ21)
    @inbounds for i in 1:n
        δs = α[i] / sqrt(1 + α[i]^2)
        ω2 = skewt_omega2_from_meas(σ21[i], ν[i], α[i])
        var_eff[i] = ω2 * (1.0 - δs^2) / max(lat.w[i], 1e-12)
    end
    return nothing
end

"""
    compute_y_eff!(y_eff, y1, σ21, lat, spec, α, ν)

Fill `y_eff` with the effective response used in the Gaussian precision assembly.
For `:gaussian` and `:t` models this is a plain copy of `y1`. For `:skew` the
truncated-normal auxiliary `lat.tpos` is used to centre the response:
`y_eff[i] = y1[i] - ω(σ21[i], α) · δ(α) · (tpos[i] − c)`, where
`c = √(2/π)` for mean-parameterized skew-normal fits and `c = 0` for
canonical-location fits.
"""
function compute_y_eff!(y_eff::Vector{Float64}, y1::Vector{Float64}, σ21::Vector{Float64},
                        lat::Latents, spec::FoldFitSpec, α::Float64, ν::Float64 = 5.0)
    n = length(y1)
    if spec.model == :gaussian || spec.model == :t
        @inbounds for i in 1:n
            y_eff[i] = y1[i]
        end
        return nothing
    elseif spec.model == :skew
        spec.skew_location in (:mean, :location) ||
            error("skew_location must be :mean or :location")
        δs = α / sqrt(1 + α^2)
        center = spec.skew_location == :mean ? sqrt(2 / pi) : 0.0
        @inbounds for i in 1:n
            ω = sqrt(skew_omega2_from_meas(σ21[i], α))
            y_eff[i] = y1[i] - ω * δs * (lat.tpos[i] - center)
        end
        return nothing
    elseif spec.model == :skewt
        # Same centring as the skew-normal, with b_ν in place of sqrt(2/pi)
        # because the half-normal latent is scaled by 1/sqrt(w).
        δs = α / sqrt(1 + α^2)
        b = skewt_b_nu(ν)
        @inbounds for i in 1:n
            ω = sqrt(skewt_omega2_from_meas(σ21[i], ν, α))
            y_eff[i] = y1[i] - ω * δs * (lat.tpos[i] - b)
        end
        return nothing
    else
        error("compute_y_eff!: unknown model $(spec.model)")
    end
end

"""
    compute_y_eff!(y_eff, y1, σ21, lat, spec, α::AbstractVector, ν::AbstractVector)

Record-indexed shape variant of [`compute_y_eff!`](@ref) for the skew-t model:
the centring of record `i` uses its own shape pair `(ν[i], α[i])`.
"""
function compute_y_eff!(y_eff::Vector{Float64}, y1::Vector{Float64}, σ21::Vector{Float64},
                        lat::Latents, spec::FoldFitSpec,
                        α::AbstractVector{Float64}, ν::AbstractVector{Float64})
    spec.model == :skewt ||
        error("compute_y_eff!: record-indexed shapes are implemented for :skewt only")
    n = length(y1)
    @inbounds for i in 1:n
        δs = α[i] / sqrt(1 + α[i]^2)
        b  = skewt_b_nu(ν[i])
        ω  = sqrt(skewt_omega2_from_meas(σ21[i], ν[i], α[i]))
        y_eff[i] = y1[i] - ω * δs * (lat.tpos[i] - b)
    end
    return nothing
end

# Hyperparameter updates

"""
    update_qprior_lambda!(hs, qsamps, eb, C0_inv_mul)

M-step for the prior mean `q_prior` and scaling hyperparameter `λ`.

Updates `hs.q_prior` as a precision-weighted average of the sample mean and
`eb.q0`, then updates `hs.λ` via the inverse-gamma posterior mode using the
mean Mahalanobis distance of `qsamps` from the updated prior mean.
"""
function update_qprior_lambda!(hs::HyperState, qsamps::AbstractMatrix,
                               eb::EBSpec, C0_inv_mul::Function)
    d, S = size(qsamps)

    μ_bar = vec(mean(qsamps, dims = 2))
    inv_λ = 1.0 / hs.λ
    inv_κ = 1.0 / eb.κ0
    hs.q_prior .= (inv_λ .* μ_bar .+ inv_κ .* eb.q0) ./ (inv_λ + inv_κ)

    D = 0.0
    tmp = zeros(Float64, d)
    @inbounds for j in 1:S
        tmp .= view(qsamps, :, j) .- hs.q_prior
        D += dot(tmp, C0_inv_mul(tmp))
    end
    D /= S
    hs.λ = (eb.bλ + 0.5 * D) / (eb.aλ + 1.0 + 0.5 * d)
    return nothing
end

"""
    update_tau_ig_from_stat!(Sbb, G, eb) -> Float64

Inverse-gamma M-step for `τ²`: returns the posterior mode
`(bτ + Sbb/2) / (aτ + 1 + G/2)` where `Sbb` is the (Rao-Blackwellised)
expected sum of squares of the random effects and `G` is the number of groups.
"""
@inline function update_tau_ig_from_stat!(Sbb::Float64, G::Int, eb::EBSpec)
    return (eb.bτ + 0.5 * Sbb) / (eb.aτ + 1.0 + 0.5 * G)
end

"""
    update_tau_pc!(Sbb, G, spec) -> Float64

Penalized-complexity M-step for `τ²`. With PC prior rate `λpc > 0`, solves
`λpc · τ³ + G · τ² - Sbb = 0` for `τ > 0` by bisection and returns `τ²`.
When `λpc == 0` the MLE `τ² = Sbb / G` is returned directly.
"""
function update_tau_pc!(Sbb::Float64, G::Int, spec::FoldFitSpec)
    G > 0 || return 0.0
    Sbb = max(Sbb, 0.0)

    λpc = max(spec.tau_pc_rate, 0.0)

    # If lambda_pc = 0, recover the MLE tau2 = Sbb / G.
    if λpc == 0.0
        return Sbb / G
    end

    # Solve λpc * τ^3 + G * τ^2 - Sbb = 0 for τ > 0 by bisection
    f(τ) = λpc * τ^3 + G * τ^2 - Sbb

    lo = 0.0
    hi = max(sqrt(Sbb / max(G, 1)), 1.0)
    while f(hi) < 0.0
        hi *= 2.0
        hi > 1e12 && error("update_tau_pc!: failed to bracket root")
    end

    for _ in 1:200
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0
            hi = mid
        else
            lo = mid
        end
        if abs(hi - lo) <= 1e-10 * (1.0 + abs(hi) + abs(lo))
            break
        end
    end

    τhat = 0.5 * (lo + hi)
    return τhat^2
end

# nu update for Student-t MCEM
#
# Uses the actual latent scale-mixture representation in the code:
#   w_i ~ Gamma(nu/2, scale = 2/nu),
#   y_i | q,b,w_i ~ N(mu_i, ((nu-2)/nu) sigma_i^2 / w_i).
#
# Objective (up to nu-independent constants), with Monte Carlo sufficient stats:
#   A = sum log w
#   B = sum w
#   C = sum w * r^2 / sigma2
#   N = total number of latent terms accumulated
#
# Q_t(nu) =
#   N * [ (nu/2) log(nu/2) - log Gamma(nu/2) ]
#   - (N/2) [ log(nu-2) - log nu ]
#   + ((nu-1)/2) A
#   - (nu/2) B
#   - (nu / (2(nu-2))) C
#   - penalty(nu)
#
# We maximize over the candidate nu-grid, with optional local refinement between
# neighboring grid points.

function update_nu_t_mcem!(ν_old::Float64,
                           sum_logw::Float64,
                           sum_w::Float64,
                           sum_wr2_over_sigma::Float64,
                           n_terms::Int,
                           spec::FoldFitSpec)

    if !spec.update_nu || n_terms <= 0
        return ν_old
    end

    ν_candidates = sort(filter(x -> x > 2.0 + 1e-6, unique(spec.ν_grid)))
    isempty(ν_candidates) && return ν_old

    ν_target = max(spec.nu_target, 2.0 + 1e-6)

    function Qν(ν::Float64)
        ν <= 2.0 + 1e-8 && return -Inf

        q =
            n_terms * ((ν / 2.0) * log(ν / 2.0) - _loggamma(ν / 2.0)) -
            (n_terms / 2.0) * (log(ν - 2.0) - log(ν)) +
            ((ν - 1.0) / 2.0) * sum_logw -
            (ν / 2.0) * sum_w -
            (ν / (2.0 * (ν - 2.0))) * sum_wr2_over_sigma

        if spec.nu_penalty > 0.0
            q -= 0.5 * spec.nu_penalty * (log(ν - 2.0) - log(ν_target - 2.0))^2
        end
        return q
    end

    vals = [Qν(ν) for ν in ν_candidates]
    k = argmax(vals)
    ν_hat = ν_candidates[k]

    if spec.refine_shape && 1 < k < length(ν_candidates)
        lo = ν_candidates[k - 1]
        hi = ν_candidates[k + 1]
        ν_hat = argmax_golden(Qν, lo, hi; tol = spec.shape_opt_tol)
    end

    return ν_hat
end

# alpha update for skew-normal MCEM
#
# Uses the latent-variable representation already used by compute_y_eff!/update_latents!:
#   y_eff = y - omega(alpha) delta(alpha) (t^+ - c)
# and conditional variance Gamma(alpha) = omega(alpha)^2 (1 - delta(alpha)^2).
# Here c = sqrt(2/pi) for the mean parameterization and c = 0 for the
# canonical-location parameterization.
#
# Given kept samples of:
#   residuals r_{is} = y_i^* - mu_{is},
#   centered latent c_{is} = t_{is}^+ - sqrt(2/pi),
#   effective sigma2_{is},
# we maximize the MCEM profile objective:
#
#   Q_skew(alpha) = -1/2 sum_{i,s} [ log Gamma_i(alpha) +
#                                (r_{is} - Delta_i(alpha) c_{is})^2 / Gamma_i(alpha) ]
#                  - penalty(alpha),
#
# where Delta_i(alpha) = sqrt(omega_i^2(alpha)) delta(alpha).
#
# To improve numerical stability, optimization is performed over eta = asinh(alpha),
# with alpha = sinh(eta). We first search over the supplied alpha-grid, then refine locally.

function update_alpha_skew_mcem!(α_old::Float64,
                                 resid_keep::AbstractMatrix,
                                 tcen_keep::AbstractMatrix,
                                 sigma_keep::AbstractMatrix,
                                 spec::FoldFitSpec)

    if !spec.update_alpha || isempty(resid_keep)
        return α_old
    end

    α_candidates = sort(unique(spec.α_grid))
    isempty(α_candidates) && return α_old

    function Qα(α::Float64)
        δ = α / sqrt(1.0 + α^2)
        denom = 1.0 - (2.0 / pi) * δ^2
        denom <= 0.0 && return -Inf

        q = 0.0
        n, S = size(resid_keep)

        @inbounds for s in 1:S
            for i in 1:n
                σ2i = sigma_keep[i, s]
                ω2 = σ2i / denom
                Δ = sqrt(ω2) * δ
                Γ = ω2 * (1.0 - δ^2)

                ri = resid_keep[i, s]
                ci = tcen_keep[i, s]
                u = ri - Δ * ci

                q += -0.5 * (log(Γ) + (u * u) / Γ)
            end
        end

        if spec.alpha_penalty > 0.0
            η = asinh(α)
            q -= 0.5 * spec.alpha_penalty * η^2
        end
        return q
    end

    vals = [Qα(α) for α in α_candidates]
    k = argmax(vals)
    α_hat = α_candidates[k]

    if spec.refine_shape && 1 < k < length(α_candidates)
        η_lo = asinh(α_candidates[k - 1])
        η_hi = asinh(α_candidates[k + 1])
        Qη(η) = Qα(sinh(η))
        η_hat = argmax_golden(Qη, η_lo, η_hi; tol = spec.shape_opt_tol)
        α_hat = sinh(η_hat)
    end

    return α_hat
end

@inline _stdnormal_pdf(x::Float64) = inv(sqrt(2.0 * pi)) * exp(-0.5 * x * x)

function _truncnorm_positive_mean(m::Float64, v::Float64)
    s = sqrt(max(v, eps(Float64)))
    a = -m / s
    denom = ccdf(Normal(), a)
    ratio = if denom > 1e-12
        _stdnormal_pdf(a) / denom
    elseif a > 0.0
        a + inv(max(a, 1e-6))
    else
        0.0
    end
    return max(0.0, m + s * ratio)
end

function _precision_cov_quadratic(tmp::Vector{Float64}, F)
    z = F.U' \ tmp
    return dot(z, z)
end

function _gaussian_linear_moments!(
    μlin::Vector{Float64},
    vlin::Vector{Float64},
    Arows::AbstractMatrix,
    μq::Vector{Float64},
    Fq,
    tmp::Vector{Float64}
)
    n, d = size(Arows)
    @inbounds for i in 1:n
        m = 0.0
        for j in 1:d
            aij = Arows[i, j]
            tmp[j] = aij
            m += aij * μq[j]
        end
        μlin[i] = m
        vlin[i] = max(_precision_cov_quadratic(tmp, Fq), 0.0)
    end
    return nothing
end

function _collapsed_linear_moments!(
    μlin::Vector{Float64},
    vlin::Vector{Float64},
    Arows::AbstractMatrix,
    cp::CollapsedPosterior,
    tmp::Vector{Float64}
)
    n, d = size(Arows)
    @inbounds for i in 1:n
        g = cp.gid[i]
        dinv = cp.D_bb_inv[g]
        row_mean = 0.0
        b_rhs = cp.r_b[g]
        for j in 1:d
            aij = Arows[i, j]
            pq = cp.P_bq[g, j]
            row_mean += aij * cp.μ_q[j]
            b_rhs -= pq * cp.μ_q[j]
            tmp[j] = aij - dinv * pq
        end
        μlin[i] = row_mean + dinv * b_rhs
        vlin[i] = max(_precision_cov_quadratic(tmp, cp.F_q) + dinv, 0.0)
    end
    return nothing
end

function _update_structured_latents!(
    lat::Latents,
    y1::Vector{Float64},
    σ21::Vector{Float64},
    μlin::Vector{Float64},
    vlin::Vector{Float64},
    spec::FoldFitSpec,
    ν::Float64,
    α::Float64
)
    spec.model == :skewt &&
        error("structured variational inference is not derived for the skew-t; " *
              "use method = :collapsed")
    n = length(y1)
    sum_logw = 0.0
    sum_w = 0.0
    sum_wr2_over_sigma = 0.0
    n_t_terms = 0

    if spec.model == :t
        shape = (ν + 1.0) / 2.0
        @inbounds for i in 1:n
            er2 = (y1[i] - μlin[i])^2 + max(vlin[i], 0.0)
            s2 = t_scale2_from_meas(σ21[i], ν)
            scale = 2.0 / (ν + er2 / s2)
            ew = max(shape * scale, 1e-12)
            lat.w[i] = ew
            sum_logw += digamma(shape) + log(scale)
            sum_w += ew
            sum_wr2_over_sigma += ew * er2 / σ21[i]
        end
        n_t_terms = n
    elseif spec.model == :skew
        spec.skew_location in (:mean, :location) ||
            error("skew_location must be :mean or :location")
        δs = α / sqrt(1.0 + α^2)
        mHN = sqrt(2.0 / pi)
        @inbounds for i in 1:n
            ω2 = skew_omega2_from_meas(σ21[i], α)
            Δ = sqrt(ω2) * δs
            Γ = ω2 * (1.0 - δs^2)
            denom = Γ
            vpost = 1.0 / (1.0 + (Δ * Δ) / denom)
            center_shift = spec.skew_location == :mean ? Δ * mHN : 0.0
            mpost = vpost * (Δ / denom) * (y1[i] - μlin[i] + center_shift)
            lat.tpos[i] = _truncnorm_positive_mean(mpost, vpost)
        end
    end

    return sum_logw, sum_w, sum_wr2_over_sigma, n_t_terms
end

function _structured_sample_posterior!(
    qkeep::Matrix{Float64},
    bkeep::Union{Matrix{Float64},Nothing},
    rng::AbstractRNG,
    cp::Union{CollapsedPosterior,Nothing},
    μq::Union{Vector{Float64},Nothing},
    Fq,
    hierarchical::Bool
)
    keep_target = size(qkeep, 2)
    if hierarchical
        cp === nothing && error("_structured_sample_posterior!: missing collapsed posterior")
        bwork = zeros(Float64, length(cp.groups))
        @inbounds for s in 1:keep_target
            q = sample_q_collapsed(rng, cp)
            qkeep[:, s] .= q
            sample_b_conditional!(bwork, rng, q, cp)
            bkeep[:, s] .= bwork
        end
    else
        μq === nothing && error("_structured_sample_posterior!: missing posterior mean")
        @inbounds for s in 1:keep_target
            qkeep[:, s] .= sample_from_precision(rng, μq, Fq)
        end
    end
    return nothing
end

"""
    _collapsed_tau_stat(q, cp::CollapsedPosterior) -> Float64

`E[||b||^2 | q] = sum_g (mu_g(q)^2 + v_g)` for one draw of `q`, the Rao-Blackwell statistic of
the between-source variance.  It is a function of its own so that its loops run on arrays of
known type.  Inside `fit_fold_mcem` the draw `q` has no type the compiler can infer
(`sample_q_collapsed` returns `Any`, because the field `F_q` of `CollapsedPosterior` is declared
with the abstract type `Cholesky`), and the same loops written there cost three dynamic calls for
every term.  The arithmetic and its order are those of the loops it replaces.
"""
function _collapsed_tau_stat(q::AbstractVector{Float64}, cp::CollapsedPosterior)
    P_bq = cp.P_bq; r_b = cp.r_b; D_bb_inv = cp.D_bb_inv
    G = length(D_bb_inv); d = length(q)
    (size(P_bq) == (G, d) && length(r_b) == G) ||
        throw(DimensionMismatch("_collapsed_tau_stat: q has $d entries and the collapsed posterior $(size(P_bq)) cross terms for $G sources"))
    sum_var = sum(D_bb_inv)
    sum_mu2 = 0.0
    @inbounds for g in 1:G
        residual = r_b[g]
        for j in 1:d
            residual -= P_bq[g, j] * q[j]
        end
        μ_bg = D_bb_inv[g] * residual
        sum_mu2 += μ_bg * μ_bg
    end
    return sum_mu2 + sum_var
end

function _structured_tau_stat(qkeep::Matrix{Float64}, cp::CollapsedPosterior)
    d, keep = size(qkeep)
    keep > 0 || return 0.0
    G = length(cp.D_bb_inv)
    sum_var = sum(cp.D_bb_inv)
    Sbb = 0.0
    @inbounds for s in 1:keep
        sum_mu2 = 0.0
        for g in 1:G
            residual = cp.r_b[g]
            for j in 1:d
                residual -= cp.P_bq[g, j] * qkeep[j, s]
            end
            μ_bg = cp.D_bb_inv[g] * residual
            sum_mu2 += μ_bg * μ_bg
        end
        Sbb += sum_mu2 + sum_var
    end
    return Sbb / keep
end

"""
    fit_fold_structured_vi(rng, A0, b0, Arows, y, σ2, grp, eb, hs_init, spec) -> FoldFit

Structured variational approximation for one fold. Robust latent variables are
replaced by deterministic conditional moments, while the Gaussian block for
`q` and optional source effects uses the same collapsed sparse posterior algebra
as the Rao-Blackwellised sampler. Final Gaussian draws are returned in the
standard `FoldFit` container so downstream predictive code is unchanged.
"""
function fit_fold_structured_vi(
    rng::AbstractRNG,
    A0::AbstractMatrix,
    b0::AbstractVector,
    Arows::AbstractMatrix,
    y::Vector{Float64},
    σ2::Vector{Float64},
    grp::Vector{String},
    eb::EBSpec,
    hs_init::HyperState,
    spec::FoldFitSpec
) :: FoldFit
    spec.contam && error("fit_fold_structured_vi: contamination mixtures require method=:collapsed")
    _validate_fold_fit_inputs("fit_fold_structured_vi", A0, b0, Arows, y, σ2, grp, eb, hs_init, spec)

    n, d = size(Arows)
    hierarchical = spec.hierarchical
    hs = HyperState(copy(hs_init.q_prior), hs_init.λ, hierarchical ? hs_init.τ² : 0.0, hs_init.π)
    grp_levels = hierarchical ? sort!(unique(grp)) : String[]
    G = length(grp_levels)

    lat = Latents(ones(Float64, n), zeros(Float64, n), zeros(Int, n))
    if spec.model == :skew
        fill!(lat.tpos, sqrt(2.0 / pi))
    elseif spec.model == :skewt
        # w starts at its prior mean of one, so the half-normal latent starts at
        # the same point as in the skew-normal case.
        fill!(lat.tpos, skewt_b_nu(max(spec.ν_init, 2.05)))
        fill!(lat.w, 1.0)
    end

    ν = (spec.model in (:t, :skewt)) ? max(spec.ν_init, 2.05) : 5.0
    α = (spec.model in (:skew, :skewt)) ? spec.α_init : 0.0
    C0_inv_mul = make_C0_inv_mul(eb.C0; allow_ridge = spec.allow_ridge)

    y1      = zeros(Float64, n)
    σ21     = zeros(Float64, n)
    y_eff   = zeros(Float64, n)
    var_eff = zeros(Float64, n)
    μlin    = zeros(Float64, n)
    vlin    = zeros(Float64, n)
    quadtmp = zeros(Float64, d)

    ci = hierarchical ? precompute_collapsed_invariants(Arows, grp, eb;
             allow_ridge=spec.allow_ridge) : nothing
    gi = !hierarchical ? precompute_gaussian_invariants(Arows, eb;
             allow_ridge=spec.allow_ridge) : nothing

    _P_qq      = zeros(Float64, d, d)
    _P_qb      = hierarchical ? zeros(Float64, d, G) : zeros(Float64, 0, 0)
    _D_bb_diag = hierarchical ? zeros(Float64, G) : Float64[]
    _r_b       = hierarchical ? zeros(Float64, G) : Float64[]
    _r_q_data  = zeros(Float64, d)
    _P_q_marg  = zeros(Float64, d, d)
    _r_q_marg  = zeros(Float64, d)
    _D_bb_inv  = hierarchical ? zeros(Float64, G) : Float64[]
    _P_bq      = hierarchical ? zeros(Float64, G, d) : zeros(Float64, 0, 0)
    _Prec      = zeros(Float64, d, d)
    _rhs       = zeros(Float64, d)

    keep_target = max(spec.n_keep_min, (spec.n_mcmc - spec.burnin) ÷ max(spec.thin, 1))
    keep_target >= 1 || error("fit_fold_structured_vi: keep_target < 1; increase n_mcmc or reduce burnin")
    last_qsamps = Matrix{Float64}(undef, d, 0)
    last_bsamps = hierarchical ? Matrix{Float64}(undef, G, 0) : nothing

    cp = nothing
    μq_post = nothing
    Fq_post = nothing

    for outer in 1:spec.n_outer
        λ_old, τ²_old, π_old = hs.λ, hs.τ², hs.π

        compute_y1_sigma21!(y1, σ21, y, σ2, lat, spec)
        compute_var_eff!(var_eff, σ21, lat, spec, ν, α)
        compute_y_eff!(y_eff, y1, σ21, lat, spec, α, ν)

        if hierarchical
            cp = posterior_q_collapsed_fast(
                A0, b0, Arows, y_eff, var_eff, ci, eb, hs;
                allow_ridge = spec.allow_ridge,
                P_qq = _P_qq, P_qb = _P_qb,
                D_bb_diag = _D_bb_diag, r_b = _r_b,
                r_q_data = _r_q_data,
                P_q_marg = _P_q_marg, r_q_marg = _r_q_marg,
                D_bb_inv = _D_bb_inv, P_bq = _P_bq
            )
            _collapsed_linear_moments!(μlin, vlin, Arows, cp, quadtmp)
        else
            μq_post, Fq_post = posterior_q_gaussian_fast(
                A0, b0, Arows, y_eff, var_eff, gi, eb, hs;
                allow_ridge = spec.allow_ridge,
                Prec = _Prec, rhs = _rhs
            )
            _gaussian_linear_moments!(μlin, vlin, Arows, μq_post, Fq_post, quadtmp)
        end

        sum_logw, sum_w, sum_wr2_over_sigma, n_t_terms =
            _update_structured_latents!(lat, y1, σ21, μlin, vlin, spec, ν, α)

        compute_var_eff!(var_eff, σ21, lat, spec, ν, α)
        compute_y_eff!(y_eff, y1, σ21, lat, spec, α, ν)

        if hierarchical
            cp = posterior_q_collapsed_fast(
                A0, b0, Arows, y_eff, var_eff, ci, eb, hs;
                allow_ridge = spec.allow_ridge,
                P_qq = _P_qq, P_qb = _P_qb,
                D_bb_diag = _D_bb_diag, r_b = _r_b,
                r_q_data = _r_q_data,
                P_q_marg = _P_q_marg, r_q_marg = _r_q_marg,
                D_bb_inv = _D_bb_inv, P_bq = _P_bq
            )
            _collapsed_linear_moments!(μlin, vlin, Arows, cp, quadtmp)
        else
            μq_post, Fq_post = posterior_q_gaussian_fast(
                A0, b0, Arows, y_eff, var_eff, gi, eb, hs;
                allow_ridge = spec.allow_ridge,
                Prec = _Prec, rhs = _rhs
            )
            _gaussian_linear_moments!(μlin, vlin, Arows, μq_post, Fq_post, quadtmp)
        end

        qkeep = zeros(Float64, d, keep_target)
        bkeep = hierarchical ? zeros(Float64, G, keep_target) : nothing
        _structured_sample_posterior!(qkeep, bkeep, rng, cp, μq_post, Fq_post, hierarchical)

        update_qprior_lambda!(hs, qkeep, eb, C0_inv_mul)

        if hierarchical
            Sbb_stat = _structured_tau_stat(qkeep, cp)
            if spec.tau_update == :pc
                hs.τ² = update_tau_pc!(Sbb_stat, G, spec)
            elseif spec.tau_update == :ig
                hs.τ² = update_tau_ig_from_stat!(Sbb_stat, G, eb)
            else
                error("Unknown tau_update = $(spec.tau_update); use :ig or :pc")
            end
        else
            hs.τ² = 0.0
        end

        if spec.model == :t && spec.update_nu
            ν = update_nu_t_mcem!(
                ν,
                sum_logw,
                sum_w,
                sum_wr2_over_sigma,
                n_t_terms,
                spec
            )
        elseif spec.model == :skew && spec.update_alpha
            center = spec.skew_location == :mean ? sqrt(2.0 / pi) : 0.0
            resid_keep = reshape(y1 .- μlin, n, 1)
            tcen_keep = reshape(lat.tpos .- center, n, 1)
            sigma_keep = reshape(copy(σ21), n, 1)
            α = update_alpha_skew_mcem!(α, resid_keep, tcen_keep, sigma_keep, spec)
        end

        last_qsamps = qkeep
        last_bsamps = bkeep

        if spec.adaptive_stop && outer ≥ spec.min_outer
            if check_convergence(λ_old, τ²_old, π_old, hs.λ, hs.τ², hs.π, spec.rtol_outer)
                break
            end
        end
    end

    τ²samps = hierarchical ? fill(hs.τ², size(last_qsamps, 2)) : fill(0.0, size(last_qsamps, 2))
    return FoldFit(
        last_qsamps,
        hierarchical ? last_bsamps : nothing,
        grp_levels,
        τ²samps,
        hs.π,
        ν,
        α,
        hs
    )
end

# Full fit_fold_mcem

"""
    fit_fold_mcem(rng, A0, b0, Arows, y, σ2, grp, eb, hs_init, spec; method=:collapsed) -> FoldFit

Run the full MCEM algorithm for one cross-validation fold and return a `FoldFit`.

Alternates between an inner Gibbs sampler (E-step) that draws from the joint
posterior of `(q, b, latents)` and an outer M-step that updates the empirical
Bayes hyperparameters `(q_prior, λ, τ², π)` and, for robust models, the shape
parameters `ν` (Student-t) or `α` (skew-normal).

# Arguments
- `A0`, `b0`: additional prior precision and mean term.
- `Arows`: `n × d` design matrix.
- `y`, `σ2`: observations and measurement variances.
- `grp`: group labels for the random effects (length `n`).
- `eb`: empirical Bayes prior specification.
- `hs_init`: initial hyperparameter state.
- `spec`: algorithmic configuration (`FoldFitSpec`).
- `method`: `:collapsed` (default, Rao-Blackwellised), `:original` (joint sampler),
  or `:structured_vi` (deterministic latent moments with the collapsed Gaussian block).

# Returns
A `FoldFit` containing posterior samples, group levels, hyperparameter traces,
and estimated shape parameters.
"""
function fit_fold_mcem(
    rng::AbstractRNG,
    A0::AbstractMatrix,
    b0::AbstractVector,
    Arows::AbstractMatrix,
    y::Vector{Float64},
    σ2::Vector{Float64},
    grp::Vector{String},
    eb::EBSpec,
    hs_init::HyperState,
    spec::FoldFitSpec;
    method::Symbol = :collapsed,   # :original, :collapsed, or :structured_vi
    ν_rec::Union{Nothing,Vector{Float64}} = nothing,
    α_rec::Union{Nothing,Vector{Float64}} = nothing,
    shape_groups::Union{Nothing,Vector{String}} = nothing
) :: FoldFit

    if method == :structured_vi
        (ν_rec === nothing && α_rec === nothing) ||
            error("fit_fold_mcem: record-indexed shapes are not implemented for :structured_vi")
        return fit_fold_structured_vi(rng, A0, b0, Arows, y, σ2, grp, eb, hs_init, spec)
    end

    _validate_fold_fit_inputs("fit_fold_mcem", A0, b0, Arows, y, σ2, grp, eb, hs_init, spec)

    n, d = size(Arows)
    hierarchical = spec.hierarchical

    # Record-indexed shape parameters (per quantity type).  Only meaningful for
    # the skew-t; the shapes are fixed inputs, so the MCEM shape updates must be
    # disabled.
    per_record_shape = (ν_rec !== nothing) || (α_rec !== nothing)
    if per_record_shape
        spec.model == :skewt ||
            error("fit_fold_mcem: record-indexed shapes are implemented for :skewt only")
        (ν_rec !== nothing && α_rec !== nothing) ||
            error("fit_fold_mcem: supply both ν_rec and α_rec, or neither")
        length(ν_rec) == n ||
            error("fit_fold_mcem: ν_rec must have length n = $n; got $(length(ν_rec))")
        length(α_rec) == n ||
            error("fit_fold_mcem: α_rec must have length n = $n; got $(length(α_rec))")
        all(x -> isfinite(x) && x > 2.0, ν_rec) ||
            error("fit_fold_mcem: every ν_rec entry must be finite and > 2")
        all(isfinite, α_rec) ||
            error("fit_fold_mcem: α_rec contains non-finite values")
        (spec.update_nu || spec.update_alpha) &&
            error("fit_fold_mcem: record-indexed shapes require update_nu = update_alpha = false")
    end

    # Fully Bayesian shape estimation: per-group MH on (ν, α) with the
    # augmentation latents integrated out of the acceptance ratio.
    mh_shapes = spec.update_shapes_mh
    idx_by_shape_group = Dict{String,Vector{Int}}()
    shape_levels = Dict{String,NTuple{2,Float64}}()
    shape_accept = Dict{String,Vector{Int}}()
    if mh_shapes
        spec.model == :skewt ||
            error("fit_fold_mcem: update_shapes_mh is implemented for :skewt only")
        per_record_shape ||
            error("fit_fold_mcem: update_shapes_mh needs ν_rec/α_rec as initial values")
        shape_groups !== nothing ||
            error("fit_fold_mcem: update_shapes_mh needs shape_groups (one label per record)")
        length(shape_groups) == n ||
            error("fit_fold_mcem: shape_groups must have length n = $n")
        spec.contam &&
            error("fit_fold_mcem: update_shapes_mh does not support the contamination mixture")
        for (i, t) in enumerate(shape_groups)
            push!(get!(idx_by_shape_group, t, Int[]), i)
        end
        for (t, idx) in idx_by_shape_group
            shape_levels[t] = (ν_rec[idx[1]], α_rec[idx[1]])
        end
    end

    posterior_refresh = max(spec.posterior_refresh, 1)

    hs = HyperState(copy(hs_init.q_prior), hs_init.λ, hierarchical ? hs_init.τ² : 0.0, hs_init.π)

    grp_levels = hierarchical ? sort!(unique(grp)) : String[]
    G = length(grp_levels)

    lat = Latents(ones(Float64, n), zeros(Float64, n), zeros(Int, n))

    ν = (spec.model in (:t, :skewt)) ? max(spec.ν_init, 2.05) : 5.0
    α = (spec.model in (:skew, :skewt)) ? spec.α_init : 0.0

    # Shape arguments threaded through the augmented updates: scalars for the
    # shared-shape families, vectors when shapes are record-indexed.  Julia
    # dispatch selects the matching methods of compute_var_eff!, compute_y_eff!
    # and update_latents!.  The MH shape sampler mutates the vectors, so it
    # gets its own copies.
    νa = per_record_shape ? (mh_shapes ? copy(ν_rec) : ν_rec) : ν
    αa = per_record_shape ? (mh_shapes ? copy(α_rec) : α_rec) : α

    C0_inv_mul = make_C0_inv_mul(eb.C0; allow_ridge = spec.allow_ridge)

    last_qsamps = Matrix{Float64}(undef, d, 0)
    last_bsamps = hierarchical ? Matrix{Float64}(undef, G, 0) : nothing
    last_shape_draws = nothing

    # Shared preallocations
    y1      = zeros(Float64, n)
    σ21     = zeros(Float64, n)
    y_eff   = zeros(Float64, n)
    var_eff = zeros(Float64, n)
    μlin    = zeros(Float64, n)

    gid_post = zeros(Int, n)
    b_work = hierarchical ? zeros(Float64, G) : Float64[]

    mHN = sqrt(2.0 / pi)
    if spec.model == :skew
        fill!(lat.tpos, mHN)
    elseif spec.model == :skewt
        if per_record_shape
            @inbounds for i in 1:n
                lat.tpos[i] = skewt_b_nu(ν_rec[i])
            end
        else
            fill!(lat.tpos, skewt_b_nu(ν))
        end
        fill!(lat.w, 1.0)
    end

    # Robust / contamination models require conditional refresh as latents evolve
    needs_aug_refresh = (spec.model != :gaussian) || spec.contam

    # Precompute invariants once: C0^{-1}, group maps, and sparse row supports.
    ci = nothing
    gi = nothing
    if method == :collapsed && hierarchical
        ci = precompute_collapsed_invariants(Arows, grp, eb;
                 allow_ridge=spec.allow_ridge)
    elseif method == :collapsed && !hierarchical
        gi = precompute_gaussian_invariants(Arows, eb;
                 allow_ridge=spec.allow_ridge)
    end

    # Preallocate work arrays for posterior_q_collapsed_fast
    _P_qq      = zeros(Float64, d, d)
    _P_qb      = hierarchical ? zeros(Float64, d, G) : zeros(Float64, 0, 0)
    _D_bb_diag = hierarchical ? zeros(Float64, G) : Float64[]
    _r_b       = hierarchical ? zeros(Float64, G) : Float64[]
    _r_q_data  = zeros(Float64, d)
    _P_q_marg  = zeros(Float64, d, d)
    _r_q_marg  = zeros(Float64, d)
    _D_bb_inv  = hierarchical ? zeros(Float64, G) : Float64[]
    _P_bq      = hierarchical ? zeros(Float64, G, d) : zeros(Float64, 0, 0)

    # Preallocate for non-hierarchical fast path
    _Prec = zeros(Float64, d, d)
    _rhs  = zeros(Float64, d)

    # Warm-start from the Gaussian posterior mean at w_i = 1.
    q_warm = zeros(Float64, d)
    b_warm = hierarchical ? zeros(Float64, G) : Float64[]
    try
        if hierarchical
            if ci !== nothing
                cp_init = posterior_q_collapsed_fast(
                    A0, b0, Arows, y, σ2, ci, eb, hs;
                    allow_ridge = spec.allow_ridge,
                    P_qq = _P_qq, P_qb = _P_qb,
                    D_bb_diag = _D_bb_diag, r_b = _r_b,
                    r_q_data = _r_q_data,
                    P_q_marg = _P_q_marg, r_q_marg = _r_q_marg,
                    D_bb_inv = _D_bb_inv, P_bq = _P_bq
                )
                q_warm .= cp_init.μ_q
                μ_b_init, _ = conditional_b_moments(q_warm, cp_init)
                b_warm .= μ_b_init
            else
                μq_init, μb_init, _, _, _ = posterior_qb_gaussian(
                    A0, b0, Arows, y, σ2, grp, eb, hs;
                    hierarchical=true, allow_ridge=spec.allow_ridge)
                q_warm .= μq_init
                if μb_init !== nothing; b_warm .= μb_init; end
            end
        else
            if gi !== nothing
                μq_init, _ = posterior_q_gaussian_fast(
                    A0, b0, Arows, y, σ2, gi, eb, hs;
                    allow_ridge=spec.allow_ridge, Prec=_Prec, rhs=_rhs)
                q_warm .= μq_init
            else
                μq_init, _ = posterior_q_gaussian(
                    A0, b0, Arows, y, σ2, eb, hs; allow_ridge=spec.allow_ridge)
                q_warm .= μq_init
            end
        end
    catch e
        @warn "Warm-start failed ($e); starting from prior mean"
        q_warm .= hs.q_prior
    end

    for outer in 1:spec.n_outer
        λ_old, τ²_old, π_old = hs.λ, hs.τ², hs.π

        keep_target = max(spec.n_keep_min, (spec.n_mcmc - spec.burnin) ÷ max(spec.thin, 1))
        keep_target >= 1 || error("fit_fold_mcem: keep_target < 1; increase n_mcmc or reduce burnin")

        qkeep = zeros(Float64, d, keep_target)
        bkeep = hierarchical ? zeros(Float64, G, keep_target) : zeros(Float64, 0, 0)
        shape_keep = mh_shapes ?
            Dict(t => zeros(Float64, 2, keep_target) for t in keys(idx_by_shape_group)) :
            nothing
        keep = 0

        # For Student-t ν update
        sum_logw = 0.0
        sum_w = 0.0
        sum_wr2_over_sigma = 0.0
        n_t_terms = 0

        # For skew α update
        need_alpha_stats = (spec.model == :skew) && spec.update_alpha
        resid_keep = need_alpha_stats ? zeros(Float64, n, keep_target) : zeros(Float64, 0, 0)
        tcen_keep  = need_alpha_stats ? zeros(Float64, n, keep_target) : zeros(Float64, 0, 0)
        sigma_keep = need_alpha_stats ? zeros(Float64, n, keep_target) : zeros(Float64, 0, 0)

        # For τ update under collapsed RB
        Sbb_rb_sum = 0.0
        Sbb_rb_n = 0

        # Conditional objects refreshed inside the inner loop
        μq_post = zeros(Float64, d)
        μb_post = hierarchical ? zeros(Float64, G) : Float64[]
        F_joint = nothing
        cp = nothing
        gmap_post = Dict{String,Int}()

        q = copy(q_warm)
        b = hierarchical ? copy(b_warm) : Float64[]

        for it in 1:spec.n_mcmc
            # (A) Current effective data from current latent state
            compute_y1_sigma21!(y1, σ21, y, σ2, lat, spec)
            compute_var_eff!(var_eff, σ21, lat, spec, νa, αa)
            compute_y_eff!(y_eff, y1, σ21, lat, spec, αa, νa)

            # (B) Refresh the Gaussian conditional when needed
            do_refresh =
                (!needs_aug_refresh && it == 1) ||
                (needs_aug_refresh && (it == 1 || posterior_refresh == 1 || (it % posterior_refresh == 1)))

            if method == :collapsed
                if hierarchical
                    if do_refresh || cp === nothing
                        cp = posterior_q_collapsed_fast(
                            A0, b0, Arows, y_eff, var_eff, ci, eb, hs;
                            allow_ridge = spec.allow_ridge,
                            P_qq = _P_qq, P_qb = _P_qb,
                            D_bb_diag = _D_bb_diag, r_b = _r_b,
                            r_q_data = _r_q_data,
                            P_q_marg = _P_q_marg, r_q_marg = _r_q_marg,
                            D_bb_inv = _D_bb_inv, P_bq = _P_bq
                        )
                        @inbounds for i in 1:n
                            gid_post[i] = cp.gid[i]
                        end
                    end

                    q = sample_q_collapsed(rng, cp)
                    sample_b_conditional!(b, rng, q, cp)

                else
                    if do_refresh || F_joint === nothing
                        if gi !== nothing
                            μq_post, Fq = posterior_q_gaussian_fast(
                                A0, b0, Arows, y_eff, var_eff, gi, eb, hs;
                                allow_ridge = spec.allow_ridge,
                                Prec = _Prec, rhs = _rhs
                            )
                        else
                            μq_post, Fq = posterior_q_gaussian(
                                A0, b0, Arows, y_eff, var_eff, eb, hs;
                                allow_ridge = spec.allow_ridge
                            )
                        end
                        F_joint = Fq
                    end
                    q = sample_from_precision(rng, μq_post, F_joint)
                end

            elseif method == :original
                if hierarchical
                    if do_refresh || F_joint === nothing
                        μq_post, μb_post, F, gmap_post, _ = posterior_qb_gaussian(
                            A0, b0, Arows, y_eff, var_eff, grp, eb, hs;
                            hierarchical = true,
                            allow_ridge = spec.allow_ridge
                        )
                        F_joint = F
                        @inbounds for i in 1:n
                            gid_post[i] = gmap_post[grp[i]]
                        end
                    end

                    θ = sample_from_precision(rng, vcat(μq_post, μb_post), F_joint)
                    q = θ[1:d]
                    b = θ[d+1:end]

                else
                    if do_refresh || F_joint === nothing
                        μq_post, Fq = posterior_q_gaussian(
                            A0, b0, Arows, y_eff, var_eff, eb, hs;
                            allow_ridge = spec.allow_ridge
                        )
                        F_joint = Fq
                    end
                    q = sample_from_precision(rng, μq_post, F_joint)
                end
            else
                error("fit_fold_mcem: unknown method = $method; use :original, :collapsed, or :structured_vi")
            end

            # (C) Current linear predictor
            mul!(μlin, Arows, q)
            if hierarchical
                @inbounds for i in 1:n
                    μlin[i] += b[gid_post[i]]
                end
            end

            # (D) Update the shape parameters (marginalizing the latents), then
            # the latent variables conditional on current (q, b) and shapes.
            # The order matters: (shapes, latents) form one valid blocked draw.
            if mh_shapes
                update_shapes_mh!(shape_levels, νa, αa, idx_by_shape_group,
                                  y1, μlin, σ21, spec, rng; accept = shape_accept)
            end
            update_latents!(lat, y1, μlin, σ21, spec.model, νa, αa, rng;
                skew_location = spec.skew_location)

            if spec.contam
                update_contamination!(lat, y, μlin, σ2, hs, spec, ν, α, rng)
                zsum = sum(lat.z)
                hs.π = (eb.aπ + zsum) / (eb.aπ + eb.bπ + n)
            end

            # (E) Store kept samples and accumulate shape / τ statistics
            if it > spec.burnin && ((it - spec.burnin) % max(spec.thin, 1) == 0)
                keep += 1
                if keep > keep_target
                    break
                end

                qkeep[:, keep] .= q
                if hierarchical
                    bkeep[:, keep] .= b
                end
                if mh_shapes
                    for (t, m) in shape_keep
                        m[1, keep], m[2, keep] = shape_levels[t]
                    end
                end

                # Recompute current y1,σ21 after possible contamination update so
                # the stored state statistics reflect the current chain state.
                compute_y1_sigma21!(y1, σ21, y, σ2, lat, spec)

                if spec.model == :t && spec.update_nu
                    @inbounds for i in 1:n
                        ri = y1[i] - μlin[i]
                        wi = lat.w[i]
                        sum_logw += log(wi)
                        sum_w += wi
                        sum_wr2_over_sigma += wi * (ri * ri) / σ21[i]
                    end
                    n_t_terms += n
                elseif need_alpha_stats
                    center = spec.skew_location == :mean ? mHN : 0.0
                    @inbounds for i in 1:n
                        resid_keep[i, keep] = y1[i] - μlin[i]
                        tcen_keep[i, keep] = lat.tpos[i] - center
                        sigma_keep[i, keep] = σ21[i]
                    end
                end

                # Collapsed Rao-Blackwell tau statistic.
                if hierarchical && method == :collapsed
                    # E[||b||^2 | q, current conditional] = sum_g (mu_g(q)^2 + v_g).
                    # Exact when posterior_refresh == 1; approximate otherwise.
                    Sbb_rb_sum += _collapsed_tau_stat(q, cp)
                    Sbb_rb_n += 1
                end
            end
        end # inner loop

        qkeep_use = qkeep[:, 1:keep]
        bkeep_use = hierarchical ? bkeep[:, 1:keep] : nothing

        # M-step for q_prior, lambda, tau2, nu, and alpha.
        update_qprior_lambda!(hs, qkeep_use, eb, C0_inv_mul)

        if hierarchical
            Sbb_stat = 0.0
            if method == :collapsed
                Sbb_stat = Sbb_rb_sum / max(Sbb_rb_n, 1)
            else
                # Monte Carlo statistic from sampled random effects.
                @inbounds for s in 1:keep
                    Sbb_stat += sum(abs2, view(bkeep_use, :, s))
                end
                Sbb_stat /= max(keep, 1)
            end

            if spec.tau_update == :pc
                hs.τ² = update_tau_pc!(Sbb_stat, G, spec)
            elseif spec.tau_update == :ig
                hs.τ² = update_tau_ig_from_stat!(Sbb_stat, G, eb)
            else
                error("Unknown tau_update = $(spec.tau_update); use :ig or :pc")
            end
        else
            hs.τ² = 0.0
        end

        if spec.model == :t && spec.update_nu
            ν = update_nu_t_mcem!(
                ν,
                sum_logw,
                sum_w,
                sum_wr2_over_sigma,
                n_t_terms,
                spec
            )
        elseif spec.model == :skew && spec.update_alpha
            α = update_alpha_skew_mcem!(
                α,
                resid_keep[:, 1:keep],
                tcen_keep[:, 1:keep],
                sigma_keep[:, 1:keep],
                spec
            )
        end

        # The shape arguments handed to compute_var_eff!, compute_y_eff! and
        # update_latents! are bound once before the outer loop, so without this
        # refresh the M-step updates above never reach the sampler and every
        # outer iteration keeps augmenting at the initial shape (a :t fit with
        # update_nu stays at ν_init, a :skew fit with update_alpha at α_init).
        # The per-record path is excluded: there the shapes are fixed inputs, or
        # are mutated in place by the MH sampler.
        if !per_record_shape
            νa = ν
            αa = α
        end

        last_qsamps = qkeep_use
        last_bsamps = bkeep_use
        if mh_shapes
            last_shape_draws = Dict(t => m[:, 1:keep] for (t, m) in shape_keep)
        end

        # Propagate posterior mean as warm-start for next outer iteration
        if keep > 0
            q_warm .= vec(mean(qkeep_use, dims=2))
            if hierarchical && bkeep_use !== nothing
                b_warm .= vec(mean(bkeep_use, dims=2))
            end
        end

        # Adaptive outer stopping.
        if spec.adaptive_stop && outer ≥ spec.min_outer
            if check_convergence(λ_old, τ²_old, π_old, hs.λ, hs.τ², hs.π, spec.rtol_outer)
                break
            end
        end
    end # outer loop

    τ²samps = hierarchical ? fill(hs.τ², size(last_qsamps, 2)) : fill(0.0, size(last_qsamps, 2))

    return FoldFit(
        last_qsamps,
        hierarchical ? last_bsamps : nothing,
        grp_levels,
        τ²samps,
        hs.π,
        ν,
        α,
        hs,
        mh_shapes ? last_shape_draws : nothing
    )
end
