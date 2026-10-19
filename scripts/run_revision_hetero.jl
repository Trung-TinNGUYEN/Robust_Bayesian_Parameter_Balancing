# run_revision_hetero.jl
#
# Learned record-level uncertainties, added for the revision.
#
# Every analysis so far assigns each record the same uncertainty, either the
# half-decade convention or one estimated global scale on it.  The curated
# data contradict that: pooled within-replicate dispersion differs by up to a
# factor of two between quantity types (shikimate kcat 2.23 against KM 4.41
# natural-log units) and further between enzymes.  A single scale therefore
# has to be simultaneously too wide for the well-measured records and too
# narrow for the badly measured ones, which no marginal coverage number can
# reveal.
#
# This driver replaces the single scale with a shrunk hierarchy of scales
# estimated from the replication already present in BRENDA.  Records that
# share an organism AND a derived-quantity row share an identical model mean
# by construction (identical row of the dependency matrix, identical organism
# offset), so their deviations from the group mean are pure record error and
# the scale is identified WITHOUT fitting the model:
#
#     c_s = sum_g sum_{i in g, s(i)=s} (y_i - ybar_g)^2
#           / sum_g sum_{i in g, s(i)=s} sigma_conv^2 (1 - 1/n_g).
#
# Constructions compared, all sharing the same sampler, protocol and folds:
#
#   convention   the half-decade convention (c = 1), for reference;
#   global_cell  one scale from the within-cell iteration of
#                run_revision_sigmacal.jl (the current recommended default);
#   global       one scale from the replication estimator above (isolates the
#                change of estimator from the change of resolution);
#   type         one scale per quantity type (kcat, KM);
#   enzyme       one scale per enzyme-by-type stratum, shrunk to its type;
#   quantity     one scale per derived-quantity row, shrunk to its
#                enzyme-by-type stratum, which is shrunk to its type;
#   tau_type     pooled replication scale, per-type between-organism variance
#   joint2m      the two adopted refinements together: per-type between-organism
#                variance (as tau_eb) with the mean-matched two-class record
#                population (as scale_mix2m); shares the global fit
#                from a fit-free quantity-row moment estimator (degenerate
#                where the record scale is large; kept for the record);
#   both         per-type replication scale plus the same fit-free tau;
#   tau_eb       pooled replication scale, per-type between-organism variance
#                by the Paule-Mandel marginal moment estimator on the
#                fitted cell residuals (unattenuated, unlike the posterior
#                second moment, which shrinks toward the pooled variance);
#   joint        per-type replication scale AND per-type model-based tau:
#                both variance components stratified, each estimated where it
#                is identified;
#   scale_mix    the record error scale treated as a RANDOM EFFECT per
#                replicate group: each group's scale estimate is a noisy
#                chi-square observation of a latent group scale, the latent
#                scales are given an inverse-gamma population whose shape is
#                identified from the spread of the estimates beyond their
#                chi-square noise, and the predictive record error becomes the
#                variance-matched scale mixture, Student-t-like tails with
#                degrees of freedom 2a.  This is what "per-record errors"
#                amount to when they must be learned from replication rather
#                than reported by the database.
#   scale_mix13  the same population with the shape floor lowered to 1.3, and
#   scale_mix20  raised to 2.0: the sensitivity of the far-tail repair to the
#                floor, which the curated data press against on every pathway;
#   scale_lnmix  a LOG-NORMAL population of group scales with the unclamped
#                excess log-variance as its spread and the mean pinned to the
#                pooled scale: every moment is finite, so no floor is needed,
#                and agreement with scale_mix shows the repair is not a
#                property of the floor;
#   scale_mixml  the inverse-gamma population fitted by MAXIMUM LIKELIHOOD on
#                the replicate statistics (each group's scale estimate is a
#                scaled chi-square given its latent scale, so the marginal of
#                the estimate under the population is a beta-prime law), which
#                unlike the log-moment estimator is not driven by the few
#                groups whose records nearly coincide;
#   scale_mix2   a TWO-COMPONENT discrete scale population (a tight and a
#                loose class of replicate groups) fitted by EM on the same
#                statistics: the alternative a shape estimate pinned to a
#                boundary usually calls for.  The three population models are
#                compared by their log-likelihood on the replicate statistics.
#   tau_post     the single replication scale with the between-organism
#                variance INTEGRATED OVER ITS POSTERIOR rather than plugged
#                in: tau2 is drawn per prediction draw from the scaled
#                inverse-chi-square posterior with as many degrees of freedom
#                as the training fold has organism-by-type cells, which is the
#                plug-in underdispersion a reviewer would suspect behind the
#                residual record-averaged gap.
#
# Shrinkage is empirical-Bayes on the log scale: a raw stratum estimate with
# df degrees of freedom has sampling variance approximately 2/df, the
# between-stratum variance psi^2 is estimated by moments at each level, and
# the posterior weight is psi^2 / (psi^2 + 2/df).  Strata unseen or
# underdetermined in a training fold fall back up the hierarchy, so a
# held-out organism always receives a well-defined scale.
#
# Everything is estimated on the training fold, so the leave-one-source-out
# protocol remains leakage-free.
#
# Reported per construction: record- and organism-averaged coverage, mean
# width, the 95% interval (Winkler) score, point accuracy, and CONDITIONAL
# coverage: coverage within each quantity type and each enzyme-by-type
# stratum, summarised by the worst and the mean absolute deviation from
# nominal.  Marginal coverage can be nominal while conditional coverage is
# not, and that is exactly what one global scale produces.
#
# Outputs, written under output/<pathway>/results/revision/:
#   hetero_summary_<pathway>.csv     one row per construction
#   hetero_coverage_<pathway>.csv    coverage at every nominal level
#   hetero_strata_<pathway>.csv      per-stratum scales and coverage
#
# Environment controls:
#   PB_REV_PATHWAYS  subset of {glycolysis,tca,pentose_phosphate,shikimate}
#   PB_REV_SIGMACAL_ITERS  EB iterations for the within-cell scale (default 4)
#   PB_REV_OUTER_MULT  multiplies the number of outer EM iterations of every fit
#                    (default 1, which is the published 3, or 4 on the TCA cycle),
#                    for the EM-iteration sensitivity run, which the shorter article does not report
#   PB_REV_METHODS   comma-separated subset of the constructions (default all)
#   PB_PRIOR_SCALE   multiplies the prior covariance C_prior (default 1.0), for
#                    the prior-sensitivity runs; the empirical-Bayes scale
#                    lambda is still estimated on top of it
#   PB_REV_SUFFIX    suffix for the output file names (default none)
#   PB_REV_SEED      master seed (default 20260805); a second seed gives the
#                    Monte Carlo standard error of every reported coverage
#   PB_REV_EXCLUDE_MUTANTS  drop records whose BRENDA comment names a
#                    site-directed mutant and not the wild type, for the
#                    enzyme-form sensitivity analysis (mutant_flags.jl)

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
include(joinpath(dirname(@__DIR__), "src", "mutant_flags.jl"))
include(joinpath(dirname(@__DIR__), "src", "experiment_runners.jl"))

const REV_LEVELS = vcat(0.01, collect(0.05:0.05:0.95), 0.99)
# Matches run_revision_sigmacal.jl and run_revision_conformal.jl, so the
# global_cell construction reproduces the published within-cell results exactly.
const REV_SEED = parse(Int, get(ENV, "PB_REV_SEED", "20260805"))
const REV_PATHWAYS = Symbol.(strip.(split(get(ENV, "PB_REV_PATHWAYS",
    "glycolysis,tca,pentose_phosphate,shikimate"), ",")))
const SIGMACAL_ITERS = parse(Int, get(ENV, "PB_REV_SIGMACAL_ITERS", "4"))
const OUTER_MULT = parse(Int, get(ENV, "PB_REV_OUTER_MULT", "1"))
const C_MIN, C_MAX = 0.25, 40.0
const N_PRED = 800
const DF_MIN = 4          # minimum replication df for a stratum to be used
const MIN_STRAT_REC = 20  # minimum held-out records for a conditional-coverage stratum

const ALL_METHODS = ["convention", "global_cell", "global", "type", "enzyme", "quantity",
                     "tau_type", "both", "tau_eb", "joint", "scale_mix",
                     "scale_mix13", "scale_mix20", "scale_lnmix", "scale_mixml",
                     "scale_mix2", "scale_mix2m", "joint2m", "tau_post",
                     "tau_rel", "joint2m_rel"]
const METHODS = let sel = strip.(split(get(ENV, "PB_REV_METHODS", ""), ","))
    isempty(get(ENV, "PB_REV_METHODS", "")) ? ALL_METHODS : filter(in(sel), ALL_METHODS)
end
const PRIOR_SCALE = parse(Float64, get(ENV, "PB_PRIOR_SCALE", "1.0"))
const OUT_SUFFIX = get(ENV, "PB_REV_SUFFIX", "")
OUTER_MULT == 1 || !isempty(OUT_SUFFIX) || error("PB_REV_OUTER_MULT is not 1: set PB_REV_SUFFIX as well, so that the published outputs are kept")
const EXCLUDE_MUTANTS = haskey(ENV, "PB_REV_EXCLUDE_MUTANTS")
const TAU_MIN, TAU_MAX = 0.02, 20.0

# Which per-record scale each construction uses, and which fit it therefore
# shares.  `tau_type` reuses the `global` fit and `both` reuses the `type` fit:
# they differ from those only in the between-organism variance used at the
# predictive stage, so no additional sampler run is needed.
const SCALE_OF = Dict("convention" => "convention", "global_cell" => "global_cell",
                      "global" => "global", "type" => "type", "enzyme" => "enzyme",
                      "quantity" => "quantity", "tau_type" => "global", "both" => "type",
                      "tau_eb" => "global", "joint" => "type",
                      "scale_mix" => "global", "scale_mix13" => "global",
                      "scale_mix20" => "global", "scale_lnmix" => "global",
                      "scale_mixml" => "global", "scale_mix2" => "global",
                      "scale_mix2m" => "global", "joint2m" => "global",
                      "tau_post" => "global", "tau_rel" => "global",
                      "joint2m_rel" => "global")
const STRAT_TAU = Set(["tau_type", "both"])
const STRAT_TAU_EB = Set(["tau_eb", "joint", "joint2m"])
# The released specification: per-type variance with the pooled value as the fallback
# wherever the per-type estimate leaves [0.2, 5] times the pooled one (hierarchy_utils.jl).
const STRAT_TAU_REL = Set(["tau_rel", "joint2m_rel"])

"""
    _inv_trigamma(y) -> x  with  polygamma(1, x) = y
"""
function _inv_trigamma(y::Float64)
    x = y > 1.0 ? 1.0 / sqrt(y) : 0.5 + 1.0 / y
    for _ in 1:60
        f = polygamma(1, x) - y
        x = clamp(x - f / polygamma(2, x), 1e-3, 1e7)
    end
    x
end

"""
    scale_population(y, sigma2, repkey, idx, c_global; floor = 1.6) -> (a, b, a_raw, v)

Inverse-gamma population of the replicate-group error scales.  Each group with
n_g >= 2 records supplies chat_g with df_g = n_g - 1 and, conditionally,
log chat_g = log c_g + log(chi^2_df / df), so across groups

  Var(log chat) = Var(log c) + E[psi'(df/2)] + Var(psi(df/2) - log(df/2)),

and the excess of the observed spread over the chi-square terms identifies the
population's log-scale variance psi'(a).  The scale b is set so the population
mean b/(a-1) equals the fit-free global scale: the mixture is variance-matched
and differs from the single scale only in its tails.  A spread at or below the
chi-square noise returns a -> large, the single-scale limit.

Also returned: the UNCLAMPED shape `a_raw` implied by the excess log-variance
`v` (Inf when the excess is nil), so that a shape pinned to the floor is
reported as such rather than hidden by the clamp.
"""
function _group_log_scales(y, sigma2, repkey::Vector{String}, idx)
    bygroup = Dict{String,Vector{Int}}()
    for i in idx
        push!(get!(bygroup, repkey[i], Int[]), i)
    end
    logs = Float64[]; dfs = Float64[]
    for (_, ig) in bygroup
        ng = length(ig); ng >= 2 || continue
        ybar = mean(y[i] for i in ig)
        S = sum((y[i] - ybar)^2 for i in ig)
        D = sum(sigma2[i] * (1.0 - 1.0 / ng) for i in ig)
        D > 0 || continue
        push!(logs, log(clamp(S / D, 1e-8, 1e8)))
        push!(dfs, ng - 1.0)
    end
    logs, dfs
end

function _excess_logvar(logs::Vector{Float64}, dfs::Vector{Float64})
    noise = mean(polygamma(1, df / 2) for df in dfs)
    bias = var([digamma(df / 2) - log(df / 2) for df in dfs])
    var(logs) - noise - bias
end

function scale_population(y, sigma2, repkey::Vector{String}, idx, c_global::Float64;
                          floor::Float64 = 1.6)
    logs, dfs = _group_log_scales(y, sigma2, repkey, idx)
    length(logs) >= 20 || return (200.0, c_global * 199.0, Inf, 0.0)
    v = _excess_logvar(logs, dfs)
    v <= 1e-4 && return (200.0, c_global * 199.0, Inf, v)
    a_raw = _inv_trigamma(v)
    a = clamp(a_raw, floor, 200.0)
    (a, c_global * (a - 1.0), a_raw, v)
end

"""
    scale_population_boot(y, sigma2, repkey, idx; B = 200, seed = 1) -> (a_lo, a_hi)

Percentile interval for the unclamped shape from `B` resamples of the
replicate groups.  Uses its own RNG so that no other draw in the driver moves.
"""
function scale_population_boot(y, sigma2, repkey::Vector{String}, idx; B::Int = 200,
                               seed::Int = 1)
    logs, dfs = _group_log_scales(y, sigma2, repkey, idx)
    m = length(logs)
    m >= 20 || return (NaN, NaN)
    rng = MersenneTwister(seed)
    as = Float64[]
    for _ in 1:B
        ii = rand(rng, 1:m, m)
        v = _excess_logvar(logs[ii], dfs[ii])
        push!(as, v > 1e-4 ? _inv_trigamma(v) : Inf)
    end
    (quantile(as, 0.025), quantile(as, 0.975))
end

"""
    _group_ratio_stats(y, sigma2, repkey, idx) -> (r, df)

Replicate-group scale estimates r_g = S_g / D_g and their degrees of freedom.
Conditionally on the group's latent scale c_g,  r_g * df_g / c_g ~ chi^2_df,
i.e. r_g ~ Gamma(df_g/2, 2 c_g/df_g).
"""
function _group_ratio_stats(y, sigma2, repkey::Vector{String}, idx)
    bygroup = Dict{String,Vector{Int}}()
    for i in idx
        push!(get!(bygroup, repkey[i], Int[]), i)
    end
    r = Float64[]; dfs = Float64[]
    for (_, ig) in bygroup
        ng = length(ig); ng >= 2 || continue
        ybar = mean(y[i] for i in ig)
        S = sum((y[i] - ybar)^2 for i in ig)
        D = sum(sigma2[i] * (1.0 - 1.0 / ng) for i in ig)
        D > 0 || continue
        push!(r, max(S / D, 1e-10)); push!(dfs, ng - 1.0)
    end
    r, dfs
end

_ll_single(r, dfs, c) = sum(logpdf(Gamma(dfs[g] / 2, 2c / dfs[g]), r[g]) for g in eachindex(r))

"Marginal log-likelihood of the group statistics under c_g ~ InverseGamma(a, b)."
function _ll_ig(r, dfs, a, b)
    # x = r df / (2b) ~ BetaPrime(df/2, a), with Jacobian df/(2b)
    s = 0.0
    for g in eachindex(r)
        x = r[g] * dfs[g] / (2b)
        s += logpdf(BetaPrime(dfs[g] / 2, a), x) + log(dfs[g] / (2b))
    end
    s
end

"""
    scale_population_fit(y, sigma2, repkey, idx) -> NamedTuple

Three population models for the replicate-group scales, each fitted by maximum
likelihood on the same statistics and compared by log-likelihood:

  single   one scale c for every group                  (1 parameter)
  ig       c_g ~ InverseGamma(a, b)                      (2 parameters)
  two      c_g = c1 with probability pi, else c2         (3 parameters)

The inverse-gamma fit is by profile likelihood over a log-grid for the shape
with a golden-section search for the scale; the two-component fit by EM with
closed-form M steps (the weighted MLE of a Gamma scale).
"""
function scale_population_fit(y, sigma2, repkey::Vector{String}, idx)
    r, dfs = _group_ratio_stats(y, sigma2, repkey, idx)
    m = length(r)
    m >= 20 || return (c = NaN, ll_single = NaN, a = NaN, b = NaN, ll_ig = NaN,
                       pi1 = NaN, c1 = NaN, c2 = NaN, ll_two = NaN, n_groups = m)
    c_hat = sum(dfs .* r) / sum(dfs)
    ll_s = _ll_single(r, dfs, c_hat)
    # inverse-gamma: profile over log a
    function best_b(a)
        lo, hi = log(1e-3), log(1e4)
        φ = (sqrt(5) - 1) / 2
        x1 = hi - φ * (hi - lo); x2 = lo + φ * (hi - lo)
        f1 = _ll_ig(r, dfs, a, exp(x1)); f2 = _ll_ig(r, dfs, a, exp(x2))
        for _ in 1:60
            if f1 < f2
                lo = x1; x1 = x2; f1 = f2; x2 = lo + φ * (hi - lo); f2 = _ll_ig(r, dfs, a, exp(x2))
            else
                hi = x2; x2 = x1; f2 = f1; x1 = hi - φ * (hi - lo); f1 = _ll_ig(r, dfs, a, exp(x1))
            end
        end
        bb = exp(0.5 * (lo + hi)); (bb, _ll_ig(r, dfs, a, bb))
    end
    a_best, b_best, ll_i = NaN, NaN, -Inf
    for la in range(log(0.2), log(100.0); length = 80)
        bb, ll = best_b(exp(la))
        if ll > ll_i
            a_best, b_best, ll_i = exp(la), bb, ll
        end
    end
    # two-component EM
    c1 = quantile(r, 0.25); c2 = quantile(r, 0.75); pi1 = 0.5
    c1 == c2 && (c2 = 2c1 + 1e-3)
    w = Vector{Float64}(undef, m)
    for _ in 1:200
        for g in 1:m
            l1 = log(pi1) + logpdf(Gamma(dfs[g] / 2, 2c1 / dfs[g]), r[g])
            l2 = log(1 - pi1) + logpdf(Gamma(dfs[g] / 2, 2c2 / dfs[g]), r[g])
            mx = max(l1, l2)
            w[g] = exp(l1 - mx) / (exp(l1 - mx) + exp(l2 - mx))
        end
        c1 = sum(w .* dfs .* r) / max(sum(w .* dfs), 1e-12)
        c2 = sum((1 .- w) .* dfs .* r) / max(sum((1 .- w) .* dfs), 1e-12)
        pi1 = clamp(mean(w), 1e-3, 1 - 1e-3)
    end
    c1 > c2 && ((c1, c2, pi1) = (c2, c1, 1 - pi1))
    ll_t = sum(begin
        l1 = log(pi1) + logpdf(Gamma(dfs[g] / 2, 2c1 / dfs[g]), r[g])
        l2 = log(1 - pi1) + logpdf(Gamma(dfs[g] / 2, 2c2 / dfs[g]), r[g])
        mx = max(l1, l2); mx + log(exp(l1 - mx) + exp(l2 - mx))
    end for g in 1:m)
    (c = c_hat, ll_single = ll_s, a = a_best, b = b_best, ll_ig = ll_i,
     pi1 = pi1, c1 = c1, c2 = c2, ll_two = ll_t, n_groups = m)
end

function posterior_q_moments(fit::FoldFit, Arows_tr::AbstractMatrix)
    M = Arows_tr * fit.qsamps
    return vec(mean(M, dims = 2)), vec(var(M, dims = 2))
end

"""
    c_update_within(y, m, sigma2, cells, idx) -> c

The within-cell moment estimator of run_revision_sigmacal.jl, kept verbatim so
that the `global_cell` construction reproduces the published default.
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

"""
    replication_stats(y, sigma2, repkey, strat, idx) -> Dict(stratum => (S, D, df))

Sufficient statistics of the replication estimator restricted to each stratum:
the sum of squared deviations from the replicate-group mean, the matching sum
of convention variances scaled by (1 - 1/n_g), and the degrees of freedom.
Replicate groups are records sharing an organism and a derived-quantity row,
which share an identical model mean by construction.
"""
function replication_stats(y, sigma2, repkey::Vector{String}, strat::Vector{String},
                           idx::AbstractVector{Int})
    bygroup = Dict{String,Vector{Int}}()
    for i in idx
        push!(get!(bygroup, repkey[i], Int[]), i)
    end
    out = Dict{String,Tuple{Float64,Float64,Int}}()
    for (_, ig) in bygroup
        ng = length(ig)
        ng >= 2 || continue
        ybar = mean(y[i] for i in ig)
        for i in ig
            s = strat[i]
            S, D, df = get(out, s, (0.0, 0.0, 0))
            out[s] = (S + (y[i] - ybar)^2,
                      D + sigma2[i] * (1.0 - 1.0 / ng),
                      df + 1)
        end
    end
    # Degrees of freedom: each group of size n_g contributes n_g - 1, shared
    # across the strata present in it in proportion to their record counts.
    dfs = Dict{String,Float64}()
    for (_, ig) in bygroup
        ng = length(ig)
        ng >= 2 || continue
        for s in unique(strat[i] for i in ig)
            k = count(i -> strat[i] == s, ig)
            dfs[s] = get(dfs, s, 0.0) + (ng - 1) * k / ng
        end
    end
    Dict(s => (v[1], v[2], get(dfs, s, 0.0)) for (s, v) in out)
end

raw_scale(S, D) = D > 0 ? clamp(S / D, C_MIN, C_MAX) : NaN

"""
    shrink_level(stats, parent_of, parent_scale) -> Dict(stratum => c)

Empirical-Bayes shrinkage of the raw stratum scales toward their parent on the
log scale.  A raw estimate with `df` degrees of freedom has sampling variance
about 2/df for log c; the between-stratum variance is estimated by moments and
the weight is psi^2 / (psi^2 + 2/df).  Strata below `DF_MIN` take the parent.
"""
function shrink_level(stats::Dict{String,Tuple{Float64,Float64,Float64}},
                      parent_of::Dict{String,String},
                      parent_scale::Dict{String,Float64})
    usable = [(s, raw_scale(v[1], v[2]), v[3]) for (s, v) in stats
              if v[3] >= DF_MIN && isfinite(raw_scale(v[1], v[2]))]
    out = Dict{String,Float64}()
    isempty(usable) && return out
    resid = [log(c) - log(get(parent_scale, get(parent_of, s, ""), NaN))
             for (s, c, _) in usable]
    keep = findall(isfinite, resid)
    isempty(keep) && return out
    vhat = length(keep) >= 2 ? var(resid[keep]) : 0.0
    noise = mean(2.0 / usable[i][3] for i in keep)
    psi2 = max(vhat - noise, 0.0)
    for (s, c, df) in usable
        p = get(parent_scale, get(parent_of, s, ""), NaN)
        isfinite(p) || continue
        w = psi2 <= 0 ? 0.0 : psi2 / (psi2 + 2.0 / df)
        out[s] = clamp(exp(w * log(c) + (1 - w) * log(p)), C_MIN, C_MAX)
    end
    out
end

"""
    fit_scales(y, sigma2, repkey, kind, eck, qrow, idx) -> NamedTuple

The full shrunk hierarchy of scales, estimated on `idx` only: a global scale,
per-type scales shrunk to it, per-enzyme-by-type scales shrunk to their type,
and per-quantity scales shrunk to their enzyme-by-type stratum.
"""
function fit_scales(y, sigma2, repkey, kind, eck, qrow, idx)
    allk = fill("all", length(y))
    gstat = replication_stats(y, sigma2, repkey, allk, idx)
    cg = haskey(gstat, "all") ? raw_scale(gstat["all"][1], gstat["all"][2]) : 1.0
    isfinite(cg) || (cg = 1.0)
    groot = Dict("all" => cg)

    tstat = replication_stats(y, sigma2, repkey, kind, idx)
    tpar = Dict(s => "all" for s in keys(tstat))
    ctype = shrink_level(tstat, tpar, groot)

    estat = replication_stats(y, sigma2, repkey, eck, idx)
    epar = Dict{String,String}()
    for i in idx
        epar[eck[i]] = kind[i]
    end
    cenz = shrink_level(estat, epar, ctype)

    qstat = replication_stats(y, sigma2, repkey, qrow, idx)
    qpar = Dict{String,String}()
    for i in idx
        qpar[qrow[i]] = eck[i]
    end
    cqty = shrink_level(qstat, qpar, cenz)

    (; cg, ctype, cenz, cqty)
end

"""
    scale_vector(method, sc, kind, eck, qrow, idx, cell_scale) -> Vector{Float64}

Per-record scale factors under one construction, falling back up the hierarchy
whenever a finer stratum was not determined in the training fold.
"""
function scale_vector(method::String, sc, kind, eck, qrow, idx, cell_scale::Float64)
    n = length(idx)
    out = Vector{Float64}(undef, n)
    for (j, i) in enumerate(idx)
        ct = get(sc.ctype, kind[i], sc.cg)
        ce = get(sc.cenz, eck[i], ct)
        cq = get(sc.cqty, qrow[i], ce)
        out[j] = method == "convention"  ? 1.0 :
                 method == "global_cell" ? cell_scale :
                 method == "global"      ? sc.cg :
                 method == "type"        ? ct :
                 method == "enzyme"      ? ce : cq
    end
    out
end

"""
    tau_by_stratum(y, sigma2, cscale, qrow, org, strat, idx) -> (tau_g, Dict(stratum => tau2))

Between-organism variance, identified from the same design that identifies the
record scale and again without fitting the model.  Records sharing a
derived-quantity row share an identical model mean, so for one quantity row the
spread of its per-organism means is the between-organism variance plus the
record noise those means inherit:

    E sum_o (ybar_qo - ybar_q)^2 = (G_q - 1) tau^2
                                   + (1 - 1/G_q) sum_o c sigma_conv^2 / n_qo,

which inverts for tau^2 after subtracting the second term.  Strata are
partitions of the quantity rows, so a per-type variance is a pooled sum over
that type's rows.  The per-stratum estimates are shrunk to the pooled one on
the log scale exactly as the record scales are.
"""
function tau_by_stratum(y, sigma2, cscale::Vector{Float64}, qrow::Vector{String},
                        org::Vector{String}, strat::Vector{String},
                        idx::AbstractVector{Int})
    byq = Dict{String,Dict{String,Vector{Int}}}()
    for i in idx
        push!(get!(get!(byq, qrow[i], Dict{String,Vector{Int}}()), org[i], Int[]), i)
    end
    acc = Dict{String,Tuple{Float64,Float64,Float64}}()  # stratum => (SS, correction, df)
    for (q, byo) in byq
        Gq = length(byo)
        Gq >= 2 || continue
        means = Dict(o => mean(y[i] for i in ig) for (o, ig) in byo)
        gm = mean(values(means))
        ss = sum((means[o] - gm)^2 for o in keys(means))
        corr = 0.0
        for (o, ig) in byo
            nqo = length(ig)
            corr += (1.0 - 1.0 / Gq) * mean(cscale[i] * sigma2[i] for i in ig) / nqo
        end
        s = strat[first(first(values(byo)))]
        S0, C0, D0 = get(acc, s, (0.0, 0.0, 0.0))
        acc[s] = (S0 + ss, C0 + corr, D0 + (Gq - 1))
    end
    isempty(acc) && return (1.0, Dict{String,Float64}())
    SS = sum(v[1] for v in values(acc)); CC = sum(v[2] for v in values(acc))
    DD = sum(v[3] for v in values(acc))
    tau_g = DD > 0 ? clamp((SS - CC) / DD, TAU_MIN, TAU_MAX) : 1.0
    stats = Dict(s => (max(v[1] - v[2], TAU_MIN * v[3]), v[3], v[3])
                 for (s, v) in acc if v[3] >= DF_MIN)
    raw = Dict(s => clamp(v[1] / v[2], TAU_MIN, TAU_MAX) for (s, v) in stats)
    out = Dict{String,Float64}()
    if length(raw) >= 2
        resid = [log(r) - log(tau_g) for r in values(raw)]
        noise = mean(2.0 / stats[s][3] for s in keys(raw))
        psi2 = max(var(collect(resid)) - noise, 0.0)
        for (s, r) in raw
            w = psi2 <= 0 ? 0.0 : psi2 / (psi2 + 2.0 / stats[s][3])
            out[s] = clamp(exp(w * log(r) + (1 - w) * log(tau_g)), TAU_MIN, TAU_MAX)
        end
    end
    (tau_g, out)
end

interval_score(l, u, y, alpha) =
    (u - l) + (2 / alpha) * max(l - y, 0.0) + (2 / alpha) * max(y - u, 0.0)

function run_pathway(pathway::Symbol)
    root = joinpath(dirname(@__DIR__), "output", string(pathway))
    results_dir = joinpath(root, "results", "revision"); mkpath(results_dir)
    brenda = load(joinpath(root, "data", "brenda_inputs_$(pathway).jld2"))
    post = load(joinpath(root, "data", "posterior_$(pathway).jld2"))
    y = Vector{Float64}(brenda["y"]); sigma2 = Vector{Float64}(brenda["σ2"])
    Arows = Matrix{Float64}(brenda["Arows"]); org = Vector{String}(brenda["grp"])
    meta = brenda["meta"]
    kinds = Vector{String}(getproperty(meta, :kind))
    ecs = string.(getproperty(meta, :ec_code))
    qrows = string.(getproperty(meta, :Q_row))
    q0 = vec(Array(post["q_prior"])); C0 = PRIOR_SCALE .* Matrix(Array(post["C_prior"]))
    eb = EBSpec(q0 = q0, C0 = C0, κ0 = parse(Float64, get(ENV, "PB_EB_KAPPA0", "50.0")))
    d = size(Arows, 2)
    A0, b0 = _prepare_information_pair(Arows, d, Matrix{Float64}(brenda["A0"]),
                                       Vector{Float64}(brenda["b0"]);
                                       deduplicate_thermo_block = true)
    if EXCLUDE_MUTANTS
        mm = mutant_mask(dirname(@__DIR__), CSVNAME_OF[pathway], y, org, kinds, ecs,
                         string.(getproperty(meta, :substrate)))
        keep = findall(.!mm.mutant)
        @printf("  enzyme-form filter: %d of %d records are mutant-only, keeping %d\n",
                count(mm.mutant), length(y), length(keep))
        y = y[keep]; sigma2 = sigma2[keep]; Arows = Arows[keep, :]
        org = org[keep]; kinds = kinds[keep]; ecs = ecs[keep]; qrows = qrows[keep]
    end
    org_type = [string(org[i], "|", kinds[i]) for i in eachindex(org)]
    eck = [string(ecs[i], "|", kinds[i]) for i in eachindex(org)]
    qkey = [string(qrows[i], "|", kinds[i]) for i in eachindex(org)]
    repkey = [string(org[i], "|", qrows[i]) for i in eachindex(org)]

    slow = pathway in (:tca,)
    spec = FoldFitSpec(model = :gaussian, hierarchical = true,
        n_outer = (slow ? 4 : 3) * OUTER_MULT, n_mcmc = slow ? 250 : 150,
        burnin = slow ? 80 : 40, thin = 1,
        n_keep_min = 60, n_pred_draws = 400, b_draws_ll = 50,
        posterior_refresh = 1, adaptive_stop = false, rtol_outer = 1e-3,
        min_outer = 2, tau_update = :ig, tau_pc_rate = 0.0,
        ν_init = 50.0, α_init = 0.0,
        update_nu = false, update_alpha = false, refine_shape = false,
        shape_opt_tol = 1e-3, skew_location = :mean,
        nu_penalty = 0.0, nu_target = 50.0, alpha_penalty = 0.0)

    folds = sort!(unique(org)); nf = length(folds); n = length(y); K = length(REV_LEVELS)
    idx95 = findfirst(x -> isapprox(x, 0.95; atol = 1e-8), REV_LEVELS)
    idx99 = findfirst(x -> isapprox(x, 0.99; atol = 1e-8), REV_LEVELS)

    covered = Dict(m => Matrix{Bool}(undef, n, K) for m in METHODS)
    width95 = Dict(m => Vector{Float64}(undef, n) for m in METHODS)
    isc95 = Dict(m => Vector{Float64}(undef, n) for m in METHODS)
    predm = Dict(m => Vector{Float64}(undef, n) for m in METHODS)
    scalerec = Dict(m => Vector{Float64}(undef, n) for m in METHODS)
    taurec = Dict(m => Vector{Float64}(undef, n) for m in METHODS)
    amixs = fill(NaN, nf); araws = fill(NaN, nf); vexs = fill(NaN, nf)
    aboot_lo = fill(NaN, nf); aboot_hi = fill(NaN, nf)
    amls = fill(NaN, nf); bmls = fill(NaN, nf); pi1s = fill(NaN, nf)
    cgs = fill(NaN, nf); ecs2 = fill(NaN, nf)
    c1s = fill(NaN, nf); c2s = fill(NaN, nf)
    dll_ig = fill(NaN, nf); dll_two = fill(NaN, nf); ngroups = fill(0, nf)
    seeds = _pb_thread_seeds(MersenneTwister(REV_SEED), nf)

    println("\n=== $(pathway): N=$n organisms=$nf")

    Threads.@threads :dynamic for f in 1:nf
        # Common random numbers across constructions: every construction starts
        # this fold from an identical RNG state (the re-seeding below), so a
        # reported contrast between two constructions carries no sampler noise
        # from the other five.
        te = findall(==(folds[f]), org); tr = findall(!=(folds[f]), org)
        Ar_tr = Arows[tr, :]
        sc = fit_scales(y, sigma2, repkey, kinds, eck, qkey, tr)
        amix, bmix, araw, vex = scale_population(y, sigma2, repkey, tr, sc.cg)
        amixs[f] = amix; araws[f] = araw; vexs[f] = vex
        amix13, bmix13 = scale_population(y, sigma2, repkey, tr, sc.cg; floor = 1.3)
        amix20, bmix20 = scale_population(y, sigma2, repkey, tr, sc.cg; floor = 2.0)
        # log-normal population: unclamped spread, mean pinned to the pooled scale
        s_ln = sqrt(max(vex, 1e-4)); m_ln = log(sc.cg) - 0.5 * s_ln^2
        aboot_lo[f], aboot_hi[f] = scale_population_boot(y, sigma2, repkey, tr; seed = f)
        pf = scale_population_fit(y, sigma2, repkey, tr)
        amls[f] = pf.a; bmls[f] = pf.b; pi1s[f] = pf.pi1; c1s[f] = pf.c1; c2s[f] = pf.c2
        cgs[f] = sc.cg
        ecs2[f] = pf.pi1 * pf.c1 + (1 - pf.pi1) * pf.c2      # mean of the fitted population
        dll_ig[f] = pf.ll_ig - pf.ll_single; dll_two[f] = pf.ll_two - pf.ll_single
        ngroups[f] = pf.n_groups

        # The published within-cell scale, for the global_cell construction.
        rng_f = _pb_rng_from_seed(seeds[f])
        ccell = 1.0
        hs0 = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
        fit0 = fit_fold_mcem(rng_f, A0, b0, Ar_tr, y[tr], ccell .* sigma2[tr],
                             org_type[tr], eb, hs0, spec; method = :collapsed)
        for _ in 1:SIGMACAL_ITERS
            m, _ = posterior_q_moments(fit0, Ar_tr)
            ccell = c_update_within(y[tr], m, sigma2[tr], org_type[tr], eachindex(tr))
            hs0 = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
            fit0 = fit_fold_mcem(rng_f, A0, b0, Ar_tr, y[tr], ccell .* sigma2[tr],
                                 org_type[tr], eb, hs0, spec; method = :collapsed)
        end

        # One set of prediction primitives per fold, shared by every
        # construction, so all cross-construction contrasts are paired in the
        # prediction stage as well as in the fits.
        rng_p = _pb_rng_from_seed(seeds[f])
        nte = length(te)
        Uprim = rand(rng_p, N_PRED, nte)
        Z1 = randn(rng_p, N_PRED, nte)
        Z2 = randn(rng_p, N_PRED, nte)
        Uc = rand(rng_p, N_PRED, nte)      # scale_mix latent-scale primitives
        Ut = rand(rng_p, N_PRED, nte)      # tau_post latent-variance primitives

        fitcache = Dict{String,Any}()
        for m in METHODS
            scname = SCALE_OF[m]
            rng_f = _pb_rng_from_seed(seeds[f])
            ctr = scale_vector(scname, sc, kinds, eck, qkey, tr, ccell)
            cte = scale_vector(scname, sc, kinds, eck, qkey, te, ccell)
            if scname == "global_cell"
                fit = fit0
            elseif haskey(fitcache, scname)
                fit = fitcache[scname]
            else
                hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
                fit = fit_fold_mcem(rng_f, A0, b0, Ar_tr, y[tr], ctr .* sigma2[tr],
                                    org_type[tr], eb, hs, spec; method = :collapsed)
                fitcache[scname] = fit
            end
            τ2_base = fit.hs.τ²
            # Constructions that stratify the between-organism variance replace
            # the fitted scalar by the type-specific moment estimate, computed
            # on the training fold under that construction's record scales.
            τ2_of = Dict{String,Float64}()
            if m in STRAT_TAU
                cfull = scale_vector(scname, sc, kinds, eck, qkey, 1:length(y), ccell)
                tg, tk = tau_by_stratum(y, sigma2, cfull, qrows, org, kinds, tr)
                τ2_base = tg
                τ2_of = tk
            elseif m in STRAT_TAU_EB
                τ2_of = tau_pm_by_kind(y, sigma2, ctr, Ar_tr, fit, org_type[tr],
                                       kinds[tr], tr, τ2_base)
            elseif m in STRAT_TAU_REL
                raw = tau_pm_by_kind(y, sigma2, ctr, Ar_tr, fit, org_type[tr],
                                     kinds[tr], tr, τ2_base)
                τ2_of, _ = tau2_by_type_with_fallback(raw, kinds[tr], τ2_base)
            end
            S = size(fit.qsamps, 2)
            for (j, i) in enumerate(te)
                a = view(Arows, i, :)
                mu_s = [dot(a, view(fit.qsamps, :, s)) for s in 1:S]
                sd_i = sqrt(cte[j] * sigma2[i])
                τ2 = get(τ2_of, kinds[i], τ2_base)
                draws = Vector{Float64}(undef, N_PRED)
                if m in ("scale_mix", "scale_mix13", "scale_mix20")
                    mix = m == "scale_mix"   ? InverseGamma(amix, bmix) :
                          m == "scale_mix13" ? InverseGamma(amix13, bmix13) :
                                               InverseGamma(amix20, bmix20)
                    for r in 1:N_PRED
                        cr = quantile(mix, Uc[r, j])
                        draws[r] = mu_s[1 + floor(Int, Uprim[r, j] * S)] +
                                   sqrt(τ2) * Z1[r, j] + sqrt(cr * sigma2[i]) * Z2[r, j]
                    end
                elseif m == "scale_mixml"
                    mixml = InverseGamma(amls[f], bmls[f])
                    for r in 1:N_PRED
                        cr = quantile(mixml, Uc[r, j])
                        draws[r] = mu_s[1 + floor(Int, Uprim[r, j] * S)] +
                                   sqrt(τ2) * Z1[r, j] + sqrt(cr * sigma2[i]) * Z2[r, j]
                    end
                elseif m in ("scale_mix2", "scale_mix2m", "joint2m", "joint2m_rel")
                    # scale_mix2  propagates the maximum-likelihood two-class
                    # population as fitted; scale_mix2m rescales both class
                    # scales so that the population mean equals the pooled
                    # replication scale, which makes the construction
                    # variance-matched, so that only the SHAPE of the record
                    # error differs from the single-scale construction.
                    # joint2m is scale_mix2m on top of the per-type between-organism
                    # variance (tau_eb): the two adopted refinements run together.
                    k2 = m in ("scale_mix2m", "joint2m", "joint2m_rel") && isfinite(ecs2[f]) && ecs2[f] > 0 ?
                         cgs[f] / ecs2[f] : 1.0
                    for r in 1:N_PRED
                        cr = k2 * (Uc[r, j] < pi1s[f] ? c1s[f] : c2s[f])
                        draws[r] = mu_s[1 + floor(Int, Uprim[r, j] * S)] +
                                   sqrt(τ2) * Z1[r, j] + sqrt(cr * sigma2[i]) * Z2[r, j]
                    end
                elseif m == "scale_lnmix"
                    for r in 1:N_PRED
                        cr = exp(m_ln + s_ln * quantile(Normal(), Uc[r, j]))
                        draws[r] = mu_s[1 + floor(Int, Uprim[r, j] * S)] +
                                   sqrt(τ2) * Z1[r, j] + sqrt(cr * sigma2[i]) * Z2[r, j]
                    end
                elseif m == "tau_post"
                    Gc = length(fit.grp_levels)
                    chi = Chisq(Gc)
                    for r in 1:N_PRED
                        τ2r = τ2 * Gc / quantile(chi, Ut[r, j])
                        draws[r] = mu_s[1 + floor(Int, Uprim[r, j] * S)] +
                                   sqrt(τ2r) * Z1[r, j] + sd_i * Z2[r, j]
                    end
                else
                    for r in 1:N_PRED
                        draws[r] = mu_s[1 + floor(Int, Uprim[r, j] * S)] +
                                   sqrt(τ2) * Z1[r, j] + sd_i * Z2[r, j]
                    end
                end
                sort!(draws)
                predm[m][i] = mean(mu_s)
                scalerec[m][i] = cte[j]
                taurec[m][i] = τ2
                for (k, lv) in enumerate(REV_LEVELS)
                    lo = quantile(draws, (1 - lv) / 2; sorted = true)
                    hi = quantile(draws, (1 + lv) / 2; sorted = true)
                    covered[m][i, k] = lo <= y[i] <= hi
                    if k == idx95
                        width95[m][i] = hi - lo
                        isc95[m][i] = interval_score(lo, hi, y[i], 0.05)
                    end
                end
            end
        end
    end

    orgset = sort!(unique(org))
    byorg = Dict(o => findall(==(o), org) for o in orgset)
    typeset = sort!(unique(kinds))
    ecset = [s for s in sort!(unique(eck)) if count(==(s), eck) >= MIN_STRAT_REC]

    sumrows = NamedTuple[]
    covdf = DataFrame(pathway = String[], method = String[], nominal = Float64[],
                      empirical = Float64[])
    stratrows = NamedTuple[]
    for m in METHODS
        cov_org = mean(mean(covered[m][byorg[o], idx95]) for o in orgset)
        devs = Float64[]
        for s in ecset
            is = findall(==(s), eck)
            cv = mean(covered[m][is, idx95])
            push!(devs, abs(cv - 0.95))
            push!(stratrows, (pathway = string(pathway), method = m, level = "enzyme",
                              stratum = s, n_rec = length(is), coverage = cv,
                              width = mean(width95[m][is]),
                              scale = mean(scalerec[m][is]),
                              tau2 = mean(taurec[m][is])))
        end
        tcov = Dict{String,Float64}()
        for s in typeset
            is = findall(==(s), kinds)
            cv = mean(covered[m][is, idx95])
            tcov[s] = cv
            push!(stratrows, (pathway = string(pathway), method = m, level = "type",
                              stratum = s, n_rec = length(is), coverage = cv,
                              width = mean(width95[m][is]),
                              scale = mean(scalerec[m][is]),
                              tau2 = mean(taurec[m][is])))
        end
        push!(sumrows, (pathway = string(pathway), method = m,
            Cov95 = mean(covered[m][:, idx95]), Cov95_org = cov_org,
            Width95 = mean(width95[m]), IS95 = mean(isc95[m]),
            IS95_org = mean(mean(isc95[m][byorg[o]]) for o in orgset),
            RMSE = sqrt(mean(abs2, y .- predm[m])),
            Cov95_kcat = get(tcov, "kcat", NaN), Cov95_KM = get(tcov, "KM", NaN),
            CondGapMax = isempty(devs) ? NaN : maximum(devs),
            CondGapMean = isempty(devs) ? NaN : mean(devs),
            n_strata = length(ecset),
            scale_min = minimum(scalerec[m]), scale_max = maximum(scalerec[m]),
            Cov99 = mean(covered[m][:, idx99]),
            a_mix = m == "scale_mix" ? mean(filter(isfinite, amixs)) :
                    m == "scale_mix13" ? 1.3 : m == "scale_mix20" ? 2.0 : NaN,
            a_raw = m == "scale_mix" ? (fa = filter(isfinite, araws); isempty(fa) ? Inf : median(fa)) : NaN,
            v_excess = m == "scale_mix" ? median(filter(isfinite, vexs)) : NaN,
            a_boot_lo = m == "scale_mix" ? median(filter(isfinite, aboot_lo)) : NaN,
            a_boot_hi = m == "scale_mix" ? (fh = filter(isfinite, aboot_hi); isempty(fh) ? Inf : median(fh)) : NaN,
            a_ml = m == "scale_mixml" ? median(filter(isfinite, amls)) : NaN,
            b_ml = m == "scale_mixml" ? median(filter(isfinite, bmls)) : NaN,
            pi_two = m in ("scale_mix2", "scale_mix2m", "joint2m", "joint2m_rel") ? median(filter(isfinite, pi1s)) : NaN,
            c1_two = m == "scale_mix2" ? median(filter(isfinite, c1s)) :
                     m in ("scale_mix2m", "joint2m", "joint2m_rel") ? median(filter(isfinite, c1s .* cgs ./ ecs2)) : NaN,
            c2_two = m == "scale_mix2" ? median(filter(isfinite, c2s)) :
                     m in ("scale_mix2m", "joint2m", "joint2m_rel") ? median(filter(isfinite, c2s .* cgs ./ ecs2)) : NaN,
            mean_two = m == "scale_mix2" ? median(filter(isfinite, ecs2)) :
                       m in ("scale_mix2m", "joint2m", "joint2m_rel") ? median(filter(isfinite, cgs)) : NaN,
            pooled_scale = median(filter(isfinite, cgs)),
            dloglik_vs_single = m == "scale_mixml" ? median(filter(isfinite, dll_ig)) :
                                m in ("scale_mix2", "scale_mix2m", "joint2m", "joint2m_rel") ? median(filter(isfinite, dll_two)) : NaN,
            n_replicate_groups = m in ("scale_mixml", "scale_mix2", "scale_mix2m", "joint2m", "joint2m_rel") ? median(ngroups) : NaN,
            tau2_kcat = (kk = findall(==("kcat"), kinds); isempty(kk) ? NaN : mean(taurec[m][kk])),
            tau2_KM = (kk = findall(==("KM"), kinds); isempty(kk) ? NaN : mean(taurec[m][kk])),
            n_folds = nf))
        for (k, lv) in enumerate(REV_LEVELS)
            push!(covdf, (string(pathway), m, lv, mean(covered[m][:, k])))
        end
        @printf("  %-12s Cov95=%.4f Cov99=%.4f org=%.4f W95=%.2f IS95=%.2f kcat=%.3f KM=%.3f condmax=%.3f condmean=%.3f c=[%.2f,%.2f] tau2=[%.2f,%.2f]\n",
                m, mean(covered[m][:, idx95]), mean(covered[m][:, idx99]), cov_org, mean(width95[m]),
                mean(isc95[m]), get(tcov, "kcat", NaN), get(tcov, "KM", NaN),
                isempty(devs) ? NaN : maximum(devs), isempty(devs) ? NaN : mean(devs),
                minimum(scalerec[m]), maximum(scalerec[m]),
                minimum(taurec[m]), maximum(taurec[m]))
    end

    # Per-record indicators for the cluster-aware inference driver
    # (run_revision_hetero_inference.jl): one BitVector of 95% coverage hits per
    # construction, with the organism and quantity-type labels.
    save(joinpath(results_dir, "hetero_percell_$(pathway)$(OUT_SUFFIX).jld2"), Dict(
        "org" => org, "kind" => kinds,
        "cov95" => Dict(m => Vector{Bool}(covered[m][:, idx95]) for m in METHODS),
        "cov99" => Dict(m => Vector{Bool}(covered[m][:, idx99]) for m in METHODS),
        "width95" => Dict(m => width95[m] for m in METHODS)))
    @printf("  scale population a_mix = %.2f (mean over folds); unclamped a_raw median = %.2f, excess log-variance = %.3f, bootstrap 95%% for a_raw = [%.2f, %.2f]\n",
            mean(filter(isfinite, amixs)), median(filter(isfinite, araws)),
            median(filter(isfinite, vexs)), median(filter(isfinite, aboot_lo)),
            (fh = filter(isfinite, aboot_hi); isempty(fh) ? Inf : median(fh)))
    @printf("  population mean of the fitted two-class law = %.2f against a pooled scale of %.2f (ratio %.2f); the mean-pinned variant rescales both classes by %.2f\n",
            median(filter(isfinite, ecs2)), median(filter(isfinite, cgs)),
            median(filter(isfinite, ecs2 ./ cgs)), median(filter(isfinite, cgs ./ ecs2)))
    @printf("  population fits (medians over folds): IG by ML a=%.2f b=%.2f dll=%+.1f; two-component pi=%.2f c1=%.2f c2=%.2f dll=%+.1f; groups=%d\n",
            median(filter(isfinite, amls)), median(filter(isfinite, bmls)), median(filter(isfinite, dll_ig)),
            median(filter(isfinite, pi1s)), median(filter(isfinite, c1s)), median(filter(isfinite, c2s)),
            median(filter(isfinite, dll_two)), Int(median(ngroups)))
    CSV.write(joinpath(results_dir, "hetero_summary_$(pathway)$(OUT_SUFFIX).csv"), DataFrame(sumrows))
    CSV.write(joinpath(results_dir, "hetero_coverage_$(pathway)$(OUT_SUFFIX).csv"), covdf)
    CSV.write(joinpath(results_dir, "hetero_strata_$(pathway)$(OUT_SUFFIX).csv"), DataFrame(stratrows))
    return nothing
end

println("Revision heteroscedastic error-scale analysis")
println("  threads = $(Threads.nthreads())")
println("  constructions = $(join(METHODS, ","))  prior scale = $(PRIOR_SCALE)  suffix = '$(OUT_SUFFIX)'")
println("  started = $(now())")
for pw in REV_PATHWAYS
    run_pathway(pw)
end
println("Done at $(now())")
