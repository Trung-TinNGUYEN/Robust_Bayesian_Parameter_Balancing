# hierarchy_utils.jl
#
# Estimators for the between-source variance of the hierarchical model that are
# shared by the revision drivers and the command-line tool, so that the released
# specification (a between-source variance stratified by quantity type) is one
# piece of code rather than three copies.

"""
    paule_mandel_tau2(rs, vs, tau2_max) -> tau2

Paule-Mandel estimate of a between-group variance from group means `rs` with
within-group variances `vs`: the value of `tau2` at which the weighted sum of
squares `sum(w_i (r_i - rbar_w)^2)` with `w_i = 1 / (tau2 + v_i)` equals its
expectation `G - 1`.  Returns `0` when the groups are no more dispersed than
their own variances imply, `tau2_max` when the equation has no root below it,
and `NaN` with fewer than five groups.
"""
function paule_mandel_tau2(rs::Vector{Float64}, vs::Vector{Float64}, τ2_max::Float64)
    G = length(rs); G >= 5 || return NaN
    F(t) = (w = [1.0 / (t + v) for v in vs]; sw = sum(w);
            rw = sum(w[i] * rs[i] for i in 1:G) / sw;
            sum(w[i] * (rs[i] - rw)^2 for i in 1:G) - (G - 1))
    F(0.0) <= 0 && return 0.0
    hi = 1.0
    while F(hi) > 0 && hi < τ2_max; hi *= 2.0; end
    F(hi) > 0 && return τ2_max
    lo = 0.0
    for _ in 1:60
        mid = 0.5 * (lo + hi); F(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    0.5 * (lo + hi)
end
const _pm_tau2 = paule_mandel_tau2

"""
    tau_pm_by_kind(y, sigma2, ctr, Ar, fit, cells, kinds, idx, tau2_pool) -> Dict

Between-source variance per quantity type, by the Paule-Mandel estimator applied
to the cell means of the marginal consensus residuals.  For every source cell
(the `cells` label of each record in `idx`) the mean residual of its records
about the posterior-mean fitted values is one group; its variance is the
record-error part, `sum(c_j sigma2_j) / n^2`, plus the posterior variance of
the fitted cell mean, so that posterior uncertainty in the consensus is not
mistaken for between-source spread.  Types with fewer than five cells are
omitted; estimates are clamped to `[0.05, 8] * tau2_pool`.
"""
function tau_pm_by_kind(y, sigma2, ctr, Ar, fit, cells, kinds, idx, τ2_pool)
    M = Ar * fit.qsamps
    mbar = vec(mean(M, dims = 2))
    bycell = Dict{String,Vector{Int}}()
    for (j, c) in enumerate(cells)
        push!(get!(bycell, c, Int[]), j)
    end
    stats = Dict{String,Vector{Tuple{Float64,Float64}}}()
    for (_, jg) in bycell
        ng = length(jg)
        rbar = mean(y[idx[j]] - mbar[j] for j in jg)
        vg = sum(ctr[j] * sigma2[idx[j]] for j in jg) / ng^2
        ug = var(vec(mean(view(M, jg, :), dims = 1)))
        push!(get!(stats, kinds[jg[1]], Tuple{Float64,Float64}[]),
              (rbar, 1.0 / max(vg + ug, 1e-8)))
    end
    out = Dict{String,Float64}()
    for (k, v) in stats
        length(v) >= 5 || continue
        t = paule_mandel_tau2([x[1] for x in v], [1.0 / x[2] for x in v], 8.0 * τ2_pool)
        isfinite(t) && (out[k] = clamp(t, 0.05 * τ2_pool, 8.0 * τ2_pool))
    end
    out
end

"""
    tau2_by_type_with_fallback(raw, kinds, tau2_pool; lo = 0.2, hi = 5.0)

The released specification's rule: use the per-type estimate where it lies
within `[lo, hi]` times the pooled value and the pooled value otherwise.
Returns `(tau2_of, source)` dictionaries keyed by type.
"""
function tau2_by_type_with_fallback(raw::Dict{String,Float64}, kinds, τ2_pool::Float64;
                                    lo::Float64 = 0.2, hi::Float64 = 5.0)
    τ2_of = Dict{String,Float64}(); src = Dict{String,String}()
    for k in unique(kinds)
        t = get(raw, k, NaN)
        if isfinite(t) && lo * τ2_pool <= t <= hi * τ2_pool
            τ2_of[k] = t; src[k] = "per_type"
        else
            τ2_of[k] = τ2_pool; src[k] = "pooled_fallback"
        end
    end
    τ2_of, src
end
