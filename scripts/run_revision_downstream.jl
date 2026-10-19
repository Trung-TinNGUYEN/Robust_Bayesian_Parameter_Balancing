# run_revision_downstream.jl
#
# Downstream propagation of balanced posteriors through a kinetic rate law,
# added for the revision.
#
# Reviewer request addressed here:
#   "The manuscript motivates the method through downstream kinetic and dynamic
#    modelling, but no balanced posterior is propagated through a kinetic model.
#    It therefore remains unclear whether the differences in posterior
#    uncertainty materially affect predicted concentrations, fluxes,
#    sensitivities, or experimental-design decisions."
#
# What is computed.  For every fitted model cell we draw from the posterior of
# the log-basis parameters, map each draw through the dependency matrix to a
# thermodynamically consistent set of derived quantities (standard chemical
# potentials, Michaelis constants, forward and reverse catalytic constants), and
# evaluate three downstream quantities per reaction:
#
#   1. the convenience-kinetics reaction rate v_j per unit enzyme at a reference
#      metabolite state,
#   2. the equilibrium constant K_eq,j implied by the standard chemical
#      potentials, and
#   3. the scaled elasticity d log v_j / d log c_i of the rate with respect to
#      each substrate concentration, the quantity that controls the local
#      sensitivity structure of a kinetic model.
#
# Because every draw is a thermodynamically consistent parameter set, the
# resulting predictive distributions inherit the Haldane constraints exactly.
# The comparison of interest is the WIDTH of these downstream intervals across
# model cells: a parameter posterior that is too narrow produces flux and
# elasticity predictions that are correspondingly overconfident.
#
# Convenience kinetics (Liebermeister and Klipp, 2006) for reaction j with
# substrate stoichiometries s_i and product stoichiometries p_i, writing
# a_i = c_i / K_M(j,i):
#
#   v_j = e_j [ kcat_f prod_S a_i^{s_i} - kcat_r prod_P a_i^{p_i} ]
#         / [ prod_S sum_{m=0}^{s_i} a_i^m + prod_P sum_{m=0}^{p_i} a_i^m - 1 ]
#
# Water is treated as a pure-activity species: it carries no Michaelis constant
# in the observation map, so it is excluded from the rate law.
#
# Model cells.  The first five cells use the half-decade record scale as given
# (record-scale multiplier c = 1).  The sixth cell, gaussian_cest_RE_orgtype, is
# the pre-specified default of the cross-validation (the global_cell
# construction of run_revision_hetero.jl): Gaussian likelihood, organism-by-type
# random intercept, and a single record-scale multiplier c estimated by the
# within-cell moment iteration.  It is fitted at c = 1, then c is re-estimated
# from the posterior-mean consensus residuals within organism-by-type cells and
# the model refitted, PB_REV_SIGMACAL_ITERS times (clamped to [C_MIN, C_MAX]),
# continuing the same random-number stream.  All cells use kappa0 = 50 (the
# EBSpec default, as in the cross-validated default), and every hierarchical
# cell, including the new one, draws new-organism offsets from its own pooled
# tau^2.  The new cell is appended last, so the rows of the first five cells
# are unchanged, and the width ratio logv_width_vs_gaussian_fix still refers to
# the gaussian_Fix population target.
#
# Outputs, written under output/<pathway>/results/revision/:
#   downstream_reactions_<pathway>.csv      per reaction and model cell
#   downstream_summary_<pathway>.csv        per model cell, aggregated over
#                                           reactions, with the fitted tau^2 and
#                                           the record-scale multiplier c
#   downstream_elasticity_<pathway>.csv     per reaction, substrate and model cell
#   downstream_concentration_<pathway>.csv  median rate-interval width per model
#                                           cell at each reference concentration,
#                                           in cell order (as listed in `cells`),
#                                           then target, then concentration
#
# Environment controls:
#   PB_REV_PATHWAYS        comma-separated subset of {glycolysis,tca} (default both)
#   PB_REV_CONC            reference metabolite concentration in M (default 1e-3)
#   PB_REV_CONC_SWEEP      reference concentrations for the width sweep
#                          (default 1e-4,1e-3,1e-2)
#   PB_REV_SIGMACAL_ITERS  within-cell iterations for the estimated record scale
#                          of gaussian_cest_RE_orgtype (default 4)
#   PB_REV_OUTER_MULT      multiplies the number of outer EM iterations of every
#                          fit (default 1, the published setting), for the
#                          EM-iteration sensitivity run (S1 Text)
#   PB_EB_KAPPA0           prior-mean concentration of every fit (default 50, the
#                          prior mean updated, as in the cross-validation; 1e-9
#                          holds it at the thermodynamic prior, as the released
#                          parameter set does, for the propagation of the
#                          shorter manuscript)
#   PB_REV_SUFFIX          appended to every output file name (default empty). A
#                          run with PB_REV_OUTER_MULT other than 1, or with
#                          PB_EB_KAPPA0 other than 50, must set it, so that it
#                          cannot overwrite the published files

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

const REV_SEED = 20260805
const REV_PATHWAYS = Symbol.(strip.(split(get(ENV, "PB_REV_PATHWAYS", "glycolysis,tca"), ",")))
const REF_CONC = parse(Float64, get(ENV, "PB_REV_CONC", "1e-3"))   # molar
# Reference states used for the sensitivity check reported in S1 Text.
const CONC_SWEEP = [parse(Float64, s) for s in
                    split(get(ENV, "PB_REV_CONC_SWEEP", "1e-4,1e-3,1e-2"), ",")]
# Within-cell record-scale iteration, as in run_revision_hetero.jl.
const SIGMACAL_ITERS = parse(Int, get(ENV, "PB_REV_SIGMACAL_ITERS", "4"))
const OUTER_MULT = parse(Int, get(ENV, "PB_REV_OUTER_MULT", "1"))
const OUT_SUFFIX = get(ENV, "PB_REV_SUFFIX", "")
OUTER_MULT == 1 || !isempty(OUT_SUFFIX) || error("PB_REV_OUTER_MULT is not 1: set PB_REV_SUFFIX as well, so that the published outputs are kept")
const KAPPA0 = parse(Float64, get(ENV, "PB_EB_KAPPA0", "50.0"))
KAPPA0 == 50.0 || !isempty(OUT_SUFFIX) || error("PB_EB_KAPPA0 is not 50: set PB_REV_SUFFIX as well, so that the published outputs are kept")
const C_MIN, C_MAX = 0.25, 40.0

"""
    c_update_within(y, m, sigma2, cells, idx) -> c

The within-cell moment estimator of the record-scale multiplier, copied
verbatim from run_revision_hetero.jl (clamp [C_MIN, C_MAX] = [0.25, 40]) so that
the gaussian_cest_RE_orgtype cell matches the pre-specified default.
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

_sample_skewness(x) = (m = mean(x); s = std(x, corrected = false);
                       s <= 0 ? 0.0 : mean(((x .- m) ./ s) .^ 3))
_sample_exkurtosis(x) = (m = mean(x); s = std(x, corrected = false);
                         s <= 0 ? 0.0 : mean(((x .- m) ./ s) .^ 4) - 3.0)

function alpha_from_skewness(g1::Float64)
    s = sign(g1); a = min(abs(g1), 0.985)
    a < 1e-4 && return 0.0
    f(delta) = ((4.0 - pi) / 2.0) * (delta * sqrt(2.0 / pi))^3 /
               (1.0 - (2.0 / pi) * delta^2)^1.5
    lo, hi = 0.0, 0.999999
    for _ in 1:200
        mid = 0.5 * (lo + hi); f(mid) < a ? (lo = mid) : (hi = mid)
    end
    delta = clamp(0.5 * (lo + hi), 0.0, 0.999)
    return s * delta / sqrt(1.0 - delta^2)
end
nu_from_kurtosis(k::Float64) = k <= 0.05 ? 50.0 : clamp(4.0 + 6.0 / k, 2.5, 50.0)

function ridge_qhat(Arows, y, sigma2, q0, C0)
    C0inv = inv(Symmetric(Matrix{Float64}(C0)))
    P = C0inv + Arows' * (Arows ./ sigma2)
    return Symmetric(P) \ (Arows' * (y ./ sigma2) + C0inv * q0)
end

"""
    convenience_rate(kcat_f, kcat_r, a_sub, s_sub, a_prod, s_prod) -> v

Convenience-kinetics rate per unit enzyme.  `a_*` are concentrations already
divided by the corresponding Michaelis constants; `s_*` are the stoichiometric
coefficients (positive integers).
"""
function convenience_rate(kcat_f::Float64, kcat_r::Float64,
                          a_sub::Vector{Float64}, s_sub::Vector{Int},
                          a_prod::Vector{Float64}, s_prod::Vector{Int})
    num_f = kcat_f
    @inbounds for i in eachindex(a_sub)
        num_f *= a_sub[i]^s_sub[i]
    end
    num_r = kcat_r
    @inbounds for i in eachindex(a_prod)
        num_r *= a_prod[i]^s_prod[i]
    end
    den_s = 1.0
    @inbounds for i in eachindex(a_sub)
        t = 0.0
        for m in 0:s_sub[i]
            t += a_sub[i]^m
        end
        den_s *= t
    end
    den_p = 1.0
    @inbounds for i in eachindex(a_prod)
        t = 0.0
        for m in 0:s_prod[i]
            t += a_prod[i]^m
        end
        den_p *= t
    end
    return (num_f - num_r) / (den_s + den_p - 1.0)
end

"""
    build_reaction_spec(rxns, h2o_names, species_names, KM_map, kcatf, kcatr, N)

Collect, for every reaction, the substrate and product species (excluding
water), their stoichiometric coefficients, and the rows of the derived-quantity
vector holding the corresponding Michaelis and catalytic constants.
"""
function build_reaction_spec(rxns, h2o_names::Vector{String}, species_names::Vector{String},
                             KM_map, kcatf, kcatr, N::AbstractMatrix)
    specs = Vector{NamedTuple}(undef, length(rxns))
    for (j, r) in enumerate(rxns)
        rname = r[1]
        sub_i = Int[]; sub_s = Int[]; sub_km = Int[]; sub_names = String[]
        pro_i = Int[]; pro_s = Int[]; pro_km = Int[]; pro_names = String[]
        for i in eachindex(species_names)
            nu = N[i, j]
            nu == 0 && continue
            sname = species_names[i]
            sname in h2o_names && continue
            haskey(KM_map, (rname, sname)) || continue
            if nu < 0
                push!(sub_i, i); push!(sub_s, Int(abs(nu)))
                push!(sub_km, KM_map[(rname, sname)]); push!(sub_names, sname)
            else
                push!(pro_i, i); push!(pro_s, Int(nu))
                push!(pro_km, KM_map[(rname, sname)]); push!(pro_names, sname)
            end
        end
        specs[j] = (name = rname,
                    sub_idx = sub_i, sub_stoich = sub_s, sub_km_row = sub_km,
                    sub_names = sub_names,
                    pro_idx = pro_i, pro_stoich = pro_s, pro_km_row = pro_km,
                    pro_names = pro_names,
                    kcatf_row = kcatf[rname], kcatr_row = kcatr[rname],
                    stoich = N[:, j])
    end
    return specs
end

"""
    propagate(qsamps, Q, specs, mu0_rows, conc) -> (logv, logKeq, elasticity)

Map posterior draws of the basis parameters into downstream kinetic quantities.
`logv` and `logKeq` are base-10 logarithms; `elasticity[j][k]` holds the scaled
elasticity of reaction `j` with respect to its `k`-th substrate.
"""
function propagate(qsamps::Matrix{Float64}, Q::Matrix{Float64}, specs,
                   mu0_rows::Vector{Int}, conc::Float64)
    S = size(qsamps, 2)
    J = length(specs)
    X = Q * qsamps                         # derived quantities, natural log scale
    logv = fill(NaN, J, S)
    signv = zeros(Int, J, S)
    logKeq = fill(NaN, J, S)
    elast = [fill(NaN, length(sp.sub_idx), S) for sp in specs]
    ln10 = log(10.0)
    h = 1e-4                               # log-scale step for the elasticity

    for s in 1:S
        x = view(X, :, s)
        for (j, sp) in enumerate(specs)
            kf = exp(x[sp.kcatf_row])
            kr = exp(x[sp.kcatr_row])
            a_sub = [conc / exp(x[row]) for row in sp.sub_km_row]
            a_pro = [conc / exp(x[row]) for row in sp.pro_km_row]
            v = convenience_rate(kf, kr, a_sub, sp.sub_stoich, a_pro, sp.pro_stoich)
            signv[j, s] = sign(v)
            logv[j, s] = log10(abs(v) + 1e-300)

            # ln K_eq = - sum_i nu_ij mu0_i, with mu0 already in RT units.
            acc = 0.0
            for i in eachindex(mu0_rows)
                nu = sp.stoich[i]
                nu == 0 && continue
                acc -= nu * x[mu0_rows[i]]
            end
            logKeq[j, s] = acc / ln10

            for k in eachindex(sp.sub_idx)
                a_pert = copy(a_sub)
                a_pert[k] *= exp(h)
                v_pert = convenience_rate(kf, kr, a_pert, sp.sub_stoich, a_pro, sp.pro_stoich)
                elast[j][k, s] = (abs(v) > 1e-300 && abs(v_pert) > 1e-300) ?
                                 (log(abs(v_pert)) - log(abs(v))) / h : NaN
            end
        end
    end
    return logv, signv, logKeq, elast
end

_ciwidth(v) = (w = filter(isfinite, v); length(w) < 10 ? NaN :
               quantile(w, 0.975) - quantile(w, 0.025))
_median(v) = (w = filter(isfinite, v); isempty(w) ? NaN : median(w))

"""
    add_organism_effect(qsamps, mur_idx, tau2, rng) -> qsamps_shifted

Return posterior draws for the parameters of a *new, previously unmeasured* source
organism, obtained by adding an organism-specific offset drawn from the fitted
between-source distribution.

The offset is applied to the reaction-velocity block of the basis vector (the
`mu_r` entries).  Every point of the basis satisfies the Haldane relationships, so
any offset on a basis coordinate leaves them intact; what singles out this block is
that forward and reverse catalytic constants both carry coefficient `+1` on `mu_r`
and no Michaelis or equilibrium constant depends on it, so the offset scales both
turnover numbers of a reaction by the same factor and leaves the rest unchanged.  Shifting the derived quantities directly
would not have this property, because a source effect estimated on the measurement
scale is not itself constrained by thermodynamics; this is a limitation of the
additive source-effect model and is discussed in S1 Text.
"""
function add_organism_effect(qsamps::Matrix{Float64}, mur_idx::Vector{Int},
                             tau2::Float64, rng::AbstractRNG)
    tau2 <= 0 && return copy(qsamps)
    out = copy(qsamps)
    sd = sqrt(tau2)
    @inbounds for s in axes(out, 2)
        eta = sd * randn(rng)
        for i in mur_idx
            out[i, s] += eta
        end
    end
    return out
end

function run_pathway(pathway::Symbol, setup)
    rxns = setup.rxns
    Qfull = Matrix{Float64}(setup.Q)
    h2o = setup.h2o
    root = joinpath(dirname(@__DIR__), "output", string(pathway))
    results_dir = joinpath(root, "results", "revision")
    mkpath(results_dir)

    brenda = load(joinpath(root, "data", "brenda_inputs_$(pathway).jld2"))
    post = load(joinpath(root, "data", "posterior_$(pathway).jld2"))
    y = Vector{Float64}(brenda["y"]); sigma2 = Vector{Float64}(brenda["σ2"])
    Arows = Matrix{Float64}(brenda["Arows"]); org = Vector{String}(brenda["grp"])
    kinds = Vector{String}(getproperty(brenda["meta"], :kind))
    org_type = [string(org[i], "|", kinds[i]) for i in eachindex(org)]
    q0 = vec(Array(post["q_prior"])); C0 = Matrix(Array(post["C_prior"]))
    eb = EBSpec(q0 = q0, C0 = C0, κ0 = KAPPA0)
    d = size(Arows, 2)
    A0, b0 = _prepare_information_pair(Arows, d, Matrix{Float64}(brenda["A0"]),
                                       Vector{Float64}(brenda["b0"]);
                                       deduplicate_thermo_block = true)

    species_names, reaction_names = rn_entities(rxns)
    (mu0_map, KM_map, kcatf, kcatr) = observation_matrix_index_maps(rxns)
    (_, mur_param_map, _) = parameter_index_maps(rxns)
    mur_idx = sort([mur_param_map[r] for r in reaction_names])
    rn = reaction_network(rxns)
    N = netstoichmat(rn)
    specs = build_reaction_spec(rxns, h2o, species_names, KM_map, kcatf, kcatr, N)
    mu0_rows = [mu0_map[s] for s in species_names]

    println("\n=== $(pathway) downstream propagation: $(length(specs)) reactions, " *
            "reference concentration $(REF_CONC) M")

    qh = ridge_qhat(Arows, y, sigma2, q0, C0)
    r = (y .- Arows * qh) ./ sqrt.(sigma2)
    nu0 = nu_from_kurtosis(_sample_exkurtosis(r))
    al0 = alpha_from_skewness(_sample_skewness(r))
    @printf("  moment-based shape start: nu0=%.2f alpha0=%.3f\n", nu0, al0)

    is_tca = pathway == :tca
    function spec_for(model, hier; estimate = false)
        FoldFitSpec(
            model = model, hierarchical = hier,
            n_outer = OUTER_MULT * (model == :gaussian ? (is_tca ? 4 : 3) : (is_tca ? 5 : 4)),
            n_mcmc = 600, burnin = 150, thin = 1, n_keep_min = 400,
            n_pred_draws = 400, b_draws_ll = 50, posterior_refresh = 1,
            adaptive_stop = false, rtol_outer = 1e-3, min_outer = 2,
            tau_update = :ig, tau_pc_rate = 0.0,
            ν_init = model == :t ? nu0 : 50.0,
            α_init = model == :skew ? al0 : 0.0,
            update_nu = estimate && model == :t,
            update_alpha = estimate && model == :skew,
            refine_shape = estimate, shape_opt_tol = 1e-3, skew_location = :mean,
            nu_penalty = 0.0, nu_target = nu0, alpha_penalty = 0.0)
    end

    # Sixth field: record scale.  :convention keeps c = 1 (the half-decade scale
    # as given); :global_cell estimates c by the within-cell iteration.  New
    # cells are appended at the end so the rows of the earlier cells, and the
    # gaussian_Fix baseline, are unchanged.
    cells = [
        ("gaussian_Fix",             :gaussian, false, org,      false, :convention),
        ("gaussian_RE_org",          :gaussian, true,  org,      false, :convention),
        ("gaussian_RE_orgtype",      :gaussian, true,  org_type, false, :convention),
        ("t_moment_RE_org",          :t,        true,  org,      true,  :convention),
        ("t_moment_RE_orgtype",      :t,        true,  org_type, true,  :convention),
        ("gaussian_cest_RE_orgtype", :gaussian, true,  org_type, false, :global_cell),
    ]
    cell_rank = Dict(c[1] => k for (k, c) in enumerate(cells))

    rxn_rows = DataFrame(pathway = String[], cell = String[], target = String[],
                         reaction = String[],
                         logv_median = Float64[], logv_ci95_width = Float64[],
                         logKeq_median = Float64[], logKeq_ci95_width = Float64[],
                         frac_forward = Float64[])
    ela_rows = DataFrame(pathway = String[], cell = String[], target = String[],
                         reaction = String[], substrate = String[],
                         elasticity_median = Float64[], elasticity_ci95_width = Float64[])
    conc_rows = DataFrame(pathway = String[], cell = String[], target = String[],
                          concentration_M = Float64[], median_logv_width = Float64[])
    sum_rows = DataFrame(pathway = String[], cell = String[], target = String[],
                         tau2 = Float64[], nu = Float64[], alpha = Float64[],
                         n_draws = Int[], median_logv_width = Float64[],
                         median_logKeq_width = Float64[],
                         median_elasticity_width = Float64[],
                         logv_width_vs_gaussian_fix = Float64[],
                         c_record_scale = Float64[])

    baseline_width = NaN
    for (label, model, hier, grp, est, scale) in cells
        print("  -> $label ... "); flush(stdout)
        t0 = time()
        spec = spec_for(model, hier; estimate = est)
        hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
        # A fresh stream per cell; the within-cell iteration below continues it.
        rng = MersenneTwister(REV_SEED)
        fit = fit_fold_mcem(rng, A0, b0, Arows, y, sigma2,
                            grp, eb, hs, spec; method = :collapsed)
        c_hat = 1.0
        if scale == :global_cell
            # Pre-specified default: re-estimate c from the posterior-mean
            # consensus residuals within the grouping cells, then refit.
            for it in 1:SIGMACAL_ITERS
                m = vec(mean(Arows * fit.qsamps, dims = 2))
                c_hat = c_update_within(y, m, sigma2, grp, eachindex(y))
                hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
                fit = fit_fold_mcem(rng, A0, b0, Arows, y, c_hat .* sigma2,
                                    grp, eb, hs, spec; method = :collapsed)
                @printf("\n       c iteration %d: c = %.4f, tau2 = %.4f", it, c_hat, fit.hs.τ²)
            end
        elseif scale != :convention
            error("unknown record scale $(scale)")
        end

        # Two downstream targets.  `population` uses the posterior for the
        # consensus parameters; `new_organism` additionally draws the offset that a
        # previously unmeasured source organism would carry, which is the quantity
        # a modeller needs when parameterising an organism whose enzymes were not
        # among the curated records.
        targets = [("population", fit.qsamps)]
        if hier && fit.hs.τ² > 0
            push!(targets, ("new_organism",
                            add_organism_effect(fit.qsamps, mur_idx, fit.hs.τ²,
                                                MersenneTwister(REV_SEED + 991))))
        end

        for (target, qs) in targets
            # Sensitivity of the rate-interval width to the reference metabolite
            # state.  Only the width summary is retained for the sweep points; the
            # per-reaction detail is stored for the primary reference state.
            for cc in CONC_SWEEP
                cc == REF_CONC && continue
                lv, _, _, _ = propagate(qs, Qfull, specs, mu0_rows, cc)
                w = [x for x in (_ciwidth(view(lv, j, :)) for j in eachindex(specs))
                     if isfinite(x)]
                isempty(w) && continue
                push!(conc_rows, (string(pathway), label, target, cc, median(w)))
            end

            logv, signv, logKeq, elast = propagate(qs, Qfull, specs, mu0_rows, REF_CONC)
            wv = Float64[]; wk = Float64[]; we = Float64[]
            for (j, sp) in enumerate(specs)
                w1 = _ciwidth(view(logv, j, :)); w2 = _ciwidth(view(logKeq, j, :))
                push!(rxn_rows, (string(pathway), label, target, sp.name,
                                 _median(view(logv, j, :)), w1,
                                 _median(view(logKeq, j, :)), w2,
                                 mean(view(signv, j, :) .> 0)))
                isfinite(w1) && push!(wv, w1); isfinite(w2) && push!(wk, w2)
                for k in eachindex(sp.sub_idx)
                    w3 = _ciwidth(view(elast[j], k, :))
                    push!(ela_rows, (string(pathway), label, target, sp.name,
                                     sp.sub_names[k], _median(view(elast[j], k, :)), w3))
                    isfinite(w3) && push!(we, w3)
                end
            end

            mv = isempty(wv) ? NaN : median(wv)
            # The reference width is the fixed-effects consensus (gaussian_Fix,
            # population), which is the first cell and target processed.
            (label == "gaussian_Fix" && target == "population") && (baseline_width = mv)
            push!(sum_rows, (string(pathway), label, target, fit.hs.τ²,
                             model == :t ? fit.ν : NaN, model == :skew ? fit.α : NaN,
                             size(qs, 2), mv,
                             isempty(wk) ? NaN : median(wk),
                             isempty(we) ? NaN : median(we),
                             mv / baseline_width, c_hat))
            @printf("\n       %-13s median 95%% width: log10 v = %.2f, log10 Keq = %.2f, elasticity = %.3f",
                    target, mv, isempty(wk) ? NaN : median(wk),
                    isempty(we) ? NaN : median(we))
        end
        @printf("   [%.1f s]\n", time() - t0)
        flush(stdout)
    end

    CSV.write(joinpath(results_dir, "downstream_reactions_$(pathway)$(OUT_SUFFIX).csv"), rxn_rows)
    CSV.write(joinpath(results_dir, "downstream_elasticity_$(pathway)$(OUT_SUFFIX).csv"), ela_rows)
    CSV.write(joinpath(results_dir, "downstream_summary_$(pathway)$(OUT_SUFFIX).csv"), sum_rows)
    # Append the primary reference state so the sweep table is self-contained.
    for r in eachrow(sum_rows)
        push!(conc_rows, (r.pathway, r.cell, r.target, REF_CONC, r.median_logv_width))
    end
    # Rows are ordered by the position of the cell in `cells`.  For the first
    # five cells this is the same as the alphabetical order used originally, so
    # their rows keep their positions and the appended cell comes last.
    CSV.write(joinpath(results_dir, "downstream_concentration_$(pathway)$(OUT_SUFFIX).csv"),
              sort(conc_rows, [order(:cell, by = lab -> cell_rank[lab]),
                               :target, :concentration_M]))
    println("\n  $(pathway) downstream summary")
    show(stdout, sum_rows; allrows = true, allcols = true); println()
    return nothing
end

# Pathway reconstruction (mirrors run_all_paper_experiments.jl).
const mdl = load_yeast_reconstruction()
const dict_reactions = reaction_data_dict(mdl)

function tca_setup()
    rxns = generate_reactions(
        ["r_0300","r_0302","r_0280","r_0658","r_0832","r_0831",
         "r_0505","r_1022","r_1021","r_0451","r_0713"],
        dict_reactions, protons = false, proton_name = "s_0799")
    h2o = ["s_0807"]
    return (rxns = rxns, h2o = h2o, Q = observation_matrix(rxns, h2o_strings = h2o))
end

function glycolysis_setup()
    rxns = generate_reactions(
        ["r_0534","r_0467","r_0886","r_0450","r_1054",
         "r_0486","r_0892","r_0893","r_0366","r_0962"],
        dict_reactions, protons = false, proton_name = "s_0794")
    h2o = ["s_0803"]
    return (rxns = rxns, h2o = h2o, Q = observation_matrix(rxns, h2o_strings = h2o))
end

println("Revision downstream propagation")
println("  threads  = $(Threads.nthreads())")
println("  pathways = $(REV_PATHWAYS)")
println("  started  = $(now())")
flush(stdout)

for pw in REV_PATHWAYS
    t0 = time()
    setup = pw == :tca ? tca_setup() : glycolysis_setup()
    run_pathway(pw, setup)
    @printf("[%s] elapsed %.1f min\n", pw, (time() - t0) / 60)
    flush(stdout)
end
println("Done at $(now())")
