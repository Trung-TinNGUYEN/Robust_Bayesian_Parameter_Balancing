# run_revision_rate_simulation.jl
#
# Coverage of predicted reaction rates in simulation, added after the shortened
# manuscript (October 2026).
#
# Question.  The Propagation paragraph shows that the rate intervals of the
# pre-specified default are wider than those of standard parameter balancing,
# but width alone does not say which intervals are right.  Where the true
# parameters are known, the rate intervals can be checked directly: how often
# does the 95% interval of the rate per unit enzyme contain the true rate?
#
# Why a new design.  The synthetic scenarios of S1 Text D.4 use a random
# dependency matrix with no reactions, so no rate can be computed there.  This
# driver builds the same two scenarios on the real glycolysis and TCA-cycle
# structure: the full observation matrix Q from yeast-GEM, the curated record
# design (which quantity each record measures, and which organism it comes
# from), and the thermodynamic block of the real fits.
#
# Truth.  The basis vector q_true is the posterior mean of the released
# parameter set (output/paramset/basis_posterior_mean_<pathway>.csv).  The true
# consensus rate of reaction j is the convenience rate at 1 mM of every
# metabolite (water excluded), evaluated at Q q_true, as in
# run_revision_downstream.jl.
#
# Records.  For replicate r, every curated record i is replaced by
#
#   y_i = (Q q_true)_{row(i)} + eta_{cell(i)} + e_i,
#
# with one offset eta per organism-by-type cell (organism and quantity type,
# kcat or KM), eta ~ N(0, tau2_true), and record error e_i ~ N(0, c_i sigma2),
# sigma2 = (0.5 ln 10)^2 the half-decade convention.  tau2_true and the record
# scale are those the pre-specified default estimated on the real records of
# the pathway (downstream_summary_<pathway>.csv, cell gaussian_cest_RE_orgtype).
#
#   scenario B  homoscedastic: c_i = c_true for every record, so the default's
#               single estimated record scale is correctly specified;
#   scenario D  inverse-gamma population, as in S1 Text D.4: every replicate
#               group (one organism, one quantity) draws its own scale
#               c_g ~ InverseGamma(3, 8), with population mean 4, while every
#               model fits a single scale.
#
# The thermodynamic information vector is redrawn as b0 = A0 q_true + A0^{1/2} z,
# which is its sampling distribution under the Gaussian thermodynamic
# likelihood the fits assume (A0 is kept).
#
# New organisms.  A new organism carries one kcat offset and one KM offset,
# both N(0, tau2_true), on the measurement scale: all its turnover numbers are
# multiplied by exp(eta_kcat) and all its Michaelis constants by exp(eta_KM).
# NNEW new organisms are drawn per replicate, and their true rates follow.
#
# Models, fitted to the synthetic records of each replicate.  All five use the
# Gaussian likelihood and the sampler budget of run_revision_downstream.jl
# (600 draws per EM iteration, 150 burn-in, 3 EM iterations, 4 on the TCA cycle).
#
#   spb       standard parameter balancing as fitted in the paper: fixed
#             effects, record scale held at the half-decade convention (c = 1),
#             kappa0 = 50
#   default   the pre-specified default (cell gaussian_cest_RE_orgtype of
#             run_revision_downstream.jl): organism-by-type random intercept,
#             record scale from the within-cell iteration (one fit at c = 1,
#             then 4 refits, clamped to [0.25, 40]), pooled tau2, kappa0 = 50
#   released  the configuration of the released parameter set
#             (run_revision_paramset.jl): organism-by-type random intercept,
#             fit-free replication record scale, prior mean held (kappa0 = 1e-9),
#             and a per-type tau2 for new organisms (tau2_by_type_with_fallback)
#   spb_held, default_held
#             spb and default with the prior mean held (kappa0 = 1e-9), the two
#             fits of the propagation of the shorter manuscript
#             (run_revision_downstream.jl with PB_EB_KAPPA0=1e-9).  They were
#             added after the first three; each model draws from its own random
#             streams, so adding them left the results of the first three
#             unchanged
#
# Rate intervals.  Each posterior draw of q gives rates at 1 mM.  Targets:
#
#   consensus  the posterior draws as they are (all five models)
#   new_vel    the paper's new-organism interval: one offset ~ N(0, tau2_hat) per
#              draw on the reaction-velocity block of q, which scales both
#              turnover numbers of every reaction and leaves Michaelis
#              constants unchanged (add_organism_effect of
#              run_revision_downstream.jl); for the released configuration the
#              offset uses the kcat type's tau2
#   new_full   as new_vel, plus a second, independent offset on every Michaelis
#              constant (tau2_hat, or the KM type's tau2 for the released
#              configuration), as the record model itself assumes
#
# Both new-organism targets draw OFFSETS_PER_DRAW offsets per posterior draw.
# Standard parameter balancing has no new-organism interval: its consensus
# interval is what it offers for an organism without records.
#
# Scores per replicate, reaction, model and target: whether the 95% interval
# (2.5% and 97.5% quantiles of the signed rate draws) contains the true
# consensus rate; the fraction of the NNEW new organisms whose true rate it
# contains; and the width of the 95% interval of log10 |rate| in decades (the
# width the paper reports).  The parameter-level coverage of the consensus
# intervals over the curated rows of Q q_true is kept as well, for comparison
# with S1 Text D.4.
#
# Outputs, written under output/rate_checks/:
#   rate_simulation_rows.csv     one row per pathway, scenario, replicate,
#                                model, target and reaction
#   rate_simulation_fits.csv     one row per pathway, scenario, replicate and
#                                model: c, tau2 (pooled and per type), lambda,
#                                parameter-level consensus coverage
#   rate_simulation_summary.csv  coverage and width per pathway, scenario,
#                                model and target, with the standard error
#                                over replicates
#
# Environment controls:
#   PB_RSIM_PATHWAYS   subset of {glycolysis,tca} (default both)
#   PB_RSIM_SCENARIOS  subset of {B,D} (default both)
#   PB_RSIM_REPS       replicates per pathway and scenario (default 100)
#   PB_RSIM_NEWORG     new organisms per replicate (default 400)
#   PB_RSIM_SEED       base seed (default 20261019)
#   PB_RSIM_OUTER_MULT multiplies the number of EM iterations of every fit
#                      (default 1), for the EM-budget sensitivity run
#   PB_RSIM_TRUTH      mean (default): q_true is the released posterior mean;
#                      draw: every replicate draws q_true from the released
#                      posterior, N(mean, basis_posterior_cov_<pathway>.csv),
#                      so that the truth is not fixed where the prior and the
#                      curated records put it
#   PB_REV_SUFFIX      output-name suffix (default empty; required when either
#                      of the two controls above is not at its default)
#
# Run (replicates in parallel over Julia threads, BLAS on one thread):
#   julia --project=. --threads=8 scripts/run_revision_rate_simulation.jl

import Pkg
Pkg.activate(dirname(@__DIR__))

using Dates, LinearAlgebra, Printf, Random, Statistics
using CSV, DataFrames, Distributions, JLD2, SpecialFunctions

# PATHWAY_SPECS and the reconstruction helpers (its main() is guarded).
include("build_pathway_inputs.jl")
include(joinpath(dirname(@__DIR__), "src", "parameter_balancing_core.jl"))
include(joinpath(dirname(@__DIR__), "src", "prediction_utils.jl"))
include(joinpath(dirname(@__DIR__), "src", "hierarchy_utils.jl"))
include(joinpath(dirname(@__DIR__), "src", "experiment_runners.jl"))

BLAS.set_num_threads(1)

const SIM_SEED = parse(Int, get(ENV, "PB_RSIM_SEED", "20261019"))
const PATHWAYS = Symbol.(strip.(split(get(ENV, "PB_RSIM_PATHWAYS", "glycolysis,tca"), ",")))
const SCENARIOS = String.(strip.(split(get(ENV, "PB_RSIM_SCENARIOS", "B,D"), ",")))
const NREP = parse(Int, get(ENV, "PB_RSIM_REPS", "100"))
const NNEW = parse(Int, get(ENV, "PB_RSIM_NEWORG", "400"))
const OUT_SUFFIX = get(ENV, "PB_REV_SUFFIX", "")
const OUTER_MULT = parse(Int, get(ENV, "PB_RSIM_OUTER_MULT", "1"))
const TRUTH = get(ENV, "PB_RSIM_TRUTH", "mean")
TRUTH in ("mean", "draw") || error("PB_RSIM_TRUTH must be mean or draw")
(OUTER_MULT == 1 && TRUTH == "mean") || !isempty(OUT_SUFFIX) ||
    error("set PB_REV_SUFFIX for a sensitivity run, so that the main outputs are kept")
const OFFSETS_PER_DRAW = 4
const REF_CONC = 1e-3                     # molar, the paper's reference state
const SIGMACAL_ITERS = 4                  # as the pre-specified default
const C_MIN, C_MAX = 0.25, 40.0           # run_revision_hetero.jl clamp
const IG_SHAPE, IG_SCALE = 3.0, 8.0       # scenario D of S1 Text D.4
const OUTDIR = joinpath(dirname(@__DIR__), "output", "rate_checks")
all(in(("B", "D")), SCENARIOS) || error("PB_RSIM_SCENARIOS must be a subset of B,D")

# ---- copies of the published helpers ----------------------------------------
# run_revision_downstream.jl and run_revision_paramset.jl have no main guard, so
# the functions below are verbatim copies.

# convenience_rate of run_revision_downstream.jl.
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

# build_reaction_spec of run_revision_downstream.jl.
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

# c_update_within of run_revision_hetero.jl / run_revision_downstream.jl.
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

# replication_stats and raw_scale of run_revision_paramset.jl (fit-free record scale).
function replication_stats(y, sigma2, repkey, strat, idx)
    bygroup = Dict{String,Vector{Int}}()
    for i in idx
        push!(get!(bygroup, repkey[i], Int[]), i)
    end
    out = Dict{String,Tuple{Float64,Float64,Int}}()
    for (_, ig) in bygroup
        ng = length(ig); ng >= 2 || continue
        ybar = mean(y[i] for i in ig)
        for i in ig
            s = strat[i]
            S, D, df = get(out, s, (0.0, 0.0, 0))
            out[s] = (S + (y[i] - ybar)^2, D + sigma2[i] * (1.0 - 1.0 / ng), df + 1)
        end
    end
    Dict(s => (v[1], v[2], Float64(v[3])) for (s, v) in out)
end
raw_scale(S, D) = D > 0 ? clamp(S / D, C_MIN, C_MAX) : NaN

# ---- rates -------------------------------------------------------------------

"Signed convenience rate per unit enzyme of reaction `sp` at log quantities `x`."
function rate_at(x::AbstractVector{Float64}, sp, conc::Float64)
    kf = exp(x[sp.kcatf_row]); kr = exp(x[sp.kcatr_row])
    a_sub = [conc / exp(x[r]) for r in sp.sub_km_row]
    a_pro = [conc / exp(x[r]) for r in sp.pro_km_row]
    return convenience_rate(kf, kr, a_sub, sp.sub_stoich, a_pro, sp.pro_stoich)
end

"Rates (J x size(X, 2)) of every reaction for every column of log quantities `X`."
function rates_of(X::AbstractMatrix{Float64}, specs)
    V = Matrix{Float64}(undef, length(specs), size(X, 2))
    for s in axes(X, 2)
        x = view(X, :, s)
        for (j, sp) in enumerate(specs)
            V[j, s] = rate_at(x, sp, REF_CONC)
        end
    end
    return V
end

"Shift every turnover-number row of `X` by `ek` and every Michaelis row by `em`, per column."
function shift_rows!(X::AbstractMatrix{Float64}, kcat_rows, km_rows, ek, em)
    for s in axes(X, 2)
        for r in kcat_rows
            X[r, s] += ek[s]
        end
        for r in km_rows
            X[r, s] += em[s]
        end
    end
    return X
end

log10abs(v) = log10(abs(v) + 1e-300)
_w95(v) = quantile(v, 0.975) - quantile(v, 0.025)

# ---- pathway setup -------------------------------------------------------------

const MDL = load_yeast_reconstruction(joinpath(dirname(@__DIR__), "data", "yeast-GEM.xml"))
const DICT_RXNS = reaction_data_dict(MDL)

function pathway_setup(pathway::Symbol)
    spec = PATHWAY_SPECS[string(pathway)]
    rxns = generate_reactions(spec.reaction_ids, DICT_RXNS, protons = false,
                              proton_name = spec.proton_name)
    Q = Matrix{Float64}(observation_matrix(rxns, h2o_strings = spec.h2o))
    species_names, reaction_names = rn_entities(rxns)
    (_, KM_map, kcatf, kcatr) = observation_matrix_index_maps(rxns)
    (_, mur_map, _) = parameter_index_maps(rxns)
    mur_idx = sort([mur_map[r] for r in reaction_names])
    N = netstoichmat(reaction_network(rxns))
    specs = build_reaction_spec(rxns, spec.h2o, species_names, KM_map, kcatf, kcatr, N)
    kcat_rows = sort(vcat([kcatf[r] for r in reaction_names], [kcatr[r] for r in reaction_names]))
    km_rows = sort(unique(vcat([vcat(sp.sub_km_row, sp.pro_km_row) for sp in specs]...)))
    # The velocity offset of the paper is a shift of the turnover-number rows only.
    for r in reaction_names
        Q[kcatf[r], mur_map[r]] == 1 && Q[kcatr[r], mur_map[r]] == 1 ||
            error("$pathway $r: turnover-number rows do not load +1 on the velocity coordinate")
    end
    all(iszero, Q[km_rows, mur_idx]) || error("$pathway: Michaelis rows load on velocity coordinates")

    root = joinpath(dirname(@__DIR__), "output", string(pathway))
    brenda = load(joinpath(root, "data", "brenda_inputs_$(pathway).jld2"))
    post = load(joinpath(root, "data", "posterior_$(pathway).jld2"))
    sigma2 = Vector{Float64}(brenda["σ2"]); Arows = Matrix{Float64}(brenda["Arows"])
    org = Vector{String}(brenda["grp"]); meta = brenda["meta"]
    kinds = Vector{String}(getproperty(meta, :kind)); qrows = Vector{Int}(getproperty(meta, :Q_row))
    Q[qrows, :] == Arows || error("$pathway: stored design rows differ from the full Q")
    q0 = vec(Array(post["q_prior"])); C0 = Matrix(Array(post["C_prior"]))
    d = size(Arows, 2)
    A0, b0 = _prepare_information_pair(Arows, d, Matrix{Float64}(brenda["A0"]),
                                       Vector{Float64}(brenda["b0"]);
                                       deduplicate_thermo_block = true)
    E = eigen(Symmetric(A0))
    A0half = E.vectors * Diagonal(sqrt.(max.(E.values, 0.0)))

    qt = CSV.read(joinpath(dirname(@__DIR__), "output", "paramset", "basis_posterior_mean_$(pathway).csv"), DataFrame)
    q_true = Vector{Float64}(qt.q_mean)
    length(q_true) == d || error("$pathway: q_true has $(length(q_true)) entries, the basis has $d")
    Cq = Matrix{Float64}(CSV.read(joinpath(dirname(@__DIR__), "output", "paramset",
                                           "basis_posterior_cov_$(pathway).csv"), DataFrame))
    size(Cq) == (d, d) || error("$pathway: posterior covariance is not $d x $d")
    Eq = eigen(Symmetric(Cq))
    Lq = Eq.vectors * Diagonal(sqrt.(max.(Eq.values, 0.0)))
    ds = CSV.read(joinpath(root, "results", "revision", "downstream_summary_$(pathway).csv"), DataFrame)
    dr = ds[(ds.cell .== "gaussian_cest_RE_orgtype") .& (ds.target .== "population"), :]
    nrow(dr) == 1 || error("$pathway: no default row in downstream_summary")

    org_type = [string(org[i], "|", kinds[i]) for i in eachindex(org)]
    repkey = [string(org[i], "|", qrows[i]) for i in eachindex(org)]
    return (pathway = pathway, Q = Q, specs = specs, mur_idx = mur_idx,
            kcat_rows = kcat_rows, km_rows = km_rows,
            sigma2 = sigma2, Arows = Arows, org = org, kinds = kinds, qrows = qrows,
            org_type = org_type, repkey = repkey,
            cell_levels = sort(unique(org_type)), rep_levels = sort(unique(repkey)),
            rec_rows = sort(unique(qrows)),
            q0 = q0, C0 = C0, A0 = A0, A0half = A0half, q_true = q_true, Lq = Lq,
            tau2_true = dr.tau2[1], c_true = dr.c_record_scale[1])
end

# The sampler budget of run_revision_downstream.jl (spec_for(:gaussian, hier)).
function sim_spec(pathway::Symbol, hier::Bool)
    is_tca = pathway == :tca
    return FoldFitSpec(
        model = :gaussian, hierarchical = hier,
        n_outer = OUTER_MULT * (is_tca ? 4 : 3), n_mcmc = 600, burnin = 150, thin = 1,
        n_keep_min = 400,
        n_pred_draws = 400, b_draws_ll = 50, posterior_refresh = 1,
        adaptive_stop = false, rtol_outer = 1e-3, min_outer = 2,
        tau_update = :ig, tau_pc_rate = 0.0, ν_init = 50.0, α_init = 0.0,
        update_nu = false, update_alpha = false, refine_shape = false,
        shape_opt_tol = 1e-3, skew_location = :mean,
        nu_penalty = 0.0, nu_target = 50.0, alpha_penalty = 0.0)
end

# ---- one replicate ---------------------------------------------------------------

function one_replicate(P, scen::String, rep::Int, seed::UInt32)
    rng = _pb_rng_from_seed(seed)
    n = length(P.sigma2); d = length(P.q_true)
    # The truth of this replicate (its own stream, so the record draws are unchanged).
    qt = TRUTH == "mean" ? P.q_true :
         P.q_true .+ P.Lq * randn(MersenneTwister(Int(seed) + 7), d)

    # Synthetic records and thermodynamic information vector.
    eta = Dict{String,Float64}()
    for c in P.cell_levels
        eta[c] = sqrt(P.tau2_true) * randn(rng)
    end
    cvec = if scen == "B"
        fill(P.c_true, n)
    else
        cg = Dict{String,Float64}()
        for g in P.rep_levels
            cg[g] = rand(rng, InverseGamma(IG_SHAPE, IG_SCALE))
        end
        [cg[P.repkey[i]] for i in 1:n]
    end
    y = P.Arows * qt .+ [eta[P.org_type[i]] for i in 1:n] .+
        sqrt.(cvec .* P.sigma2) .* randn(rng, n)
    b0s = P.A0 * qt .+ P.A0half * randn(rng, d)

    spec_fe = sim_spec(P.pathway, false); spec_h = sim_spec(P.pathway, true)
    eb50 = EBSpec(q0 = P.q0, C0 = P.C0, κ0 = 50.0)
    ebheld = EBSpec(q0 = P.q0, C0 = P.C0, κ0 = 1e-9)
    hs_of(eb, sp) = init_hyper(eb; λ0 = sp.λ_init, τ²0 = sp.τ²_init, π0 = sp.π_init)

    # Standard parameter balancing.
    t0 = time()
    rngA = MersenneTwister(Int(seed) + 1)
    fit_s = fit_fold_mcem(rngA, P.A0, b0s, P.Arows, y, P.sigma2, P.org, eb50,
                          hs_of(eb50, spec_fe), spec_fe; method = :collapsed)
    # Pre-specified default.
    rngB = MersenneTwister(Int(seed) + 2)
    fit_d = fit_fold_mcem(rngB, P.A0, b0s, P.Arows, y, P.sigma2, P.org_type, eb50,
                          hs_of(eb50, spec_h), spec_h; method = :collapsed)
    c_d = 1.0
    for _ in 1:SIGMACAL_ITERS
        m = vec(mean(P.Arows * fit_d.qsamps, dims = 2))
        c_d = c_update_within(y, m, P.sigma2, P.org_type, 1:n)
        fit_d = fit_fold_mcem(rngB, P.A0, b0s, P.Arows, y, c_d .* P.sigma2, P.org_type, eb50,
                              hs_of(eb50, spec_h), spec_h; method = :collapsed)
    end
    # Released configuration.
    gstat = replication_stats(y, P.sigma2, P.repkey, fill("all", n), 1:n)
    c_r = haskey(gstat, "all") ? raw_scale(gstat["all"][1], gstat["all"][2]) : 1.0
    isfinite(c_r) || (c_r = 1.0)
    rngC = MersenneTwister(Int(seed) + 3)
    fit_r = fit_fold_mcem(rngC, P.A0, b0s, P.Arows, y, c_r .* P.sigma2, P.org_type, ebheld,
                          hs_of(ebheld, spec_h), spec_h; method = :collapsed)
    raw_t = tau_pm_by_kind(y, P.sigma2, fill(c_r, n), P.Arows, fit_r, P.org_type, P.kinds,
                           collect(1:n), fit_r.hs.τ²)
    τ2_of, _ = tau2_by_type_with_fallback(raw_t, P.kinds, fit_r.hs.τ²)
    # Standard parameter balancing and the default with the prior mean held.
    rngD = MersenneTwister(Int(seed) + 5)
    fit_sh = fit_fold_mcem(rngD, P.A0, b0s, P.Arows, y, P.sigma2, P.org, ebheld,
                           hs_of(ebheld, spec_fe), spec_fe; method = :collapsed)
    rngE = MersenneTwister(Int(seed) + 6)
    fit_dh = fit_fold_mcem(rngE, P.A0, b0s, P.Arows, y, P.sigma2, P.org_type, ebheld,
                           hs_of(ebheld, spec_h), spec_h; method = :collapsed)
    c_dh = 1.0
    for _ in 1:SIGMACAL_ITERS
        m = vec(mean(P.Arows * fit_dh.qsamps, dims = 2))
        c_dh = c_update_within(y, m, P.sigma2, P.org_type, 1:n)
        fit_dh = fit_fold_mcem(rngE, P.A0, b0s, P.Arows, y, c_dh .* P.sigma2, P.org_type, ebheld,
                               hs_of(ebheld, spec_h), spec_h; method = :collapsed)
    end
    t_fit = time() - t0

    # Truth: consensus rates and NNEW new organisms.
    Xtrue = P.Q * qt
    v_true = vec(rates_of(reshape(Xtrue, :, 1), P.specs))
    rngN = MersenneTwister(Int(seed) + 4)
    ek_new = sqrt(P.tau2_true) .* randn(rngN, NNEW)
    em_new = sqrt(P.tau2_true) .* randn(rngN, NNEW)
    Xnew = shift_rows!(repeat(Xtrue, 1, NNEW), P.kcat_rows, P.km_rows, ek_new, em_new)
    V_new = rates_of(Xnew, P.specs)

    models = [("spb", fit_s, 0.0, 0.0, 1.0), ("default", fit_d, fit_d.hs.τ², fit_d.hs.τ², c_d),
              ("released", fit_r, τ2_of["kcat"], τ2_of["KM"], c_r),
              ("spb_held", fit_sh, 0.0, 0.0, 1.0),
              ("default_held", fit_dh, fit_dh.hs.τ², fit_dh.hs.τ², c_dh)]
    rows = NamedTuple[]; fits = NamedTuple[]
    for (k, (mname, fit, t2k, t2m, cc)) in enumerate(models)
        qs = fit.qsamps; S = size(qs, 2)
        X = P.Q * qs
        # Parameter-level coverage of the consensus over the curated rows.
        pc = mean(begin
                      v = view(X, r, :)
                      quantile(v, 0.025) <= Xtrue[r] <= quantile(v, 0.975)
                  end for r in P.rec_rows)
        push!(fits, (pathway = string(P.pathway), scenario = scen, rep = rep, model = mname,
                     c = cc, tau2_pooled = fit.hs.τ², tau2_kcat = t2k, tau2_KM = t2m,
                     lambda = fit.hs.λ, n_draws = S, param_cov95_curated_rows = pc,
                     fit_seconds = k == 1 ? t_fit : NaN))
        targets = [("consensus", rates_of(X, P.specs))]
        if !startswith(mname, "spb")
            rngP = MersenneTwister(Int(seed) + 10 + k)
            Xr = repeat(X, 1, OFFSETS_PER_DRAW)
            ek = sqrt(t2k) .* randn(rngP, size(Xr, 2))
            em = sqrt(t2m) .* randn(rngP, size(Xr, 2))
            Xv = shift_rows!(copy(Xr), P.kcat_rows, P.km_rows, ek, zeros(size(Xr, 2)))
            Xf = shift_rows!(copy(Xr), P.kcat_rows, P.km_rows, ek, em)
            push!(targets, ("new_vel", rates_of(Xv, P.specs)))
            push!(targets, ("new_full", rates_of(Xf, P.specs)))
        end
        for (tname, V) in targets, (j, sp) in enumerate(P.specs)
            v = view(V, j, :)
            lo, hi = quantile(v, 0.025), quantile(v, 0.975)
            push!(rows, (pathway = string(P.pathway), scenario = scen, rep = rep,
                model = mname, target = tname, reaction = sp.name,
                v_true = v_true[j], lo = lo, hi = hi,
                covers_consensus = lo <= v_true[j] <= hi,
                frac_new_covered = mean(x -> lo <= x <= hi, view(V_new, j, :)),
                width95_log10 = _w95(log10abs.(v)),
                frac_positive = mean(>(0), v)))
        end
    end
    return rows, fits
end

# ---- main -------------------------------------------------------------------------

println("Revision rate simulation")
println("  threads   = $(Threads.nthreads()) (BLAS 1)")
println("  pathways  = $(PATHWAYS), scenarios = $(SCENARIOS), replicates = $(NREP), new organisms = $(NNEW)")
println("  truth     = $(TRUTH), EM iterations x $(OUTER_MULT), suffix = '$(OUT_SUFFIX)'")
println("  started   = $(now())")
flush(stdout)

all_rows = NamedTuple[]; all_fits = NamedTuple[]
for (ip, pw) in enumerate(PATHWAYS)
    P = pathway_setup(pw)
    @printf("\n=== %s: %d records, %d organism-by-type cells, %d replicate groups, %d reactions; tau2_true = %.3f, c_true = %.3f\n",
            pw, length(P.sigma2), length(P.cell_levels), length(P.rep_levels), length(P.specs),
            P.tau2_true, P.c_true)
    flush(stdout)
    for (is, scen) in enumerate(SCENARIOS)
        t0 = time()
        seeds = _pb_thread_seeds(MersenneTwister(SIM_SEED + 100 * ip + is), NREP)
        res = Vector{Any}(undef, NREP)
        Threads.@threads :dynamic for r in 1:NREP
            res[r] = one_replicate(P, scen, r, seeds[r])
        end
        for r in 1:NREP
            append!(all_rows, res[r][1]); append!(all_fits, res[r][2])
        end
        @printf("  scenario %s: %d replicates in %.1f min\n", scen, NREP, (time() - t0) / 60)
        flush(stdout)
    end
end
rdf = DataFrame(all_rows); fdf = DataFrame(all_fits)

# Summary: coverage of the true consensus rate and of the new organisms' true
# rates, averaged over replicates and reactions; the standard error is that of
# the replicate means.
srows = NamedTuple[]
for g in groupby(rdf, [:pathway, :scenario, :model, :target])
    perrep = combine(groupby(g, :rep), :covers_consensus => mean => :cc,
                     :frac_new_covered => mean => :cn)
    R = nrow(perrep)
    push!(srows, (pathway = g.pathway[1], scenario = g.scenario[1], model = g.model[1],
        target = g.target[1], n_reps = R, n_reactions = length(unique(g.reaction)),
        cov95_consensus = mean(perrep.cc), se_consensus = std(perrep.cc) / sqrt(R),
        cov95_new_organism = mean(perrep.cn), se_new_organism = std(perrep.cn) / sqrt(R),
        median_width95_decades = median(g.width95_log10)))
end
sdf = DataFrame(srows)
fsum = combine(groupby(fdf, [:pathway, :scenario, :model]),
               :c => median => :median_c, :tau2_pooled => median => :median_tau2,
               :tau2_kcat => median => :median_tau2_kcat, :tau2_KM => median => :median_tau2_KM,
               :param_cov95_curated_rows => mean => :param_cov95)

mkpath(OUTDIR)
CSV.write(joinpath(OUTDIR, "rate_simulation_rows$(OUT_SUFFIX).csv"), rdf)
CSV.write(joinpath(OUTDIR, "rate_simulation_fits$(OUT_SUFFIX).csv"), fdf)
CSV.write(joinpath(OUTDIR, "rate_simulation_summary$(OUT_SUFFIX).csv"), sdf)
println("\nRate coverage (95% intervals)")
show(stdout, sdf; allrows = true, allcols = true); println()
println("\nFitted hyperparameters and parameter-level coverage")
show(stdout, fsum; allrows = true, allcols = true); println()
println("\nDone at $(now())")
