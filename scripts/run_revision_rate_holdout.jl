# run_revision_rate_holdout.jl
#
# Held-out check of predicted reaction rates on the curated records, added after
# the shortened manuscript (October 2026).
#
# Question.  The leave-one-source-out cross-validation of the paper checks one
# record at a time.  A modeller uses the parameters jointly, through a rate law,
# so the check that matters downstream is whether the predicted rate of an
# organism the fit has not seen is calibrated.  Here the rate is built from the
# held-out organism's own records, and compared with the predictive
# distribution of the same quantity.
#
# Units.  A unit is one organism and one reaction for which the organism has
# forward turnover-number records (the kcat+ row of the reaction) and Michaelis
# constant records for every substrate of the forward direction (water
# excluded).  Products are not needed, because the rate checked is the forward
# rate with no product present, the initial rate an in vitro assay measures,
# evaluated with every substrate at 1 mM:
#
#   v = kcat+ prod_i a_i^{s_i} / prod_i sum_{m=0}^{s_i} a_i^m,   a_i = 1 mM / KM_i,
#
# the convenience rate of run_revision_downstream.jl with product
# concentrations at zero.  Too few organisms have records for both directions
# and every product to check the net rate at the reference state.  The
# organism's observed rate uses the mean of its log records of each quantity.
#
# Folds.  Every organism with at least one unit is held out in its published
# leave-one-source-out fold (organisms sorted, fold seeds from
# _pb_thread_seeds(MersenneTwister(20260805), n_folds)), and two models are
# refitted exactly as run_revision_ecoli_holdout.jl refits them for E. coli:
#
#   (i)  fixed effects: cell gaussian_Fix of run_revision_realdata.jl, the
#        standard parameter balancing of the paper;
#   (ii) pre-specified default: construction global_cell of
#        run_revision_hetero.jl (organism-by-type random intercept, record
#        scale c from the within-cell iteration, pooled tau^2).
#
# Both use kappa0 = 50, as published.  The per-record 95% coverage of the
# held-out organism's records reproduces the saved per-record files
# (realdata_percell_<pathway>.jld2, hetero_percell_<pathway>.jld2), and the
# driver checks that it does for every fold.  Where those files are absent (they
# come from run_revision_realdata.jl and run_revision_hetero.jl), the check is
# skipped and its columns are left empty.
#
# Predictive draws of the observed rate.  For unit u with n_k turnover records
# and n_i Michaelis records of substrate i, each draw r takes one posterior
# column, the organism's offsets and the errors of the record means:
#
#   log kcat+   a_k' q + eta_kcat + sqrt(c sigma2 / n_k) e_k
#   log KM_i    a_i' q + eta_KM   + sqrt(c sigma2 / n_i) e_i
#
# and evaluates the rate above.  The variants are
#
#   spb       fixed effects: no offsets, c = 1 (the half-decade convention)
#   default   eta_kcat and eta_KM independent N(0, tau^2), the record model
#             of the default
#   velocity  eta_kcat only (eta_KM = 0), the simplification of the paper's
#             new-organism rate intervals, which offset the reaction-velocity
#             block of the basis and so leave Michaelis constants unchanged
#   per_type  as default, with the per-type tau^2 of the released
#             specification (tau2_by_type_with_fallback) estimated on the
#             training fold
#
# Each variant is drawn at record level (as above, the distribution of what the
# held-out organism's records show, which is the calibrated comparison) and at
# organism level (no record error, the interval a modeller would use for the
# organism's rate; its width is reported, and its coverage of record means is
# not a calibration check).
#
# Scores per unit and variant: coverage of the observed log rate by the
# central 50, 80, 90 and 95% intervals, the PIT, the width of the 95% interval
# in decades, the CRPS in log10 units, and, for the same draws, whether the 95%
# intervals of the turnover-number mean and of each Michaelis mean cover the
# observed means.  km_sensitivity, the derivative of the observed log rate
# with respect to a common shift of all its log Michaelis constants, says how
# far the rate depends on the Michaelis constants at 1 mM (0 when every
# substrate is saturating, the number of substrates when none is).
#
# Outputs, written under output/rate_checks/:
#   rate_holdout_units.csv       one row per unit, variant and level
#   rate_holdout_summary.csv     per pathway and pooled, per variant and level:
#                                coverage, an organism-clustered bootstrap
#                                interval for the 95% coverage, PIT, width, CRPS,
#                                with strata by km_sensitivity
#   rate_holdout_regression.csv  per fold: reproduced and published per-record
#                                95% coverage of the held-out organism
#
# Environment controls:
#   PB_REV_PATHWAYS   subset of {glycolysis,tca,pentose_phosphate,shikimate}
#                     (default all four)
#   PB_RHO_NDRAW      predictive draws per unit (default 4000)
#   PB_RHO_DRAW_SEED  base seed of the unit draws (default 20261020)
#   PB_RHO_CONC       substrate concentration in M (default 1e-3)
#   PB_REV_SUFFIX     output-name suffix (default empty)
#
# Run (folds in parallel over Julia threads, BLAS on one thread):
#   julia --project=. --threads=8 scripts/run_revision_rate_holdout.jl

import Pkg
Pkg.activate(dirname(@__DIR__))

using Dates, LinearAlgebra, Printf, Random, Statistics
using CSV, DataFrames, Distributions, JLD2, SpecialFunctions

# PATHWAY_SPECS and the full observation matrix (its main() is guarded).
include("build_pathway_inputs.jl")
include(joinpath(dirname(@__DIR__), "src", "parameter_balancing_core.jl"))
include(joinpath(dirname(@__DIR__), "src", "prediction_utils.jl"))
include(joinpath(dirname(@__DIR__), "src", "hierarchy_utils.jl"))
include(joinpath(dirname(@__DIR__), "src", "experiment_runners.jl"))
include(joinpath(dirname(@__DIR__), "src", "mutant_flags.jl"))

BLAS.set_num_threads(1)

const REV_LEVELS = vcat(0.01, collect(0.05:0.05:0.95), 0.99)
const REV_SEED = 20260805                 # published fold seed (realdata, hetero)
const ALL_PATHWAYS = [:glycolysis, :tca, :pentose_phosphate, :shikimate]
const REV_PATHWAYS = Symbol.(strip.(split(get(ENV, "PB_REV_PATHWAYS",
    "glycolysis,tca,pentose_phosphate,shikimate"), ",")))
const NDRAW = parse(Int, get(ENV, "PB_RHO_NDRAW", "4000"))
const DRAW_SEED = parse(Int, get(ENV, "PB_RHO_DRAW_SEED", "20261020"))
const CONC = parse(Float64, get(ENV, "PB_RHO_CONC", "1e-3"))
const OUT_SUFFIX = get(ENV, "PB_REV_SUFFIX", "")
const KAPPA0 = 50.0
const SIGMACAL_ITERS = 4
const C_MIN, C_MAX = 0.25, 40.0           # run_revision_hetero.jl clamp
const N_PRED = 800                        # run_revision_hetero.jl draws per record
const LEVELS = [0.5, 0.8, 0.9, 0.95]
const NBOOT = 2000
const OUTDIR = joinpath(dirname(@__DIR__), "output", "rate_checks")

# ---- copies of the published helpers (run_revision_ecoli_holdout.jl) ----------

# cell_spec(:gaussian, false, pathway) of run_revision_realdata.jl (cell gaussian_Fix).
function fe_spec(pathway::Symbol)
    is_tca = pathway == :tca
    return FoldFitSpec(
        model = :gaussian, hierarchical = false,
        n_outer = is_tca ? 4 : 3, n_mcmc = is_tca ? 250 : 150, burnin = is_tca ? 80 : 40,
        thin = 1, n_keep_min = 60, n_pred_draws = 400, b_draws_ll = 50,
        posterior_refresh = 1, adaptive_stop = true, rtol_outer = 1e-3, min_outer = 2,
        tau_update = :ig, tau_pc_rate = 0.0, ν_init = 50.0, α_init = 0.0,
        update_nu = false, update_alpha = false, refine_shape = false, shape_opt_tol = 1e-3,
        skew_location = :mean, nu_penalty = 0.0, nu_target = 50.0, alpha_penalty = 0.0)
end

# The sampler specification of run_revision_hetero.jl.
function pd_spec(pathway::Symbol)
    slow = pathway in (:tca,)
    return FoldFitSpec(model = :gaussian, hierarchical = true,
        n_outer = slow ? 4 : 3, n_mcmc = slow ? 250 : 150,
        burnin = slow ? 80 : 40, thin = 1,
        n_keep_min = 60, n_pred_draws = 400, b_draws_ll = 50,
        posterior_refresh = 1, adaptive_stop = false, rtol_outer = 1e-3,
        min_outer = 2, tau_update = :ig, tau_pc_rate = 0.0,
        ν_init = 50.0, α_init = 0.0,
        update_nu = false, update_alpha = false, refine_shape = false,
        shape_opt_tol = 1e-3, skew_location = :mean,
        nu_penalty = 0.0, nu_target = 50.0, alpha_penalty = 0.0)
end

function posterior_q_moments(fit::FoldFit, Arows_tr::AbstractMatrix)
    M = Arows_tr * fit.qsamps
    return vec(mean(M, dims = 2)), vec(var(M, dims = 2))
end

# The within-cell moment estimator of run_revision_sigmacal.jl / run_revision_hetero.jl.
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

# crps_from_samples of prediction_utils.jl, as copied in run_revision_invivo_check.jl.
function crps_samples(x::Vector{Float64}, y::Float64)
    m = length(x)
    xs = sort(x)
    term1 = mean(abs.(xs .- y))
    s = 0.0
    @inbounds for i in 1:m
        s += (2i - m - 1) * xs[i]
    end
    return term1 - (1.0 / (m * m)) * s
end

# ---- the forward rate ---------------------------------------------------------

"ln of the forward convenience rate with no product, from ln kcat+ and the substrate ln KMs."
function ln_forward_rate(lnk::Float64, lnKM::AbstractVector{Float64}, s::Vector{Int}, conc::Float64)
    acc = lnk
    @inbounds for i in eachindex(s)
        lna = log(conc) - lnKM[i]
        a = exp(lna)
        t = 0.0
        for m in 0:s[i]
            t += a^m
        end
        acc += s[i] * lna - log(t)
    end
    return acc
end

"d ln v / d(common shift of all ln KM) at the observed values; 0 if every substrate saturates."
function km_sensitivity(lnKM::AbstractVector{Float64}, s::Vector{Int}, conc::Float64)
    out = 0.0
    for i in eachindex(s)
        a = conc / exp(lnKM[i])
        num = sum(m * a^m for m in 0:s[i]); den = sum(a^m for m in 0:s[i])
        out += s[i] - num / den
    end
    return out
end

pit_midrank(x, z) = (count(<(z), x) + 0.5 * count(==(z), x)) / length(x)

# ---- pathway setup ----------------------------------------------------------------

const MDL = load_yeast_reconstruction(joinpath(dirname(@__DIR__), "data", "yeast-GEM.xml"))
const DICT_RXNS = reaction_data_dict(MDL)

function pathway_setup(pathway::Symbol)
    root = joinpath(dirname(@__DIR__), "output", string(pathway))
    brenda = load(joinpath(root, "data", "brenda_inputs_$(pathway).jld2"))
    post = load(joinpath(root, "data", "posterior_$(pathway).jld2"))
    y = Vector{Float64}(brenda["y"]); sigma2 = Vector{Float64}(brenda["σ2"])
    Arows = Matrix{Float64}(brenda["Arows"]); org = Vector{String}(brenda["grp"])
    meta = brenda["meta"]
    kinds = Vector{String}(getproperty(meta, :kind))
    ecs = string.(getproperty(meta, :ec_code))
    subs = string.(getproperty(meta, :substrate))
    qrows = Vector{Int}(getproperty(meta, :Q_row))
    q0 = vec(Array(post["q_prior"])); C0 = Matrix(Array(post["C_prior"]))
    eb = EBSpec(q0 = q0, C0 = C0, κ0 = KAPPA0)
    d = size(Arows, 2)
    A0, b0 = _prepare_information_pair(Arows, d, Matrix{Float64}(brenda["A0"]),
                                       Vector{Float64}(brenda["b0"]);
                                       deduplicate_thermo_block = true)
    all(==(sigma2[1]), sigma2) || error("record variances differ; the draws assume one convention")
    org_type = [string(org[i], "|", kinds[i]) for i in eachindex(org)]
    mm = mutant_mask(dirname(@__DIR__), CSVNAME_OF[pathway], y, org, kinds, ecs, subs)

    spec = PATHWAY_SPECS[string(pathway)]
    rxns = generate_reactions(spec.reaction_ids, DICT_RXNS, protons = false,
                              proton_name = spec.proton_name)
    Q = Matrix{Float64}(observation_matrix(rxns, h2o_strings = spec.h2o))
    size(Q, 2) == d || error("full Q has $(size(Q, 2)) columns, inputs have $d")
    Q[qrows, :] == Arows || error("stored design rows differ from the full Q")
    species_names, _ = rn_entities(rxns)
    (_, KM_map, kcatf, kcatr) = observation_matrix_index_maps(rxns)
    N = netstoichmat(reaction_network(rxns))
    specs = build_reaction_spec(rxns, spec.h2o, species_names, KM_map, kcatf, kcatr, N)

    # Units: organism and reaction with kcat+ records and records for every substrate KM.
    units = NamedTuple[]
    for o in sort(unique(org)), (j, sp) in enumerate(specs)
        isempty(sp.sub_km_row) && continue
        ik = findall(i -> org[i] == o && qrows[i] == sp.kcatf_row, eachindex(y))
        isempty(ik) && continue
        im = [findall(i -> org[i] == o && qrows[i] == r, eachindex(y)) for r in sp.sub_km_row]
        any(isempty, im) && continue
        push!(units, (organism = o, j = j, ik = ik, im = im))
    end
    return (pathway = pathway, y = y, sigma2 = sigma2, Arows = Arows, org = org,
            kinds = kinds, ecs = ecs, qrows = qrows, eb = eb, A0 = A0, b0 = b0,
            org_type = org_type, mutant = mm.mutant, Q = Q, specs = specs,
            spec = spec, units = units)
end

# ---- one fold ---------------------------------------------------------------------

function run_fold(P, f::Int, o::String, seed::UInt32, pw_pos::Int)
    y = P.y; sigma2 = P.sigma2; org = P.org
    te = findall(==(o), org); tr = findall(!=(o), org)
    idx95 = findfirst(x -> isapprox(x, 0.95; atol = 1e-8), REV_LEVELS)

    # (i) fixed effects, as loso_by_key runs the fold.
    spec_fe = fe_spec(P.pathway)
    rng = _pb_rng_from_seed(seed)
    hs = init_hyper(P.eb; λ0 = spec_fe.λ_init, τ²0 = spec_fe.τ²_init, π0 = spec_fe.π_init)
    fit_fe = fit_fold_mcem(rng, P.A0, P.b0, P.Arows[tr, :], y[tr], sigma2[tr], org[tr], P.eb, hs,
                           spec_fe; method = :collapsed)
    fe_cov = Bool[]
    for i in te
        p = predict_row_metrics_mc_hier(rng, view(P.Arows, i, :), y[i], sigma2[i], org[i],
                                        fit_fe, spec_fe; cover_levels = REV_LEVELS)
        push!(fe_cov, p.covered[idx95])
    end

    # (ii) pre-specified default, as the hetero fold loop runs global_cell.
    spec = pd_spec(P.pathway)
    rng_f = _pb_rng_from_seed(seed)
    Ar_tr = P.Arows[tr, :]
    ccell = 1.0
    hs0 = init_hyper(P.eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
    fit0 = fit_fold_mcem(rng_f, P.A0, P.b0, Ar_tr, y[tr], ccell .* sigma2[tr],
                         P.org_type[tr], P.eb, hs0, spec; method = :collapsed)
    for _ in 1:SIGMACAL_ITERS
        m, _ = posterior_q_moments(fit0, Ar_tr)
        ccell = c_update_within(y[tr], m, sigma2[tr], P.org_type[tr], eachindex(tr))
        hs0 = init_hyper(P.eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
        fit0 = fit_fold_mcem(rng_f, P.A0, P.b0, Ar_tr, y[tr], ccell .* sigma2[tr],
                             P.org_type[tr], P.eb, hs0, spec; method = :collapsed)
    end
    τ2 = fit0.hs.τ²
    S_pd = size(fit0.qsamps, 2)
    rng_p = _pb_rng_from_seed(seed)
    nte = length(te)
    Uprim = rand(rng_p, N_PRED, nte)
    Z1 = randn(rng_p, N_PRED, nte)
    Z2 = randn(rng_p, N_PRED, nte)
    lv = REV_LEVELS[idx95]
    pd_cov = Bool[]
    for (j, i) in enumerate(te)
        a = view(P.Arows, i, :)
        mu_s = [dot(a, view(fit0.qsamps, :, s)) for s in 1:S_pd]
        sd_i = sqrt(ccell * sigma2[i])
        draws = Vector{Float64}(undef, N_PRED)
        for r in 1:N_PRED
            draws[r] = mu_s[1 + floor(Int, Uprim[r, j] * S_pd)] +
                       sqrt(τ2) * Z1[r, j] + sd_i * Z2[r, j]
        end
        sort!(draws)
        lo = quantile(draws, (1 - lv) / 2; sorted = true)
        hi = quantile(draws, (1 + lv) / 2; sorted = true)
        push!(pd_cov, lo <= y[i] <= hi)
    end
    # Per-type between-organism variance of the released specification, on this fold.
    raw_t = tau_pm_by_kind(y, sigma2, fill(ccell, length(tr)), Ar_tr, fit0, P.org_type[tr],
                           P.kinds[tr], tr, τ2)
    τ2_rel, _ = tau2_by_type_with_fallback(raw_t, P.kinds[tr], τ2)
    t2k = get(τ2_rel, "kcat", τ2); t2m = get(τ2_rel, "KM", τ2)

    # ---- unit draws ----
    σ2 = sigma2[1]
    rd = MersenneTwister(DRAW_SEED + 1000 * pw_pos + f)
    S_fe = size(fit_fe.qsamps, 2)
    rows = NamedTuple[]
    for u in P.units
        u.organism == o || continue
        sp = P.specs[u.j]
        ns = length(sp.sub_km_row); s = sp.sub_stoich
        Rr = vcat(sp.kcatf_row, sp.sub_km_row)
        Mfe = P.Q[Rr, :] * fit_fe.qsamps
        Mpd = P.Q[Rr, :] * fit0.qsamps
        nk = length(u.ik); nm = length.(u.im)
        ybar_k = mean(y[u.ik]); ybar_m = [mean(y[ii]) for ii in u.im]
        z_obs = ln_forward_rate(ybar_k, ybar_m, s, CONC) / log(10)
        sens = km_sensitivity(ybar_m, s, CONC)
        U = rand(rd, NDRAW); Zk = randn(rd, NDRAW); Zm = randn(rd, NDRAW)
        Ek = randn(rd, NDRAW); Em = randn(rd, NDRAW, ns)
        col_fe = 1 .+ floor.(Int, U .* S_fe); col_pd = 1 .+ floor.(Int, U .* S_pd)
        # (variant, level, posterior block, columns, tau_kcat, tau_KM, record scale)
        variants = [("spb", "record", Mfe, col_fe, 0.0, 0.0, 1.0),
                    ("spb", "organism", Mfe, col_fe, 0.0, 0.0, 0.0),
                    ("default", "record", Mpd, col_pd, τ2, τ2, ccell),
                    ("default", "organism", Mpd, col_pd, τ2, τ2, 0.0),
                    ("velocity", "record", Mpd, col_pd, τ2, 0.0, ccell),
                    ("velocity", "organism", Mpd, col_pd, τ2, 0.0, 0.0),
                    ("per_type", "record", Mpd, col_pd, t2k, t2m, ccell),
                    ("per_type", "organism", Mpd, col_pd, t2k, t2m, 0.0)]
        for (vname, level, M, col, tk, tm, cc) in variants
            x = Vector{Float64}(undef, NDRAW)
            lk = Vector{Float64}(undef, NDRAW); lm = Matrix{Float64}(undef, NDRAW, ns)
            buf = Vector{Float64}(undef, ns)
            for r in 1:NDRAW
                c = col[r]
                lk[r] = M[1, c] + sqrt(tk) * Zk[r] + sqrt(cc * σ2 / nk) * Ek[r]
                for i in 1:ns
                    buf[i] = M[1 + i, c] + sqrt(tm) * Zm[r] + sqrt(cc * σ2 / nm[i]) * Em[r, i]
                    lm[r, i] = buf[i]
                end
                x[r] = ln_forward_rate(lk[r], buf, s, CONC) / log(10)
            end
            qs = quantile(x, [0.025, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.975])
            cov_at(l) = (k = findfirst(==(round((1 - l) / 2, digits = 4)), [0.025, 0.05, 0.10, 0.25]);
                         qs[k] <= z_obs <= qs[end + 1 - k])
            in95(v, z) = quantile(v, 0.025) <= z <= quantile(v, 0.975)
            push!(rows, (pathway = string(P.pathway), fold = f, organism = o,
                reaction_id = sp.name, ec_code = P.spec.ec_codes[sp.name],
                n_substrates = ns, n_kcat_records = nk, n_KM_records = sum(nm),
                any_mutant = any(P.mutant[u.ik]) || any(any(P.mutant[ii]) for ii in u.im),
                log10_rate_obs = z_obs, km_sensitivity = sens,
                variant = vname, level = level,
                tau2_kcat = tk, tau2_KM = tm, c_record_scale = level == "record" ? cc : NaN,
                lo95 = qs[1], median = qs[5], hi95 = qs[end],
                covered50 = cov_at(0.5), covered80 = cov_at(0.8),
                covered90 = cov_at(0.9), covered95 = cov_at(0.95),
                pit = pit_midrank(x, z_obs), width95_decades = qs[end] - qs[1],
                crps_log10 = crps_samples(x, z_obs),
                error_log10 = z_obs - qs[5],
                kcat_mean_covered95 = in95(lk, ybar_k),
                frac_KM_means_covered95 = mean(in95(view(lm, :, i), ybar_m[i]) for i in 1:ns)))
        end
    end
    reg = (pathway = string(P.pathway), fold = f, organism = o, n_records = nte,
           fe_cov95 = mean(fe_cov), pd_cov95 = mean(pd_cov),
           c_record_scale = ccell, tau2_pooled = τ2, tau2_kcat = t2k, tau2_KM = t2m,
           S_fe = S_fe, S_pd = S_pd)
    return rows, reg, fe_cov, pd_cov, te
end

# ---- main ---------------------------------------------------------------------------

println("Revision rate hold-out")
println("  threads  = $(Threads.nthreads()) (BLAS 1)")
println("  pathways = $(REV_PATHWAYS), draws per unit = $(NDRAW), substrate concentration = $(CONC) M")
println("  started  = $(now())")
flush(stdout)

all_rows = NamedTuple[]; regs = NamedTuple[]
for pw in REV_PATHWAYS
    t0 = time()
    P = pathway_setup(pw)
    folds = sort!(unique(P.org)); nf = length(folds)
    seeds = _pb_thread_seeds(MersenneTwister(REV_SEED), nf)
    orgs = sort(unique([u.organism for u in P.units]))
    @printf("\n=== %s: %d organisms, %d units from %d organisms\n", pw, nf, length(P.units), length(orgs))
    flush(stdout)
    pw_pos = findfirst(==(pw), ALL_PATHWAYS)
    res = Vector{Any}(undef, length(orgs))
    Threads.@threads :dynamic for k in eachindex(orgs)
        f = findfirst(==(orgs[k]), folds)
        res[k] = run_fold(P, f, orgs[k], seeds[f], pw_pos)
    end
    # Regression against the saved per-record files, where they exist.
    results_dir = joinpath(dirname(@__DIR__), "output", string(pw), "results", "revision")
    f_fe = joinpath(results_dir, "realdata_percell_$(pw).jld2")
    f_pd = joinpath(results_dir, "hetero_percell_$(pw).jld2")
    have_ref = isfile(f_fe) && isfile(f_pd)
    cov_fe = cov_pd = nothing
    if have_ref
        ref = load(f_fe)
        ref["org"] == P.org || error("realdata_percell organism vector differs from the inputs")
        cov_fe = ref["percell"]["gaussian_Fix"].covered95
        href = load(f_pd)
        href["org"] == P.org || error("hetero_percell organism vector differs from the inputs")
        cov_pd = href["cov95"]["global_cell"]
    end
    for k in eachindex(orgs)
        rows, reg, fe_cov, pd_cov, te = res[k]
        append!(all_rows, rows)
        push!(regs, merge(reg, (fe_identical = have_ref ? fe_cov == cov_fe[te] : missing,
                                pd_identical = have_ref ? pd_cov == cov_pd[te] : missing)))
    end
    if have_ref
        nid = count(r -> r.pathway == string(pw) && r.fe_identical && r.pd_identical, regs)
        @printf("  regression: %d of %d folds reproduce the published per-record coverage of both models [%.1f min]\n",
                nid, length(orgs), (time() - t0) / 60)
    else
        @printf("  regression skipped: the per-record files of the published runs are absent [%.1f min]\n",
                (time() - t0) / 60)
    end
    flush(stdout)
end
udf = DataFrame(all_rows); rdf = DataFrame(regs)

# ---- summary ----
"Organism-clustered bootstrap interval for the mean of `v` (clusters `cl`)."
function cluster_boot(v::AbstractVector, cl::AbstractVector; nboot = NBOOT, seed = 1)
    keys_ = unique(cl); byk = Dict(k => findall(==(k), cl) for k in keys_)
    rng = MersenneTwister(seed); m = Float64[]
    for _ in 1:nboot
        idx = reduce(vcat, [byk[keys_[rand(rng, 1:length(keys_))]] for _ in keys_])
        push!(m, mean(v[idx]))
    end
    return quantile(m, 0.025), quantile(m, 0.975)
end

srows = NamedTuple[]
strata = [("all", r -> true), ("km_sensitive", r -> r.km_sensitivity >= 0.5),
          ("saturated", r -> r.km_sensitivity < 0.5), ("no_mutant", r -> !r.any_mutant)]
for pw in vcat(string.(REV_PATHWAYS), "pooled"), (vname, level) in
        [(v, l) for v in ("spb", "default", "velocity", "per_type") for l in ("record", "organism")],
        (stname, stf) in strata
    m = (udf.variant .== vname) .& (udf.level .== level)
    pw == "pooled" || (m .&= udf.pathway .== pw)
    d = udf[m, :]
    d = d[[stf(r) for r in eachrow(d)], :]
    nrow(d) == 0 && continue
    cl = string.(d.pathway, "|", d.organism)
    ci = cluster_boot(Float64.(d.covered95), cl)
    push!(srows, (pathway = pw, variant = vname, level = level, stratum = stname,
        n_units = nrow(d), n_organisms = length(unique(cl)),
        cov50 = mean(d.covered50), cov80 = mean(d.covered80), cov90 = mean(d.covered90),
        cov95 = mean(d.covered95), cov95_ci_lo = ci[1], cov95_ci_hi = ci[2],
        mean_pit = mean(d.pit), median_width95_decades = median(d.width95_decades),
        mean_crps_log10 = mean(d.crps_log10), median_error_log10 = median(d.error_log10),
        kcat_mean_cov95 = mean(d.kcat_mean_covered95),
        KM_means_cov95 = mean(d.frac_KM_means_covered95)))
end
sdf = DataFrame(srows)

mkpath(OUTDIR)
CSV.write(joinpath(OUTDIR, "rate_holdout_units$(OUT_SUFFIX).csv"), udf)
CSV.write(joinpath(OUTDIR, "rate_holdout_summary$(OUT_SUFFIX).csv"), sdf)
CSV.write(joinpath(OUTDIR, "rate_holdout_regression$(OUT_SUFFIX).csv"), rdf)
@printf("\nRegression: %d of %d folds identical for fixed effects, %d for the default (%d not checked)\n",
        count(isequal(true), rdf.fe_identical), nrow(rdf), count(isequal(true), rdf.pd_identical),
        count(ismissing, rdf.fe_identical))
println("\nRecord-level rate coverage, all units")
show(stdout, sdf[(sdf.level .== "record") .& (sdf.stratum .== "all"), :]; allrows = true, allcols = true)
println("\n\nRecord-level rate coverage by stratum, pooled")
show(stdout, sdf[(sdf.level .== "record") .& (sdf.pathway .== "pooled"), :]; allrows = true, allcols = true)
println("\n\nOrganism-level interval widths, pooled")
show(stdout, select(sdf[(sdf.level .== "organism") .& (sdf.pathway .== "pooled") .& (sdf.stratum .== "all"), :],
                    :variant, :n_units, :median_width95_decades, :cov95); allrows = true, allcols = true)
println("\nDone at $(now())")
