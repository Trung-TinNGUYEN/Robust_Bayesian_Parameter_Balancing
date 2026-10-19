# run_revision_ecoli_holdout.jl
#
# Escherichia coli hold-out with joint posterior predictive draws, added for
# the revision.
#
# Purpose.  An external check of the released intervals needs predictions for
# an organism whose records the fit has not seen.  E. coli is the natural
# choice: it has in vivo turnover estimates (Davidi et al. 2016, used by
# run_revision_invivo_check.jl) that are independent of BRENDA, but it also has
# BRENDA records in every pathway, so a fit on all records would not be a
# prediction.  This driver removes every E. coli record of a pathway, refits
# two models on the remaining records, and draws joint predictions for every
# forward turnover number (kcat+ row of the full observation matrix Q).
#
# The fold is the published leave-one-source-out fold.  Organisms are sorted,
# the fold seeds come from _pb_thread_seeds(MersenneTwister(20260805), n_folds)
# and the E. coli fold uses its own seed, exactly as loso_by_key
# (run_revision_realdata.jl) and the fold loop of run_revision_hetero.jl do.
# The per-record 95% coverage of the held-out E. coli records therefore
# reproduces the saved per-record files, and the driver checks that it does:
#
#   (i)  fixed effects: cell gaussian_Fix of run_revision_realdata.jl (Gaussian
#        likelihood, no random effect, adaptive MCEM stop), compared with
#        realdata_percell_<pathway>.jld2 (coverage, CRPS, log score, error);
#   (ii) pre-specified default: construction global_cell of
#        run_revision_hetero.jl (organism-by-type random intercept; record
#        scale c from the within-cell iteration, one fit at c = 1 then
#        PB_REV_SIGMACAL_ITERS refits, clamped to [C_MIN, C_MAX]; pooled
#        between-organism variance tau^2 = the fitted scalar), compared with
#        hetero_percell_<pathway>.jld2 (coverage and interval width).
#
# Both use kappa0 = 50 (the prior mean is updated), the value of the published
# runs; the regression holds only at that value.  The fitting helpers below are
# verbatim copies from those drivers, which have no main guard and so cannot be
# included.
#
# Joint predictive draws.  For every forward kcat row j (and the reverse rows
# with PB_ECOLI_REVERSE=1) and draw r = 1..NDRAW, with one posterior column s_r
# and one E. coli offset eta_r shared by all rows of the pathway, and
# independent record errors:
#
#   fixed effects, organism level   a_j' q_{s_r}                   (the consensus)
#   fixed effects, record level     a_j' q_{s_r} + sigma e_rj
#   default, organism level         a_j' q_{s_r} + eta_r,          eta_r = tau z_r
#   default, record level           a_j' q_{s_r} + eta_r + sqrt(c) sigma e_rj
#
# with sigma^2 = (0.5 ln 10)^2 the record convention.  The two models share the
# random primitives (uniform u_r for the column, s_r = 1 + floor(u_r S), z_r
# and e_rj), so contrasts between them carry no extra Monte Carlo noise.  Each
# row a_j is taken from the full Q built from yeast-GEM (build_pathway_inputs.jl),
# which is checked to equal the stored design rows where the row has records;
# rows without any record (TCA r_0280, r_0831, r_0505; pentose phosphate
# r_0091) are predicted from Q alone.
#
# Outputs, written under output/<pathway>/results/revision/:
#   ecoli_holdout_quantities_<pathway>.csv  one row per target kcat row:
#       record counts, E. coli held-out medians, and 2.5/50/97.5% quantiles of
#       the four predictive distributions in ln(1/s) and in 1/s
#   ecoli_holdout_records_<pathway>.csv     one row per held-out E. coli record:
#       value, mutant flag, reproduced and published 95% coverage, and the joint
#       draw record-level intervals where the record lies on a target row
#   ecoli_holdout_draws_<pathway>.jld2      the NDRAW x rows draw matrices, the
#       target rows of Q, the kept posterior draws of both fits, c, tau^2,
#       lambda, seeds and the regression result
#
# Environment controls:
#   PB_REV_PATHWAYS        subset of {glycolysis,tca,pentose_phosphate,shikimate}
#                          (default all four)
#   PB_ECOLI_LABELS        exact (default): hold out "escherichia coli" only, the
#                          published fold, with the regression check;
#                          k12: also hold out "escherichia coli k-12" (present in
#                          TCA only); the check is skipped where the held-out set
#                          differs from the published fold, and the outputs get
#                          the suffix _k12
#   PB_ECOLI_NDRAW         joint draws per pathway (default 4000)
#   PB_ECOLI_DRAW_SEED     base seed of the joint draws (default 20260918; the
#                          pathway's position in the full list is added)
#   PB_ECOLI_REVERSE       1 to add the reverse kcat rows (default 0)
#   PB_EB_KAPPA0           prior-mean concentration (default 50, as published)
#   PB_REV_SIGMACAL_ITERS  within-cell refits for c (default 4, as published)
#   PB_REV_SUFFIX          output-name suffix (overrides the _k12 default)
#   PB_REV_OUTER_MULT      multiplies the number of outer EM iterations of both
#                          fits (default 1, as published), for the EM-iteration
#                          sensitivity run (S1 Text); the regression against the
#                          published per-record files holds only at 1
#
# Run (single thread is enough, about two minutes):
#   julia --project=. --threads=1 scripts/run_revision_ecoli_holdout.jl

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

const REV_LEVELS = vcat(0.01, collect(0.05:0.05:0.95), 0.99)
const REV_SEED = 20260805                 # published fold seed (realdata, hetero)
const ALL_PATHWAYS = [:glycolysis, :tca, :pentose_phosphate, :shikimate]
const REV_PATHWAYS = Symbol.(strip.(split(get(ENV, "PB_REV_PATHWAYS",
    "glycolysis,tca,pentose_phosphate,shikimate"), ",")))
const LABEL_MODE = lowercase(get(ENV, "PB_ECOLI_LABELS", "exact"))
LABEL_MODE in ("exact", "k12") || error("PB_ECOLI_LABELS must be exact or k12")
const ECOLI = "escherichia coli"
const HELDOUT = LABEL_MODE == "exact" ? [ECOLI] : [ECOLI, "escherichia coli k-12"]
const NDRAW = parse(Int, get(ENV, "PB_ECOLI_NDRAW", "4000"))
const DRAW_SEED = parse(Int, get(ENV, "PB_ECOLI_DRAW_SEED", "20260918"))
const WITH_REVERSE = get(ENV, "PB_ECOLI_REVERSE", "0") == "1"
const KAPPA0 = parse(Float64, get(ENV, "PB_EB_KAPPA0", "50.0"))
const SIGMACAL_ITERS = parse(Int, get(ENV, "PB_REV_SIGMACAL_ITERS", "4"))
const OUT_SUFFIX = get(ENV, "PB_REV_SUFFIX", LABEL_MODE == "exact" ? "" : "_k12")
const OUTER_MULT = parse(Int, get(ENV, "PB_REV_OUTER_MULT", "1"))
OUTER_MULT == 1 || haskey(ENV, "PB_REV_SUFFIX") || error("PB_REV_OUTER_MULT is not 1: set PB_REV_SUFFIX as well, so that the published outputs are kept")
const C_MIN, C_MAX = 0.25, 40.0           # run_revision_hetero.jl clamp
const N_PRED = 800                        # run_revision_hetero.jl draws per record

# ---- copies of the published helpers ----------------------------------------

# cell_spec(:gaussian, false, pathway) of run_revision_realdata.jl (cell gaussian_Fix).
function fe_spec(pathway::Symbol)
    is_tca = pathway == :tca
    return FoldFitSpec(
        model = :gaussian, hierarchical = false,
        n_outer = OUTER_MULT * (is_tca ? 4 : 3), n_mcmc = is_tca ? 250 : 150, burnin = is_tca ? 80 : 40,
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
        n_outer = OUTER_MULT * (slow ? 4 : 3), n_mcmc = slow ? 250 : 150,
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

qtl(v) = quantile(v, [0.025, 0.5, 0.975])
maxabsdiff(a, b) = isempty(a) ? 0.0 : maximum(abs.(a .- b))

# ---- full observation matrix -------------------------------------------------

const MDL = load_yeast_reconstruction(joinpath(dirname(@__DIR__), "data", "yeast-GEM.xml"))
const DICT_RXNS = reaction_data_dict(MDL)

function full_design(pathway::Symbol)
    spec = PATHWAY_SPECS[string(pathway)]
    rxns = generate_reactions(spec.reaction_ids, DICT_RXNS, protons = false,
                              proton_name = spec.proton_name)
    Q = Matrix{Float64}(observation_matrix(rxns, h2o_strings = spec.h2o))
    (_, _, kf, kr) = observation_matrix_index_maps(rxns)
    return (spec = spec, Q = Q,
            fwd = [kf[r] for r in spec.reaction_ids],
            rev = [kr[r] for r in spec.reaction_ids])
end

# ---- one pathway -------------------------------------------------------------

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
    subs = string.(getproperty(meta, :substrate))
    qrows = Vector{Int}(getproperty(meta, :Q_row))
    q0 = vec(Array(post["q_prior"])); C0 = Matrix(Array(post["C_prior"]))
    eb = EBSpec(q0 = q0, C0 = C0, κ0 = KAPPA0)
    d = size(Arows, 2)
    A0, b0 = _prepare_information_pair(Arows, d, Matrix{Float64}(brenda["A0"]),
                                       Vector{Float64}(brenda["b0"]);
                                       deduplicate_thermo_block = true)
    org_type = [string(org[i], "|", kinds[i]) for i in eachindex(org)]
    all(==(sigma2[1]), sigma2) || error("record variances differ; the draws assume one convention")
    σ2 = sigma2[1]
    mm = mutant_mask(dirname(@__DIR__), CSVNAME_OF[pathway], y, org, kinds, ecs, subs)

    # Full Q, and the check that the stored design rows are its rows.
    fd = full_design(pathway)
    size(fd.Q, 2) == d || error("full Q has $(size(fd.Q, 2)) columns, inputs have $d")
    fd.Q[qrows, :] == Arows || error("stored design rows differ from the full Q")
    nr = length(fd.spec.reaction_ids)
    fd.fwd == collect(d - nr + 1:d) || error("unexpected kcat+ row layout")
    fd.rev == collect(d + 1:d + nr) || error("unexpected kcat- row layout")

    # The published E. coli fold.
    folds = sort!(unique(org)); nf = length(folds)
    f = findfirst(==(ECOLI), folds)
    f === nothing && error("no records labelled '$ECOLI' in $pathway")
    seeds = _pb_thread_seeds(MersenneTwister(REV_SEED), nf)
    held = intersect(HELDOUT, folds)
    te = findall(in(held), org); tr = findall(!in(held), org)
    same_fold = te == findall(==(folds[f]), org)
    nte = length(te)
    @printf("\n=== %s: N=%d, organisms=%d, E. coli fold %d of %d, held out %s: %d records (%d kcat), training %d records from %d organisms\n",
            pathway, length(y), nf, f, nf, join(repr.(held), " + "), nte,
            count(i -> kinds[i] == "kcat", te), length(tr), length(unique(org[tr])))
    flush(stdout)
    idx95 = findfirst(x -> isapprox(x, 0.95; atol = 1e-8), REV_LEVELS)

    # ---- (i) fixed effects, as loso_by_key runs the fold -----------------------
    t0 = time()
    spec_fe = fe_spec(pathway)
    rng = _pb_rng_from_seed(seeds[f])
    hs = init_hyper(eb; λ0 = spec_fe.λ_init, τ²0 = spec_fe.τ²_init, π0 = spec_fe.π_init)
    fit_fe = fit_fold_mcem(rng, A0, b0, Arows[tr, :], y[tr], sigma2[tr], org[tr], eb, hs,
                           spec_fe; method = :collapsed)
    fe_cov = Bool[]; fe_crps = Float64[]; fe_ll = Float64[]; fe_err = Float64[]
    for i in te
        p = predict_row_metrics_mc_hier(rng, view(Arows, i, :), y[i], sigma2[i], org[i],
                                        fit_fe, spec_fe; cover_levels = REV_LEVELS)
        push!(fe_cov, p.covered[idx95]); push!(fe_crps, p.crps)
        push!(fe_ll, p.loglik); push!(fe_err, p.pred_error)
    end
    t_fe = time() - t0

    # ---- (ii) pre-specified default, as the hetero fold loop runs global_cell --
    t0 = time()
    spec = pd_spec(pathway)
    rng_f = _pb_rng_from_seed(seeds[f])
    Ar_tr = Arows[tr, :]
    ccell = 1.0
    hs0 = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
    fit0 = fit_fold_mcem(rng_f, A0, b0, Ar_tr, y[tr], ccell .* sigma2[tr],
                         org_type[tr], eb, hs0, spec; method = :collapsed)
    c_trace = [ccell]
    for _ in 1:SIGMACAL_ITERS
        m, _ = posterior_q_moments(fit0, Ar_tr)
        ccell = c_update_within(y[tr], m, sigma2[tr], org_type[tr], eachindex(tr))
        push!(c_trace, ccell)
        hs0 = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
        fit0 = fit_fold_mcem(rng_f, A0, b0, Ar_tr, y[tr], ccell .* sigma2[tr],
                             org_type[tr], eb, hs0, spec; method = :collapsed)
    end
    t_pd = time() - t0
    τ2 = fit0.hs.τ²
    S_pd = size(fit0.qsamps, 2)
    rng_p = _pb_rng_from_seed(seeds[f])
    Uprim = rand(rng_p, N_PRED, nte)
    Z1 = randn(rng_p, N_PRED, nte)
    Z2 = randn(rng_p, N_PRED, nte)
    lv = REV_LEVELS[idx95]
    pd_cov = Bool[]; pd_w95 = Float64[]
    for (j, i) in enumerate(te)
        a = view(Arows, i, :)
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
        push!(pd_cov, lo <= y[i] <= hi); push!(pd_w95, hi - lo)
    end

    # Post hoc diagnostic only (not used in any draw): the per-type
    # between-organism variance of the released specification's estimator on
    # this training fold, against the pooled value the default uses.
    raw_t = tau_pm_by_kind(y, sigma2, fill(ccell, length(tr)), Ar_tr, fit0, org_type[tr],
                           kinds[tr], tr, τ2)
    τ2_rel, τ2_src = tau2_by_type_with_fallback(raw_t, kinds[tr], τ2)

    # ---- regression against the saved per-record files -------------------------
    reg = Dict{String,Any}("checked" => same_fold)
    if same_fold
        ref = load(joinpath(results_dir, "realdata_percell_$(pathway).jld2"))
        ref["org"] == org || error("realdata_percell organism vector differs from the inputs")
        rc = ref["percell"]["gaussian_Fix"]
        href = load(joinpath(results_dir, "hetero_percell_$(pathway).jld2"))
        href["org"] == org || error("hetero_percell organism vector differs from the inputs")
        reg["fe_cov95_published"] = Vector{Bool}(rc.covered95[te])
        reg["fe_identical"] = fe_cov == rc.covered95[te]
        reg["fe_maxdiff_crps"] = maxabsdiff(fe_crps, rc.crps[te])
        reg["fe_maxdiff_loglik"] = maxabsdiff(fe_ll, rc.loglik[te])
        reg["fe_maxdiff_error"] = maxabsdiff(fe_err, rc.pred_error[te])
        reg["pd_cov95_published"] = Vector{Bool}(href["cov95"]["global_cell"][te])
        reg["pd_identical"] = pd_cov == href["cov95"]["global_cell"][te]
        reg["pd_maxdiff_width95"] = maxabsdiff(pd_w95, href["width95"]["global_cell"][te])
        @printf("  regression, fixed effects: E. coli Cov95 %.4f (published %.4f), identical=%s, max |diff| CRPS %.1e, log score %.1e, error %.1e\n",
                mean(fe_cov), mean(rc.covered95[te]), reg["fe_identical"],
                reg["fe_maxdiff_crps"], reg["fe_maxdiff_loglik"], reg["fe_maxdiff_error"])
        @printf("  regression, default:       E. coli Cov95 %.4f (published %.4f), identical=%s, max |diff| width95 %.1e\n",
                mean(pd_cov), mean(reg["pd_cov95_published"]), reg["pd_identical"],
                reg["pd_maxdiff_width95"])
    else
        @printf("  regression not applicable (held-out set differs from the published fold); E. coli Cov95 fixed effects %.4f, default %.4f\n",
                mean(fe_cov), mean(pd_cov))
    end
    @printf("  fixed effects: S=%d, lambda=%.4f [%.1f s]; default: S=%d, c=%.3f (trace %s), pooled tau2=%.3f, lambda=%.4f [%.1f s]\n",
            size(fit_fe.qsamps, 2), fit_fe.hs.λ, t_fe, S_pd, ccell,
            join((@sprintf("%.3f", c) for c in c_trace), " -> "), τ2, fit0.hs.λ, t_pd)
    @printf("  diagnostic only: per-type tau2 kcat %.3f (%s), KM %.3f (%s)\n",
            get(raw_t, "kcat", NaN), get(τ2_src, "kcat", "absent"),
            get(raw_t, "KM", NaN), get(τ2_src, "KM", "absent"))
    flush(stdout)

    # ---- joint predictive draws for the target kcat rows ------------------------
    tq = WITH_REVERSE ? vcat(fd.fwd, fd.rev) : copy(fd.fwd)
    ridx = WITH_REVERSE ? vcat(1:nr, 1:nr) : collect(1:nr)
    tdir = WITH_REVERSE ? vcat(fill("forward", nr), fill("reverse", nr)) : fill("forward", nr)
    J = length(tq)
    At = fd.Q[tq, :]
    M_fe = At * fit_fe.qsamps
    M_pd = At * fit0.qsamps
    S_fe = size(fit_fe.qsamps, 2)
    pw_pos = findfirst(==(pathway), ALL_PATHWAYS)
    dseed = DRAW_SEED + pw_pos
    rd = MersenneTwister(dseed)
    u = rand(rd, NDRAW)
    z = randn(rd, NDRAW)
    e = randn(rd, NDRAW, J)
    col_fe = 1 .+ floor.(Int, u .* S_fe)
    col_pd = 1 .+ floor.(Int, u .* S_pd)
    eta = sqrt(τ2) .* z
    fe_org = Matrix{Float64}(undef, NDRAW, J); pd_org = similar(fe_org)
    for jj in 1:J, r in 1:NDRAW
        fe_org[r, jj] = M_fe[jj, col_fe[r]]
        pd_org[r, jj] = M_pd[jj, col_pd[r]] + eta[r]
    end
    fe_rec = fe_org .+ sqrt(σ2) .* e
    pd_rec = pd_org .+ sqrt(ccell * σ2) .* e

    # ---- per-quantity table -------------------------------------------------------
    qs = Dict(:fe_org => fe_org, :fe_rec => fe_rec, :pd_org => pd_org, :pd_rec => pd_rec)
    rows = NamedTuple[]
    for jj in 1:J
        qr = tq[jj]; rid = fd.spec.reaction_ids[ridx[jj]]
        ref_sub = tdir[jj] == "forward" ? fd.spec.forward_metabolites[rid] :
                                          fd.spec.reverse_metabolites[rid]
        iall = findall(i -> qrows[i] == qr, eachindex(y))
        itr = [i for i in iall if !(org[i] in held)]
        ite = [i for i in iall if org[i] in held]
        iwt = [i for i in ite if !mm.mutant[i]]
        qv = Dict(k => qtl(view(v, :, jj)) for (k, v) in qs)
        nin(k) = count(i -> qv[k][1] <= y[i] <= qv[k][3], ite)
        push!(rows, (pathway = string(pathway), Q_row = qr, reaction_id = rid,
            direction = tdir[jj], ec_code = fd.spec.ec_codes[rid], reference_substrate = ref_sub,
            design_source = isempty(iall) ? "full_Q_no_records" : "records",
            n_records_all = length(iall), n_organisms_all = length(unique(org[iall])),
            n_records_train = length(itr), n_organisms_train = length(unique(org[itr])),
            n_ecoli = length(ite), n_ecoli_mutant = count(i -> mm.mutant[i], ite),
            ecoli_median_ln = isempty(ite) ? NaN : median(y[ite]),
            ecoli_median_ln_nonmutant = isempty(iwt) ? NaN : median(y[iwt]),
            n_ecoli_in_fe_rec95 = nin(:fe_rec), n_ecoli_in_pd_rec95 = nin(:pd_rec),
            fe_org_lo = qv[:fe_org][1], fe_org_med = qv[:fe_org][2], fe_org_hi = qv[:fe_org][3],
            fe_rec_lo = qv[:fe_rec][1], fe_rec_med = qv[:fe_rec][2], fe_rec_hi = qv[:fe_rec][3],
            pd_org_lo = qv[:pd_org][1], pd_org_med = qv[:pd_org][2], pd_org_hi = qv[:pd_org][3],
            pd_rec_lo = qv[:pd_rec][1], pd_rec_med = qv[:pd_rec][2], pd_rec_hi = qv[:pd_rec][3],
            fe_org_lo_1s = exp(qv[:fe_org][1]), fe_org_med_1s = exp(qv[:fe_org][2]),
            fe_org_hi_1s = exp(qv[:fe_org][3]),
            fe_rec_lo_1s = exp(qv[:fe_rec][1]), fe_rec_med_1s = exp(qv[:fe_rec][2]),
            fe_rec_hi_1s = exp(qv[:fe_rec][3]),
            pd_org_lo_1s = exp(qv[:pd_org][1]), pd_org_med_1s = exp(qv[:pd_org][2]),
            pd_org_hi_1s = exp(qv[:pd_org][3]),
            pd_rec_lo_1s = exp(qv[:pd_rec][1]), pd_rec_med_1s = exp(qv[:pd_rec][2]),
            pd_rec_hi_1s = exp(qv[:pd_rec][3]),
            c_record_scale = ccell, tau2_pooled = τ2))
    end
    qdf = DataFrame(rows)

    # ---- per-record table -----------------------------------------------------------
    col_of = Dict(q => jj for (jj, q) in enumerate(tq))
    recs = NamedTuple[]
    for (k, i) in enumerate(te)
        jj = kinds[i] == "kcat" ? get(col_of, qrows[i], 0) : 0
        dirn = kinds[i] != "kcat" ? "KM" : (qrows[i] <= d ? "forward" : "reverse")
        rid = kinds[i] == "kcat" ? fd.spec.reaction_ids[qrows[i] <= d ? qrows[i] - (d - nr) : qrows[i] - d] : ""
        iv(key) = jj == 0 ? (NaN, NaN) : (qtl(view(qs[key], :, jj))[1], qtl(view(qs[key], :, jj))[3])
        fr = iv(:fe_rec); pr = iv(:pd_rec)
        push!(recs, (pathway = string(pathway), record_index = i, organism = org[i],
            kind = kinds[i], ec_code = ecs[i], substrate = subs[i], Q_row = qrows[i],
            reaction_id = rid, direction = dirn, y_ln = y[i],
            mutant = mm.mutant[i], wildtype_labelled = mm.wildtype[i],
            fe_cov95_fold = fe_cov[k],
            fe_cov95_published = same_fold ? reg["fe_cov95_published"][k] : missing,
            pd_cov95_fold = pd_cov[k],
            pd_cov95_published = same_fold ? reg["pd_cov95_published"][k] : missing,
            pd_width95_fold = pd_w95[k],
            joint_fe_rec_lo = fr[1], joint_fe_rec_hi = fr[2],
            joint_in_fe_rec95 = jj == 0 ? missing : fr[1] <= y[i] <= fr[2],
            joint_pd_rec_lo = pr[1], joint_pd_rec_hi = pr[2],
            joint_in_pd_rec95 = jj == 0 ? missing : pr[1] <= y[i] <= pr[2]))
    end
    rdf = DataFrame(recs)

    CSV.write(joinpath(results_dir, "ecoli_holdout_quantities_$(pathway)$(OUT_SUFFIX).csv"), qdf)
    CSV.write(joinpath(results_dir, "ecoli_holdout_records_$(pathway)$(OUT_SUFFIX).csv"), rdf)
    jldsave(joinpath(results_dir, "ecoli_holdout_draws_$(pathway)$(OUT_SUFFIX).jld2");
            fe_org, fe_rec, pd_org, pd_rec,
            Q_rows = tq, reaction_ids = fd.spec.reaction_ids[ridx], directions = tdir,
            A_target = At, fe_qsamps = fit_fe.qsamps, pd_qsamps = fit0.qsamps,
            col_fe, col_pd, eta,
            c_record_scale = ccell, c_trace, tau2_pooled = τ2,
            tau2_per_type_diagnostic = raw_t, tau2_rel_diagnostic = τ2_rel,
            lambda_fe = fit_fe.hs.λ, lambda_pd = fit0.hs.λ, sigma2 = σ2,
            heldout_labels = held, heldout_index = te, train_index = tr,
            fold_index = f, n_folds = nf, fold_seed = Int(seeds[f]), draw_seed = dseed,
            ndraw = NDRAW, kappa0 = KAPPA0, sigmacal_iters = SIGMACAL_ITERS,
            regression = reg)

    fw = qdf[qdf.direction .== "forward", :]
    @printf("  %d target rows (%d forward, %d without records); median 95%% width in decades: fixed effects organism %.2f, record %.2f; default organism %.2f, record %.2f\n",
            nrow(qdf), nrow(fw), count(==("full_Q_no_records"), qdf.design_source),
            median((fw.fe_org_hi .- fw.fe_org_lo) ./ log(10)),
            median((fw.fe_rec_hi .- fw.fe_rec_lo) ./ log(10)),
            median((fw.pd_org_hi .- fw.pd_org_lo) ./ log(10)),
            median((fw.pd_rec_hi .- fw.pd_rec_lo) ./ log(10)))
    kk = findall(r -> r.kind == "kcat" && !ismissing(r.joint_in_pd_rec95), eachrow(rdf))
    if !isempty(kk)
        @printf("  held-out E. coli kcat records on target rows: %d; inside joint record-level 95%%: fixed effects %d, default %d\n",
                length(kk), count(rdf.joint_in_fe_rec95[kk]), count(rdf.joint_in_pd_rec95[kk]))
    end
    show(stdout, select(qdf, :Q_row, :reaction_id, :direction, :n_records_train, :n_ecoli,
                        :ecoli_median_ln, :fe_org_lo, :fe_org_hi, :fe_rec_lo, :fe_rec_hi,
                        :pd_org_lo, :pd_org_hi, :pd_rec_lo, :pd_rec_hi);
         allrows = true, allcols = true)
    println()
    flush(stdout)
    return (pathway = string(pathway), fold = f, n_folds = nf, n_test = nte,
            n_train = length(tr), checked = same_fold,
            fe_cov95 = mean(fe_cov), pd_cov95 = mean(pd_cov),
            fe_identical = get(reg, "fe_identical", missing),
            pd_identical = get(reg, "pd_identical", missing),
            fe_maxdiff_crps = get(reg, "fe_maxdiff_crps", missing),
            pd_maxdiff_width95 = get(reg, "pd_maxdiff_width95", missing),
            S_fe = S_fe, S_pd = S_pd, lambda_fe = fit_fe.hs.λ, c = ccell, tau2 = τ2,
            tau2_kcat_per_type_diag = get(raw_t, "kcat", NaN))
end

println("Revision E. coli hold-out")
println("  threads  = $(Threads.nthreads())")
println("  pathways = $(REV_PATHWAYS)")
println("  held out = $(HELDOUT) (mode $(LABEL_MODE)), NDRAW = $(NDRAW), reverse rows = $(WITH_REVERSE), kappa0 = $(KAPPA0)")
println("  started  = $(now())")
flush(stdout)

fold_summary = NamedTuple[]
for pw in REV_PATHWAYS
    t0 = time()
    push!(fold_summary, run_pathway(pw))
    @printf("[%s] elapsed %.1f min\n", pw, (time() - t0) / 60)
    flush(stdout)
end
println("\nRegression and fit summary")
show(stdout, DataFrame(fold_summary); allrows = true, allcols = true); println()
println("Done at $(now())")
