# run_revision_dynamic.jl
#
# Dynamic propagation and rate-law sensitivity, added for the revision.
#
# Limitations addressed:
#   1. "The downstream propagation evaluates rates at a fixed reference
#      metabolite state rather than integrating Eq (ode)": this driver
#      integrates the kinetic ODE system to steady state for every posterior
#      draw, with boundary metabolites and cofactors clamped at the reference
#      concentration and unit enzyme activities, exactly the normalisation of
#      the static analysis, and compares the posterior interval widths of the
#      steady-state pathway flux with the static ones.
#   2. "Uncertainty in the choice of rate law is not propagated": the static
#      evaluation is repeated under a mass-action rate law sharing the same
#      catalytic constants and Michaelis scalings (identical numerator, no
#      saturation denominator), so the comparison isolates the rate-law choice
#      while preserving Haldane consistency.
#
# The question in both cases is not the location of the downstream
# distributions but whether the paper's headline comparison, the width of the
# consensus interval against the width of the new-organism interval, survives
# the modelling choice.  Draws whose ODE integration fails to reach steady
# state are counted and excluded.
#
# Model cells, as in run_revision_downstream.jl (same fits, same seed): the
# fixed-effects fit and the organism-by-type random-intercept fit on the
# half-decade record scale (c = 1), followed by gaussian_cest_RE_orgtype, the
# pre-specified default of the cross-validation.  That cell has the same
# random-intercept structure but a record-scale multiplier c estimated by the
# within-cell iteration of run_revision_hetero.jl (fit at c = 1, then
# PB_REV_SIGMACAL_ITERS rounds of re-estimating c and refitting, clamped to
# [C_MIN, C_MAX], continuing the same random-number stream; kappa0 = 50), and
# its new-organism offsets are drawn from its own pooled tau^2.  The new cell is
# appended last, so the rows of the first two cells are unchanged; its c and
# tau^2 are printed to the log (the same values are recorded in
# downstream_summary_<pathway>.csv).
#
# Outputs, written under output/<pathway>/results/revision/:
#   dynamic_summary_<pathway>.csv   one row per model cell and target
#
# Environment controls:
#   PB_REV_PATHWAYS        comma-separated subset of {glycolysis,tca} (default both)
#   PB_REV_DYN_DRAWS       posterior draws per cell (default 200)
#   PB_REV_SIGMACAL_ITERS  within-cell iterations for the estimated record scale
#                          of gaussian_cest_RE_orgtype (default 4)
#   PB_EB_KAPPA0           prior-mean concentration of every fit (default 50, as
#                          published; 1e-9 holds the prior mean, as in the
#                          propagation of the shorter manuscript)
#   PB_REV_SUFFIX          appended to the output file name (default empty);
#                          required when PB_EB_KAPPA0 is not 50

import Pkg
Pkg.activate(dirname(@__DIR__))

using Dates, LinearAlgebra, Printf, Random, Statistics
using ArviZ, BlockArrays, CairoMakie, Colors, CSV, DataFrames, Distributions
using JLD2, JSON, KernelDensity, LaTeXStrings, MCMCChains, ProgressMeter, SpecialFunctions
using Catalyst
using DifferentialEquations

include(joinpath(dirname(@__DIR__), "src", "metabolic_reconstruction.jl"))
include(joinpath(dirname(@__DIR__), "src", "parameter_balancing.jl"))
include(joinpath(dirname(@__DIR__), "src", "makie_functions.jl"))
include(joinpath(dirname(@__DIR__), "src", "parameter_balancing_core.jl"))
include(joinpath(dirname(@__DIR__), "src", "prediction_utils.jl"))
include(joinpath(dirname(@__DIR__), "src", "experiment_runners.jl"))

const REV_SEED = 20260805
const REV_PATHWAYS = Symbol.(strip.(split(get(ENV, "PB_REV_PATHWAYS", "glycolysis,tca"), ",")))
const N_DRAWS = parse(Int, get(ENV, "PB_REV_DYN_DRAWS", "200"))
const REF_CONC = 1e-3          # 1 mM, as in the static downstream analysis
# Within-cell record-scale iteration, as in run_revision_hetero.jl.
const SIGMACAL_ITERS = parse(Int, get(ENV, "PB_REV_SIGMACAL_ITERS", "4"))
const KAPPA0 = parse(Float64, get(ENV, "PB_EB_KAPPA0", "50.0"))
const OUT_SUFFIX = get(ENV, "PB_REV_SUFFIX", "")
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

# Boundary species clamped at the reference concentration: pathway substrates
# and products, cofactor pairs, and small inorganic species.  Internal
# metabolites, the pathway intermediates, evolve under the ODE.
const CLAMPED = Dict(
    :glycolysis => ["s_0434","s_0563","s_0394","s_1198","s_1322","s_1203",
                    "s_0803","s_1399"],
    :tca => ["s_0376","s_0807","s_0532","s_1200","s_1205","s_0460",
             "s_1097","s_1411","s_0627","s_0397","s_1326","s_0437",
             "s_1537","s_1535"],
)
# Terminal reaction whose flux is summarised (last pathway step).
const TERMINAL = Dict(:glycolysis => "r_0962", :tca => "r_0713")

function ridge_qhat(Arows, y, sigma2, q0, C0)
    C0inv = inv(Symmetric(Matrix{Float64}(C0)))
    P = C0inv + Arows' * (Arows ./ sigma2)
    return Symmetric(P) \ (Arows' * (y ./ sigma2) + C0inv * q0)
end

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

# Mass-action counterpart: identical numerator (same Haldane combination, so
# the equilibrium state is unchanged), no saturation denominator.
function massaction_rate(kcat_f::Float64, kcat_r::Float64,
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
    return num_f - num_r
end

function build_reaction_spec(rxns, h2o_names::Vector{String}, species_names::Vector{String},
                             KM_map, kcatf, kcatr, N::AbstractMatrix)
    specs = Vector{NamedTuple}(undef, length(rxns))
    for (j, r) in enumerate(rxns)
        rname = r[1]
        sub_i = Int[]; sub_s = Int[]; sub_km = Int[]
        pro_i = Int[]; pro_s = Int[]; pro_km = Int[]
        for i in eachindex(species_names)
            nu = N[i, j]
            nu == 0 && continue
            sname = species_names[i]
            sname in h2o_names && continue
            haskey(KM_map, (rname, sname)) || continue
            if nu < 0
                push!(sub_i, i); push!(sub_s, Int(abs(nu))); push!(sub_km, KM_map[(rname, sname)])
            else
                push!(pro_i, i); push!(pro_s, Int(nu)); push!(pro_km, KM_map[(rname, sname)])
            end
        end
        specs[j] = (name = rname,
                    sub_idx = sub_i, sub_stoich = sub_s, sub_km_row = sub_km,
                    pro_idx = pro_i, pro_stoich = pro_s, pro_km_row = pro_km,
                    kcatf_row = kcatf[rname], kcatr_row = kcatr[rname])
    end
    return specs
end

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

_w95(v) = (w = filter(isfinite, v); length(w) < 10 ? NaN :
           quantile(w, 0.975) - quantile(w, 0.025))
_med(v) = (w = filter(isfinite, v); isempty(w) ? NaN : median(w))

"""
    steady_flux(x, specs, species_names, clamped_idx, int_idx, N_int, jterm)
        -> (log10 |v_terminal|, converged::Bool)

Integrate the pathway ODE to steady state for one posterior draw `x` of the
derived quantities and return the terminal-reaction flux there.
"""
function steady_flux(x::AbstractVector{Float64}, specs, species_names,
                     clamped::Vector{Bool}, int_idx::Vector{Int},
                     N_int::Matrix{Float64}, jterm::Int)
    nsp = length(species_names)
    kf = [exp(x[sp.kcatf_row]) for sp in specs]
    kr = [exp(x[sp.kcatr_row]) for sp in specs]
    KMs = [[exp(x[row]) for row in sp.sub_km_row] for sp in specs]
    KMp = [[exp(x[row]) for row in sp.pro_km_row] for sp in specs]

    c = fill(REF_CONC, nsp)
    v = zeros(length(specs))
    function rates!(cvec)
        @inbounds for (j, sp) in enumerate(specs)
            a_sub = [cvec[sp.sub_idx[k]] / KMs[j][k] for k in eachindex(sp.sub_idx)]
            a_pro = [cvec[sp.pro_idx[k]] / KMp[j][k] for k in eachindex(sp.pro_idx)]
            v[j] = convenience_rate(kf[j], kr[j], a_sub, sp.sub_stoich, a_pro, sp.pro_stoich)
        end
        return v
    end

    function f!(du, u, p, t)
        @inbounds for (k, i) in enumerate(int_idx)
            c[i] = max(u[k], 1e-12)
        end
        rates!(c)
        mul!(du, N_int, v)
        return nothing
    end

    u0 = fill(REF_CONC, length(int_idx))
    prob = ODEProblem(f!, u0, (0.0, 1.0e7))
    sol = solve(prob, Rodas5(autodiff = false);
                callback = TerminateSteadyState(1e-10, 1e-8),
                abstol = 1e-10, reltol = 1e-8, maxiters = 2_000_000,
                isoutofdomain = (u, p, t) -> any(<(0.0), u),
                save_everystep = false, verbose = false)
    ok = sol.retcode == ReturnCode.Terminated || sol.retcode == ReturnCode.Success
    ok || return (NaN, false)
    @inbounds for (k, i) in enumerate(int_idx)
        c[i] = max(sol.u[end][k], 1e-12)
    end
    rates!(c)
    return (log10(abs(v[jterm]) + 1e-300), true)
end

function run_pathway(pathway::Symbol, setup)
    rxns = setup.rxns
    Qfull = Matrix{Float64}(setup.Q)
    h2o = setup.h2o
    root = joinpath(dirname(@__DIR__), "output", string(pathway))
    results_dir = joinpath(root, "results", "revision"); mkpath(results_dir)
    brenda = load(joinpath(root, "data", "brenda_inputs_$(pathway).jld2"))
    post = load(joinpath(root, "data", "posterior_$(pathway).jld2"))
    y = Vector{Float64}(brenda["y"]); sigma2 = Vector{Float64}(brenda["σ2"])
    Arows = Matrix{Float64}(brenda["Arows"]); org = Vector{String}(brenda["grp"])
    kinds = Vector{String}(getproperty(brenda["meta"], :kind))
    org_type = [string(org[i], "|", kinds[i]) for i in eachindex(org)]
    q0 = vec(Array(post["q_prior"])); C0 = Matrix(Array(post["C_prior"]))
    eb = EBSpec(q0 = q0, C0 = C0, κ0 = KAPPA0); d = size(Arows, 2)
    A0, b0 = _prepare_information_pair(Arows, d, Matrix{Float64}(brenda["A0"]),
                                       Vector{Float64}(brenda["b0"]);
                                       deduplicate_thermo_block = true)

    species_names, reaction_names = rn_entities(rxns)
    (mu0_map, KM_map, kcatf, kcatr) = observation_matrix_index_maps(rxns)
    (_, mur_param_map, _) = parameter_index_maps(rxns)
    mur_idx = sort([mur_param_map[r] for r in reaction_names])
    rn = reaction_network(rxns)
    N = Matrix{Float64}(netstoichmat(rn))
    specs = build_reaction_spec(rxns, h2o, species_names, KM_map, kcatf, kcatr, N)

    clamped = [s in CLAMPED[pathway] for s in species_names]
    int_idx = findall(.!clamped)
    N_int = N[int_idx, :]
    jterm = findfirst(sp -> sp.name == TERMINAL[pathway], specs)
    println("\n=== $(pathway): $(length(specs)) reactions, " *
            "$(length(int_idx)) internal metabolites, terminal = $(TERMINAL[pathway])")

    is_tca = pathway == :tca
    spec_for(model, hier) = FoldFitSpec(
        model = model, hierarchical = hier,
        n_outer = is_tca ? 4 : 3, n_mcmc = 600, burnin = 150, thin = 1,
        n_keep_min = 400, n_pred_draws = 400, b_draws_ll = 50,
        posterior_refresh = 1, adaptive_stop = false, rtol_outer = 1e-3,
        min_outer = 2, tau_update = :ig, tau_pc_rate = 0.0,
        ν_init = 50.0, α_init = 0.0,
        update_nu = false, update_alpha = false, refine_shape = false,
        shape_opt_tol = 1e-3, skew_location = :mean,
        nu_penalty = 0.0, nu_target = 50.0, alpha_penalty = 0.0)

    # Fourth field: record scale (:convention keeps c = 1; :global_cell
    # estimates c by the within-cell iteration).  New cells are appended last.
    cells = [("gaussian_Fix",             false, org,      :convention),
             ("gaussian_RE_orgtype",      true,  org_type, :convention),
             ("gaussian_cest_RE_orgtype", true,  org_type, :global_cell)]

    out = DataFrame(pathway = String[], cell = String[], target = String[],
                    n_draws = Int[], n_converged = Int[],
                    static_conv_median = Float64[], static_conv_w95 = Float64[],
                    static_mass_median = Float64[], static_mass_w95 = Float64[],
                    dynamic_median = Float64[], dynamic_w95 = Float64[])

    for (label, hier, grp, scale) in cells
        print("  -> $label ... "); flush(stdout); t0 = time()
        spec = spec_for(:gaussian, hier)
        hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
        # A fresh stream per cell; the within-cell iteration below continues it.
        rng = MersenneTwister(REV_SEED)
        fit = fit_fold_mcem(rng, A0, b0, Arows, y, sigma2,
                            grp, eb, hs, spec; method = :collapsed)
        c_hat = 1.0
        if scale == :global_cell
            # Pre-specified default: re-estimate c from the posterior-mean
            # consensus residuals within the grouping cells, then refit.
            for _ in 1:SIGMACAL_ITERS
                m = vec(mean(Arows * fit.qsamps, dims = 2))
                c_hat = c_update_within(y, m, sigma2, grp, eachindex(y))
                hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
                fit = fit_fold_mcem(rng, A0, b0, Arows, y, c_hat .* sigma2,
                                    grp, eb, hs, spec; method = :collapsed)
            end
        elseif scale != :convention
            error("unknown record scale $(scale)")
        end
        @printf("\n       record scale c = %.15g, tau2 = %.15g", c_hat, fit.hs.τ²)
        targets = [("population", fit.qsamps)]
        if hier && fit.hs.τ² > 0
            push!(targets, ("new_organism",
                            add_organism_effect(fit.qsamps, mur_idx, fit.hs.τ²,
                                                MersenneTwister(REV_SEED + 991))))
        end
        for (target, qs) in targets
            S = size(qs, 2)
            take = S <= N_DRAWS ? collect(1:S) :
                   round.(Int, range(1, S; length = N_DRAWS))
            X = Qfull * qs[:, take]
            n = length(take)
            st_conv = fill(NaN, n); st_mass = fill(NaN, n); dyn = fill(NaN, n)
            nconv = 0
            for (t, s) in enumerate(1:n)
                x = view(X, :, s)
                sp = specs[jterm]
                kfv = exp(x[sp.kcatf_row]); krv = exp(x[sp.kcatr_row])
                a_sub = [REF_CONC / exp(x[row]) for row in sp.sub_km_row]
                a_pro = [REF_CONC / exp(x[row]) for row in sp.pro_km_row]
                st_conv[t] = log10(abs(convenience_rate(kfv, krv, a_sub, sp.sub_stoich,
                                                        a_pro, sp.pro_stoich)) + 1e-300)
                st_mass[t] = log10(abs(massaction_rate(kfv, krv, a_sub, sp.sub_stoich,
                                                       a_pro, sp.pro_stoich)) + 1e-300)
                dyn[t], ok = steady_flux(x, specs, species_names, clamped, int_idx,
                                         N_int, jterm)
                nconv += ok ? 1 : 0
            end
            push!(out, (string(pathway), label, target, n, nconv,
                        _med(st_conv), _w95(st_conv),
                        _med(st_mass), _w95(st_mass),
                        _med(dyn), _w95(dyn)))
            @printf("\n       %-13s conv-static w95=%.2f  mass-static w95=%.2f  dynamic w95=%.2f  (%d/%d converged)",
                    target, _w95(st_conv), _w95(st_mass), _w95(dyn), nconv, n)
        end
        @printf("   [%.1f min]\n", (time() - t0) / 60)
        flush(stdout)
    end

    CSV.write(joinpath(results_dir, "dynamic_summary_$(pathway)$(OUT_SUFFIX).csv"), out)
    println("\n  $(pathway) dynamic-propagation summary")
    show(stdout, out; allrows = true, allcols = true); println()
    return nothing
end

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

println("Revision dynamic propagation and rate-law sensitivity")
println("  draws = $N_DRAWS, started = $(now())")
for pw in REV_PATHWAYS
    setup = pw == :tca ? tca_setup() : glycolysis_setup()
    run_pathway(pw, setup)
end
println("Done at $(now())")
