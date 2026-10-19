# run_revision_rate_invivo.jl
#
# In vivo check of predicted reaction rates for E. coli, added after the
# shortened manuscript (October 2026).
#
# Question.  run_revision_invivo_check.jl compares the in vivo maximal apparent
# turnover numbers of Davidi et al. (2016) with predicted turnover numbers.  A
# rate law makes a stronger prediction: at the metabolite concentrations
# measured in a growth condition, the predicted rate per enzyme should match
# the apparent rate per enzyme observed in that condition, kapp = flux /
# enzyme abundance.  This driver evaluates the convenience rate law of the
# Propagation section at measured E. coli concentrations, with the parameters
# of the E. coli hold-out fits, and asks whether the predictive intervals
# contain kapp.
#
# Inputs (built by scripts/build_invivo_rate_inputs.py, sources in
# data/external/README.md):
#   data/external/ecoli_invivo_kapp.csv            signed kapp per chain (1/s),
#                                                  per reaction and condition
#   data/external/ecoli_invivo_concentrations.csv  concentrations (mM)
#   output/<pathway>/results/revision/ecoli_holdout_draws_<pathway>.jld2
#       the posterior draws of the two E. coli hold-out fits of
#       run_revision_ecoli_holdout.jl (every E. coli record removed): fixed
#       effects (standard parameter balancing) and the pre-specified default,
#       with the default's pooled between-organism variance tau^2
#
# Predictive draws.  For each reaction and condition, NDRAW draws each take one
# posterior column (the same uniform for both fits, as in the hold-out driver)
# and evaluate the signed convenience rate per unit enzyme, with every
# substrate and product (water excluded) at its measured concentration:
#
#   spb        fixed-effects posterior, no organism offset
#   velocity   default posterior plus an E. coli offset ~ N(0, tau^2) on every
#              turnover number, the paper's new-organism rate interval
#   full       as velocity, plus an independent offset ~ N(0, tau^2) on every
#              Michaelis constant, as the default's record model assumes
#
# and, as a reference, the turnover number alone in the direction of the
# observed flux (kcat_spb, kcat_full), which is what the existing check
# compares with kmax.  Gerosa et al. report 2- and 3-phosphoglycerate as one
# pool; each draw splits it with its own phosphoglycerate mutase equilibrium
# constant (the pool is assumed at equilibrium).  The draws are at organism
# level: kapp carries its own errors (pFBA fluxes, proteomics, in vivo
# conditions) that no record error of the fit describes.
#
# Measured concentrations are used as they are.  With PB_RIV_CONC_SD = s > 0,
# every draw multiplies each concentration by its own exp(s z), z ~ N(0, 1), a
# sensitivity run for concentration error (a 2PG/3PG pool gets one factor).
#
# Scores per reaction, condition and variant: whether the central 95% interval
# of the signed draws contains kapp; the PIT of kapp; the fraction of draws
# with the observed flux direction; the log10 error of the median prediction
# where it has the observed direction; the width of the 95% interval of
# log10 |rate| in decades; and the median of ln(Gamma / Keq), the mass-action
# ratio over the equilibrium constant, from the ratio of the reverse to the
# forward term of the rate law (Haldane), which says how close to
# equilibrium each draw puts the reaction.
#
# Outputs, written under output/rate_checks/:
#   rate_invivo_rows.csv     one row per reaction, condition and variant
#   rate_invivo_summary.csv  per variant and set: coverage with an exact
#                            binomial interval, mean PIT, direction agreement,
#                            median absolute log10 error.  Sets: all pairs; the
#                            Gerosa concentrations only; and two thermodynamic
#                            classes by the default's median ln(Gamma / Keq),
#                            near equilibrium (|ln(Gamma / Keq)| < 1, Gamma
#                            within a factor e of Keq) or far from it
#
# Environment controls:
#   PB_RIV_NDRAW   draws per reaction and condition (default 4000)
#   PB_RIV_SEED    base seed (default 20261021)
#   PB_RIV_CONC_SD log-scale standard deviation of a per-draw concentration
#                  error (default 0, the measured values as they are); a run
#                  with a positive value must set PB_RIV_OUT_SUFFIX
#   PB_RIV_OUT_SUFFIX  suffix of the outputs only (default PB_REV_SUFFIX)
#   PB_REV_SUFFIX  suffix of the hold-out draw files and of the outputs (default
#                  empty)
#
# Run (seconds, after run_revision_ecoli_holdout.jl):
#   julia --project=. scripts/run_revision_rate_invivo.jl

import Pkg
Pkg.activate(dirname(@__DIR__))

using Dates, LinearAlgebra, Printf, Random, Statistics
using CSV, DataFrames, Distributions, JLD2

# PATHWAY_SPECS and the reconstruction helpers (its main() is guarded).
include("build_pathway_inputs.jl")

const NDRAW = parse(Int, get(ENV, "PB_RIV_NDRAW", "4000"))
const SEED = parse(Int, get(ENV, "PB_RIV_SEED", "20261021"))
const SUFFIX = get(ENV, "PB_REV_SUFFIX", "")
const CONC_SD = parse(Float64, get(ENV, "PB_RIV_CONC_SD", "0"))
const OUT_SUFFIX = get(ENV, "PB_RIV_OUT_SUFFIX", SUFFIX)
CONC_SD == 0 || OUT_SUFFIX != SUFFIX ||
    error("PB_RIV_CONC_SD is not 0: set PB_RIV_OUT_SUFFIX, so that the main outputs are kept")
const OUTDIR = joinpath(dirname(@__DIR__), "output", "rate_checks")
const PATHWAYS = [:glycolysis, :tca, :pentose_phosphate]

# build_reaction_spec of run_revision_downstream.jl (that driver has no main guard).
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
Convenience rate per unit enzyme (convenience_rate of run_revision_downstream.jl),
returning the rate and the forward and reverse terms of its numerator.
"""
function convenience_terms(kcat_f, kcat_r, a_sub, s_sub, a_prod, s_prod)
    num_f = kcat_f
    for i in eachindex(a_sub)
        num_f *= a_sub[i]^s_sub[i]
    end
    num_r = kcat_r
    for i in eachindex(a_prod)
        num_r *= a_prod[i]^s_prod[i]
    end
    den_s = 1.0
    for i in eachindex(a_sub)
        den_s *= sum(a_sub[i]^m for m in 0:s_sub[i])
    end
    den_p = 1.0
    for i in eachindex(a_prod)
        den_p *= sum(a_prod[i]^m for m in 0:s_prod[i])
    end
    return (num_f - num_r) / (den_s + den_p - 1.0), num_f, num_r
end

pit_midrank(x, z) = (count(<(z), x) + 0.5 * count(==(z), x)) / length(x)

function clopper_pearson(k::Int, n::Int; level::Float64 = 0.95)
    n == 0 && return (NaN, NaN)
    a = (1 - level) / 2
    lo = k == 0 ? 0.0 : quantile(Beta(k, n - k + 1), a)
    hi = k == n ? 1.0 : quantile(Beta(k + 1, n - k), 1 - a)
    return (lo, hi)
end

const MDL = load_yeast_reconstruction(joinpath(dirname(@__DIR__), "data", "yeast-GEM.xml"))
const DICT_RXNS = reaction_data_dict(MDL)

println("Revision in vivo rate check")
println("  draws per reaction and condition = $(NDRAW), hold-out suffix = '$(SUFFIX)', concentration error sd = $(CONC_SD)")
println("  started = $(now())")
flush(stdout)

kt = CSV.read(joinpath(dirname(@__DIR__), "data", "external", "ecoli_invivo_kapp.csv"), DataFrame;
              stringtype = String)
kt.included = [v isa Bool ? v : lowercase(string(v)) == "true" for v in kt.included]
ct = CSV.read(joinpath(dirname(@__DIR__), "data", "external", "ecoli_invivo_concentrations.csv"), DataFrame;
              stringtype = String)
conc = Dict((r.condition, r.bigg_id) => Float64(r.conc_mM) for r in eachrow(ct))

function run_pathway(ip::Int, pw::Symbol, kt::DataFrame, conc::Dict)
    rows = NamedTuple[]
    sub = kt[(kt.pathway .== string(pw)) .& kt.included, :]
    nrow(sub) == 0 && return rows
    spec = PATHWAY_SPECS[string(pw)]
    rxns = generate_reactions(spec.reaction_ids, DICT_RXNS, protons = false,
                              proton_name = spec.proton_name)
    Q = Matrix{Float64}(observation_matrix(rxns, h2o_strings = spec.h2o))
    species_names, _ = rn_entities(rxns)
    (mu0_map, KM_map, kcatf, kcatr) = observation_matrix_index_maps(rxns)
    N = netstoichmat(reaction_network(rxns))
    specs = build_reaction_spec(rxns, spec.h2o, species_names, KM_map, kcatf, kcatr, N)
    bigg = Dict(s => get(spec.bigg_ids, s, "") for s in species_names)
    sid_of(b) = (i = findfirst(s -> bigg[s] == b, species_names); i === nothing ? "" : species_names[i])

    D = load(joinpath(dirname(@__DIR__), "output", string(pw), "results", "revision",
                      "ecoli_holdout_draws_$(pw)$(SUFFIX).jld2"))
    qfe = Matrix{Float64}(D["fe_qsamps"]); qpd = Matrix{Float64}(D["pd_qsamps"])
    τ2 = Float64(D["tau2_pooled"])
    size(Q, 2) == size(qfe, 1) == size(qpd, 1) || error("$pw: basis dimensions differ")
    Xfe = Q * qfe; Xpd = Q * qpd
    Sfe = size(Xfe, 2); Spd = size(Xpd, 2)
    kcat_rows = sort(vcat([kcatf[r] for r in spec.reaction_ids], [kcatr[r] for r in spec.reaction_ids]))
    km_rows = sort(unique(vcat([vcat(sp.sub_km_row, sp.pro_km_row) for sp in specs]...)))
    # Rows of the phosphoglycerate mutase equilibrium (glycolysis only).
    s3 = sid_of("3pg"); s2 = sid_of("2pg")
    @printf("\n=== %s: %d reaction-condition pairs; fixed-effects draws %d, default draws %d, tau2 = %.3f\n",
            pw, nrow(sub), Sfe, Spd, τ2)

    for (k, r) in enumerate(eachrow(sub))
        j = findfirst(sp -> sp.name == r.reaction_id, specs)
        j === nothing && error("$pw: reaction $(r.reaction_id) is not in the model")
        sp = specs[j]
        names_needed = vcat(sp.sub_names, sp.pro_names)
        # Concentration of each species in M; 2PG and 3PG come from one pool.
        # One concentration-error factor per metabolite (or pool) and draw.
        ckeys = unique([bigg[s] in ("2pg", "3pg") && !haskey(conc, (r.condition, bigg[s])) ?
                        "2pg+3pg" : bigg[s] for s in names_needed])
        function conc_of(sname, x, cf)
            b = bigg[sname]
            if b in ("2pg", "3pg") && !haskey(conc, (r.condition, b))
                pool = conc[(r.condition, "2pg+3pg")] / 1000 * cf["2pg+3pg"]
                lnK = -(x[mu0_map[s2]] - x[mu0_map[s3]])      # [2PG]/[3PG] at equilibrium
                f2 = 1 / (1 + exp(-lnK))
                return b == "2pg" ? pool * f2 : pool * (1 - f2)
            end
            haskey(conc, (r.condition, b)) || error("no concentration for $b ($sname) in $(r.condition)")
            return conc[(r.condition, b)] / 1000 * cf[b]
        end
        rng = MersenneTwister(SEED + 1000 * ip + k)
        U = rand(rng, NDRAW); Zk = randn(rng, NDRAW); Zm = randn(rng, NDRAW)
        # The concentration errors use their own stream, so the main draws are unchanged.
        rngc = MersenneTwister(SEED + 1000 * ip + k + 500)
        Zc = randn(rngc, NDRAW, length(ckeys))
        col_fe = 1 .+ floor.(Int, U .* Sfe); col_pd = 1 .+ floor.(Int, U .* Spd)
        dirn = r.kapp_signed_1s > 0 ? 1 : -1
        out = Dict{String,Vector{Float64}}(v => Vector{Float64}(undef, NDRAW)
                                           for v in ("spb", "velocity", "full", "kcat_spb", "kcat_full"))
        lnratio = Dict(v => Vector{Float64}(undef, NDRAW) for v in ("spb", "velocity", "full"))
        x = Vector{Float64}(undef, size(Q, 1))
        for d in 1:NDRAW
            cf = Dict(ckeys[i] => exp(CONC_SD * Zc[d, i]) for i in eachindex(ckeys))
            for (v, X, c, ek, em) in (("spb", Xfe, col_fe[d], 0.0, 0.0),
                                      ("velocity", Xpd, col_pd[d], sqrt(τ2) * Zk[d], 0.0),
                                      ("full", Xpd, col_pd[d], sqrt(τ2) * Zk[d], sqrt(τ2) * Zm[d]))
                x .= view(X, :, c)
                for i in kcat_rows; x[i] += ek; end
                for i in km_rows; x[i] += em; end
                a_sub = [conc_of(s, x, cf) / exp(x[row]) for (s, row) in zip(sp.sub_names, sp.sub_km_row)]
                a_pro = [conc_of(s, x, cf) / exp(x[row]) for (s, row) in zip(sp.pro_names, sp.pro_km_row)]
                vrate, nf, nr = convenience_terms(exp(x[sp.kcatf_row]), exp(x[sp.kcatr_row]),
                                                  a_sub, sp.sub_stoich, a_pro, sp.pro_stoich)
                out[v][d] = vrate
                lnratio[v][d] = log(nr) - log(nf)
                if v != "velocity"
                    kc = dirn > 0 ? exp(x[sp.kcatf_row]) : -exp(x[sp.kcatr_row])
                    out[v == "spb" ? "kcat_spb" : "kcat_full"][d] = kc
                end
            end
        end
        z = Float64(r.kapp_signed_1s)
        for v in ("spb", "velocity", "full", "kcat_spb", "kcat_full")
            xs = out[v]
            lo, med, hi = quantile(xs, [0.025, 0.5, 0.975])
            psign = mean(t -> sign(t) == sign(z), xs)
            lw = log10.(abs.(xs) .+ 1e-300)
            push!(rows, (pathway = string(pw), reaction_id = r.reaction_id,
                reaction_name = r.reaction_name, iJO1366_reaction = r.iJO1366_reaction,
                condition = r.condition, concentration_source = r.concentration_source,
                kapp_signed_1s = z, variant = v,
                lo95 = lo, median = med, hi95 = hi, covered95 = lo <= z <= hi,
                pit = pit_midrank(xs, z), frac_observed_direction = psign,
                log10_error = sign(med) == sign(z) ? log10(abs(z)) - log10(abs(med)) : NaN,
                width95_log10abs = quantile(lw, 0.975) - quantile(lw, 0.025),
                median_ln_gamma_over_keq = haskey(lnratio, v) ? median(lnratio[v]) : NaN,
                n_species = length(names_needed)))
        end
    end
    return rows
end

rows = NamedTuple[]
for (ip, pw) in enumerate(PATHWAYS)
    append!(rows, run_pathway(ip, pw, kt, conc))
end
rdf = DataFrame(rows)
# Thermodynamic class of each pair, from the default's (full variant) median ln(Gamma / Keq).
lgk = Dict((r.reaction_id, r.condition) => r.median_ln_gamma_over_keq
           for r in eachrow(rdf) if r.variant == "full")
rdf.thermo_class = [abs(lgk[(r.reaction_id, r.condition)]) < 1 ? "near_equilibrium" :
                    "far_from_equilibrium" for r in eachrow(rdf)]

srows = NamedTuple[]
for v in ("spb", "velocity", "full", "kcat_spb", "kcat_full"),
        set in ("all", "gerosa2015", "near_equilibrium", "far_from_equilibrium")
    m = rdf.variant .== v
    set == "gerosa2015" && (m .&= rdf.concentration_source .== set)
    set in ("near_equilibrium", "far_from_equilibrium") && (m .&= rdf.thermo_class .== set)
    d = rdf[m, :]
    n = nrow(d); k = count(d.covered95); ci = clopper_pearson(k, n)
    e = filter(isfinite, abs.(d.log10_error))
    push!(srows, (variant = v, set = set, n = n, n_reactions = length(unique(d.reaction_id)),
        n_covered95 = k, cov95 = k / n, cov95_ci_lo = ci[1], cov95_ci_hi = ci[2],
        n_below = count(d.kapp_signed_1s .< d.lo95), n_above = count(d.kapp_signed_1s .> d.hi95),
        mean_pit = mean(d.pit), mean_frac_observed_direction = mean(d.frac_observed_direction),
        n_median_wrong_direction = count(isnan, d.log10_error),
        median_abs_log10_error = isempty(e) ? NaN : median(e),
        median_width95_log10abs = median(d.width95_log10abs)))
end
sdf = DataFrame(srows)

mkpath(OUTDIR)
CSV.write(joinpath(OUTDIR, "rate_invivo_rows$(OUT_SUFFIX).csv"), rdf)
CSV.write(joinpath(OUTDIR, "rate_invivo_summary$(OUT_SUFFIX).csv"), sdf)
println("\nPer reaction and condition")
show(stdout, select(rdf, :reaction_id, :iJO1366_reaction, :condition, :variant, :kapp_signed_1s,
                    :lo95, :median, :hi95, :covered95, :pit, :frac_observed_direction,
                    :median_ln_gamma_over_keq, :thermo_class); allrows = true, allcols = true)
println("\n\nSummary")
show(stdout, sdf; allrows = true, allcols = true)
println("\nDone at $(now())")
