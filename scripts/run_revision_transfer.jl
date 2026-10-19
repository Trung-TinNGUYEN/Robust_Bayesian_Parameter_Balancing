# run_revision_transfer.jl
#
# Reducing the unseen-organism penalty, added for the revision.
#
# The recalibrated organism-by-type hierarchy is calibrated per held-out
# organism, and the residual record-averaged gap is an estimand distinction
# (run_revision_conformal.jl).  What remains is the size of the
# unseen-organism predictive variance itself: an organism the model has never
# seen is assigned the full between-organism variance tau^2.  That treats the
# organism offset as pure noise.  It is not: the offset is a reproducible
# organism-and-laboratory fingerprint, and two sources of side information
# predict it before a single record from the target pathway is observed,
#
#   genus     the fitted offsets of congeneric organisms (same genus, same
#             quantity type) already in the training set;
#   xpath     the SAME organism's fitted offset in ANOTHER pathway: the four
#             curated pathways share 19 to 52 organisms per pair, and 64 to
#             79 per cent of records belong to organisms with records in at
#             least one other pathway;
#   transfer  the precision-weighted combination of the two.
#
# Every ingredient is estimated on training data only.  Within each
# leave-one-source-out fold of the target pathway: the genus variance share
# (from congener-pair products of fitted training offsets), the cross-pathway
# coupling rho and residual omega^2 (attenuation-corrected regression of the
# training organisms' target offsets on their donor-pathway offsets), and the
# error scale c (within-cell iteration, as in run_revision_sigmacal.jl).
# Donor-pathway offsets come from a full recalibrated fit of the donor
# pathway; the held-out organism's donor records are legitimately available
# information in the practical scenario (known organism, new pathway) and its
# target-pathway records are never touched.
#
# Outputs, written under output/<pathway>/results/revision/:
#   transfer_summary_<pathway>.csv    one row per predictive construction
#   transfer_coverage_<pathway>.csv   coverage at every nominal level
#   transfer_cells_<pathway>.csv      per held-out cell: predicted offset,
#                                     variance, signal flags, realised proxy
#
# Environment controls:
#   PB_REV_PATHWAYS  subset of {glycolysis,tca,pentose_phosphate,shikimate}
#   PB_REV_SIGMACAL_ITERS  EB iterations for the error scale (default 4)

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
include(joinpath(dirname(@__DIR__), "src", "experiment_runners.jl"))

const REV_LEVELS = vcat(0.01, collect(0.05:0.05:0.95), 0.99)
const REV_SEED = 20260812
const REV_PATHWAYS = Symbol.(strip.(split(get(ENV, "PB_REV_PATHWAYS",
    "glycolysis,tca,pentose_phosphate,shikimate"), ",")))
const ALL_PATHWAYS = [:glycolysis, :tca, :pentose_phosphate, :shikimate]
const SIGMACAL_ITERS = parse(Int, get(ENV, "PB_REV_SIGMACAL_ITERS", "4"))
const C_MIN, C_MAX = 0.25, 25.0
const Z95 = quantile(Normal(), 0.975)

genus_of(o::AbstractString) = String(first(split(o)))

function posterior_q_moments(fit::FoldFit, Arows_tr::AbstractMatrix)
    M = Arows_tr * fit.qsamps
    return vec(mean(M, dims = 2)), vec(var(M, dims = 2))
end

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

function make_spec(pathway::Symbol)
    slow = pathway in (:tca,)
    FoldFitSpec(model = :gaussian, hierarchical = true,
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

function load_pathway(pathway::Symbol)
    root = joinpath(dirname(@__DIR__), "output", string(pathway))
    brenda = load(joinpath(root, "data", "brenda_inputs_$(pathway).jld2"))
    post = load(joinpath(root, "data", "posterior_$(pathway).jld2"))
    y = Vector{Float64}(brenda["y"]); sigma2 = Vector{Float64}(brenda["σ2"])
    Arows = Matrix{Float64}(brenda["Arows"]); org = Vector{String}(brenda["grp"])
    kinds = Vector{String}(getproperty(brenda["meta"], :kind))
    q0 = vec(Array(post["q_prior"])); C0 = Matrix(Array(post["C_prior"]))
    eb = EBSpec(q0 = q0, C0 = C0); d = size(Arows, 2)
    A0, b0 = _prepare_information_pair(Arows, d, Matrix{Float64}(brenda["A0"]),
                                       Vector{Float64}(brenda["b0"]);
                                       deduplicate_thermo_block = true)
    org_type = [string(org[i], "|", kinds[i]) for i in eachindex(org)]
    (; y, sigma2, Arows, org, kinds, org_type, eb, A0, b0,
       results_dir = joinpath(root, "results", "revision"))
end

function sigmacal_fit(rng, dat, idx, spec)
    Ar = dat.Arows[idx, :]
    cg = 1.0
    hs = init_hyper(dat.eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
    fit = fit_fold_mcem(rng, dat.A0, dat.b0, Ar, dat.y[idx], cg .* dat.sigma2[idx],
                        dat.org_type[idx], dat.eb, hs, spec; method = :collapsed)
    for _ in 1:SIGMACAL_ITERS
        m, _ = posterior_q_moments(fit, Ar)
        cg = c_update_within(dat.y[idx], m, dat.sigma2[idx], dat.org_type[idx],
                             eachindex(idx))
        hs = init_hyper(dat.eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
        fit = fit_fold_mcem(rng, dat.A0, dat.b0, Ar, dat.y[idx], cg .* dat.sigma2[idx],
                            dat.org_type[idx], dat.eb, hs, spec; method = :collapsed)
    end
    fit, cg
end

function cell_offsets(fit::FoldFit, cells::Vector{String})
    counts = Dict{String,Int}()
    for c in cells
        counts[c] = get(counts, c, 0) + 1
    end
    out = Dict{String,Tuple{Float64,Float64,Int}}()
    for (i, c) in enumerate(fit.grp_levels)
        out[c] = (mean(view(fit.bsamps, i, :)), var(view(fit.bsamps, i, :)), counts[c])
    end
    out
end

"""
    genus_var(offsets, τ2) -> σ²_g

Between-genus variance share of the organism-by-type offsets, from products of
congener pairs (same genus, same quantity type): under mean-zero offsets,
E[b̂_a b̂_b] = σ²_g for distinct congeners, and the estimation noise of the
fitted offsets cancels in the cross products.
"""
function genus_var(offsets::Dict{String,Tuple{Float64,Float64,Int}}, τ2::Float64)
    groups = Dict{Tuple{String,String},Vector{Float64}}()
    for (ck, (b, _, _)) in offsets
        o, k = split(ck, "|")
        push!(get!(groups, (genus_of(o), String(k)), Float64[]), b)
    end
    num = 0.0; npairs = 0
    for (_, bs) in groups
        L = length(bs)
        L >= 2 || continue
        for a in 1:L, b in (a + 1):L
            num += bs[a] * bs[b]; npairs += 1
        end
    end
    npairs >= 8 || return 0.0
    clamp(num / npairs, 0.0, 0.9 * τ2)
end

"""
    xpath_coupling(offsets, τ2, donors) -> (ρ, ω², n_pairs)

Attenuation-corrected regression through the origin of the training organisms'
target-pathway offsets on their donor-pathway offsets, pooled over donors.
"""
function xpath_coupling(offsets::Dict{String,Tuple{Float64,Float64,Int}}, τ2::Float64,
                        donors::Vector{Dict{String,Tuple{Float64,Float64,Int}}})
    xs = Float64[]; ys = Float64[]; vxs = Float64[]; vys = Float64[]
    for (ck, (bT, vT, _)) in offsets, D in donors
        haskey(D, ck) || continue
        bD, vD, _ = D[ck]
        push!(xs, bD); push!(ys, bT); push!(vxs, vD); push!(vys, vT)
    end
    npair = length(xs)
    npair >= 10 || return (0.0, τ2, npair)
    den = sum(abs2, xs) - sum(vxs)
    den > 1e-8 || return (0.0, τ2, npair)
    ρ = clamp(sum(xs .* ys) / den, 0.0, 1.5)
    resid = mean((ys[i] - ρ * xs[i])^2 - vys[i] - ρ^2 * vxs[i] for i in eachindex(xs))
    ω2 = clamp(resid, 0.05 * τ2, τ2)
    (ρ, ω2, npair)
end

function predict_offset(cellkey::String, τ2::Float64, σ2g::Float64,
                        offsets::Dict{String,Tuple{Float64,Float64,Int}},
                        ρ::Float64, ω2::Float64,
                        donors::Vector{Dict{String,Tuple{Float64,Float64,Int}}})
    o, k = split(cellkey, "|")
    g = genus_of(o)
    σ2w = max(τ2 - σ2g, 1e-8)
    mg, Vg, hasg = 0.0, τ2, false
    if σ2g > 1e-8
        P = 1.0 / σ2g; num = 0.0
        for (ck, (b, v, _)) in offsets
            oc, kc = split(ck, "|")
            (kc == k && oc != o && genus_of(oc) == g) || continue
            w = 1.0 / (σ2w + v)
            P += w; num += b * w
        end
        if P > 1.0 / σ2g
            mg = num / P; Vg = 1.0 / P + σ2w; hasg = true
        end
    end
    mx, Vx, hasx = 0.0, τ2, false
    if ρ > 1e-8
        for D in donors
            haskey(D, cellkey) || continue
            bD, vD, _ = D[cellkey]
            cand_m = ρ * bD; cand_V = ω2 + ρ^2 * vD
            if !hasx || cand_V < Vx
                mx, Vx, hasx = cand_m, cand_V, true
            end
        end
    end
    if hasg && hasx
        Pc = 1.0 / Vg + 1.0 / Vx
        mc = (mg / Vg + mx / Vx) / Pc; Vc = 1.0 / Pc
    elseif hasg
        mc, Vc = mg, Vg
    elseif hasx
        mc, Vc = mx, Vx
    else
        mc, Vc = 0.0, τ2
    end
    (; mg, Vg, hasg, mx, Vx, hasx, mc, Vc)
end

function run_pathway(pathway::Symbol, data::Dict{Symbol,Any},
                     donor_bank::Dict{Symbol,Dict{String,Tuple{Float64,Float64,Int}}})
    dat = data[pathway]
    mkpath(dat.results_dir)
    spec = make_spec(pathway)
    donors = [donor_bank[p] for p in ALL_PATHWAYS if p != pathway]

    methods = ["gaussian", "genus", "xpath", "transfer"]
    folds = sort!(unique(dat.org)); nf = length(folds); n = length(dat.y)
    K = length(REV_LEVELS)
    idx95 = findfirst(x -> isapprox(x, 0.95; atol = 1e-8), REV_LEVELS)
    zq = [quantile(Normal(), (1 + lv) / 2) for lv in REV_LEVELS]

    covered = Dict(m => Matrix{Bool}(undef, n, K) for m in methods)
    width95 = Dict(m => Vector{Float64}(undef, n) for m in methods)
    widthpar = Dict(m => Vector{Float64}(undef, n) for m in methods)
    predm = Dict(m => Vector{Float64}(undef, n) for m in methods)
    hassig = Dict(m => fill(false, n) for m in methods)
    rhos = fill(NaN, nf); omegas = fill(NaN, nf); iccs = fill(NaN, nf)
    taus = fill(NaN, nf)
    cellrows = [NamedTuple[] for _ in 1:nf]
    seeds = _pb_thread_seeds(MersenneTwister(REV_SEED), nf)

    println("\n=== $(pathway): N=$n organisms=$nf")

    Threads.@threads :dynamic for f in 1:nf
        rng_f = _pb_rng_from_seed(seeds[f])
        te = findall(==(folds[f]), dat.org); tr = findall(!=(folds[f]), dat.org)
        fit, cg = sigmacal_fit(rng_f, dat, tr, spec)
        τ2 = fit.hs.τ²; taus[f] = τ2
        offsets = cell_offsets(fit, dat.org_type[tr])
        σ2g = genus_var(offsets, τ2)
        ρ, ω2, _ = xpath_coupling(offsets, τ2, donors)
        rhos[f] = ρ; omegas[f] = ω2; iccs[f] = σ2g / τ2

        S = size(fit.qsamps, 2)
        te_cells = sort!(unique(dat.org_type[te]))
        preds = Dict(c => predict_offset(c, τ2, σ2g, offsets, ρ, ω2, donors)
                     for c in te_cells)
        for i in te
            a = view(dat.Arows, i, :)
            mu_s = [dot(a, view(fit.qsamps, :, s)) for s in 1:S]
            m_i = mean(mu_s); v_i = var(mu_s)
            pr = preds[dat.org_type[i]]
            offs = Dict("gaussian" => (0.0, τ2, false),
                        "genus" => (pr.mg, pr.Vg, pr.hasg),
                        "xpath" => (pr.mx, pr.Vx, pr.hasx),
                        "transfer" => (pr.mc, pr.Vc, pr.hasg || pr.hasx))
            for m in methods
                mb, Vb, sig = offs[m]
                sd = sqrt(v_i + Vb + cg * dat.sigma2[i])
                ctr = m_i + mb
                predm[m][i] = ctr
                hassig[m][i] = sig
                width95[m][i] = 2 * Z95 * sd
                widthpar[m][i] = 2 * Z95 * sqrt(v_i + Vb)
                for k in 1:K
                    covered[m][i, k] = abs(dat.y[i] - ctr) <= zq[k] * sd
                end
            end
        end
        # Realised offset proxy per held-out cell, for the offset-prediction
        # diagnostics: the mean residual of the cell's records under the
        # unseen-organism mean.
        bycell = Dict{String,Vector{Int}}()
        for i in te
            push!(get!(bycell, dat.org_type[i], Int[]), i)
        end
        for (c, is) in bycell
            pr = preds[c]
            mq = [mean(dot(view(dat.Arows, i, :), view(fit.qsamps, :, s)) for s in 1:S)
                  for i in is]
            proxy = mean(dat.y[is] .- mq)
            noise = mean(cg * dat.sigma2[i] for i in is) / length(is)
            push!(cellrows[f], (organism = folds[f], cell = c,
                                kind = String(last(split(c, "|"))), n_rec = length(is),
                                proxy = proxy, proxy_noise = noise, tau2 = τ2,
                                m_genus = pr.mg, V_genus = pr.Vg, has_genus = pr.hasg,
                                m_xpath = pr.mx, V_xpath = pr.Vx, has_xpath = pr.hasx,
                                m_transfer = pr.mc, V_transfer = pr.Vc))
        end
    end

    orgset = sort!(unique(dat.org))
    byorg = Dict(o => findall(==(o), dat.org) for o in orgset)
    cells_df = DataFrame(reduce(vcat, cellrows))
    cells_df[!, :pathway] .= string(pathway)
    proxy_ok = cells_df.n_rec .>= 2

    sumrows = NamedTuple[]
    covdf = DataFrame(pathway = String[], method = String[], nominal = Float64[],
                      empirical = Float64[])
    for m in methods
        cov_org = mean(mean(covered[m][byorg[o], idx95]) for o in orgset)
        sig = hassig[m]
        mcol = Dict("gaussian" => :proxy, "genus" => :m_genus, "xpath" => :m_xpath,
                    "transfer" => :m_transfer)[m]
        pred_cells = m == "gaussian" ? zeros(sum(proxy_ok)) : cells_df[proxy_ok, mcol]
        px = cells_df.proxy[proxy_ok]
        r2 = 1.0 - sum(abs2, px .- pred_cells) / sum(abs2, px)
        rmse_cell = sqrt(mean(abs2, px .- pred_cells))
        push!(sumrows, (pathway = string(pathway), method = m,
            Cov95 = mean(covered[m][:, idx95]), Cov95_org = cov_org,
            Width95 = mean(width95[m]), Width95_param = mean(widthpar[m]),
            RMSE_rec = sqrt(mean(abs2, dat.y .- predm[m])),
            RMSE_cell = rmse_cell, R2_offset = r2,
            frac_signal = mean(sig),
            Cov95_sig = any(sig) ? mean(covered[m][sig, idx95]) : NaN,
            Width95_sig = any(sig) ? mean(width95[m][sig]) : NaN,
            Width95_gauss_sig = any(sig) ? mean(width95["gaussian"][sig]) : NaN,
            Widthpar_sig = any(sig) ? mean(widthpar[m][sig]) : NaN,
            Widthpar_gauss_sig = any(sig) ? mean(widthpar["gaussian"][sig]) : NaN,
            rho = mean(filter(isfinite, rhos)), icc = mean(filter(isfinite, iccs)),
            tau2 = mean(filter(isfinite, taus)), n_folds = nf))
        for (k, lv) in enumerate(REV_LEVELS)
            push!(covdf, (string(pathway), m, lv, mean(covered[m][:, k])))
        end
        @printf("  %-9s Cov95=%.4f org=%.4f W95=%.2f Wpar=%.2f RMSEcell=%.3f R2=%.3f sig=%.2f\n",
                m, mean(covered[m][:, idx95]), cov_org, mean(width95[m]),
                mean(widthpar[m]),
                rmse_cell, r2, mean(sig))
    end
    @printf("  rho=%.3f icc=%.3f tau2=%.3f\n", mean(filter(isfinite, rhos)),
            mean(filter(isfinite, iccs)), mean(filter(isfinite, taus)))

    CSV.write(joinpath(dat.results_dir, "transfer_summary_$(pathway).csv"),
              DataFrame(sumrows))
    CSV.write(joinpath(dat.results_dir, "transfer_coverage_$(pathway).csv"), covdf)
    CSV.write(joinpath(dat.results_dir, "transfer_cells_$(pathway).csv"), cells_df)
    return nothing
end

println("Revision offset-transfer analysis")
println("  threads = $(Threads.nthreads())")
println("  started = $(now())")

data = Dict{Symbol,Any}(p => load_pathway(p) for p in ALL_PATHWAYS)

# Donor bank: full recalibrated fit of every pathway, all organisms.
donor_bank = Dict{Symbol,Dict{String,Tuple{Float64,Float64,Int}}}()
full_seeds = _pb_thread_seeds(MersenneTwister(REV_SEED + 1), length(ALL_PATHWAYS))
for (j, p) in enumerate(ALL_PATHWAYS)
    rng = _pb_rng_from_seed(full_seeds[j])
    fit, cg = sigmacal_fit(rng, data[p], collect(eachindex(data[p].y)), make_spec(p))
    donor_bank[p] = cell_offsets(fit, data[p].org_type)
    @printf("full fit %-18s c=%.2f tau2=%.3f cells=%d\n", p, cg, fit.hs.τ²,
            length(donor_bank[p]))
end

for pw in REV_PATHWAYS
    run_pathway(pw, data, donor_bank)
end
println("Done at $(now())")
