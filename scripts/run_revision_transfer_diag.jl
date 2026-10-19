# run_revision_transfer_diag.jl
#
# Assumption-light diagnostics behind the offset-transfer null of
# run_revision_transfer.jl: for every pair of pathways, the raw Pearson
# correlation between the SAME organism-by-type cell's fitted offsets in the
# two full recalibrated fits, with a permutation p-value, overall and
# restricted to well-estimated cells (at least three records on both sides,
# where posterior shrinkage cannot hide a real signal).  The full fits and
# their seeds reproduce the donor bank of run_revision_transfer.jl exactly.
#
# Output: output/transfer_diagnostics/transfer_diag_pairs.csv

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

const REV_SEED = 20260812
const ALL_PATHWAYS = [:glycolysis, :tca, :pentose_phosphate, :shikimate]
const SIGMACAL_ITERS = parse(Int, get(ENV, "PB_REV_SIGMACAL_ITERS", "4"))
const C_MIN, C_MAX = 0.25, 25.0
const N_PERM = 20000

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
    (; y, sigma2, Arows, org, kinds, org_type, eb, A0, b0)
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

perm_p(rng, xs, ys, robs) = begin
    hits = 0
    yp = copy(ys)
    for _ in 1:N_PERM
        shuffle!(rng, yp)
        abs(cor(xs, yp)) >= abs(robs) && (hits += 1)
    end
    (hits + 1) / (N_PERM + 1)
end

println("Offset-transfer diagnostics; started $(now())")
data = Dict{Symbol,Any}(p => load_pathway(p) for p in ALL_PATHWAYS)
bank = Dict{Symbol,Dict{String,Tuple{Float64,Float64,Int}}}()
full_seeds = _pb_thread_seeds(MersenneTwister(REV_SEED + 1), length(ALL_PATHWAYS))
for (j, p) in enumerate(ALL_PATHWAYS)
    rng = _pb_rng_from_seed(full_seeds[j])
    fit, cg = sigmacal_fit(rng, data[p], collect(eachindex(data[p].y)), make_spec(p))
    bank[p] = cell_offsets(fit, data[p].org_type)
    @printf("full fit %-18s c=%.2f tau2=%.3f cells=%d\n", p, cg, fit.hs.τ²,
            length(bank[p]))
end

rng = MersenneTwister(REV_SEED + 2)
rows = NamedTuple[]
allx = Float64[]; ally = Float64[]
allxw = Float64[]; allyw = Float64[]
for i in 1:4, j in (i + 1):4
    a, b = ALL_PATHWAYS[i], ALL_PATHWAYS[j]
    shared = sort!(collect(intersect(keys(bank[a]), keys(bank[b]))))
    xs = [bank[a][c][1] for c in shared]; ys = [bank[b][c][1] for c in shared]
    well = [c for c in shared if bank[a][c][3] >= 3 && bank[b][c][3] >= 3]
    xw = [bank[a][c][1] for c in well]; yw = [bank[b][c][1] for c in well]
    r = length(xs) >= 3 ? cor(xs, ys) : NaN
    rw = length(xw) >= 3 ? cor(xw, yw) : NaN
    p_r = isfinite(r) ? perm_p(rng, xs, ys, r) : NaN
    p_rw = isfinite(rw) ? perm_p(rng, xw, yw, rw) : NaN
    append!(allx, xs); append!(ally, ys)
    append!(allxw, xw); append!(allyw, yw)
    push!(rows, (pathway_a = string(a), pathway_b = string(b),
                 n_shared = length(xs), r = r, p_perm = p_r,
                 n_well = length(xw), r_well = rw, p_perm_well = p_rw))
    @printf("%-18s x %-18s n=%3d r=%+.3f (p=%.3f)  well n=%3d r=%+.3f (p=%.3f)\n",
            a, b, length(xs), r, p_r, length(xw), rw, p_rw)
end
rp = cor(allx, ally); rpw = cor(allxw, allyw)
push!(rows, (pathway_a = "pooled", pathway_b = "all pairs",
             n_shared = length(allx), r = rp, p_perm = perm_p(rng, allx, ally, rp),
             n_well = length(allxw), r_well = rpw,
             p_perm_well = perm_p(rng, allxw, allyw, rpw)))
@printf("pooled: n=%d r=%+.3f  well n=%d r=%+.3f\n", length(allx), rp,
        length(allxw), rpw)

outdir = joinpath(dirname(@__DIR__), "output", "transfer_diagnostics"); mkpath(outdir)
CSV.write(joinpath(outdir, "transfer_diag_pairs.csv"), DataFrame(rows))
println("Done at $(now())")
