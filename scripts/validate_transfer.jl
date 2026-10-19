# validate_transfer.jl
#
# Validation of the offset-transfer layer of run_revision_transfer.jl on
# synthetic data with a known transfer structure.  Two pathways share the same
# organisms; the organism-by-type offsets decompose as
#
#     b_T = g + u + e_T,   b_D = g + u + e_D,
#
# with g a genus effect (shared by congeners and by pathways), u an
# organism effect shared across pathways, and e_T, e_D pathway-specific, so
# the cross-pathway coupling is rho = (sg2 + su2) / tau2 and the genus
# variance share is sg2 / tau2.  The record error is Gaussian with a
# misspecified scale (c = 2.5 on the nominal variance), handled by the same
# within-cell iteration as the real driver.
#
# Checks, scenario A (signal: sg2 = 0.5, su2 = 0.7, se2 = 0.3):
#   1. transfer and xpath keep 95% coverage on the held-out organisms;
#   2. on records with a signal, the transfer intervals are strictly
#      narrower than the baseline, and the parameter-level intervals are
#      narrower by at least 10%;
#   3. the predicted offsets explain a nontrivial share of the realised
#      cell-mean offsets (R2 > 0.2), and rho is recovered near its true 0.8.
# Checks, scenario B (null: sg2 = su2 = 0, se2 = 1.5):
#   4. coverage is preserved and the transfer widths and cell RMSE are within
#      5 and 10 per cent of the baseline: the clamps make the layer harmless
#      when there is nothing to transfer.
#
# Run:  julia --project=. --threads=8 scripts/validate_transfer.jl

import Pkg
Pkg.activate(dirname(@__DIR__))

using LinearAlgebra, Printf, Random, Statistics
using BlockArrays, DataFrames, Distributions, SpecialFunctions

include(joinpath(dirname(@__DIR__), "src", "parameter_balancing_core.jl"))
include(joinpath(dirname(@__DIR__), "src", "prediction_utils.jl"))

const C_TRUE = 2.5
const Z95 = quantile(Normal(), 0.975)
const SIG_ITERS = 2

genus_of(o::AbstractString) = String(first(split(o)))

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
    return clamp(num / den, 0.25, 25.0)
end

function cell_offsets(fit, cells)
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

function genus_var(offsets, τ2)
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

function xpath_coupling(offsets, τ2, donors)
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

function predict_offset(cellkey, τ2, σ2g, offsets, ρ, ω2, donors)
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

function make_pathway(rng, orgs, boff, n, d)
    grp = [orgs[1 + (i - 1) % length(orgs)] for i in 1:n]
    Arows = randn(rng, n, d) ./ sqrt(d)
    qtrue = randn(rng, d)
    sigma2 = fill(1.33, n)
    y = [dot(Arows[i, :], qtrue) + boff[grp[i]] +
         sqrt(C_TRUE * sigma2[i]) * randn(rng) for i in 1:n]
    cells = string.(grp, "|kcat")
    (; y, sigma2, Arows, grp, cells)
end

function sigmacal_fit(rng, eb, spec, A0, b0, Arows, y, sigma2, cells)
    cg = 1.0
    hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
    fit = fit_fold_mcem(rng, A0, b0, Arows, y, cg .* sigma2, cells, eb, hs, spec;
                        method = :collapsed)
    for _ in 1:SIG_ITERS
        M = Arows * fit.qsamps
        m = vec(mean(M, dims = 2))
        cg = c_update_within(y, m, sigma2, cells, eachindex(y))
        hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
        fit = fit_fold_mcem(rng, A0, b0, Arows, y, cg .* sigma2, cells, eb, hs, spec;
                            method = :collapsed)
    end
    fit, cg
end

function run_scenario(label, sg2, su2, se2)
    rng = MersenneTwister(20260812)
    G, n, d = 72, 1100, 15
    ngen = 24
    orgs = ["Genus$(1 + (j - 1) % ngen) org$j" for j in 1:G]
    geff = Dict("Genus$k" => sqrt(sg2) * randn(rng) for k in 1:ngen)
    ueff = Dict(o => sqrt(su2) * randn(rng) for o in orgs)
    bT = Dict(o => geff[genus_of(o)] + ueff[o] + sqrt(se2) * randn(rng) for o in orgs)
    bD = Dict(o => geff[genus_of(o)] + ueff[o] + sqrt(se2) * randn(rng) for o in orgs)
    T = make_pathway(rng, orgs, bT, n, d)
    D = make_pathway(rng, orgs, bD, n, d)

    eb = EBSpec(q0 = zeros(d), C0 = Matrix{Float64}(4.0I, d, d))
    spec = FoldFitSpec(model = :gaussian, hierarchical = true,
        n_outer = 3, n_mcmc = 150, burnin = 40, thin = 1, n_keep_min = 60,
        posterior_refresh = 1, adaptive_stop = false, min_outer = 2, tau_update = :ig,
        update_nu = false, update_alpha = false, refine_shape = false)
    A0 = zeros(d, d); b0 = zeros(d)

    fitD, _ = sigmacal_fit(MersenneTwister(11), eb, spec, A0, b0, D.Arows, D.y,
                           D.sigma2, D.cells)
    donors = [cell_offsets(fitD, D.cells)]

    methods = ["gaussian", "genus", "xpath", "transfer"]
    folds = sort!(unique(T.grp)); nf = length(folds)
    cov95 = Dict(m => Vector{Bool}(undef, n) for m in methods)
    w95 = Dict(m => Vector{Float64}(undef, n) for m in methods)
    wpar = Dict(m => Vector{Float64}(undef, n) for m in methods)
    sigf = Dict(m => fill(false, n) for m in methods)
    proxy = Dict{String,Float64}(); predc = Dict(m => Dict{String,Float64}() for m in methods)
    rhos = fill(NaN, nf)
    lk = ReentrantLock()
    seeds = _pb_thread_seeds(MersenneTwister(7), nf)

    Threads.@threads :dynamic for f in 1:nf
        rng_f = _pb_rng_from_seed(seeds[f])
        te = findall(==(folds[f]), T.grp); tr = findall(!=(folds[f]), T.grp)
        fit, cg = sigmacal_fit(rng_f, eb, spec, A0, b0, T.Arows[tr, :], T.y[tr],
                               T.sigma2[tr], T.cells[tr])
        τ2 = fit.hs.τ²
        offsets = cell_offsets(fit, T.cells[tr])
        σ2g = genus_var(offsets, τ2)
        ρ, ω2, _ = xpath_coupling(offsets, τ2, donors)
        rhos[f] = ρ
        S = size(fit.qsamps, 2)
        ck = string(folds[f], "|kcat")
        pr = predict_offset(ck, τ2, σ2g, offsets, ρ, ω2, donors)
        offs = Dict("gaussian" => (0.0, τ2, false), "genus" => (pr.mg, pr.Vg, pr.hasg),
                    "xpath" => (pr.mx, pr.Vx, pr.hasx),
                    "transfer" => (pr.mc, pr.Vc, pr.hasg || pr.hasx))
        mqs = Float64[]
        for i in te
            a = view(T.Arows, i, :)
            mu = [dot(a, view(fit.qsamps, :, s)) for s in 1:S]
            m_i = mean(mu); v_i = var(mu)
            push!(mqs, m_i)
            for m in methods
                mb, Vb, sg = offs[m]
                sd = sqrt(v_i + Vb + cg * T.sigma2[i])
                cov95[m][i] = abs(T.y[i] - (m_i + mb)) <= Z95 * sd
                w95[m][i] = 2 * Z95 * sd
                wpar[m][i] = 2 * Z95 * sqrt(v_i + Vb)
                sigf[m][i] = sg
            end
        end
        lock(lk) do
            proxy[ck] = mean(T.y[te]) - mean(mqs)
            for m in methods
                predc[m][ck] = offs[m][1]
            end
        end
    end

    res = Dict{String,Any}("rho" => mean(filter(isfinite, rhos)))
    pk = sort!(collect(keys(proxy)))
    px = [proxy[c] for c in pk]
    for m in methods
        pc = [predc[m][c] for c in pk]
        res["cov_$m"] = mean(cov95[m])
        res["w_$m"] = mean(w95[m])
        res["r2_$m"] = 1.0 - sum(abs2, px .- pc) / sum(abs2, px)
        res["rmse_$m"] = sqrt(mean(abs2, px .- pc))
        s = sigf[m]
        res["wsig_$m"] = any(s) ? mean(w95[m][s]) : NaN
        res["wsig0_$m"] = any(s) ? mean(w95["gaussian"][s]) : NaN
        res["wparsig_$m"] = any(s) ? mean(wpar[m][s]) : NaN
        res["wparsig0_$m"] = any(s) ? mean(wpar["gaussian"][s]) : NaN
        @printf("[%s] %-9s cov95=%.3f w=%.2f r2=%+.3f rmse_cell=%.3f\n", label, m,
                res["cov_$m"], res["w_$m"], res["r2_$m"], res["rmse_$m"])
    end
    @printf("[%s] rho_hat=%.3f (signal frac xpath %.2f, genus %.2f)\n", label,
            res["rho"], mean(sigf["xpath"]), mean(sigf["genus"]))
    res
end

println("Scenario A: signal (sg2=0.5, su2=0.7, se2=0.3; rho_true=0.8, icc=1/3)")
A = run_scenario("A", 0.5, 0.7, 0.3)
println("\nScenario B: null (sg2=0, su2=0, se2=1.5)")
B = run_scenario("B", 0.0, 0.0, 1.5)

failures = String[]
check(name, ok) = (ok ? @printf("  PASS  %s\n", name) :
                        (push!(failures, name); @printf("  FAIL  %s\n", name)))
println()
check("A: transfer keeps 95% coverage", abs(A["cov_transfer"] - 0.95) <= 0.04)
check("A: xpath keeps 95% coverage", abs(A["cov_xpath"] - 0.95) <= 0.04)
check("A: transfer narrower than baseline on signal records",
      A["wsig_transfer"] < A["wsig0_transfer"])
check("A: parameter-level intervals at least 10% narrower on signal records",
      A["wparsig_transfer"] < 0.90 * A["wparsig0_transfer"])
check("A: offsets are predicted (R2 > 0.2 for xpath and transfer)",
      A["r2_xpath"] > 0.2 && A["r2_transfer"] > 0.2)
check("A: rho recovered near 0.8", 0.5 <= A["rho"] <= 1.1)
check("B: all methods keep 95% coverage under the null",
      all(abs(B["cov_$m"] - 0.95) <= 0.04 for m in ["gaussian", "genus", "xpath", "transfer"]))
check("B: transfer widths within 5% of baseline under the null",
      B["w_transfer"] <= 1.05 * B["w_gaussian"])
check("B: transfer cell RMSE within 10% of baseline under the null",
      B["rmse_transfer"] <= 1.10 * B["rmse_gaussian"])

println()
if isempty(failures)
    println("All offset-transfer validation checks passed.")
else
    println("FAILED checks: ", join(failures, "; "))
    exit(1)
end
