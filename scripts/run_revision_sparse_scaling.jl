# run_revision_sparse_scaling.jl
#
# Sparsity-aware posterior solve for genome-scale parameter balancing, added
# for the revision.
#
# Limitation addressed:
#   The scaling discussion identifies the dense D^3 Cholesky of the posterior
#   precision as the binding constraint for genome-scale application, and
#   names sparsity-aware exploitation of the dependency structure as the
#   needed direction without attempting it.  This driver attempts it.
#
# The posterior precision of the basis parameters is
#
#     P = sum_i w_i a_i a_i'  +  (1/lambda) C0^{-1}  +  A0 ,
#
# and both structural facts that make it sparse are generic to parameter
# balancing rather than special to our pathways: an observation row touches
# only the few basis coordinates its record depends on, and the prior
# precision is block structured, with a dense thermodynamic block of size m
# (the formation-energy covariance of the m metabolites) and near-diagonal
# kinetic blocks.  The resulting precision is "sparse plus a small dense
# block", which a sparse Cholesky factorises with bounded fill-in.  Sampling
# uses the same factor: with P[p, p] = L L' from CHOLMOD, a draw is obtained
# by solving L' v = z for z ~ N(0, I) and undoing the permutation.
#
# The benchmark compares the existing dense path (assembly + dense Cholesky +
# mean solve + one sample) against the sparse path on synthetic problems of
# growing dimension with realistic row support (each record touches at most 4
# basis coordinates) and a dense thermo block of size D/10.  The correctness
# check verifies that both paths return the same posterior mean and that the
# sample covariance of the sparse draws matches the dense posterior
# covariance on a moderate problem.
#
# Outputs:
#   output/scaling/sparse_scaling.csv   one row per (D, path) with timings
#
# Environment controls:
#   PB_SPARSE_DIMS  comma-separated D values (default 250,500,1000,2000,4000)

import Pkg
Pkg.activate(dirname(@__DIR__))

using Dates, LinearAlgebra, Printf, Random, SparseArrays, Statistics
using CSV, DataFrames

const DIMS = parse.(Int, split(get(ENV, "PB_SPARSE_DIMS", "250,500,1000,2000,4000,8000"), ","))
const N_PER_D = 5          # records per basis coordinate
const SUPPORT = 4          # basis coordinates touched per record
const REPS = 3             # timing repetitions, best-of

"""
    make_problem(rng, D) -> (rows, vals, w, yv, Pi_prior)

Synthetic balancing problem with the sparsity structure that parameter
balancing actually has, rather than random supports (which produce an
expander graph and near-total Cholesky fill-in).  Coordinates 1..m are the
formation-energy (thermo) block, m = D/10, with a dense SPD prior; the
remaining kinetic coordinates are partitioned into per-reaction blocks of six
(velocities and Michaelis constants of one reaction).  A record touches two
to four coordinates of its own reaction's kinetic block, one or two thermo
coordinates from that reaction's metabolite window, and, with probability
0.3, one hub metabolite (currency metabolites such as ATP couple many
reactions).  The resulting precision is block-local with a small dense
border, the arrowhead structure a genome-scale reconstruction produces.
"""
function make_problem(rng, D)
    m = max(D ÷ 10, 5)
    n_kin = D - m
    blocksz = 6
    n_rxn = max(n_kin ÷ blocksz, 1)
    n = N_PER_D * D
    n_hub = 5
    rows = Vector{Vector{Int}}(undef, n)
    vals = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        r = 1 + (i - 1) % n_rxn                        # this record's reaction
        base = m + (r - 1) * blocksz
        k = rand(rng, 2:min(SUPPORT, blocksz))
        kin = base .+ sort!(randperm(rng, blocksz)[1:k])
        win = n_hub + 1 + (r * 3) % max(m - n_hub - 3, 1)
        thermo = [win + rand(rng, 0:2)]
        rand(rng) < 0.3 && push!(thermo, rand(rng, 1:n_hub))
        idx = sort!(unique(vcat(kin, thermo)))
        rows[i] = idx
        vals[i] = randn(rng, length(idx))
    end
    w = 1.0 ./ (0.5 .+ rand(rng, n))
    yv = randn(rng, n)
    B = randn(rng, m, m)
    thermo_prior = B * B' + m * I
    Pi = spzeros(D, D)
    Pi[1:m, 1:m] = sparse(thermo_prior)
    for j in (m + 1):D
        Pi[j, j] = 0.5 + rand(rng)
    end
    return rows, vals, w, yv, Pi
end

function assemble_dense(rows, vals, w, yv, Pi, D)
    P = Matrix(Pi)
    r = zeros(D)
    for i in eachindex(rows)
        ri = rows[i]; vi = vals[i]; wi = w[i]
        for a in eachindex(ri), b in eachindex(ri)
            P[ri[a], ri[b]] += wi * vi[a] * vi[b]
        end
        for a in eachindex(ri)
            r[ri[a]] += wi * yv[i] * vi[a]
        end
    end
    return P, r
end

function assemble_sparse(rows, vals, w, yv, Pi, D)
    I_ = Int[]; J_ = Int[]; V_ = Float64[]
    for i in eachindex(rows)
        ri = rows[i]; vi = vals[i]; wi = w[i]
        for a in eachindex(ri), b in eachindex(ri)
            push!(I_, ri[a]); push!(J_, ri[b]); push!(V_, wi * vi[a] * vi[b])
        end
    end
    P = sparse(I_, J_, V_, D, D) + Pi
    r = zeros(D)
    for i in eachindex(rows)
        ri = rows[i]; vi = vals[i]
        for a in eachindex(ri)
            r[ri[a]] += w[i] * yv[i] * vi[a]
        end
    end
    return P, r
end

"""
    sparse_solve_sample(P, r, rng) -> (mu, x)

Posterior mean and one exact sample using a sparse CHOLMOD Cholesky:
`P[p, p] = L L'`, draw z ~ N(0, I), solve `L' v = z`, unpermute.
"""
function sparse_solve_sample(P::SparseMatrixCSC, r::Vector{Float64}, rng)
    F = cholesky(Symmetric(P))
    μ = F \ r
    L = sparse(F.L)
    p = F.p
    z = randn(rng, length(r))
    v = L' \ z
    x = similar(μ)
    x[p] = v
    return μ, μ .+ x
end

function dense_solve_sample(P::Matrix{Float64}, r::Vector{Float64}, rng)
    F = cholesky(Symmetric(P))
    μ = F \ r
    z = randn(rng, length(r))
    x = F.U \ z
    return μ, μ .+ x
end

# Correctness check on a moderate problem: identical means, matching
# marginal variances from repeated sparse draws.
let rng = MersenneTwister(1)
    D = 300
    rows, vals, w, yv, Pi = make_problem(rng, D)
    Pd, rd = assemble_dense(rows, vals, w, yv, Pi, D)
    Ps, rs = assemble_sparse(rows, vals, w, yv, Pi, D)
    μd, _ = dense_solve_sample(Pd, rd, rng)
    μs, _ = sparse_solve_sample(Ps, rs, rng)
    Δμ = maximum(abs.(μd .- μs))
    S = 20_000
    F = cholesky(Symmetric(Ps)); L = sparse(F.L); p = F.p
    acc = zeros(D)
    for _ in 1:S
        v = L' \ randn(rng, D)
        x = similar(v); x[p] = v
        acc .+= x .^ 2
    end
    var_sparse = acc ./ S
    var_dense = diag(inv(Symmetric(Pd)))
    ratio = maximum(abs.(var_sparse ./ var_dense .- 1))
    @printf("correctness: max |mu_dense - mu_sparse| = %.2e, max |var ratio - 1| = %.3f over %d draws\n",
            Δμ, ratio, S)
    (Δμ < 1e-8 && ratio < 0.10) || error("sparse path failed the correctness check")
end

results = DataFrame(D = Int[], n = Int[], path = String[],
                    assemble_s = Float64[], factor_solve_sample_s = Float64[],
                    fill_nnz = Float64[])
for D in DIMS
    rng = MersenneTwister(100 + D)
    rows, vals, w, yv, Pi = make_problem(rng, D)
    n = length(w)

    ta = minimum([@elapsed assemble_sparse(rows, vals, w, yv, Pi, D) for _ in 1:REPS])
    Ps, rs = assemble_sparse(rows, vals, w, yv, Pi, D)
    ts = minimum([@elapsed sparse_solve_sample(Ps, rs, rng) for _ in 1:REPS])
    Fs = cholesky(Symmetric(Ps))
    push!(results, (D, n, "sparse", ta, ts, nnz(sparse(Fs.L)) / (D * (D + 1) / 2)))
    @printf("D=%5d n=%6d sparse: assemble %.3fs, factor+solve+sample %.3fs, fill %.3f\n",
            D, n, ta, ts, results.fill_nnz[end])

    if D <= 2000   # dense path becomes memory- and time-hostile beyond this
        td_a = minimum([@elapsed assemble_dense(rows, vals, w, yv, Pi, D) for _ in 1:REPS])
        Pd, rd = assemble_dense(rows, vals, w, yv, Pi, D)
        td = minimum([@elapsed dense_solve_sample(Pd, rd, rng) for _ in 1:REPS])
        push!(results, (D, n, "dense", td_a, td, 1.0))
        @printf("D=%5d n=%6d dense : assemble %.3fs, factor+solve+sample %.3fs\n", D, n, td_a, td)
    end
end

# Hierarchical stage: the collapsed update subtracts the Schur complement
# sum_g z_g z_g' / d_g, where z_g = sum_{i in g} w_i a_i has the union of the
# group's record supports.  Real source structure is bimodal: most organisms
# contribute records to one or two adjacent reactions (local support), while a
# few model organisms span the whole pathway (dense support).  The downdate is
# formed sparsely as P - S S' with S = [z_g / sqrt(d_g)]_g and refactorised.
println("\nHierarchical (collapsed) stage: local organisms + 5 pathway-wide organisms")
hier = DataFrame(D = Int[], G = Int[], path = String[],
                 factor_solve_sample_s = Float64[], fill_nnz = Float64[])
for D in DIMS
    rng = MersenneTwister(200 + D)
    rows, vals, w, yv, Pi = make_problem(rng, D)
    n = length(w)
    m = max(D ÷ 10, 5)
    n_kin = D - m
    n_rxn = max(n_kin ÷ 6, 1)
    G = max(D ÷ 10, 10)
    n_global = 5
    # Record i belongs to reaction 1 + (i-1) % n_rxn (as in make_problem).
    # Local organisms own a window of two adjacent reactions; global organisms
    # draw records from everywhere (10% of records, mirroring the dominant
    # model organisms in the curated data).
    grp = Vector{Int}(undef, n)
    for i in 1:n
        if rand(rng) < 0.10
            grp[i] = rand(rng, 1:n_global)
        else
            r = 1 + (i - 1) % n_rxn
            grp[i] = n_global + 1 + (r ÷ 2) % (G - n_global)
        end
    end
    tau2 = 1.0
    Ps, rs = assemble_sparse(rows, vals, w, yv, Pi, D)
    # z_g and d_g
    Iz = Int[]; Jz = Int[]; Vz = Float64[]
    dg = fill(1.0 / tau2, G)
    for i in 1:n
        g = grp[i]
        dg[g] += w[i]
        for (a, r_) in enumerate(rows[i])
            push!(Iz, r_); push!(Jz, g); push!(Vz, w[i] * vals[i][a])
        end
    end
    Z = sparse(Iz, Jz, Vz, D, G)
    S = Z * spdiagm(0 => 1.0 ./ sqrt.(dg))
    Pm = Ps - S * S'
    t = minimum([@elapsed begin
            F = cholesky(Symmetric(Pm))
            μ = F \ rs
            L = sparse(F.L)
            v = L' \ randn(rng, D)
            x = similar(μ); x[F.p] = v
        end for _ in 1:REPS])
    F = cholesky(Symmetric(Pm))
    fillr = nnz(sparse(F.L)) / (D * (D + 1) / 2)
    push!(hier, (D, G, "sparse_hier", t, fillr))
    @printf("D=%5d G=%4d hier-sparse: factor+solve+sample %.3fs, fill %.3f\n", D, G, t, fillr)

    # Woodbury variant: only the LOCAL groups are folded into the sparse
    # downdate (their supports are reaction-local, so fill stays bounded);
    # the few pathway-wide organisms enter as a rank-n_global correction via
    # the Woodbury identity.  With A = P - S_loc S_loc', W = A^{-1} S_glob and
    # M = I - S_glob' W (the capacitance), the collapsed covariance is
    # (A - S_glob S_glob')^{-1} = A^{-1} + W M^{-1} W', so the mean is
    # A^{-1} r + W M^{-1} (W' r) and an exact sample is u + W m with
    # u ~ N(0, A^{-1}) and m ~ N(0, M^{-1}): a sum of independent Gaussians
    # with exactly the target covariance.
    loc = collect(n_global+1:G)
    Sl = S[:, loc]
    Sg = Matrix(S[:, 1:n_global])
    Pl = Ps - Sl * Sl'
    tw = minimum([@elapsed begin
            F = cholesky(Symmetric(Pl))
            W = F \ Sg
            M = Symmetric(Matrix(I, n_global, n_global) - Sg' * W)
            FM = cholesky(M)
            μ = (F \ rs) + W * (FM \ (Sg' * (F \ rs)))
            L = sparse(F.L)
            v = L' \ randn(rng, D)
            u = similar(μ); u[F.p] = v
            x = u + W * (FM.U \ randn(rng, n_global))
        end for _ in 1:REPS])
    Fl = cholesky(Symmetric(Pl))
    fillw = nnz(sparse(Fl.L)) / (D * (D + 1) / 2)
    push!(hier, (D, G, "sparse_hier_woodbury", tw, fillw))
    @printf("D=%5d G=%4d hier-woodbury: factor+solve+sample %.3fs, fill %.3f\n", D, G, tw, fillw)
end

# Thermo-block stage: the formation-energy covariance is not an arbitrary
# dense matrix.  Component contribution returns it as a finite-rank factor
# model, Sigma = S_fin S_fin' + 1e6 S_inf S_inf' (see
# data/equilibrator/equilibrator_glycolysis.py), so with any ridge the solves the
# prior assembly needs reduce to the component rank k << m via the Woodbury
# identity, and the m^3 dense Cholesky disappears as well.
println("\nThermo-block stage: dense Cholesky vs factor (Woodbury) solve, k = m/10")
thermo = DataFrame(m = Int[], k = Int[], path = String[], solve_s = Float64[])
for m in [1000, 2000, 4000, 8000]
    rng = MersenneTwister(300 + m)
    k = m ÷ 10
    Sf = randn(rng, m, k)
    λ = 1.0
    r = randn(rng, m)
    if m <= 4000
        Σ = Sf * Sf' + λ * I
        td = minimum([@elapsed cholesky(Symmetric(Σ)) \ r for _ in 1:REPS])
        push!(thermo, (m, k, "dense", td))
        @printf("m=%5d k=%4d dense   : solve %.3fs\n", m, k, td)
    end
    # Woodbury: (SS' + laI)^{-1} r = (r - S (la I_k + S'S)^{-1} S' r) / la.
    tw = minimum([@elapsed begin
            M = cholesky(Symmetric(λ * Matrix(I, k, k) + Sf' * Sf))
            x = (r - Sf * (M \ (Sf' * r))) ./ λ
        end for _ in 1:REPS])
    push!(thermo, (m, k, "factor_woodbury", tw))
    @printf("m=%5d k=%4d woodbury: solve %.3fs\n", m, k, tw)
end

mkpath(joinpath(dirname(@__DIR__), "output", "scaling"))
CSV.write(joinpath(dirname(@__DIR__), "output", "scaling", "sparse_scaling.csv"), results)
CSV.write(joinpath(dirname(@__DIR__), "output", "scaling", "sparse_scaling_hier.csv"), hier)
CSV.write(joinpath(dirname(@__DIR__), "output", "scaling", "sparse_scaling_thermo.csv"), thermo)
show(stdout, results; allrows = true, allcols = true); println()
show(stdout, hier; allrows = true, allcols = true); println()
println("Done at $(now())")
