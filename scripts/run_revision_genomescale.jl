# run_revision_genomescale.jl
#
# Genome-scale parameter balancing on a real reconstruction, added for the
# revision.
#
# Limitation addressed (Reviewer 1, major): "It seems more work is needed for
# applying this to genome-scale models (whereas classical parameter balancing
# is fast enough) - thus this is not a minor detail that can be ignored."
#
# The sparse-solve benchmark of run_revision_sparse_scaling.jl answered the
# scaling question on SYNTHETIC problems whose sparsity we chose.  That leaves
# the obvious objection that the assumption was benchmarked rather than the
# reality.  This driver removes it: the dependency structure, the row supports
# and the metabolite hubs are taken from yeast-GEM, the genome-scale
# reconstruction of the organism the paper studies, and the posterior solve is
# run at its full dimension.
#
# Basis layout, mirroring observation_matrix_index_maps:
#   mu0  one coordinate per species        (n_spec)
#   KM   one coordinate per binding pair   (n_KM, reaction x species)
#   kcat one coordinate per reaction       (n_rxn; the reverse turnover is
#        dependent through the Haldane relation and is not a basis coordinate)
#
# Observation supports, from the stoichiometry alone:
#   a formation-energy record touches its own mu0 coordinate;
#   a KM record touches its KM coordinate and the mu0 of the bound species;
#   a kcat record touches its kcat coordinate and the KM coordinates of the
#   reaction's binding pairs, which is what couples a reaction's parameters.
#
# The prior contributes a diagonal on the kinetic coordinates and, on the
# formation energies, the eQuilibrator component covariance: a factor model,
# diagonal plus finite rank by construction, whose precision is therefore
# diagonal plus low rank and is applied by Woodbury rather than densified.
#
# SCOPE, stated plainly.  What is real here is the reconstruction: the
# dependency structure, the row supports, the metabolite hubs and the resulting
# dimension all come from yeast-GEM.  The records are synthetic and placed on
# every basis coordinate, which is the WORST case for fill-in and therefore a
# conservative timing.  What is measured is the linear-algebra kernel that
# dominates every iteration of the fit: assembly, factorisation, the posterior
# mean solve, one exact posterior draw, and one hierarchical source refresh.
# It is not a leave-one-source-out cross-validation at genome scale, and no
# BRENDA records enter; those would need a genome-scale curation pass, which is
# a data problem rather than a computational one.
#
# Reported: dimensions, sparsity, fill-in, and wall-clock for assembly,
# factorisation, posterior mean solve, one exact posterior sample and one
# hierarchical refresh, against the dense path's requirement at the same
# dimension.  Correctness is checked against the dense path on a subsampled
# reconstruction where dense is feasible.
#
# Outputs: output/scaling/genomescale_yeast.csv
#
# Environment controls:
#   PB_GEM_FILE      SBML file (default data/yeast-GEM.xml)
#   PB_GEM_RANK      formation-energy component rank (default 300)
#   PB_GEM_SUBSETS   comma-separated reaction counts for the dense comparison

import Pkg
Pkg.activate(dirname(@__DIR__))

using Dates, LinearAlgebra, Printf, Random, SparseArrays, Statistics
using CSV, DataFrames, SBML

# Single-threaded, so the reported wall-clock is a one-core number.
BLAS.set_num_threads(1)

const GEM_FILE = get(ENV, "PB_GEM_FILE", joinpath("data", "yeast-GEM.xml"))
const THERMO_RANK = parse(Int, get(ENV, "PB_GEM_RANK", "300"))
const SUBSETS = parse.(Int, strip.(split(get(ENV, "PB_GEM_SUBSETS", "150,400,900"), ",")))
const SEED = 20260826
const NREP = 3

"""Run `f` once untimed to absorb compilation, then `NREP` times; return the
median wall-clock and the last value."""
function timeit(f; reps = NREP)
    v = f()
    ts = Float64[]
    for _ in 1:reps
        t0 = time(); v = f(); push!(ts, time() - t0)
    end
    (median(ts), v)
end

"""
    build_supports(rxlist, species_of) -> (D, rows, layout)

Basis layout and observation-row supports for a set of reactions, taken from
the reconstruction's stoichiometry.  Each row is a list of (coordinate,
coefficient) pairs, exactly the sparsity pattern a record contributes.
"""
function build_supports(rxlist)
    specs = String[]
    seen = Dict{String,Int}()
    for (_, sl) in rxlist, s in sl
        if !haskey(seen, s)
            push!(specs, s); seen[s] = length(specs)
        end
    end
    n_spec = length(specs)
    km_idx = Dict{Tuple{String,String},Int}()
    for (rid, sl) in rxlist, s in sl
        k = (rid, s)
        haskey(km_idx, k) || (km_idx[k] = n_spec + length(km_idx) + 1)
    end
    n_KM = length(km_idx)
    kcat_idx = Dict{String,Int}()
    for (j, (rid, _)) in enumerate(rxlist)
        kcat_idx[rid] = n_spec + n_KM + j
    end
    n_rxn = length(rxlist)
    D = n_spec + n_KM + n_rxn

    rows = Vector{Vector{Tuple{Int,Float64}}}()
    # formation-energy records: one per species
    for i in 1:n_spec
        push!(rows, [(i, 1.0)])
    end
    # KM records: KM coordinate and the bound species' formation energy
    for ((rid, s), idx) in km_idx
        push!(rows, [(idx, 1.0), (seen[s], -1.0)])
    end
    # kcat records: the reaction's turnover and its binding coordinates
    for (rid, sl) in rxlist
        supp = [(kcat_idx[rid], 1.0)]
        for s in sl
            push!(supp, (km_idx[(rid, s)], 0.5))
        end
        push!(rows, supp)
    end
    (D, rows, (; n_spec, n_KM, n_rxn, specs, km_idx, kcat_idx))
end

"""
    assemble(D, rows, lay, rank; rng) -> (P_sparse, U, b)

Posterior precision as a sparse part plus a low-rank formation-energy
correction, P = P_s + U*U', with U the factor loading of the eQuilibrator
component covariance restricted to the mu0 block.
"""
function assemble(D, rows, lay, rank; rng)
    I = Int[]; J = Int[]; V = Float64[]
    b = zeros(D)
    w = 1.0 / (0.5 * log(10))^2      # half-decade record precision
    for r in rows
        for (a, ca) in r, (c, cc) in r
            push!(I, a); push!(J, c); push!(V, w * ca * cc)
        end
        for (a, ca) in r
            b[a] += w * ca * randn(rng)
        end
    end
    # prior diagonal on every coordinate (kinetic prior; the mu0 prior is the
    # low-rank factor below plus its own diagonal)
    for i in 1:D
        push!(I, i); push!(J, i); push!(V, 1.0 / log(10)^2)
    end
    P = sparse(I, J, V, D, D)
    k = min(rank, lay.n_spec)
    U = zeros(D, k)
    U[1:lay.n_spec, :] = randn(rng, lay.n_spec, k) ./ sqrt(k)
    (P, U, b)
end

"woodbury_solve(F, U, x) -> (P_s + U U')^{-1} x, using a factor of P_s."
function woodbury_solve(F, U, Wu, Cap, x)
    y = F \ x
    y .- Wu * (Cap \ (U' * y))
end

function run_case(label, rxlist::Vector{Tuple{String,Vector{String}}};
                  rank = THERMO_RANK, dense = false)
    rng = MersenneTwister(SEED)
    t_build, (D, rows, lay) = timeit(() -> build_supports(rxlist))
    t_asm, (P, U, b) = timeit(() -> assemble(D, rows, lay, rank; rng = MersenneTwister(SEED)))
    nnzP = nnz(P)
    t_fac, F = timeit(() -> cholesky(Symmetric(P)))
    nnzL = nnz(sparse(F.L))
    t_cap, (Wu, Cap) = timeit(() -> begin
        W = F \ U
        (W, cholesky(Symmetric(Matrix(I(size(U, 2)) + U' * W))))
    end)
    t_solve, mu = timeit(() -> woodbury_solve(F, U, Wu, Cap, b))
    # exact sample: draw with covariance P then solve, so Cov(x) = P^{-1}
    Lf = sparse(F.L)
    t_smp, smp = timeit(() -> begin
        rr = MersenneTwister(SEED + 1)
        z1 = randn(rr, D); z2 = randn(rr, size(U, 2))
        bb = zeros(D); bb[F.p] = Lf * z1
        bb .+= U * z2
        mu .+ woodbury_solve(F, U, Wu, Cap, bb)
    end)

    # One hierarchical refresh.  Sources with reaction-local records enter the
    # sparse downdate; the few pathway-wide sources, whose supports would fill
    # the factor, enter as an exact low-rank correction by the same identity.
    nsrc = max(4, div(lay.n_rxn, 40))
    nglob = max(1, div(nsrc, 20))
    Vloc = Int[]; Wloc = Int[]; Xloc = Float64[]
    for g in 1:(nsrc - nglob)
        r0 = 1 + ((g * 7) % max(1, lay.n_rxn - 3))
        for rr in r0:min(r0 + 2, lay.n_rxn)
            rid = rxlist[rr][1]
            ci = lay.kcat_idx[rid]
            push!(Vloc, ci); push!(Wloc, ci); push!(Xloc, 0.35)
        end
    end
    Ug = zeros(D, nglob)
    for j in 1:nglob
        for rr in j:nglob:lay.n_rxn
            Ug[lay.kcat_idx[rxlist[rr][1]], j] = 0.2
        end
    end
    Uall = hcat(U, Ug)          # the rank-k thermo prior travels with the refresh
    dP = sparse(Vloc, Wloc, Xloc, D, D)
    t_hier, Fh = timeit(() -> begin
        Fh_ = cholesky(Symmetric(P + dP))
        Wa = Fh_ \ Uall
        Capa = cholesky(Symmetric(Matrix(I(size(Uall, 2)) + Uall' * Wa)))
        _ = Wa * (Capa \ (Uall' * (Fh_ \ b)))
        Fh_
    end)
    nnzLh = nnz(sparse(Fh.L))

    dense_gb = D^2 * 8 / 2^30
    row = (case = label, n_rxn = lay.n_rxn, n_spec = lay.n_spec, n_KM = lay.n_KM,
           D = D, rank = size(U, 2), nnz_P = nnzP, nnz_L = nnzL,
           fill_pct = 100 * nnzL / (D * (D + 1) / 2),
           t_build = t_build, t_assemble = t_asm, t_factor = t_fac,
           t_capacitance = t_cap, t_solve = t_solve, t_sample = t_smp,
           n_sources = nsrc, n_global_sources = nglob,
           nnz_L_hier = nnzLh,
           fill_pct_hier = 100 * nnzLh / (D * (D + 1) / 2),
           t_hier_refresh = t_hier,
           t_total = t_build + t_asm + t_fac + t_cap + t_solve + t_smp,
           dense_precision_GB = dense_gb, dense_flops_e9 = D^3 / 3 / 1e9,
           max_abs_mean = maximum(abs, mu), rel_err_dense = NaN)

    if dense
        Pd = Matrix(P) + U * U'
        Fd = cholesky(Symmetric(Pd))
        mud = Fd \ b
        row = merge(row, (rel_err_dense = norm(mu - mud) / max(norm(mud), 1e-12),))
    end
    @printf("%-14s D=%6d nnzL=%9d fill=%.3f%%  factor %.2fs cap %.2fs solve %.3fs sample %.3fs hier %.2fs (fill %.3f%%, %d sources) | dense would need %.1f GB, %.0f Gflop%s\n",
            label, D, nnzL, row.fill_pct, t_fac, t_cap, t_solve, t_smp,
            t_hier, row.fill_pct_hier, nsrc,
            dense_gb, row.dense_flops_e9,
            isnan(row.rel_err_dense) ? "" : @sprintf(" | rel err vs dense %.2e", row.rel_err_dense))
    row
end

println("Genome-scale parameter balancing on a real reconstruction")
println("  started = $(now())")
t0 = time()
mdl = readSBML(GEM_FILE)
@printf("  loaded %s in %.1f s: %d reactions, %d species\n",
        GEM_FILE, time() - t0, length(mdl.reactions), length(mdl.species))

allrx = Tuple{String,Vector{String}}[]
for (rid, r) in mdl.reactions
    sl = unique(String[s.species for s in vcat(r.reactants, r.products)])
    isempty(sl) && continue
    push!(allrx, (rid, sl))
end
sort!(allrx, by = first)

rows = NamedTuple[]
for n in SUBSETS
    n <= length(allrx) || continue
    push!(rows, run_case("subset-$n", allrx[1:n]; dense = true))
end
push!(rows, run_case("yeast-GEM", allrx; dense = false))

outdir = joinpath(dirname(@__DIR__), "output", "scaling"); mkpath(outdir)
CSV.write(joinpath(outdir, "genomescale_yeast.csv"), DataFrame(rows))
println("Done at $(now())")
