# run_parallel_scaling_benchmark.jl
#
# Small, reproducible timing experiment for the repository-level threaded
# execution paths.  Run this script several times with different Julia thread
# counts, for example:
#
#   julia --project=. --threads=1 scripts/run_parallel_scaling_benchmark.jl
#   julia --project=. --threads=2 scripts/run_parallel_scaling_benchmark.jl
#   julia --project=. --threads=4 scripts/run_parallel_scaling_benchmark.jl
#
# Each run writes one CSV file under output/scaling/.  Combine the generated
# files to summarize speedup across thread counts.

using CSV
using DataFrames
using LinearAlgebra
using Random
using Statistics

BLAS.set_num_threads(1)

include(joinpath(dirname(@__DIR__), "src", "parameter_balancing_core.jl"))
include(joinpath(dirname(@__DIR__), "src", "prediction_utils.jl"))
include(joinpath(dirname(@__DIR__), "src", "experiment_runners.jl"))

const OUTDIR = joinpath(dirname(@__DIR__), "output", "scaling")
mkpath(OUTDIR)

function _parallel_benchmark_problem(; n::Int = 96, d::Int = 20, G::Int = 24, seed::Int = 101)
    rng = MersenneTwister(seed)
    A = randn(rng, n, d) ./ sqrt(d)
    q = randn(rng, d)
    grp = ["source_" * string(1 + mod(i - 1, G)) for i in 1:n]
    b_by_group = Dict("source_$g" => 0.25 * randn(rng) for g in 1:G)
    σ2 = fill(0.35^2, n)
    y = A * q .+ [b_by_group[g] for g in grp] .+ sqrt.(σ2) .* randn(rng, n)
    eb = EBSpec(q0 = zeros(d), C0 = Matrix{Float64}(I, d, d))
    A0 = zeros(d, d)
    b0 = zeros(d)
    return A0, b0, A, y, σ2, grp, eb
end

function _benchmark_spec()
    return FoldFitSpec(
        model = :t,
        hierarchical = true,
        n_outer = 3,
        n_mcmc = 160,
        burnin = 40,
        thin = 2,
        n_keep_min = 50,
        n_pred_draws = 120,
        b_draws_ll = 30,
        posterior_refresh = 1,
        adaptive_stop = false,
        min_outer = 1,
        tau_update = :ig,
        ν_init = 5.0,
        update_nu = false,
        update_alpha = false,
        refine_shape = false,
    )
end

function _elapsed_after_warmup(f)
    GC.gc()
    f()
    GC.gc()
    return @elapsed f()
end

function benchmark_record_loo()
    A0, b0, A, y, σ2, grp, eb = _parallel_benchmark_problem()
    spec = _benchmark_spec()
    f() = leave_one_out_cv(
        MersenneTwister(2024), A0, b0, A, y, σ2, grp, eb, spec;
        cover_levels = [0.8, 0.95],
        method = :collapsed,
        parallel = nothing,
    )
    return _elapsed_after_warmup(f)
end

function benchmark_source_loso()
    A0, b0, A, y, σ2, grp, eb = _parallel_benchmark_problem()
    spec = _benchmark_spec()
    f() = leave_one_source_out_cv(
        MersenneTwister(2025), A0, b0, A, y, σ2, grp, eb, spec;
        cover_levels = [0.8, 0.95],
        method = :collapsed,
        store_fits = false,
        parallel = nothing,
    )
    return _elapsed_after_warmup(f)
end

function benchmark_sweep_recovery()
    spec = FoldFitSpec(
        model = :gaussian,
        hierarchical = false,
        n_outer = 1,
        n_mcmc = 120,
        burnin = 30,
        thin = 2,
        n_keep_min = 40,
        n_pred_draws = 120,
        adaptive_stop = false,
        min_outer = 1,
        update_nu = false,
        update_alpha = false,
        refine_shape = false,
    )
    posterior_file = joinpath(dirname(@__DIR__), "output", "tca", "data", "posterior_tca.jld2")
    f() = run_sweep_recovery(
        posterior_file;
        n_reps = 48,
        seed = 2026,
        data_model = :gaussian,
        fit_model = :gaussian,
        nominal_levels = [0.95],
        spec = spec,
        method = :collapsed,
        parallel = nothing,
    )
    return _elapsed_after_warmup(f)
end

threads = Threads.nthreads()
rows = [
    (experiment = "record_loo_cv", threads = threads, seconds = benchmark_record_loo()),
    (experiment = "source_loso_cv", threads = threads, seconds = benchmark_source_loso()),
    (experiment = "synthetic_recovery", threads = threads, seconds = benchmark_sweep_recovery()),
]

df = DataFrame(rows)
outfile = joinpath(OUTDIR, "parallel_thread_scaling_threads$(threads).csv")
CSV.write(outfile, df)
println(df)
println("Saved: $outfile")
