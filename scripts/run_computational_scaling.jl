# run_computational_scaling.jl
#
# Comprehensive computational scaling benchmarks for the paper.
# Measures wall-clock time of fit_fold_mcem across structural factors
# (basis dimension d, observations n, sources G) and algorithmic factors
# (MCEM iterations, posterior samples per iteration). Results are saved
# under output/scaling/ as CSV files and PDFs.
#
# Run after installing all project dependencies via `Pkg.instantiate()`.
#
# STRUCTURAL FACTORS (biologically relevant):
#   Exp 1. Basis dimension d: pathway size
#   Exp 2. Number of observations n: data density
#   Exp 3. Number of sources G: database richness
#   Exp 4. RE overhead ratio: direct Fixed vs RE comparison
#
# ALGORITHMIC FACTORS (on real pathway data):
#   Exp 5. Number of outer MCEM iterations (both pathways)
#   Exp 6. Number of posterior samples per iteration (both pathways)
#
# Timing: single fit_fold_mcem call, JIT-warmed, median over n_reps.  Within one
# experiment all fits (every problem size and every model) are taken in turn, round
# after round, so that a burst of other work on the machine falls on a few fits of
# every curve and not on all fits of one point (time_fits_in_turn).  Linear algebra
# runs on one thread, as in the other timing drivers.  Until 10 October 2026 the
# fits of one point were timed in a row with the default number of threads, and a
# run of that day showed slow columns in the figures.

import Pkg
Pkg.activate(".")
Pkg.instantiate()

# Standard libraries
using Dates
using LinearAlgebra
using Printf
using Random
using Statistics

# External packages
using ArviZ
using BlockArrays
using CairoMakie
using Colors
using CSV
using DataFrames
using Distributions
using JLD2
using JSON
using KernelDensity
using LaTeXStrings
using MCMCChains
using ProgressMeter
using SpecialFunctions

# Local source files
include(joinpath(dirname(@__DIR__), "src", "metabolic_reconstruction.jl"))
include(joinpath(dirname(@__DIR__), "src", "parameter_balancing.jl"))
include(joinpath(dirname(@__DIR__), "src", "makie_functions.jl"))
include(joinpath(dirname(@__DIR__), "src", "parameter_balancing_core.jl"))
include(joinpath(dirname(@__DIR__), "src", "prediction_utils.jl"))

# Experiment and plotting utilities
include(joinpath(dirname(@__DIR__), "src", "experiment_runners.jl"))
include(joinpath(dirname(@__DIR__), "src", "plot_utils.jl"))
include(joinpath(dirname(@__DIR__), "src", "plot_paper.jl"))

const SEED = 42
const METHOD = _default_sampler_method()

BLAS.set_num_threads(1)
const SCALING_N_REPS = parse(Int, get(ENV, "PB_SCALING_N_REPS", "10"))

function _method_suffix(method::Symbol)
    return method == :collapsed ? "" : "_$(String(method))"
end

function _method_filename(stem::AbstractString, ext::AbstractString, method::Symbol)
    return string(stem, _method_suffix(method), ext)
end

function _method_title_prefix(method::Symbol)
    return method == :collapsed ? "" : "Structured VI: "
end

# BW-friendly style system
# Fixed: open markers and dashed lines
# RE: filled markers and solid lines
# Each family: unique color and marker shape

struct BenchStyle
    color       :: Any
    marker      :: Symbol
    markersize  :: Int
    strokewidth :: Float64
    strokecolor :: Any
    fill        :: Any
    linewidth   :: Float64
    linestyle   :: Any
end

const BENCH_STYLES = Dict{String, BenchStyle}(
    "Gaussian Fixed"  => BenchStyle(:steelblue3,  :utriangle, 10, 1.8, :steelblue3,  :white,        2.4, :dash),
    "Gaussian RE"     => BenchStyle(:steelblue3,  :circle,    10, 0.7, :black,        :steelblue3,   2.4, :solid),
    "Student-t Fixed" => BenchStyle(:forestgreen,  :rect,      10, 1.8, :forestgreen,  :white,        2.4, :dash),
    "Student-t RE"    => BenchStyle(:forestgreen,  :diamond,   11, 0.7, :black,        :forestgreen,  2.4, :solid),
    "Skew-N Fixed"    => BenchStyle(:firebrick,    :dtriangle, 10, 1.8, :firebrick,    :white,        2.4, :dash),
    "Skew-N RE"       => BenchStyle(:firebrick,    :star5,     11, 0.7, :black,        :firebrick,    2.4, :solid),
)

const BENCH_LEGEND_LABELS = Dict(
    "Gaussian Fixed"  => "Gaussian FE",
    "Gaussian RE"     => "Gaussian RE",
    "Student-t Fixed" => "Student-t FE",
    "Student-t RE"    => "Student-t RE",
    "Skew-N Fixed"    => "Skew-normal FE",
    "Skew-N RE"       => "Skew-normal RE",
)

const MODELS_ORDERED = [
    "Gaussian Fixed", "Gaussian RE",
    "Student-t Fixed", "Student-t RE",
    "Skew-N Fixed", "Skew-N RE"
]

const MODEL_CONFIGS = [
    (:gaussian, false, "Gaussian Fixed"),
    (:gaussian, true,  "Gaussian RE"),
    (:t,        false, "Student-t Fixed"),
    (:t,        true,  "Student-t RE"),
    (:skew,     false, "Skew-N Fixed"),
    (:skew,     true,  "Skew-N RE"),
]

"""
    generate_benchmark_data(rng, d, n, G; tau2, sigma2_range, n_thermo)
        -> (Arows, y, sigma2, grp, q_prior, C_prior, A0, b0)

Generate a random benchmark dataset with `d` latent parameters, `n` observations,
and `G` source groups.  Observation design rows have 1–3 nonzero entries.
If `n_thermo > 0`, appends exact identity-row thermodynamic constraints as `(A0, b0)`.
"""
function generate_benchmark_data(
    rng::AbstractRNG, d::Int, n::Int, G::Int;
    τ²::Float64 = 1.0,
    σ²_range::Tuple{Float64,Float64} = (0.1, 2.0),
    n_thermo::Int = 0
)
    q_prior = zeros(Float64, d)
    C_prior = Matrix{Float64}(5.0 * I(d))
    q_true  = randn(rng, d) .* 0.5

    Arows = zeros(Float64, n, d)
    for i in 1:n
        k = rand(rng, 1:min(3, d))
        cols = sort(randperm(rng, d)[1:k])
        for c in cols; Arows[i, c] = randn(rng); end
    end

    grp = ["src_$(mod1(i, G))" for i in 1:n]
    η   = Dict(g => randn(rng) * sqrt(τ²) for g in unique(grp))
    σ2  = σ²_range[1] .+ (σ²_range[2] - σ²_range[1]) .* rand(rng, n)
    y   = [dot(view(Arows, i, :), q_true) + η[grp[i]] + randn(rng)*sqrt(σ2[i]) for i in 1:n]

    n_th = min(n_thermo, d)
    if n_th > 0
        Q0 = zeros(Float64, n_th, d); for j in 1:n_th; Q0[j,j] = 1.0; end
        y0 = q_true[1:n_th] + randn(rng, n_th) .* 0.1
        Σ0 = Symmetric(0.01 * Matrix{Float64}(I(n_th)))
        F  = cholesky(Σ0)
        A0 = Matrix{Float64}(Q0' * (F \ Q0))
        b0 = Vector{Float64}(Q0' * (F \ y0))
    else
        A0 = zeros(Float64, d, d); b0 = zeros(Float64, d)
    end

    return Arows, y, σ2, grp, q_prior, C_prior, A0, b0
end

"""
    make_bench_spec(model, hierarchical; n_outer, n_mcmc, burnin) -> FoldFitSpec

Build a `FoldFitSpec` for timing benchmarks: fixed outer/MCMC counts, no adaptive
stopping, shape updates disabled, and a minimal prediction budget.
"""
function make_bench_spec(model::Symbol, hierarchical::Bool;
    n_outer::Int = 5, n_mcmc::Int = 100, burnin::Int = 25
)
    FoldFitSpec(
        model = model, hierarchical = hierarchical,
        n_outer = n_outer, n_mcmc = n_mcmc, burnin = burnin,
        thin = 1, n_keep_min = max(n_mcmc - burnin, 50),
        n_pred_draws = 100, b_draws_ll = 20,
        posterior_refresh = 1,
        adaptive_stop = false,
        rtol_outer = 1e-10, min_outer = n_outer,
        tau_update = :ig, tau_pc_rate = 0.0,
        ν_init = 8.0, α_init = 0.0,
        update_nu = false, update_alpha = false,
        refine_shape = false, shape_opt_tol = 1e-3,
        nu_penalty = 0.0, nu_target = 8.0, alpha_penalty = 0.0
    )
end

"""
    time_single_fit(Arows, y, sigma2, grp, eb, A0, b0, spec; seed, n_reps) -> Float64

Time one `fit_fold_mcem` call: one JIT warmup run followed by `n_reps` timed runs.
Returns the median elapsed time in seconds.
"""
function time_single_fit(
    Arows::Matrix{Float64}, y::Vector{Float64},
    σ2::Vector{Float64}, grp::Vector{String},
    eb::EBSpec, A0::Matrix{Float64}, b0::Vector{Float64},
    spec::FoldFitSpec;
    seed::Int = SEED, n_reps::Int = 3, method::Symbol = METHOD
)
    # JIT warmup (excluded from timing)
    rng_w = MersenneTwister(seed)
    hs_w  = init_hyper(eb; λ0=spec.λ_init, τ²0=spec.τ²_init, π0=spec.π_init)
    fit_fold_mcem(rng_w, A0, b0, Arows, y, σ2, grp, eb, hs_w, spec; method=method)

    times = Float64[]
    for rep in 1:n_reps
        rng_r = MersenneTwister(seed + rep)
        hs_r  = init_hyper(eb; λ0=spec.λ_init, τ²0=spec.τ²_init, π0=spec.π_init)
        GC.gc()
        t = @elapsed fit_fold_mcem(rng_r, A0, b0, Arows, y, σ2, grp, eb, hs_r, spec; method=method)
        push!(times, t)
    end
    return median(times)
end

"""
    time_fits_in_turn(jobs; seed, n_reps, method) -> Vector{Float64}

Time a set of fits against each other.  `jobs` is a vector of named tuples
`(Arows, y, σ2, grp, eb, A0, b0, spec)`.  Every job is fitted once to warm up.  Then come
`n_reps` rounds, and in each round every job is fitted once, in turn.  Returns the median
seconds of each job over the rounds, with the seeds of `time_single_fit`.
"""
function time_fits_in_turn(jobs; seed::Int = SEED, n_reps::Int = 3, method::Symbol = METHOD)
    fit_once(job, s) = begin
        sp = job.spec
        hs = init_hyper(job.eb; λ0=sp.λ_init, τ²0=sp.τ²_init, π0=sp.π_init)
        fit_fold_mcem(MersenneTwister(s), job.A0, job.b0, job.Arows, job.y, job.σ2, job.grp,
                      job.eb, hs, sp; method=method)
    end
    for job in jobs
        fit_once(job, seed)                         # JIT warmup (excluded from timing)
    end
    times = [Float64[] for _ in jobs]
    for rep in 1:n_reps, (j, job) in enumerate(jobs)
        GC.gc()
        push!(times[j], @elapsed fit_once(job, seed + rep))
    end
    return [median(t) for t in times]
end

_job(Ar, y, s2, gr, eb, A0, b0, sp) = (Arows=Ar, y=y, σ2=s2, grp=gr, eb=eb, A0=A0, b0=b0, spec=sp)

# Plotting
function _plot_scaling(df::DataFrame, xcol::Symbol, xlabel::String, title::String;
    outpath::Union{String,Nothing} = nothing, xscale = identity
)
    fig = Figure(size = (600, 420), fontsize = 18)
    ax  = Axis(fig[1,1],
        xlabel=xlabel,
        ylabel="Wall-clock time (s)",
        title=title,
        xscale=xscale,
        xlabelsize=20,
        ylabelsize=20,
        titlesize=19,
        xticklabelsize=16,
        yticklabelsize=16)
    leg_e = Any[]; leg_l = String[]
    for label in MODELS_ORDERED
        sub = filter(r -> r.model == label, df)
        nrow(sub) == 0 && continue
        sty = BENCH_STYLES[label]
        lines!(ax, sub[!,xcol], sub.time_s,
            color=sty.color, linestyle=sty.linestyle, linewidth=sty.linewidth)
        scatter!(ax, sub[!,xcol], sub.time_s,
            color=sty.fill, marker=sty.marker, markersize=sty.markersize,
            strokewidth=sty.strokewidth, strokecolor=sty.strokecolor)
        push!(leg_e, MarkerElement(marker=sty.marker, color=sty.fill,
            markersize=sty.markersize, strokewidth=sty.strokewidth,
            strokecolor=sty.strokecolor))
        push!(leg_l, BENCH_LEGEND_LABELS[label])
    end
    Legend(fig[1,2], leg_e, leg_l, "Model",
        framevisible=true,
        labelsize=16,
        titlesize=17,
        tellwidth=true,
        padding=(8,8,6,6),
        rowgap=5)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

function _plot_re_overhead(df::DataFrame; outpath::Union{String,Nothing} = nothing)
    fig = Figure(size = (600, 420), fontsize = 18)
    ax  = Axis(fig[1,1],
        xlabel="Number of basis quantities D",
        ylabel="Random-effects / fixed-effects time ratio",
        title="Random-effects overhead",
        xlabelsize=20,
        ylabelsize=20,
        titlesize=19,
        xticklabelsize=16,
        yticklabelsize=16)
    hlines!(ax, [1.0], linestyle=:dash, color=:gray60, linewidth=1.4)
    fam_sty = Dict(
        "Gaussian"  => (color=:steelblue3,  marker=:circle),
        "Student-t" => (color=:forestgreen,  marker=:diamond),
        "Skew-N"    => (color=:firebrick,    marker=:star5),
    )
    leg_e = Any[]; leg_l = String[]
    for family in ["Gaussian", "Student-t", "Skew-N"]
        sub = filter(r -> r.family == family, df)
        nrow(sub) == 0 && continue
        sty = fam_sty[family]
        lines!(ax, sub.d, sub.ratio, color=sty.color, linewidth=2.4)
        scatter!(ax, sub.d, sub.ratio, color=sty.color, marker=sty.marker,
            markersize=11, strokewidth=0.7, strokecolor=:black)
        push!(leg_e, MarkerElement(marker=sty.marker, color=sty.color,
            markersize=11, strokewidth=0.7, strokecolor=:black))
        push!(leg_l, family == "Skew-N" ? "Skew-normal" : family)
    end
    Legend(fig[1,2], leg_e, leg_l, "Likelihood",
        framevisible=true,
        labelsize=16,
        titlesize=17,
        tellwidth=true,
        padding=(8,8,6,6),
        rowgap=5)
    !isnothing(outpath) && save(outpath, fig)
    return fig
end

function _method_plot_name(method::Symbol)
    return method == :collapsed ? "Collapsed MCEM" : "Structured VI"
end

function regenerate_scaling_plots_from_csv(;
    outdir::String = "output/scaling",
    method::Symbol = METHOD
)
    pd = joinpath(outdir, "plots")
    mkpath(pd)

    read_benchmark(stem) =
        CSV.read(joinpath(outdir, _method_filename(stem, ".csv", method)), DataFrame)

    method_name = _method_plot_name(method)

    df_d = read_benchmark("scaling_d")
    _plot_scaling(df_d, :d, "Number of basis quantities D",
        "$(method_name): basis dimension";
        outpath=joinpath(pd, _method_filename("scaling_d", ".pdf", method)))

    df_n = read_benchmark("scaling_n")
    _plot_scaling(df_n, :n, "Number of observations N",
        "$(method_name): observation count";
        outpath=joinpath(pd, _method_filename("scaling_n", ".pdf", method)))

    df_G = read_benchmark("scaling_G")
    _plot_scaling(df_G, :G, "Number of source groups G",
        "$(method_name): source count";
        outpath=joinpath(pd, _method_filename("scaling_G", ".pdf", method)))

    re_path = joinpath(outdir, _method_filename("re_overhead", ".csv", method))
    if isfile(re_path)
        df_re = CSV.read(re_path, DataFrame)
        _plot_re_overhead(df_re;
            outpath=joinpath(pd, _method_filename("re_overhead_ratio", ".pdf", method)))
    end

    for pw in [:glycolysis, :tca]
        ps = pw == :glycolysis ? "glycolysis" : "tca"
        pathway_name = pw == :glycolysis ? "glycolysis" : "TCA cycle"

        df_out = read_benchmark("scaling_outer_$(ps)")
        _plot_scaling(df_out, :n_outer, "Number of outer fitting iterations",
            "$(method_name): outer iterations ($(pathway_name))";
            outpath=joinpath(pd, _method_filename("scaling_outer_$(ps)", ".pdf", method)))

        df_samp = read_benchmark("scaling_samples_$(ps)")
        _plot_scaling(df_samp, :n_mcmc, "Number of posterior samples",
            "$(method_name): posterior samples ($(pathway_name))";
            outpath=joinpath(pd, _method_filename("scaling_samples_$(ps)", ".pdf", method)))
    end

    println("Regenerated $(method_name) scaling plots from existing CSV files in $(pd)")
    return nothing
end

# STRUCTURAL EXPERIMENTS (synthetic data, varying d / n / G)

function experiment_scaling_d(;
    d_values=[20,40,60,80,100,120,150], n_per_d=500, G=30,
    n_outer=5, n_mcmc=100, n_reps=3, outdir="output/scaling",
    method::Symbol = METHOD
)
    mkpath(outdir)
    println("\n=== Exp 1: Scaling with basis dimension d ===")
    jobs = NamedTuple[]; tags = NamedTuple[]
    for d in d_values
        n = max(n_per_d, 5*d)
        rng = MersenneTwister(SEED)
        Ar, y, s2, gr, qp, Cp, A0, b0 =
            generate_benchmark_data(rng, d, n, G; τ²=1.0, n_thermo=min(div(d,3),30))
        eb = EBSpec(q0=qp, C0=Cp)
        println("  d=$d, n=$n")
        for (m, re, lab) in MODEL_CONFIGS
            sp = make_bench_spec(m, re; n_outer=n_outer, n_mcmc=n_mcmc)
            push!(jobs, _job(Ar, y, s2, gr, eb, A0, b0, sp))
            push!(tags, (d=d, n=n, G=G, model=lab, is_re=re))
        end
    end
    ts = time_fits_in_turn(jobs; n_reps=n_reps, method=method)
    rows = [merge(k, (time_s=t, method=String(method))) for (k, t) in zip(tags, ts)]
    df = DataFrame(rows)
    CSV.write(joinpath(outdir, _method_filename("scaling_d", ".csv", method)), df)
    df
end

function experiment_scaling_n(;
    d=60, n_values=[100,250,500,1000,2000,4000], G=30,
    n_outer=5, n_mcmc=100, n_reps=3, outdir="output/scaling",
    method::Symbol = METHOD
)
    mkpath(outdir)
    println("\n=== Exp 2: Scaling with observation count n ===")
    jobs = NamedTuple[]; tags = NamedTuple[]
    for n in n_values
        rng = MersenneTwister(SEED)
        Ar, y, s2, gr, qp, Cp, A0, b0 =
            generate_benchmark_data(rng, d, n, G; τ²=1.0, n_thermo=20)
        eb = EBSpec(q0=qp, C0=Cp)
        println("  n=$n")
        for (m, re, lab) in MODEL_CONFIGS
            sp = make_bench_spec(m, re; n_outer=n_outer, n_mcmc=n_mcmc)
            push!(jobs, _job(Ar, y, s2, gr, eb, A0, b0, sp))
            push!(tags, (d=d, n=n, G=G, model=lab, is_re=re))
        end
    end
    ts = time_fits_in_turn(jobs; n_reps=n_reps, method=method)
    rows = [merge(k, (time_s=t, method=String(method))) for (k, t) in zip(tags, ts)]
    df = DataFrame(rows)
    CSV.write(joinpath(outdir, _method_filename("scaling_n", ".csv", method)), df)
    df
end

function experiment_scaling_G(;
    d=60, n=1000, G_values=[5,10,20,50,100,200],
    n_outer=5, n_mcmc=100, n_reps=3, outdir="output/scaling",
    method::Symbol = METHOD
)
    mkpath(outdir)
    println("\n=== Exp 3: Scaling with source count G ===")
    jobs = NamedTuple[]; tags = NamedTuple[]
    for G in G_values
        rng = MersenneTwister(SEED)
        Ar, y, s2, gr, qp, Cp, A0, b0 =
            generate_benchmark_data(rng, d, n, G; τ²=1.0, n_thermo=20)
        eb = EBSpec(q0=qp, C0=Cp)
        println("  G=$G")
        for (m, re, lab) in MODEL_CONFIGS
            sp = make_bench_spec(m, re; n_outer=n_outer, n_mcmc=n_mcmc)
            push!(jobs, _job(Ar, y, s2, gr, eb, A0, b0, sp))
            push!(tags, (d=d, n=n, G=G, model=lab, is_re=re))
        end
    end
    ts = time_fits_in_turn(jobs; n_reps=n_reps, method=method)
    rows = [merge(k, (time_s=t, method=String(method))) for (k, t) in zip(tags, ts)]
    df = DataFrame(rows)
    CSV.write(joinpath(outdir, _method_filename("scaling_G", ".csv", method)), df)
    df
end

function experiment_re_overhead(;
    d_values=[20,40,60,80,100,120], n_per_d=500, G=30,
    n_outer=5, n_mcmc=100, n_reps=3, outdir="output/scaling",
    method::Symbol = METHOD
)
    mkpath(outdir)
    println("\n=== Exp 4: RE overhead ratio ===")
    jobs = NamedTuple[]; tags = NamedTuple[]
    for d in d_values
        n = max(n_per_d, 5*d)
        rng = MersenneTwister(SEED)
        Ar, y, s2, gr, qp, Cp, A0, b0 =
            generate_benchmark_data(rng, d, n, G; τ²=1.0, n_thermo=min(div(d,3),30))
        eb = EBSpec(q0=qp, C0=Cp)
        for ms in [:gaussian, :t, :skew]
            sf = make_bench_spec(ms, false; n_outer=n_outer, n_mcmc=n_mcmc)
            sr = make_bench_spec(ms, true;  n_outer=n_outer, n_mcmc=n_mcmc)
            fam = ms==:gaussian ? "Gaussian" : ms==:t ? "Student-t" : "Skew-N"
            push!(jobs, _job(Ar, y, s2, gr, eb, A0, b0, sf))     # fixed effects, then
            push!(jobs, _job(Ar, y, s2, gr, eb, A0, b0, sr))     # random effects
            push!(tags, (d=d, n=n, G=G, family=fam))
        end
    end
    ts = time_fits_in_turn(jobs; n_reps=n_reps, method=method)
    rows = NamedTuple[]
    for (i, k) in enumerate(tags)
        tf, tr = ts[2i - 1], ts[2i]
        push!(rows, merge(k, (time_fix=tf, time_re=tr, ratio=tr/tf, method=String(method))))
    end
    df = DataFrame(rows)
    CSV.write(joinpath(outdir, _method_filename("re_overhead", ".csv", method)), df)
    df
end

# ALGORITHMIC EXPERIMENTS (real pathway data)

function load_pathway_data(pathway::Symbol)
    mdl = load_yeast_reconstruction()
    dr  = reaction_data_dict(mdl)

    if pathway == :glycolysis
        rxns = generate_reactions(
            ["r_0534","r_0467","r_0886","r_0450","r_1054",
             "r_0486","r_0892","r_0893","r_0366","r_0962"],
            dr, protons=false, proton_name="s_0794")
        h2o = ["s_0803"]
        eq_path  = "data/equilibrator/equilibrator_glycolysis.json"
        kc_path  = "data/brenda/kcat_values_glycolysis.csv"
        km_path  = "data/brenda/KM_values_glycolysis.csv"
        br_path  = "output/glycolysis/data/brenda_inputs_glycolysis.jld2"
        pw_str   = "Glycolysis"
        bigg_ids = Dict("s_0434"=>"atp","s_0563"=>"glc__D","s_0394"=>"adp",
            "s_0568"=>"g6p","s_0557"=>"f6p","s_0555"=>"fdp","s_0629"=>"dhap",
            "s_0764"=>"g3p","s_1198"=>"nad","s_1322"=>"pi","s_0075"=>"13dpg",
            "s_1203"=>"nadh","s_0260"=>"3pg","s_0188"=>"2pg","s_0803"=>"h2o",
            "s_1360"=>"pep","s_1399"=>"pyr")
    else
        rxns = generate_reactions(
            ["r_0300","r_0302","r_0280","r_0658","r_0832","r_0831",
             "r_0505","r_1022","r_1021","r_0451","r_0713"],
            dr, protons=false, proton_name="s_0799")
        h2o = ["s_0807"]
        eq_path  = "data/equilibrator/equilibrator_tca_cycle.json"
        kc_path  = "data/brenda/kcat_values_tca_cycle.csv"
        km_path  = "data/brenda/KM_values_tca_cycle.csv"
        br_path  = "output/tca/data/brenda_inputs_tca.jld2"
        pw_str   = "TCA"
        bigg_ids = Dict("s_0376"=>"accoa","s_0807"=>"h2o","s_1273"=>"oaa",
            "s_0524"=>"cit","s_0532"=>"coa","s_0517"=>"acon_C","s_0941"=>"icit",
            "s_1200"=>"nad","s_0182"=>"akg","s_0460"=>"co2","s_1205"=>"nadh",
            "s_1097"=>"lpam","s_1411"=>"sdhlam","s_0627"=>"dhlam",
            "s_1464"=>"succoa","s_0397"=>"adp","s_1326"=>"pi","s_0437"=>"atp",
            "s_1460"=>"succ","s_1537"=>"q6","s_0727"=>"fum","s_1535"=>"q6h2",
            "s_0068"=>"mal__L")
    end

    (eq_names, ΔGf_mu, ΔGf_cov) = load_equilibrator_data(eq_path)
    (kcat_med, KM_med)           = load_BRENDA_data(kc_path, km_path)
    global Q = observation_matrix(rxns, h2o_strings=h2o)
    (species, _) = rn_entities(rxns)
    (μ0_map, _, _, _) = observation_matrix_index_maps(rxns)

    met_obs = [bigg_ids[s] for s in species]
    μ0eq = Dict(bigg_ids[k] => v for (k,v) in μ0_map)
    (μ0_mean, μ0_cov, Q_μ0) = chemical_potential_obs(ΔGf_mu, ΔGf_cov, eq_names, met_obs, μ0eq)

    Σ0 = Symmetric(Matrix{Float64}(μ0_cov)); F0 = cholesky(Σ0)
    A0 = Matrix{Float64}(Q_μ0' * (F0 \ Matrix{Float64}(Q_μ0)))
    b0 = Vector{Float64}(Q_μ0' * (F0 \ Vector{Float64}(μ0_mean)))

    (q_prior, C_prior) = thermodynamic_prior(rxns, h2o)
    d  = size(Q, 2)
    eb = EBSpec(q0=q_prior, C0=C_prior)

    brenda = load(br_path)
    Arows = Matrix{Float64}(brenda["Arows"])
    y     = Vector{Float64}(brenda["y"])
    σ2    = Vector{Float64}(brenda["σ2"])
    grp   = Vector{String}(brenda["grp"])

    return Arows, y, σ2, grp, eb, A0, b0, d, pw_str
end

function experiment_scaling_outer(;
    pathway::Symbol = :glycolysis,
    outer_values::Vector{Int} = [3, 5, 10, 15],
    n_mcmc::Int = 150, n_reps::Int = 3,
    outdir::String = "output/scaling",
    method::Symbol = METHOD
)
    mkpath(outdir)
    Ar, y, s2, gr, eb, A0, b0, d, pw = load_pathway_data(pathway)
    n = length(y); G = length(unique(gr))
    println("\n=== Exp 5: Outer iterations ($pw, d=$d, n=$n, G=$G) ===")

    jobs = NamedTuple[]; tags = NamedTuple[]
    for nout in outer_values
        println("  n_outer=$nout")
        for (m, re, lab) in MODEL_CONFIGS
            sp = make_bench_spec(m, re; n_outer=nout, n_mcmc=n_mcmc)
            push!(jobs, _job(Ar, y, s2, gr, eb, A0, b0, sp))
            push!(tags, (n_outer=nout, n_mcmc=n_mcmc, model=lab, is_re=re))
        end
    end
    ts = time_fits_in_turn(jobs; n_reps=n_reps, method=method)
    rows = [merge(k, (time_s=t, pathway=pw, d=d, method=String(method))) for (k, t) in zip(tags, ts)]
    df = DataFrame(rows)
    CSV.write(joinpath(outdir, _method_filename("scaling_outer_$(lowercase(pw))", ".csv", method)), df)
    return df
end

function experiment_scaling_samples(;
    pathway::Symbol = :glycolysis,
    sample_values::Vector{Int} = [25, 50, 100, 150],
    n_outer::Int = 5, n_reps::Int = 3,
    outdir::String = "output/scaling",
    method::Symbol = METHOD
)
    mkpath(outdir)
    Ar, y, s2, gr, eb, A0, b0, d, pw = load_pathway_data(pathway)
    n = length(y); G = length(unique(gr))
    println("\n=== Exp 6: MCMC samples ($pw, d=$d, n=$n, G=$G) ===")

    jobs = NamedTuple[]; tags = NamedTuple[]
    for nmc in sample_values
        burn = max(div(nmc, 4), 10)
        println("  n_mcmc=$nmc (burnin=$burn)")
        for (m, re, lab) in MODEL_CONFIGS
            sp = make_bench_spec(m, re; n_outer=n_outer, n_mcmc=nmc, burnin=burn)
            push!(jobs, _job(Ar, y, s2, gr, eb, A0, b0, sp))
            push!(tags, (n_outer=n_outer, n_mcmc=nmc, model=lab, is_re=re))
        end
    end
    ts = time_fits_in_turn(jobs; n_reps=n_reps, method=method)
    rows = [merge(k, (time_s=t, pathway=pw, d=d, method=String(method))) for (k, t) in zip(tags, ts)]
    df = DataFrame(rows)
    CSV.write(joinpath(outdir, _method_filename("scaling_samples_$(lowercase(pw))", ".csv", method)), df)
    return df
end

# Main runner
function run_all_scaling_benchmarks(;
    outdir::String = "output/scaling", n_reps::Int = 3,
    method::Symbol = METHOD
)
    mkpath(outdir)
    pd = joinpath(outdir, "plots"); mkpath(pd)

    println("="^70)
    println("COMPUTATIONAL SCALING BENCHMARKS")
    println("Inference method: $(method)")
    println("="^70)

    # --- Structural ---
    df_d = experiment_scaling_d(; n_reps=n_reps, outdir=outdir, method=method)
    _plot_scaling(df_d, :d, "Number of basis quantities D",
        "$(_method_plot_name(method)): basis dimension";
        outpath=joinpath(pd, _method_filename("scaling_d", ".pdf", method)))

    df_n = experiment_scaling_n(; n_reps=n_reps, outdir=outdir, method=method)
    _plot_scaling(df_n, :n, "Number of observations N",
        "$(_method_plot_name(method)): observation count";
        outpath=joinpath(pd, _method_filename("scaling_n", ".pdf", method)))

    df_G = experiment_scaling_G(; n_reps=n_reps, outdir=outdir, method=method)
    _plot_scaling(df_G, :G, "Number of source groups G",
        "$(_method_plot_name(method)): source count";
        outpath=joinpath(pd, _method_filename("scaling_G", ".pdf", method)))

    df_re = experiment_re_overhead(; n_reps=n_reps, outdir=outdir, method=method)
    _plot_re_overhead(df_re; outpath=joinpath(pd, _method_filename("re_overhead_ratio", ".pdf", method)))

    # --- Algorithmic (both pathways) ---
    for pw in [:glycolysis, :tca]
        ps = pw == :glycolysis ? "glycolysis" : "tca"

        df_out = experiment_scaling_outer(;
            pathway=pw, outer_values=[3,5,10,15],
            n_mcmc=150, n_reps=n_reps, outdir=outdir, method=method)
        pathway_name = pw == :glycolysis ? "glycolysis" : "TCA cycle"
        _plot_scaling(df_out, :n_outer, "Number of outer fitting iterations",
            "$(_method_plot_name(method)): outer iterations ($(pathway_name))";
            outpath=joinpath(pd, _method_filename("scaling_outer_$(ps)", ".pdf", method)))

        df_samp = experiment_scaling_samples(;
            pathway=pw, sample_values=[25,50,100,150],
            n_outer=5, n_reps=n_reps, outdir=outdir, method=method)
        _plot_scaling(df_samp, :n_mcmc, "Number of posterior samples",
            "$(_method_plot_name(method)): posterior samples ($(pathway_name))";
            outpath=joinpath(pd, _method_filename("scaling_samples_$(ps)", ".pdf", method)))
    end

    # --- Summary ---
    println("\n" * "="^70)
    println("ALL SCALING BENCHMARKS COMPLETE")
    println("Results: $outdir"); println("Plots:   $pd")
    println("="^70)

    println("\n--- RE Overhead Summary ---")

    return (d=df_d, n=df_n, G=df_G, re=df_re)
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = run_all_scaling_benchmarks(n_reps = SCALING_N_REPS, method = METHOD)
end
