# run_revision_engine_timing.jl
#
# Wall-clock time of one fit under the two inference engines, the collapsed sampler and
# the structured variational option, on the four curated pathways.
#
# Question addressed:
#   What does a fit cost on the data of the paper, and is the variational option enough
#   faster to earn its place in the text?  The driver times all four pathways, without
#   source effects (the Gaussian fixed-effects model, sampler only), with organism
#   effects and with the organism-by-type effects of the pre-specified default, at two
#   budgets:
#     common   15 EM iterations of 150 draws on every pathway (2250 draws);
#     default  the draws of one fit of the pre-specified default on that pathway.  The
#              default alternates five EM runs, each of three iterations of 150 draws,
#              or of four iterations of 250 draws on the TCA cycle
#              (run_revision_sigmacal.jl), so this is 15 iterations of 150 draws
#              (the common budget again) on glycolysis, the pentose phosphate pathway
#              and the shikimate pathway, and 20 iterations of 250 draws (5000 draws)
#              on the TCA cycle.
#   What is timed is a single run of that many iterations on all records of the
#   pathway, with the fit specification of the default (run_revision_sigmacal.jl): each
#   EM iteration discards its first 40 draws, or 80 where it makes 250, and keeps the
#   rest.  The restarts of the five runs and the four updates of the record scale
#   between them are not timed.  The fixed-effects model is run through the same
#   iterations and draws as the others, so its time is not that of the single linear
#   solve of standard parameter balancing.
#
#   The two engines are compared with the shape held fixed (8 degrees of freedom for
#   the Student-t likelihood, zero skewness for the skew-normal one), so that both do
#   the same work.  The Student-t and skew-normal fits with estimated shape that the
#   paper reports update the shape at every EM iteration (cell_spec in
#   run_revision_realdata.jl).  Both engines are timed once more with that update, with
#   organism-by-type effects (rows with shape_updated = true).  Those rows keep the
#   budget of the default and do not stop early, whereas the reported fits make one run
#   of four iterations, or five on the TCA cycle.  The Student-t fit starts from the
#   moment estimate of the degrees of freedom on all records, as the reported fits do on
#   a fold.  The skew-normal fit starts from zero skewness and not from its moment
#   estimate: from the moment estimate its shape runs to the bound of the grid on these
#   data and its draws stop being finite, which is the failure that the paper reports,
#   and a failing fit has no cost worth printing.  From zero the update does its full
#   work at every iteration (and the estimate stays at zero, the fixed point of the
#   update).  The shape that each model starts from and the shape on which its warm-up
#   fit ends are stored.
#
# Timing.  A timed fit is a single call to fit_fold_mcem.  Every model of one pathway
# and budget is first fitted once without being timed.  Then come rounds, and in each
# round every model is fitted once, in turn.  There are at least PB_TIMING_REPS rounds
# (default 10), and more only if those rounds, the garbage collection before every fit
# included, have lasted less than PB_TIMING_BLOCK_SECONDS (default 60).  In both stored
# runs every block made ten rounds.  The time of a model is the median of its fits over
# the rounds.
# Other work on this machine comes in bursts of seconds to a minute.  With the models
# taken in turn a burst falls on a few of the fits of every model, which the median
# ignores, and not on all fits of one model, as it did when each model was timed ten
# times in a row.  The fastest fit of each model is kept too (columns ending in _min):
# other work can only add time, so a median far above its minimum was disturbed.  A
# median close to its minimum does not exclude that the whole block ran slowly, which
# only a second run can show.  Linear algebra runs on one thread, as in
# run_revision_genomescale.jl, so every time is that of one core.  The inputs (records,
# thermodynamic block, prior) are read from the stored input files of each pathway.
#
# Output: output/scaling/engine_timing_four_pathways.csv, one row per pathway, budget,
#   likelihood, effect structure and shape setting, with the seconds of each engine,
#   their ratio and difference, the number of rounds, the shape the fit started from
#   and the shape each engine ended on (degrees of freedom or skewness, NaN for a
#   Gaussian fit) and whether every draw of both warm-up fits was finite.  Nothing else
#   is written.  With PB_TIMING_TAG set the file name ends in that tag, so that a repeat
#   run does not replace the stored file.  A second run with the same settings, made
#   directly after the stored one, is kept as engine_timing_four_pathways_second_run.csv
#   (PB_TIMING_TAG=_second_run).
#
# Run it alone, and send its log to a folder that no service synchronises: a log line
# written into a synchronised folder starts a burst of that service.
#
# Environment controls:
#   PB_REV_PATHWAYS          comma-separated subset (default: all four)
#   PB_TIMING_REPS           smallest number of rounds (default 10)
#   PB_TIMING_BLOCK_SECONDS  smallest duration of the rounds of one block (default 60)
#   PB_TIMING_TAG            suffix of the output file name (default none)
#
# Run: julia --project=. scripts/run_revision_engine_timing.jl > /tmp/revision_engine_timing.log

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

const TIMING_SEED = 42
const TIMING_PATHWAYS = Symbol.(strip.(split(get(ENV, "PB_REV_PATHWAYS",
                                                 "glycolysis,tca,pentose_phosphate,shikimate"), ",")))
const TIMING_REPS = parse(Int, get(ENV, "PB_TIMING_REPS", "10"))
const TIMING_BLOCK_SECONDS = parse(Float64, get(ENV, "PB_TIMING_BLOCK_SECONDS", "60"))
const TIMING_TAG = get(ENV, "PB_TIMING_TAG", "")
const COMMON_BUDGET = (15, 150)

BLAS.set_num_threads(1)

# Draws of one fit of the pre-specified default: five runs of three iterations of 150
# draws, or of four iterations of 250 draws on the TCA cycle.
default_budget(pathway::Symbol) = pathway == :tca ? (20, 250) : (15, 150)

# The fit specification of the pre-specified default (run_revision_sigmacal.jl), with
# the number of EM iterations, the likelihood and the shape as the things that vary.
# An iteration of 250 draws discards 80, one of 150 draws discards 40, as there.  With
# update_shape the shape settings are those of the reported fits with estimated shape
# (cell_spec in run_revision_realdata.jl), without their early stopping.
timing_spec(model::Symbol, hierarchical::Bool, n_outer::Int, n_mcmc::Int;
            nu::Float64 = 8.0, alpha::Float64 = 0.0, update_shape::Bool = false) = FoldFitSpec(
    model = model, hierarchical = hierarchical,
    n_outer = n_outer, n_mcmc = n_mcmc, burnin = n_mcmc == 250 ? 80 : 40, thin = 1,
    n_keep_min = 60, n_pred_draws = 400, b_draws_ll = 50,
    posterior_refresh = 1, adaptive_stop = false, rtol_outer = 1e-3,
    min_outer = 2, tau_update = :ig, tau_pc_rate = 0.0,
    ν_init = model == :t ? max(nu, 2.05) : 50.0,
    α_init = model == :skew ? alpha : 0.0,
    update_nu = update_shape && model == :t,
    update_alpha = update_shape && model == :skew,
    refine_shape = update_shape, shape_opt_tol = 1e-3,
    skew_location = :mean,
    nu_penalty = 0.0, nu_target = model == :t ? max(nu, 2.05) : 50.0, alpha_penalty = 0.0)

# The moment start of the shape, from run_revision_realdata.jl, unchanged.
_sample_skewness(x) = (m = mean(x); s = std(x, corrected = false);
                       s <= 0 ? 0.0 : mean(((x .- m) ./ s) .^ 3))
_sample_exkurtosis(x) = (m = mean(x); s = std(x, corrected = false);
                         s <= 0 ? 0.0 : mean(((x .- m) ./ s) .^ 4) - 3.0)

function alpha_from_skewness(g1::Float64)
    s = sign(g1)
    a = min(abs(g1), 0.985)
    a < 1e-4 && return 0.0
    f(delta) = ((4.0 - pi) / 2.0) * (delta * sqrt(2.0 / pi))^3 /
               (1.0 - (2.0 / pi) * delta^2)^1.5
    lo, hi = 0.0, 0.999999
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        f(mid) < a ? (lo = mid) : (hi = mid)
    end
    delta = clamp(0.5 * (lo + hi), 0.0, 0.999)
    return s * delta / sqrt(1.0 - delta^2)
end

nu_from_kurtosis(k::Float64) = k <= 0.05 ? 50.0 : clamp(4.0 + 6.0 / k, 2.5, 50.0)

function ridge_qhat(Arows::AbstractMatrix, y::Vector{Float64}, sigma2::Vector{Float64},
                    q0::Vector{Float64}, C0::AbstractMatrix)
    C0inv = inv(Symmetric(Matrix{Float64}(C0)))
    P = C0inv + Arows' * (Arows ./ sigma2)
    rhs = Arows' * (y ./ sigma2) + C0inv * q0
    return Symmetric(P) \ rhs
end

# The shape a fit ended on: degrees of freedom, skewness, or NaN for a Gaussian fit.
final_shape(fit, model::Symbol) = model == :t ? Float64(fit.ν) : model == :skew ? Float64(fit.α) : NaN

"""
    timed_rounds(jobs, A0, b0, Arows, y, sigma2, eb) -> (times, warm, rounds)

Fit every job once without timing it, then time one fit of every job in each round.
Returns the seconds of each job, one entry per round, the warm-up fits and the number
of rounds.
"""
function timed_rounds(jobs, A0, b0, Arows, y, sigma2, eb)
    fit_once(job, seed) = begin
        spec = job.spec
        hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
        fit_fold_mcem(MersenneTwister(seed), A0, b0, Arows, y, sigma2, job.grp, eb, hs, spec;
                      method = job.engine)
    end
    warm = [fit_once(job, TIMING_SEED) for job in jobs]          # warm-up, not timed
    times = [Float64[] for _ in jobs]
    started = time()
    rounds = 0
    while rounds < TIMING_REPS || time() - started < TIMING_BLOCK_SECONDS
        rounds += 1
        for (j, job) in enumerate(jobs)
            GC.gc()
            push!(times[j], @elapsed fit_once(job, TIMING_SEED + rounds))
        end
    end
    return times, warm, rounds
end

function time_pathway!(out::DataFrame, pathway::Symbol)
    root = joinpath(dirname(@__DIR__), "output", string(pathway))
    brenda = load(joinpath(root, "data", "brenda_inputs_$(pathway).jld2"))
    post = load(joinpath(root, "data", "posterior_$(pathway).jld2"))
    y = Vector{Float64}(brenda["y"]); sigma2 = Vector{Float64}(brenda["σ2"])
    Arows = Matrix{Float64}(brenda["Arows"]); org = Vector{String}(brenda["grp"])
    kinds = Vector{String}(getproperty(brenda["meta"], :kind))
    org_type = [string(org[i], "|", kinds[i]) for i in eachindex(org)]
    q0 = vec(Array(post["q_prior"])); C0 = Matrix(Array(post["C_prior"]))
    eb = EBSpec(q0 = q0, C0 = C0); d = size(Arows, 2)
    A0, b0 = _prepare_information_pair(Arows, d, Matrix{Float64}(brenda["A0"]),
                                       Vector{Float64}(brenda["b0"]);
                                       deduplicate_thermo_block = true)

    # moment start of the shape on all records, as the reported fits take it on a fold
    r = (y .- Arows * ridge_qhat(Arows, y, sigma2, q0, C0)) ./ sqrt.(sigma2)
    nu0 = nu_from_kurtosis(_sample_exkurtosis(r))
    al0 = alpha_from_skewness(_sample_skewness(r))
    @printf("  %-18s moment estimates on all records: %.2f degrees of freedom, skewness parameter %.2f (the skew-normal fits start from zero)\n",
            string(pathway), nu0, al0)

    budgets = unique([COMMON_BUDGET, default_budget(pathway)])
    for (n_outer, n_mcmc) in budgets
        label = (n_outer, n_mcmc) == COMMON_BUDGET ? "common" : "default"
        is_default = (n_outer, n_mcmc) == default_budget(pathway)
        jobs = NamedTuple[]
        add(likelihood, model, effects, shape_updated, engine, grp, spec) =
            push!(jobs, (likelihood = likelihood, model = model, effects = effects,
                         shape_updated = shape_updated, engine = engine, grp = grp, spec = spec,
                         start = model == :t ? spec.ν_init : model == :skew ? spec.α_init : NaN))
        # the Gaussian fixed-effects model, run through the same iterations and draws
        add("Gaussian", :gaussian, "none", false, :collapsed, org,
            timing_spec(:gaussian, false, n_outer, n_mcmc))
        for (effects, grp) in (("organism", org), ("organism_by_type", org_type))
            for (model, likelihood) in ((:gaussian, "Gaussian"), (:t, "Student-t"), (:skew, "Skew-normal"))
                spec = timing_spec(model, true, n_outer, n_mcmc)
                add(likelihood, model, effects, false, :collapsed, grp, spec)
                add(likelihood, model, effects, false, :structured_vi, grp, spec)
            end
        end
        # both engines once more with the shape updated at every EM iteration, the
        # Student-t fit from its moment start and the skew-normal fit from zero skewness
        for (model, likelihood) in ((:t, "Student-t"), (:skew, "Skew-normal"))
            spec = timing_spec(model, true, n_outer, n_mcmc; nu = nu0, alpha = 0.0, update_shape = true)
            add(likelihood, model, "organism_by_type", true, :collapsed, org_type, spec)
            add(likelihood, model, "organism_by_type", true, :structured_vi, org_type, spec)
        end

        times, warm, rounds = timed_rounds(jobs, A0, b0, Arows, y, sigma2, eb)

        # one row per model, the two engines side by side
        for (j, job) in enumerate(jobs)
            job.engine == :collapsed || continue
            k = findfirst(i -> jobs[i].engine == :structured_vi &&
                               jobs[i].likelihood == job.likelihood &&
                               jobs[i].effects == job.effects &&
                               jobs[i].shape_updated == job.shape_updated, eachindex(jobs))
            t_mc, t_mc_min = median(times[j]), minimum(times[j])
            t_vi, t_vi_min = k === nothing ? (NaN, NaN) : (median(times[k]), minimum(times[k]))
            groups = job.effects == "none" ? 0 : length(unique(job.grp))
            shape_mc = final_shape(warm[j], job.model)
            shape_vi = k === nothing ? NaN : final_shape(warm[k], job.model)
            finite = all(isfinite, warm[j].qsamps) && (k === nothing || all(isfinite, warm[k].qsamps))
            push!(out, (string(pathway), label, is_default, job.likelihood, job.effects,
                        job.shape_updated, length(y), groups, d, n_outer, n_mcmc, job.spec.burnin,
                        rounds, t_mc, t_vi, t_mc / t_vi, t_mc - t_vi, t_mc_min, t_vi_min,
                        job.start, shape_mc, shape_vi, finite))
            @printf("  %-18s %2d x %3d  %-12s %-17s %-13s sampler %6.3f s (fastest %6.3f) | variational %6.3f s (fastest %6.3f) | ratio %5.1f | difference %5.2f s | %d rounds | shape %6.2f %6.2f%s\n",
                    string(pathway), n_outer, n_mcmc, job.likelihood, job.effects,
                    job.shape_updated ? "shape updated" : "shape fixed",
                    t_mc, t_mc_min, t_vi, t_vi_min, t_mc / t_vi, t_mc - t_vi, rounds,
                    shape_mc, shape_vi, finite ? "" : " | NOT FINITE")
        end
        flush(stdout)
    end
    return out
end

println("Fit time of the collapsed sampler against the structured variational option")
println("  started = $(now()), threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads())")
println("  budgets = 15 EM iterations of 150 draws on every pathway, and 20 of 250 on the TCA cycle (its default)")
println("  median over at least $(TIMING_REPS) rounds and $(TIMING_BLOCK_SECONDS) s of rounds per block, the models taken in turn, after one warm-up each")
out = DataFrame(pathway = String[], budget = String[], is_default_budget = Bool[],
                likelihood = String[], effects = String[], shape_updated = Bool[],
                n_records = Int[], n_groups = Int[], d = Int[],
                n_outer = Int[], n_mcmc = Int[], burnin = Int[], rounds = Int[],
                seconds_sampler = Float64[], seconds_variational = Float64[],
                ratio = Float64[], difference_s = Float64[],
                seconds_sampler_min = Float64[], seconds_variational_min = Float64[],
                shape_start = Float64[], shape_sampler = Float64[], shape_variational = Float64[],
                finite = Bool[])
for pw in TIMING_PATHWAYS
    time_pathway!(out, pw)
end
outdir = joinpath(dirname(@__DIR__), "output", "scaling"); mkpath(outdir)
CSV.write(joinpath(outdir, "engine_timing_four_pathways$(TIMING_TAG).csv"), out)
show(stdout, out; allrows = true, allcols = true); println()
println("Done at $(now())")
