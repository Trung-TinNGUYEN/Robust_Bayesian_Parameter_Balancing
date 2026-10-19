# run_all_paper_experiments.jl
#
# Main experiment driver for the paper.
#
# Default execution reconstructs the TCA-cycle and glycolysis observation
# systems; runs the BRENDA leave-one-source-out, optional record-level
# leave-one-out, and synthetic experiments for all six model configurations;
# and regenerates the corresponding result tables and paper figures.
#
# Computational scaling and hypothesis-testing analyses are run separately in:
#
#   run_computational_scaling.jl
#   hypothesis_tests_and_biology.jl
#
# Required inputs:
#
#   data/yeast-GEM.xml
#   data/equilibrator/equilibrator_tca_cycle.json
#   data/equilibrator/equilibrator_glycolysis.json
#   data/brenda/kcat_values_tca_cycle.csv
#   data/brenda/KM_values_tca_cycle.csv
#   data/brenda/kcat_values_glycolysis.csv
#   data/brenda/KM_values_glycolysis.csv
#   output/tca/data/brenda_inputs_tca.jld2
#   output/glycolysis/data/brenda_inputs_glycolysis.jld2
#   output/tca/data/posterior_tca.jld2
#   output/glycolysis/data/posterior_glycolysis.jld2
#
# Outputs are written under:
#
#   output/tca/results/
#   output/tca/plots/brenda/
#   output/tca/plots/brenda_record_loo/
#   output/tca/plots/synthetic/
#   output/glycolysis/results/
#   output/glycolysis/plots/brenda/
#   output/glycolysis/plots/brenda_record_loo/
#   output/glycolysis/plots/synthetic/
#
# Tables, JLD2 files, and figures use fixed filenames and are overwritten when
# the corresponding stage is rerun. Runtime is machine-dependent and dominated
# by MCEM/MCMC LOSO fits. No dry-run or smoke-run flag is implemented.

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

# Configuration

const SEED = 1
Random.seed!(SEED)
const NOMINAL_LEVELS = vcat(0.01, collect(0.05:0.05:0.95), 0.99)

function _env_bool(name::String, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    return raw in ("1", "true", "yes", "y", "on")
end

const RUN_RECORD_LOO = _env_bool("PB_RUN_RECORD_LOO", true)
const PARALLEL_CV_ENABLED = _pb_should_parallel(2, "PB_PARALLEL_CV", nothing)
const PARALLEL_SWEEP_ENABLED = _pb_should_parallel(2, "PB_PARALLEL_SWEEPS", nothing)

println("Julia threads: $(Threads.nthreads())")
println("Parallel cross-validation folds: $(PARALLEL_CV_ENABLED)")
println("Parallel synthetic recovery sweeps: $(PARALLEL_SWEEP_ENABLED)")

# Workflow:
# 1. Define shared model specifications and result exporters.
# 2. Build pathway-specific reaction systems and observation maps.
# 3. Run BRENDA and synthetic LOSO experiments for TCA and glycolysis.
# 4. Regenerate summaries, per-model CSV tables, and paper plots.
# 5. Leave scaling and hypothesis-testing analyses to the dedicated scripts.

"""
    make_spec(model, hierarchical; pathway) -> FoldFitSpec

Build the `FoldFitSpec` for one experiment cell, with chain lengths tuned per
pathway (TCA uses longer chains than glycolysis due to higher parameter count).
"""
function make_spec(model::Symbol, hierarchical::Bool; pathway::Symbol=:tca)
    is_robust = model != :gaussian
    if pathway == :tca
        n_out = is_robust ? 5 : 4; n_mc = 250; burn = 80
    else
        n_out = is_robust ? 4 : 3; n_mc = 150; burn = 40
    end
    return FoldFitSpec(
        model=model, hierarchical=hierarchical,
        n_outer=n_out, n_mcmc=n_mc, burnin=burn,
        thin=2, n_keep_min=50, n_pred_draws=400, b_draws_ll=30,
        posterior_refresh=1, adaptive_stop=true, rtol_outer=5e-4, min_outer=4,
        tau_update=(hierarchical ? :pc : :ig),
        tau_pc_rate=(hierarchical ? 0.005 : 0.0),
        ν_init=50.0,
        α_init=0.0,
        update_nu=false,
        update_alpha=(model == :skew),
        refine_shape=false,
        shape_opt_tol=1e-3,
        nu_penalty=0.0,
        nu_target=50.0,
        alpha_penalty=0.0)
end

"""
    make_spec_synth(model, hierarchical) -> FoldFitSpec

Build the `FoldFitSpec` used by the synthetic LOSO helper. These settings match
the paper experiment configuration used by the default synthetic runs.
"""
function make_spec_synth(model::Symbol, hierarchical::Bool)
    is_robust = model != :gaussian
    return FoldFitSpec(
        model=model, hierarchical=hierarchical,
        n_outer=(is_robust ? 6 : 4), n_mcmc=200, burnin=60,
        thin=2, n_keep_min=50, n_pred_draws=400, b_draws_ll=30,
        posterior_refresh=1, adaptive_stop=true, rtol_outer=5e-4, min_outer=4,
        tau_update=(hierarchical ? :pc : :ig),
        tau_pc_rate=(hierarchical ? 0.005 : 0.0),
        ν_init=5.0, α_init=0.0,
        update_nu=false, update_alpha=(model == :skew),
        refine_shape=is_robust, shape_opt_tol=1e-3,
        nu_penalty=0.1, nu_target=5.0, alpha_penalty=0.0)
end

# Synthetic experiment settings used by `run_synthetic_experiment`.
const TAU2_DATA = 30.0    # moderate source heterogeneity
const NU_DATA = 5.0
const ALPHA_DATA = 4.0
const N_REPS_SYN = 15

# Result summaries and exports

"""
    brenda_summary_table(results; level=0.95) -> DataFrame

Summarize one set of cross-validation results by model, random-effect setting,
predictive score, empirical coverage, final hyperparameters, and fold count.
The helper is also used for synthetic results because the `CVResult` fields are
shared across experiment types.
"""
function brenda_summary_table(results::Dict{Symbol, CVResult}; level::Float64=0.95)
    rows = NamedTuple[]
    for key in sort(collect(keys(results)))
        res = results[key]
        idx_lev = findfirst(x -> isapprox(x, level, atol=1e-5), res.nominal_levels)
        cov95 = idx_lev !== nothing ? mean(res.pred_coverage[:, idx_lev]) : NaN
        crps_c = filter(isfinite, res.crps)
        wis_c  = filter(isfinite, res.wis)
        pred_err = res.pred_error === nothing ? Float64[] :
            filter(isfinite, Vector{Float64}(res.pred_error))
        push!(rows, (
            Model   = string(res.model),
            RE      = res.has_random_effects,
            CRPS    = isempty(crps_c) ? NaN : round(mean(crps_c); digits=4),
            WIS     = isempty(wis_c)  ? NaN : round(mean(wis_c);  digits=4),
            Cov95   = round(cov95; digits=4),
            PredBias = isempty(pred_err) ? NaN : round(mean(pred_err); digits=4),
            PredRMSE = isempty(pred_err) ? NaN : round(sqrt(mean(abs2, pred_err)); digits=4),
            nu      = res.ν === nothing ? missing : round(res.ν; digits=2),
            alpha   = res.α === nothing ? missing : round(res.α; digits=2),
            tau2    = round(res.τ²_final; digits=4),
            lambda  = round(res.λ_final; digits=4),
            n_folds = res.n_folds))
    end
    df = DataFrame(rows); sort!(df, [:Model, :RE]); return df
end

"""
    _save_results_csv(results, results_dir) -> nothing

Write per-model coverage and interval-width matrices to
`joinpath(results_dir, "per_model")`. Existing files with the same names are
overwritten by `CSV.write`.
"""
function _save_results_csv(results::Dict{Symbol, CVResult}, results_dir::String; subdir::String = "per_model")
    per_dir = joinpath(results_dir, subdir)
    mkpath(per_dir)
    for (key, res) in results
        ks = string(key)
        # Coverage matrix: rows = observations, columns = nominal levels.
        cov_df = DataFrame(res.pred_coverage, :auto)
        rename!(cov_df, [Symbol("level_$(round(l; digits=3))") for l in res.nominal_levels])
        if !isnothing(res.param_names) && length(res.param_names) == nrow(cov_df)
            insertcols!(cov_df, 1, :param => res.param_names)
        end
        CSV.write(joinpath(per_dir, "$(ks)_coverage.csv"), cov_df)

        # Width matrix: rows = observations, columns = nominal levels.
        wid_df = DataFrame(res.interval_width, :auto)
        rename!(wid_df, [Symbol("level_$(round(l; digits=3))") for l in res.nominal_levels])
        if !isnothing(res.param_names) && length(res.param_names) == nrow(wid_df)
            insertcols!(wid_df, 1, :param => res.param_names)
        end
        CSV.write(joinpath(per_dir, "$(ks)_widths.csv"), wid_df)
    end
    println("  [CSV] per-model coverage + width tables saved to $per_dir")
end

# Shared experiment runners

"""
    run_brenda_experiment(; kwargs...) -> Dict{Symbol,CVResult}

Run BRENDA leave-one-source-out cross-validation for the six paper model cells:
Gaussian, Student-t, and skew-normal likelihoods, each with fixed effects and
random source effects. The function writes:

- `brenda_summary.csv`
- `brenda_loso.jld2`
- `per_model/*_coverage.csv`
- `per_model/*_widths.csv`
- BRENDA paper plots and forest-comparison plots under `plots_dir`
"""
function run_brenda_experiment(;
    y, σ2, Arows, grp, eb, param_names, obs_names,
    pathway_sym, pathway_str, results_dir, plots_dir,
    A0 :: Union{Nothing,AbstractMatrix} = nothing,
    b0 :: Union{Nothing,AbstractVector} = nothing,
    obs_map              :: Union{Nothing,AbstractMatrix} = nothing,
    obs_database_mean    :: Union{Nothing,AbstractVector} = nothing,
    obs_database_std     :: Union{Nothing,AbstractVector} = nothing,
    kinetic_divisor :: Float64 = log(10.0),
    holdout_unit :: Symbol = :source
)
    holdout_unit in (:source, :record) ||
        error("run_brenda_experiment: holdout_unit must be :source or :record")

    models = [(:gaussian,false),(:gaussian,true),(:t,false),(:t,true),(:skew,false),(:skew,true)]
    results  = Dict{Symbol, CVResult}()
    forests  = Dict{Symbol, ForestPayload}()
    cv_runner = holdout_unit == :source ? run_brenda_loso_experiment : run_brenda_loo_experiment
    holdout_label = holdout_unit == :source ? "LOSO" : "record LOO"

    can_forest = !isnothing(obs_map) && !isnothing(obs_database_mean) &&
                 !isnothing(obs_database_std) && !isnothing(obs_names)
    if !can_forest
        @warn "run_brenda_experiment: forest prerequisites missing; skipping forest plots."
    end

    for (model, has_re) in models
        key  = Symbol("$(model)_$(has_re ? "RE" : "Fix")")
        println("  -> $key")
        spec = make_spec(model, has_re; pathway = pathway_sym)

        # Run held-out CV + full-data fit (for coverage/scoring metrics)
        res = cv_runner(
            Vector{Float64}(y), Vector{Float64}(σ2),
            Matrix{Float64}(Arows), Vector{String}(grp);
            pathway  = pathway_str,
            spec     = spec,
            eb       = eb,
            seed     = SEED,
            param_names    = param_names,
            nominal_levels = NOMINAL_LEVELS,
            method   = :collapsed,
            A0 = A0, b0 = b0,
            obs_map  = obs_map,
            obs_names = obs_names,
            obs_database_mean = obs_database_mean,
            obs_database_std = obs_database_std,
            deduplicate_thermo_block = true
        )
        results[key] = res

        # Compute forest payload via closed-form posterior (exact for Gaussian)
        # Apply same dedup check as LOSO CV for consistency
        if can_forest
            τ²_forest = has_re ? max(res.τ²_final, 0.0) : 0.0
            d_local = size(Arows, 2)
            _A0_forest, _b0_forest = _prepare_information_pair(
                Matrix{Float64}(Arows), d_local, A0, b0;
                deduplicate_thermo_block = true
            )
            println("    [forest] closed-form with τ²=$(round(τ²_forest, digits=4)), " *
                    "A0 norm=$(round(norm(_A0_forest), digits=2))")
            forests[key] = compute_closedform_forest_payload(
                Matrix{Float64}(Arows),
                Vector{Float64}(y),
                Vector{Float64}(σ2),
                Vector{Float64}(eb.q0),
                Matrix{Float64}(eb.C0),
                Matrix{Float64}(obs_map),
                Vector{String}(obs_names),
                Vector{Float64}(obs_database_mean),
                Vector{Float64}(obs_database_std);
                τ² = τ²_forest,
                A0 = _A0_forest,
                b0 = _b0_forest
            )
        end
    end

    summary = brenda_summary_table(results)
    summary_file = holdout_unit == :source ? "brenda_summary.csv" : "brenda_record_loo_summary.csv"
    jld_file = holdout_unit == :source ? "brenda_loso.jld2" : "brenda_record_loo.jld2"
    per_model_dir = holdout_unit == :source ? "per_model" : "per_model_record_loo"

    CSV.write(joinpath(results_dir, summary_file), summary)
    JLD2.jldsave(joinpath(results_dir, jld_file); results...)
    _save_results_csv(results, results_dir; subdir = per_model_dir)

    println("\n $pathway_str BRENDA $holdout_label Summary ")
    show(stdout, summary; allrows=true, allcols=true); println()

    plot_paper_brenda(results;
        outdir          = plots_dir,
        pathway         = pathway_str,
        kinetic_divisor = kinetic_divisor)

    # Combined panel: heatmap row plus scoring bars.
    plot_brenda_combined_panel(results;
        level    = 0.95,
        pathway  = pathway_str,
        outpath  = joinpath(plots_dir, "combined_panel.pdf"))

    if !isempty(forests)
        plot_forest_comparisons(forests;
            outdir          = joinpath(plots_dir, "forest"),
            kinetic_divisor = kinetic_divisor)
    end

    return results
end

"""
    run_synthetic_experiment(; kwargs...) -> Dict{Symbol,CVResult}

Run the synthetic LOSO experiment for the same six model cells and regenerate
synthetic paper plots. The function writes:

- `synthetic_summary.csv`
- `synthetic.jld2`
- `per_model/*_coverage.csv`
- `per_model/*_widths.csv`
- synthetic paper plots and forest-comparison plots under `plots_dir`
"""
function run_synthetic_experiment(;
    posterior_file, pathway_str, param_names, obs_names,
    results_dir, plots_dir,
    A0 :: Union{Nothing,AbstractMatrix} = nothing,
    b0 :: Union{Nothing,AbstractVector} = nothing,
    obs_map         :: Union{Nothing,AbstractMatrix} = nothing,
    kinetic_divisor :: Float64 = log(10.0)
)
    results = Dict{Symbol, CVResult}()
    forests = Dict{Symbol, ForestPayload}()

    can_forest = !isnothing(obs_map) && !isnothing(obs_names)
    if !can_forest
        @warn "run_synthetic_experiment: obs_map / obs_names not provided; " *
              "forest plots will be skipped."
    end

    τ²_data_path = lowercase(pathway_str) == "tca" ? TAU2_DATA : TAU2_DATA / 6

    for (model, has_re) in [(:gaussian,false),(:gaussian,true),
                             (:t,false),(:t,true),(:skew,false),(:skew,true)]
        key  = Symbol("$(model)_$(has_re ? "RE" : "Fix")")
        println("  -> synth $key")
        spec = make_spec_synth(model, has_re)

        # Run LOSO CV + full-data fit (for coverage/scoring metrics)
        res = run_synthetic_loso_experiment(
            posterior_file;
            pathway       = pathway_str,
            n_reps        = N_REPS_SYN,
            seed          = SEED,
            data_model    = model,
            fit_model     = model,
            data_has_re   = true,
            fit_has_re    = has_re,
            τ²_data       = τ²_data_path,
            ν_data        = NU_DATA,
            α_data        = ALPHA_DATA,
            nominal_levels = NOMINAL_LEVELS,
            spec          = spec,
            param_names   = param_names,
            method        = :collapsed,
            A0 = A0, b0 = b0,
            obs_map  = obs_map,
            obs_names = obs_names,
            deduplicate_thermo_block = true
        )
        results[key] = res

        # Compute forest payload via the closed-form posterior.
        if can_forest
            τ²_forest = has_re ? max(res.τ²_final, 0.0) : 0.0
            println("    [forest] closed-form with τ²=$(round(τ²_forest, digits=4))")
            forests[key] = compute_synthetic_closedform_forest(
                posterior_file,
                Matrix{Float64}(obs_map),
                Vector{String}(obs_names);
                n_reps      = N_REPS_SYN,
                seed        = SEED,
                data_model  = model,
                data_has_re = true,
                τ²_data     = τ²_data_path,
                ν_data      = NU_DATA,
                α_data      = ALPHA_DATA,
                τ²_fit      = τ²_forest,
                A0          = A0,
                b0          = b0
            )
        end
    end

    summary = brenda_summary_table(results)
    CSV.write(joinpath(results_dir, "synthetic_summary.csv"), summary)
    JLD2.jldsave(joinpath(results_dir, "synthetic.jld2"); results...)
    _save_results_csv(results, results_dir)

    println("\n $pathway_str Synthetic Summary ")
    show(stdout, summary; allrows=true, allcols=true); println()

    plot_paper_synthetic(results;
        outdir          = plots_dir,
        pathway         = pathway_str,
        kinetic_divisor = kinetic_divisor)

    if !isempty(forests)
        plot_synthetic_forest_comparisons(forests;
            outdir          = joinpath(plots_dir, "forest"),
            kinetic_divisor = kinetic_divisor)
    end

    return results
end

"""
    _mean_cov_safe(res, level, use_hdi; target=:auto) -> Float64

Return mean coverage at `level`. `target=:predictive` uses LOSO predictive
coverage, `target=:parameter` uses parameter-recovery intervals, and
`target=:auto` tries parameter recovery before predictive coverage. In automatic
mode, synthetic experiments treat all-zero parameter coverage as a valid result;
experiments without a known parameter truth fall back to predictive coverage.
"""
function _mean_cov_safe(res::CVResult, level::Float64, use_hdi::Bool; target::Symbol = :auto)
    idx = findfirst(x -> isapprox(x, level, atol = 1e-5), res.nominal_levels)
    isnothing(idx) && return NaN

    target in (:auto, :parameter, :predictive) ||
        error("coverage target must be :auto, :parameter, or :predictive")

    if target == :predictive
        return _mean_coverage_column(res.pred_coverage, idx; zero_is_missing = false)
    end

    param_mat = use_hdi ? res.coverage_HDI : res.coverage_CI
    target == :parameter &&
        return _mean_coverage_column(param_mat, idx; zero_is_missing = false)

    zero_is_missing = !(res.experiment_type == :synthetic && res.q_true !== nothing)
    param_cov = _mean_coverage_column(param_mat, idx; zero_is_missing = zero_is_missing)
    isfinite(param_cov) && return param_cov

    return _mean_coverage_column(res.pred_coverage, idx; zero_is_missing = false)
end

function _mean_coverage_column(mat, idx::Int; zero_is_missing::Bool)
    isnothing(mat) && return NaN
    size(mat, 2) < idx && return NaN
    vals = Float64[Float64(x) for x in mat[:, idx] if isfinite(x)]
    isempty(vals) && return NaN
    zero_is_missing && all(iszero, vals) && return NaN
    return mean(vals)
end

"""
    run_coverage_sweep_experiment(; kwargs...) -> CoverageSweepResult

Sweep a distributional parameter in the synthetic data-generating process and
compare Gaussian fixed-effects fitting with the matched robust fixed-effects
model. Source heterogeneity, sparse observations, and rogue measurements are
disabled so the sweep isolates distributional misspecification. The skew sweep
uses a location-parameterized skew-normal DGP with variance-matched scale;
computational scaling and biological hypothesis tests are run by separate
scripts.
"""
function run_coverage_sweep_experiment(;
    posterior_file :: String,
    sweep_param    :: Symbol,              # :nu or :alpha
    sweep_values   :: Vector{Float64},
    pathway_str    :: String,
    param_names    :: Vector{String},
    results_dir    :: String,
    A0 = nothing, b0 = nothing,
    obs_map  = nothing,
    obs_names :: Union{Vector{String}, Nothing} = nothing,
    n_reps    :: Int   = 200,
    seed      :: Int   = SEED,
    nominal_levels = NOMINAL_LEVELS,
    coverage_target::Symbol = :parameter
) :: CoverageSweepResult

    isfile(posterior_file) ||
        error("posterior_file not found: $posterior_file. Run the main experiment driver first.")
    mkpath(results_dir)

    sweep_param in (:nu, :alpha) ||
        error("sweep_param must be :nu or :alpha, got $sweep_param")

    true_dgp = sweep_param == :nu ? :t : :skew
    robust_model = true_dgp
    skew_location = sweep_param == :alpha ? :location : :mean

    coverage_target == :parameter ||
        error("coverage_target must be :parameter for recovery sweeps")

    coverage_idx = findfirst(x -> isapprox(x, 0.95, atol = 1e-5), nominal_levels)
    isnothing(coverage_idx) && error("nominal_levels must include 0.95")

    println("\n  Coverage sweep: $sweep_param over $(length(sweep_values)) points " *
            "(pathway=$(pathway_str), DGP=$(true_dgp), tau2_data=0, " *
            "coverage=$(coverage_target), generator=shape_sweep, " *
            "skew_location=$(skew_location), n_reps=$n_reps, recovery=replicated)")

    is_tca = lowercase(pathway_str) == "tca"
    n_mc_gauss = 400
    n_mc_robust = robust_model == :skew ? (is_tca ? 1600 : 2000) : 600
    thin_robust = robust_model == :skew ? 4 : 2
    n_keep_robust = robust_model == :skew ? 250 : 200
    n_out_gauss = 1
    n_out_robust = 1
    n_pred = 800

    # A single outer pass keeps the empirical-Bayes calibration fixed across
    # repeated synthetic data sets. Re-estimating q_prior and lambda inside
    # each replicate under-covers even at the Gaussian null point.
    spec_gauss = FoldFitSpec(
        model = :gaussian, hierarchical = false,
        n_outer = n_out_gauss, n_mcmc = n_mc_gauss,
        burnin = div(n_mc_gauss, 4),
        thin = 2, n_keep_min = 150, n_pred_draws = n_pred, b_draws_ll = 50,
        posterior_refresh = 1, adaptive_stop = false, rtol_outer = 5e-4,
        min_outer = 1, tau_update = :ig, tau_pc_rate = 0.0,
        ν_init = 50.0, α_init = 0.0,
        update_nu = false, update_alpha = false,
        refine_shape = false, shape_opt_tol = 1e-3,
        skew_location = :mean,
        nu_penalty = 0.0, nu_target = 50.0, alpha_penalty = 0.0)

    function _spec_robust(nu_init::Float64, alpha_init::Float64)
        FoldFitSpec(
            model = robust_model, hierarchical = false,
            n_outer = n_out_robust, n_mcmc = n_mc_robust,
            burnin = div(n_mc_robust, 4),
            thin = thin_robust, n_keep_min = n_keep_robust, n_pred_draws = n_pred, b_draws_ll = 50,
            posterior_refresh = 1, adaptive_stop = false, rtol_outer = 5e-4,
            min_outer = 1, tau_update = :ig, tau_pc_rate = 0.0,
            ν_init = nu_init, α_init = alpha_init,
            update_nu = false, update_alpha = false,
            refine_shape = false, shape_opt_tol = 1e-3,
            skew_location = skew_location,
            nu_penalty = 0.0, nu_target = nu_init, alpha_penalty = 0.0)
    end

    gc_ci = Float64[]
    gc_hdi = Float64[]
    rc_ci = Float64[]
    rc_hdi = Float64[]
    gc_abs_bias = Float64[]
    rc_abs_bias = Float64[]
    gc_rmse = Float64[]
    rc_rmse = Float64[]
    gc_mahal = Float64[]
    rc_mahal = Float64[]
    gc_width95 = Float64[]
    rc_width95 = Float64[]
    robust_nu = Float64[]
    robust_alpha = Float64[]

    for val in sweep_values
        nu_data = sweep_param == :nu ? val : 50.0
        alpha_data = sweep_param == :alpha ? val : 0.0
        nu_fit = robust_model == :t ? max(nu_data, 2.05) : 50.0
        alpha_fit = robust_model == :skew ? alpha_data : 0.0
        push!(robust_nu, robust_model == :t ? nu_fit : NaN)
        push!(robust_alpha, robust_model == :skew ? alpha_fit : NaN)

        @printf("    %s = %6.2f  [nu_data=%.1f, alpha_data=%.1f, fit_shape=%.1f]\n",
                string(sweep_param), val, nu_data, alpha_data,
                robust_model == :t ? nu_fit : alpha_fit)

        for (fit_model, spec, ci_vec, hdi_vec, bias_vec, rmse_vec, mahal_vec, width_vec) in [
                (:gaussian, spec_gauss, gc_ci, gc_hdi, gc_abs_bias, gc_rmse, gc_mahal, gc_width95),
                (robust_model, _spec_robust(nu_fit, alpha_fit), rc_ci, rc_hdi, rc_abs_bias, rc_rmse, rc_mahal, rc_width95)]

            res = try
                run_sweep_recovery(
                    posterior_file;
                    n_reps = n_reps,
                    seed = seed,
                    data_model = true_dgp,
                    fit_model = fit_model,
                    ν_data = nu_data,
                    α_data = alpha_data,
                    nominal_levels = nominal_levels,
                    spec = spec,
                    method = :collapsed,
                    A0 = A0, b0 = b0,
                    deduplicate_thermo_block = true,
                    skew_location = skew_location)
            catch err
                @warn "Coverage sweep fit failed" pathway=pathway_str parameter=sweep_param value=val fit_model=fit_model exception=(err, catch_backtrace())
                nothing
            end

            if isnothing(res)
                push!(ci_vec, NaN)
                push!(hdi_vec, NaN)
                push!(bias_vec, NaN)
                push!(rmse_vec, NaN)
                push!(mahal_vec, NaN)
                push!(width_vec, NaN)
            else
                push!(ci_vec, mean(res.coverage_CI[:, coverage_idx]))
                push!(hdi_vec, mean(res.coverage_HDI[:, coverage_idx]))
                push!(bias_vec, res.mean_abs_bias)
                push!(rmse_vec, res.mean_rmse)
                push!(mahal_vec, res.mahal_mse)
                push!(width_vec, mean(res.interval_width_CI[:, coverage_idx]))
            end
        end
    end

    null_idx = sweep_param == :nu ?
        findlast(v -> v >= 30.0, sweep_values) :
        findfirst(v -> abs(v) <= 0.5, sweep_values)

    if !isnothing(null_idx)
        @printf("  Null point %.2f: Gaussian=%.3f, %s=%.3f\n",
                sweep_values[null_idx], gc_ci[null_idx], string(robust_model), rc_ci[null_idx])
        if isfinite(gc_ci[null_idx]) && gc_ci[null_idx] < 0.80
            @warn "Gaussian coverage is low at the near-null sweep point" pathway=pathway_str parameter=sweep_param coverage=gc_ci[null_idx]
        end
    end

    sweep_res = CoverageSweepResult(
        sweep_param, sweep_values,
        gc_ci, gc_hdi, rc_ci, rc_hdi,
        robust_model, n_reps, pathway_str, coverage_target)

    tag  = sweep_param == :nu ? "nu" : "alpha"
    base = joinpath(results_dir, "coverage_sweep_$(tag)_$(lowercase(pathway_str))")
    JLD2.jldsave("$(base).jld2"; sweep_res = sweep_res)

    CSV.write("$(base).csv", DataFrame(
        sweep_val = sweep_values,
        gauss_ci = gc_ci,
        gauss_hdi = gc_hdi,
        robust_ci = rc_ci,
        robust_hdi = rc_hdi,
        robust_model = string(robust_model),
        pathway = pathway_str,
        n_reps = n_reps,
        sweep_param = string(sweep_param),
        gauss_mean_abs_bias = gc_abs_bias,
        robust_mean_abs_bias = rc_abs_bias,
        gauss_mean_rmse = gc_rmse,
        robust_mean_rmse = rc_rmse,
        gauss_mahal_mse = gc_mahal,
        robust_mahal_mse = rc_mahal,
        gauss_width95 = gc_width95,
        robust_width95 = rc_width95,
        tau2_data = 0.0,
        data_dgp = string(true_dgp),
        coverage_target = string(coverage_target),
        recovery = "replicated",
        generator = "shape_sweep",
        calibration = "fixed_eb_one_outer",
        shape_mode = "matched",
        skew_location = string(skew_location),
        sparsity = 0.0,
        n_rogue = 0,
        robust_nu = robust_nu,
        robust_alpha = robust_alpha))

    println("  Saved: $(base).jld2 and $(base).csv")

    return sweep_res
end

# Main execution: TCA cycle

println("\nTCA CYCLE")

# Input data
equilibrator_data_path_tca = "data/equilibrator/equilibrator_tca_cycle.json"
kcat_data_path_tca = "data/brenda/kcat_values_tca_cycle.csv"
KM_data_path_tca   = "data/brenda/KM_values_tca_cycle.csv"

# Model reconstruction
mdl = load_yeast_reconstruction()
dict_reactions = reaction_data_dict(mdl)

# Pathway definition
function tca_cycle_reactions()
    generate_reactions(
        ["r_0300","r_0302","r_0280","r_0658","r_0832","r_0831",
         "r_0505","r_1022","r_1021","r_0451","r_0713"],
        dict_reactions, protons=false, proton_name="s_0799")
end
rxns_tca = tca_cycle_reactions()

bigg_ids_tca = Dict("s_0376"=>"accoa","s_0807"=>"h2o","s_1273"=>"oaa","s_0524"=>"cit","s_0532"=>"coa",
    "s_0517"=>"acon_C","s_0941"=>"icit","s_1200"=>"nad","s_0182"=>"akg","s_0460"=>"co2",
    "s_1205"=>"nadh","s_1097"=>"lpam","s_1411"=>"sdhlam","s_0627"=>"dhlam","s_1464"=>"succoa",
    "s_0397"=>"adp","s_1326"=>"pi","s_0437"=>"atp","s_1460"=>"succ","s_1537"=>"q6",
    "s_0727"=>"fum","s_1535"=>"q6h2","s_0068"=>"mal__L")
brenda_metabolite_ids_tca = Dict("s_0376"=>"acetyl-CoA","s_0807"=>"H2O","s_1273"=>"oxaloacetate",
    "s_0524"=>"citrate","s_0532"=>"CoA","s_0517"=>"cis-aconitate","s_0941"=>"isocitrate",
    "s_1200"=>"NAD+","s_0182"=>"2-oxoglutarate","s_0460"=>"CO2","s_1205"=>"NADH",
    "s_1097"=>"lipoamide","s_1411"=>"succinyldihydrolipoamide","s_0627"=>"DL-dihydrolipoamide",
    "s_1464"=>"succinyl-CoA","s_0397"=>"ADP","s_1326"=>"phosphate","s_0437"=>"ATP",
    "s_1460"=>"succinate","s_1537"=>"ubiquinone","s_0727"=>"fumarate","s_1535"=>"ubiquinol",
    "s_0068"=>"L-malate")
bigg_reaction_ids_tca = Dict("r_0300"=>"CSm","r_0302"=>"ACN_a_m","r_0280"=>"ACN_b_m",
    "r_0658"=>"ICDHxm","r_0832"=>"AKGDam","r_0831"=>"AKGDbm","r_0505"=>"GCC2cm",
    "r_1022"=>"SUCOASm","r_1021"=>"SUCD2_u6m","r_0451"=>"FUMm","r_0713"=>"MDHm")
ec_codes_tca = Dict("r_0300"=>"2.3.3.1","r_0302"=>"4.2.1.3","r_0280"=>"4.2.1.3",
    "r_0658"=>"1.1.1.41","r_0832"=>"1.2.4.2","r_0831"=>"2.3.1.61","r_0505"=>"1.8.1.4",
    "r_1022"=>"6.2.1.5","r_1021"=>"1.3.5.1","r_0451"=>"4.2.1.2","r_0713"=>"1.1.1.37")
forward_metabolites_tca = Dict("r_0300"=>"acetyl-CoA","r_0302"=>"citrate","r_0280"=>"cis-aconitate",
    "r_0658"=>"isocitrate","r_0832"=>"2-oxoglutarate","r_0831"=>"succinyldihydrolipoamide",
    "r_0505"=>"DL-dihydrolipoamide","r_1022"=>"succinyl-CoA","r_1021"=>"succinate",
    "r_0451"=>"fumarate","r_0713"=>"L-malate")
reverse_metabolites_tca = Dict("r_0300"=>"citrate","r_0302"=>"cis-aconitate","r_0280"=>"isocitrate",
    "r_0658"=>"2-oxoglutarate","r_0832"=>"succinyldihydrolipoamide",
    "r_0831"=>"DL-dihydrolipoamide","r_0505"=>"lipoamide","r_1022"=>"succinate",
    "r_1021"=>"fumarate","r_0451"=>"L-malate","r_0713"=>"oxaloacetate")

(eq_names_tca, ΔGf_mu_tca, ΔGf_cov_tca) = load_equilibrator_data(equilibrator_data_path_tca)
(kcat_med_tca, KM_med_tca) = load_BRENDA_data(kcat_data_path_tca, KM_data_path_tca)

h2o_tca = ["s_0807"]
Q_tca = observation_matrix(rxns_tca, h2o_strings=h2o_tca)
# `Q` is kept as a top-level compatibility binding for legacy helper code.
Q = Q_tca
(species_tca, reactions_tca) = rn_entities(rxns_tca)
(μ0_map_tca, KM_map_tca, kcatf_tca, kcatr_tca) = observation_matrix_index_maps(rxns_tca)

met_obs_tca = [bigg_ids_tca[s] for s in species_tca]
μ0eq_tca = Dict(bigg_ids_tca[k] => v for (k,v) in μ0_map_tca)
(μ0_mean_tca, μ0_cov_tca, Q_μ0_tca) = chemical_potential_obs(ΔGf_mu_tca, ΔGf_cov_tca, eq_names_tca, met_obs_tca, μ0eq_tca)
(KM_mu_tca, KM_cov_tca, Q_KM_tca) = KM_obs(rxns_tca, KM_med_tca, ec_codes_tca, brenda_metabolite_ids_tca, KM_map_tca)
(kcat_mu_tca, kcat_cov_tca, Q_kcat_tca) = kcat_obs(rxns_tca, kcat_med_tca, ec_codes_tca, forward_metabolites_tca, reverse_metabolites_tca, kcatf_tca, kcatr_tca)

Q_obs_tca = vcat(Q_μ0_tca, Q_KM_tca, Q_kcat_tca)
μ_obs_tca = vcat(μ0_mean_tca, KM_mu_tca, kcat_mu_tca)
μ_cov_tca = BlockDiagonal([μ0_cov_tca, KM_cov_tca, kcat_cov_tca])
(q_prior_tca, C_prior_tca) = thermodynamic_prior(rxns_tca, h2o_tca)
(q_post_tca, C_post_tca) = infer_parameters(Q_obs_tca, μ_obs_tca, μ_cov_tca, q_prior_tca, C_prior_tca)

d_tca = size(Q_tca, 2)
println("TCA: d=$d_tca, n_obs=$(size(Q_obs_tca,1))")
println("TCA obs sigma2 range: ", extrema(diag(μ_cov_tca)))

# Parameter and observable labels used in result tables and forest plots.
const PARAM_NAMES_TCA = make_param_names(
    μ0_map=μ0_map_tca, KM_map=KM_map_tca, kcatfmap=kcatf_tca, kcatrmap=kcatr_tca,
    bigg_ids=bigg_ids_tca, bigg_reaction_ids=bigg_reaction_ids_tca,
    array_species=species_tca, array_reactions=reactions_tca, d=d_tca)

# Observable names follow the same iteration order as KM_obs and kcat_obs.
# They are not derived from Q sparsity because kcat/KM rows have multiple nonzeros.
const _μ0_names_tca, _KM_names_tca, _kcat_names_tca = build_obs_block_names(
    rxns_tca,
    [bigg_ids_tca[s] for s in species_tca],
    KM_med_tca,
    kcat_med_tca,
    ec_codes_tca,
    brenda_metabolite_ids_tca,
    forward_metabolites_tca,
    reverse_metabolites_tca,
    bigg_reaction_ids_tca;
    display_metabolite_ids = bigg_ids_tca,
    extract_name_fn = extract_name
)
const OBS_NAMES_TCA = make_observable_names_from_obs(
    μ0_names  = _μ0_names_tca,
    KM_names  = _KM_names_tca,
    kcat_names = _kcat_names_tca
)

# Verify observable name block counts.
let n_mu0  = count(n -> startswith(n, "μ0:") || startswith(n, "mu0:"), OBS_NAMES_TCA),
    n_kcat = count(n -> startswith(n, "kcat:"), OBS_NAMES_TCA),
    n_km   = count(n -> startswith(n, "KM:"), OBS_NAMES_TCA)
    println("OBS_NAMES_TCA: $(length(OBS_NAMES_TCA)) total = $n_mu0 μ0 + $n_km KM + $n_kcat kcat")
    println("  Q blocks: Q_μ0=$(size(Q_μ0_tca,1)) rows, Q_KM=$(size(Q_KM_tca,1)) rows, Q_kcat=$(size(Q_kcat_tca,1)) rows")
    n_obs_expected = size(Q_μ0_tca,1) + size(Q_KM_tca,1) + size(Q_kcat_tca,1)
    length(OBS_NAMES_TCA) == n_obs_expected ||
        error("OBS_NAMES_TCA length $(length(OBS_NAMES_TCA)) != expected $n_obs_expected")
    n_kcat == 0 && @warn "No kcat entries in OBS_NAMES_TCA"
    n_km == 0 && @warn "No KM entries in OBS_NAMES_TCA"
    # Print representative observable names.
    n_kcat > 0 && println("  kcat sample: ", OBS_NAMES_TCA[findfirst(n -> startswith(n, "kcat:"), OBS_NAMES_TCA)])
    n_km > 0 && println("  KM sample: ", OBS_NAMES_TCA[findfirst(n -> startswith(n, "KM:"), OBS_NAMES_TCA)])
end

# Output locations
const TCA_ROOT    = joinpath(dirname(@__DIR__), "output", "tca")
const TCA_DATA    = joinpath(TCA_ROOT, "data")
const TCA_RESULTS = joinpath(TCA_ROOT, "results")
const TCA_BRENDA_PLOTS = joinpath(TCA_ROOT, "plots", "brenda")
const TCA_BRENDA_RECORD_PLOTS = joinpath(TCA_ROOT, "plots", "brenda_record_loo")
const TCA_SYNTH_PLOTS  = joinpath(TCA_ROOT, "plots", "synthetic")
const TCA_RECORD_RESULTS = joinpath(TCA_RESULTS, "record_loo")
for d in [TCA_DATA, TCA_RESULTS, TCA_RECORD_RESULTS, TCA_BRENDA_PLOTS, TCA_BRENDA_RECORD_PLOTS, TCA_SYNTH_PLOTS]; mkpath(d); end
const TCA_POSTERIOR = joinpath(TCA_DATA, "posterior_tca.jld2")
const TCA_BRENDA   = joinpath(TCA_DATA, "brenda_inputs_tca.jld2")

# Compute TCA thermodynamic information pair and database reference.
A0_tca, b0_tca = make_information_pair(Q_μ0_tca, μ0_mean_tca, Matrix(μ0_cov_tca))

db_mean_tca, db_std_tca = build_database_reference_from_blocks(
    μ0_mean = μ0_mean_tca,
    μ0_cov = Matrix(μ0_cov_tca),
    KM_mu = KM_mu_tca,
    KM_cov = Matrix(KM_cov_tca),
    kcat_mu = kcat_mu_tca,
    kcat_cov = Matrix(kcat_cov_tca)
)
println("TCA A0 norm: ", norm(A0_tca), ", db_mean finite: ", count(isfinite, db_mean_tca), "/", d_tca)

# BRENDA experiments
println("\n TCA: BRENDA LOSO ")
tca_brenda_data = load(TCA_BRENDA)
tca_eb = EBSpec(q0=q_prior_tca, C0=C_prior_tca)

# Long-running stage: fits all six BRENDA model configurations for TCA.
results_tca_brenda = run_brenda_experiment(
    y = Vector{Float64}(tca_brenda_data["y"]),
    σ2 = Vector{Float64}(tca_brenda_data["σ2"]),
    Arows = Matrix{Float64}(tca_brenda_data["Arows"]),
    grp = Vector{String}(tca_brenda_data["grp"]),
    eb = tca_eb,
    param_names = PARAM_NAMES_TCA,
    obs_names = OBS_NAMES_TCA,
    pathway_sym = :tca,
    pathway_str = "TCA",
    results_dir = TCA_RESULTS,
    plots_dir = TCA_BRENDA_PLOTS,
    A0 = A0_tca,
    b0 = b0_tca,
    obs_map = Q_obs_tca,
    obs_database_mean = db_mean_tca,
    obs_database_std = db_std_tca,
    kinetic_divisor = log(10.0)
)

results_tca_brenda_record = nothing
if RUN_RECORD_LOO
    println("\n TCA: BRENDA record-level LOO ")
    results_tca_brenda_record = run_brenda_experiment(
        y = Vector{Float64}(tca_brenda_data["y"]),
        σ2 = Vector{Float64}(tca_brenda_data["σ2"]),
        Arows = Matrix{Float64}(tca_brenda_data["Arows"]),
        grp = Vector{String}(tca_brenda_data["grp"]),
        eb = tca_eb,
        param_names = PARAM_NAMES_TCA,
        obs_names = OBS_NAMES_TCA,
        pathway_sym = :tca,
        pathway_str = "TCA",
        results_dir = TCA_RECORD_RESULTS,
        plots_dir = TCA_BRENDA_RECORD_PLOTS,
        A0 = A0_tca,
        b0 = b0_tca,
        obs_map = Q_obs_tca,
        obs_database_mean = db_mean_tca,
        obs_database_std = db_std_tca,
        kinetic_divisor = log(10.0),
        holdout_unit = :record
    )
else
    println("\n TCA: BRENDA record-level LOO skipped (PB_RUN_RECORD_LOO=false)")
end

# Synthetic experiments
println("\n TCA: Synthetic ")

# Legacy sweep helpers in `experiment_runners.jl` expect these globals. The
# sweep helper itself is not called in this script.
const PATHWAY = "tca"
const OUTPUT_ROOT = TCA_ROOT
const DATA_DIR = TCA_DATA
const RESULTS_DIR = TCA_RESULTS
const PLOTS_DIR = joinpath(TCA_ROOT, "plots")
const PLOTS_SYNTHETIC = TCA_SYNTH_PLOTS
const PLOTS_FOREST = joinpath(TCA_SYNTH_PLOTS, "forest")
const PLOTS_SWEEP = joinpath(TCA_ROOT, "plots", "sweep")
for d in [PLOTS_SYNTHETIC, PLOTS_FOREST, PLOTS_SWEEP]; mkpath(d); end
const POSTERIOR_FILE = TCA_POSTERIOR

# Long-running stage: fits all six synthetic model configurations for TCA.
results_tca_synth = run_synthetic_experiment(
    posterior_file = TCA_POSTERIOR,
    pathway_str = "tca",
    param_names = PARAM_NAMES_TCA,
    obs_names = OBS_NAMES_TCA,
    results_dir = TCA_RESULTS,
    plots_dir = TCA_SYNTH_PLOTS,
    A0 = A0_tca,
    b0 = b0_tca,
    obs_map = Q_obs_tca
)

# Main execution: glycolysis

println("\nGLYCOLYSIS")

# Input data
equilibrator_data_path_g = "data/equilibrator/equilibrator_glycolysis.json"
kcat_data_path_g = "data/brenda/kcat_values_glycolysis.csv"
KM_data_path_g   = "data/brenda/KM_values_glycolysis.csv"

# Pathway definition
function glycolysis_reactions()
    generate_reactions(
        ["r_0534","r_0467","r_0886","r_0450","r_1054",
         "r_0486","r_0892","r_0893","r_0366","r_0962"],
        dict_reactions, protons=false, proton_name="s_0794")
end
rxns_g = glycolysis_reactions()

bigg_ids_g = Dict(
    "s_0434"=>"atp","s_0563"=>"glc__D","s_0394"=>"adp","s_0568"=>"g6p",
    "s_0557"=>"f6p","s_0555"=>"fdp","s_0629"=>"dhap","s_0764"=>"g3p",
    "s_1198"=>"nad","s_1322"=>"pi","s_0075"=>"13dpg","s_1203"=>"nadh",
    "s_0260"=>"3pg","s_0188"=>"2pg","s_0803"=>"h2o","s_1360"=>"pep","s_1399"=>"pyr")
brenda_metabolite_ids_g = Dict(
    "s_0434"=>"ATP","s_0563"=>"D-glucose","s_0394"=>"ADP",
    "s_0568"=>"D-glucose 6-phosphate","s_0557"=>"D-fructose 6-phosphate",
    "s_0555"=>"D-fructose 1,6-bisphosphate","s_0629"=>"dihydroxyacetone phosphate",
    "s_0764"=>"D-glyceraldehyde 3-phosphate","s_1198"=>"NAD+","s_1322"=>"phosphate",
    "s_0075"=>"3-phospho-D-glyceroyl phosphate","s_1203"=>"NADH",
    "s_0260"=>"3-phospho-D-glycerate","s_0188"=>"2-phospho-D-glycerate",
    "s_0803"=>"H2O","s_1360"=>"phosphoenolpyruvate","s_1399"=>"pyruvate")
bigg_reaction_ids_g = Dict(
    "r_0534"=>"HEX1","r_0467"=>"PGI","r_0886"=>"PFK","r_0450"=>"FBA",
    "r_1054"=>"TPI","r_0486"=>"GAPD","r_0892"=>"PGK","r_0893"=>"PGM",
    "r_0366"=>"ENO","r_0962"=>"PYK")
ec_codes_g = Dict(
    "r_0534"=>"2.7.1.1","r_0467"=>"5.3.1.9","r_0886"=>"2.7.1.11",
    "r_0450"=>"4.1.2.13","r_1054"=>"5.3.1.1","r_0486"=>"1.2.1.12",
    "r_0892"=>"2.7.2.3","r_0893"=>"5.4.2.11","r_0366"=>"4.2.1.11","r_0962"=>"2.7.1.40")
forward_metabolites_g = Dict(
    "r_0534"=>"D-glucose","r_0467"=>"D-glucose 6-phosphate",
    "r_0886"=>"D-fructose 6-phosphate","r_0450"=>"D-fructose 1,6-bisphosphate",
    "r_1054"=>"dihydroxyacetone phosphate","r_0486"=>"D-glyceraldehyde 3-phosphate",
    "r_0892"=>"3-phospho-D-glyceroyl phosphate","r_0893"=>"3-phospho-D-glycerate",
    "r_0366"=>"2-phospho-D-glycerate","r_0962"=>"phosphoenolpyruvate")
reverse_metabolites_g = Dict(
    "r_0534"=>"D-glucose 6-phosphate","r_0467"=>"D-fructose 6-phosphate",
    "r_0886"=>"D-fructose 1,6-bisphosphate","r_0450"=>"D-glyceraldehyde 3-phosphate",
    "r_1054"=>"D-glyceraldehyde 3-phosphate","r_0486"=>"3-phospho-D-glyceroyl phosphate",
    "r_0892"=>"3-phospho-D-glycerate","r_0893"=>"2-phospho-D-glycerate",
    "r_0366"=>"phosphoenolpyruvate","r_0962"=>"pyruvate")

(eq_names_g, ΔGf_mu_g, ΔGf_cov_g) = load_equilibrator_data(equilibrator_data_path_g)
(kcat_med_g, KM_med_g) = load_BRENDA_data(kcat_data_path_g, KM_data_path_g)

h2o_g = ["s_0803"]
Q_g = observation_matrix(rxns_g, h2o_strings=h2o_g)
# Update the legacy compatibility binding to the active pathway.
Q = Q_g
(species_g, reactions_g) = rn_entities(rxns_g)
(μ0_map_g, KM_map_g, kcatf_g, kcatr_g) = observation_matrix_index_maps(rxns_g)

met_obs_g = [bigg_ids_g[s] for s in species_g]
μ0eq_g = Dict(bigg_ids_g[k] => v for (k,v) in μ0_map_g)
(μ0_mean_g, μ0_cov_g, Q_μ0_g) = chemical_potential_obs(ΔGf_mu_g, ΔGf_cov_g, eq_names_g, met_obs_g, μ0eq_g)
(KM_mu_g, KM_cov_g, Q_KM_g) = KM_obs(rxns_g, KM_med_g, ec_codes_g, brenda_metabolite_ids_g, KM_map_g)
(kcat_mu_g, kcat_cov_g, Q_kcat_g) = kcat_obs(rxns_g, kcat_med_g, ec_codes_g, forward_metabolites_g, reverse_metabolites_g, kcatf_g, kcatr_g)

Q_obs_g = vcat(Q_μ0_g, Q_KM_g, Q_kcat_g)
μ_obs_g = vcat(μ0_mean_g, KM_mu_g, kcat_mu_g)
μ_cov_g = BlockDiagonal([μ0_cov_g, KM_cov_g, kcat_cov_g])
(q_prior_g, C_prior_g) = thermodynamic_prior(rxns_g, h2o_g)
(q_post_g, C_post_g) = infer_parameters(Q_obs_g, μ_obs_g, μ_cov_g, q_prior_g, C_prior_g)

d_g = size(Q_g, 2)
println("Glycolysis: d=$d_g, n_obs=$(size(Q_obs_g,1))")
println("Glycolysis obs sigma2 range: ", extrema(diag(μ_cov_g)))

# Parameter and observable labels used in result tables and forest plots.
const PARAM_NAMES_GLYC = make_param_names(
    μ0_map=μ0_map_g, KM_map=KM_map_g, kcatfmap=kcatf_g, kcatrmap=kcatr_g,
    bigg_ids=bigg_ids_g, bigg_reaction_ids=bigg_reaction_ids_g,
    array_species=species_g, array_reactions=reactions_g, d=d_g)

const _μ0_names_g, _KM_names_g, _kcat_names_g = build_obs_block_names(
    rxns_g,
    [bigg_ids_g[s] for s in species_g],
    KM_med_g,
    kcat_med_g,
    ec_codes_g,
    brenda_metabolite_ids_g,
    forward_metabolites_g,
    reverse_metabolites_g,
    bigg_reaction_ids_g;
    display_metabolite_ids = bigg_ids_g,
    extract_name_fn = extract_name
)
const OBS_NAMES_GLYC = make_observable_names_from_obs(
    μ0_names  = _μ0_names_g,
    KM_names  = _KM_names_g,
    kcat_names = _kcat_names_g
)

# Verify observable name block counts.
let n_mu0  = count(n -> startswith(n, "μ0:") || startswith(n, "mu0:"), OBS_NAMES_GLYC),
    n_kcat = count(n -> startswith(n, "kcat:"), OBS_NAMES_GLYC),
    n_km   = count(n -> startswith(n, "KM:"), OBS_NAMES_GLYC)
    println("OBS_NAMES_GLYC: $(length(OBS_NAMES_GLYC)) total = $n_mu0 μ0 + $n_km KM + $n_kcat kcat")
    println("  Q blocks: Q_μ0=$(size(Q_μ0_g,1)) rows, Q_KM=$(size(Q_KM_g,1)) rows, Q_kcat=$(size(Q_kcat_g,1)) rows")
    n_obs_expected = size(Q_μ0_g,1) + size(Q_KM_g,1) + size(Q_kcat_g,1)
    length(OBS_NAMES_GLYC) == n_obs_expected ||
        error("OBS_NAMES_GLYC length $(length(OBS_NAMES_GLYC)) != expected $n_obs_expected")
    n_kcat > 0 && println("  kcat sample: ", OBS_NAMES_GLYC[findfirst(n -> startswith(n, "kcat:"), OBS_NAMES_GLYC)])
    n_km > 0 && println("  KM sample: ", OBS_NAMES_GLYC[findfirst(n -> startswith(n, "KM:"), OBS_NAMES_GLYC)])
end

# Output locations
const GLYC_ROOT    = joinpath(dirname(@__DIR__), "output", "glycolysis")
const GLYC_DATA    = joinpath(GLYC_ROOT, "data")
const GLYC_RESULTS = joinpath(GLYC_ROOT, "results")
const GLYC_BRENDA_PLOTS = joinpath(GLYC_ROOT, "plots", "brenda")
const GLYC_BRENDA_RECORD_PLOTS = joinpath(GLYC_ROOT, "plots", "brenda_record_loo")
const GLYC_SYNTH_PLOTS  = joinpath(GLYC_ROOT, "plots", "synthetic")
const GLYC_RECORD_RESULTS = joinpath(GLYC_RESULTS, "record_loo")
for d in [GLYC_DATA, GLYC_RESULTS, GLYC_RECORD_RESULTS, GLYC_BRENDA_PLOTS, GLYC_BRENDA_RECORD_PLOTS, GLYC_SYNTH_PLOTS]; mkpath(d); end
const GLYC_POSTERIOR = joinpath(GLYC_DATA, "posterior_glycolysis.jld2")
const GLYC_BRENDA   = joinpath(GLYC_DATA, "brenda_inputs_glycolysis.jld2")

# Compute glycolysis thermodynamic information pair and database reference.
A0_g, b0_g = make_information_pair(Q_μ0_g, μ0_mean_g, Matrix(μ0_cov_g))

# Observable-space database reference with length n_obs.
db_mean_g, db_std_g = build_database_reference_from_blocks(
    μ0_mean = μ0_mean_g,
    μ0_cov = Matrix(μ0_cov_g),
    KM_mu = KM_mu_g,
    KM_cov = Matrix(KM_cov_g),
    kcat_mu = kcat_mu_g,
    kcat_cov = Matrix(kcat_cov_g)
)
println("Glycolysis A0 norm: ", norm(A0_g),
    ", db_mean length: ", length(db_mean_g),
    " (obs_names: ", length(OBS_NAMES_GLYC), ")",
    ", db_mean finite: ", count(isfinite, db_mean_g))

# BRENDA experiments
println("\n Glycolysis: BRENDA LOSO ")
glyc_brenda_data = load(GLYC_BRENDA)
glyc_eb = EBSpec(q0=q_prior_g, C0=C_prior_g)
println("Glycolysis BRENDA: n=$(length(glyc_brenda_data["y"])), d=$(size(glyc_brenda_data["Arows"],2)), sources=$(length(unique(glyc_brenda_data["grp"])))")

# Long-running stage: fits all six BRENDA model configurations for glycolysis.
results_glyc_brenda = run_brenda_experiment(
    y = Vector{Float64}(glyc_brenda_data["y"]),
    σ2 = Vector{Float64}(glyc_brenda_data["σ2"]),
    Arows = Matrix{Float64}(glyc_brenda_data["Arows"]),
    grp = Vector{String}(glyc_brenda_data["grp"]),
    eb = glyc_eb,
    param_names = PARAM_NAMES_GLYC,
    obs_names = OBS_NAMES_GLYC,
    pathway_sym = :glycolysis,
    pathway_str = "Glycolysis",
    results_dir = GLYC_RESULTS,
    plots_dir = GLYC_BRENDA_PLOTS,
    A0 = A0_g,
    b0 = b0_g,
    obs_map = Q_obs_g,
    obs_database_mean = db_mean_g,
    obs_database_std = db_std_g,
    kinetic_divisor = log(10.0)
)

results_glyc_brenda_record = nothing
if RUN_RECORD_LOO
    println("\n Glycolysis: BRENDA record-level LOO ")
    results_glyc_brenda_record = run_brenda_experiment(
        y = Vector{Float64}(glyc_brenda_data["y"]),
        σ2 = Vector{Float64}(glyc_brenda_data["σ2"]),
        Arows = Matrix{Float64}(glyc_brenda_data["Arows"]),
        grp = Vector{String}(glyc_brenda_data["grp"]),
        eb = glyc_eb,
        param_names = PARAM_NAMES_GLYC,
        obs_names = OBS_NAMES_GLYC,
        pathway_sym = :glycolysis,
        pathway_str = "Glycolysis",
        results_dir = GLYC_RECORD_RESULTS,
        plots_dir = GLYC_BRENDA_RECORD_PLOTS,
        A0 = A0_g,
        b0 = b0_g,
        obs_map = Q_obs_g,
        obs_database_mean = db_mean_g,
        obs_database_std = db_std_g,
        kinetic_divisor = log(10.0),
        holdout_unit = :record
    )
else
    println("\n Glycolysis: BRENDA record-level LOO skipped (PB_RUN_RECORD_LOO=false)")
end

# Synthetic experiments
println("\n Glycolysis: Synthetic ")

# Long-running stage: fits all six synthetic model configurations for glycolysis.
results_glyc_synth = run_synthetic_experiment(
    posterior_file = GLYC_POSTERIOR,
    pathway_str = "glycolysis",
    param_names = PARAM_NAMES_GLYC,
    obs_names = OBS_NAMES_GLYC,
    results_dir = GLYC_RESULTS,
    plots_dir = GLYC_SYNTH_PLOTS,
    A0 = A0_g,
    b0 = b0_g,
    obs_map = Q_obs_g
)

# Output summary
println("\nMAIN LOSO EXPERIMENTS COMPLETE")
println("TCA BRENDA:      $TCA_BRENDA_PLOTS")
RUN_RECORD_LOO && println("TCA BRENDA LOO:  $TCA_BRENDA_RECORD_PLOTS")
println("TCA Synthetic:   $TCA_SYNTH_PLOTS")
println("Glyc BRENDA:     $GLYC_BRENDA_PLOTS")
RUN_RECORD_LOO && println("Glyc BRENDA LOO: $GLYC_BRENDA_RECORD_PLOTS")
println("Glyc Synthetic:  $GLYC_SYNTH_PLOTS")


# Coverage sweep grids
const NU_SWEEP_VALS    = [2.1, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
const ALPHA_SWEEP_VALS = [-10.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 10.0]
const N_REPS_SWEEP     = 200

println("\n TCA: Coverage sweep - nu (distributional misspecification, tau2=0) ")
sweep_tca_nu = run_coverage_sweep_experiment(
    posterior_file = TCA_POSTERIOR,
    sweep_param    = :nu,
    sweep_values   = NU_SWEEP_VALS,
    pathway_str    = "TCA",
    param_names    = PARAM_NAMES_TCA,
    results_dir    = TCA_RESULTS,
    A0 = A0_tca, b0 = b0_tca,
    obs_map = Q_obs_tca, obs_names = OBS_NAMES_TCA,
    n_reps  = N_REPS_SWEEP)

plot_coverage_sweep(sweep_tca_nu;
    outpath = joinpath(TCA_SYNTH_PLOTS, "coverage_vs_df.pdf"))
plot_point_accuracy_sweep(joinpath(TCA_RESULTS, "coverage_sweep_nu_tca.csv");
    outpath = joinpath(TCA_SYNTH_PLOTS, "point_accuracy_vs_df.pdf"))

println("\n TCA: Coverage sweep - alpha (distributional misspecification, tau2=0) ")
sweep_tca_alpha = run_coverage_sweep_experiment(
    posterior_file = TCA_POSTERIOR,
    sweep_param    = :alpha,
    sweep_values   = ALPHA_SWEEP_VALS,
    pathway_str    = "TCA",
    param_names    = PARAM_NAMES_TCA,
    results_dir    = TCA_RESULTS,
    A0 = A0_tca, b0 = b0_tca,
    obs_map = Q_obs_tca, obs_names = OBS_NAMES_TCA,
    n_reps  = N_REPS_SWEEP)

plot_coverage_sweep(sweep_tca_alpha;
    outpath = joinpath(TCA_SYNTH_PLOTS, "coverage_vs_skew.pdf"))
plot_point_accuracy_sweep(joinpath(TCA_RESULTS, "coverage_sweep_alpha_tca.csv");
    outpath = joinpath(TCA_SYNTH_PLOTS, "point_accuracy_vs_skew.pdf"))

println("\n Glycolysis: Coverage sweep - nu ")
sweep_glyc_nu = run_coverage_sweep_experiment(
    posterior_file = GLYC_POSTERIOR,
    sweep_param    = :nu,
    sweep_values   = NU_SWEEP_VALS,
    pathway_str    = "glycolysis",
    param_names    = PARAM_NAMES_GLYC,
    results_dir    = GLYC_RESULTS,
    A0 = A0_g, b0 = b0_g,
    obs_map = Q_obs_g, obs_names = OBS_NAMES_GLYC,
    n_reps  = N_REPS_SWEEP)

plot_coverage_sweep(sweep_glyc_nu;
    outpath = joinpath(GLYC_SYNTH_PLOTS, "coverage_vs_df.pdf"))
plot_point_accuracy_sweep(joinpath(GLYC_RESULTS, "coverage_sweep_nu_glycolysis.csv");
    outpath = joinpath(GLYC_SYNTH_PLOTS, "point_accuracy_vs_df.pdf"))

println("\n Glycolysis: Coverage sweep - alpha ")
sweep_glyc_alpha = run_coverage_sweep_experiment(
    posterior_file = GLYC_POSTERIOR,
    sweep_param    = :alpha,
    sweep_values   = ALPHA_SWEEP_VALS,
    pathway_str    = "glycolysis",
    param_names    = PARAM_NAMES_GLYC,
    results_dir    = GLYC_RESULTS,
    A0 = A0_g, b0 = b0_g,
    obs_map = Q_obs_g, obs_names = OBS_NAMES_GLYC,
    n_reps  = N_REPS_SWEEP)

plot_coverage_sweep(sweep_glyc_alpha;
    outpath = joinpath(GLYC_SYNTH_PLOTS, "coverage_vs_skew.pdf"))
plot_point_accuracy_sweep(joinpath(GLYC_RESULTS, "coverage_sweep_alpha_glycolysis.csv");
    outpath = joinpath(GLYC_SYNTH_PLOTS, "point_accuracy_vs_skew.pdf"))

println("\nCOVERAGE SWEEPS COMPLETE")
println("TCA coverage plots:  $TCA_SYNTH_PLOTS")
println("Glyc coverage plots: $GLYC_SYNTH_PLOTS")
