#!/usr/bin/env julia

using RobustBayesianParameterBalancing
using CSV
using DataFrames
using LinearAlgebra
using Random
using Statistics
using Distributions
using SpecialFunctions

const HELP_TEXT = """
Parameter balancing command-line helper

Usage:
  julia --project=. bin/parameter-balance.jl prior-summary [--prior FILE] [--output FILE]
  julia --project=. bin/parameter-balance.jl validate-table --data FILE [--prior FILE] [--output FILE]
  julia --project=. bin/parameter-balance.jl pseudo-fill --data FILE --template FILE --output FILE [--prior FILE]
  julia --project=. bin/parameter-balance.jl balance-gaussian --Q FILE --x FILE --Cx FILE --q-prior FILE --C-prior FILE --output FILE
  julia --project=. bin/parameter-balance.jl balance-hierarchical --Q FILE --x FILE --Cx FILE --q-prior FILE --C-prior FILE --group FILE --output FILE [--mcmc N] [--outer N] [--seed N] [--error-scale auto|replication|NUMBER] [--replicate-group FILE]
  julia --project=. bin/parameter-balance.jl kineticize --sbml FILE --balanced FILE --parameters FILE [--output-sbml FILE]

Commands:
  prior-summary         Export the bundled or user-supplied prior table in normalized CSV form.
  validate-table        Validate an SBtab-like quantity table against the prior definitions.
  pseudo-fill           Add prior pseudo-observations for missing slots in a quantity table.
  balance-gaussian      Run the closed-form Gaussian Bayesian parameter-balancing update.
  balance-hierarchical  Run the hierarchical model recommended in the paper: a Gaussian
                        likelihood with a source-level random intercept (--group gives one
                        source label per observation, e.g. organism or organism|type).
                        Reports the posterior mean and standard deviation of the basis
                        parameters together with the fitted between-source variance tau2.
                        --error-scale replication estimates the multiplicative
                        error scale from replicate records alone, with no model
                        fit: pass --replicate-group with one label per
                        observation identifying records that measure the same
                        quantity in the same source, whose spread is then pure
                        measurement error.  This is the recommended setting.
                        --error-scale auto additionally estimates a multiplicative
                        record-level error scale from the data by an empirical-Bayes
                        moment iteration, instead of trusting the supplied variances;
                        a numeric value applies a fixed scale.
                        With --error-scale replication the two-class population of
                        replicate-group scales (a tight and a loose class) is also
                        fitted by expectation-maximisation on the replicate statistics
                        and reported: --predict FILE then writes, for every row of Q,
                        the posterior of the derived quantity and the predictive
                        interval for a previously unmeasured source, under the single
                        scale and under the two-class population.
                        --prior-mean fixed (default) keeps the prior mean at the
                        supplied thermodynamic prior; --prior-mean adaptive shrinks it
                        toward the posterior mean (empirical Bayes), which narrows the
                        parameter intervals below nominal on synthetic data and is
                        kept only for reproducing the cross-validation runs.
                        --predict-levels 0.95,0.99 sets the predictive levels.
                        --output-cov FILE writes the posterior covariance of the basis
                        vector q, from which the covariance of any derived quantity
                        Q q follows as Q C Q'.
                        --type FILE (one quantity-type label per observation, e.g.
                        kcat or KM) stratifies the between-source variance by type
                        with the Paule-Mandel estimator on the source-cell means,
                        falling back to the pooled value for a type whose estimate
                        is outside 0.2 to 5 times the pooled one; this is the
                        released specification of the paper, and --predict then
                        uses each row's own type variance.
  kineticize            Copy an SBML file if requested and export a balanced-parameter sidecar.
"""

function parse_options(args)
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            key = a[3:end]
            i == length(args) && error("Missing value for option $a")
            opts[key] = args[i + 1]
            i += 2
        else
            error("Unexpected argument: $a")
        end
    end
    return opts
end

function require_option(opts, key)
    haskey(opts, key) || error("Missing required option --$key")
    return opts[key]
end

function read_table(path)
    delim = endswith(lowercase(path), ".tsv") ? '\t' : ','
    return DataFrame(CSV.File(path; delim = delim, missingstring = [""]))
end

function read_numeric_matrix(path)
    try
        return Matrix{Float64}(CSV.read(path, DataFrame))
    catch
        return Matrix{Float64}(CSV.read(path, DataFrame; header = false))
    end
end

read_numeric_vector(path) = vec(read_numeric_matrix(path))

function maybe_write(df::DataFrame, path::Union{Nothing,String})
    if isnothing(path)
        show(stdout, df; allrows = true, allcols = true)
        println()
    else
        delim = endswith(lowercase(path), ".tsv") ? '\t' : ','
        CSV.write(path, df; delim = delim)
        println("Wrote $path")
    end
    return df
end

function load_prior_from_opts(opts)
    return haskey(opts, "prior") ? load_parameter_balancing_prior(opts["prior"]) :
           load_parameter_balancing_prior()
end

function cmd_prior_summary(opts)
    prior = load_prior_from_opts(opts)
    df = DataFrame(
        quantity_type = collect(keys(prior.entries)),
        unit = [v.unit for v in values(prior.entries)],
        biological_element = string.([v.biological_element for v in values(prior.entries)]),
        mathematical_type = string.([v.mathematical_type for v in values(prior.entries)]),
        dependence = string.([v.dependence for v in values(prior.entries)]),
        prior_median = [v.prior_median for v in values(prior.entries)],
        lower_bound = [v.lower_bound for v in values(prior.entries)],
        upper_bound = [v.upper_bound for v in values(prior.entries)]
    )
    sort!(df, :quantity_type)
    maybe_write(df, get(opts, "output", nothing))
end

function cmd_validate_table(opts)
    prior = load_prior_from_opts(opts)
    data = read_table(require_option(opts, "data"))
    report = validate_pb_inputs(nothing, data; prior = prior)
    maybe_write(report, get(opts, "output", nothing))
end

function cmd_pseudo_fill(opts)
    prior = load_prior_from_opts(opts)
    data = read_table(require_option(opts, "data"))
    template = read_table(require_option(opts, "template"))
    filled = fill_pseudo_observations(data, prior; template = template)
    maybe_write(filled, require_option(opts, "output"))
end

function cmd_balance_gaussian(opts)
    Q = read_numeric_matrix(require_option(opts, "Q"))
    x = read_numeric_vector(require_option(opts, "x"))
    Cx = read_numeric_matrix(require_option(opts, "Cx"))
    q_prior = read_numeric_vector(require_option(opts, "q-prior"))
    C_prior = read_numeric_matrix(require_option(opts, "C-prior"))
    result = balance_parameters((Q = Q, x = x, Cx = Cx, q_prior = q_prior, C_prior = C_prior))
    export_balanced_table(result; format = :csv, path = require_option(opts, "output"))
    println("Wrote $(opts["output"])")
end

function cmd_balance_hierarchical(opts)
    Q = read_numeric_matrix(require_option(opts, "Q"))
    x = read_numeric_vector(require_option(opts, "x"))
    Cx = read_numeric_matrix(require_option(opts, "Cx"))
    q_prior = read_numeric_vector(require_option(opts, "q-prior"))
    C_prior = read_numeric_matrix(require_option(opts, "C-prior"))
    grp_df = read_table(require_option(opts, "group"))
    grp = String.(strip.(string.(grp_df[!, 1])))

    n, d = size(Q)
    length(x) == n || error("x must have length $(n) to match Q")
    length(grp) == n || error("--group must supply one label per observation; got $(length(grp)), expected $n")
    sigma2 = size(Cx, 2) == 1 ? vec(Cx) : diag(Cx)
    length(sigma2) == n || error("Cx must provide $n variances (vector or diagonal)")
    all(>(0), sigma2) || error("Cx variances must be positive")

    n_mcmc = parse(Int, get(opts, "mcmc", "2000"))
    n_outer = parse(Int, get(opts, "outer", "4"))
    seed = parse(Int, get(opts, "seed", "20260805"))
    scale_opt = get(opts, "error-scale", "1.0")

    spec = FoldFitSpec(model = :gaussian, hierarchical = true,
        n_outer = n_outer, n_mcmc = n_mcmc, burnin = div(n_mcmc, 4), thin = 1,
        n_keep_min = 60, posterior_refresh = 1, adaptive_stop = true,
        rtol_outer = 1e-3, min_outer = 2, tau_update = :ig,
        update_nu = false, update_alpha = false, refine_shape = false)
    prior_mean_opt = get(opts, "prior-mean", "fixed")
    prior_mean_opt in ("fixed", "adaptive") || error("--prior-mean must be fixed or adaptive")
    eb = EBSpec(q0 = q_prior, C0 = C_prior, κ0 = prior_mean_opt == "fixed" ? 1e-9 : 50.0)
    A0 = zeros(d, d); b0 = zeros(d)

    fit_once(s2) = begin
        hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
        fit_fold_mcem(Random.MersenneTwister(seed), A0, b0, Q, x, s2, grp, eb, hs, spec;
                      method = :collapsed)
    end

    c = 1.0
    mixture = nothing
    fit = fit_once(sigma2)
    if scale_opt == "auto"
        # Empirical-Bayes within-cell moment iteration for a multiplicative
        # error scale.  Records in the same source cell share the same random
        # effect exactly, so the within-cell dispersion of the marginal
        # consensus residuals r_i = x_i - a_i' qhat identifies c free of the
        # between-source variance; cells with a single record contribute
        # nothing.  See run_revision_sigmacal.jl for the derivation.
        bygroup = Dict{String,Vector{Int}}()
        for i in 1:n
            push!(get!(bygroup, grp[i], Int[]), i)
        end
        for _ in 1:4
            m = vec(mean(Q * fit.qsamps, dims = 2))
            num = 0.0; den = 0.0
            for (_, ig) in bygroup
                ng = length(ig)
                ng >= 2 || continue
                rbar = mean(x[i] - m[i] for i in ig)
                for i in ig
                    num += (x[i] - m[i] - rbar)^2
                    den += sigma2[i] * (1.0 - 1.0 / ng)
                end
            end
            den > 0 || error("--error-scale auto needs at least one source with two or more observations")
            c = clamp(num / den, 0.25, 25.0)
            fit = fit_once(c .* sigma2)
        end
    elseif scale_opt == "replication"
        # Replication estimator.  Records that measure the same quantity in the
        # same source share an identical model mean exactly, so their deviations
        # from the group mean are pure record error and c is identified without
        # any fit.  See run_revision_hetero.jl for the derivation and
        # validate_hetero.jl for the validation.
        repfile = get(opts, "replicate-group", nothing)
        repfile === nothing &&
            error("--error-scale replication needs --replicate-group with one label per observation")
        rep = string.(vec(Matrix(read_table(repfile))[:, 1]))
        length(rep) == n ||
            error("--replicate-group must supply one label per observation; got $(length(rep)), expected $n")
        bygroup = Dict{String,Vector{Int}}()
        for i in 1:n
            push!(get!(bygroup, rep[i], Int[]), i)
        end
        num = 0.0; den = 0.0
        for (_, ig) in bygroup
            ng = length(ig)
            ng >= 2 || continue
            xbar = mean(x[i] for i in ig)
            for i in ig
                num += (x[i] - xbar)^2
                den += sigma2[i] * (1.0 - 1.0 / ng)
            end
        end
        den > 0 ||
            error("--error-scale replication needs at least one replicate group with two or more observations")
        c = clamp(num / den, 0.25, 40.0)
        fit = fit_once(c .* sigma2)
        # Two-class population of replicate-group scales, by EM on the group
        # statistics r_g = S_g / D_g with df_g = n_g - 1, for which
        # r_g df_g / c_g ~ chi^2_df given the group's true scale c_g.
        rg = Float64[]; dfg = Float64[]
        for (_, ig) in bygroup
            ng = length(ig); ng >= 2 || continue
            xbar = mean(x[i] for i in ig)
            S = sum((x[i] - xbar)^2 for i in ig); D = sum(sigma2[i] * (1.0 - 1.0 / ng) for i in ig)
            D > 0 || continue
            push!(rg, max(S / D, 1e-10)); push!(dfg, ng - 1.0)
        end
        if length(rg) >= 20
            c1 = quantile(rg, 0.25); c2 = quantile(rg, 0.75); pi1 = 0.5
            c1 == c2 && (c2 = 2c1 + 1e-3)
            w = similar(rg)
            for _ in 1:200
                for g in eachindex(rg)
                    l1 = log(pi1) + logpdf(Gamma(dfg[g] / 2, 2c1 / dfg[g]), rg[g])
                    l2 = log(1 - pi1) + logpdf(Gamma(dfg[g] / 2, 2c2 / dfg[g]), rg[g])
                    mx = max(l1, l2); w[g] = exp(l1 - mx) / (exp(l1 - mx) + exp(l2 - mx))
                end
                c1 = sum(w .* dfg .* rg) / max(sum(w .* dfg), 1e-12)
                c2 = sum((1 .- w) .* dfg .* rg) / max(sum((1 .- w) .* dfg), 1e-12)
                pi1 = clamp(mean(w), 1e-3, 1 - 1e-3)
            end
            c1 > c2 && ((c1, c2, pi1) = (c2, c1, 1 - pi1))
            ll1 = sum(logpdf(Gamma(dfg[g] / 2, 2c / dfg[g]), rg[g]) for g in eachindex(rg))
            ll2 = sum(begin
                l1 = log(pi1) + logpdf(Gamma(dfg[g] / 2, 2c1 / dfg[g]), rg[g])
                l2 = log(1 - pi1) + logpdf(Gamma(dfg[g] / 2, 2c2 / dfg[g]), rg[g])
                mx = max(l1, l2); mx + log(exp(l1 - mx) + exp(l2 - mx))
            end for g in eachindex(rg))
            # Match the population mean to the pooled replication scale, so that the
            # mixture changes only the SHAPE of the record error and not its variance.
            # The paper adopts this mean-matched form; the form fitted directly has a
            # mean 0.65 to 0.91 of the pooled scale and quietly narrows the record
            # variance as well as reshaping it.
            mean_fit = pi1 * c1 + (1 - pi1) * c2
            k2 = mean_fit > 0 ? c / mean_fit : 1.0
            mixture = (pi1 = pi1, c1 = k2 * c1, c2 = k2 * c2, mean_fit = mean_fit,
                       rescale = k2, dll = ll2 - ll1, groups = length(rg))
            println("Two-class scale population over $(length(rg)) replicate groups: " *
                    "$(round(100pi1, digits = 0))% tight, $(round(100(1 - pi1), digits = 0))% loose; " *
                    "fitted mean $(round(mean_fit, sigdigits = 3)) rescaled by " *
                    "$(round(k2, sigdigits = 3)) to match the pooled scale, giving " *
                    "c1 = $(round(k2 * c1, sigdigits = 3)) and c2 = $(round(k2 * c2, sigdigits = 3)); " *
                    "log-likelihood gain over the single scale $(round(ll2 - ll1, digits = 1)).")
        else
            println("Two-class scale population not fitted: fewer than 20 replicate groups.")
        end
    elseif scale_opt != "1.0"
        c = parse(Float64, scale_opt)
        c > 0 || error("--error-scale must be positive")
        fit = fit_once(c .* sigma2)
    end

    q_mean = vec(mean(fit.qsamps, dims = 2))
    q_sd = vec(std(fit.qsamps, dims = 2))
    out = DataFrame(index = 1:d, q_mean = q_mean, q_sd = q_sd)
    maybe_write(out, require_option(opts, "output"))
    # The posterior covariance of the basis vector: the covariance of any derived
    # quantity Q q is Q C Q', so a kinetic model can use the joint uncertainty.
    covfile = get(opts, "output-cov", nothing)
    if covfile !== nothing
        Cq = cov(fit.qsamps, dims = 2)
        maybe_write(DataFrame(Cq, [Symbol("q", j) for j in 1:d]), covfile)
        println("Wrote the posterior covariance of the basis vector to $covfile")
    end
    println("Fitted between-source variance tau2 = $(round(fit.hs.τ², sigdigits = 4)) " *
            "over $(length(fit.grp_levels)) sources; error scale c = $(round(c, sigdigits = 4)); " *
            "prior mean $(prior_mean_opt).")

    # Between-source variance stratified by quantity type (the released
    # specification): Paule-Mandel on the source-cell means of the consensus
    # residuals, with the pooled value as the fallback where the estimator loses
    # the signal.  --type gives one label per observation.
    typefile = get(opts, "type", nothing)
    kinds = nothing
    τ2_of = Dict{String,Float64}()
    if typefile !== nothing
        kinds = String.(strip.(string.(read_table(typefile)[!, 1])))
        length(kinds) == n || error("--type must supply one label per observation; got $(length(kinds)), expected $n")
        τ2_pool = fit.hs.τ²
        # The released set forms its cells as organism-by-type; compose them here so
        # that the per-type estimator is stratified correctly whether --group carries
        # organism labels or organism-by-type labels (the composition is idempotent
        # for the latter only up to the label text, which does not affect grouping).
        cells = [string(grp[i], "|", kinds[i]) for i in 1:n]
        raw = tau_pm_by_kind(x, sigma2, fill(c, n), Q, fit, cells, kinds, collect(1:n), τ2_pool)
        τ2_of, τ2_src = tau2_by_type_with_fallback(raw, kinds, τ2_pool)
        for k in sort(collect(keys(τ2_of)))
            println("Between-source variance for type $k = $(round(τ2_of[k], sigdigits = 4)) " *
                    "($(τ2_src[k]); Paule-Mandel estimate $(round(get(raw, k, NaN), sigdigits = 4)), " *
                    "pooled $(round(τ2_pool, sigdigits = 4)))")
        end
    end

    # Predictive intervals for a previously unmeasured source, per row of Q, under
    # the single scale and, when fitted, the two-class scale population.
    predfile = get(opts, "predict", nothing)
    if predfile !== nothing
        levels = parse.(Float64, split(get(opts, "predict-levels", "0.95,0.99"), ","))
        Qfull = Q
        S = size(fit.qsamps, 2)
        M = Qfull * fit.qsamps
        τ2 = fit.hs.τ²
        s2ref = median(sigma2)
        rng = Random.MersenneTwister(seed + 1)
        ndraw = 4000
        rows = NamedTuple[]
        for i in 1:size(Qfull, 1)
            mu = vec(M[i, :])
            z1 = randn(rng, ndraw); z2 = randn(rng, ndraw); u = rand(rng, ndraw)
            idx = rand(rng, 1:S, ndraw)
            τ2i = kinds === nothing ? τ2 : get(τ2_of, kinds[i], τ2)
            single = sort!([mu[idx[r]] + sqrt(τ2i) * z1[r] + sqrt(c * s2ref) * z2[r] for r in 1:ndraw])
            mixd = mixture === nothing ? nothing :
                sort!([mu[idx[r]] + sqrt(τ2i) * z1[r] +
                       sqrt((u[r] < mixture.pi1 ? mixture.c1 : mixture.c2) * s2ref) * z2[r] for r in 1:ndraw])
            rec = Dict{Symbol,Any}(:row => i, :posterior_mean => mean(mu), :posterior_sd => std(mu),
                                   :tau2_used => τ2i)
            for lv in levels
                tag = replace(string(lv), "." => "")
                rec[Symbol("single_lo_$tag")] = quantile(single, (1 - lv) / 2; sorted = true)
                rec[Symbol("single_hi_$tag")] = quantile(single, (1 + lv) / 2; sorted = true)
                if mixd !== nothing
                    rec[Symbol("twoclass_lo_$tag")] = quantile(mixd, (1 - lv) / 2; sorted = true)
                    rec[Symbol("twoclass_hi_$tag")] = quantile(mixd, (1 + lv) / 2; sorted = true)
                end
            end
            push!(rows, NamedTuple(rec))
        end
        pred = DataFrame(rows)
        lead = [:row, :posterior_mean, :posterior_sd, :tau2_used]
        select!(pred, vcat(lead, sort(setdiff(propertynames(pred), lead))))
        maybe_write(pred, predfile)
        println("Wrote new-source predictive intervals for $(size(Qfull, 1)) rows to $predfile " *
                "(record variance taken as the median supplied variance times the error scale).")
    end
end

function cmd_kineticize(opts)
    balanced = read_table(require_option(opts, "balanced"))
    info = kineticize_sbml(require_option(opts, "sbml"), balanced;
        output_path = get(opts, "output-sbml", nothing),
        parameter_table_path = require_option(opts, "parameters"),
        overwrite_existing = true)
    println(info.message)
    println("Parameter table: $(info.parameter_table_path)")
    isnothing(info.sbml_path) || println("SBML path: $(info.sbml_path)")
end

function main(args)
    isempty(args) && (print(HELP_TEXT); return)
    command = first(args)
    command in ("-h", "--help", "help") && (print(HELP_TEXT); return)
    opts = parse_options(args[2:end])
    if command == "prior-summary"
        cmd_prior_summary(opts)
    elseif command == "validate-table"
        cmd_validate_table(opts)
    elseif command == "pseudo-fill"
        cmd_pseudo_fill(opts)
    elseif command == "balance-gaussian"
        cmd_balance_gaussian(opts)
    elseif command == "balance-hierarchical"
        cmd_balance_hierarchical(opts)
    elseif command == "kineticize"
        cmd_kineticize(opts)
    else
        error("Unknown command '$command'. Run with --help for usage.")
    end
end

main(ARGS)
