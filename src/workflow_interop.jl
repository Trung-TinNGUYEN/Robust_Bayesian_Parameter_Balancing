using CSV
using DataFrames
using LinearAlgebra
using Statistics
using Random
using Distributions

Base.@kwdef struct PBQuantityPrior
    quantity_type::String
    unit::String = ""
    biological_element::Symbol = :Unknown
    mathematical_type::Symbol = :Unknown
    prior_median::Float64 = NaN
    prior_std::Union{Nothing,Float64} = nothing
    prior_geometric_std::Union{Nothing,Float64} = nothing
    lower_bound::Float64 = -Inf
    upper_bound::Float64 = Inf
    data_std::Union{Nothing,Float64} = nothing
    data_geometric_std::Union{Nothing,Float64} = nothing
    dependence::Symbol = :Unknown
    use_as_prior::Bool = true
    abbreviation::String = ""
    matrix_info::String = ""
end

struct PBPrior
    table::DataFrame
    entries::Dict{String,PBQuantityPrior}
    basic_quantities::Vector{String}
    derived_quantities::Vector{String}
end

Base.@kwdef struct PBOptions
    use_pseudo_values::Bool = true
    ph::Float64 = 7.0
    temperature::Float64 = 298.15
    overwrite_kinetics::Bool = true
    cell_volume::Float64 = 43.0
    parametrisation::Symbol = :hal
    enzyme_prefactor::Bool = true
    default_inhibition::Symbol = :complete
    default_activation::Symbol = :complete
    model_name::String = "outputname"
    boundary_values::Symbol = :ignore
    size_limit::Int = 300
    raw::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

Base.@kwdef struct BalanceResult
    q_mean::Vector{Float64}
    q_cov::Matrix{Float64}
    q_samples::Union{Matrix{Float64},Nothing} = nothing
    observable_mean::Union{Vector{Float64},Nothing} = nothing
    observable_cov::Union{Matrix{Float64},Nothing} = nothing
    parameter_names::Vector{String} = String[]
    observable_names::Vector{String} = String[]
    quantity_types::Vector{String} = String[]
    units::Vector{String} = String[]
    prior::Union{PBPrior,Nothing} = nothing
    metadata::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

_pb_default_prior_path() = joinpath(dirname(@__DIR__), "data", "pb_prior.tsv")
_pb_default_options_path() = joinpath(dirname(@__DIR__), "data", "pb_options.tsv")

function _pb_read_tsv(path::AbstractString; header::Int = 2)
    isfile(path) || error("File not found: $path")
    return DataFrame(CSV.File(path; delim = '\t', header = header,
                              normalizenames = false, missingstring = [""]))
end

_pb_clean_name(s) = lowercase(replace(String(s), r"^!+" => "", r"[^A-Za-z0-9]+" => "_", r"_+" => "_", r"^_|_$" => ""))

function _canonicalize_pb_columns(df::DataFrame)
    out = copy(df)
    rename_map = Dict{Symbol,Symbol}()
    for nm in names(out)
        cn = _pb_clean_name(nm)
        target =
            cn in ("quantitytype", "quantity_type") ? :quantity_type :
            cn in ("reaction_sbml_reaction_id", "reaction_id", "reaction") ? :reaction :
            cn in ("compound_sbml_species_id", "species_id", "compound", "species") ? :compound :
            cn in ("mean", "median", "value") ? :mean :
            cn in ("std", "standard_deviation") ? :std :
            cn in ("geometricstd", "geometric_std") ? :geometric_std :
            cn in ("unit", "units") ? :unit :
            cn in ("source", "reference") ? :source :
            cn in ("ispseudo", "is_pseudo") ? :is_pseudo :
            cn in ("biologicalelement", "biological_element") ? :biological_element :
            cn in ("mathematicaltype", "mathematical_type") ? :mathematical_type :
            cn in ("priormedian", "prior_median") ? :prior_median :
            cn in ("priorstd", "prior_std") ? :prior_std :
            cn in ("priorgeometricstd", "prior_geometric_std") ? :prior_geometric_std :
            cn in ("lowerbound", "lower_bound") ? :lower_bound :
            cn in ("upperbound", "upper_bound") ? :upper_bound :
            cn in ("datastd", "data_std") ? :data_std :
            cn in ("datageometricstd", "data_geometric_std") ? :data_geometric_std :
            cn in ("dependence",) ? :dependence :
            cn in ("useaspriorinformation", "use_as_prior_information", "use_as_prior") ? :use_as_prior :
            cn in ("abbreviation",) ? :abbreviation :
            cn in ("matrixinfo", "matrix_info") ? :matrix_info :
            Symbol(cn)
        if target != Symbol(nm) && !(target in values(rename_map))
            rename_map[Symbol(nm)] = target
        end
    end
    rename!(out, rename_map)
    return out
end

_pb_str(x; default = "") = (ismissing(x) || isnothing(x)) ? default : strip(string(x))

function _pb_float(x; default = NaN)
    (ismissing(x) || isnothing(x) || strip(string(x)) == "") && return default
    x isa Number && return Float64(x)
    return parse(Float64, strip(string(x)))
end

function _pb_maybe_float(x)
    v = _pb_float(x; default = NaN)
    return isfinite(v) ? v : nothing
end

function _pb_bool(x; default = false)
    (ismissing(x) || isnothing(x) || strip(string(x)) == "") && return default
    s = lowercase(strip(string(x)))
    return s in ("1", "true", "t", "yes", "y", "on")
end

function _pb_symbol(x; default = :Unknown)
    s = _pb_str(x)
    isempty(s) && return default
    s = replace(s, r"[^A-Za-z0-9]+" => "_", r"_+" => "_", r"^_|_$" => "")
    isempty(s) && return default
    return Symbol(s)
end

_pb_hascol(df::DataFrame, col::Symbol) = col in propertynames(df)

function _pb_get(row, col::Symbol, default = missing)
    try
        col in propertynames(row) && return row[col]
    catch
        return default
    end
    return default
end
_pb_get_nt(data, key::Symbol, default = nothing) = hasproperty(data, key) ? getproperty(data, key) : default
_pb_get_dict(data::AbstractDict, key::Symbol, default = nothing) = get(data, key, get(data, String(key), default))

function _pb_data_value(data, key::Symbol, default = nothing)
    data isa AbstractDict && return _pb_get_dict(data, key, default)
    return _pb_get_nt(data, key, default)
end

function _pb_quantity_key(row)
    return (_pb_str(_pb_get(row, :quantity_type)), _pb_str(_pb_get(row, :reaction)), _pb_str(_pb_get(row, :compound)))
end

function _pb_quantity_prior_from_row(row)
    math_type = _pb_symbol(_pb_get(row, :mathematical_type), default = :Unknown)
    dep = _pb_symbol(_pb_get(row, :dependence), default = :Unknown)
    return PBQuantityPrior(
        quantity_type = _pb_str(_pb_get(row, :quantity_type)),
        unit = _pb_str(_pb_get(row, :unit)),
        biological_element = _pb_symbol(_pb_get(row, :biological_element), default = :Unknown),
        mathematical_type = math_type,
        prior_median = _pb_float(_pb_get(row, :prior_median)),
        prior_std = _pb_maybe_float(_pb_get(row, :prior_std)),
        prior_geometric_std = _pb_maybe_float(_pb_get(row, :prior_geometric_std)),
        lower_bound = _pb_float(_pb_get(row, :lower_bound), default = -Inf),
        upper_bound = _pb_float(_pb_get(row, :upper_bound), default = Inf),
        data_std = _pb_maybe_float(_pb_get(row, :data_std)),
        data_geometric_std = _pb_maybe_float(_pb_get(row, :data_geometric_std)),
        dependence = dep,
        use_as_prior = _pb_bool(_pb_get(row, :use_as_prior), default = true),
        abbreviation = _pb_str(_pb_get(row, :abbreviation)),
        matrix_info = _pb_str(_pb_get(row, :matrix_info))
    )
end

"""
    load_parameter_balancing_prior([path]) -> PBPrior

Load a classical parameter-balancing prior table in SBtab-style TSV form.
When `path` is omitted, the bundled defaults in `data/pb_prior.tsv` are used.
The returned `PBPrior` keeps the normalized table and a dictionary keyed by
quantity type.
"""
function load_parameter_balancing_prior(path::Union{Nothing,AbstractString} = nothing)
    path = isnothing(path) ? _pb_default_prior_path() : path
    df = _canonicalize_pb_columns(_pb_read_tsv(path; header = 2))
    _pb_hascol(df, :quantity_type) || error("Prior table must contain a QuantityType column")
    filter!(r -> !isempty(_pb_str(_pb_get(r, :quantity_type))), df)
    entries = Dict{String,PBQuantityPrior}()
    for row in eachrow(df)
        qtype = _pb_str(_pb_get(row, :quantity_type))
        entries[qtype] = _pb_quantity_prior_from_row(row)
    end
    basic = sort([k for (k, v) in entries if v.dependence == :Basic])
    derived = sort([k for (k, v) in entries if v.dependence == :Derived])
    return PBPrior(df, entries, basic, derived)
end

function _parse_pb_options(df::DataFrame)
    opts = Dict{Symbol,Any}()
    id_col = Symbol(names(df)[1])
    val_col = Symbol(names(df)[2])
    for row in eachrow(df)
        key = Symbol(_pb_clean_name(_pb_get(row, id_col)))
        val = _pb_str(_pb_get(row, val_col))
        isempty(String(key)) && continue
        opts[key] = val
    end
    return opts
end

"""
    load_parameter_balancing_options([path]) -> PBOptions

Load classical parameter-balancing options from an SBtab-style TSV file.
"""
function load_parameter_balancing_options(path::Union{Nothing,AbstractString} = nothing)
    path = isnothing(path) ? _pb_default_options_path() : path
    df = _pb_read_tsv(path; header = 2)
    raw = _parse_pb_options(df)
    getraw(key, default) = get(raw, key, default)
    return PBOptions(
        use_pseudo_values = _pb_bool(getraw(:use_pseudo_values, true), default = true),
        ph = _pb_float(get(raw, :ph, get(raw, :p_h, 7.0)), default = 7.0),
        temperature = _pb_float(getraw(:temperature, 298.15), default = 298.15),
        overwrite_kinetics = _pb_bool(getraw(:overwrite_kinetics, true), default = true),
        cell_volume = _pb_float(getraw(:cell_volume, 43.0), default = 43.0),
        parametrisation = _pb_symbol(getraw(:parametrisation, "hal"), default = :hal),
        enzyme_prefactor = _pb_bool(getraw(:enzyme_prefactor, true), default = true),
        default_inhibition = _pb_symbol(getraw(:default_inhibition, "complete"), default = :complete),
        default_activation = _pb_symbol(getraw(:default_activation, "complete"), default = :complete),
        model_name = String(getraw(:model_name, "outputname")),
        boundary_values = _pb_symbol(getraw(:boundary_values, "ignore"), default = :ignore),
        size_limit = Int(round(_pb_float(getraw(:size_limit, 300), default = 300))),
        raw = raw
    )
end

function _pb_add_issue!(rows, severity::String, component::String, message::String)
    push!(rows, (severity = severity, component = component, message = message))
    return rows
end

"""
    validate_pb_inputs(model, data; prior=nothing, options=nothing, strict=false)

Validate a reaction model, an observation table, or a matrix-based inference
payload before balancing. The result is a DataFrame with `severity`,
`component`, and `message` columns. With `strict=true`, any error row raises an
exception after the report is assembled.
"""
function validate_pb_inputs(model, data; prior = nothing, options = nothing, strict::Bool = false)
    prior = isnothing(prior) ? load_parameter_balancing_prior() : prior
    options = isnothing(options) ? load_parameter_balancing_options() : options
    issues = NamedTuple[]

    for required in ("standard chemical potential", "Michaelis constant", "catalytic rate constant geometric mean")
        haskey(prior.entries, required) ||
            _pb_add_issue!(issues, "warning", "prior", "Missing prior entry for '$required'.")
    end

    for (qtype, entry) in prior.entries
        entry.lower_bound < entry.upper_bound ||
            _pb_add_issue!(issues, "error", "prior", "Lower bound is not below upper bound for '$qtype'.")
        if entry.mathematical_type == :Multiplicative && isfinite(entry.lower_bound) && entry.lower_bound <= 0
            _pb_add_issue!(issues, "warning", "prior", "Multiplicative quantity '$qtype' has a nonpositive lower bound.")
        end
        if isnothing(entry.prior_std) && isnothing(entry.prior_geometric_std)
            _pb_add_issue!(issues, "warning", "prior", "No prior standard deviation is specified for '$qtype'.")
        end
    end

    if model isa AbstractVector && !isempty(model)
        if length(model) > options.size_limit
            _pb_add_issue!(issues, "warning", "model", "Reaction count $(length(model)) exceeds the configured size limit $(options.size_limit).")
        end
        for (name, rxn) in model
            try
                ns = num_substrates(rxn)
                ns > 6 && _pb_add_issue!(issues, "warning", "model", "Reaction '$name' has $ns participating stoichiometric terms.")
            catch err
                _pb_add_issue!(issues, "warning", "model", "Could not inspect reaction '$name': $(err).")
            end
        end
    end

    if data isa DataFrame
        df = _canonicalize_pb_columns(data)
        _pb_hascol(df, :quantity_type) ||
            _pb_add_issue!(issues, "error", "data", "Observation table lacks a QuantityType column.")
        if !(_pb_hascol(df, :mean) || _pb_hascol(df, :prior_median))
            _pb_add_issue!(issues, "error", "data", "Observation table lacks a Mean, Median, Value, or PriorMedian column.")
        end
        if _pb_hascol(df, :quantity_type)
            for row in eachrow(df)
                qtype = _pb_str(_pb_get(row, :quantity_type))
                isempty(qtype) && continue
                if !haskey(prior.entries, qtype)
                    _pb_add_issue!(issues, "warning", "data", "Unknown quantity type '$qtype'.")
                    continue
                end
                entry = prior.entries[qtype]
                if entry.mathematical_type == :Multiplicative && _pb_hascol(df, :mean)
                    val = _pb_float(_pb_get(row, :mean))
                    isfinite(val) && val <= 0 &&
                        _pb_add_issue!(issues, "error", "data", "Multiplicative quantity '$qtype' has nonpositive native mean $val.")
                end
            end
        end
    else
        Q = _pb_data_value(data, :Q)
        x = _pb_data_value(data, :x)
        Cx = _pb_data_value(data, :Cx)
        if !(isnothing(Q) || isnothing(x) || isnothing(Cx))
            size(Q, 1) == length(x) || _pb_add_issue!(issues, "error", "data", "Rows of Q do not match length(x).")
            size(Cx, 1) == size(Cx, 2) == length(x) || _pb_add_issue!(issues, "error", "data", "Cx must be square with size length(x).")
        else
            _pb_add_issue!(issues, "warning", "data", "Data are neither a quantity DataFrame nor a matrix payload containing Q, x, and Cx.")
        end
    end

    report = DataFrame(issues)
    if nrow(report) == 0
        report = DataFrame(severity = String[], component = String[], message = String[])
    end
    if strict && any(report.severity .== "error")
        error("Parameter-balancing input validation failed:\n$(report)")
    end
    return report
end

function _pb_table_row(qtype, reaction, compound, entry; source = "", is_pseudo = false)
    return (
        QuantityType = qtype,
        Reaction = reaction,
        Compound = compound,
        Mean = missing,
        Std = something(entry.data_std, entry.prior_std, missing),
        GeometricStd = something(entry.data_geometric_std, entry.prior_geometric_std, missing),
        Unit = entry.unit,
        Source = source,
        IsPseudo = is_pseudo
    )
end

"""
    make_empty_quantity_table(rxns; prior, options, include_derived=false)

Create an SBtab-like quantity table for the reactions and species represented by
`rxns`. The table is intended as a data-entry template for kinetic measurements.
"""
function make_empty_quantity_table(rxns; prior = load_parameter_balancing_prior(),
                                   options = load_parameter_balancing_options(),
                                   include_derived::Bool = false)
    rows = NamedTuple[]
    qtypes = include_derived ? sort(collect(keys(prior.entries))) : prior.basic_quantities
    species_names = String[]
    reaction_names = String[]
    binding_pairs = Tuple{String,String}[]
    try
        mu0_map, mur_map, mub_map = parameter_index_maps(rxns)
        species_names = sort(String[k for (k, _) in mu0_map])
        reaction_names = sort(String[k for (k, _) in mur_map])
        binding_pairs = sort(Tuple{String,String}[k for (k, _) in mub_map])
    catch
        reaction_names = sort(String[first(r) for r in rxns])
    end

    for qtype in qtypes
        entry = prior.entries[qtype]
        be = entry.biological_element
        if be in (:Species, :SpeciesReaction)
            for compound in species_names
                push!(rows, _pb_table_row(qtype, "", compound, entry))
            end
        elseif be in (:Reaction, :ReactionSpecies)
            for reaction in reaction_names
                push!(rows, _pb_table_row(qtype, reaction, "", entry))
            end
        elseif be in (:Reaction_Species, :ReactionSpecies, :SpeciesReaction)
            for (reaction, compound) in binding_pairs
                push!(rows, _pb_table_row(qtype, reaction, compound, entry))
            end
        else
            push!(rows, _pb_table_row(qtype, "", "", entry))
        end
    end
    return DataFrame(rows)
end

"""
    fill_pseudo_observations(data, prior; template=nothing, use_pseudo_values=true)

Append prior pseudo-observations for missing quantities. Existing rows are kept.
If `template` is supplied, its reaction/compound keys define the possible
quantity slots; otherwise the slots already present in `data` are used.
"""
function fill_pseudo_observations(data::DataFrame, prior::PBPrior = load_parameter_balancing_prior();
                                  template::Union{Nothing,DataFrame} = nothing,
                                  use_pseudo_values::Bool = true)
    out = _canonicalize_pb_columns(data)
    use_pseudo_values || return out

    tmpl = isnothing(template) ? out : _canonicalize_pb_columns(template)
    existing = Set(_pb_quantity_key(row) for row in eachrow(out))
    rows = NamedTuple[]
    for row in eachrow(tmpl)
        qtype = _pb_str(_pb_get(row, :quantity_type))
        isempty(qtype) && continue
        haskey(prior.entries, qtype) || continue
        key = _pb_quantity_key(row)
        key in existing && continue
        entry = prior.entries[qtype]
        val = entry.prior_median
        push!(rows, (
            quantity_type = qtype,
            reaction = _pb_str(_pb_get(row, :reaction)),
            compound = _pb_str(_pb_get(row, :compound)),
            mean = val,
            std = something(entry.data_std, entry.prior_std, missing),
            geometric_std = something(entry.data_geometric_std, entry.prior_geometric_std, missing),
            unit = entry.unit,
            source = "pseudo_prior",
            is_pseudo = true
        ))
        push!(existing, key)
    end
    isempty(rows) && return out
    return vcat(out, DataFrame(rows); cols = :union)
end

function _pb_row_cov(samples::AbstractMatrix)
    size(samples, 2) > 1 || return zeros(Float64, size(samples, 1), size(samples, 1))
    return Matrix(cov(Matrix(samples'); dims = 1))
end

function _pb_vector_or_default(x, n, prefix)
    isnothing(x) && return ["$prefix$(i)" for i in 1:n]
    return String.(collect(x))
end

"""
    balance_parameters(data, [prior]; model=:gaussian, robust=:none, hierarchical=false)

High-level balancing wrapper. Matrix payloads with `Q`, `x`, `Cx`, `q_prior`,
and `C_prior` use the closed-form Gaussian update. Raw fold payloads with
`A0`, `b0`, `Arows`, `y`, `sigma2`, `grp`, and `eb` call the MCEM/structured-VI
core and return a common `BalanceResult`.
"""
function balance_parameters(data, prior::Union{PBPrior,Nothing} = nothing;
                            model::Symbol = :gaussian,
                            robust::Symbol = :none,
                            hierarchical::Bool = false,
                            rng::AbstractRNG = MersenneTwister(1),
                            method::Symbol = :collapsed,
                            spec::Union{FoldFitSpec,Nothing} = nothing,
                            kwargs...)
    prior_obj = isnothing(prior) ? load_parameter_balancing_prior() : prior
    Q = _pb_data_value(data, :Q)
    x = _pb_data_value(data, :x)
    Cx = _pb_data_value(data, :Cx)
    q_prior = _pb_data_value(data, :q_prior)
    C_prior = _pb_data_value(data, :C_prior)

    if !(isnothing(Q) || isnothing(x) || isnothing(Cx) || isnothing(q_prior) || isnothing(C_prior))
        q_post, C_post = infer_parameters(Q, x, Cx, q_prior, C_prior)
        pname = _pb_vector_or_default(_pb_data_value(data, :parameter_names), length(q_post), "q")
        oname = _pb_vector_or_default(_pb_data_value(data, :observable_names), size(Q, 1), "x")
        qtype = String.(collect(something(_pb_data_value(data, :quantity_types), fill("", length(q_post)))))
        units = String.(collect(something(_pb_data_value(data, :units), fill("", length(q_post)))))
        return BalanceResult(
            q_mean = collect(q_post),
            q_cov = Matrix(C_post),
            observable_mean = collect(Q * q_post),
            observable_cov = Matrix(Q * C_post * transpose(Q)),
            parameter_names = pname,
            observable_names = oname,
            quantity_types = qtype,
            units = units,
            prior = prior_obj,
            metadata = Dict{Symbol,Any}(
                :inference_method => :gaussian_closed_form,
                :Q => Matrix(Q),
                :x => collect(x),
                :Cx => Matrix(Cx),
                :q_prior => collect(q_prior),
                :C_prior => Matrix(C_prior)
            )
        )
    end

    A0 = _pb_data_value(data, :A0)
    b0 = _pb_data_value(data, :b0)
    Arows = _pb_data_value(data, :Arows)
    y = _pb_data_value(data, :y)
    sigma2 = _pb_data_value(data, :sigma2)
    isnothing(sigma2) && (sigma2 = _pb_data_value(data, :σ2))
    grp = _pb_data_value(data, :grp)
    eb = _pb_data_value(data, :eb)
    if any(isnothing, (A0, b0, Arows, y, sigma2, grp, eb))
        error("balance_parameters requires either a Gaussian matrix payload or a fold payload containing A0, b0, Arows, y, sigma2, grp, and eb.")
    end

    fit_model = robust in (:student_t, :t) ? :t :
                robust in (:skew, :skew_normal) ? :skew :
                model
    spec = isnothing(spec) ? FoldFitSpec(model = fit_model, hierarchical = hierarchical) : spec
    hs = init_hyper(eb; λ0 = spec.λ_init, τ²0 = spec.τ²_init, π0 = spec.π_init)
    fit = fit_fold_mcem(rng, A0, b0, Matrix(Arows), collect(Float64, y),
                        collect(Float64, sigma2), String.(grp), eb, hs, spec;
                        method = method)
    q_mean = vec(mean(fit.qsamps; dims = 2))
    q_cov = _pb_row_cov(fit.qsamps)
    pname = _pb_vector_or_default(_pb_data_value(data, :parameter_names), length(q_mean), "q")
    return BalanceResult(
        q_mean = q_mean,
        q_cov = q_cov,
        q_samples = fit.qsamps,
        parameter_names = pname,
        quantity_types = String.(collect(something(_pb_data_value(data, :quantity_types), fill("", length(q_mean))))),
        units = String.(collect(something(_pb_data_value(data, :units), fill("", length(q_mean))))),
        prior = prior_obj,
        metadata = Dict{Symbol,Any}(:inference_method => method, :fold_fit => fit, :spec => spec, :eb => eb)
    )
end

_pb_z_for_level(level) = quantile(Normal(), 0.5 + level / 2)

"""
    posterior_observable_summary(result, Q; levels=[0.68, 0.95])

Propagate a balanced posterior through an observable matrix and return posterior
means, standard deviations, and interval endpoints.
"""
function posterior_observable_summary(result::BalanceResult, Q; levels = [0.68, 0.95],
                                      observable_names = nothing,
                                      quantity_types = nothing,
                                      units = nothing)
    mu = collect(Q * result.q_mean)
    cov_obs = Matrix(Q * result.q_cov * transpose(Q))
    sd = sqrt.(max.(diag(cov_obs), 0.0))
    names_obs = _pb_vector_or_default(observable_names, length(mu), "x")
    df = DataFrame(name = names_obs,
                   quantity_type = String.(collect(something(quantity_types, fill("", length(mu))))),
                   unit = String.(collect(something(units, fill("", length(mu))))),
                   mean = mu,
                   sd = sd)
    ysamps = isnothing(result.q_samples) ? nothing : Matrix(Q * result.q_samples)
    for level in levels
        suffix = replace(string(round(level * 100; digits = 1)), "." => "_")
        if isnothing(ysamps)
            z = _pb_z_for_level(level)
            df[!, Symbol("lo_$suffix")] = mu .- z .* sd
            df[!, Symbol("hi_$suffix")] = mu .+ z .* sd
        else
            alpha = (1 - level) / 2
            df[!, Symbol("lo_$suffix")] = [quantile(view(ysamps, i, :), alpha) for i in axes(ysamps, 1)]
            df[!, Symbol("hi_$suffix")] = [quantile(view(ysamps, i, :), 1 - alpha) for i in axes(ysamps, 1)]
        end
    end
    return df
end

function _pb_parameter_summary(result::BalanceResult; levels = [0.68, 0.95])
    n = length(result.q_mean)
    names_par = isempty(result.parameter_names) ? ["q$(i)" for i in 1:n] : result.parameter_names
    qtypes = isempty(result.quantity_types) ? fill("", n) : result.quantity_types
    units = isempty(result.units) ? fill("", n) : result.units
    sd = sqrt.(max.(diag(result.q_cov), 0.0))
    df = DataFrame(name = names_par, quantity_type = qtypes, unit = units,
                   mean = result.q_mean, sd = sd)
    for level in levels
        suffix = replace(string(round(level * 100; digits = 1)), "." => "_")
        if isnothing(result.q_samples)
            z = _pb_z_for_level(level)
            df[!, Symbol("lo_$suffix")] = result.q_mean .- z .* sd
            df[!, Symbol("hi_$suffix")] = result.q_mean .+ z .* sd
        else
            alpha = (1 - level) / 2
            df[!, Symbol("lo_$suffix")] = [quantile(view(result.q_samples, i, :), alpha) for i in 1:n]
            df[!, Symbol("hi_$suffix")] = [quantile(view(result.q_samples, i, :), 1 - alpha) for i in 1:n]
        end
    end
    return df
end

function _pb_sbtab_string(df::DataFrame)
    io = IOBuffer()
    println(io, "!!SBtab TableType='Quantity' TableName='BalancedParameters'")
    println(io, join("!" .* names(df), '\t'))
    CSV.write(io, df; delim = '\t', header = false)
    return String(take!(io))
end

"""
    export_balanced_table(result; format=:dataframe, path=nothing, Q=nothing)

Export a posterior summary as a `DataFrame`, CSV/TSV file, or SBtab-like TSV
string. If `Q` is supplied, the exported table is in observable space.
"""
function export_balanced_table(result::BalanceResult; format::Symbol = :dataframe,
                               path::Union{Nothing,AbstractString} = nothing,
                               Q = nothing,
                               levels = [0.68, 0.95])
    df = isnothing(Q) ? _pb_parameter_summary(result; levels = levels) :
         posterior_observable_summary(result, Q; levels = levels,
                                      observable_names = result.observable_names)
    if format == :dataframe
        return df
    elseif format == :csv
        isnothing(path) || CSV.write(path, df)
        return df
    elseif format == :tsv
        isnothing(path) || CSV.write(path, df; delim = '\t')
        return df
    elseif format == :sbtab
        txt = _pb_sbtab_string(df)
        if !isnothing(path)
            open(path, "w") do io
                write(io, txt)
            end
        end
        return txt
    else
        error("Unknown export format $format; use :dataframe, :csv, :tsv, or :sbtab.")
    end
end

function _pb_gaussian_entropy(C::AbstractMatrix)
    n = size(C, 1)
    sgn, logdet = logabsdet(Symmetric(C))
    sgn > 0 || return NaN
    return 0.5 * (n * (1 + log(2pi)) + logdet)
end

function _pb_prior_cov_from_result(result::BalanceResult, prior::PBPrior)
    haskey(result.metadata, :C_prior) && return Matrix(result.metadata[:C_prior])
    n = length(result.q_mean)
    vars = fill(NaN, n)
    for i in 1:n
        qtype = i <= length(result.quantity_types) ? result.quantity_types[i] : ""
        if haskey(prior.entries, qtype)
            entry = prior.entries[qtype]
            sd = !isnothing(entry.prior_std) ? entry.prior_std :
                 (!isnothing(entry.prior_geometric_std) ? log(entry.prior_geometric_std) : nothing)
            !isnothing(sd) && (vars[i] = sd^2)
        end
    end
    all(isfinite, vars) || return nothing
    return Diagonal(vars)
end

"""
    information_gain(result, prior)

Compute Gaussian entropy reduction from prior to posterior. Returns one total
row and per-parameter marginal entropy reductions when prior variances are
available.
"""
function information_gain(result::BalanceResult, prior::Union{PBPrior,Nothing} = result.prior)
    isnothing(prior) && (prior = load_parameter_balancing_prior())
    C0 = _pb_prior_cov_from_result(result, prior)
    isnothing(C0) && error("Prior covariance is unavailable; supply C_prior in result metadata or quantity types matching the prior table.")
    C1 = result.q_cov
    rows = NamedTuple[(name = "total",
                       prior_entropy = _pb_gaussian_entropy(Matrix(C0)),
                       posterior_entropy = _pb_gaussian_entropy(C1),
                       information_gain = _pb_gaussian_entropy(Matrix(C0)) - _pb_gaussian_entropy(C1),
                       sd_ratio = NaN)]
    names_par = isempty(result.parameter_names) ? ["q$(i)" for i in 1:length(result.q_mean)] : result.parameter_names
    for i in eachindex(result.q_mean)
        v0 = C0[i, i]
        v1 = C1[i, i]
        if v0 > 0 && v1 > 0
            h0 = 0.5 * log(2pi * exp(1) * v0)
            h1 = 0.5 * log(2pi * exp(1) * v1)
            push!(rows, (name = names_par[i], prior_entropy = h0,
                         posterior_entropy = h1, information_gain = h0 - h1,
                         sd_ratio = sqrt(v1 / v0)))
        end
    end
    return DataFrame(rows)
end

"""
    check_bounds(result, prior; scale=:native, level=0.95)

Compare posterior means and intervals with prior-table lower/upper bounds.
For multiplicative quantities on the native scale, log posterior summaries are
exponentiated before comparison.
"""
function check_bounds(result::BalanceResult, prior::PBPrior = load_parameter_balancing_prior();
                      scale::Symbol = :native, level::Float64 = 0.95)
    n = length(result.q_mean)
    names_par = isempty(result.parameter_names) ? ["q$(i)" for i in 1:n] : result.parameter_names
    qtypes = isempty(result.quantity_types) ? fill("", n) : result.quantity_types
    sd = sqrt.(max.(diag(result.q_cov), 0.0))
    z = _pb_z_for_level(level)
    rows = NamedTuple[]
    for i in 1:n
        qtype = qtypes[i]
        haskey(prior.entries, qtype) || continue
        entry = prior.entries[qtype]
        mean_i = result.q_mean[i]
        lo = mean_i - z * sd[i]
        hi = mean_i + z * sd[i]
        lower = entry.lower_bound
        upper = entry.upper_bound
        if entry.mathematical_type == :Multiplicative && scale == :native
            mean_i, lo, hi = exp(mean_i), exp(lo), exp(hi)
        elseif entry.mathematical_type == :Multiplicative && scale == :log
            lower = lower > 0 ? log(lower) : -Inf
            upper = upper > 0 ? log(upper) : Inf
        end
        push!(rows, (
            name = names_par[i],
            quantity_type = qtype,
            mean = mean_i,
            lower_interval = lo,
            upper_interval = hi,
            lower_bound = lower,
            upper_bound = upper,
            mean_below = isfinite(lower) && mean_i < lower,
            mean_above = isfinite(upper) && mean_i > upper,
            interval_below = isfinite(lower) && lo < lower,
            interval_above = isfinite(upper) && hi > upper
        ))
    end
    return DataFrame(rows)
end

"""
    kineticize_sbml(model, balanced_table; parametrisation=:hal, output_path=nothing)

Prepare an SBML-compatible parameter sidecar for downstream kineticization. This
function does not synthesize complete kinetic laws; it copies the SBML file when
`output_path` is supplied and writes a balanced-parameter CSV sidecar that can
be consumed by SBML or kinetic-law tooling.
"""
function kineticize_sbml(model, balanced_table; parametrisation::Symbol = :hal,
                         output_path::Union{Nothing,AbstractString} = nothing,
                         overwrite_existing::Bool = false,
                         parameter_table_path::Union{Nothing,AbstractString} = nothing)
    df = balanced_table isa BalanceResult ? export_balanced_table(balanced_table; format = :dataframe) :
         DataFrame(balanced_table)
    sbml_path = nothing
    if model isa AbstractString
        if isnothing(output_path)
            sbml_path = model
        else
            if isfile(output_path) && !overwrite_existing
                error("Output path exists: $output_path. Pass overwrite_existing=true to replace it.")
            end
            cp(model, output_path; force = overwrite_existing)
            sbml_path = output_path
        end
    end
    if isnothing(parameter_table_path)
        parameter_table_path = isnothing(sbml_path) ? "balanced_parameters.csv" :
            replace(String(sbml_path), r"\.[^.]*$" => "_balanced_parameters.csv")
    end
    CSV.write(parameter_table_path, df)
    return (
        sbml_path = sbml_path,
        parameter_table_path = parameter_table_path,
        status = :parameter_export_only,
        parametrisation = parametrisation,
        message = "Balanced parameters were exported as a sidecar table; complete kinetic-law synthesis is intentionally left to downstream SBML tooling."
    )
end
