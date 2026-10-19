using SBMLToolkit, SBML, DataFrames, ModelingToolkit, Catalyst
@parameters t
extract_name(x) = string(tosymbol(x,escape=false))

"""
    load_yeast_reconstruction(file = joinpath("data", "yeast-GEM.xml"))

Load a yeast genome-scale metabolic reconstruction from an SBML file.

# Arguments
- `file`: Path to the SBML model file.

# Returns
An `SBML.Model` object with local parameters promoted and functions expanded.
"""
function load_yeast_reconstruction(file = joinpath("data", "yeast-GEM.xml"))
    @assert SBMLToolkit.checksupport_file(file)
    mdl = readSBML(
        file, doc -> begin
            set_level_and_version(3, 1)(doc)
            convert_promotelocals_expandfuns(doc)
        end
    )
    return mdl
end

nothingtonan(x) = isnothing(x) ? NaN : x

"""
    species_data(mdl)

Return a `DataFrame` of all species in the SBML model.

Each row corresponds to one species and contains its symbol, name, compartment,
boundary condition, chemical formula, charge, initial concentration, initial
amount, constant flag, and metaid. Missing numeric values are stored as `NaN`.
"""
function species_data(mdl)
    df = DataFrame(
        symbol=String[], name=String[], compartment = Symbol[], boundary_condition = Bool[],
        formula = Union{Nothing,String}[], charge = Int[], initial_concentration = Float64[],
        initial_amount = Float64[], constant = Bool[], metaid = String[]
    )
    for (k,v) in mdl.species
        push!(df,
            [k, v.name, Symbol(v.compartment), v.boundary_condition, v.formula, v.charge,
            nothingtonan(v.initial_concentration), nothingtonan(v.initial_amount), v.constant, v.metaid]
        )
    end
    return df
end

"""
    reaction_data(mdl)

Return a `DataFrame` of all reactions in the SBML model.

Each row contains the reaction symbol, name, reactant references, product
references, metaid, and SBO term.
"""
function reaction_data(mdl)
    df = DataFrame(
        symbol = String[], name = String[], reactants = Vector{SBML.SpeciesReference}[], products = Vector{SBML.SpeciesReference}[],
        metaid = Union{Nothing,String}[], sbo = Union{Nothing,String}[]
    )
    for (k,v) in mdl.reactions
        push!(df, [k, v.name, v.reactants, v.products, v.metaid, v.sbo])
    end
    return df
end

"""
    species_data_dict(mdl)

Return a `Dict` mapping each reaction symbol to its non-symbol fields from `reaction_data`.

Note: this function calls `reaction_data` internally; use `species_data` to
obtain species information.
"""
function species_data_dict(mdl)
    df = reaction_data(mdl)
    return Dict(row[1] => row[2:end] for row in eachrow(df))
end

"""
    reaction_data_dict(mdl)

Return a `Dict` mapping each reaction symbol to its non-symbol fields from `reaction_data`.
"""
function reaction_data_dict(mdl)
    df = reaction_data(mdl)
    return Dict(row[1] => row[2:end] for row in eachrow(df))
end

mtk_symbol(x) = only(@variables($(x)(t)))
species_symbol(x) = only(@species($(x)(t)))

"""
    generate_reaction_object(reactants, products; protons=true, proton_name="h")

Construct a `Catalyst.Reaction` from lists of `SBML.SpeciesReference` objects.

# Arguments
- `reactants`: Vector of `SBML.SpeciesReference` for the reactant side.
- `products`: Vector of `SBML.SpeciesReference` for the product side.

# Keywords
- `protons`: If `false`, species whose name matches `proton_name` are excluded.
- `proton_name`: Name used to identify proton species (default `"h"`).

# Returns
A `Catalyst.Reaction` with rate constant 1 and the given stoichiometries.
"""
function generate_reaction_object(reactants, products; protons=true, proton_name="h")
    @parameters t
    include_species(sref) = (protons == true || sref.species != proton_name)
    reactant_names = [sref.species for sref in reactants if include_species(sref)]
    reactant_vars = species_symbol.(Symbol.(reactant_names))
    reactant_stoichiometries = [sref.stoichiometry for sref in reactants if include_species(sref)]
    product_names = [Symbol(sref.species) for sref in products if include_species(sref)]
    procut_vars = species_symbol.(Symbol.(product_names))
    product_stoichiometries = [sref.stoichiometry for sref in products if include_species(sref)]
    return Catalyst.Reaction(
        1, reactant_vars, procut_vars, reactant_stoichiometries, product_stoichiometries
    )
end

"""
    generate_reactions(rxn_list, dict; protons=true, proton_name="h")

Build a vector of `(name, Reaction)` tuples from a list of reaction names and a data dict.

# Arguments
- `rxn_list`: Iterable of reaction name keys present in `dict`.
- `dict`: Dict mapping reaction names to structs with `.reactants` and `.products` fields.

# Keywords
- `protons`: Passed through to `generate_reaction_object`.
- `proton_name`: Passed through to `generate_reaction_object`.

# Returns
Vector of `(name, Catalyst.Reaction)` tuples, one per entry in `rxn_list`.
"""
generate_reactions(rxn_list,dict; protons=true, proton_name="h") = [
    (r, generate_reaction_object(dict[r].reactants, dict[r].products, protons=protons, proton_name=proton_name))
    for r in rxn_list
]

"""
    contraction_matrices(rxns, identical_compounds=[])

Build projection matrices that fold identical compounds into a single representative.

When multiple species in `identical_compounds` refer to the same physical compound
(e.g., the same metabolite in different compartments that should share a parameter),
the matrices `L`, `R`, and `Lx` collapse the duplicates in observation space,
parameter space, and the left-side of the parameter vector respectively.

# Arguments
- `rxns`: Vector of `(name, Reaction)` tuples, as returned by `generate_reactions`.
- `identical_compounds`: Vector of compound-name groups; each inner vector lists
  species that are treated as identical.

# Returns
Tuple `(L, R, Lx)` of sparse integer matrices.
- `L`: Left projection onto the reduced observation vector (removes duplicate rows).
- `R`: Right expansion that sums duplicate parameter columns into the first index.
- `Lx`: Left projection onto the reduced parameter vector (removes duplicate entries).
"""
function contraction_matrices(rxns::Vector{<:Tuple}, identical_compounds=[])
    (array_species, _) = rn_entities(rxns)
    idx_compounds = [[findfirst(s -> s == c, array_species) for c in cs] for cs in identical_compounds]
    Q = observation_matrix(rxns)
    (ny,nx) = size(Q)

    I_left = Matrix{Int}(I, ny, ny)
    idx_remove = vcat([idx[2:end] for idx in idx_compounds]...)
    idx_keep = [i for i in 1:ny if i ∉ idx_remove]
    L = sparse(I_left[idx_keep,:])

    I_right = Matrix{Int}(I, nx, nx)
    for idxs in idx_compounds
        i_sum = idxs[1]
        I_right[:,i_sum] = sum(I_right[:,idxs], dims=2)
    end
    idx_keep = [i for i in 1:nx if i ∉ idx_remove]
    R = sparse(I_right[:,idx_keep])

    I_left = Matrix{Int}(I, nx, nx)
    idx_remove = vcat([idx[2:end] for idx in idx_compounds]...)
    idx_keep = [i for i in 1:nx if i ∉ idx_remove]
    Lx = sparse(I_left[idx_keep,:])

    return (L,R,Lx)
end

"""
    reaction_network(rxns)

Construct a `Catalyst.ReactionSystem` from a vector of `(name, Reaction)` tuples.
"""
reaction_network(rxns::Vector{<:Tuple}) = ReactionSystem([r[2] for r in rxns], t, name=:__)

"""
    rn_entities(rxns)

Return `(array_species, array_reactions)` string name vectors for the reaction network.

# Arguments
- `rxns`: Vector of `(name, Reaction)` tuples.

# Returns
Tuple of `(array_species, array_reactions)` where each is a `Vector{String}` in the
order used by Catalyst's stoichiometry matrices.
"""
function rn_entities(rxns::Vector{<:Tuple})
    rn = reaction_network(rxns)
    array_reactions = [r[1] for r in rxns]
    array_species = [extract_name(s) for s in species(rn)]
    return (array_species, array_reactions)
end

"""
    observation_matrix_index_maps(rxns)

Return bijective index maps from observable names to rows of the observation matrix `Q`.

# Arguments
- `rxns`: Vector of `(name, Reaction)` tuples.

# Returns
Tuple `(μ0_map, KM_map, kcatfmap, kcatrmap)` of `Bijection` objects mapping:
- `μ0_map`: species name mapped to the row index of its standard chemical potential.
- `KM_map`: `(reaction, species)` mapped to the row index of its binding potential.
- `kcatfmap`: reaction name mapped to the row index of its forward kcat.
- `kcatrmap`: reaction name mapped to the row index of its reverse kcat.
"""
function observation_matrix_index_maps(rxns::Vector{<:Tuple})
    rn = reaction_network(rxns)
    (array_species, array_reactions) = rn_entities(rxns)
    n_spec = length(array_species)
    n_rxn = length(array_reactions)

    KM_map = binding_potential_map(rxns)
    n_KM = length(KM_map)

    μ0_map = Bijection(Dict(array_species .=> 1:n_spec))
    KM_map = Bijection(Dict(k => v + n_spec for (k,v) in KM_map))
    kcatfmap = Bijection(Dict(array_reactions .=> (1:n_rxn) .+ n_spec .+ n_KM))
    kcatrmap = Bijection(Dict(array_reactions .=> (1:n_rxn) .+ n_spec .+ n_KM .+ n_rxn))
    return (μ0_map, KM_map, kcatfmap, kcatrmap)
end

"""
    binding_potential_map(rxns)

Return a `Dict` mapping each `(reaction_name, species_name)` pair to its KM row index.

The index is assigned in column-major order over the binary net-stoichiometry matrix,
matching the ordering used in `KM_link_matrix`.

# Arguments
- `rxns`: Vector of `(name, Reaction)` tuples.

# Returns
`Dict{Tuple{String,String}, Int}` from `(reaction, species)` to 1-based row index.
"""
function binding_potential_map(rxns)
    rn = reaction_network(rxns)
    (array_species, array_reactions) = rn_entities(rxns)
    μb_map = Dict{Tuple{String,String},Int}()
    Nt = abs.(netstoichmat(rn)) .> 0
    row = 0
    for i_reac in axes(Nt,2), i_spec in axes(Nt,1)
        if Nt[i_spec,i_reac]
            row += 1
            μb_map[(array_reactions[i_reac],array_species[i_spec])] = row
        end
    end
    return μb_map
end

"""
    parameter_index_maps(rxns)

Return bijective index maps from parameter names to positions in the independent parameter vector.

# Arguments
- `rxns`: Vector of `(name, Reaction)` tuples.

# Returns
Tuple `(μ0_map, μr_map, μb_map)` of `Bijection` objects:
- `μ0_map`: species name mapped to the index of its standard chemical potential `μ0`.
- `μr_map`: reaction name mapped to the index of its reaction potential `μr`.
- `μb_map`: `(reaction, species)` mapped to the index of its binding potential `μb`.
"""
function parameter_index_maps(rxns::Vector{<:Tuple})
    (array_species, array_reactions) = rn_entities(rxns)
    n_spec = length(array_species)
    n_rxn = length(array_reactions)
    μb_map = binding_potential_map(rxns)

    μ0_map = Bijection(Dict(array_species .=> 1:n_spec))
    μr_map = Bijection(Dict(array_reactions .=> (1:n_rxn) .+ n_spec))
    μb_map = Bijection(Dict(k => v  + n_spec + n_rxn for (k,v) in μb_map))
    return (μ0_map, μr_map, μb_map)
end

"""
    thermodynamic_prior(rxns, h2o_strings=[], identical_compounds=[]; kwargs...)

Compute the prior mean and covariance for independent parameters from a reaction list.

Builds the observation matrix, applies contraction for identical compounds, then
propagates Gaussian priors on standard chemical potentials, binding potentials, and
velocity potentials back to the independent parameter space via the inverse observation map.

# Arguments
- `rxns`: Vector of `(name, Reaction)` tuples.
- `h2o_strings`: Species names to treat as water (their KM dependence is zeroed out).
- `identical_compounds`: Groups of species names to collapse into one representative.

# Keywords
- `μ0_mean`, `μ0_std`: Prior mean and std for standard chemical potentials (in RT units).
- `log_KM_mean`, `log_KM_std`: Prior mean and std for log KM values.
- `μv_mean`, `μv_std`: Prior mean and std for log velocity potentials.

# Returns
Tuple `(q_prior, C_prior)`: prior mean vector and covariance matrix in parameter space.
"""
function thermodynamic_prior(rxns::Vector{<:Tuple}, h2o_strings=[], identical_compounds=[];
    μ0_mean=-8.8e5/(8.314*298.15),μ0_std=6.8e5/(8.314*298.15),log_KM_mean=log(1e-4),log_KM_std=log(10),μv_mean=log(10),μv_std=log(10))
    Q = observation_matrix(rxns, h2o_strings=h2o_strings)
    L,R,Lx = contraction_matrices(rxns, identical_compounds)
    Q_compact = L*Q*R
    n_reac = length(rxns)
    Q_map = Q_compact[1:end-n_reac,:]
    Q_map[end-n_reac+1:end,:] = (Q_map[end-n_reac+1:end,:] + Q_compact[end-n_reac+1:end,:])/2
    @assert size(Q_map,1) == size(Q_map,2)

    (array_species, array_reactions) = rn_entities(rxns)
    n_species = length(array_species) + size(L,1) - size(L,2)
    n_KM = size(Q_map,1) - n_species - n_reac

    db_mean = [
        μ0_mean*ones(n_species);
        log_KM_mean*ones(n_KM);
        μv_mean*ones(n_reac)
    ]
    db_cov = BlockDiagonal([
        μ0_std^2*I(n_species),
        log_KM_std^2*I(n_KM),
        μv_std^2*I(n_reac)
    ])

    Q_inv = inv(Q_map)
    q_prior = Q_inv*db_mean
    C_prior = Q_inv*db_cov*transpose(Q_inv)

    return q_prior,C_prior
end

"""
    thermodynamic_prior(rn; kwargs...)

Compute the prior mean and covariance directly from a `ReactionSystem`.

Propagates independent Gaussian priors on `μ0`, `μb`, and `μa` through the
stoichiometric relationships to produce a joint prior over all parameters.

# Arguments
- `rn`: A `Catalyst.ReactionSystem`.

# Keywords
- `μ0_mean`, `μ0_std`: Prior mean and std for standard chemical potentials.
- `log_KM_mean`, `log_KM_std`: Prior mean and std for log KM values.
- `μv_mean`, `μv_std`: Prior mean and std for log velocity potentials.

# Returns
Tuple `(q, C)`: prior mean vector and full joint covariance matrix ordered
as `[μ0; μa; μb]`.
"""
function thermodynamic_prior(rn;
    μ0_mean=-8.8e5/(8.314*298.15),μ0_std=6.8e5/(8.314*298.15),log_KM_mean=log(1e-4),log_KM_std=log(10),μv_mean=log(10),μv_std=log(10))
    N = netstoichmat(rn)
    NM = KM_link_matrix(rn)
    (NKMf, NKMr) = KM_stoichiometry(rn)

    n_spec = size(N,1)
    n_rxn = size(N,2)
    n_KM = size(NM,1)

    μ0_mean_vec = μ0_mean*ones(n_spec)
    μb_mean = log_KM_mean*ones(n_KM) + NM*μ0_mean_vec
    μa_mean = μv_mean*ones(n_rxn) - 1/2*(NKMf + NKMr)*μb_mean

    cov_μ0_μ0 = μ0_std^2 * I
    cov_μb_μb = log_KM_std^2*I + NM*cov_μ0_μ0*transpose(NM)
    cov_μa_μa = μv_std^2*I + 1/4*(NKMf + NKMr)*cov_μb_μb*transpose(NKMf + NKMr)
    cov_μb_μ0 = NM*cov_μ0_μ0
    cov_μa_μ0 = -1/2*(NKMf + NKMr)*NM*cov_μ0_μ0
    cov_μa_μb = -1/2*(NKMf + NKMr)*cov_μb_μb

    q = [μ0_mean_vec;μa_mean;μb_mean]
    C = [
        cov_μ0_μ0 transpose(cov_μa_μ0) transpose(cov_μb_μ0);
        cov_μa_μ0 cov_μa_μa cov_μa_μb;
        cov_μb_μ0 transpose(cov_μa_μb) cov_μb_μb
    ]

    return (q,C)
end

"""
    infer_parameters(Q, x, Cx, q_prior, C_prior)

Solve the thermodynamic parameter balancing problem via Gaussian conjugate update.

Given observations `x` related to parameters `q` by `x ≈ Q*q` with noise covariance
`Cx`, and a Gaussian prior `(q_prior, C_prior)`, returns the posterior mean and
covariance.

# Arguments
- `Q`: Observation matrix mapping parameters to observables.
- `x`: Observed values vector.
- `Cx`: Observation noise covariance matrix.
- `q_prior`: Prior mean of the parameter vector.
- `C_prior`: Prior covariance of the parameter vector.

# Returns
Tuple `(q_post, C_post)`: posterior mean and covariance.
"""
function infer_parameters(Q,x,Cx,q_prior,C_prior)
    C_post = inv(inv(C_prior) + transpose(Q)*(Cx\Q))
    q_post = C_post*(transpose(Q)*(Cx\x) + C_prior\q_prior)
    return (q_post,C_post)
end

function _infer_parameters(Q,x,Cx,q_prior,C_prior)
    C_post = inv(inv(C_prior) + transpose(Q)*inv(Cx)*Q)
    q_post = C_post*(transpose(Q)*inv(Cx)*x + inv(C_prior)*q_prior)
    return (q_post,C_post)
end

"""
    load_equilibrator_data(equilibrator_data_path)

Load eQuilibrator thermodynamic data from a JSON file and convert to RT units.

# Arguments
- `equilibrator_data_path`: Path to the JSON file produced by eQuilibrator.

# Returns
Tuple `(eq_metabolite_names, ΔGf_mu, ΔGf_cov)`:
- `eq_metabolite_names`: `Vector{String}` of metabolite names.
- `ΔGf_mu`: Vector of mean formation free energies divided by RT.
- `ΔGf_cov`: Matrix of formation free energy covariances divided by RT².
"""
function load_equilibrator_data(equilibrator_data_path)
    # Load data into Julia
    dict = JSON.parsefile(equilibrator_data_path)
    eq_metabolite_names = string.(dict["metabolite_names"])
    RT = 8.314e-3 * 298.15
    ΔGf_mu = float.(dict["dgf_prime_mu"])/RT
    ΔGf_cov = float.(hcat(dict["dgf_prime_cov"]...))/(RT^2)
    return (eq_metabolite_names, ΔGf_mu, ΔGf_cov)
end

"""
    load_BRENDA_data(kcat_data_path, KM_data_path)

Load BRENDA kcat and KM data from CSV files and return median log-values by EC–substrate key.

Negative values are removed before aggregation. KM values are converted from mM to M
(divided by 1000) before taking the log.

# Arguments
- `kcat_data_path`: Path to a CSV file with columns `ec_code`, `substrate`, `kcat`.
- `KM_data_path`: Path to a CSV file with columns `ec_code`, `substrate`, `KM`.

# Returns
Tuple `(dict_kcat_medians, dict_KM_medians)`:
- `dict_kcat_medians`: `Dict` from `(ec_code, substrate)` to median log kcat.
- `dict_KM_medians`: `Dict` from `(ec_code, substrate)` to median log KM (in M).
"""
function load_BRENDA_data(kcat_data_path,KM_data_path)
    df_kcat = DataFrame(CSV.File(kcat_data_path))
    filter!(r -> r.kcat >= 0, df_kcat) # Remove negative values
    df_kcat.ec_substrate = [(r.ec_code, r.substrate) for r in eachrow(df_kcat)]
    df_kcat.log_kcat = log.(df_kcat.kcat)
    df_kcat_groups = groupby(df_kcat, :ec_substrate)
    df_kcat_medians = combine(df_kcat_groups, :log_kcat => median)
    dict_kcat_medians = Dict(df_kcat_medians.ec_substrate .=> df_kcat_medians.log_kcat_median)

    df_KM = DataFrame(CSV.File(KM_data_path))
    filter!(r -> r.KM >= 0, df_KM) # Remove negative values
    df_KM.ec_substrate = [(r.ec_code, r.substrate) for r in eachrow(df_KM)]
    df_KM.log_KM = log.(df_KM.KM/1000)
    df_KM_groups = groupby(df_KM, :ec_substrate)
    df_KM_medians = combine(df_KM_groups, :log_KM => median)
    dict_KM_medians = Dict(df_KM_medians.ec_substrate .=> df_KM_medians.log_KM_median)

    return (dict_kcat_medians, dict_KM_medians)
end

function print_observation_eqs(rxns; h2o_strings=[])
    Q = observation_matrix(rxns, h2o_strings=h2o_strings)
    (μ0_map, KM_map, kcatfmap, kcatrmap) = observation_matrix_index_maps(rxns)
    (bg_μ0_map, bg_μr_map, bg_μb_map) = parameter_index_maps(rxns)
    for i in axes(Q,1)
        row = Q[i,:]
        if i in 1:length(μ0_map)
            print("μ0[$(μ0_map(i))] = ")
        elseif i in length(μ0_map)+1:length(μ0_map)+length(KM_map)
            print("KM[$(KM_map(i))] = ")
        elseif i in length(μ0_map)+length(KM_map)+1:length(μ0_map)+length(KM_map)+length(kcatfmap)
            print("kcatf[$(kcatfmap(i))] = ")
        else
            print("kcatr[$(kcatrmap(i))] = ")
        end
        for (j,v) in enumerate(row)
            if v != 0
                if j in 1:length(bg_μ0_map)
                    print("$v*μ0[$(bg_μ0_map(j))] ")
                elseif j in length(bg_μ0_map)+1:length(bg_μ0_map)+length(bg_μr_map)
                    print("+ $v*μr[$(bg_μr_map(j))] ")
                else
                    print("+ $v*μb[$(bg_μb_map(j))] ")
                end
            end
        end
        print("\n")
    end
end

"""
    chemical_potential_obs(ΔGf_mu, ΔGf_cov, eq_metabolite_names, metabolites_obs, μ0_map)

Extract chemical potential observations for a subset of metabolites.

# Arguments
- `ΔGf_mu`: Full vector of mean formation free energies (RT units).
- `ΔGf_cov`: Full covariance matrix of formation free energies (RT units).
- `eq_metabolite_names`: Ordered list of all metabolite names in the eQuilibrator dataset.
- `metabolites_obs`: Names of the metabolites to extract observations for.
- `μ0_map`: Bijection from species name to observation matrix row index.

# Returns
Tuple `(μ0_mean, μ0_cov, Q_μ0)`: subset mean, subset covariance, and the
corresponding observation matrix rows.
"""
function chemical_potential_obs(ΔGf_mu,ΔGf_cov,eq_metabolite_names,metabolites_obs,μ0_map)
    idx_metabolites = [μ0_map[m] for m in metabolites_obs]
    idx_map = [findfirst(x -> x == s, eq_metabolite_names) for s in metabolites_obs]
    μ0_mean = ΔGf_mu[idx_map]
    μ0_cov = ΔGf_cov[idx_map,idx_map]
    Q_μ0 = Q[idx_metabolites, :]
    return (μ0_mean, μ0_cov, Q_μ0)
end

"""
    KM_obs(rxns, dict_KM_medians, reaction_ids, metabolite_ids, KM_map; KM_std=0.5*log(10))

Extract KM observations that have matching entries in the BRENDA median dict.

# Arguments
- `rxns`: Vector of `(name, Reaction)` tuples.
- `dict_KM_medians`: Dict from `(ec_code, substrate)` to median log KM.
- `reaction_ids`: Dict mapping reaction name to EC code.
- `metabolite_ids`: Dict mapping species name to BRENDA substrate name.
- `KM_map`: Bijection from `(reaction, species)` to observation matrix row index.

# Keywords
- `KM_std`: Log-scale standard deviation assumed for all KM observations.

# Returns
Tuple `(KM_mu, KM_cov, Q_KM)`: observation means, diagonal covariance, and
the corresponding observation matrix rows.
"""
function KM_obs(rxns, dict_KM_medians, reaction_ids, metabolite_ids, KM_map; KM_std=0.5*log(10))
    KM_mu = Float64[]
    idx_KM = Int[]
    for r in rxns
        rid = reaction_ids[r[1]]
        substrates = vcat(r[2].substrates, r[2].products)
        for s in substrates
            k = (rid, metabolite_ids[extract_name(s)])
            k2 = (r[1], extract_name(s))
            if haskey(dict_KM_medians, k)
                push!(KM_mu, dict_KM_medians[k])
                push!(idx_KM, KM_map[k2])
            end
        end
    end
    KM_cov = Diagonal(KM_std^2*ones(length(KM_mu)))
    Q_KM = Q[idx_KM, :]
    return (KM_mu, KM_cov, Q_KM)
end

"""
    kcat_obs(rxns, dict_kcat_medians, reaction_ids, forward_metabolites, reverse_metabolites,
             kcatfmap, kcatrmap; kcat_std=0.5*log(10))

Extract kcat observations that have matching entries in the BRENDA median dict.

# Arguments
- `rxns`: Vector of `(name, Reaction)` tuples.
- `dict_kcat_medians`: Dict from `(ec_code, substrate)` to median log kcat.
- `reaction_ids`: Dict mapping reaction name to EC code.
- `forward_metabolites`: Dict mapping reaction name to its forward reference substrate.
- `reverse_metabolites`: Dict mapping reaction name to its reverse reference substrate.
- `kcatfmap`: Bijection from reaction name to forward kcat row index.
- `kcatrmap`: Bijection from reaction name to reverse kcat row index.

# Keywords
- `kcat_std`: Log-scale standard deviation assumed for all kcat observations.

# Returns
Tuple `(kcat_mu, kcat_cov, Q_kcat)`: observation means, diagonal covariance, and
the corresponding observation matrix rows.
"""
function kcat_obs(rxns, dict_kcat_medians, reaction_ids, forward_metabolites, reverse_metabolites, kcatfmap, kcatrmap; kcat_std=0.5*log(10))
    kcat_mu = Float64[]
    idx_kcat = Int[]
    for r in rxns
        rid = reaction_ids[r[1]]
        reactant = forward_metabolites[r[1]]
        k = (rid, reactant)
        if haskey(dict_kcat_medians, k)
            push!(kcat_mu, dict_kcat_medians[k])
            push!(idx_kcat, kcatfmap[r[1]])
        end
        product = reverse_metabolites[r[1]]
        k = (rid, product)
        if haskey(dict_kcat_medians, k)
            push!(kcat_mu, dict_kcat_medians[k])
            push!(idx_kcat, kcatrmap[r[1]])
        end
    end
    kcat_cov = Diagonal(kcat_std^2*ones(length(kcat_mu)))
    Q_kcat = Q[idx_kcat, :]
    return (kcat_mu, kcat_cov, Q_kcat)
end

"""
    posterior_dicts(rxns, q_post_full, C_post_full; h2o_strings=[])

Unpack the posterior parameter vector into named dictionaries for each parameter type.

# Arguments
- `rxns`: Vector of `(name, Reaction)` tuples.
- `q_post_full`: Posterior mean vector.
- `C_post_full`: Posterior covariance matrix.

# Keywords
- `h2o_strings`: Species names treated as water; their binding potentials are set to `Inf`.

# Returns
Tuple `(dict_μ0_params, dict_μr_params, dict_μb_params)`, each a `Dict` mapping
names to `(mean, std)` tuples.
"""
function posterior_dicts(rxns, q_post_full, C_post_full; h2o_strings=[])
    (bg_μ0_map, bg_μr_map, bg_μb_map) = parameter_index_maps(rxns)

    dict_μ0_params = Dict{String,Tuple{Float64,Float64}}()
    dict_μr_params = Dict{String,Tuple{Float64,Float64}}()
    dict_μb_params = Dict{Tuple{String,String},Tuple{Float64,Float64}}()
    for (s,i) in bg_μ0_map
        v = q_post_full[i]
        std = sqrt(C_post_full[i,i])
        dict_μ0_params[s] = (v,std)
    end

    for (r,i) in bg_μr_map
        v = q_post_full[i]
        std = sqrt(C_post_full[i,i])
        dict_μr_params[r] = (v,std)
    end

    for ((r,s),i) in bg_μb_map
        if s in h2o_strings
            v = Inf
            dict_μb_params[(r,s)] = (v,0.0)
        else
            v = q_post_full[i]
            std = sqrt(C_post_full[i,i])
            dict_μb_params[(r,s)] = (v,std)
        end
    end
    return (dict_μ0_params, dict_μr_params, dict_μb_params)
end

"""
    print_parameters(dict_μ0_params, dict_μr_params, dict_μb_params; h2o_strings=[])

Print standard chemical potentials, reaction potentials, and binding potentials to stdout.
"""
function print_parameters(dict_μ0_params, dict_μr_params, dict_μb_params; h2o_strings=[])
    println("Standard chemical potentials")
    for (s,(v,std)) in dict_μ0_params
        v = round(v,digits=1)
        std = round(std,digits=1)
        println("$s: $v ± $std")
    end
    print("\n")
    println("Reaction potentials")
    for (r,(v,std)) in dict_μr_params
        v = round(v,digits=1)
        std = round(std,digits=1)
        println("$r: $v ± $std")
    end
    print("\n")
    println("Binding potentials")
    for ((r,s),(v,std)) in dict_μb_params
        if s in h2o_strings
            v = Inf
            println("($r,$s): $v")
        else
            v = round(v,digits=1)
            std = round(std,digits=1)
            println("($r,$s): $v ± $std")
        end
    end
end

function plot_KM_comparison(q_post,C_post,Q,KM_map,dict_KM_medians,reaction_ids,metabolite_ids;
    print_labels=false)
    μ_model = Q*q_post

    # Compare literature parameters with inferred parameters
    KM_names = String[]
    KM_literature = Float64[]
    KM_inferred = Float64[]
    logKM_std = Float64[]
    for ((r,s),i) in KM_map
        rid = reaction_ids[r]
        met = metabolite_ids[s]
        if haskey(dict_KM_medians, (rid, met))
            push!(KM_names, "$r: $met")
            push!(KM_literature, 1000*exp(dict_KM_medians[(rid, met)]))
            push!(KM_inferred, 1000*exp(μ_model[i]))
            push!(logKM_std, C_post[i,i])
        end
    end
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="KM (literature) [mM]", ylabel="KM (inferred) [mM]",
        xscale=log10, yscale=log10)
    CairoMakie.scatter!(ax,KM_literature,KM_inferred)
    if print_labels
        text!(ax,KM_literature,KM_inferred,text=KM_names)
    end
    lines!(ax,[1e-2, 1e1], [1e-2, 1e1], color=:black)
    return fig
end

function plot_kcat_comparison(q_post_full,Q,kcatfmap,kcatrmap,dict_kcat_medians,reaction_ids,forward_metabolites,reverse_metabolites; print_labels=false)
    μ_model = Q*q_post_full

    kcat_names = String[]
    kcat_literature = Float64[]
    kcat_inferred = Float64[]
    for (r,i) in kcatfmap
        rid = reaction_ids[r]
        met = forward_metabolites[r]
        if haskey(dict_kcat_medians, (rid, met))
            push!(kcat_names, "$r: fwd")
            push!(kcat_literature, exp(dict_kcat_medians[(rid, met)]))
            push!(kcat_inferred, exp(μ_model[i]))
        end
    end
    for (r,i) in kcatrmap
        rid = reaction_ids[r]
        met = reverse_metabolites[r]
        if haskey(dict_kcat_medians, (rid, met))
            push!(kcat_names, "$r: rev")
            push!(kcat_literature, exp(dict_kcat_medians[(rid, met)]))
            push!(kcat_inferred, exp(μ_model[i]))
        end
    end
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="kcat (literature) [1/s]", ylabel="kcat (inferred) [1/s]",
        xscale=log10, yscale=log10)
    CairoMakie.scatter!(ax,kcat_literature,kcat_inferred)
    if print_labels
        text!(ax,kcat_literature,kcat_inferred,text=kcat_names)
    end
    lines!(ax,10 .^ [-1.5, 3.5], 10 .^ [-1.5, 3.5], color=:black)
    fig
end

"""
    extract_result_comparison(rxns, q_post_full, C_post_full, Q,
        eq_metabolite_names, ΔGf_mu, ΔGf_cov,
        dict_KM_medians, dict_kcat_medians, reaction_ids, metabolite_ids,
        forward_metabolites, reverse_metabolites;
        metabolite_string_mapping=identity, reaction_string_mapping=identity)

Build DataFrames comparing posterior model predictions to literature values for
standard chemical potentials, KM values, and kcat values.

# Arguments
- `rxns`: Vector of `(name, Reaction)` tuples.
- `q_post_full`, `C_post_full`: Posterior mean and covariance in observable space.
- `Q`: Full observation matrix.
- `eq_metabolite_names`, `ΔGf_mu`, `ΔGf_cov`: eQuilibrator reference data.
- `dict_KM_medians`, `dict_kcat_medians`: BRENDA median log-value dicts.
- `reaction_ids`, `metabolite_ids`: Dicts mapping model names to database identifiers.
- `forward_metabolites`, `reverse_metabolites`: Dicts mapping reaction to reference substrate.

# Keywords
- `metabolite_string_mapping`: Optional function to rename metabolite keys for display.
- `reaction_string_mapping`: Optional function to rename reaction keys for display.

# Returns
Tuple `(μ0_comparison, KM_comparison, kcat_comparison)`: three `DataFrame`s with
columns `name`, `model_mean`, `model_std`, `data_mean`, `data_std`, `mean_diff`.
"""
function extract_result_comparison(rxns,q_post_full,C_post_full,Q,
    eq_metabolite_names, ΔGf_mu, ΔGf_cov,
    dict_KM_medians, dict_kcat_medians, reaction_ids, metabolite_ids,
    forward_metabolites, reverse_metabolites;
    metabolite_string_mapping=identity, reaction_string_mapping=identity)
    dict_ΔGf = Dict(eq_metabolite_names .=> ΔGf_mu)
    ΔGf_std = diag(sqrt(Diagonal(ΔGf_cov)))
    dict_ΔGf_std = Dict(eq_metabolite_names .=> ΔGf_std)
    μ_model = Q*q_post_full
    x_cov = Q*C_post_full*transpose(Q)
    (μ0_map, KM_map, kcatfmap, kcatrmap) = observation_matrix_index_maps(rxns)

    μ0_comparison = DataFrame(
        name=String[],model_mean=Float64[],model_std=Float64[],data_mean=Float64[],data_std=Float64[]
    )
    KM_comparison = DataFrame(
        name=String[],model_mean=Float64[],model_std=Float64[],data_mean=Float64[],data_std=Float64[]
    )
    kcat_comparison = DataFrame(
        name=String[],model_mean=Float64[],model_std=Float64[],data_mean=Float64[],data_std=Float64[]
    )

    for (s,i) in μ0_map
        v = μ_model[i]
        std = sqrt(x_cov[i,i])
        skey = metabolite_string_mapping(s)
        if haskey(dict_ΔGf, skey)
            lit_val = dict_ΔGf[skey]
            lit_std = dict_ΔGf_std[skey]
        else
            lit_val = lit_std = NaN
        end
        push!(μ0_comparison,(name=metabolite_string_mapping(s),model_mean=v,model_std=std,data_mean=lit_val,data_std=lit_std))
    end

    for ((r,s),i) in KM_map
        v = μ_model[i]
        std = sqrt(x_cov[i,i])
        if haskey(dict_KM_medians, (reaction_ids[r], metabolite_ids[s]))
            lit_val = dict_KM_medians[(reaction_ids[r], metabolite_ids[s])]
            lit_std = 0.5*log(10)
        else
            lit_val = lit_std = NaN
        end
        push!(KM_comparison,(name="$(reaction_string_mapping(r)):$(metabolite_string_mapping(s))",model_mean=v,model_std=std,data_mean=lit_val,data_std=lit_std))
    end

    for (r,i) in kcatfmap
        v = μ_model[i]
        std = sqrt(x_cov[i,i])
        if haskey(dict_kcat_medians, (reaction_ids[r], forward_metabolites[r]))
            lit_val = dict_kcat_medians[(reaction_ids[r], forward_metabolites[r])]
            lit_std = 0.5*log(10)
        else
            lit_val = lit_std = NaN
        end
        push!(kcat_comparison,(name="$(reaction_string_mapping(r)):fwd",model_mean=v,model_std=std,data_mean=lit_val,data_std=lit_std))
    end

    for (r,i) in kcatrmap
        v = μ_model[i]
        std = sqrt(x_cov[i,i])
        if haskey(dict_kcat_medians, (reaction_ids[r], reverse_metabolites[r]))
            lit_val = dict_kcat_medians[(reaction_ids[r], reverse_metabolites[r])]
            lit_std = 0.5*log(10)
        else
            lit_val = lit_std = NaN
        end
        push!(kcat_comparison,(name="$(reaction_string_mapping(r)):rev",model_mean=v,model_std=std,data_mean=lit_val,data_std=lit_std))
    end
    calculate_differernce_of_mean!(μ0_comparison)
    calculate_differernce_of_mean!(KM_comparison)
    calculate_differernce_of_mean!(kcat_comparison)
    return (μ0_comparison,KM_comparison,kcat_comparison)
end

"""
    calculate_differernce_of_mean!(df)

Add a `mean_diff` column equal to `model_mean - data_mean` to `df` in place.

Note: the function name contains a typo (`differernce`) that is preserved
for backward compatibility.
"""
function calculate_differernce_of_mean!(df)
    df.mean_diff = df.model_mean - df.data_mean
end

"""
    print_result_comparison(df)

Print each row of a comparison DataFrame as
`name: model_mean plus/minus model_std (obs: data_mean plus/minus data_std)`.
"""
function print_result_comparison(df)
    for r in eachrow(df)
        v = round(r.model_mean, digits=1)
        std = round(r.model_std, digits=1)
        print("$(r.name): $v ± $std ")
        if ~isnan(r.data_mean)
            lit_val = round(r.data_mean, digits=1)
            lit_std = round(r.data_std, digits=1)
            print("(obs: $lit_val ± $lit_std)")
        end
        print("\n")
    end
    return nothing
end

"""
    print_enzyme_kinetics_parameter_comparison(μ0_comparison, KM_comparison, kcat_comparison)

Print all three comparison DataFrames (standard chemical potentials, KM, kcat) in sequence.
"""
function print_enzyme_kinetics_parameter_comparison(μ0_comparison,KM_comparison,kcat_comparison)
    println("Standard chemical potentials:")
    print_result_comparison(μ0_comparison)
    println("")
    println("KM values:")
    print_result_comparison(KM_comparison)
    println("")
    println("kcat values:")
    print_result_comparison(kcat_comparison)
    return nothing
end
