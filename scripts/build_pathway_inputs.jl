# build_pathway_inputs.jl
#
# Rebuild the record-level input files consumed by run_all_paper_experiments.jl
# and the revision drivers, for a given pathway specification:
#
#   output/<pathway>/data/brenda_inputs_<pathway>.jld2
#       y      :: Vector{Float64}   natural-log observations (kcat in 1/s, KM in M)
#       σ2     :: Vector{Float64}   constant (0.5*log(10))^2 per record (half-decade std)
#       Arows  :: Matrix{Float64}   Q[meta.Q_row, :] rows of the full observation matrix
#       grp    :: Vector{String}    lowercase BRENDA organism (leave-one-source-out group)
#       meta   :: DataFrame         kind, ec_code, substrate, Q_row, group_id
#       A0, b0                      thermodynamic (eQuilibrator) information pair
#
#   output/<pathway>/data/posterior_<pathway>.jld2
#       q_prior, C_prior            thermodynamic prior in parameter space
#       q_post,  C_post             Gaussian parameter-balancing posterior
#       Q_obs                       stacked observed rows [Q_mu0; Q_KM; Q_kcat]
#       μ_cov                       BlockDiagonal([μ0_cov, KM_cov, kcat_cov])
#
# Construction conventions (recovered from run_all_paper_experiments.jl,
# metabolic_reconstruction.jl / parameter_balancing.jl and the existing files):
#
#   * The full observation matrix Q = observation_matrix(rxns; h2o_strings) has
#     rows ordered [μ0 (per species); log KM (per (reaction,species) pair,
#     column-major over the binary net-stoichiometry matrix); kcat+ (per
#     reaction); kcat- (per reaction)] and columns [μ0; μr; μb].
#   * Record-level y: one row per BRENDA CSV record with value > 0 whose
#     (ec_code, substrate) pair maps to an observable of the pathway.
#     kcat records come first (kcat CSV file order), then KM records
#     (KM CSV file order). y = log(kcat [1/s]) resp. log(KM [mM] / 1000).
#     Substrate matching is case-insensitive (meta keeps the CSV casing);
#     the aggregate median pipeline in run_all_paper_experiments.jl is
#     case-sensitive, so e.g. TCA "Lipoamide" records enter the record-level
#     file but not the median observations.
#   * (ec, substrate) -> Q_row lookup is last-write-wins:
#       kcat: for r in rxns assign (ec, forward_metabolite) -> kcatf row,
#             then for r in rxns assign (ec, reverse_metabolite) -> kcatr row.
#       KM:   for r in rxns, for s in vcat(substrates, products) with
#             (r, s) in the binding-potential map, assign
#             (ec, brenda_metabolite_id[s]) -> KM row.
#     (Matters only when one EC maps to several reactions, e.g. aconitase
#     4.2.1.3 = r_0302 + r_0280 in the TCA pathway: cis-aconitate kcat
#     records land on kcat-(r_0302), its KM records on KM(r_0280).)
#   * σ2 = (0.5*log(10))^2 for every record (half-decade log10 standard
#     deviation), matching KM_obs / kcat_obs defaults.
#   * grp = lowercase(organism) from the CSV `species` column.
#   * A0, b0 are the information pair of the Gaussian likelihood of the
#     eQuilibrator block, the form S1 Text states:
#         A0 = Q_μ0' Σ⁻¹ Q_μ0,   b0 = Q_μ0' Σ⁻¹ μ0_mean,
#     where Σ is the eQuilibrator covariance in RT units (data/equilibrator/*.py
#     writes sigma_fin sigma_fin' + 1e6 sigma_inf sigma_inf' in (kJ/mol)^2 and
#     load_equilibrator_data divides it by RT^2), as make_information_pair of
#     experiment_runners.jl also computes.  Until 20 October 2026 the stored
#     files held the squared-inverse form Q_μ0' Σ⁻² Q_μ0 and Q_μ0' Σ⁻² μ0_mean,
#     which this script reproduced: a whitening that used the full Cholesky
#     solve (Σ⁻¹ Q) where the triangular factor (L⁻¹ Q) was meant.  That form
#     gave each eigen-direction of Σ 1/variance times its weight, so the
#     well-determined formation energies were overweighted.
#     make_information_pair_legacy keeps it for comparison.
#
# Usage:
#   julia --project=. scripts/build_pathway_inputs.jl <pathway> [options]
#     <pathway>          glycolysis | tca (extend PATHWAY_SPECS for new ones)
#     --outdir=DIR       output root (default "output"); files are written to
#                        DIR/<pathway>/data/. Existing files are never
#                        overwritten unless --force is given.
#     --force            allow overwriting existing output files
#     --verify           after building, compare against the reference files in
#                        output/<pathway>/data/ and report max abs differences
#
# Example (verification without touching the originals):
#   julia --project=. scripts/build_pathway_inputs.jl glycolysis --outdir=/tmp/rebuild --verify
#   julia --project=. scripts/build_pathway_inputs.jl tca        --outdir=/tmp/rebuild --verify

import Pkg
Pkg.activate(dirname(@__DIR__))

using LinearAlgebra
using Statistics
using BlockDiagonals
using CSV
using DataFrames
using JLD2
using JSON

include(joinpath(dirname(@__DIR__), "src", "metabolic_reconstruction.jl"))
include(joinpath(dirname(@__DIR__), "src", "parameter_balancing.jl"))

# Pathway specifications

struct PathwaySpec
    name::String                       # tag used in output file names
    reaction_ids::Vector{String}       # yeast-GEM reaction ids, pathway order
    proton_name::String                # yeast-GEM species id of H+ (excluded)
    h2o::Vector{String}                # yeast-GEM species id(s) of water
    bigg_ids::Dict{String,String}      # species id -> eQuilibrator/BiGG name
    brenda_metabolite_ids::Dict{String,String}  # species id -> BRENDA substrate name
    ec_codes::Dict{String,String}      # reaction id -> EC number
    forward_metabolites::Dict{String,String}    # reaction id -> kcat+ reference substrate
    reverse_metabolites::Dict{String,String}    # reaction id -> kcat- reference substrate
    equilibrator_json::String
    kcat_csv::String
    KM_csv::String
end

const PATHWAY_SPECS = Dict{String,PathwaySpec}()

PATHWAY_SPECS["tca"] = PathwaySpec(
    "tca",
    ["r_0300","r_0302","r_0280","r_0658","r_0832","r_0831",
     "r_0505","r_1022","r_1021","r_0451","r_0713"],
    "s_0799",
    ["s_0807"],
    Dict("s_0376"=>"accoa","s_0807"=>"h2o","s_1273"=>"oaa","s_0524"=>"cit","s_0532"=>"coa",
        "s_0517"=>"acon_C","s_0941"=>"icit","s_1200"=>"nad","s_0182"=>"akg","s_0460"=>"co2",
        "s_1205"=>"nadh","s_1097"=>"lpam","s_1411"=>"sdhlam","s_0627"=>"dhlam","s_1464"=>"succoa",
        "s_0397"=>"adp","s_1326"=>"pi","s_0437"=>"atp","s_1460"=>"succ","s_1537"=>"q6",
        "s_0727"=>"fum","s_1535"=>"q6h2","s_0068"=>"mal__L"),
    Dict("s_0376"=>"acetyl-CoA","s_0807"=>"H2O","s_1273"=>"oxaloacetate",
        "s_0524"=>"citrate","s_0532"=>"CoA","s_0517"=>"cis-aconitate","s_0941"=>"isocitrate",
        "s_1200"=>"NAD+","s_0182"=>"2-oxoglutarate","s_0460"=>"CO2","s_1205"=>"NADH",
        "s_1097"=>"lipoamide","s_1411"=>"succinyldihydrolipoamide","s_0627"=>"DL-dihydrolipoamide",
        "s_1464"=>"succinyl-CoA","s_0397"=>"ADP","s_1326"=>"phosphate","s_0437"=>"ATP",
        "s_1460"=>"succinate","s_1537"=>"ubiquinone","s_0727"=>"fumarate","s_1535"=>"ubiquinol",
        "s_0068"=>"L-malate"),
    Dict("r_0300"=>"2.3.3.1","r_0302"=>"4.2.1.3","r_0280"=>"4.2.1.3",
        "r_0658"=>"1.1.1.41","r_0832"=>"1.2.4.2","r_0831"=>"2.3.1.61","r_0505"=>"1.8.1.4",
        "r_1022"=>"6.2.1.5","r_1021"=>"1.3.5.1","r_0451"=>"4.2.1.2","r_0713"=>"1.1.1.37"),
    Dict("r_0300"=>"acetyl-CoA","r_0302"=>"citrate","r_0280"=>"cis-aconitate",
        "r_0658"=>"isocitrate","r_0832"=>"2-oxoglutarate","r_0831"=>"succinyldihydrolipoamide",
        "r_0505"=>"DL-dihydrolipoamide","r_1022"=>"succinyl-CoA","r_1021"=>"succinate",
        "r_0451"=>"fumarate","r_0713"=>"L-malate"),
    Dict("r_0300"=>"citrate","r_0302"=>"cis-aconitate","r_0280"=>"isocitrate",
        "r_0658"=>"2-oxoglutarate","r_0832"=>"succinyldihydrolipoamide",
        "r_0831"=>"DL-dihydrolipoamide","r_0505"=>"lipoamide","r_1022"=>"succinate",
        "r_1021"=>"fumarate","r_0451"=>"L-malate","r_0713"=>"oxaloacetate"),
    "data/equilibrator/equilibrator_tca_cycle.json",
    "data/brenda/kcat_values_tca_cycle.csv",
    "data/brenda/KM_values_tca_cycle.csv")

PATHWAY_SPECS["glycolysis"] = PathwaySpec(
    "glycolysis",
    ["r_0534","r_0467","r_0886","r_0450","r_1054",
     "r_0486","r_0892","r_0893","r_0366","r_0962"],
    "s_0794",
    ["s_0803"],
    Dict("s_0434"=>"atp","s_0563"=>"glc__D","s_0394"=>"adp","s_0568"=>"g6p",
        "s_0557"=>"f6p","s_0555"=>"fdp","s_0629"=>"dhap","s_0764"=>"g3p",
        "s_1198"=>"nad","s_1322"=>"pi","s_0075"=>"13dpg","s_1203"=>"nadh",
        "s_0260"=>"3pg","s_0188"=>"2pg","s_0803"=>"h2o","s_1360"=>"pep","s_1399"=>"pyr"),
    Dict("s_0434"=>"ATP","s_0563"=>"D-glucose","s_0394"=>"ADP",
        "s_0568"=>"D-glucose 6-phosphate","s_0557"=>"D-fructose 6-phosphate",
        "s_0555"=>"D-fructose 1,6-bisphosphate","s_0629"=>"dihydroxyacetone phosphate",
        "s_0764"=>"D-glyceraldehyde 3-phosphate","s_1198"=>"NAD+","s_1322"=>"phosphate",
        "s_0075"=>"3-phospho-D-glyceroyl phosphate","s_1203"=>"NADH",
        "s_0260"=>"3-phospho-D-glycerate","s_0188"=>"2-phospho-D-glycerate",
        "s_0803"=>"H2O","s_1360"=>"phosphoenolpyruvate","s_1399"=>"pyruvate"),
    Dict("r_0534"=>"2.7.1.1","r_0467"=>"5.3.1.9","r_0886"=>"2.7.1.11",
        "r_0450"=>"4.1.2.13","r_1054"=>"5.3.1.1","r_0486"=>"1.2.1.12",
        "r_0892"=>"2.7.2.3","r_0893"=>"5.4.2.11","r_0366"=>"4.2.1.11","r_0962"=>"2.7.1.40"),
    Dict("r_0534"=>"D-glucose","r_0467"=>"D-glucose 6-phosphate",
        "r_0886"=>"D-fructose 6-phosphate","r_0450"=>"D-fructose 1,6-bisphosphate",
        "r_1054"=>"dihydroxyacetone phosphate","r_0486"=>"D-glyceraldehyde 3-phosphate",
        "r_0892"=>"3-phospho-D-glyceroyl phosphate","r_0893"=>"3-phospho-D-glycerate",
        "r_0366"=>"2-phospho-D-glycerate","r_0962"=>"phosphoenolpyruvate"),
    Dict("r_0534"=>"D-glucose 6-phosphate","r_0467"=>"D-fructose 6-phosphate",
        "r_0886"=>"D-fructose 1,6-bisphosphate","r_0450"=>"D-glyceraldehyde 3-phosphate",
        "r_1054"=>"D-glyceraldehyde 3-phosphate","r_0486"=>"3-phospho-D-glyceroyl phosphate",
        "r_0892"=>"3-phospho-D-glycerate","r_0893"=>"2-phospho-D-glycerate",
        "r_0366"=>"phosphoenolpyruvate","r_0962"=>"pyruvate"),
    "data/equilibrator/equilibrator_glycolysis.json",
    "data/brenda/kcat_values_glycolysis.csv",
    "data/brenda/KM_values_glycolysis.csv")

# Pentose phosphate pathway, added for the third-pathway generalisation check.
# Curation choices, recorded here because a reviewer will ask:
#   * BRENDA substrate names use the dominant spelling by record count; the
#     pipeline matches one exact (case-insensitive) name per species, so
#     records under variant spellings ("6-phospho-D-gluconate",
#     "D-ribose-5-phosphate") are dropped, exactly as in the other pathways.
#     "6-phosphogluconate" (68 records) is preferred over
#     "6-phospho-D-gluconate" (38).
#   * EC 2.2.1.1 maps to both transketolase reactions (like TCA aconitase);
#     the kcat direction references are chosen distinct (xu5p -> TKT1 forward,
#     e4p -> TKT2 forward, s7p / f6p the respective reverses) so the
#     last-write-wins lookup attributes each record to one reaction, and KM
#     collisions resolve to the later reaction in list order (r_1050).
#   * Reaction directions are as written in yeast-GEM; r_0982 is
#     ru5p -> r5p, so the abundant D-ribose 5-phosphate kcat records attach
#     to the reverse turnover number.
PATHWAY_SPECS["pentose_phosphate"] = PathwaySpec(
    "pentose_phosphate",
    ["r_0466","r_0091","r_0889","r_0982","r_0984","r_1049","r_1050","r_1048"],
    "s_0794",
    ["s_0803"],
    Dict("s_0568"=>"g6p","s_1207"=>"nadp","s_0335"=>"6pgl","s_1212"=>"nadph",
        "s_0803"=>"h2o","s_0340"=>"6pgc","s_0456"=>"co2","s_0577"=>"ru5p__D",
        "s_1408"=>"r5p","s_0581"=>"xu5p__D","s_0764"=>"g3p","s_1427"=>"s7p",
        "s_0551"=>"e4p","s_0557"=>"f6p"),
    Dict("s_0568"=>"D-glucose 6-phosphate","s_1207"=>"NADP+",
        "s_0335"=>"6-phospho-D-glucono-1,5-lactone","s_1212"=>"NADPH",
        "s_0803"=>"H2O","s_0340"=>"6-phosphogluconate","s_0456"=>"CO2",
        "s_0577"=>"D-ribulose 5-phosphate","s_1408"=>"D-ribose 5-phosphate",
        "s_0581"=>"D-xylulose 5-phosphate","s_0764"=>"glyceraldehyde 3-phosphate",
        "s_1427"=>"sedoheptulose 7-phosphate","s_0551"=>"D-erythrose 4-phosphate",
        "s_0557"=>"D-fructose 6-phosphate"),
    Dict("r_0466"=>"1.1.1.49","r_0091"=>"3.1.1.31","r_0889"=>"1.1.1.44",
        "r_0982"=>"5.3.1.6","r_0984"=>"5.1.3.1","r_1049"=>"2.2.1.1",
        "r_1050"=>"2.2.1.1","r_1048"=>"2.2.1.2"),
    Dict("r_0466"=>"D-glucose 6-phosphate",
        "r_0091"=>"6-phospho-D-glucono-1,5-lactone",
        "r_0889"=>"6-phosphogluconate",
        "r_0982"=>"D-ribulose 5-phosphate",
        "r_0984"=>"D-ribulose 5-phosphate",
        "r_1049"=>"D-xylulose 5-phosphate",
        "r_1050"=>"D-erythrose 4-phosphate",
        "r_1048"=>"glyceraldehyde 3-phosphate"),
    Dict("r_0466"=>"6-phospho-D-glucono-1,5-lactone",
        "r_0091"=>"6-phosphogluconate",
        "r_0889"=>"D-ribulose 5-phosphate",
        "r_0982"=>"D-ribose 5-phosphate",
        "r_0984"=>"D-xylulose 5-phosphate",
        "r_1049"=>"sedoheptulose 7-phosphate",
        "r_1050"=>"D-fructose 6-phosphate",
        "r_1048"=>"D-fructose 6-phosphate"),
    "data/equilibrator/equilibrator_pentose_phosphate.json",
    "data/brenda/kcat_values_pentose_phosphate.csv",
    "data/brenda/KM_values_pentose_phosphate.csv")

# Shikimate (aromatic amino acid biosynthesis) pathway, added as the
# fourth-pathway generalisation check OUTSIDE central carbon metabolism.
# Curation notes, recorded because a reviewer will ask:
#   * r_0042 is the cytosolic DAHP synthase; the mitochondrial duplicate
#     r_0020 (same EC) is deliberately excluded.
#   * Five reactions are catalysed by the pentafunctional ARO1 protein and
#     carry all five EC annotations in the SBML; the one-EC-per-reaction
#     assignment below is therefore hard-coded, never read from the SBML.
#   * Substrate names use the dominant BRENDA spelling by record count:
#     "3-phosphoshikimate" (68 records) over "shikimate 3-phosphate" (19),
#     "3-deoxy-D-arabino-hept-2-ulosonate 7-phosphate" (17) over the
#     hyphenation variants; variant spellings are dropped, as in the other
#     pathways.
#   * r_0996 is written reductive (3-dehydroshikimate + NADPH -> shikimate),
#     the reverse of the EC 1.1.1.25 assay convention, so the abundant
#     shikimate-oxidation turnover records attach to the reverse turnover
#     number, analogous to r_0982 in the pentose phosphate spec.
#   * eQuilibrator resolves the yeast-GEM BiGG id "skm5p" (not the E. coli
#     synonym "s3p"), verified by smoke test.
PATHWAY_SPECS["shikimate"] = PathwaySpec(
    "shikimate",
    ["r_0042","r_0040","r_0039","r_0996","r_0997","r_0065","r_0279"],
    "s_0794",
    ["s_0803"],
    Dict("s_0551"=>"e4p","s_1360"=>"pep","s_0803"=>"h2o","s_0349"=>"2dda7p",
        "s_1322"=>"pi","s_0210"=>"3dhq","s_0211"=>"3dhsk","s_1212"=>"nadph",
        "s_1207"=>"nadp","s_1429"=>"skm","s_0434"=>"atp","s_0394"=>"adp",
        "s_0261"=>"skm5p","s_0324"=>"3psme","s_0515"=>"chor"),
    Dict("s_0551"=>"D-erythrose 4-phosphate","s_1360"=>"phosphoenolpyruvate",
        "s_0803"=>"H2O",
        "s_0349"=>"3-deoxy-D-arabino-hept-2-ulosonate 7-phosphate",
        "s_1322"=>"phosphate","s_0210"=>"3-dehydroquinate",
        "s_0211"=>"3-dehydroshikimate","s_1212"=>"NADPH","s_1207"=>"NADP+",
        "s_1429"=>"shikimate","s_0434"=>"ATP","s_0394"=>"ADP",
        "s_0261"=>"3-phosphoshikimate",
        "s_0324"=>"5-enolpyruvylshikimate 3-phosphate","s_0515"=>"chorismate"),
    Dict("r_0042"=>"2.5.1.54","r_0040"=>"4.2.3.4","r_0039"=>"4.2.1.10",
        "r_0996"=>"1.1.1.25","r_0997"=>"2.7.1.71","r_0065"=>"2.5.1.19",
        "r_0279"=>"4.2.3.5"),
    Dict("r_0042"=>"phosphoenolpyruvate",
        "r_0040"=>"3-deoxy-D-arabino-hept-2-ulosonate 7-phosphate",
        "r_0039"=>"3-dehydroquinate",
        "r_0996"=>"3-dehydroshikimate",
        "r_0997"=>"shikimate",
        "r_0065"=>"3-phosphoshikimate",
        "r_0279"=>"5-enolpyruvylshikimate 3-phosphate"),
    Dict("r_0042"=>"3-deoxy-D-arabino-hept-2-ulosonate 7-phosphate",
        "r_0040"=>"3-dehydroquinate",
        "r_0039"=>"3-dehydroshikimate",
        "r_0996"=>"shikimate",
        "r_0997"=>"3-phosphoshikimate",
        "r_0065"=>"5-enolpyruvylshikimate 3-phosphate",
        "r_0279"=>"chorismate"),
    "data/equilibrator/equilibrator_shikimate.json",
    "data/brenda/kcat_values_shikimate.csv",
    "data/brenda/KM_values_shikimate.csv")

# Helpers

"""
    make_information_pair_gaussian(Q_block, mu_block, Sigma_block) -> (A0, b0)

Information pair of a Gaussian likelihood mu_block ~ N(Q_block q, Sigma_block):
A0 = Q' Σ⁻¹ Q and b0 = Q' Σ⁻¹ μ, computed with one Cholesky solve.  This is the
form the stored inputs use since 20 October 2026.
"""
function make_information_pair_gaussian(
    Q_block::AbstractMatrix,
    μ_block::AbstractVector,
    Σ_block::AbstractMatrix
)
    Qm = Matrix{Float64}(Q_block)
    μv = Vector{Float64}(μ_block)

    size(Qm, 1) == length(μv) || error("make_information_pair_gaussian: dimension mismatch")
    size(Σ_block, 1) == length(μv) && size(Σ_block, 2) == length(μv) ||
        error("make_information_pair_gaussian: covariance dimension mismatch")

    F = cholesky(Symmetric(Matrix{Float64}(Σ_block)))
    A0 = transpose(Qm) * (F \ Qm)
    b0 = transpose(Qm) * (F \ μv)

    return Matrix(Symmetric(A0)), Vector(b0)
end

"""
    make_information_pair_legacy(Q_block, mu_block, Sigma_block) -> (A0, b0)

Thermodynamic information pair in the squared-inverse form that the stored
brenda_inputs files held until 20 October 2026, kept for comparison only:

    ZQ = Σ⁻¹ Q,  Zμ = Σ⁻¹ μ,  A0 = ZQ' ZQ,  b0 = ZQ' Zμ

i.e. A0 = Q' Σ⁻² Q and b0 = Q' Σ⁻² μ. Note this is NOT the same as the
current `make_information_pair` in experiment_runners.jl (Q' Σ⁻¹ Q, Q' Σ⁻¹ μ);
the stored files predate it and verifiably use the squared form. The Cholesky
solve reproduces the stored values most closely (max abs deviation of order
1e-10 on entries of order 1e5, i.e. pure floating-point/BLAS noise).
"""
function make_information_pair_legacy(
    Q_block::AbstractMatrix,
    μ_block::AbstractVector,
    Σ_block::AbstractMatrix
)
    Qm = Matrix{Float64}(Q_block)
    μv = Vector{Float64}(μ_block)

    size(Qm, 1) == length(μv) || error("make_information_pair_legacy: dimension mismatch")
    size(Σ_block, 1) == length(μv) && size(Σ_block, 2) == length(μv) ||
        error("make_information_pair_legacy: covariance dimension mismatch")

    F = cholesky(Symmetric(Matrix{Float64}(Σ_block)))
    ZQ = F \ Qm
    Zμ = F \ μv

    A0 = transpose(ZQ) * ZQ
    b0 = transpose(ZQ) * Zμ

    return Matrix(A0), Vector(b0)
end

"""
    build_pathway_inputs(spec; mdl) -> NamedTuple

Reconstruct the full observation system and both record-level and aggregate
(median-based) data products for `spec`. `mdl` is the loaded yeast-GEM model.
"""
function build_pathway_inputs(spec::PathwaySpec; mdl)
    dict_reactions = reaction_data_dict(mdl)
    rxns = generate_reactions(spec.reaction_ids, dict_reactions,
                              protons=false, proton_name=spec.proton_name)

    # Full observation matrix and index maps. chemical_potential_obs/KM_obs/
    # kcat_obs read the global binding `Q` (legacy convention kept by the
    # driver scripts), so set it before calling them.
    global Q = observation_matrix(rxns, h2o_strings=spec.h2o)
    (array_species, array_reactions) = rn_entities(rxns)
    (μ0_map, KM_map, kcatfmap, kcatrmap) = observation_matrix_index_maps(rxns)

    # Aggregate (median) observation system, as in run_all_paper_experiments.jl
    (eq_names, ΔGf_mu, ΔGf_cov) = load_equilibrator_data(spec.equilibrator_json)
    (kcat_med, KM_med) = load_BRENDA_data(spec.kcat_csv, spec.KM_csv)

    met_obs = [spec.bigg_ids[s] for s in array_species]
    μ0eq = Dict(spec.bigg_ids[k] => v for (k, v) in μ0_map)
    (μ0_mean, μ0_cov, Q_μ0) = chemical_potential_obs(ΔGf_mu, ΔGf_cov, eq_names, met_obs, μ0eq)
    (KM_mu, KM_cov, Q_KM) = KM_obs(rxns, KM_med, spec.ec_codes, spec.brenda_metabolite_ids, KM_map)
    (kcat_mu, kcat_cov, Q_kcat) = kcat_obs(rxns, kcat_med, spec.ec_codes,
        spec.forward_metabolites, spec.reverse_metabolites, kcatfmap, kcatrmap)

    Q_obs = vcat(Q_μ0, Q_KM, Q_kcat)
    μ_obs = vcat(μ0_mean, KM_mu, kcat_mu)
    μ_cov = BlockDiagonal([μ0_cov, Matrix(KM_cov), Matrix(kcat_cov)])
    (q_prior, C_prior) = thermodynamic_prior(rxns, spec.h2o)
    (q_post, C_post) = infer_parameters(Q_obs, μ_obs, μ_cov, q_prior, C_prior)

    # Thermodynamic information pair (eQuilibrator block only), Gaussian likelihood
    A0, b0 = make_information_pair_gaussian(Q_μ0, μ0_mean, Matrix(μ0_cov))

    # (ec_code, lowercase(substrate)) -> Q row lookups, last write wins.
    # Substrate matching is case-insensitive: the TCA CSVs write "Lipoamide"
    # while the pathway spec uses "lipoamide", and the stored reference file
    # includes those records. meta keeps the CSV's original casing.
    # kcat: all forward assignments first, then all reverse assignments.
    kcat_rows = Dict{Tuple{String,String},Int}()
    for r in rxns
        kcat_rows[(spec.ec_codes[r[1]], lowercase(spec.forward_metabolites[r[1]]))] = kcatfmap[r[1]]
    end
    for r in rxns
        kcat_rows[(spec.ec_codes[r[1]], lowercase(spec.reverse_metabolites[r[1]]))] = kcatrmap[r[1]]
    end
    # KM: reactions in pathway order, substrates then products within a reaction.
    KM_rows = Dict{Tuple{String,String},Int}()
    for r in rxns
        rid = spec.ec_codes[r[1]]
        for s in vcat(r[2].substrates, r[2].products)
            sname = extract_name(s)
            haskey(KM_map, (r[1], sname)) || continue
            KM_rows[(rid, lowercase(spec.brenda_metabolite_ids[sname]))] = KM_map[(r[1], sname)]
        end
    end

    # Record-level data: kcat CSV pass, then KM CSV pass, in file order.
    y = Float64[]
    grp = String[]
    meta = DataFrame(kind=String[], ec_code=String[], substrate=String[],
                     Q_row=Int[], group_id=String[])

    df_kcat = DataFrame(CSV.File(spec.kcat_csv))
    for rec in eachrow(df_kcat)
        (ismissing(rec.kcat) || rec.kcat <= 0) && continue
        sub = String(rec.substrate)
        key = (String(rec.ec_code), lowercase(sub))
        haskey(kcat_rows, key) || continue
        g = lowercase(String(rec.species))
        push!(y, log(rec.kcat))
        push!(grp, g)
        push!(meta, (kind="kcat", ec_code=key[1], substrate=sub,
                     Q_row=kcat_rows[key], group_id=g))
    end

    df_KM = DataFrame(CSV.File(spec.KM_csv))
    for rec in eachrow(df_KM)
        (ismissing(rec.KM) || rec.KM <= 0) && continue
        sub = String(rec.substrate)
        key = (String(rec.ec_code), lowercase(sub))
        haskey(KM_rows, key) || continue
        g = lowercase(String(rec.species))
        push!(y, log(rec.KM / 1000))    # mM -> M
        push!(grp, g)
        push!(meta, (kind="KM", ec_code=key[1], substrate=sub,
                     Q_row=KM_rows[key], group_id=g))
    end

    σ2 = fill((0.5 * log(10))^2, length(y))
    Arows = Matrix{Float64}(Q[meta.Q_row, :])

    return (; rxns, Q, array_species, array_reactions,
            y, σ2, Arows, grp, meta, A0, b0,
            q_prior, C_prior, q_post, C_post, Q_obs, μ_obs, μ_cov)
end

function save_pathway_inputs(build, spec::PathwaySpec, outdir::String; force::Bool=false)
    data_dir = joinpath(outdir, spec.name, "data")
    mkpath(data_dir)
    brenda_path = joinpath(data_dir, "brenda_inputs_$(spec.name).jld2")
    posterior_path = joinpath(data_dir, "posterior_$(spec.name).jld2")
    for p in (brenda_path, posterior_path)
        if isfile(p) && !force
            error("refusing to overwrite existing file $(p); pass --force to allow")
        end
    end
    jldsave(brenda_path;
        y = build.y, σ2 = build.σ2, Arows = build.Arows, grp = build.grp,
        meta = build.meta, A0 = build.A0, b0 = build.b0)
    jldsave(posterior_path;
        q_prior = build.q_prior, C_prior = build.C_prior,
        q_post = build.q_post, C_post = build.C_post,
        Q_obs = build.Q_obs, μ_cov = build.μ_cov)
    println("wrote $(brenda_path)")
    println("wrote $(posterior_path)")
    return (brenda_path, posterior_path)
end

# Verification against the reference files

_maxabs(a, b) = isempty(a) ? 0.0 : maximum(abs.(Float64.(a) .- Float64.(b)))

# Max abs difference relative to the largest magnitude in the reference array.
function _relscale(a, b)
    isempty(a) && return 0.0
    scale = max(maximum(abs.(Float64.(b))), eps())
    return _maxabs(a, b) / scale
end

"""
    _blockdiag_matrix(x) -> Matrix{Float64}

Dense matrix from a BlockDiagonal, including instances serialized by older
BlockDiagonals versions that JLD2 loads as reconstructed types.
"""
function _blockdiag_matrix(x)
    if x isa AbstractMatrix
        return Matrix{Float64}(x)
    end
    blocks = [Matrix{Float64}(b) for b in x.blocks]
    n = sum(size(b, 1) for b in blocks)
    M = zeros(n, n)
    o = 0
    for b in blocks
        k = size(b, 1)
        M[o+1:o+k, o+1:o+k] = b
        o += k
    end
    return M
end

function verify_pathway(build, spec::PathwaySpec; refdir::String="output")
    ref_brenda = joinpath(refdir, spec.name, "data", "brenda_inputs_$(spec.name).jld2")
    ref_posterior = joinpath(refdir, spec.name, "data", "posterior_$(spec.name).jld2")
    ok = true

    # Verdict per array: bit-identical, or within floating-point noise of the
    # linear-algebra pipeline (max abs diff below 1e-10 relative to the array
    # scale, from Σ solves / matrix inversions on a different machine/BLAS).
    fp_tol = 1e-10

    println("\n== verification: $(spec.name) vs $(ref_brenda) ==")
    ref = load(ref_brenda)
    if length(build.y) != length(ref["y"])
        println("  SIZE MISMATCH: rebuilt n=$(length(build.y)) vs reference n=$(length(ref["y"]))")
        return false
    end
    for (nm, a, b) in [("y", build.y, ref["y"]), ("σ2", build.σ2, ref["σ2"]),
                       ("Arows", build.Arows, ref["Arows"]),
                       ("A0", build.A0, ref["A0"]), ("b0", build.b0, ref["b0"])]
        δ = _maxabs(a, b)
        ρ = _relscale(a, b)
        ok &= ρ <= fp_tol
        tag = δ == 0.0 ? "bit-identical" : (ρ <= fp_tol ? "fp-precision" : "MISMATCH")
        println("  max|Δ| $(nm) = $(δ) (rel $(ρ)) [$(tag)]")
    end
    g_eq = build.grp == Vector{String}(ref["grp"])
    ok &= g_eq
    println("  grp identical = $(g_eq)")
    mref = ref["meta"]
    for c in [:kind, :ec_code, :substrate, :Q_row, :group_id]
        c_eq = build.meta[!, c] == mref[!, c]
        ok &= c_eq
        println("  meta.$(c) identical = $(c_eq)")
    end

    println("== verification: $(spec.name) vs $(ref_posterior) ==")
    refp = load(ref_posterior)
    for (nm, a, b) in [("q_prior", build.q_prior, refp["q_prior"]),
                       ("C_prior", build.C_prior, refp["C_prior"]),
                       ("q_post", build.q_post, refp["q_post"]),
                       ("C_post", build.C_post, refp["C_post"]),
                       ("Q_obs", build.Q_obs, refp["Q_obs"]),
                       ("μ_cov", _blockdiag_matrix(build.μ_cov), _blockdiag_matrix(refp["μ_cov"]))]
        δ = _maxabs(a, b)
        ρ = _relscale(a, b)
        ok &= ρ <= fp_tol
        tag = δ == 0.0 ? "bit-identical" : (ρ <= fp_tol ? "fp-precision" : "MISMATCH")
        println("  max|Δ| $(nm) = $(δ) (rel $(ρ)) [$(tag)]")
    end
    println(ok ? "  RESULT: match (bit-identical or floating-point precision)" :
                 "  RESULT: differences found (see above)")
    return ok
end

# Main

function main(args)
    if isempty(args) || args[1] in ("-h", "--help")
        println("usage: julia --project=. scripts/build_pathway_inputs.jl <pathway> [--outdir=DIR] [--force] [--verify]")
        println("pathways: ", join(sort(collect(keys(PATHWAY_SPECS))), ", "))
        return
    end
    pathway = args[1]
    haskey(PATHWAY_SPECS, pathway) ||
        error("unknown pathway '$(pathway)'; available: $(join(sort(collect(keys(PATHWAY_SPECS))), ", ")). " *
              "Add a PathwaySpec to PATHWAY_SPECS for a new pathway.")
    outdir = "output"
    force = false
    verify = false
    for a in args[2:end]
        if startswith(a, "--outdir=")
            outdir = a[length("--outdir=")+1:end]
        elseif a == "--force"
            force = true
        elseif a == "--verify"
            verify = true
        else
            error("unknown argument $(a)")
        end
    end

    spec = PATHWAY_SPECS[pathway]
    println("loading yeast-GEM reconstruction ...")
    mdl = load_yeast_reconstruction(joinpath(dirname(@__DIR__), "data", "yeast-GEM.xml"))
    println("building $(pathway) ...")
    build = build_pathway_inputs(spec; mdl=mdl)
    println("  d = $(size(build.Q, 2)) parameters, $(size(build.Q, 1)) observables")
    println("  records: n = $(length(build.y)) " *
            "($(sum(build.meta.kind .== "kcat")) kcat + $(sum(build.meta.kind .== "KM")) KM), " *
            "$(length(unique(build.grp))) sources")
    save_pathway_inputs(build, spec, outdir; force=force)
    if verify
        verify_pathway(build, spec)
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
