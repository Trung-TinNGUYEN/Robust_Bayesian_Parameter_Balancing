# run_revision_record_range.jl
#
# Adds the range of each released quantity's own records to the released parameter
# sets, and flags rows whose consensus median lies outside that range.  A consensus
# can leave its records when a sparsely measured quantity is coupled through the
# Haldane relationships to a better-measured one, so the flag tells a modeller where
# the balanced value and the direct measurements disagree.  No model is refitted.
#
# Input:  output/<pathway>/data/brenda_inputs_<pathway>.jld2 and the released tables
# Output: record_min, record_max and consensus_outside_records columns appended to
#         S1_Table.csv (all records) and S26_Table.csv (records not flagged as mutants)
#
# Environment controls:
#   PB_RECORD_RANGE_DIR     folder that holds the two tables (default the folder set
#                           in SUB below)
#   PB_RECORD_RANGE_TABLES  the two table files, all records first, comma-separated
#                           (default the two named above)

import Pkg
Pkg.activate(dirname(@__DIR__))

using CSV, DataFrames, JLD2, Printf, Statistics
include(joinpath(dirname(@__DIR__), "src", "mutant_flags.jl"))

const SUB = get(ENV, "PB_RECORD_RANGE_DIR", joinpath(dirname(@__DIR__), "supporting_tables"))
const TABLES = String.(strip.(split(get(ENV, "PB_RECORD_RANGE_TABLES", "S1_Table.csv,S26_Table.csv"), ",")))
length(TABLES) == 2 || error("PB_RECORD_RANGE_TABLES must name two tables")
to_natural(kind, v) = kind == "kcat" ? exp(v) : exp(v) * 1000

function record_ranges(pathway::Symbol; wildtype::Bool)
    b = load(joinpath(dirname(@__DIR__), "output", string(pathway), "data", "brenda_inputs_$(pathway).jld2"))
    y = Vector{Float64}(b["y"]); org = Vector{String}(b["grp"]); meta = b["meta"]
    kinds = Vector{String}(getproperty(meta, :kind)); ecs = string.(getproperty(meta, :ec_code))
    subs = string.(getproperty(meta, :substrate)); qrows = Vector{Int}(getproperty(meta, :Q_row))
    keep = collect(eachindex(y))
    if wildtype
        mm = mutant_mask(dirname(@__DIR__), CSVNAME_OF[pathway], y, org, kinds, ecs, subs)
        keep = findall(.!mm.mutant)
    end
    out = Dict{Tuple{Int,String},Tuple{Float64,Float64,Int}}()
    for i in keep
        k = (qrows[i], kinds[i]); v = to_natural(kinds[i], y[i])
        lo, hi, n = get(out, k, (Inf, -Inf, 0))
        out[k] = (min(lo, v), max(hi, v), n + 1)
    end
    return out
end

for (file, wildtype) in ((TABLES[1], false), (TABLES[2], true))
    path = joinpath(SUB, file)
    df = DataFrame(CSV.File(path))
    for c in (:record_min, :record_max, :consensus_outside_records)
        c in propertynames(df) && select!(df, Not(c))
    end
    rmin = Vector{Float64}(undef, nrow(df)); rmax = similar(rmin); outside = falses(nrow(df))
    cache = Dict{String,Any}(); mismatch = 0
    for (r, row) in enumerate(eachrow(df))
        pw = string(row.pathway)
        rg = get!(() -> record_ranges(Symbol(pw); wildtype = wildtype), cache, pw)
        lo, hi, n = get(rg, (Int(row.Q_row), string(row.kind)), (NaN, NaN, 0))
        n == row.n_records || (mismatch += 1)
        rmin[r] = lo; rmax[r] = hi
        outside[r] = row.consensus_median < lo || row.consensus_median > hi
    end
    df.record_min = rmin; df.record_max = rmax; df.consensus_outside_records = outside
    CSV.write(path, df)
    @printf("%s: %d rows, record-count mismatches %d, consensus outside its records in %d\n", file, nrow(df), mismatch, count(outside))
    for pw in unique(df.pathway)
        s = df[(df.pathway .== pw), :]
        @printf("  %-18s %d of %d (kcat %d, KM %d); flagged rows with at most 3 records: %d\n", pw, count(s.consensus_outside_records), nrow(s),
            count(s.consensus_outside_records .& (s.kind .== "kcat")), count(s.consensus_outside_records .& (s.kind .== "KM")),
            count(s.consensus_outside_records .& (s.n_records .<= 3)))
    end
    f = df[df.consensus_outside_records, :]
    for row in eachrow(first(sort(f, :n_records), min(6, nrow(f))))
        @printf("    %s %s %s %s: consensus %.3g, records %.3g to %.3g (n=%d)\n", row.pathway, row.kind, row.ec_code, row.substrate, row.consensus_median, row.record_min, row.record_max, row.n_records)
    end
end
println("RECORD RANGE DONE")
