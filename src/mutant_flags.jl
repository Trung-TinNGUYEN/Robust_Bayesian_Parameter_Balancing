# mutant_flags.jl
#
# Which curated records are site-directed mutants?
#
# BRENDA's free-text commentary, kept in the `metadata` column of the
# extraction, names the enzyme form a record was measured on: comments such as
# "#12# mutant S151C, pH 7.0" or "#3# wild-type enzyme".  The record-level
# input builder never reads that column, so every analysis in the paper pools
# wild-type and engineered enzymes.  A site-directed mutant is not a
# measurement of the wild-type parameter, so this file supplies the flag
# needed to quantify the consequence and to repeat the analyses without them.
#
# The flag is conservative in the direction that matters for a sensitivity
# analysis: a record counts as a mutant only when the comment names a mutant
# or a mutation and does NOT also name the wild type, so records whose comment
# compares a mutant with the wild-type value are kept.
#
# Alignment to the record-level inputs is the same greedy exact match used by
# run_revision_publication.jl (organism, EC number, substrate and log value, in
# CSV file order), and every record must be matched or the alignment is
# rejected.

using CSV, DataFrames

const MUTANT_PATTERNS = [
    r"\bmutant\b", r"\bmutants\b", r"\bmutation\b", r"\bmutated\b",
    r"\bvariant\b", r"\bchimera\b", r"\btruncat", r"\bdeletion mutant\b",
    r"\b[ACDEFGHIKLMNPQRSTVWY]\d{1,4}[ACDEFGHIKLMNPQRSTVWY]\b",  # e.g. S151C
]
const WILDTYPE_PATTERNS = [r"\bwild[- ]?type\b", r"\bwildtype\b", r"\bwt\b"]

"True when the comment names a mutant form and not the wild type."
function is_mutant_comment(comment::AbstractString)
    c = lowercase(comment)
    any(occursin(p, c) for p in WILDTYPE_PATTERNS) && return false
    # the residue-code pattern is case sensitive, so test it on the raw text
    any(occursin(p, c) for p in MUTANT_PATTERNS[1:end-1]) && return true
    occursin(MUTANT_PATTERNS[end], comment)
end

"Comment names the wild type explicitly."
is_wildtype_comment(comment::AbstractString) =
    any(occursin(p, lowercase(comment)) for p in WILDTYPE_PATTERNS)

"""
    record_comments(dir, csvname, y, grp, kinds, ecs, subs) -> Vector{String}

The BRENDA comment of every record of the record-level input file, in input
order, recovered by aligning the extraction CSVs to the inputs.
"""
function record_comments(dir::String, csvname::String, y, grp, kinds, ecs, subs)
    out = fill("", length(y))
    for (kind, file, col) in (("kcat", "kcat_values_$(csvname).csv", :kcat),
                              ("KM", "KM_values_$(csvname).csv", :KM))
        df = CSV.read(joinpath(dir, "data", "brenda", file), DataFrame; stringtype = String)
        idx = findall(==(kind), kinds)
        j = 1
        for r in eachrow(df)
            j > length(idx) && break
            i = idx[j]
            v = r[col]
            (v isa Number && v > 0) || continue
            ylog = kind == "kcat" ? log(Float64(v)) : log(Float64(v) / 1000)
            if lowercase(string(r.species)) == grp[i] && string(r.ec_code) == ecs[i] &&
               lowercase(string(r.substrate)) == lowercase(subs[i]) &&
               isapprox(ylog, y[i]; atol = 1e-9)
                out[i] = string(coalesce(r.metadata, ""))
                j += 1
            end
        end
        j - 1 == length(idx) ||
            error("mutant-flag alignment failed for $kind in $file: matched $(j-1) of $(length(idx))")
    end
    out
end

"Mutant flag per record of the record-level input file."
function mutant_mask(dir::String, csvname::String, y, grp, kinds, ecs, subs)
    cs = record_comments(dir, csvname, y, grp, kinds, ecs, subs)
    (mutant = BitVector(is_mutant_comment.(cs)),
     wildtype = BitVector(is_wildtype_comment.(cs)),
     comments = cs)
end

const CSVNAME_OF = Dict(:glycolysis => "glycolysis", :tca => "tca_cycle",
                        :pentose_phosphate => "pentose_phosphate",
                        :shikimate => "shikimate")

"Literature reference indices in a BRENDA comment, e.g. the text <12,13> gives 12 and 13."
function parse_refs_local(comment::AbstractString)
    out = String[]
    for m in eachmatch(r"<([0-9,\s]+)>", comment)
        for r in split(m.captures[1], ",")
            t = strip(r)
            isempty(t) || push!(out, t)
        end
    end
    unique(out)
end
