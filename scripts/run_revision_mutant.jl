# run_revision_mutant.jl
#
# Enzyme form: how much of the curated record dispersion is site-directed
# mutants?  Added for the revision.
#
# BRENDA's commentary names the enzyme form a record was measured on, and the
# record-level input builder does not read it, so every analysis in the paper
# pools wild-type and engineered enzymes.  This driver quantifies that, entirely
# without fitting the model, using the same replication design as the
# record-scale estimator (mutant_flags.jl for the flag).
#
# Reported per pathway and quantity type:
#   the share of records that are mutant-only and wild-type-labelled;
#   the fit-free replication record scale c estimated from all replicate
#   groups, from groups containing no mutant record, and from groups whose
#   records are all mutants, which separates "mutants are noisier" from
#   "mutants shift the value";
#   the median log value of mutant and non-mutant records of the same
#   derived quantity, paired within quantity, which is the size of the shift
#   a pooled analysis absorbs;
#   the two-class population fitted to mutant-free groups alone.
#
# Output: output/mutant_structure_all.csv

import Pkg
Pkg.activate(dirname(@__DIR__))

using CSV, DataFrames, JLD2, Printf, Statistics

include(joinpath(dirname(@__DIR__), "src", "mutant_flags.jl"))

const SIG2_CONV = (0.5 * log(10))^2
const PATHWAYS = [:glycolysis, :tca, :pentose_phosphate, :shikimate]

"Fit-free replication scale over the replicate groups selected by `pick`."
function rep_scale(y, sigma2, repkey, mutant, idx, pick::Symbol)
    groups = Dict{String,Vector{Int}}()
    for i in idx
        push!(get!(groups, repkey[i], Int[]), i)
    end
    num = 0.0; den = 0.0; ngrp = 0
    for (_, ig) in groups
        length(ig) >= 2 || continue
        nm = count(mutant[i] for i in ig)
        pick == :nomutant && nm > 0 && continue
        pick == :allmutant && nm < length(ig) && continue
        ybar = mean(y[i] for i in ig)
        for i in ig
            num += (y[i] - ybar)^2
            den += sigma2[i] * (1.0 - 1.0 / length(ig))
        end
        ngrp += 1
    end
    (den > 0 ? num / den : NaN, ngrp)
end

rows = NamedTuple[]
for p in PATHWAYS
    b = load(joinpath(dirname(@__DIR__), "output", string(p), "data", "brenda_inputs_$(p).jld2"))
    y = Vector{Float64}(b["y"]); sigma2 = Vector{Float64}(b["σ2"])
    grp = Vector{String}(b["grp"]); meta = b["meta"]
    kinds = Vector{String}(meta.kind); ecs = string.(meta.ec_code)
    subs = string.(meta.substrate); qrows = string.(meta.Q_row)
    mm = mutant_mask(dirname(@__DIR__), CSVNAME_OF[p], y, grp, kinds, ecs, subs)
    repkey = [string(grp[i], "|", qrows[i]) for i in eachindex(y)]

    for kind in ("all", "kcat", "KM")
        idx = kind == "all" ? collect(eachindex(y)) : findall(==(kind), kinds)
        c_all, n_all = rep_scale(y, sigma2, repkey, mm.mutant, idx, :any)
        c_nom, n_nom = rep_scale(y, sigma2, repkey, mm.mutant, idx, :nomutant)
        c_mut, n_mut = rep_scale(y, sigma2, repkey, mm.mutant, idx, :allmutant)
        # paired shift: within a derived quantity, median of mutant minus
        # median of non-mutant records, over quantities holding both
        byq = Dict{String,Vector{Int}}()
        for i in idx
            push!(get!(byq, qrows[i], Int[]), i)
        end
        shifts = Float64[]
        for (_, iq) in byq
            mu = [y[i] for i in iq if mm.mutant[i]]
            nm = [y[i] for i in iq if !mm.mutant[i]]
            (length(mu) >= 3 && length(nm) >= 3) || continue
            push!(shifts, median(mu) - median(nm))
        end
        push!(rows, (pathway = string(p), subset = kind, n_records = length(idx),
            n_mutant = count(mm.mutant[idx]), share_mutant = mean(mm.mutant[idx]),
            n_wildtype_labelled = count(mm.wildtype[idx]),
            share_wildtype_labelled = mean(mm.wildtype[idx]),
            c_all_groups = c_all, n_groups_all = n_all,
            c_mutant_free_groups = c_nom, n_groups_mutant_free = n_nom,
            c_all_mutant_groups = c_mut, n_groups_all_mutant = n_mut,
            n_quantities_paired = length(shifts),
            median_log_shift_mutant_minus_wildtype = isempty(shifts) ? NaN : median(shifts),
            fold_shift = isempty(shifts) ? NaN : exp(median(shifts))))
        @printf("%-18s %-4s n=%5d mutant=%4.1f%%  c: all=%5.2f (%3d grp) mutant-free=%5.2f (%3d) all-mutant=%5.2f (%3d)  median shift=%+.2f log (%.2f-fold, %d quantities)\n",
                string(p), kind, length(idx), 100mean(mm.mutant[idx]), c_all, n_all,
                c_nom, n_nom, c_mut, n_mut,
                isempty(shifts) ? NaN : median(shifts), isempty(shifts) ? NaN : exp(median(shifts)),
                length(shifts))
    end
end
df = DataFrame(rows)
CSV.write(joinpath(dirname(@__DIR__), "output", "mutant_structure_all.csv"), df)
println("wrote output/mutant_structure_all.csv")
