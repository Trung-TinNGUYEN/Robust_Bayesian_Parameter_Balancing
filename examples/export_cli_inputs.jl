# Export one pathway's record-level inputs as the CSV files the command-line tool reads,
# with a per-record label table so that every row of Q can be traced to its quantity.
# Usage: julia --project=. examples/export_cli_inputs.jl <repo root> <pathway> <output dir>
import Pkg; Pkg.activate(ARGS[1])
using JLD2, CSV, DataFrames
root, pw, out = ARGS[1], ARGS[2], ARGS[3]; mkpath(out)
b = load(joinpath(root, "output", pw, "data", "brenda_inputs_$(pw).jld2"))
p = load(joinpath(root, "output", pw, "data", "posterior_$(pw).jld2"))
y = Vector{Float64}(b["y"]); s2 = Vector{Float64}(b["σ2"]); A = Matrix{Float64}(b["Arows"])
org = Vector{String}(b["grp"]); meta = b["meta"]
kinds = string.(getproperty(meta, :kind)); qrows = string.(getproperty(meta, :Q_row))
ecs = string.(getproperty(meta, :ec_code)); subs = string.(getproperty(meta, :substrate))
CSV.write(joinpath(out, "Q.csv"), DataFrame(A, :auto))
CSV.write(joinpath(out, "x.csv"), DataFrame(x = y))
CSV.write(joinpath(out, "Cx.csv"), DataFrame(var = s2))
CSV.write(joinpath(out, "q_prior.csv"), DataFrame(q = vec(Array(p["q_prior"]))))
CSV.write(joinpath(out, "C_prior.csv"), DataFrame(Matrix(Array(p["C_prior"])), :auto))
# The released specification's random intercept is per organism-by-type cell.
CSV.write(joinpath(out, "groups.csv"), DataFrame(group = [string(org[i], "|", kinds[i]) for i in eachindex(y)]))
CSV.write(joinpath(out, "replicates.csv"), DataFrame(rep = [string(org[i], "|", qrows[i]) for i in eachindex(y)]))
CSV.write(joinpath(out, "types.csv"), DataFrame(type = kinds))
CSV.write(joinpath(out, "labels.csv"), DataFrame(row = eachindex(y), organism = org, kind = kinds,
                                                 ec_code = ecs, substrate = subs, Q_row = qrows))
println("exported ", length(y), " records and ", size(A, 2), " basis coordinates for ", pw, " to ", out)
