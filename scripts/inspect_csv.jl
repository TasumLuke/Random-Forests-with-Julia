# scripts/inspect_csv.jl
using Pkg
Pkg.activate("..")
using CSV, DataFrames, JSON3, Statistics

# Args: csv_path
csv_path = ARGS[1]
df = CSV.read(csv_path, DataFrame; missingstring="", ignorerepeated=true)
cols = names(df)
# take first 5 rows as preview
preview = Dict{String,Any}()
n = min(5, nrow(df))
for c in cols
    preview[string(c)] = Vector{Any}()
end
for i in 1:n
    for c in cols
        push!(preview[string(c)], df[i, c])
    end
end

out = Dict("columns" => [ string(c) for c in cols ], "preview" => [ Dict(k=>preview[k][i] for k in keys(preview)) for i in 1:n ])
println(JSON3.write(out))
