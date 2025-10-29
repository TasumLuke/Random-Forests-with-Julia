module EWSFData
using CSV, DataFrames, CategoricalArrays, Statistics, StatsBase
export data_load_csv, data_preprocess, Encoders

mutable struct Encoders
    feature_encoders::Dict{Symbol,Dict{Any,Int}}   # maps categorical values -> ints
    target_encoder::Dict{Any,Int}
    inv_target::Dict{Int,Any}
    target_type::Symbol
    feature_types::Dict{Symbol,Symbol}             # :numeric or :categorical
    feature_order::Vector{Symbol}
end

# load CSV file
function data_load_csv(path::AbstractString)
    df = CSV.read(path, DataFrame)
    return df
end

# decide whether column is numeric
is_numeric_col(col) = eltype(col) <: Number || all(x->(x===missing ? true : tryparse(Float64, string(x)) !== nothing), col)

# preprocess: returns (processed_df, encoders)
function data_preprocess(df::DataFrame, target_col::AbstractString, feature_cols::Vector{AbstractString})
    features_sym = Symbol.(feature_cols)
    target_sym = Symbol(target_col)
    sub = copy(df[:, [feature_cols... , target_col]])
    fenc = Dict{Symbol,Dict{Any,Int}}()
    ftypes = Dict{Symbol,Symbol}()

    for f in features_sym
        col = sub[!, f]
        if is_numeric_col(col)
            ftypes[f] = :numeric
            arr = [ x===missing ? missing : tryparse(Float64,string(x)) for x in col ]
            med = median(skipmissing(arr))
            sub[!, f] = [ x===missing ? med : Float64(x) for x in arr ]
        else
            ftypes[f] = :categorical
            c = categorical(sub[!, f])
            lvls = levels(c)
            mapdict = Dict{Any,Int}()
            for (i,l) in enumerate(lvls); mapdict[l]=i; end
            fenc[f] = mapdict
            sub[!, f] = [ mapdict[string(x)] for x in c ]
        end
    end

    # target
    tcol = sub[!, target_sym]
    if is_numeric_col(tcol)
        arr = [x===missing ? missing : tryparse(Float64,string(x)) for x in tcol]
        if length(unique(skipmissing(arr))) <= 10
            levels = unique(skipmissing(arr))
            tmap = Dict{Any,Int}()
            invt = Dict{Int,Any}()
            i=1
            for v in levels; tmap[v]=i; invt[i]=v; i+=1 end
            sub[!, target_sym] = [ tmap[x] for x in arr ]
            ttype = :classification
            tenc = tmap
            inv = invt
        else
            # treat as numeric (not supported for EWSF classifier)
            sub[!, target_sym] = [ x===missing ? median(skipmissing(arr)) : Float64(x) for x in arr ]
            ttype = :numeric
            tenc = Dict{Any,Int}()
            inv = Dict{Int,Any}()
        end
    else
        c = categorical(tcol)
        lvls = levels(c)
        tmap = Dict{Any,Int}()
        invt = Dict{Int,Any}()
        for (i,l) in enumerate(lvls); tmap[l]=i; invt[i]=l; end
        sub[!, target_sym] = [ tmap[string(x)] for x in c ]
        ttype = :classification
        tenc = tmap
        inv = invt
    end

    enc = Encoders(fenc, tenc, inv, ttype, ftypes, features_sym)
    return (sub, enc)
end

end # module
