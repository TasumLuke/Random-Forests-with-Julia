# scripts/train_api.jl
using Pkg
Pkg.activate("..")
using CSV, DataFrames, JSON3, EWSF

# ARGS:
# 1 csv_path
# 2 target_col
# 3 features_csv (comma-separated)
# 4 out_model_path
# 5 n_trees
# 6 max_depth
# 7 min_leaf
# 8 n_subforests

csv_path = ARGS[1]
target_col = ARGS[2]
features = split(ARGS[3], ',')
out_model = ARGS[4]
n_trees = parse(Int, ARGS[5])
max_depth = parse(Int, ARGS[6])
min_leaf = parse(Int, ARGS[7])
n_subforests = parse(Int, ARGS[8])

try
    df = CSV.read(csv_path, DataFrame)
    (dfclean, enc) = EWSF.EWSFData.data_preprocess(df, target_col, features)
    model = EWSF.EWSFModel.train_ewsf(dfclean, enc; n_trees=n_trees, max_depth=max_depth, min_leaf=min_leaf, n_subforests=n_subforests)
    # Save model to out_model (path provided by server)
    open(out_model, "w") do io
        serialize(io, model)
    end
    println(JSON3.write(Dict("status"=>"ok", "model"=>out_model)))
catch e
    println(JSON3.write(Dict("status"=>"error", "message"=>string(e))))
    exit(1)
end
