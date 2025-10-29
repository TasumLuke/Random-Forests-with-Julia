module EWSF

# Public API
# Core deps
using CSV, DataFrames, CategoricalArrays, StatsBase, Random, Serialization, Statistics

# include components
include("utils.jl")
include("data.jl")
include("tree.jl")
include("model.jl")

# export main functions and types
export train_ewsf, predict_model, save_model, load_model, adapt_weights!, EWSFModel, SubForest

end # module
