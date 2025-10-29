# run_train.jl
# Simple entrypoint to start interactive training.
using Pkg
Pkg.activate(".")  # optional if you created a project
using CSV, DataFrames

include("src/main.jl")
EWSF = Main.EWSF

# CLI entry
EWSF.run_train()
