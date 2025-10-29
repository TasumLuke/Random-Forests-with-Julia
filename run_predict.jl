# run_predict.jl
using Pkg
Pkg.activate(".")
include("src/main.jl")
EWSF = Main.EWSF

EWSF.run_predict()
