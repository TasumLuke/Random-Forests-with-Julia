# scripts/predict_api.jl
using Pkg
Pkg.activate("..")
using CSV, DataFrames, JSON3, Serialization, EWSF

csv_path = ARGS[1]
model_path = ARGS[2]
out_predictions = ARGS[3]

try
    df_new = CSV.read(csv_path, DataFrame)
    model = open(model_path) do io
        deserialize(io)
    end
    pred_df = EWSF.EWSFModel.predict_model(model, df_new)
    CSV.write(out_predictions, pred_df)
    println(JSON3.write(Dict("status"=>"ok", "predictions"=>out_predictions)))
catch e
    println(JSON3.write(Dict("status"=>"error", "message"=>string(e))))
    exit(1)
end
