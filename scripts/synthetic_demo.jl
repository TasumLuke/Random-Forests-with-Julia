# scripts/synthetic_demo.jl
using Pkg
Pkg.activate(".")
using CSV, DataFrames, Random, Statistics
using EWSF

# Generate synthetic training data
function generate_data(n=1000)
    Random.seed!(123)
    age = randn(n) * 10 .+ 50     # numeric
    chol = randn(n) * 30 .+ 200
    smoker = rand(n) .< 0.3
    smoker_s = ifelse.(smoker, "yes", "no")
    # target depends on age and chol and smoking
    target = [ (age[i] > 52 || chol[i] > 240 || smoker[i]) ? "disease" : "healthy" for i in 1:n ]
    df = DataFrame(age=age, cholesterol=chol, smoker=smoker_s, target=target)
    return df
end

train = generate_data(800)
test = generate_data(200)

# Save sample
CSV.write("train_synthetic.csv", train)
CSV.write("test_synthetic.csv", test)

# Preprocess via package helper
(df_clean, enc) = EWSF.EWSFData.data_preprocess(train, "target", ["age","cholesterol","smoker"])

# Train model
model = EWSF.EWSFModel.train_ewsf(df_clean, enc; n_trees=60, n_subforests=3, max_depth=6)

println("Initial subforest weights: ", [sf.weight for sf in model.subforests])

# Create a drifted new dataset: shift age distribution + change smoker proportion
function create_drifted(df)
    df2 = deepcopy(df)
    df2.age .= df2.age .+ 5.0   # mean shift
    # raise smoker fraction
    for i in 1:nrow(df2)
        if rand() < 0.2
            df2.smoker[i] = "yes"
        end
    end
    return df2
end

incoming = create_drifted(test)

# Preprocess incoming using encoders — we will align columns to same format (simulate CSV reading)
# For adaptation we need the original preprocessed train_df and new_df with same columns
train_pre = df_clean
# Build new_df like preprocessed structure:
new_df = DataFrame(age = Float64[], cholesterol=Float64[], smoker=Int[])
# Map smoker categories to integers using encoder map
map_smoker = enc.feature_encoders[:smoker]
for r in eachrow(incoming)
    push!(new_df, (Float64(r.age), Float64(r.cholesterol), get(map_smoker, r.smoker, 0)))
end
rename!(new_df, [:age, :cholesterol, :smoker])

# Also we need to add target column for predict/permutation function convenience; set dummy (not used)
new_df_targeted = deepcopy(train_pre)
new_df_targeted[!, :] .= 0
new_df_targeted[!, 1:size(new_df,2)] = new_df

println("Running adapt_weights! with permutation-based drift detection...")
summary = EWSF.EWSFModel.adapt_weights!(model, train_pre, new_df_targeted; p_threshold=0.05, n_perm=200, lambda=3.0)
println("Drift p-values (sample):")
for (k,v) in first(summary["pvals"], 5)
    println(k, " => ", round(v, digits=4))
end
println("New weights: ", summary["new_weights"])
