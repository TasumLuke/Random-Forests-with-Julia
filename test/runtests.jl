using Test, Random, DataFrames, CSV
using EWSF

@testset "Drift adaptation changes weights" begin
    # small synthetic dataset
    Random.seed!(999)
    n = 200
    age = randn(n) * 5 .+ 50
    chol = randn(n) * 20 .+ 200
    smoker = [ rand() < 0.2 ? "yes" : "no" for i in 1:n ]
    target = [ (age[i] > 52 || chol[i] > 230 || smoker[i] == "yes") ? "d" : "h" for i in 1:n ]
    train = DataFrame(age=age, cholesterol=chol, smoker=smoker, target=target)

    (train_clean, enc) = EWSF.EWSFData.data_preprocess(train, "target", ["age","cholesterol","smoker"])
    model = EWSF.EWSFModel.train_ewsf(train_clean, enc; n_trees=30, n_subforests=3, max_depth=5)
    weights_before = [sf.weight for sf in model.subforests]

    # create drift by shifting age and smoker rate
    new = deepcopy(train)
    new.age .= new.age .+ 4.0
    for i in 1:n
        if rand() < 0.3
            new.smoker[i] = "yes"
        end
    end

    # build preprocessed new to align
    map_smoker = enc.feature_encoders[:smoker]
    new_proc = DataFrame(age = Float64[], cholesterol=Float64[], smoker=Int[])
    for r in eachrow(new)
        push!(new_proc, (Float64(r.age), Float64(r.cholesterol), get(map_smoker, r.smoker, 0)))
    end
    rename!(new_proc, [:age, :cholesterol, :smoker])
    # mimic train_clean structure for adapt_weights!
    summary = EWSF.EWSFModel.adapt_weights!(model, train_clean, new_proc; p_threshold=0.05, n_perm=200, lambda=2.5)
    weights_after = [sf.weight for sf in model.subforests]

    @test sum(abs.(weights_after .- weights_before)) > 1e-6  # weights must change
end
