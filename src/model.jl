module EWSFModel
using ..EWSFTree
using ..EWSFData
using ..EWSFUtils
using Random, Serialization, Statistics
export train_ewsf, predict_model, save_model, load_model, adapt_weights!, EWSFModel, SubForest

mutable struct SubForest
    trees::Vector{EWSFTree.TreeNode}
    perm::Vector{Int}
    feature_types::Vector{Symbol}
    weight::Float64
end

mutable struct EWSFModel
    subforests::Vector{SubForest}
    encoders::EWSFData.Encoders
    feature_names::Vector{Symbol}
    original_feature_types::Dict{Symbol,Symbol}
end

# Train EWSF with multiple subforests (different alpha emphasis)
function train_ewsf(df::DataFrame, enc::EWSFData.Encoders; n_trees::Int=90, max_depth::Int=8, min_leaf::Int=3, n_subforests::Int=3)
    features = enc.feature_order
    p = length(features)
    # Build X, y from preprocessed df (features then target at end)
    X = convert(Matrix{Float64}, Matrix(df[:, features]))
    # find target col as the remaining column
    target_cols = setdiff(names(df), features)
    if length(target_cols) != 1
        error("Preprocessed df expected to contain features then exactly one target column.")
    end
    y = Vector{Int}(df[:, target_cols[1]])
    ftypes = [ enc.feature_types[f] for f in features ]

    base_scores = EWSFTree.feature_scores(X, ftypes)
    subforests = Vector{SubForest}()
    # choose alphas to produce variety
    alphas = range(0.5, stop=2.0, length=n_subforests)
    rng = MersenneTwister(42)
    trees_per_sub = max(1, Int(round(n_trees / n_subforests)))
    for alpha in alphas
        # build permutation biased by scores^alpha
        scores = base_scores .^ alpha
        if sum(scores) == 0.0
            probs = fill(1.0/length(scores), length(scores))
        else
            probs = scores ./ sum(scores)
        end
        perm = sample(collect(1:p), Weights(probs), p; replace=false)
        X_perm = X[:, perm]
        ftypes_perm = [ ftypes[j] for j in perm ]
        trees = EWSFTree.train_forest(X_perm, y, ftypes_perm; n_trees=trees_per_sub, max_depth=max_depth, min_leaf=min_leaf, alpha=1.0, rng=rng)
        push!(subforests, SubForest(trees, perm, ftypes_perm, 1.0 / n_subforests))
    end
    model = EWSFModel(subforests, enc, features, enc.feature_types)
    return model
end

# prediction
function predict_model(model::EWSFModel, newdf::DataFrame)
    enc = model.encoders
    features = model.feature_names
    n = nrow(newdf)
    p = length(features)
    # build Xnew matrix
    Xnew = zeros(n, p)
    for (j,f) in enumerate(features)
        if enc.feature_types[f] == :numeric
            for i in 1:n
                v = newdf[i, string(f)]
                Xnew[i,j] = tryparse(Float64,string(v)) === nothing ? 0.0 : Float64(v)
            end
        else
            mapdict = get(enc.feature_encoders, f, Dict{Any,Int}())
            for i in 1:n
                v = newdf[i, string(f)]
                Xnew[i,j] = get(mapdict, v, 0)
            end
        end
    end

    # per-subforest predictions (majority vote within subforest)
    m = length(model.subforests)
    sub_preds = [ Vector{Int}() for _ in 1:m ]
    for (si, sf) in enumerate(model.subforests)
        Xp = Xnew[:, sf.perm]
        # predict for each tree, then majority among trees
        preds_matrix = [ [ EWSFTree.predict_tree(tree, view(Xp,i,:)) for i in 1:size(Xp,1) ] for tree in sf.trees ]
        # majority vote per sample
        pvec = [ EWSFUtils.mode_label([ preds_matrix[t][i] for t in 1:length(preds_matrix) ]) for i in 1:size(Xp,1) ]
        sub_preds[si] = pvec
    end

    # weighted ensemble vote
    final_preds = Vector{Int}(undef, n)
    for i in 1:n
        votes = Dict{Int,Float64}()
        for (si,sf) in enumerate(model.subforests)
            lab = sub_preds[si][i]
            votes[lab] = get(votes, lab, 0.0) + sf.weight
        end
        final_preds[i] = findmax(collect(values(votes)))[2] # this returns index not label => fix:
        # better: choose key with maximum value
        best_lab = nothing; best_w = -Inf
        for (k,v) in votes
            if v > best_w
                best_lab = k; best_w = v
            end
        end
        final_preds[i] = best_lab
    end

    # map back to original labels
    inv = model.encoders.inv_target
    return [ get(inv, p, p) for p in final_preds ]
end

# save & load
function save_model(path::AbstractString, model::EWSFModel)
    open(path,"w") do io
        serialize(io, model)
    end
end

function load_model(path::AbstractString)
    return open(path,"r") do io
        deserialize(io)
    end
end

# ---------- DRIFT DETECTION (permutation p-values) ----------
# KS statistic for numeric
function ks_stat(x::Vector{Float64}, y::Vector{Float64})
    allv = sort(unique(vcat(x, y)))
    cdfx = [ sum(x .<= v) / length(x) for v in allv ]
    cdfy = [ sum(y .<= v) / length(y) for v in allv ]
    return maximum(abs.(cdfx .- cdfy))
end

# chi2 stat for categorical encoded as ints
function chi2_stat(x::Vector{Int}, y::Vector{Int})
    cats = unique(vcat(x,y))
    n1 = length(x); n2 = length(y)
    tot = n1 + n2
    counts_tot = Dict{Int,Int}()
    for v in vcat(x,y)
        counts_tot[v] = get(counts_tot, v, 0) + 1
    end
    stat = 0.0
    for c in cats
        obs1 = sum(x .== c)
        exp1 = counts_tot[c] * (n1 / tot)
        if exp1 > 0
            stat += (obs1 - exp1)^2 / exp1
        end
    end
    return stat
end

# permutation p-value wrapper (universal): for numeric (ks) or categorical (chi2)
function permutation_pvalue(x, y; n_perm::Int=500, rng::AbstractRNG=MersenneTwister(123))
    obs_stat = 0.0
    if eltype(x) <: Real && eltype(y) <: Real
        xfloat = Float64.(x); yfloat = Float64.(y)
        obs_stat = ks_stat(xfloat, yfloat)
        combined = vcat(xfloat, yfloat)
        n1 = length(xfloat)
        cnt = 0
        for i in 1:n_perm
            perm = rand(rng, combined)
            s1 = perm[1:n1]; s2 = perm[n1+1:end]
            val = ks_stat(s1, s2)
            if val >= obs_stat
                cnt += 1
            end
        end
        return (cnt + 1) / (n_perm + 1), obs_stat
    else
        # categorical ints
        xi = Int.(x); yi = Int.(y)
        obs_stat = chi2_stat(xi, yi)
        combined = vcat(xi, yi)
        n1 = length(xi)
        cnt = 0
        for i in 1:n_perm
            perm = rand(rng, combined)
            s1 = perm[1:n1]; s2 = perm[n1+1:end]
            val = chi2_stat(s1, s2)
            if val >= obs_stat
                cnt += 1
            end
        end
        return (cnt + 1) / (n_perm + 1), obs_stat
    end
end

# compute p-values per feature between training data (original) and incoming data
function feature_drift_pvalues(model::EWSFModel, train_df::DataFrame, new_df::DataFrame; n_perm::Int=500)
    enc = model.encoders
    features = model.feature_names
    pvals = Dict{Symbol,Float64}()
    stats = Dict{Symbol,Float64}()
    rng = MersenneTwister(2025)
    for f in features
        if enc.feature_types[f] == :numeric
            train_vals = Float64.(train_df[!, string(f)])
            new_vals = Float64.(new_df[!, string(f)])
            pv, st = permutation_pvalue(train_vals, new_vals, n_perm=n_perm, rng=rng)
            pvals[f] = pv; stats[f] = st
        else
            mapdict = get(enc.feature_encoders, f, Dict{Any,Int}())
            train_vals = Int.(train_df[!, string(f)])
            # for new data map unseen -> 0 (new category)
            new_raw = new_df[!, string(f)]
            new_vals = [ get(mapdict, v, 0) for v in new_raw ]
            pv, st = permutation_pvalue(train_vals, new_vals, n_perm=n_perm, rng=rng)
            pvals[f] = pv; stats[f] = st
        end
    end
    return (pvals, stats)
end

# adapt_weights! : reduces subforest weights proportionally to fraction of their features that show drift (p < threshold)
function adapt_weights!(model::EWSFModel, train_df::DataFrame, new_df::DataFrame; p_threshold::Float64=0.05, n_perm::Int=500, lambda::Float64=3.0)
    (pvals, stats) = feature_drift_pvalues(model, train_df, new_df, n_perm=n_perm)
    # compute for each subforest the fraction of features with p < threshold
    drift_scores = Float64[]
    for sf in model.subforests
        pf = sf.perm
        # perm indices correspond to feature positions in model.feature_names
        feat_syms = model.feature_names[pf]
        n = length(feat_syms)
        if n == 0
            push!(drift_scores, 0.0)
            continue
        end
        cnt = 0
        for f in feat_syms
            if pvals[f] < p_threshold
                cnt += 1
            end
        end
        push!(drift_scores, cnt / n)
    end
    # update weights: weight_new = weight_old * exp(-lambda * drift_score)
    new_w = [ sf.weight * exp(-lambda * ds) for (sf, ds) in zip(model.subforests, drift_scores) ]
    total = sum(new_w)
    if total == 0.0
        # fallback uniform
        for sf in model.subforests
            sf.weight = 1.0 / length(model.subforests)
        end
    else
        for (i, sf) in enumerate(model.subforests)
            sf.weight = new_w[i] / total
        end
    end
    # Return summary
    return Dict("pvals" => pvals, "stats" => stats, "drift_scores" => drift_scores, "new_weights" => [sf.weight for sf in model.subforests])
end

end # module
