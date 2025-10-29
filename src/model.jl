module EWSFModel

using ..EWSFTree
using ..EWSFData
using ..EWSFUtils
using Random, Serialization, Statistics, DataFrames, StatsBase

export train_ewsf, predict_model, save_model, load_model, adapt_weights!, feature_drift_pvalues, permutation_pvalue, EWSFModel, SubForest, TreeWithOOB


# Data structures


mutable struct TreeWithOOB
    tree::EWSFTree.TreeNode
    oob_idxs::Vector{Int}   # indices of training rows that were OOB for this tree
end

mutable struct SubForest
    trees::Vector{TreeWithOOB}
    perm::Vector{Int}
    feature_types::Vector{Symbol}
    weight::Float64
end

mutable struct EWSFModel
    subforests::Vector{SubForest}
    encoders::EWSFData.Encoders
    feature_names::Vector{Symbol}
    original_feature_types::Dict{Symbol,Symbol}
    classes::Vector{Any}   # original class labels (in order corresponding to model internals)
end


# Helper utilities


# Find the most frequent key in a Dict{K,Int} (returns key with max count), or nothing if empty
function _mode_from_counts(d::Dict{Int,Int})
    if isempty(d)
        return nothing
    end
    bestk = first(keys(d))
    bestv = d[bestk]
    for (k,v) in d
        if v > bestv
            bestk = k
            bestv = v
        end
    end
    return bestk
end


# Training (OOB-weighted subforests)


"""
    train_ewsf(df::DataFrame, enc::EWSFData.Encoders; n_trees=90, max_depth=8, min_leaf=3, n_subforests=3)

Train an EWSFModel on the preprocessed DataFrame `df` (output of `data_preprocess`) using encoders `enc`.

- Builds `n_subforests`, each containing `n_trees / n_subforests` trees.
- Each tree is trained on a bootstrap sample and records OOB indices.
- Each subforest's initial weight is proportional to its OOB accuracy (normalized).
"""
function train_ewsf(df::DataFrame, enc::EWSFData.Encoders; n_trees::Int=90, max_depth::Int=8, min_leaf::Int=3, n_subforests::Int=3)
    features = enc.feature_order
    p = length(features)

    # Build X, y from preprocessed df (assumes df contains exactly features then single target column)
    X = convert(Matrix{Float64}, Matrix(df[:, features]))
    target_cols = setdiff(names(df), features)
    if length(target_cols) != 1
        error("Preprocessed df expected to contain features then exactly one target column.")
    end
    y = Vector{Int}(df[:, target_cols[1]])
    ftypes = [ enc.feature_types[f] for f in features ]

    # base feature scores (variance/entropy)
    base_scores = EWSFTree.feature_scores(X, ftypes)
    subforests = Vector{SubForest}()
    alphas = range(0.5, stop=2.0, length=n_subforests)  # different emphasis per subforest
    rng = MersenneTwister(42)
    trees_per_sub = max(1, Int(round(n_trees / max(n_subforests,1))))
    n = size(X,1)

    for alpha in alphas
        # bias permutation by base_scores^alpha
        scores = base_scores .^ alpha
        probs = sum(scores) == 0.0 ? fill(1.0/length(scores), length(scores)) : scores ./ sum(scores)
        perm = sample(collect(1:p), Weights(probs), p; replace=false)
        X_perm = X[:, perm]
        ftypes_perm = [ ftypes[j] for j in perm ]

        # Build trees with OOB tracking
        twobs = Vector{TreeWithOOB}()
        for t in 1:trees_per_sub
            boot_idxs = rand(rng, 1:n, n)                # bootstrap indices with replacement
            oob_idxs = setdiff(1:n, unique(boot_idxs))  # indices not included in bootstrap
            Xb = X_perm[boot_idxs, :]
            yb = y[boot_idxs]
            tree = EWSFTree.build_tree(Xb, yb, ftypes_perm; max_depth=max_depth, min_leaf=min_leaf, m_features=max(1,Int(round(sqrt(p)))), rng=rng)
            push!(twobs, TreeWithOOB(tree, oob_idxs))
        end

        # Compute OOB accuracy for this subforest
        correct = 0
        counted = 0
        for i in 1:n
            votes = Dict{Int,Int}()
            for tow in twobs
                if i in tow.oob_idxs
                    lab = EWSFTree.predict_tree(tow.tree, view(X_perm, i, :))
                    votes[lab] = get(votes, lab, 0) + 1
                end
            end
            if isempty(votes)
                continue
            end
            pred_label = _mode_from_counts(votes)
            counted += 1
            if pred_label == y[i]
                correct += 1
            end
        end
        oob_acc = counted == 0 ? 0.0 : correct / counted
        initial_weight = max(oob_acc, 1e-6)  # small floor so weight isn't zero
        push!(subforests, SubForest(twobs, perm, ftypes_perm, initial_weight))
    end

    # Normalize subforest weights
    total = sum(sf.weight for sf in subforests)
    if total == 0.0
        for sf in subforests
            sf.weight = 1.0 / length(subforests)
        end
    else
        for sf in subforests
            sf.weight = sf.weight / total
        end
    end

    # Prepare class labels in a deterministic order (sort integer keys if possible)
    inv = enc.inv_target
    int_keys = sort(collect(keys(inv)))
    classes = [ inv[k] for k in int_keys ]

    model = EWSFModel(subforests, enc, features, enc.feature_types, classes)
    return model
end


# Prediction (soft probabilities + labels)


"""
    predict_model(model::EWSFModel, newdf::DataFrame) -> DataFrame

Predict class probabilities and label for `newdf` which must have the same feature column names used at training.
Returns a DataFrame with one column per class (probabilities) and a `:prediction` column with highest-probability label.
"""
function predict_model(model::EWSFModel, newdf::DataFrame)
    enc = model.encoders
    features = model.feature_names
    n = nrow(newdf)
    p = length(features)

    # Build numeric matrix Xnew with same feature ordering
    Xnew = zeros(n, p)
    for (j,f) in enumerate(features)
        if enc.feature_types[f] == :numeric
            for i in 1:n
                v = newdf[i, string(f)]
                Xnew[i,j] = tryparse(Float64, string(v)) === nothing ? 0.0 : Float64(v)
            end
        else
            mapdict = get(enc.feature_encoders, f, Dict{Any,Int}())
            for i in 1:n
                v = newdf[i, string(f)]
                Xnew[i,j] = get(mapdict, v, 0)  # unseen -> 0
            end
        end
    end

    # Determine class integer ordering and mapping to columns
    inv = model.encoders.inv_target
    class_ints = sort(collect(keys(inv)))
    labs_ordered = [ inv[i] for i in class_ints ]
    class_to_col = Dict{Int,Int}()
    for (idx, ci) in enumerate(class_ints)
        class_to_col[ci] = idx
    end
    K = length(class_ints)

    # Probability matrix: rows samples x cols classes
    prob_mat = zeros(n, K)

    # For each subforest compute per-sample probabilities (from tree votes) and add weighted contribution
    for sf in model.subforests
        Xp = Xnew[:, sf.perm]  # permute columns to match training
        tcount = length(sf.trees)
        if tcount == 0
            continue
        end

        # Gather predictions: tree x sample
        preds = Array{Int}(undef, tcount, n)
        for (ti, tow) in enumerate(sf.trees)
            for i in 1:n
                preds[ti,i] = EWSFTree.predict_tree(tow.tree, view(Xp, i, :))
            end
        end

        # For each sample compute counts across trees and add weighted fraction to prob_mat
        for i in 1:n
            counts = Dict{Int,Int}()
            for ti in 1:tcount
                counts[preds[ti,i]] = get(counts, preds[ti,i], 0) + 1
            end
            for (class_int, cnt) in counts
                col_idx = get(class_to_col, class_int, nothing)
                if col_idx !== nothing
                    prob_mat[i, col_idx] += sf.weight * (cnt / tcount)
                end
            end
        end
    end

    # Normalize per-row and produce predictions
    preds_out = Vector{Any}(undef, n)
    for i in 1:n
        s = sum(prob_mat[i, :])
        if s > 0
            prob_mat[i, :] ./= s
        else
            prob_mat[i, :] .= 1.0 / K  # fallback uniform
        end
        idx = argmax(prob_mat[i, :])
        preds_out[i] = labs_ordered[idx]
    end

    # Build DataFrame output
    colnames = [ string(l) for l in labs_ordered ]
    out_df = DataFrame()
    for k in 1:K
        out_df[!, Symbol(colnames[k])] = prob_mat[:, k]
    end
    out_df[!, :prediction] = preds_out
    return out_df
end


# Save / Load


function save_model(path::AbstractString, model::EWSFModel)
    open(path, "w") do io
        serialize(io, model)
    end
end

function load_model(path::AbstractString)
    open(path, "r") do io
        return deserialize(io)
    end
end


# Permutation-based drift helpers


# KS statistic for numeric arrays (two-sample)
function ks_stat(x::Vector{Float64}, y::Vector{Float64})
    allv = sort(unique(vcat(x, y)))
    cdfx = [ sum(x .<= v) / length(x) for v in allv ]
    cdfy = [ sum(y .<= v) / length(y) for v in allv ]
    return maximum(abs.(cdfx .- cdfy))
end

# Pearson-like chi2 for categorical integer-encoded arrays
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

"""
    permutation_pvalue(x, y; n_perm=500, rng=MersenneTwister(123))

Computes permutation p-value for two-sample test:

- If inputs are numeric (or convertible to Float64) uses KS statistic;
- Otherwise treats as categorical ints and uses chi2-like stat.

Returns (pvalue, observed_statistic).
"""
function permutation_pvalue(x, y; n_perm::Int=500, rng::AbstractRNG=MersenneTwister(123))
    # Try numeric first (convertible to Float64)
    is_numeric = try
        Float64.(x); Float64.(y)
        true
    catch
        false
    end

    if is_numeric
        xfloat = Float64.(x); yfloat = Float64.(y)
        obs_stat = ks_stat(xfloat, yfloat)
        combined = vcat(xfloat, yfloat)
        n1 = length(xfloat)
        cnt = 0
        for i in 1:n_perm
            perm = combined[randperm(rng, length(combined))]
            s1 = perm[1:n1]
            s2 = perm[n1+1:end]
            val = ks_stat(s1, s2)
            if val >= obs_stat
                cnt += 1
            end
        end
        return ( (cnt + 1) / (n_perm + 1), obs_stat )
    else
        xi = Int.(x); yi = Int.(y)
        obs_stat = chi2_stat(xi, yi)
        combined = vcat(xi, yi)
        n1 = length(xi)
        cnt = 0
        for i in 1:n_perm
            perm = combined[randperm(rng, length(combined))]
            s1 = perm[1:n1]
            s2 = perm[n1+1:end]
            val = chi2_stat(s1, s2)
            if val >= obs_stat
                cnt += 1
            end
        end
        return ( (cnt + 1) / (n_perm + 1), obs_stat )
    end
end

# Compute p-values & stats per feature between train_df and new_df
function feature_drift_pvalues(model::EWSFModel, train_df::DataFrame, new_df::DataFrame; n_perm::Int=500)
    enc = model.encoders
    features = model.feature_names
    pvals = Dict{Symbol,Float64}()
    stats = Dict{Symbol,Float64}()
    rng = MersenneTwister(2025)
    for f in features
        fname = string(f)
        if enc.feature_types[f] == :numeric
            train_vals = Float64.(train_df[!, fname])
            new_vals = Float64.(new_df[!, fname])
            pv, st = permutation_pvalue(train_vals, new_vals, n_perm=n_perm, rng=rng)
            pvals[f] = pv; stats[f] = st
        else
            # categorical: training encodings exist in enc.feature_encoders
            train_vals = Int.(train_df[!, fname])
            mapdict = get(enc.feature_encoders, f, Dict{Any,Int}())
            # Map new raw values to codes using mapdict; unseen -> 0
            raw_new = new_df[!, fname]
            new_vals = [ get(mapdict, v, 0) for v in raw_new ]
            pv, st = permutation_pvalue(train_vals, new_vals, n_perm=n_perm, rng=rng)
            pvals[f] = pv; stats[f] = st
        end
    end
    return (pvals, stats)
end

# Adapt weights based on drift


"""
    adapt_weights!(model, train_df, new_df; p_threshold=0.05, n_perm=500, lambda=3.0)

Compute per-feature permutation p-values between `train_df` (preprocessed original training frame)
and `new_df` (preprocessed incoming frame). For each subforest, compute fraction of its features
that have p < p_threshold; then downweight subforests proportional to exp(-lambda * drift_fraction).

Returns a Dict with keys:
- "pvals" => Dict{feature => pvalue}
- "stats" => Dict{feature => observed_statistic}
- "drift_scores" => Vector{Float64} per subforest
- "new_weights" => Vector{Float64} per subforest (normalized)
"""
function adapt_weights!(model::EWSFModel, train_df::DataFrame, new_df::DataFrame; p_threshold::Float64=0.05, n_perm::Int=500, lambda::Float64=3.0)
    (pvals, stats) = feature_drift_pvalues(model, train_df, new_df, n_perm=n_perm)
    drift_scores = Float64[]
    for sf in model.subforests
        # sf.perm are indices into model.feature_names (1-based)
        feat_syms = model.feature_names[sf.perm]
        n = length(feat_syms)
        if n == 0
            push!(drift_scores, 0.0)
            continue
        end
        cnt = 0
        for f in feat_syms
            if get(pvals, f, 1.0) < p_threshold
                cnt += 1
            end
        end
        push!(drift_scores, cnt / n)
    end

    # Apply soft weight decay: w_new = w_old * exp(-lambda * drift_score)
    new_w = [ sf.weight * exp(-lambda * ds) for (sf, ds) in zip(model.subforests, drift_scores) ]
    total = sum(new_w)
    if total == 0.0
        # fallback to uniform
        for sf in model.subforests
            sf.weight = 1.0 / length(model.subforests)
        end
    else
        for (i, sf) in enumerate(model.subforests)
            sf.weight = new_w[i] / total
        end
    end

    return Dict("pvals" => pvals, "stats" => stats, "drift_scores" => drift_scores, "new_weights" => [sf.weight for sf in model.subforests])
end

end # module
