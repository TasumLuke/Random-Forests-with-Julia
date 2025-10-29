module EWSFModel
using ..EWSFTree
using ..EWSFData
using ..EWSFUtils
using Random, Serialization, Statistics, DataFrames
export train_ewsf, predict_model, save_model, load_model, adapt_weights!, EWSFModel, SubForest

# SubForest now carries per-tree OOB indices optionally
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
    classes::Vector{Any}   # class labels in canonical order
end

# Train with explicit bootstrap and compute OOB accuracies for each subforest
function train_ewsf(df::DataFrame, enc::EWSFData.Encoders; n_trees::Int=90, max_depth::Int=8, min_leaf::Int=3, n_subforests::Int=3)
    features = enc.feature_order
    p = length(features)
    # Build X, y from preprocessed df (features then target at end)
    X = convert(Matrix{Float64}, Matrix(df[:, features]))
    target_cols = setdiff(names(df), features)
    if length(target_cols) != 1
        error("Preprocessed df expected to contain features then exactly one target column.")
    end
    y = Vector{Int}(df[:, target_cols[1]])
    ftypes = [ enc.feature_types[f] for f in features ]

    base_scores = EWSFTree.feature_scores(X, ftypes)
    subforests = Vector{SubForest}()
    alphas = range(0.5, stop=2.0, length=n_subforests)
    rng = MersenneTwister(42)
    trees_per_sub = max(1, Int(round(n_trees / n_subforests)))
    n = size(X,1)

    for alpha in alphas
        scores = base_scores .^ alpha
        if sum(scores) == 0.0
            probs = fill(1.0/length(scores), length(scores))
        else
            probs = scores ./ sum(scores)
        end
        perm = sample(collect(1:p), Weights(probs), p; replace=false)
        X_perm = X[:, perm]
        ftypes_perm = [ ftypes[j] for j in perm ]

        twobs = Vector{TreeWithOOB}()
        for t in 1:trees_per_sub
            # bootstrap sample indices with replacement
            boot_idxs = rand(rng, 1:n, n)
            # OOB indices are those not present in boot_idxs
            oob_idxs = setdiff(1:n, unique(boot_idxs))
            Xb = X_perm[boot_idxs, :]
            yb = y[boot_idxs]
            tree = EWSFTree.build_tree(Xb, yb, ftypes_perm; max_depth=max_depth, min_leaf=min_leaf, m_features=max(1,Int(round(sqrt(p)))), rng=rng)
            push!(twobs, TreeWithOOB(tree, oob_idxs))
        end

        # Compute OOB predictions for all rows
        n_classes = length(unique(y))
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
                # no OOB votes for this sample (rare); skip
                continue
            end
            # majority vote
            pred = EWSFUtils.mode_label(collect(Iterators.flatten([ fill(k, v) for (k,v) in votes ])))
            counted += 1
            if pred == y[i]
                correct += 1
            end
        end
        oob_acc = counted == 0 ? 0.0 : correct / counted
        # initialize weight to oob_acc (small floor)
        initial_weight = max(oob_acc, 1e-4)
        push!(subforests, SubForest(twobs, perm, ftypes_perm, initial_weight))
    end

    # Normalize weights
    total = sum(sf.weight for sf in subforests)
    if total == 0.0
        for sf in subforests
            sf.weight = 1.0/length(subforests)
        end
    else
        for sf in subforests
            sf.weight = sf.weight / total
        end
    end

    # gather class labels in order (map ints -> original labels using enc.inv_target)
    classes = sort(collect(values(enc.inv_target)))  # best-effort ordering
    model = EWSFModel(subforests, enc, features, enc.feature_types, classes)
    return model
end

# Predict: return DataFrame with probability columns (one per class) + prediction (argmax)
function predict_model(model::EWSFModel, newdf::DataFrame)
    enc = model.encoders
    features = model.feature_names
    n = nrow(newdf)
    p = length(features)

    # build Xnew
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

    # Determine label set from encoder
    inv = model.encoders.inv_target
    # sort keys by integer label
    labs_ordered = [ inv[i] for i in sort(collect(keys(inv))) ]
    K = length(labs_ordered)
    # prepare probability matrix: rows = samples, cols = class ints in ascending order
    prob_mat = zeros(n, K)

    # For each subforest compute per-sample class probabilities (from tree votes) and add weighted contribution
    for sf in model.subforests
        Xp = Xnew[:, sf.perm]
        # per-tree predictions: matrix trees x samples
        tcount = length(sf.trees)
        if tcount == 0
            continue
        end
        preds = Array{Int}(undef, tcount, n)
        for (ti, tow) in enumerate(sf.trees)
            for i in 1:n
                preds[ti,i] = EWSFTree.predict_tree(tow.tree, view(Xp, i, :))
            end
        end
        # per-sample probability distribution within subforest
        for i in 1:n
            counts = Dict{Int,Int}()
            for ti in 1:tcount
                counts[preds[ti,i]] = get(counts, preds[ti,i], 0) + 1
            end
            for (class_int, cnt) in counts
                # class_int corresponds to integer labels used in training; we need column index
                col_idx = findfirst(==(class_int), sort(collect(keys(inv))))  # index of class_int in sorted keys
                if col_idx === nothing
                    continue
                end
                prob_mat[i, col_idx] += sf.weight * (cnt / tcount)
            end
        end
    end

    # Normalize rows (in case numerical rounding) and produce predictions
    preds = Vector{Any}(undef, n)
    for i in 1:n
        s = sum(prob_mat[i, :])
        if s > 0
            prob_mat[i, :] ./= s
        else
            # uniform fallback
            prob_mat[i, :] .= 1.0 / K
        end
        # argmax gives index; map to label
        idx = argmax(prob_mat[i, :])
        preds[i] = labs_ordered[idx]
    end

    # produce DataFrame: one column per class label with probability, plus "prediction"
    colnames = [ string(l) for l in labs_ordered ]
    df_out = DataFrame()
    for k in 1:K
        df_out[!, Symbol(colnames[k])] = prob_mat[:, k]
    end
    df_out[!, :prediction] = preds
    return df_out
end

# save & load unchanged
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

# Permutation p-value and adapt_weights! as before (unchanged) — ensure adapt_weights! updates sf.weight, leaving OOB-based weights as prior
# ... include the adapt_weights! and helpers from earlier model.jl (copy those in verbatim)
# For brevity in this message, assume adapt_weights! and its helpers (permutation_pvalue, feature_drift_pvalues) are copied verbatim from prior version.
# (When you paste the file, include those functions here unchanged.)

end # module
