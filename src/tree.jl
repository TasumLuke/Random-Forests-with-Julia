module EWSFTree
using Statistics, StatsBase, Random
export TreeNode, build_tree, predict_tree, feature_scores, train_forest

mutable struct TreeNode
    feature::Int
    threshold::Float64
    is_categorical::Bool
    left::Union{TreeNode,Nothing}
    right::Union{TreeNode,Nothing}
    prediction::Int
    is_leaf::Bool
end
TreeNode() = TreeNode(-1, 0.0, false, nothing, nothing, -1, true)

# gini impurity
function gini(labels::Vector{Int})
    n = length(labels)
    if n == 0 return 0.0 end
    counts = countmap(labels)
    s = 0.0
    for c in values(counts)
        p = c / n
        s += p * (1.0 - p)
    end
    return s
end

# mode label
function mode_label(labels::Vector{Int})
    counts = countmap(labels)
    # argmax
    best = first(keys(counts))
    bestc = counts[best]
    for (k,v) in counts
        if v > bestc
            best = k; bestc = v
        end
    end
    return best
end

# find best split among given feature indices
function find_best_split(X::Matrix{Float64}, y::Vector{Int}, feature_indices::Vector{Int}, feature_types::Vector{Symbol}, min_leaf::Int)
    n,p = size(X)
    parent_imp = gini(y)
    best_gain = 0.0
    best_feat = -1
    best_thresh = 0.0
    best_is_cat = false

    for f in feature_indices
        col = X[:, f]
        if feature_types[f] == :numeric
            order = sortperm(col)
            vals = col[order]
            labs = y[order]
            if minimum(vals) == maximum(vals) continue end
            for i in min_leaf:(n - min_leaf)
                if vals[i] == vals[i+1] continue end
                thr = (vals[i] + vals[i+1]) / 2.0
                left_idx = findall(x->x <= thr, col)
                right_idx = findall(x->x > thr, col)
                if length(left_idx) < min_leaf || length(right_idx) < min_leaf continue end
                g_left = gini(y[left_idx])
                g_right = gini(y[right_idx])
                gain = parent_imp - (length(left_idx)/n)*g_left - (length(right_idx)/n)*g_right
                if gain > best_gain
                    best_gain = gain; best_feat = f; best_thresh = thr; best_is_cat = false
                end
            end
        else
            vals = unique(col)
            for v in vals
                left_idx = findall(x->x == v, col)
                right_idx = setdiff(1:n, left_idx)
                if length(left_idx) < min_leaf || length(right_idx) < min_leaf continue end
                g_left = gini(y[left_idx])
                g_right = gini(y[right_idx])
                gain = parent_imp - (length(left_idx)/n)*g_left - (length(right_idx)/n)*g_right
                if gain > best_gain
                    best_gain = gain; best_feat = f; best_thresh = float(v); best_is_cat = true
                end
            end
        end
    end
    return (best_feat, best_thresh, best_is_cat, best_gain)
end

# build tree recursively
function build_tree(X::Matrix{Float64}, y::Vector{Int}, feature_types::Vector{Symbol}; max_depth::Int=8, min_leaf::Int=3, m_features::Int=0, rng::AbstractRNG=Random.GLOBAL_RNG)
    n,p = size(X)
    if m_features <= 0
        m_features = max(1, Int(round(sqrt(p))))
    end
    node = TreeNode()
    node.prediction = mode_label(y)

    if max_depth == 0 || length(unique(y)) == 1 || n <= 2*min_leaf
        node.is_leaf = true
        return node
    end

    feat_list = shuffle!(rng, collect(1:p))
    feat_list = feat_list[1:min(m_features, length(feat_list))]

    (best_feat, best_thr, best_is_cat, best_gain) = find_best_split(X, y, feat_list, feature_types, min_leaf)
    if best_feat == -1 || best_gain <= 0.0
        node.is_leaf = true
        return node
    end

    node.feature = best_feat
    node.threshold = best_thr
    node.is_categorical = best_is_cat
    if node.is_categorical
        left_idx = findall(x->x == node.threshold, X[:, node.feature])
    else
        left_idx = findall(x->x <= node.threshold, X[:, node.feature])
    end
    right_idx = setdiff(1:n, left_idx)
    if isempty(left_idx) || isempty(right_idx)
        node.is_leaf = true
        return node
    end
    node.is_leaf = false
    node.left = build_tree(X[left_idx, :], y[left_idx], feature_types; max_depth=max_depth-1, min_leaf=min_leaf, m_features=m_features, rng=rng)
    node.right = build_tree(X[right_idx, :], y[right_idx], feature_types; max_depth=max_depth-1, min_leaf=min_leaf, m_features=m_features, rng=rng)
    return node
end

# predict single sample
function predict_tree(node::TreeNode, x::AbstractVector{Float64})
    if node.is_leaf
        return node.prediction
    end
    if node.is_categorical
        if x[node.feature] == node.threshold
            return predict_tree(node.left, x)
        else
            return predict_tree(node.right, x)
        end
    else
        if x[node.feature] <= node.threshold
            return predict_tree(node.left, x)
        else
            return predict_tree(node.right, x)
        end
    end
end

# feature scores: numeric -> variance, categorical -> entropy
function feature_scores(X::Matrix{Float64}, feature_types::Vector{Symbol})
    p = size(X,2)
    scores = zeros(p)
    for j in 1:p
        col = X[:,j]
        if feature_types[j] == :numeric
            scores[j] = var(col)
        else
            freq = countmap(col)
            n = sum(values(freq))
            entr = 0.0
            for v in values(freq)
                p_ = v / n
                entr -= p_ * (p_ == 0 ? 0.0 : log2(p_))
            end
            scores[j] = entr
        end
    end
    ssum = sum(scores)
    if ssum > 0.0
        scores ./= ssum
    else
        scores .= 1.0 / p
    end
    return scores
end

# train_forest: returns vector of trees and for convenience returns the permutation used for each tree (we will not permute here; permutations handled by model)
function train_forest(X::Matrix{Float64}, y::Vector{Int}, feature_types::Vector{Symbol}; n_trees::Int=100, max_depth::Int=8, min_leaf::Int=3, alpha::Float64=1.0, rng::AbstractRNG = Random.GLOBAL_RNG)
    n,p = size(X)
    scores = feature_scores(X, feature_types)
    w = scores .^ alpha
    if sum(w) == 0.0
        w .= 1.0/p
    else
        w ./= sum(w)
    end
    trees = Vector{TreeNode}()
    for t in 1:n_trees
        idxs = rand(rng, 1:n, n)
        Xb = X[idxs, :]
        yb = y[idxs]
        m_features = max(1, Int(round(sqrt(p))))
        # sample subset of features to bias splitting
        feat_inds = sample(collect(1:p), Weights(w), min(m_features, p); replace=false)
        # To bias, we will pass m_features = length(feat_inds) so tree splitting considers that many features when sampling
        tree = build_tree(Xb, yb, feature_types; max_depth=max_depth, min_leaf=min_leaf, m_features=length(feat_inds), rng=rng)
        push!(trees, tree)
    end
    return trees
end

end # module
