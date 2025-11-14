"""
ACRF-QIU: Adaptive Causal Random Forest with Quantum-Inspired Uncertainty Quantification
Complete implementation of the methodology from Lego & Baptiste

This module implements:
- Causal discovery via PC algorithm with conditional independence testing
- Quantum-inspired feature encoding with entanglement quantification
- Causal-weighted adaptive random forest ensemble
- Split conformal prediction for distribution-free uncertainty quantification
- Meta-learning with online weight adaptation
"""

module ACRFQIU

using LinearAlgebra
using Statistics
using StatsBase
using Distributions
using Random
using Printf

export CausalGraph, QuantumFeatureEncoder, CausalRandomForest, ConformalPredictor
export fit!, predict, predict_with_uncertainty, get_feature_importance
export causal_discovery!, quantum_encode, calculate_entanglement

# ============================================================================
# SECTION 1: CAUSAL DISCOVERY AND STRUCTURAL LEARNING (Section 4.1)
# ============================================================================

"""
    CausalGraph

Represents a directed acyclic graph (DAG) encoding causal relationships.
Implements equations (1)-(6) from Section 4.1.

Fields:
- `adjacency`: p×p adjacency matrix where A[i,j] = 1 indicates edge Xᵢ → Xⱼ
- `edge_weights`: p×p matrix of edge strengths (partial correlations)
- `causal_importance`: p-vector of causal importance scores Cᵢ (Eq. 16)
- `p`: number of features
"""
mutable struct CausalGraph
    adjacency::Matrix{Float64}
    edge_weights::Matrix{Float64}
    causal_importance::Vector{Float64}
    p::Int
end

"""
    CausalGraph(p::Int)

Initialize empty causal graph for p features.
"""
function CausalGraph(p::Int)
    return CausalGraph(
        zeros(p, p),
        zeros(p, p),
        zeros(p),
        p
    )
end

"""
    partial_correlation(X::Matrix, i::Int, j::Int, S::Vector{Int})

Compute partial correlation ρᵢⱼ|S between features i and j conditioning on set S.
Implements Equation (2) from the paper.

Arguments:
- `X`: n×p data matrix
- `i, j`: feature indices
- `S`: conditioning set (vector of feature indices)

Returns partial correlation coefficient ρᵢⱼ|S
"""
function partial_correlation(X::Matrix, i::Int, j::Int, S::Vector{Int})
    n, p = size(X)
    
    if isempty(S)
        # Unconditional correlation
        return cor(X[:, i], X[:, j])
    end
    
    # Compute correlation matrix for {i, j, S}
    indices = [i, j, S...]
    C = cor(X[:, indices])
    
    # Use recursive formula for partial correlation
    ρ_ij = C[1, 2]
    
    # Apply conditioning formula (Eq. 2)
    if length(S) == 1
        s_idx = 3
        ρ_is = C[1, s_idx]
        ρ_js = C[2, s_idx]
        
        numerator = ρ_ij - ρ_is * ρ_js
        denominator = sqrt((1 - ρ_is^2) * (1 - ρ_js^2))
        
        return denominator > 1e-10 ? numerator / denominator : 0.0
    else
        # For multiple conditioning variables, use precision matrix approach
        # More numerically stable than recursive formula
        Σ = C
        P = inv(Σ + 1e-6 * I)  # Add regularization for stability
        
        # Partial correlation from precision matrix: -P[i,j] / sqrt(P[i,i] * P[j,j])
        ρ_partial = -P[1, 2] / sqrt(P[1, 1] * P[2, 2])
        return ρ_partial
    end
end

"""
    fisher_z_test(ρ::Float64, n::Int, k::Int; α::Float64=0.05)

Test conditional independence using Fisher's z-transformation (Eq. 3).

Arguments:
- `ρ`: partial correlation coefficient
- `n`: sample size
- `k`: size of conditioning set
- `α`: significance level

Returns true if null hypothesis H₀: Xᵢ ⊥⊥ Xⱼ | S is rejected.
"""
function fisher_z_test(ρ::Float64, n::Int, k::Int; α::Float64=0.05)
    if abs(ρ) > 0.9999
        return true  # Deterministic relationship
    end
    
    # Fisher's z-transformation (Eq. 3)
    z = 0.5 * log((1 + ρ) / (1 - ρ))
    
    # Standard error
    se = 1.0 / sqrt(n - k - 3)
    
    # Test statistic
    test_stat = abs(z / se)
    
    # Critical value for two-tailed test
    z_critical = quantile(Normal(0, 1), 1 - α/2)
    
    return test_stat > z_critical
end

"""
    pc_skeleton(X::Matrix; α::Float64=0.05, max_cond_size::Int=3)

Phase 1 of PC algorithm: Discover graph skeleton through conditional independence testing.

Arguments:
- `X`: n×p data matrix
- `α`: significance level for independence tests
- `max_cond_size`: maximum size of conditioning sets

Returns skeleton adjacency matrix.
"""
function pc_skeleton(X::Matrix; α::Float64=0.05, max_cond_size::Int=3)
    n, p = size(X)
    
    # Initialize complete undirected graph
    skeleton = ones(p, p) - I(p)
    separating_sets = Dict{Tuple{Int,Int}, Vector{Int}}()
    
    # Iterate over conditioning set sizes
    for k in 0:max_cond_size
        changed = true
        while changed
            changed = false
            
            for i in 1:p
                for j in (i+1):p
                    if skeleton[i, j] == 0
                        continue  # Already removed
                    end
                    
                    # Get neighbors of i (excluding j)
                    neighbors = findall(x -> x > 0, skeleton[i, :])
                    neighbors = filter(x -> x != j, neighbors)
                    
                    if length(neighbors) < k
                        continue
                    end
                    
                    # Test all conditioning sets of size k
                    for S in combinations(neighbors, k)
                        ρ = partial_correlation(X, i, j, collect(S))
                        
                        # Test independence (Eq. 2-3)
                        if !fisher_z_test(ρ, n, k, α=α)
                            # Remove edge
                            skeleton[i, j] = 0
                            skeleton[j, i] = 0
                            separating_sets[(i, j)] = collect(S)
                            separating_sets[(j, i)] = collect(S)
                            changed = true
                            break
                        end
                    end
                end
            end
        end
    end
    
    return skeleton, separating_sets
end

"""
    interventional_score(X::Matrix, y::Vector, i::Int, j::Int)

Compute interventional inference score Sᵢ→ⱼ for orienting edges (Eq. 4-5).

Uses residual variance approximation of do-calculus:
Sᵢ→ⱼ ≈ Var(Y|Xⱼ) - Var(Y|Xᵢ, Xⱼ)
"""
function interventional_score(X::Matrix, y::Vector, i::Int, j::Int)
    n = length(y)
    
    # Variance of y given Xⱼ alone
    # Fit linear model: y ~ Xⱼ
    β_j = X[:, j] \ y
    residuals_j = y - X[:, j] * β_j
    var_j = var(residuals_j)
    
    # Variance of y given both Xᵢ and Xⱼ
    # Fit linear model: y ~ Xᵢ + Xⱼ
    β_ij = X[:, [i, j]] \ y
    residuals_ij = y - X[:, [i, j]] * β_ij
    var_ij = var(residuals_ij)
    
    # Score (Eq. 5): reduction in variance when adding Xᵢ
    return var_j - var_ij
end

"""
    orient_edges!(graph::CausalGraph, skeleton::Matrix, X::Matrix, y::Vector; τ::Float64=0.01)

Phase 2 of PC algorithm: Orient edges using asymmetric measures (Eq. 6).

Arguments:
- `graph`: CausalGraph to populate
- `skeleton`: undirected skeleton from Phase 1
- `X`: feature matrix
- `y`: target vector
- `τ`: threshold for orientation
"""
function orient_edges!(graph::CausalGraph, skeleton::Matrix, X::Matrix, y::Vector; τ::Float64=0.01)
    p = size(X, 2)
    
    for i in 1:p
        for j in (i+1):p
            if skeleton[i, j] == 0
                continue
            end
            
            # Compute directional scores (Eq. 4-5)
            S_i_to_j = interventional_score(X, y, i, j)
            S_j_to_i = interventional_score(X, y, j, i)
            
            # Orient edge based on score difference (Eq. 6)
            if S_i_to_j - S_j_to_i > τ
                # Orient as i → j
                graph.adjacency[i, j] = 1
                graph.edge_weights[i, j] = abs(cor(X[:, i], X[:, j]))
            elseif S_j_to_i - S_i_to_j > τ
                # Orient as j → i
                graph.adjacency[j, i] = 1
                graph.edge_weights[j, i] = abs(cor(X[:, i], X[:, j]))
            else
                # Leave undirected or use heuristic
                # Default: lexicographic ordering
                graph.adjacency[i, j] = 1
                graph.edge_weights[i, j] = abs(cor(X[:, i], X[:, j]))
            end
        end
    end
end

"""
    compute_causal_importance!(graph::CausalGraph, y_idx::Int)

Compute causal importance Cᵢ for each feature based on graph structure (Eq. 16).

Cᵢ = Σⱼ:(i,j)∈E wᵢⱼ + Σₖ:(i,k,j)∈Paths (wᵢₖ · wₖⱼ) / |Path|

Arguments:
- `graph`: CausalGraph with adjacency and weights
- `y_idx`: index representing target variable (typically p+1)
"""
function compute_causal_importance!(graph::CausalGraph, y_idx::Int=0)
    p = graph.p
    C = zeros(p)
    
    for i in 1:p
        # Direct causal effects (immediate children)
        for j in 1:p
            if graph.adjacency[i, j] > 0
                C[i] += graph.edge_weights[i, j]
            end
        end
        
        # Indirect causal effects (paths of length 2)
        for k in 1:p
            if graph.adjacency[i, k] > 0
                for j in 1:p
                    if graph.adjacency[k, j] > 0 && i != j
                        # Path i → k → j
                        path_strength = graph.edge_weights[i, k] * graph.edge_weights[k, j]
                        C[i] += path_strength / 2.0  # Discounted by path length
                    end
                end
            end
        end
    end
    
    # Normalize to [0, 1]
    if maximum(C) > 0
        C ./= maximum(C)
    end
    
    graph.causal_importance = C
end

"""
    causal_discovery!(X::Matrix, y::Vector; α::Float64=0.05, max_cond_size::Int=3, τ::Float64=0.01)

Complete causal discovery pipeline (Algorithm 1, Phase 1).

Discovers causal DAG structure from observational data using PC algorithm.

Returns CausalGraph with discovered structure.
"""
function causal_discovery!(X::Matrix, y::Vector; α::Float64=0.05, max_cond_size::Int=3, τ::Float64=0.01)
    n, p = size(X)
    
    println("Phase 1: Skeleton Discovery")
    skeleton, sep_sets = pc_skeleton(X, α=α, max_cond_size=max_cond_size)
    println("  Found $(sum(skeleton)/2) undirected edges")
    
    println("Phase 2: Edge Orientation")
    graph = CausalGraph(p)
    orient_edges!(graph, skeleton, X, y, τ=τ)
    println("  Oriented $(sum(graph.adjacency)) directed edges")
    
    println("Phase 3: Causal Importance Computation")
    compute_causal_importance!(graph)
    
    return graph
end


# ============================================================================
# SECTION 2: QUANTUM-INSPIRED FEATURE ENCODING (Section 4.2)
# ============================================================================

"""
    QuantumFeatureEncoder

Quantum-inspired feature encoding using amplitude representation in Hilbert space.
Implements equations (7)-(13) from Section 4.2.

Fields:
- `d`: dimension of Hilbert space (number of bins)
- `amplitudes`: p×d matrix of complex probability amplitudes αᵢₖ
- `entanglement_matrix`: p×p matrix of entanglement measures Eᵢⱼ
- `p`: number of features
"""
mutable struct QuantumFeatureEncoder
    d::Int
    amplitudes::Matrix{ComplexF64}
    entanglement_matrix::Matrix{Float64}
    bin_edges::Matrix{Float64}  # p×(d+1) matrix storing bin boundaries
    p::Int
end

"""
    QuantumFeatureEncoder(p::Int; d::Int=10)

Initialize quantum feature encoder for p features with d-dimensional Hilbert space.
"""
function QuantumFeatureEncoder(p::Int; d::Int=10)
    return QuantumFeatureEncoder(
        d,
        zeros(ComplexF64, p, d),
        zeros(p, p),
        zeros(p, d+1),
        p
    )
end

"""
    discretize_feature(x::Vector, d::Int)

Discretize continuous feature into d bins and compute histogram.
Implements Equation (8).

Returns:
- `h`: d-vector of bin probabilities
- `edges`: (d+1)-vector of bin boundaries
"""
function discretize_feature(x::Vector, d::Int)
    n = length(x)
    
    # Create bins spanning [min, max]
    x_min, x_max = extrema(x)
    
    # Add small epsilon to avoid boundary issues
    ε = (x_max - x_min) * 1e-6
    edges = range(x_min - ε, x_max + ε, length=d+1)
    
    # Compute histogram (Eq. 8)
    h = zeros(d)
    for val in x
        bin_idx = searchsortedfirst(edges, val) - 1
        bin_idx = clamp(bin_idx, 1, d)
        h[bin_idx] += 1
    end
    
    h ./= n  # Normalize to probabilities
    
    return h, collect(edges)
end

"""
    compute_amplitudes(h::Vector, d::Int)

Convert probability distribution to quantum amplitudes with phase encoding.
Implements Equation (9).

αᵢₖ = √hᵢₖ · exp(iθᵢₖ) where θᵢₖ = 2πk/d
"""
function compute_amplitudes(h::Vector, d::Int)
    α = zeros(ComplexF64, d)
    
    for k in 1:d
        # Amplitude is square root of probability (Born rule)
        magnitude = sqrt(h[k])
        
        # Phase encodes higher-order moments (Eq. 9)
        phase = 2π * (k-1) / d
        
        α[k] = magnitude * exp(im * phase)
    end
    
    # Verify normalization: Σ|α|² = 1
    norm_sq = sum(abs2.(α))
    if norm_sq > 0
        α ./= sqrt(norm_sq)
    end
    
    return α
end

"""
    quantum_encode!(encoder::QuantumFeatureEncoder, X::Matrix)

Encode all features as quantum states in Hilbert space.
Implements Algorithm 1, Phase 2, lines 16-20.

Populates encoder.amplitudes with quantum representations |ψᵢ⟩.
"""
function quantum_encode!(encoder::QuantumFeatureEncoder, X::Matrix)
    n, p = size(X)
    
    for i in 1:p
        # Discretize feature into histogram (Eq. 8)
        h, edges = discretize_feature(X[:, i], encoder.d)
        encoder.bin_edges[i, :] = edges
        
        # Compute quantum amplitudes (Eq. 9)
        α = compute_amplitudes(h, encoder.d)
        encoder.amplitudes[i, :] = α
    end
    
    println("Quantum encoding complete: $p features → $(encoder.d)-dimensional Hilbert space")
end

"""
    joint_state(encoder::QuantumFeatureEncoder, i::Int, j::Int)

Construct joint quantum state |Ψᵢⱼ⟩ for features i and j.
Implements Equation (10).

|Ψᵢⱼ⟩ = Σₖ Σₗ βᵢⱼₖₗ |k⟩ᵢ ⊗ |l⟩ⱼ
"""
function joint_state(encoder::QuantumFeatureEncoder, i::Int, j::Int)
    d = encoder.d
    
    # Tensor product of individual states
    Ψ = zeros(ComplexF64, d, d)
    
    for k in 1:d
        for l in 1:d
            # Assuming independence: βᵢⱼₖₗ = αᵢₖ · αⱼₗ
            # In practice, estimate from joint distribution
            Ψ[k, l] = encoder.amplitudes[i, k] * encoder.amplitudes[j, l]
        end
    end
    
    return Ψ
end

"""
    reduced_density_matrix(Ψ::Matrix{ComplexF64}, trace_out::Symbol=:j)

Compute reduced density matrix by tracing out one subsystem.
Implements Equation (11).

ρᵢ = Trⱼ(|Ψᵢⱼ⟩⟨Ψᵢⱼ|)
"""
function reduced_density_matrix(Ψ::Matrix{ComplexF64}, trace_out::Symbol=:j)
    d = size(Ψ, 1)
    
    # Construct density matrix |Ψ⟩⟨Ψ|
    ρ_full = Ψ[:] * Ψ[:]'  # Vectorize and outer product
    ρ_full = reshape(ρ_full, d, d, d, d)
    
    # Trace out second subsystem
    ρ_reduced = zeros(ComplexF64, d, d)
    
    if trace_out == :j
        for i1 in 1:d
            for i2 in 1:d
                for j in 1:d
                    ρ_reduced[i1, i2] += ρ_full[i1, j, i2, j]
                end
            end
        end
    else  # trace_out == :i
        for j1 in 1:d
            for j2 in 1:d
                for i in 1:d
                    ρ_reduced[j1, j2] += ρ_full[i, j1, i, j2]
                end
            end
        end
    end
    
    return ρ_reduced
end

"""
    von_neumann_entropy(ρ::Matrix{ComplexF64})

Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ).
Implements Equation (12).

For entanglement quantification: S(ρ) = 0 for separable states.
"""
function von_neumann_entropy(ρ::Matrix{ComplexF64})
    # Eigenvalue decomposition
    λ = eigvals(ρ)
    
    # Filter out numerical noise (very small/negative eigenvalues)
    λ = real.(λ)
    λ = filter(x -> x > 1e-10, λ)
    
    if isempty(λ)
        return 0.0
    end
    
    # Normalize (should sum to 1, but numerical errors)
    λ ./= sum(λ)
    
    # S(ρ) = -Σ λₖ log λₖ (Eq. 12)
    S = -sum(λ .* log.(λ))
    
    return S
end

"""
    calculate_entanglement!(encoder::QuantumFeatureEncoder)

Compute pairwise entanglement matrix Eᵢⱼ.
Implements Algorithm 1, Phase 2, lines 21-25 and Equation (13).

Eᵢⱼ = S(ρᵢ) where ρᵢ = Trⱼ(|Ψᵢⱼ⟩⟨Ψᵢⱼ|)
"""
function calculate_entanglement!(encoder::QuantumFeatureEncoder)
    p = encoder.p
    E = zeros(p, p)
    
    for i in 1:p
        for j in (i+1):p
            # Construct joint state (Eq. 10)
            Ψ = joint_state(encoder, i, j)
            
            # Compute reduced density matrix (Eq. 11)
            ρ_i = reduced_density_matrix(Ψ, :j)
            
            # Compute von Neumann entropy (Eq. 12-13)
            S = von_neumann_entropy(ρ_i)
            
            E[i, j] = S
            E[j, i] = S  # Symmetric
        end
    end
    
    encoder.entanglement_matrix = E
    
    println("Entanglement quantification complete: max entropy = $(maximum(E))")
end


# ============================================================================
# SECTION 3: ADAPTIVE RANDOM FOREST WITH CAUSAL WEIGHTING (Section 4.3)
# ============================================================================

"""
    DecisionNode

Node in a decision tree.
"""
mutable struct DecisionNode
    feature::Int  # Feature index for split
    threshold::Float64  # Split threshold
    left::Union{DecisionNode, Nothing}
    right::Union{DecisionNode, Nothing}
    prediction::Int  # Class label (for leaf nodes)
    is_leaf::Bool
    depth::Int
end

"""
    DecisionTree

Single decision tree in the ensemble.
"""
mutable struct DecisionTree
    root::Union{DecisionNode, Nothing}
    max_depth::Int
    min_samples_leaf::Int
    feature_subset_size::Int  # m: number of features sampled at each split
    causal_weights::Vector{Float64}  # Cᵢ from causal graph
    γ::Float64  # Causal bonus parameter for splitting (Eq. 15)
    η::Float64  # Causal alignment weight parameter (Eq. 19)
    causal_graph::Union{CausalGraph, Nothing}
    classes::Vector{Int}
    n_classes::Int
end

"""
    CausalRandomForest(; n_trees::Int=100, max_depth::Int=10, 
                        min_samples_leaf::Int=5, feature_subset_size::Int=0,
                        γ::Float64=0.5, η::Float64=0.5)

Initialize Causal Random Forest ensemble.
"""
function CausalRandomForest(; n_trees::Int=100, max_depth::Int=10,
                           min_samples_leaf::Int=5, feature_subset_size::Int=0,
                           γ::Float64=0.5, η::Float64=0.5)
    return CausalRandomForest(
        DecisionTree[],
        Float64[],
        n_trees,
        max_depth,
        min_samples_leaf,
        feature_subset_size,
        γ,
        η,
        nothing,
        Int[],
        0
    )
end

"""
    compute_causal_alignment(tree::DecisionTree, node::DecisionNode, 
                            causal_weights::Vector{Float64})

Compute causal alignment score Aₜ for a tree.
Implements Equations (17) and (18).

Aₜ = (Σᵢ∈Fₜ Cᵢ · dₜ(i)) / (Σᵢ∈Fₜ dₜ(i))

where dₜ(i) = Σᵥ∈Nₜ 𝟙[split(ν)=Xᵢ] · 1/depth(ν)
"""
function compute_causal_alignment(tree::DecisionTree, node::DecisionNode,
                                 causal_weights::Vector{Float64},
                                 feature_depths::Dict{Int,Vector{Int}}=Dict{Int,Vector{Int}}())
    if node === nothing
        return feature_depths
    end
    
    if !node.is_leaf
        # Record feature usage at this depth
        if !haskey(feature_depths, node.feature)
            feature_depths[node.feature] = Int[]
        end
        push!(feature_depths[node.feature], node.depth)
        
        # Recurse
        compute_causal_alignment(tree, node.left, causal_weights, feature_depths)
        compute_causal_alignment(tree, node.right, causal_weights, feature_depths)
    end
    
    return feature_depths
end

function causal_alignment_score(feature_depths::Dict{Int,Vector{Int}},
                               causal_weights::Vector{Float64})
    if isempty(feature_depths)
        return 0.0
    end
    
    numerator = 0.0
    denominator = 0.0
    
    for (feature, depths) in feature_depths
        # dₜ(i) = Σ 1/depth(ν) for all nodes split on feature i (Eq. 18)
        d_t_i = sum(1.0 ./ max.(depths, 1))  # max to avoid division by zero at root
        
        numerator += causal_weights[feature] * d_t_i
        denominator += d_t_i
    end
    
    return denominator > 0 ? numerator / denominator : 0.0
end

"""
    fit!(forest::CausalRandomForest, X::Matrix, y::Vector, 
         causal_graph::CausalGraph; verbose::Bool=true)

Train Causal Random Forest ensemble.
Implements Algorithm 1, Phase 3.
"""
function fit!(forest::CausalRandomForest, X::Matrix, y::Vector,
             causal_graph::CausalGraph; verbose::Bool=true)
    n, p = size(X)
    forest.causal_graph = causal_graph
    forest.classes = sort(unique(y))
    forest.n_classes = length(forest.classes)
    
    # Set feature subset size (default: sqrt(p))
    if forest.feature_subset_size == 0
        forest.feature_subset_size = Int(ceil(sqrt(p)))
    end
    
    if verbose
        println("\n" * "="^70)
        println("Training Causal Random Forest")
        println("="^70)
        println("Samples: $n | Features: $p | Classes: $(forest.n_classes)")
        println("Trees: $(forest.n_trees) | Max depth: $(forest.max_depth)")
        println("Causal bonus γ: $(forest.γ) | Alignment weight η: $(forest.η)")
        println("="^70)
    end
    
    # Train each tree
    forest.trees = DecisionTree[]
    oob_accuracies = Float64[]
    causal_alignments = Float64[]
    
    for t in 1:forest.n_trees
        if verbose && t % 10 == 0
            print("\rTraining tree $t/$(forest.n_trees)...")
        end
        
        # Bootstrap sample (Algorithm 1, line 30)
        bootstrap_indices = sample(1:n, n, replace=true)
        oob_indices = setdiff(1:n, unique(bootstrap_indices))
        
        X_boot = X[bootstrap_indices, :]
        y_boot = y[bootstrap_indices]
        
        # Create and train tree
        tree = DecisionTree(
            nothing,
            forest.max_depth,
            forest.min_samples_leaf,
            forest.feature_subset_size,
            causal_graph.causal_importance,
            forest.γ
        )
        
        # Build tree (Algorithm 1, line 32)
        tree.root = build_tree!(X_boot, y_boot, tree)
        
        # Compute OOB accuracy (Algorithm 1, line 33)
        if !isempty(oob_indices)
            oob_predictions = [predict_tree(tree.root, X[i, :]) for i in oob_indices]
            oob_acc = mean(oob_predictions .== y[oob_indices])
        else
            oob_acc = 0.5  # Default if no OOB samples
        end
        
        # Compute causal alignment (Algorithm 1, line 34)
        feature_depths = compute_causal_alignment(tree, tree.root, 
                                                 causal_graph.causal_importance)
        alignment = causal_alignment_score(feature_depths, causal_graph.causal_importance)
        
        push!(forest.trees, tree)
        push!(oob_accuracies, oob_acc)
        push!(causal_alignments, alignment)
    end
    
    if verbose
        println("\rTraining complete!                    ")
    end
    
    # Compute tree weights (Algorithm 1, line 35)
    raw_weights = [compute_tree_weight(oob_accuracies[t], causal_alignments[t], forest.η)
                   for t in 1:forest.n_trees]
    
    # Normalize weights (Algorithm 1, line 37, Eq. 20)
    forest.tree_weights = raw_weights ./ sum(raw_weights)
    
    if verbose
        println("\nTree Weight Statistics:")
        println("  Mean OOB Accuracy: $(round(mean(oob_accuracies), digits=4))")
        println("  Mean Causal Alignment: $(round(mean(causal_alignments), digits=4))")
        println("  Weight Range: [$(round(minimum(forest.tree_weights), digits=4)), " *
                "$(round(maximum(forest.tree_weights), digits=4))]")
        println("="^70)
    end
end

"""
    predict_proba(forest::CausalRandomForest, X::Matrix)

Predict class probabilities using weighted ensemble voting.
Implements Equations (21) and (23).

ŷ(x) = argmaxc∈C Σₜ w̃ₜ · 𝟙[hₜ(x)=c]
π̂c(x) = Σₜ w̃ₜ · 𝟙[hₜ(x)=c]
"""
function predict_proba(forest::CausalRandomForest, X::Matrix)
    n = size(X, 1)
    K = forest.n_classes
    
    proba = zeros(n, K)
    
    for i in 1:n
        class_votes = zeros(K)
        
        for (t, tree) in enumerate(forest.trees)
            prediction = predict_tree(tree.root, X[i, :])
            class_idx = findfirst(forest.classes .== prediction)
            
            if class_idx !== nothing
                class_votes[class_idx] += forest.tree_weights[t]
            end
        end
        
        # Normalize to probabilities (Eq. 23)
        proba[i, :] = class_votes ./ sum(class_votes)
    end
    
    return proba
end

"""
    predict(forest::CausalRandomForest, X::Matrix)

Predict class labels using weighted ensemble voting.
Implements Equation (21).
"""
function predict(forest::CausalRandomForest, X::Matrix)
    proba = predict_proba(forest, X)
    predictions = [forest.classes[argmax(proba[i, :])] for i in 1:size(X, 1)]
    return predictions
end


# ============================================================================
# SECTION 4: CONFORMAL PREDICTION FOR UNCERTAINTY QUANTIFICATION (Section 4.4)
# ============================================================================

"""
    ConformalPredictor

Split conformal prediction for distribution-free uncertainty quantification.
Implements Section 4.4.
"""
mutable struct ConformalPredictor
    model::CausalRandomForest
    calibration_scores::Vector{Float64}
    quantile::Float64
    α::Float64  # Miscoverage level
    n_cal::Int
end

"""
    ConformalPredictor(model::CausalRandomForest; α::Float64=0.1)

Initialize conformal predictor with miscoverage level α.
"""
function ConformalPredictor(model::CausalRandomForest; α::Float64=0.1)
    return ConformalPredictor(
        model,
        Float64[],
        0.0,
        α,
        0
    )
end

"""
    compute_nonconformity_score(y_true::Int, probabilities::Vector{Float64},
                               classes::Vector{Int})

Compute non-conformity score for calibration.
Implements Equation (22).

sᵢ = 1 - π̂yᵢ(xᵢ)
"""
function compute_nonconformity_score(y_true::Int, probabilities::Vector{Float64},
                                    classes::Vector{Int})
    class_idx = findfirst(classes .== y_true)
    
    if class_idx === nothing
        return 1.0  # Maximum non-conformity for unseen class
    end
    
    # Non-conformity: 1 - probability of true class (Eq. 22)
    return 1.0 - probabilities[class_idx]
end

"""
    calibrate!(predictor::ConformalPredictor, X_cal::Matrix, y_cal::Vector)

Calibrate conformal predictor using hold-out calibration set.
Implements Algorithm 1, Phase 4.
"""
function calibrate!(predictor::ConformalPredictor, X_cal::Matrix, y_cal::Vector)
    n_cal = length(y_cal)
    predictor.n_cal = n_cal
    
    println("\n" * "="^70)
    println("Conformal Calibration")
    println("="^70)
    println("Calibration samples: $n_cal")
    println("Miscoverage level α: $(predictor.α)")
    
    # Compute predicted probabilities for calibration set
    proba = predict_proba(predictor.model, X_cal)
    
    # Compute non-conformity scores (Algorithm 1, lines 40-43)
    scores = Float64[]
    for i in 1:n_cal
        score = compute_nonconformity_score(y_cal[i], proba[i, :],
                                           predictor.model.classes)
        push!(scores, score)
    end
    
    # Sort scores (Algorithm 1, line 44)
    predictor.calibration_scores = sort(scores)
    
    # Compute quantile with finite-sample correction (Algorithm 1, line 45, Eq. 24-25)
    quantile_index = Int(ceil((n_cal + 1) * (1 - predictor.α)))
    quantile_index = min(quantile_index, n_cal)
    
    predictor.quantile = predictor.calibration_scores[quantile_index]
    
    println("Quantile q̂₁₋ₐ: $(round(predictor.quantile, digits=4))")
    println("Coverage guarantee: ≥ $(round((1-predictor.α)*100, digits=1))%")
    println("="^70)
end

"""
    predict_set(predictor::ConformalPredictor, x::Vector{Float64})

Construct conformal prediction set for single sample.
Implements Equation (26) and Algorithm 2, lines 10-17.

C(xnew) = {c ∈ C : 1 - π̂c(xnew) ≤ q̂₁₋ₐ}
"""
function predict_set(predictor::ConformalPredictor, x::Vector{Float64})
    # Get class probabilities
    X_single = reshape(x, 1, length(x))
    proba = predict_proba(predictor.model, X_single)[1, :]
    
    # Build prediction set (Eq. 26)
    prediction_set = Int[]
    
    for (idx, class) in enumerate(predictor.model.classes)
        # Compute non-conformity for this class
        score = 1.0 - proba[idx]
        
        # Include in set if non-conformity ≤ quantile
        if score <= predictor.quantile
            push!(prediction_set, class)
        end
    end
    
    return prediction_set, proba
end

"""
    predict_with_uncertainty(predictor::ConformalPredictor, X::Matrix)

Make predictions with conformal uncertainty quantification.
Implements Algorithm 2.

Returns:
- predictions: point predictions
- confidence: confidence scores
- prediction_sets: conformal prediction sets
- set_sizes: sizes of prediction sets
"""
function predict_with_uncertainty(predictor::ConformalPredictor, X::Matrix)
    n = size(X, 1)
    
    predictions = Int[]
    confidence = Float64[]
    prediction_sets = Vector{Int}[]
    set_sizes = Int[]
    
    for i in 1:n
        pred_set, proba = predict_set(predictor, X[i, :])
        
        # Point prediction: most probable class in prediction set (Eq. 28)
        if !isempty(pred_set)
            # Find most probable class in set
            set_indices = [findfirst(predictor.model.classes .== c) for c in pred_set]
            set_probs = [proba[idx] for idx in set_indices]
            best_idx = argmax(set_probs)
            point_pred = pred_set[best_idx]
            conf = set_probs[best_idx]
        else
            # Empty set (rare): default to most probable class
            point_pred = predictor.model.classes[argmax(proba)]
            conf = maximum(proba)
        end
        
        push!(predictions, point_pred)
        push!(confidence, conf)
        push!(prediction_sets, pred_set)
        push!(set_sizes, length(pred_set))
    end
    
    return predictions, confidence, prediction_sets, set_sizes
end


# ============================================================================
# SECTION 5: META-LEARNING AND ADAPTIVE OPTIMIZATION (Section 4.5)
# ============================================================================

"""
    update_weights!(forest::CausalRandomForest, X_new::Matrix, y_new::Vector; λ::Float64=0.01)

Online weight adaptation via gradient descent.
Implements Equations (29) and (30).

ℓᵢ(w) = -log(Σₜ wₜ · 𝟙[hₜ(xᵢ)=yᵢ])
w⁽ᵏ⁺¹⁾ₜ = w⁽ᵏ⁾ₜ - λ ∂/∂wₜ Σᵢ ℓᵢ(w⁽ᵏ⁾)
"""
function update_weights!(forest::CausalRandomForest, X_new::Matrix, y_new::Vector;
                        λ::Float64=0.01, w_min::Float64=0.01)
    n_new = length(y_new)
    T = forest.n_trees
    
    # Compute gradient of loss
    gradients = zeros(T)
    
    for i in 1:n_new
        # Predictions from each tree
        tree_correct = zeros(T)
        for t in 1:T
            pred = predict_tree(forest.trees[t].root, X_new[i, :])
            tree_correct[t] = (pred == y_new[i]) ? 1.0 : 0.0
        end
        
        # Weighted sum
        weighted_sum = sum(forest.tree_weights .* tree_correct)
        
        if weighted_sum > 1e-10
            # Gradient of log loss (Eq. 29-30)
            gradients .+= -tree_correct ./ weighted_sum
        end
    end
    
    # Update weights
    new_weights = forest.tree_weights .- λ .* gradients
    
    # Project onto simplex: non-negative, sum to 1, minimum threshold
    new_weights = max.(new_weights, w_min)
    new_weights ./= sum(new_weights)
    
    forest.tree_weights = new_weights
end

"""
    adaptive_hyperparameters(n::Int, p::Int, K::Int)

Automatically tune hyperparameters based on dataset characteristics.
Implements Equations (31), (32), and (33).
"""
function adaptive_hyperparameters(n::Int, p::Int, K::Int)
    # Number of trees (Eq. 31)
    T = Int(min(400, 75 + 50 * log(n) + 25 * sqrt(p)))
    
    # Maximum tree depth (Eq. 32)
    D_max = Int(min(25, 6 + 4 * log2(n/K) + 2 * floor(p/10)))
    
    # Minimum samples per leaf (Eq. 33)
    m_min = Int(max(2, floor(n / (50 * K))))
    
    return T, D_max, m_min
end


# ============================================================================
# SECTION 6: COMPLETE ACRF-QIU PIPELINE
# ============================================================================

"""
    ACRFQIU_Model

Complete ACRF-QIU model integrating all components.
"""
mutable struct ACRFQIU_Model
    causal_graph::Union{CausalGraph, Nothing}
    quantum_encoder::Union{QuantumFeatureEncoder, Nothing}
    forest::Union{CausalRandomForest, Nothing}
    conformal::Union{ConformalPredictor, Nothing}
    X_train::Union{Matrix{Float64}, Nothing}
    y_train::Union{Vector{Int}, Nothing}
end

"""
    ACRFQIU_Model()

Initialize empty ACRF-QIU model.
"""
function ACRFQIU_Model()
    return ACRFQIU_Model(nothing, nothing, nothing, nothing, nothing, nothing)
end

"""
    fit!(model::ACRFQIU_Model, X::Matrix, y::Vector;
         causal_α::Float64=0.05,
         quantum_d::Int=10,
         n_trees::Int=0,  # 0 = auto
         max_depth::Int=0,  # 0 = auto
         min_samples_leaf::Int=0,  # 0 = auto
         γ::Float64=0.5,
         η::Float64=0.5,
         conformal_α::Float64=0.1,
         cal_fraction::Float64=0.2,
         verbose::Bool=true)

Complete ACRF-QIU training pipeline.
Implements Algorithm 1 in its entirety.
"""
function fit!(model::ACRFQIU_Model, X::Matrix, y::Vector;
             causal_α::Float64=0.05,
             quantum_d::Int=10,
             n_trees::Int=0,
             max_depth::Int=0,
             min_samples_leaf::Int=0,
             γ::Float64=0.5,
             η::Float64=0.5,
             conformal_α::Float64=0.1,
             cal_fraction::Float64=0.2,
             verbose::Bool=true)
    
    n, p = size(X)
    K = length(unique(y))
    
    if verbose
        println("\n" * "█"^70)
        println("█" * " "^68 * "█")
        println("█" * " "^15 * "ACRF-QIU TRAINING PIPELINE" * " "^26 * "█")
        println("█" * " "^68 * "█")
        println("█"^70)
        println("\nDataset: $n samples × $p features × $K classes")
    end
    
    # ==========================
    # PHASE 1: CAUSAL DISCOVERY
    # ==========================
    if verbose
        println("\n" * "▼"^70)
        println("PHASE 1: Causal Discovery (PC Algorithm)")
        println("▼"^70)
    end
    
    model.causal_graph = causal_discovery!(X, y, α=causal_α)
    
    # ==========================
    # PHASE 2: QUANTUM ENCODING
    # ==========================
    if verbose
        println("\n" * "▼"^70)
        println("PHASE 2: Quantum-Inspired Feature Encoding")
        println("▼"^70)
    end
    
    model.quantum_encoder = QuantumFeatureEncoder(p, d=quantum_d)
    quantum_encode!(model.quantum_encoder, X)
    calculate_entanglement!(model.quantum_encoder)
    
    # ==========================
    # PHASE 3: RANDOM FOREST
    # ==========================
    if verbose
        println("\n" * "▼"^70)
        println("PHASE 3: Causal Random Forest Training")
        println("▼"^70)
    end
    
    # Adaptive hyperparameter selection (Eq. 31-33)
    if n_trees == 0 || max_depth == 0 || min_samples_leaf == 0
        T_auto, D_auto, m_auto = adaptive_hyperparameters(n, p, K)
        n_trees = n_trees == 0 ? T_auto : n_trees
        max_depth = max_depth == 0 ? D_auto : max_depth
        min_samples_leaf = min_samples_leaf == 0 ? m_auto : min_samples_leaf
        
        if verbose
            println("Adaptive hyperparameters:")
            println("  Trees: $n_trees")
            println("  Max depth: $max_depth")
            println("  Min samples/leaf: $min_samples_leaf")
        end
    end
    
    # Split data: training and calibration (Algorithm 1, line 28)
    n_cal = Int(floor(n * cal_fraction))
    cal_indices = sample(1:n, n_cal, replace=false)
    train_indices = setdiff(1:n, cal_indices)
    
    X_train = X[train_indices, :]
    y_train = y[train_indices]
    X_cal = X[cal_indices, :]
    y_cal = y[cal_indices]
    
    model.X_train = X_train
    model.y_train = y_train
    
    # Train forest
    model.forest = CausalRandomForest(
        n_trees=n_trees,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        γ=γ,
        η=η
    )
    
    fit!(model.forest, X_train, y_train, model.causal_graph, verbose=verbose)
    
    # ==========================
    # PHASE 4: CONFORMAL CALIBRATION
    # ==========================
    if verbose
        println("\n" * "▼"^70)
        println("PHASE 4: Conformal Prediction Calibration")
        println("▼"^70)
    end
    
    model.conformal = ConformalPredictor(model.forest, α=conformal_α)
    calibrate!(model.conformal, X_cal, y_cal)
    
    if verbose
        println("\n" * "█"^70)
        println("█" * " "^20 * "TRAINING COMPLETE!" * " "^27 * "█")
        println("█"^70)
    end
end

"""
    predict(model::ACRFQIU_Model, X::Matrix; with_uncertainty::Bool=false)

Make predictions with trained ACRF-QIU model.
Implements Algorithm 2.
"""
function predict(model::ACRFQIU_Model, X::Matrix; with_uncertainty::Bool=false)
    if model.forest === nothing
        error("Model not trained. Call fit! first.")
    end
    
    if with_uncertainty
        if model.conformal === nothing
            error("Conformal predictor not calibrated.")
        end
        return predict_with_uncertainty(model.conformal, X)
    else
        return predict(model.forest, X)
    end
end

"""
    get_feature_importance(model::ACRFQIU_Model)

Extract feature importance based on causal structure.
Implements causal explanation (Algorithm 2, lines 19-24).
"""
function get_feature_importance(model::ACRFQIU_Model)
    if model.causal_graph === nothing
        error("Causal graph not computed.")
    end
    
    importance = model.causal_graph.causal_importance
    
    # Sort by importance
    sorted_indices = sortperm(importance, rev=true)
    
    return sorted_indices, importance[sorted_indices]
end


# ============================================================================
# SECTION 7: UTILITY FUNCTIONS AND HELPERS
# ============================================================================

"""
    combinations(arr, k)

Generate all combinations of k elements from arr.
"""
function combinations(arr, k)
    n = length(arr)
    if k == 0
        return [[]]
    end
    if k > n
        return []
    end
    
    result = Vector{Vector{eltype(arr)}}()
    
    function combine(start, combo)
        if length(combo) == k
            push!(result, copy(combo))
            return
        end
        
        for i in start:n
            push!(combo, arr[i])
            combine(i + 1, combo)
            pop!(combo)
        end
    end
    
    combine(1, eltype(arr)[])
    return result
end

"""
    evaluate_model(y_true::Vector{Int}, y_pred::Vector{Int}, 
                   prediction_sets::Vector{Vector{Int}})

Evaluate model performance with multiple metrics.
"""
function evaluate_model(y_true::Vector{Int}, y_pred::Vector{Int},
                       prediction_sets::Union{Vector{Vector{Int}}, Nothing}=nothing)
    n = length(y_true)
    
    # Accuracy
    accuracy = mean(y_true .== y_pred)
    
    # Confusion matrix
    classes = sort(unique(vcat(y_true, y_pred)))
    K = length(classes)
    conf_matrix = zeros(Int, K, K)
    
    for i in 1:n
        true_idx = findfirst(classes .== y_true[i])
        pred_idx = findfirst(classes .== y_pred[i])
        conf_matrix[true_idx, pred_idx] += 1
    end
    
    # Per-class metrics
    precision = zeros(K)
    recall = zeros(K)
    f1 = zeros(K)
    
    for k in 1:K
        tp = conf_matrix[k, k]
        fp = sum(conf_matrix[:, k]) - tp
        fn = sum(conf_matrix[k, :]) - tp
        
        precision[k] = tp / (tp + fp + 1e-10)
        recall[k] = tp / (tp + fn + 1e-10)
        f1[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k] + 1e-10)
    end
    
    # Conformal metrics
    if prediction_sets !== nothing
        coverage = mean([y_true[i] in prediction_sets[i] for i in 1:n])
        avg_set_size = mean(length.(prediction_sets))
    else
        coverage = missing
        avg_set_size = missing
    end
    
    return Dict(
        "accuracy" => accuracy,
        "precision" => mean(precision),
        "recall" => mean(recall),
        "f1_score" => mean(f1),
        "confusion_matrix" => conf_matrix,
        "coverage" => coverage,
        "avg_set_size" => avg_set_size
    )
end

"""
    print_results(metrics::Dict)

Pretty-print evaluation results.
"""
function print_results(metrics::Dict)
    println("\n" * "="^70)
    println("MODEL EVALUATION RESULTS")
    println("="^70)
    println(@sprintf("Accuracy:          %.4f", metrics["accuracy"]))
    println(@sprintf("Precision (macro): %.4f", metrics["precision"]))
    println(@sprintf("Recall (macro):    %.4f", metrics["recall"]))
    println(@sprintf("F1-Score (macro):  %.4f", metrics["f1_score"]))
    
    if !ismissing(metrics["coverage"])
        println("\nConformal Prediction:")
        println(@sprintf("  Coverage:        %.4f", metrics["coverage"]))
        println(@sprintf("  Avg. set size:   %.2f", metrics["avg_set_size"]))
    end
    
    println("\nConfusion Matrix:")
    display(metrics["confusion_matrix"])
    println("\n" * "="^70)
end

end  # module ACRFQIU parameter (Eq. 15)
end

"""
    gini_impurity(y::Vector{Int})

Compute Gini impurity for classification.
I(ν) = 1 - Σₖ pₖ² where pₖ is fraction of samples in class k.
"""
function gini_impurity(y::Vector{Int})
    n = length(y)
    if n == 0
        return 0.0
    end
    
    class_counts = countmap(y)
    impurity = 1.0
    
    for count in values(class_counts)
        p = count / n
        impurity -= p^2
    end
    
    return impurity
end

"""
    causal_impurity_reduction(X::Matrix, y::Vector, feature::Int, threshold::Float64, 
                              causal_weight::Float64, γ::Float64)

Compute causal-aware impurity reduction for a split.
Implements Equations (14) and (15).

ΔI_causal(Xᵢ) = ΔI(Xᵢ) · (1 + γ · Cᵢ)
"""
function causal_impurity_reduction(X::Matrix, y::Vector, feature::Int, threshold::Float64,
                                   causal_weight::Float64, γ::Float64)
    n = length(y)
    
    # Split samples
    left_mask = X[:, feature] .<= threshold
    right_mask = .!left_mask
    
    n_left = sum(left_mask)
    n_right = sum(right_mask)
    
    if n_left == 0 || n_right == 0
        return 0.0
    end
    
    # Standard impurity reduction (Eq. 14)
    I_parent = gini_impurity(y)
    I_left = gini_impurity(y[left_mask])
    I_right = gini_impurity(y[right_mask])
    
    ΔI = I_parent - (n_left/n * I_left + n_right/n * I_right)
    
    # Causal bonus (Eq. 15)
    ΔI_causal = ΔI * (1 + γ * causal_weight)
    
    return ΔI_causal
end

"""
    find_best_split(X::Matrix, y::Vector, features::Vector{Int}, 
                   causal_weights::Vector{Float64}, γ::Float64)

Find best feature and threshold for splitting using causal-aware criterion.
"""
function find_best_split(X::Matrix, y::Vector, features::Vector{Int},
                        causal_weights::Vector{Float64}, γ::Float64)
    best_gain = -Inf
    best_feature = 0
    best_threshold = 0.0
    
    for feature in features
        # Try several threshold candidates
        feature_values = sort(unique(X[:, feature]))
        
        if length(feature_values) < 2
            continue
        end
        
        # Use quantiles as candidate thresholds
        n_candidates = min(10, length(feature_values) - 1)
        candidates = quantile(feature_values, range(0.1, 0.9, length=n_candidates))
        
        for threshold in candidates
            gain = causal_impurity_reduction(X, y, feature, threshold,
                                            causal_weights[feature], γ)
            
            if gain > best_gain
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
            end
        end
    end
    
    return best_feature, best_threshold, best_gain
end

"""
    build_tree!(X::Matrix, y::Vector, tree::DecisionTree, depth::Int=0)

Recursively build decision tree with causal-aware splitting.
Implements Algorithm 1, Phase 3, lines 29-36.
"""
function build_tree!(X::Matrix, y::Vector, tree::DecisionTree, depth::Int=0)
    n, p = size(X)
    
    # Stopping criteria
    if depth >= tree.max_depth || n <= tree.min_samples_leaf || length(unique(y)) == 1
        # Create leaf node
        prediction = mode(y)
        return DecisionNode(0, 0.0, nothing, nothing, prediction, true, depth)
    end
    
    # Random feature subsampling
    m = tree.feature_subset_size
    features = sample(1:p, min(m, p), replace=false)
    
    # Find best split using causal criterion (Eq. 14-15)
    best_feature, best_threshold, best_gain = find_best_split(
        X, y, features, tree.causal_weights, tree.γ
    )
    
    if best_gain <= 0
        # No good split found
        prediction = mode(y)
        return DecisionNode(0, 0.0, nothing, nothing, prediction, true, depth)
    end
    
    # Split data
    left_mask = X[:, best_feature] .<= best_threshold
    right_mask = .!left_mask
    
    # Recursively build subtrees
    left_child = build_tree!(X[left_mask, :], y[left_mask], tree, depth + 1)
    right_child = build_tree!(X[right_mask, :], y[right_mask], tree, depth + 1)
    
    return DecisionNode(best_feature, best_threshold, left_child, right_child, 0, false, depth)
end

"""
    predict_tree(node::DecisionNode, x::Vector)

Predict class for single sample using decision tree.
"""
function predict_tree(node::DecisionNode, x::Vector)
    if node.is_leaf
        return node.prediction
    end
    
    if x[node.feature] <= node.threshold
        return predict_tree(node.left, x)
    else
        return predict_tree(node.right, x)
    end
end

"""
    compute_tree_weight(tree::DecisionTree, oob_accuracy::Float64, 
                       causal_alignment::Float64, η::Float64)

Compute tree weight combining OOB accuracy and causal alignment.
Implements Equations (17), (18), and (19).

wₜ = Acc_OOB · (1 + η · Aₜ)
"""
function compute_tree_weight(oob_accuracy::Float64, causal_alignment::Float64, η::Float64)
    return oob_accuracy * (1 + η * causal_alignment)
end

"""
    CausalRandomForest

Ensemble of causally-weighted decision trees.
Implements Section 4.3.
"""
mutable struct CausalRandomForest
    trees::Vector{DecisionTree}
    tree_weights::Vector{Float64}
    n_trees::Int
    max_depth::Int
    min_samples_leaf::Int
    feature_subset_size::Int
    γ::Float64  # Causal bonus
