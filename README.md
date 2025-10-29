
# EWSF: Entropy-Weighted SubForest (Julia)

**Version:** 0.2.0  
**Purpose:** A Random-Forest–style classifier in Julia that:
- Automatically parses CSVs and detects column types,
- Lets you choose features and target,
- Trains multiple *entropy‑weighted sub‑forests* (different feature-emphasis levels),
- Initializes sub‑forest weights from OOB accuracy,
- Detects input-data drift (per‑feature permutation KS / chi‑square tests) and adapts sub‑forest weights automatically,
- Produces soft probabilities (per-class probabilities) and hard predictions,
- Includes tools to explain model importance (permutation importance),
- Provides a synthetic demo, tests, and a Jupyter notebook.

---

## Repository layout

```
ewsf-julia/
├─ Project.toml
├─ Manifest.toml                # generate locally with `Pkg.instantiate()`
├─ README.md                    # ← this file
├─ src/
│  ├─ EWSF.jl                   # package entry point
│  ├─ data.jl
│  ├─ tree.jl
│  ├─ model.jl
│  └─ utils.jl
├─ scripts/
│  ├─ synthetic_demo.jl
│  └─ ewsf_demo.ipynb
├─ test/
│  └─ runtests.jl
└─ sample_data.csv
```

---

## Quick requirements

- **Julia** 1.8 or later recommended.
- Packages: `CSV`, `DataFrames`, `CategoricalArrays`, `StatsBase`, `Random`, `Serialization`, `Test`, `Plots`, `IJulia`, `StatsPlots`.
- Install dependencies via the Julia package manager (instructions below).

---

## Setup (one-time)

1. Clone or copy the repository into a folder, e.g.:

```bash
git clone <your-repo-url> ewsf-julia
cd ewsf-julia
```

2. Activate the project and install dependencies in Julia REPL:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()   # will install packages and create Manifest.toml for your environment
```

If you prefer to add packages manually:

```julia
Pkg.activate(".")
Pkg.add(["CSV","DataFrames","CategoricalArrays","StatsBase","Plots","IJulia","StatsPlots"])
```

> **Note:** For exact reproducibility across machines, run the `Pkg.instantiate()` command which will write a complete `Manifest.toml` tailored for your environment.

---

## Files 
- `run_train.jl` (if present): simple CLI wrapper to run interactive training.
- `run_predict.jl` (if present): wrapper to load a saved model and run predictions on a CSV.
- `scripts/synthetic_demo.jl`: generates synthetic data, trains model, synthesizes drift, runs `adapt_weights!` and prints summaries.
- `scripts/ewsf_demo.ipynb`: Jupyter notebook demonstrating training, drift detection, plotting p-values, and model performance.
- `test/runtests.jl`: unit tests verifying drift adaptation and that the package components run.

---

## How to train on your CSV (interactive CLI)

1. Start Julia in repo root:

```bash
julia --project=.
```

2. From the Julia REPL you may run interactive training (if `run_train.jl` exists) or call package functions:

```julia
include("run_train.jl")
```

`run_train.jl` will prompt:

- Path to CSV (press Enter to use `sample_data.csv`),
- Index of target column,
- Feature selection (use all or pick indices),
- Hyperparameters: `n_trees`, `max_depth`, `min_leaf`, `n_subforests`,
- A filename to save the trained model (e.g. `ewsf_model.bin`).

**Under the hood**:
- The CSV is preprocessed: numeric columns converted to floats, missing numeric values filled with medians; categorical columns are encoded to integer codes with encoders saved in the model.
- The training routine builds `n_subforests` each with `n_trees / n_subforests` trees. Sub‑forests are biased to emphasize different groups of informative features by raising feature scores to different exponents (alpha).
- Each tree is trained with bootstrap sampling and collects OOB indices for that tree. After subforest training, OOB accuracy becomes its initial weight (normalized across subforests).

---

## Predicting on new data (CSV)

1. Use the CLI wrapper:

```bash
julia --project=. run_predict.jl
```

It will prompt for:
- Path to new CSV file,
- Path to saved model file (e.g. `ewsf_model.bin`).

Or call from Julia:

```julia
using Serialization
model = Serialization.deserialize("ewsf_model.bin")
using EWSF
pred_df = EWSF.EWSFModel.predict_model(model, new_dataframe)
```

**Important Input Format Notes**:

- The new CSV must have the same feature column names used in training (case-sensitive), and must include exactly the same columns (or at least the columns used by the model). Extra columns are ignored.
- For categorical columns, unseen categories in new data are mapped to code `0` (unknown), which the model can handle, but it's best to keep training set representative.

**Output**:
- `predict_model` returns a `DataFrame` where each column corresponds to the probability for a class label and a final `:prediction` column with the highest-probability label.

---

## Drift detection and automatic weight adaptation

To detect drift between your original training dataset and a new incoming dataset, call:

```julia
summary = EWSF.EWSFModel.adapt_weights!(model, train_df, new_df; p_threshold=0.05, n_perm=500, lambda=3.0)
```

Where:

- `train_df` is the preprocessed training DataFrame (the *same* object returned by `data_preprocess`).
- `new_df` is a preprocessed-like DataFrame for incoming data. For convenience the package provides helper functions to convert raw incoming CSVs into the form expected (see demo).
- `p_threshold` is the significance threshold (default `0.05`).
- `n_perm` is number of permutations used to compute permutation p-values (bigger → more accurate, slower).
- `lambda` controls aggressiveness of weight decay: new weight = old_weight * exp(−lambda * drift_fraction). Higher `lambda` reduces weight faster.

`summary` contains:
- `"pvals"`: per-feature permutation p-values,
- `"stats"`: per-feature statistics (KS or chi2),
- `"drift_scores"`: fraction of a subforest’s features flagged as drifting,
- `"new_weights"`: updated weights per subforest (normalized).

**Recommended workflow**:
- Use `feature_drift_pvalues` to inspect which features changed most.
- If drift is pervasive (many features with small p-values), consider retraining using recent labeled examples.
- The `adapt_weights!` routine adjusts ensemble behavior when minor/partial drift occurs by down‑weighting subforests that rely on drifting features.

---

## Running the synthetic demo & notebook

**Synthetic CLI demo**:

```bash
julia --project=. scripts/synthetic_demo.jl
```

This script:
- Generates synthetic train/test data,
- Trains EWSF,
- Induces drift (age shift & more smokers),
- Converts incoming data to encoded form,
- Runs `adapt_weights!` and prints p-values & updated weights.

**Notebook**:

Start the notebook server from the package root:

```bash
julia --project=.
julia> using IJulia; notebook(dir="scripts")
```

Open `scripts/ewsf_demo.ipynb`. The notebook shows plots of p-values, pre/post-adaptation weights, and simple accuracy checks.

---

## Running tests

From the package root execute:

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

Or in the REPL:

```julia
using Pkg
Pkg.activate(".")
Pkg.test()
```

This runs `test/runtests.jl` which contains unit tests (e.g., ensuring `adapt_weights!` changes subforest weights when drift exists).

---


## Troubleshooting

- **"column not found" on predict**: ensure column names in new CSV exactly match the feature names used for training (case-sensitive).
- **Strange zeros in categorical mapping**: unseen categories in new data map to code `0`. If these are common, consider re-encoding the model with a broader training set or mapping unknowns explicitly.
- **Slow permutation p-values**: reduce `n_perm` (e.g., 200) for exploratory runs. For publication-quality p-values, use `n_perm >= 1000`.
- **Stack overflow / recursion limit during tree building**: limit `max_depth` (default ≤ 8) or increase Julia recursion limit cautiously.

---

## Example quick commands

```bash
# Activate & instantiate
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run synthetic demo
julia --project=. scripts/synthetic_demo.jl

# Launch demo notebook
julia --project=. -e 'using IJulia; notebook(dir="scripts")'
```

---

Thank you — enjoy experimenting with EWSF. If you'd like, I will now:
- produce a fully resolved `Manifest.toml` for macOS/Linux/Windows for a specific Julia version (tell me OS and version), **or**
- add a simple web UI for uploading CSV and viewing p-values and predictions.
