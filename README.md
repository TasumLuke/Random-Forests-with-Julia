
# EWSF — Entropy-Weighted SubForest (Julia)

**Version:** 0.2.0  
**Purpose:** A novel Random-Forest–style classifier in Julia that:
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

## 1 — Setup (one-time)

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

## 2 — Files you will interact with

- `run_train.jl` (if present): simple CLI wrapper to run interactive training.
- `run_predict.jl` (if present): wrapper to load a saved model and run predictions on a CSV.
- `scripts/synthetic_demo.jl`: generates synthetic data, trains model, synthesizes drift, runs `adapt_weights!` and prints summaries.
- `scripts/ewsf_demo.ipynb`: Jupyter notebook demonstrating training, drift detection, plotting p-values, and model performance.
- `test/runtests.jl`: unit tests verifying drift adaptation and that the package components run.

---

## 3 — How to train on your CSV (interactive CLI)

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

## 4 — Predicting on new data (CSV)

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

## 5 — Drift detection and automatic weight adaptation

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

## 6 — Running the synthetic demo & notebook

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

## 7 — Running tests

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

## 8 — Using real data — checklist & tips

1. **CSV format**:
   - First row: header with column names.
   - Columns may be numeric or categorical (strings). The script will detect data types automatically.
   - Avoid complex nested types or arrays in CSV cells.

2. **Target column**:
   - Choose a categorical target (classification).
   - If your target is numeric but has ≤ 10 distinct values, the package will coerce it to categorical automatically; for regression tasks (continuous targets) the current implementation focuses on classification — contact me if you want regression support.

3. **Missing values**:
   - Numeric missing values are imputed with the column median (during preprocessing).
   - Categorical missing values are treated like any other level (and encoded).

4. **Data leakage**:
   - Ensure your CSV split (training vs incoming/test) does not leak future info into training. Drift adaptation assumes training data reflects the distribution of the label–feature relationship at training time.

5. **Scaling & performance**:
   - The package uses pure-Julia nested loops and recursion. For medium datasets (thousands of rows, tens of features) it will be fine. For very large datasets (100k+ rows or many features) consider:
     - Increasing `n_subforests` carefully,
     - Decreasing `n_trees` or `n_perm` for drift testing,
     - Using a compiled tree library later for speed, while retaining EWSF wrappers.

6. **Interpreting weights**:
   - Each subforest has a weight representing relative trust (initially from OOB accuracy). After `adapt_weights!` a subforest that relies on drifting features will receive lower weight, making the ensemble rely more on robust subforests.

---

## 9 — Advanced usage & extension ideas

- **OOB-based online monitoring**: periodically generate OOB metrics incrementally as new labeled data arrives and trigger retraining once performance drops below a threshold.
- **Hybrid retrain strategy**: use `adapt_weights!` for small drift; if many features drift, perform incremental or full retraining on recent labeled examples.
- **Regression support**: implement mean-squared-error splits in `tree.jl` and adapt the prediction to return means and variances.
- **Visualization**: extend the Jupyter notebook to show per-feature CDFs pre/post drift (for KS) and contingency tables (for categorical features).
- **Packaging & distribution**: publish `EWSF` on GitHub and register with Julia General registry if you want others to `Pkg.add("EWSF")`.

---

## 10 — Troubleshooting

- **"column not found" on predict**: ensure column names in new CSV exactly match the feature names used for training (case-sensitive).
- **Strange zeros in categorical mapping**: unseen categories in new data map to code `0`. If these are common, consider re-encoding the model with a broader training set or mapping unknowns explicitly.
- **Slow permutation p-values**: reduce `n_perm` (e.g., 200) for exploratory runs. For publication-quality p-values, use `n_perm >= 1000`.
- **Stack overflow / recursion limit during tree building**: limit `max_depth` (default ≤ 8) or increase Julia recursion limit cautiously.

---

## 11 — License & attribution

You can place your preferred license here (MIT, Apache‑2.0, etc.). Example:

```
MIT License
Copyright (c) 2025 You
Permission is hereby granted...
```

---

## 12 — Contact / next steps

If you want me to:
- Convert the package into a registered Julia package,
- Add regression support,
- Create a minimal Dash/Genie web UI for uploading CSVs and exploring p-values interactively,
- Or produce an exact `Manifest.toml` for a specific OS and Julia registry snapshot — tell me your OS and Julia version and I will produce the manifest.

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
