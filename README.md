# AdaptALS: Adaptive Hybrid ALS Recommender System

An adaptive hybrid recommender system that combines a long-term **implicit-feedback ALS** model with a lightweight short-term latent preference model to adapt to changing user preferences while reducing unnecessary full ALS retraining.

The project is implemented using **PySpark**, **NumPy**, and the **MovieLens-1M** dataset.

---

## Overview

Traditional recommender systems often retrain their global model at fixed time intervals, even when user preferences have not changed substantially.

**AdaptALS** separates adaptation into two levels:

* **Long-term adaptation:** a global ALS model captures stable user-item preferences.
* **Short-term adaptation:** recent interactions update a temporary latent preference representation without retraining ALS.

The disagreement between long- and short-term recommendations determines the short-term contribution. This signal is then used to decide whether a full ALS retraining is necessary.

```text
MovieLens-1M
     │
     ▼
Chronological Split
     │
     ├── Training ──► Initial ALS ──► Long-Term Model
     │
     └── Validation ──► Parameter Selection
                              │
                         γ*, λ*, τ*
                              │
                              ▼
                        Test Stream
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          Long-Term ALS             Short-Term Model
                 │                         │
                 └──────────┬──────────────┘
                            ▼
                     Dynamic Weighting
                            │
                            ▼
                    Hybrid Recommendation
                            │
                            ▼
                    Weekly Average w₂
                            │
                     ┌──────┴──────┐
                     ▼             ▼
                  No Retrain    Retrain ALS
```

---

## Dataset

The project uses **MovieLens-1M**. Each interaction contains:

```text
(userId, movieId, rating, timestamp)
```

The data are divided chronologically into:

* 70% Training
* 15% Validation
* 15% Test

This preserves the temporal ordering required to evaluate preference adaptation.

---

## Core Method

### Long-Term ALS

The long-term model uses PySpark implicit-feedback ALS with:

| Parameter         | Value |
| ----------------- | ----: |
| Rank              |    32 |
| Max iterations    |    15 |
| Regularization    |  0.05 |
| Alpha             |  20.0 |
| Implicit feedback |  True |
| Seed              |    42 |

The recommendation score is:

```text
s_B(u,i) = u_u · v_i
```

The learned item factors are also reused by the short-term model.

### Short-Term Preference

The short-term component **does not train a second ALS model**. It creates a temporary user vector from recent interactions using a recency- and rating-weighted centroid:

```text
weight_m = γ^(n-1-m) × rating_m
```

```text
x̃_u = Σ(weight_m × v_i_m) / Σ(weight_m)
```

The short-term score is:

```text
s_S(u,i) = x̃_u · v_i
```

The implementation retains up to 20 recent interactions in the active buffer.

### Dynamic Fusion

The baseline and short-term Top-100 lists are compared using Jaccard dissimilarity:

```text
D(u) = 1 - |K_B ∩ K_S| / |K_B ∪ K_S|
```

Interaction confidence is:

```text
C(n) = n / (n + λ)
```

The adaptive weights are:

```text
w₂ = D(u) × C(n)
w₁ = 1 - w₂
```

The hybrid recommendation score is:

```text
s_H(u,i) = w₁s_B(u,i) + w₂s_S(u,i)
```

The candidate pool is the union of the baseline and short-term Top-100 lists.

---

## Adaptive Retraining

The test stream is processed in fixed **seven-day windows**.

For each window, the mean short-term influence is calculated:

```text
average_w₂ = mean(w₂)
```

ALS is physically retrained only when:

```text
average_w₂ ≥ τ
```

Otherwise, the current ALS factors are retained and only the short-term representation continues to adapt.

The baseline system, in contrast, physically retrains ALS after every seven-day window.

---

## Pareto-Based Parameter Selection

The adaptive parameters are selected using the validation stream.

### Search space

```text
γ ∈ {0.60, 0.70, 0.80, 0.90}

λ ∈ {2, 5, 8, 10}

τ ∈ {0.10, 0.15, ..., 0.85}
```

This produces **240 configurations**.

Each configuration is evaluated using two objectives:

1. **Maximize NDCG@10**
2. **Minimize ALS retraining count**

A configuration is considered dominated if another configuration provides at least as high NDCG and no more retraining, with at least one objective strictly better.

Only the **non-dominated configurations** are retained to form the Pareto frontier.

The frontier objectives are normalized, and the knee point is selected as the point representing the strongest trade-off between recommendation quality and retraining cost.

The resulting:

```text
(γ*, λ*, τ*)
```

is fixed before the final test evaluation.

---

## Evaluation

AdaptALS is compared against the fixed-retraining ALS baseline.

### Ranking metrics

* NDCG@10, @50, @100
* Precision@10, @50, @100
* Recall@10, @50, @100

### Rating metrics

* RMSE
* MAE

### Computational metrics

* Number of physical ALS retrainings
* Execution time

---

## Project Structure

```text
AdaptALS
│
├── data/
│   ├── ratings1.dat
│   └── desktop.ini
│
├── src/
│   ├── __init__.py
│   ├── baseline_als.py
│   ├── data_loader.py
│   ├── pareto_optimizer.py
│   ├── short_term_model.py
│   └── weighting.py
│
├── final_run_v7.py
│
├── README.md
│
└── results_plots/
    ├── 1_dataset_split_summary.png
    ├── 2_pareto_threshold_selection.png
    ├── 2a_tau_sensitivity.png
    ├── 2b_gamma_sensitivity.png
    ├── 2c_lambda_sensitivity.png
    ├── 3_weekly_test_ndcg_comparison.png
    ├── 4_1_retrain_operations.png
    ├── 4_2_computation_time.png
    ├── 4_3_rating_accuracy_errors.png
    ├── 4_4_ranking_ndcg_performance.png
    ├── evaluation_summary.txt
    └── evaluation_summary_old.txt
```

### Main Components

| File                  | Purpose                                                          |
| --------------------- | ---------------------------------------------------------------- |
| `data_loader.py`      | Loads data and creates chronological splits                      |
| `baseline_als.py`     | Implements the long-term implicit ALS model                      |
| `short_term_model.py` | Creates the short-term latent representation                     |
| `weighting.py`        | Computes dissimilarity, confidence, and adaptive weights         |
| `pareto_optimizer.py` | Finds the Pareto frontier and selects the knee point             |
| `final_run_v7.py`          | Runs validation, testing, retraining, metrics, and visualization |

---

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
pyspark
pandas
numpy
scipy
matplotlib
seaborn
```

---

## Running

Place the MovieLens-1M ratings file at the location expected by `data_loader.py`, then run:

```bash
python final_run_v7.py
```

The experiment performs:

```text
Data loading
→ Chronological split
→ Initial ALS training
→ Validation parameter search
→ Pareto selection
→ Baseline evaluation
→ AdaptALS evaluation
→ Conditional ALS retraining
→ Metrics and plots
```

---

## Outputs

Results are generated under `restults_plots/`, including:

* baseline vs. AdaptALS performance tables
* ranking and rating metrics
* retraining statistics
* execution-time results
* Pareto-selection plots
* parameter sensitivity plots
* weekly performance plots
* overall comparison plots

---

## Reproducibility

Key fixed settings include:

```text
ALS rank        = 32
ALS iterations  = 15
ALS regParam    = 0.05
ALS alpha       = 20.0
ALS seed        = 42
Window size     = 7 days
Short-term buffer = 20
```

The validation search uses the fixed parameter grid described above.

---

## Research Contribution

AdaptALS investigates whether a recommender can maintain recommendation quality while reducing unnecessary global model updates by separating:

* stable long-term preference modeling,
* fast short-term preference adaptation,
* dynamic recommendation fusion, and
* behavior-triggered ALS retraining.

The central idea is to use **recommendation drift as a signal for model refresh**, rather than treating every fixed time interval as an automatic retraining event.
