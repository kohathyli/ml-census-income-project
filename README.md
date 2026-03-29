# Census Income Classification and Customer Segmentation

This repository contains my solution to a machine learning take-home project based on weighted U.S. Census data.

The project addresses two business questions:

1. **Classification:** identify individuals who are likely to have income **above $50K** versus **at or below $50K**
2. **Segmentation:** group the population into actionable customer segments for marketing use

The goal of this project is not only to build accurate models, but also to show clear business judgment:  
- which model should be trusted for targeting  
- how model outputs should be used in practice  
- how technical clustering results can be translated into marketing personas  

---

## TL;DR

- **Best model:** CatBoost (ROC-AUC 0.956, Precision 0.891)
- **Targeting strategy:** precision-focused (~2.3% population selected)
- **Segmentation:** 5 actionable personas derived from k=6 clustering
- **Key insight:** combine model-based targeting with persona-based messaging

## Final Recommendations

### Classification

Three model families were evaluated:

- **Logistic Regression** — interpretable linear baseline  
- **Random Forest** — nonlinear tree-based benchmark  
- **CatBoost** — gradient boosting model (final)

**Final recommendation: use CatBoost for production targeting.**

Model comparison showed:

- Logistic Regression provides interpretability but limited nonlinear capacity  
- Random Forest improves performance by capturing interactions  
- CatBoost delivers the strongest overall ranking and precision on the rare positive class  

This makes CatBoost the best choice for **high-confidence, cost-sensitive targeting**.

### Segmentation

The final segmentation uses:

- **k = 6** as the technical solution  
- **5 client-facing personas** after consolidation  

**Recommendation:** Use the 5-persona framework for campaign design.

This reflects a key business insight:

> the best technical segmentation is not always the most actionable one.
---

## Repository Structure


```text
ml-census-income-project/
├── artifacts/
│   ├── classifier/                # logistic regression outputs
│   ├── random_forest_classifier/  # random forest benchmark outputs
│   ├── catboost_classifier/       # final recommended classifier outputs
│   ├── segmentation/              # default segmentation outputs
│   ├── segmentation_k6/           # forced k=6 comparison
│   ├── segmentation_k7/           # forced k=7 comparison
│   ├── threshold_analysis/        # threshold table & cost analysis
│   ├── shap_analysis/             # SHAP values and plots
│   ├── baseline_snapshot/
│   └── final_snapshot/
│
├── data/
│   ├── raw/
│   │   ├── census-bureau.columns
│   │   └── census-bureau.data
│   └── processed/
│
├── reports/
│   ├── figures/                   # EDA + SHAP + segmentation visuals
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── models/
│   │   ├── train_classifier.py    # logistic regression
│   │   ├── train_random_forest.py # random forest benchmark
│   │   ├── train_catboost.py      # final model
│   │   └── segmentation.py
│   │
│   ├── analysis/
│   │   ├── build_persona_table.py     # build client-facing persona summary table
│   │   ├── build_threshold_table.py   # build threshold / cost trade-off table
│   │   └── eda_plots.py               # generate EDA figures for the report
│   │
│   └── utils/
│       ├── io.py
│       ├── metrics.py
│       └── visualization.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Environment Setup

I recommend using a clean virtual environment.

### Option 1: `venv`

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
```

### Option 2: `conda`

```bash
conda create -n census_ml python=3.10 -y
conda activate census_ml
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Run the baseline logistic regression + default segmentation

```bash
python main.py
```

This will:

- run the **Logistic Regression baseline classifier**
- run the **default segmentation pipeline**
- save outputs under **`artifacts/`**

---

## Run Each Part Separately

### 1) Logistic Regression Baseline

```bash
python -m src.models.train_classifier
```

#### Optional arguments

```bash
python -m src.models.train_classifier \
  --data-path data/raw/census-bureau.data \
  --columns-path data/raw/census-bureau.columns \
  --output-dir artifacts/classifier
```

This script will:

- load and preprocess the raw data
- train a weighted Logistic Regression classifier
- select a threshold using validation performance
- evaluate on validation and test sets
- save metrics, plots, reports, and feature importance

**Outputs go to:**

```text
artifacts/classifier/
```

---


### 2) Random Forest Benchmark

```bash
python -m src.models.train_random_forest
```

#### Optional arguments

```bash
python -m src.models.train_random_forest \
  --data-path data/raw/census-bureau.data \
  --columns-path data/raw/census-bureau.columns \
  --output-dir artifacts/random_forest_classifier
```

This script will:

- load and preprocess the raw data
- train a weighted Random Forest classifier
- select a threshold using validation performance
- evaluate on validation and test sets
- save metrics, plots, reports, ROC data, and feature importance

**Outputs go to:**

```text
artifacts/random_forest_classifier/
```

---

### 3) Final CatBoost Classifier

```bash
python -m src.models.train_catboost
```

#### Optional arguments

```bash
python -m src.models.train_catboost \
  --data-path data/raw/census-bureau.data \
  --columns-path data/raw/census-bureau.columns \
  --output-dir artifacts/catboost_classifier
```

This script will:

- load the raw data
- preserve categorical variables in native form
- train a CatBoost classifier with weighted fitting
- choose a threshold on the validation set
- evaluate the final model on the test set
- save metrics and feature importance outputs

**Outputs go to:**

```text
artifacts/catboost_classifier/
```

---

### 4) Segmentation

```bash
python -m src.models.segmentation
```

#### Optional arguments

```bash
python -m src.models.segmentation \
  --data-path data/raw/census-bureau.data \
  --columns-path data/raw/census-bureau.columns \
  --output-dir artifacts/segmentation
```

This script will:

- preprocess the data for clustering
- reduce dimensionality with TruncatedSVD
- evaluate multiple candidate values of k
- fit KMeans on the reduced representation
- generate cluster profiles and a marketing-oriented summary

**Outputs go to:**

```text
artifacts/segmentation/
```

---

## Segmentation Comparison: k = 6 vs k = 7

To reproduce the business comparison between the two candidate segmentation solutions:

### Force k = 6

```bash
python -m src.models.segmentation --k-min 6 --k-max 6 --output-dir artifacts/segmentation_k6
```

### Force k = 7

```bash
python -m src.models.segmentation --k-min 7 --k-max 7 --output-dir artifacts/segmentation_k7
```

This comparison supports the final decision to prefer **k = 6** for client-facing use, even though **k = 7** achieved the highest silhouette score.

---

## Key Modeling Choices

## Classification

### Logistic Regression baseline

The baseline pipeline uses:

- numeric imputation
- categorical imputation
- one-hot encoding
- weighted Logistic Regression
- validation-based threshold tuning

**Why this was useful:**

- interpretable and easy to explain
- strong benchmark on mixed tabular data
- useful for showing business reasoning, not just model performance

### Random Forest (nonlinear benchmark)

The Random Forest pipeline uses:

- the same preprocessing as Logistic Regression
- ensemble of decision trees to capture nonlinearities
- weighted training
- validation-based threshold tuning

**Why this was useful:**

- captures feature interactions and nonlinear effects
- improves performance over linear models
- serves as a strong tree-based benchmark before boosting methods

### CatBoost final model

The final production recommendation uses:

- native categorical feature handling
- gradient-boosted decision trees
- weighted fitting
- validation-based threshold tuning

**Why this was selected:**

- better suited to categorical-heavy tabular data
- captures nonlinearities and interactions more effectively
- achieved the strongest ranking and targeting performance

---

## Segmentation

The segmentation pipeline uses:

- preprocessing suitable for mixed tabular data
- dimensionality reduction with TruncatedSVD
- KMeans clustering on the reduced representation
- silhouette score for initial model guidance
- manual review of cluster sizes and profiles for final business selection

**Why this approach was chosen:**

- scalable on a large dataset
- practical for sparse, mixed-type data
- easy to translate into marketing personas
- supports both technical evaluation and business interpretation

---

## Business Framing

This project is intentionally structured as a client-facing analytics solution, not just a modeling exercise.

The classification model answers:

> **Who should the client prioritize?**

The segmentation model answers:

> **How should the client approach those people differently?**

That distinction matters. A prospect with a high predicted probability of being in the `>50K` group may still require different messaging depending on whether they belong to:

- a younger working-adult segment
- a mainstream working household segment
- a small affluent niche segment

The final output is therefore a combination of:

- predictive targeting
- persona-based marketing guidance

---

## Business Insights

This project combines two complementary capabilities:

- **Classification → who to target**
- **Segmentation → how to target them**

Key takeaways:

1. **High-income targeting is feasible with strong precision.**  
   The model supports focused, high-confidence outreach.

2. **Threshold selection is a business lever.**  
   Precision vs. reach can be adjusted based on campaign cost.

3. **Customers are structurally heterogeneous.**  
   Life stage and economic profile materially affect targeting strategy.

4. **Segmentation adds critical context to prediction.**  
   High-probability customers still require different messaging.
---

## Notes on Data and Reproducibility

The project uses the provided census data files:

- `census-bureau.data`
- `census-bureau.columns`

The survey weight is used analytically, but is not used as a predictive customer feature.

---

## References

Brief references consulted while working on the project:

- project-provided census data file and column description file
- **scikit-learn** documentation for preprocessing, Logistic Regression, evaluation metrics, and KMeans clustering
- **CatBoost** documentation for classifier training and categorical feature handling
- standard references for **ROC-AUC** and **precision-recall evaluation** in imbalanced classification