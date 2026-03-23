[![Shipping files](https://github.com/neuefische/ds-ml-project-template/actions/workflows/workflow-02.yml/badge.svg?branch=main&event=workflow_dispatch)](https://github.com/neuefische/ds-ml-project-template/actions/workflows/workflow-02.yml)

# Financial Inclusion in Africa — Group 0

> *23,500 households. 4 countries. 1 clear pattern.*

**Team:** Andreas · Paulina · Rehma · Ernesto

---

## Project Summary

Only **14.3%** of surveyed adults in East Africa hold a bank account. We built a machine learning model to predict who is likely to have one — and, more importantly, to identify the structural factors that drive financial exclusion.

Our final model is an **XGBoost classifier tuned with Optuna (Bayesian optimization)**, optimized for **Recall** to minimize false negatives. In financial inclusion work, failing to identify someone who *could* be reached by financial services is more costly than a false alarm — so we prioritized recall over precision.

**Data source:** [Zindi — Financial Inclusion in Africa](https://zindi.africa/competitions/financial-inclusion-in-africa)
**Countries:** Kenya, Rwanda, Tanzania, Uganda
**Respondents:** ~23,500 households

---

## Key Findings

| # | Finding | Key Numbers | Chi² (statistical strength) |
|---|---------|-------------|------------------------------|
| 1 | Mobile phone access is the strongest predictor | 24.4% with phone vs. 7.7% without — **16.7 pp gap** | 1,033 |
| 2 | Gender gap is significant | Men: 19% · Women: 10.7% — **8.3 pp gap** | 323 |
| 3 | Education drives inclusion dramatically | University: 63% · No education: 5% — **58 pp gap** | 3,549 |
| 4 | Employment type matters — most are informal | Formal: 6% · Informal/farming: 74% · Dependent: ~20% | 3,032 |
| 5 | Urban residents are more included | Urban: 25.5% · Rural: 9.7% — **15.8 pp gap** | 179 |

> **pp = percentage points** (arithmetic difference between two percentages, not a relative change)

All chi-square values confirmed statistically significant (p < 0.001).

---

## Model Results

### Model Comparison

| Model | Recall (val) | AUC (val) | Notes |
|-------|-------------|-----------|-------|
| Logistic Regression (baseline) | ~0.47 | ~0.82 | Baseline reference |
| Random Forest (tuned) | 0.74 | 0.89 | GridSearchCV tuned |
| LightGBM | 0.76 | 0.90 | Fast gradient boosting |
| **XGBoost + Optuna** | **0.78** | **0.91** | **Final model — best recall** |

### XGBoost Tuning Strategy

| Step | Method | What changed |
|------|--------|-------------|
| 6a | Baseline XGBoost | Default parameters |
| 6b | `scale_pos_weight=16` | Correct class imbalance (14% positive class) |
| 6c | RandomizedSearchCV | Broad hyperparameter search |
| 6d | Cross-validation (StratifiedKFold, k=5) | Robust evaluation of best params |
| 6e | F2-score threshold optimization | Shift decision boundary to maximize recall |
| 6f | **Optuna Bayesian optimization** | Fine-tune: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda` |

### Optuna Best Parameters (XGBoost)

```
n_estimators:       350
max_depth:          5
learning_rate:      0.05
subsample:          0.85
colsample_bytree:   0.75
gamma:              0.1
reg_alpha:          0.5
reg_lambda:         2.0
scale_pos_weight:   16
```

**Why Optuna?** Optuna uses Bayesian optimization (Tree-structured Parzen Estimators) to intelligently explore the hyperparameter space. Unlike GridSearch, it learns from each trial which regions of the search space are promising — converging on good parameters in far fewer iterations than exhaustive search.

---

## Dataset

**Source:** Zindi Financial Inclusion in Africa competition

| Property | Value |
|----------|-------|
| Total rows | ~23,500 |
| Countries | Kenya, Rwanda, Tanzania, Uganda |
| Target variable | `bank_account` (binary: 1 = has account) |
| Positive class rate | ~14.3% |
| Class imbalance ratio | ~1:6 (account : no account) |

### Feature Categories

| Category | Features |
|----------|----------|
| Demographics | `gender_of_respondent`, `age_of_respondent`, `relationship_with_head`, `marital_status` |
| Location | `country`, `location_type` (urban/rural), `household_size` |
| Socioeconomic | `education_level`, `job_type`, `cellphone_access` |

### Job Type Distribution

| Category | Share |
|----------|-------|
| Formally employed | 6.1% |
| Informally employed | ~30% |
| Self-employed | ~25% |
| Farming/fishing/forestry | ~19% |
| Remittance dependent | 10.7% |
| Other/no income | 4.6% |
| No income | 2.7% |
| Government dependent | 1.0% |
| Don't know | 0.5% |

> ~74% of respondents fall in the informal/self-employed/farming sector. ~20% have no independent income (remittances, government dependent, no income) — these are not workers at all.

---

## Repository Structure

```
NF-ML-Project-Group-0/
│
├── data/                                   # Raw and processed datasets (not committed)
│
├── images/                                 # Saved chart images
│
├── models/                                 # Saved model files (.pkl, .joblib)
│
├── example_files/                          # Example training and prediction scripts
│
├── 260318-EDA_Demographics_EB.ipynb        # EDA: age, gender, household demographics
├── 260318-EDA_socioeconomic_variables.ipynb# EDA: education, employment, mobile, rural/urban
├── 260318-Train_Encoding_DF.ipynb          # Feature encoding for training data
├── 260318_EDA_Rehma.ipynb                  # EDA by Rehma
│
├── 260319-Model_Baseline-EB.ipynb          # Logistic regression baseline model
├── 260319-Baseline_model_final.ipynb       # Final baseline (shared)
├── 260319-Model_Encoding_Andy.ipynb        # Alternative encoding approach (Andy)
├── 260319-Decision_tree_guide_andy.ipynb   # Decision tree exploration (Andy)
├── 260319_Model_Random_Forest_Pauli.ipynb  # Random Forest model (Paulina)
├── 260319-Model_XGBoost-EB.ipynb           # XGBoost model with Optuna tuning (Ernesto)
│
├── 260320-LazyPredict-EB.ipynb             # LazyPredict model comparison
├── 260320-lgbm_model_andy.ipynb            # LightGBM model (Andy)
├── 20260219_Model_Random_Forest_Pauli.ipynb# Updated Random Forest (Paulina)
│
├── StarterNotebook.ipynb                   # Zindi starter notebook
├── EDA-and-modeling.ipynb                  # Early combined EDA/modeling exploration
│
├── submission_baseline_logreg.csv          # Zindi submission file (baseline)
├── requirements.txt
├── Makefile
└── README.md
```

### Recommended Run Order

1. `260318-EDA_Demographics_EB.ipynb` — understand the data and target distribution
2. `260318-EDA_socioeconomic_variables.ipynb` — chi-square analysis and key finding charts
3. `260318-Train_Encoding_DF.ipynb` — encode features and prepare training data
4. `260319-Model_Baseline-EB.ipynb` — logistic regression baseline
5. `260319-Model_XGBoost-EB.ipynb` — full XGBoost pipeline with Optuna (final model)
6. `260320-lgbm_model_andy.ipynb` — LightGBM comparison

---

## Optimization Approach

Our project client is **FinAfrica NGO**, whose goal is to *reach people who are excluded from financial services*. This shapes our optimization target:

- **Minimize False Negatives** (don't miss someone who could be reached)
- **Metric priority:** Recall > AUC > Precision
- **Threshold tuning:** F2-score optimization (weights recall twice as much as precision)
- **Class imbalance:** `scale_pos_weight=16` in XGBoost (inverse of class ratio)

---

## Set up your Environment

### macOS

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Or use the Makefile:

```bash
make setup
source .venv/bin/activate
```

### Windows (PowerShell)

```PowerShell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (Git Bash)

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** If `pip install --upgrade pip` fails on Windows, try `python.exe -m pip install --upgrade pip`

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `xgboost` | Final model |
| `optuna` | Bayesian hyperparameter optimization |
| `lightgbm` | LightGBM comparison model |
| `scikit-learn` | Baseline, metrics, cross-validation |
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `lazypredict` | Quick model comparison |
| `nbdime` | Notebook merge conflict resolution |

---

## Handling Merge Conflicts in Jupyter Notebooks

When working in teams, `.ipynb` files can cause messy merge conflicts because they're JSON-based.
We use **nbdime** to make this easy.

### Setup (run once)
```bash
nbdime config-git --enable
```

### When a conflict happens
```bash
nbdime mergetool
```

A web interface will open showing both notebook versions side by side.
Choose what to keep, save and close the tool, then:
```bash
git add your_notebook.ipynb
git commit -m "Resolved notebook conflict"
```

---

## Team

| Name | Focus Area |
|------|-----------|
| Ernesto | EDA, XGBoost + Optuna, Presentation |
| Andreas | EDA, Decision trees, LightGBM, encoding, Presentation |
| Paulina | EDA, Random Forest, model tuning, Presentation |
| Rehma | EDA, data exploration, Presentation |
