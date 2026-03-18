[![Shipping files](https://github.com/neuefische/ds-ml-project-template/actions/workflows/workflow-02.yml/badge.svg?branch=main&event=workflow_dispatch)](https://github.com/neuefische/ds-ml-project-template/actions/workflows/workflow-02.yml)

# Financial Inclusion in Africa — Group 0

**Team:** Andreas, Paulina, Rehma, Ernesto

---

## Project Overview

**Goal:** Build a machine learning model to predict which individuals in Kenya, Rwanda, Tanzania, and Uganda are most likely to have a bank account.

**Why it matters:** Bank account ownership is a key measure of financial inclusion, which is vital for economic growth. The model helps identify factors influencing access to formal financial services.

**Data source:** [Zindi — Financial Inclusion in Africa](https://zindi.africa/competitions/financial-inclusion-in-africa)

---

## Objectives & KPIs

**Objective:** Build a model that predicts whether an individual has a bank account to support financial inclusion strategies.

**Target variable (binary):**
- `1` = has a bank account
- `0` = does not have a bank account

**KPI:** F1-score on the test set (evaluated via Zindi leaderboard using Mean Absolute Error)

---

## Research Questions

- Which individuals are most likely to have or use a bank account?
- What is the state of financial inclusion in Kenya, Rwanda, Tanzania, and Uganda?
- Which key factors drive individuals' financial security?

---

## Dataset

**Features used:**

| Category | Features |
|---|---|
| Demographics | Gender, age, relationship with head of household, marital status |
| Location | Country, location type (urban/rural), household size |
| Socioeconomic | Education level, job type, cellphone access |

---

## Workflow

1. **Data exploration** — summary statistics, distributions, correlations, missing values
2. **Data cleaning** — handle missing values, encode categorical variables, check for duplicates
3. **Modeling** — train classifiers (e.g. logistic regression, random forest, XGBoost)
4. **Evaluation** — F1-score on test set, MAE for Zindi submission
5. **Submission** — generate prediction file and submit to Zindi leaderboard

### Key questions to answer during EDA

| Goal | Method |
|---|---|
| Which features might be useful | Checking correlations |
| Summary statistics | `df.describe()`, value counts |
| Which cleaning steps are necessary | Visualizing distributions |
| Which models might be appropriate | Exploring class balance, feature types |
| Whether data quality issues exist | Identifying missing values, duplicates, inconsistencies |

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

## Repository Structure

```
NF-ML-Project-Group-0/
├── data/               # Raw and processed datasets (not committed)
├── notebooks/          # Jupyter notebooks for EDA and modeling
├── models/             # Saved model files
├── example_files/      # Example training and prediction scripts
├── requirements.txt
└── README.md
```
