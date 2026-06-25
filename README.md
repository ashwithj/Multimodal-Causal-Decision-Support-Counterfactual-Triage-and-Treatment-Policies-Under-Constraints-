# ICU Admission Causal Inference & Mortality Prediction

A machine learning pipeline that combines **predictive modeling** and **causal inference** to answer two related clinical questions:

1. *Can we predict which patients will die in the hospital?*
2. *For which patients does ICU admission actually reduce mortality?*

Built on the [MIMIC](https://mimic.mit.edu/) clinical database using XGBoost, T-Learner meta-learners, and fairness analysis across demographic subgroups.

---

## Project Overview

Most ICU admission decisions are made heuristically. This project estimates **Individual Treatment Effects (ITEs)** — the personalized causal effect of ICU admission on in-hospital mortality for each patient — using the T-Learner framework on top of a validated XGBoost propensity model.

**Key questions answered:**
- Which patients *benefit* from ICU admission (ITE < 0 = ICU reduces mortality)?
- Which patients are harmed or unaffected (ITE ≥ 0)?
- Does the predictive model perform fairly across race, gender, and insurance subgroups?

---

## Repository Structure

```
├── data/
│   └── 0_final_dataset.csv              # Final merged MIMIC dataset (~155K patients)
│
├── scripts/
│   ├── t_learnerV2.py                   # T-Learner meta-learner (6 base models + XGBoost)
│   ├── cate_model.py                    # CATE estimation using CausalForestDML + BERTopic
│   ├── xgb_sensitivity_trimmed.py       # Sensitivity analysis on XGBoost hyperparameters
│   └── patientrank.py                   # Patient benefit ranking by ITE score
│
├── notebooks/
│   └── Notebook 3 S_T-Learner 1.ipynb  # Full T-Learner pipeline (Azure Synapse / PySpark)
│
├── results/
│   ├── Predictive Model Results/        # XGBoost evaluation, SHAP, fairness outputs
│   ├── Causal Model Results/            # T-Learner ITE distributions and comparisons
│   ├── Causal Model - Sensitivity/      # Hyperparameter sensitivity heatmaps
│   ├── Causal Model - Ranking/          # Patient benefit ranking visualizations
│   └── T Learner V4 Results/            # Final trimmed T-Learner outputs
│
└── docs/
    ├── Project Proposal.docx
    ├── Final Proposal Text.docx
    └── Group 5 Research.docx
```

---

## Dataset

**Source:** MIMIC-IV (de-identified EHR data, PhysioNet access required)

**Features include:**
- Demographics: age, gender, race, insurance, marital status
- Lab values: ALT, creatinine, hemoglobin, lactate, WBC, BUN, glucose, and 40+ others
- Admission metadata: admission type, location, ICU status
- Clinical notes (NLP-processed for BERTopic subgroup discovery)
- Diagnostic categories (one-hot encoded, 150+ categories)

**Target variable:** `hospital_expire_flag` (in-hospital mortality)  
**Treatment variable:** `icu_admitted` (binary — was the patient admitted to the ICU?)

> **Note:** Access to `0_final_dataset.csv` requires a PhysioNet credentialed account. The file is not redistributed here.

---

## Pipeline

### 1. Predictive Model (`best_model.pkl`)

An XGBoost classifier trained to predict in-hospital mortality.

| Metric | Value |
|--------|-------|
| Test AUROC | 0.8584 |
| Test AUPRC | 0.6553 |
| CV AUROC (mean ± std) | 0.858 ± 0.001 |
| Optimal threshold | 0.43 |

Hyperparameter tuning via Optuna (238 estimators, learning rate 0.05, max depth 5).

### 2. Causal Model — T-Learner (`t_learnerV2.py`)

The T-Learner trains separate outcome models for treated (ICU) and control (non-ICU) patients, then computes ITE = μ₁(x) − μ₀(x) for each patient.

Six base learners evaluated:

| Model | AUROC | PEHE | ATE Error |
|-------|-------|------|-----------|
| XGBoost | 0.906 | 0.260 | 0.097 |
| GradientBoost | 0.882 | 0.260 | 0.046 |
| Logistic Regression | 0.857 | 0.290 | 0.013 |
| SVC | 0.920 | 0.176 | 0.004 |
| Random Forest | 0.848 | 0.189 | 0.002 |
| Decision Tree | 0.827 | 0.324 | 0.052 |

**Best model:** XGBoost T-Learner (highest AUROC, competitive PEHE)

Overlap trimming applied: patients with propensity scores outside [0.1, 0.9] excluded to reduce confounding.

### 3. Patient Ranking (`patientrank.py`)

Patients ranked by ITE score from best T-Learner model:
- **ITE < 0** → ICU *reduces* mortality → patient **benefits**
- **ITE > 0** → ICU *increases* mortality → patient does **not benefit**
- **ITE ≈ 0** → No meaningful effect

Outputs include benefit group breakdowns by age, clinical profile, and ITE distribution.

### 4. Sensitivity Analysis (`xgb_sensitivity_trimmed.py`)

Grid search over 27 hyperparameter combinations (n_estimators × max_depth × learning_rate). PEHE ranges from ~0.217 to ~0.238 across configurations, confirming model stability.

### 5. Fairness Analysis

Evaluated across **gender**, **race**, **insurance**, and **age group**.

| Subgroup | AUROC | Note |
|----------|-------|------|
| Female | 0.855 | Sensitivity 10pp lower than male |
| Male | 0.855 | Higher prevalence (23.9%) |
| White | 0.837 | Largest group (n=20,713) |
| Native Hawaiian/PI | 0.951 | Small group (n=41) |
| Black/African American | 0.805 | Lower AUPRC (0.352) |
| Unable to Obtain | 0.878 | High prevalence (58%) |

Key fairness concerns: sensitivity gap by gender (9.1pp), AUROC spread of 0.3 across race subgroups, FPR disparities.

---

## Installation

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn econml bertopic sentence-transformers joblib
```

For the notebook: requires Azure Synapse / PySpark environment (or adapt to local Jupyter).

---

## Usage

```bash
# Run T-Learner causal model
python scripts/t_learnerV2.py

# Run sensitivity analysis
python scripts/xgb_sensitivity_trimmed.py

# Rank patients by predicted ICU benefit
python scripts/patientrank.py

# CATE estimation with subgroup discovery
python scripts/cate_model.py
```

All scripts expect `0_final_dataset.csv` and `best_model.pkl` in the working directory.

---

## Results Summary

- Predictive model achieves **AUROC 0.858** on held-out test set across 33,518 patients
- XGBoost T-Learner identifies a meaningful subset of patients where ICU admission is predicted to reduce mortality
- Fairness analysis flags sensitivity and FPR disparities across gender and racial subgroups — a key limitation for clinical deployment
- Sensitivity analysis confirms PEHE stability (CV ~0.22–0.24) across hyperparameter configurations

---

## Limitations

- MIMIC-IV is a single-institution dataset (BIDMC); generalizability to other health systems is unknown
- T-Learner assumes no unmeasured confounders — violated if ICU decisions are driven by factors not in the data
- Fairness gaps in sensitivity and FPR across race/gender require further mitigation before clinical use
- `best_model.pkl` serves dual purpose as both propensity model and predictive model — these should ideally be separated

---

## Team

**Group 5**

---

## License

This project uses MIMIC-IV data, which requires a credentialed PhysioNet account and completion of CITI training. Do not redistribute raw data. Code is released for academic use.
