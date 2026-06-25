"""
cate_model.py
─────────────────────────────────────────────────────────────────────────────
CATE estimation for ICU admission → in-hospital mortality
Picks up from the predictive model (xgboost_model.py) output.

Inputs expected
───────────────
  0_final_dataset.csv      — same file used by the predictive model
  best_model               — the trained XGBoost object (imported or reloaded)

Outputs
───────
  cate_results.csv         — per-patient CATE scores + subgroup labels
  cate_subgroup_summary.csv — mean CATE + CI per BERTopic subgroup
  cate_distribution.png    — CATE score distribution plot
  subgroup_cate.png        — subgroup-level CATE comparison plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# EconML — CausalForestDML is the recommended estimator for observational data
# pip install econml
from econml.dml import CausalForestDML
from econml.inference import BootstrapInference

# BERTopic for subgroup discovery from EHR notes
# pip install bertopic umap-learn hdbscan
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

import joblib

# ── 0. Reload the dataset (same as predictive model) ─────────────────────────
print("Loading dataset...")
spine_raw = pd.read_csv("0_final_dataset.csv")
spine_raw = spine_raw.iloc[:, :-53]

# ── 1. Extract what we need BEFORE any drops ──────────────────────────────────
# Keep hadm_id as index so we can link everything back
# Keep hospital_expire_flag as our outcome Y
# Keep note_text for BERTopic

hadm_ids   = spine_raw['hadm_id'].values                      # patient index
Y          = spine_raw['hospital_expire_flag'].astype(int)     # outcome: in-hospital death
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_note(text):
    if not isinstance(text, str):
        return ""
    # Remove MIMIC de-identification tokens
    text = re.sub(r'\[\*\*.*?\*\*\]', ' ', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

note_texts = spine_raw['note_text'].fillna("").apply(clean_note).tolist()


print(f"Outcome (Y) distribution:")
print(f"  In-hospital death : {Y.sum()} ({Y.mean()*100:.1f}%)")
print(f"  Survived          : {(Y==0).sum()} ({(1-Y.mean())*100:.1f}%)")

# ── 2. Reconstruct feature matrix X and treatment T ───────────────────────────
# Mirrors the exact preprocessing from the predictive model so indices align

COLS_TO_DROP = [
    'subject_id', 'hadm_id', 'admittime', 'note_text', 'diagnostic_category',
    'dischtime', 'deathtime', 'admission_type', 'admit_provider_id',
    'admission_location', 'discharge_location', 'language',
    'edregtime', 'edouttime', 'anchor_year_group', 'dod',
    'hospital_expire_flag', 'note_text', 'diagnostic_category'
]

spine = spine_raw.copy()
cols_to_drop_present = [c for c in COLS_TO_DROP if c in spine.columns]
spine = spine.drop(columns=cols_to_drop_present)

T = spine['icu_admitted'].astype(int).values          # treatment: ICU admission
X = spine.drop(columns=['icu_admitted'])

CATEGORICAL_COLS = ['gender', 'race', 'insurance', 'marital_status']
cat_cols_present = [c for c in CATEGORICAL_COLS if c in X.columns]
X = pd.get_dummies(X, columns=cat_cols_present, drop_first=False)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

feature_names = X.columns.tolist()
print(f"\nFeature matrix: {X.shape[0]} patients, {X.shape[1]} features")
print(f"Treatment (T=ICU): {T.sum()} admitted ({T.mean()*100:.1f}%)")

# ── 3. Load propensity scores from the predictive model ───────────────────────
# Option A: reload the saved model and re-score (recommended)
# Option B: if your group member exports probabilities to CSV, load that instead

print("\nLoading predictive model for propensity scores...")
try:
    best_model = joblib.load("best_model.pkl")          # ask group member to save with joblib.dump()
    propensity_scores = best_model.predict_proba(X)[:, 1]   # e(X) = P(T=1 | X)
    print(f"  Propensity score range: {propensity_scores.min():.3f} – {propensity_scores.max():.3f}")
except FileNotFoundError:
    # Fallback: if they haven't saved the model yet, fit a quick logistic regression
    print("  best_model.pkl not found — fitting fallback propensity model...")
    from sklearn.linear_model import LogisticRegression
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, T)
    propensity_scores = ps_model.predict_proba(X)[:, 1]
    print("  Warning: using fallback propensity model. Ask group member to save best_model.pkl")

# ── 4. Positivity check ───────────────────────────────────────────────────────
# Trim patients where propensity ≈ 0 or ≈ 1 — no valid counterfactual exists
TRIM_LOWER = 0.05
TRIM_UPPER = 0.95

positivity_mask = (propensity_scores > TRIM_LOWER) & (propensity_scores < TRIM_UPPER)
n_trimmed = (~positivity_mask).sum()
print(f"\nPositivity trimming [{TRIM_LOWER}, {TRIM_UPPER}]:")
print(f"  Removed : {n_trimmed} patients ({n_trimmed/len(T)*100:.1f}%)")
print(f"  Retained: {positivity_mask.sum()} patients")

X_trim   = X[positivity_mask].values
T_trim   = T[positivity_mask]
Y_trim   = Y[positivity_mask].values
ps_trim  = propensity_scores[positivity_mask]
ids_trim = hadm_ids[positivity_mask]
notes_trim = [note_texts[i] for i in np.where(positivity_mask)[0]]

# ── 5. BERTopic subgroup discovery from EHR notes ─────────────────────────────
print("\nRunning BERTopic on EHR notes...")
print("  Embedding notes with ClinicalBERT (this may take a few minutes)...")

# Use ClinicalBERT for clinical domain embeddings
# Alternative: 'emilyalsentzer/Bio_ClinicalBERT' if you have HuggingFace access
embedding_model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")

# BERTopic handles UMAP dim reduction + HDBSCAN clustering internally
topic_model = BERTopic(
    embedding_model    = embedding_model,
    nr_topics          = "auto",           # let BERTopic decide; or set e.g. 10
    min_topic_size     = 30,               # minimum patients per topic
    calculate_probabilities = False,
    verbose            = True
)

topics, _ = topic_model.fit_transform(notes_trim)
topics = np.array(topics)

# Topic -1 = outliers in BERTopic — assign to a dedicated "other" label
topic_labels = np.array([
    f"topic_{t}" if t >= 0 else "other"
    for t in topics
])

# Print topic summaries
print("\nDiscovered topics:")
topic_info = topic_model.get_topic_info()
print(topic_info[['Topic', 'Count', 'Name']].to_string(index=False))

# ── 6. Build enriched covariate matrix for CATE ───────────────────────────────
# X_cate = [static features | propensity score | topic dummies]
topic_dummies = pd.get_dummies(topic_labels, prefix='topic')
X_cate = np.hstack([
    X_trim,
    ps_trim.reshape(-1, 1),
    topic_dummies.values
])

print(f"\nCATE covariate matrix: {X_cate.shape}")

# ── 7. CausalForestDML ────────────────────────────────────────────────────────
# CausalForestDML is doubly robust — it jointly models E[Y|X] and E[T|X]
# so it's more robust to misspecification than T-Learner or X-Learner
print("\nFitting CausalForestDML...")
print("  T = ICU admission, Y = in-hospital mortality")

from sklearn.ensemble import GradientBoostingRegressor

causal_forest = CausalForestDML(
    model_y             = GradientBoostingRegressor(n_estimators=200, random_state=42),
    model_t             = GradientBoostingRegressor(n_estimators=200, random_state=42),
    n_estimators        = 1000,
    min_samples_leaf    = 10,
    max_depth           = None,
    random_state        = 42,
    verbose             = 0,
    inference           = True        # enables confidence intervals
)

causal_forest.fit(
    Y    = Y_trim,
    T    = T_trim,
    X    = X_cate,
    cache_values = True
)

print("  CausalForest fit complete.")

# ── 8. Estimate CATE scores ───────────────────────────────────────────────────
print("\nEstimating CATE scores...")
tau_hat = causal_forest.effect(X_cate)                        # point estimates τ̂(x)
tau_ci  = causal_forest.effect_interval(X_cate, alpha=0.05)   # 95% CI (lower, upper)

print(f"  Mean CATE  : {tau_hat.mean():.4f}")
print(f"  Std CATE   : {tau_hat.std():.4f}")
print(f"  CATE range : {tau_hat.min():.4f} to {tau_hat.max():.4f}")
print(f"  % patients with τ̂ < 0 (ICU reduces mortality): {(tau_hat < 0).mean()*100:.1f}%")
print(f"  % patients with τ̂ > 0 (ICU increases mortality): {(tau_hat > 0).mean()*100:.1f}%")

# ── 9. Per-patient results dataframe ─────────────────────────────────────────
results_df = pd.DataFrame({
    'hadm_id'         : ids_trim,
    'T_icu'           : T_trim,
    'Y_mortality'     : Y_trim,
    'propensity_score': ps_trim,
    'topic_label'     : topic_labels,
    'cate'            : tau_hat,
    'cate_ci_lower'   : tau_ci[0],
    'cate_ci_upper'   : tau_ci[1],
})

results_df.to_csv("cate_results.csv", index=False)
print(f"\nPer-patient results saved to cate_results.csv ({len(results_df)} rows)")

# ── 10. Subgroup CATE summary ─────────────────────────────────────────────────
print("\nSubgroup CATE summary:")
subgroup_summary = (
    results_df.groupby('topic_label')
    .agg(
        n_patients      = ('cate', 'count'),
        mean_cate       = ('cate', 'mean'),
        std_cate        = ('cate', 'std'),
        ci_lower        = ('cate_ci_lower', 'mean'),
        ci_upper        = ('cate_ci_upper', 'mean'),
        pct_icu         = ('T_icu', 'mean'),
        mortality_rate  = ('Y_mortality', 'mean'),
    )
    .reset_index()
    .sort_values('mean_cate')
)

# Add topic keywords from BERTopic for interpretability
topic_keywords = {}
for row in topic_model.get_topic_info().itertuples():
    if row.Topic >= 0:
        words = [w for w, _ in topic_model.get_topic(row.Topic)[:5]]
        topic_keywords[f"topic_{row.Topic}"] = ", ".join(words)
topic_keywords['other'] = "outliers / mixed"

subgroup_summary['top_keywords'] = subgroup_summary['topic_label'].map(
    lambda x: topic_keywords.get(x, "")
)

print(subgroup_summary.to_string(index=False))
subgroup_summary.to_csv("cate_subgroup_summary.csv", index=False)
print("\nSubgroup summary saved to cate_subgroup_summary.csv")

# ── 11. Plots ─────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── 11a. CATE distribution ────────────────────────────────────────────────────
ax = axes[0]
ax.hist(tau_hat, bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
ax.axvline(0, color='crimson', linestyle='--', linewidth=1.2, label='No effect (τ=0)')
ax.axvline(tau_hat.mean(), color='darkorange', linestyle='-',
           linewidth=1.2, label=f'Mean τ̂ = {tau_hat.mean():.3f}')
ax.set_xlabel("Estimated treatment effect τ̂(x)")
ax.set_ylabel("Number of patients")
ax.set_title("CATE distribution — ICU admission effect on mortality")
ax.legend()
ax.grid(alpha=0.3)

# ── 11b. Subgroup CATE comparison ─────────────────────────────────────────────
ax = axes[1]
sub = subgroup_summary[subgroup_summary['topic_label'] != 'other'].copy()
colors = ['#2ecc71' if v < 0 else '#e74c3c' for v in sub['mean_cate']]
bars = ax.barh(sub['topic_label'], sub['mean_cate'], color=colors, alpha=0.8)

# Error bars using mean CI width
ci_width = (sub['ci_upper'] - sub['ci_lower']) / 2
ax.errorbar(sub['mean_cate'], sub['topic_label'],
            xerr=ci_width.values, fmt='none',
            color='black', capsize=4, linewidth=1)

ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel("Mean CATE τ̂(x)  [negative = ICU reduces mortality]")
ax.set_title("CATE by BERTopic subgroup")
ax.grid(axis='x', alpha=0.3)

# Annotate with n and keywords
for i, row in sub.reset_index(drop=True).iterrows():
    ax.text(sub['mean_cate'].max() * 1.05, i,
            f"n={int(row['n_patients'])}  [{row['top_keywords'][:30]}]",
            va='center', fontsize=7, color='#444')

plt.tight_layout()
plt.savefig("subgroup_cate.png", dpi=150, bbox_inches='tight')
plt.close()

# ── 11c. CATE distribution per subgroup (violin) ──────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
subgroup_order = subgroup_summary['topic_label'].tolist()
data_by_group  = [
    results_df[results_df['topic_label'] == g]['cate'].values
    for g in subgroup_order
]
vp = ax.violinplot(data_by_group, vert=False, showmedians=True)
ax.set_yticks(range(1, len(subgroup_order) + 1))
ax.set_yticklabels(subgroup_order)
ax.axvline(0, color='crimson', linestyle='--', linewidth=1)
ax.set_xlabel("CATE τ̂(x)")
ax.set_title("CATE distribution by subgroup")
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("cate_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n── Complete. Output files: ──")
print("  cate_results.csv           (per-patient CATE scores + subgroup labels)")
print("  cate_subgroup_summary.csv  (mean CATE + CI + keywords per subgroup)")
print("  subgroup_cate.png          (subgroup comparison bar chart)")
print("  cate_distribution.png      (violin plots per subgroup)")
print("\nNext step: feed cate_subgroup_summary.csv → LLM reasoning layer")