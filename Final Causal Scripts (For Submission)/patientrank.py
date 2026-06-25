"""
patient_ranking.py
==================
Ranks patients by predicted ICU benefit using ITE scores
from the best T-Learner model (XGBoost).

ITE < 0  → ICU REDUCES mortality → patient BENEFITS
ITE > 0  → ICU INCREASES mortality → patient does NOT benefit
ITE = 0  → no effect

Outputs:
  patient_ranking.csv              — all test patients ranked by ITE
  benefit_groups_summary.csv       — summary stats by benefit group
  ranking_ite_distribution.png     — ITE distribution with benefit zones
  ranking_top_features.png         — feature comparison across benefit groups
  ranking_age_distribution.png     — age distribution by benefit group
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA + ITE SCORES
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PATIENT BENEFIT RANKING")
print("=" * 60)

print("\n[1] Loading data and ITE scores...")
spine = pd.read_csv("0_final_dataset.csv")
spine = spine.iloc[:, :-53]

# Load ITE scores from best model (XGBoost T-Learner)
ite_scores = np.load("causal_output_trimmed/hospital_expire_flag/T-Learner (GradientBoost)_ITE.npy")
print(f"  ITE scores loaded: {len(ite_scores)} patients")

# ── Recreate the exact same split used in t_learner_v3.py ────────────────────
Y = spine['hospital_expire_flag'].astype(float)
w = spine['icu_admitted'].astype(float)

COLS_TO_DROP = [
    'subject_id', 'hadm_id', 'admittime', 'note_text', 'diagnostic_category',
    'dischtime', 'deathtime', 'admission_type', 'admit_provider_id',
    'admission_location', 'discharge_location', 'language',
    'edregtime', 'edouttime', 'anchor_year_group', 'dod',
    'hospital_expire_flag', 'icu_admitted', 'diagnostic_category'
]

# Keep clinical columns for interpretation
KEEP_COLS = [
    'anchor_age', 'gender', 'race', 'insurance', 'marital_status',
    'hospital_expire_flag', 'icu_admitted'
]
keep_present = [c for c in KEEP_COLS if c in spine.columns]
clinical_df  = spine[keep_present].copy()

cols_present = [c for c in COLS_TO_DROP if c in spine.columns]
X = spine.drop(columns=cols_present, errors='ignore')

CATEGORICAL_COLS = ['gender', 'race', 'insurance', 'marital_status']
cat_cols_present = [c for c in CATEGORICAL_COLS if c in X.columns]
X = pd.get_dummies(X, columns=cat_cols_present, drop_first=False)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X = X.drop(columns=X.select_dtypes(include='object').columns, errors='ignore')

# Recreate split — must match t_learner_v3.py exactly (test_size=0.40)
_, X_test, _, y_test, _, w_test, _, clin_test = train_test_split(
    X, Y, w, clinical_df,
    test_size=0.40, random_state=RANDOM_STATE, stratify=Y
)

X_test   = X_test.reset_index(drop=True)
y_test   = y_test.reset_index(drop=True)
w_test   = w_test.reset_index(drop=True)
clin_test= clin_test.reset_index(drop=True)

print(f"  Test set size : {len(X_test)}")
print(f"  ITE scores    : {len(ite_scores)}")

# Verify sizes match
if len(ite_scores) != len(X_test):
    print(f"  ⚠️  Size mismatch! ITE={len(ite_scores)}, Test={len(X_test)}")
    print(f"  Trimming to smaller size...")
    n = min(len(ite_scores), len(X_test))
    ite_scores = ite_scores[:n]
    X_test     = X_test.iloc[:n]
    y_test     = y_test.iloc[:n]
    w_test     = w_test.iloc[:n]
    clin_test  = clin_test.iloc[:n]

# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD RANKING DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Building patient ranking...")

ranking_df = clin_test.copy()
ranking_df['ite']              = ite_scores
ranking_df['actual_mortality'] = y_test.values
ranking_df['actual_icu']       = w_test.values

# Sort by ITE ascending — most negative = most benefit
ranking_df = ranking_df.sort_values('ite', ascending=True).reset_index(drop=True)
ranking_df['benefit_rank'] = ranking_df.index + 1  # 1 = most benefit

# ── Benefit groups ────────────────────────────────────────────────────────────
# Based on ITE quartiles
quartiles = pd.qcut(ranking_df['ite'], q=4,
                    labels=['High Benefit', 'Moderate Benefit',
                            'Low Benefit', 'Potential Harm'])
ranking_df['benefit_group'] = quartiles

# Also add a simple binary: benefits vs does not benefit
ranking_df['benefits_from_icu'] = (ranking_df['ite'] < 0).map(
    {True: 'Benefits (ITE < 0)', False: 'Does Not Benefit (ITE ≥ 0)'}
)

print(f"  ITE range: {ite_scores.min():.4f} to {ite_scores.max():.4f}")
print(f"  Mean ITE : {ite_scores.mean():.4f}")
print(f"  % patients who benefit (ITE < 0): {(ite_scores < 0).mean()*100:.1f}%")
print(f"\n  Benefit group distribution:")
print(ranking_df['benefit_group'].value_counts().to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 3. SUMMARY STATS BY BENEFIT GROUP
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Computing summary statistics by benefit group...")

summary_cols = ['ite', 'actual_mortality', 'actual_icu']
if 'anchor_age' in ranking_df.columns:
    summary_cols.append('anchor_age')

group_summary = ranking_df.groupby('benefit_group')[summary_cols].agg(
    ['mean', 'std', 'count']
).round(4)

print(group_summary.to_string())

# Simpler summary for CSV
simple_summary = ranking_df.groupby('benefit_group').agg(
    n_patients        = ('ite', 'count'),
    mean_ite          = ('ite', 'mean'),
    std_ite           = ('ite', 'std'),
    pct_actual_died   = ('actual_mortality', 'mean'),
    pct_actual_icu    = ('actual_icu', 'mean'),
    mean_age          = ('anchor_age', 'mean') if 'anchor_age' in ranking_df.columns else ('ite', 'count'),
).round(4)

simple_summary.to_csv("benefit_groups_summary.csv")
print("\n  Saved: benefit_groups_summary.csv")

# Top 10% most likely to benefit
top_10pct = ranking_df.head(int(len(ranking_df) * 0.10))
print(f"\n  Top 10% most likely to benefit ({len(top_10pct)} patients):")
print(f"    Mean ITE          : {top_10pct['ite'].mean():.4f}")
print(f"    Actual mortality  : {top_10pct['actual_mortality'].mean()*100:.1f}%")
print(f"    Actual ICU rate   : {top_10pct['actual_icu'].mean()*100:.1f}%")
if 'anchor_age' in top_10pct.columns:
    print(f"    Mean age          : {top_10pct['anchor_age'].mean():.1f}")

# Bottom 10% (most likely to be harmed)
bot_10pct = ranking_df.tail(int(len(ranking_df) * 0.10))
print(f"\n  Bottom 10% least likely to benefit ({len(bot_10pct)} patients):")
print(f"    Mean ITE          : {bot_10pct['ite'].mean():.4f}")
print(f"    Actual mortality  : {bot_10pct['actual_mortality'].mean()*100:.1f}%")
print(f"    Actual ICU rate   : {bot_10pct['actual_icu'].mean()*100:.1f}%")
if 'anchor_age' in bot_10pct.columns:
    print(f"    Mean age          : {bot_10pct['anchor_age'].mean():.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SAVE FULL RANKING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Saving full patient ranking...")
ranking_df.to_csv("patient_ranking_trimmed.csv", index=False)
print("  Saved: patient_ranking.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 5. PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Generating plots...")

GROUP_COLORS = {
    'High Benefit'     : '#2166ac',
    'Moderate Benefit' : '#74add1',
    'Low Benefit'      : '#fdae61',
    'Potential Harm'   : '#d73027',
}

# ── Plot 1: ITE distribution with benefit zones ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(ranking_df['ite'], bins=80, ax=ax,
             color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.2)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No effect (ITE=0)')
ax.axvline(ranking_df['ite'].mean(), color='black', linewidth=2,
           linestyle='-', label=f"Mean ITE = {ranking_df['ite'].mean():.4f}")

# Shade benefit zone
x_min = ranking_df['ite'].min()
ax.axvspan(x_min, 0, alpha=0.1, color='blue', label='Benefit zone (ITE < 0)')
ax.axvspan(0, ranking_df['ite'].max(), alpha=0.1, color='red', label='Harm zone (ITE > 0)')

ax.set_xlabel("Individual Treatment Effect (ITE)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Distribution of Predicted ICU Benefit\n(ITE < 0 = ICU reduces mortality)", fontsize=13)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ranking_ite_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: ranking_ite_distribution.png")

# ── Plot 2: Age distribution by benefit group (if age available) ──────────────
if 'anchor_age' in ranking_df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    for group, color in GROUP_COLORS.items():
        subset = ranking_df[ranking_df['benefit_group'] == group]['anchor_age'].dropna()
        if len(subset) > 0:
            sns.kdeplot(subset, ax=ax, label=group, color=color, linewidth=2.5)
    ax.axvline(ranking_df['anchor_age'].mean(), color='black',
               linestyle='--', linewidth=1.5, label='Overall mean age')
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Age Distribution by Benefit Group", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ranking_age_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ranking_age_distribution.png")

# ── Plot 3: Mortality and ICU rate by benefit group ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

groups     = list(GROUP_COLORS.keys())
colors     = list(GROUP_COLORS.values())
mort_rates = [ranking_df[ranking_df['benefit_group']==g]['actual_mortality'].mean()*100
              for g in groups]
icu_rates  = [ranking_df[ranking_df['benefit_group']==g]['actual_icu'].mean()*100
              for g in groups]

ax = axes[0]
bars = ax.bar(range(len(groups)), mort_rates, color=colors, alpha=0.85)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups, rotation=15, ha='right', fontsize=9)
ax.set_ylabel("Actual Mortality Rate (%)", fontsize=11)
ax.set_title("Actual Mortality by Benefit Group", fontsize=12)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
bars2 = ax.bar(range(len(groups)), icu_rates, color=colors, alpha=0.85)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups, rotation=15, ha='right', fontsize=9)
ax.set_ylabel("Actual ICU Rate (%)", fontsize=11)
ax.set_title("Actual ICU Admission Rate by Benefit Group", fontsize=12)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.suptitle("Clinical Characteristics by Predicted Benefit Group", fontsize=13)
plt.tight_layout()
plt.savefig("ranking_clinical_profile.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: ranking_clinical_profile.png")

# ── Plot 4: Mean ITE by benefit group (ranked bar) ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
mean_ites = [ranking_df[ranking_df['benefit_group']==g]['ite'].mean() for g in groups]
bars3 = ax.bar(range(len(groups)), mean_ites, color=colors, alpha=0.85)
ax.axhline(0, color='black', linestyle='--', linewidth=1.2)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups, rotation=15, ha='right', fontsize=9)
ax.set_ylabel("Mean ITE", fontsize=11)
ax.set_title("Mean ITE by Benefit Group\n(negative = ICU reduces mortality)", fontsize=12)
for bar in bars3:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            yval + (0.001 if yval >= 0 else -0.003),
            f'{yval:.4f}', ha='center', va='bottom', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("ranking_mean_ite_by_group.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: ranking_mean_ite_by_group.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RANKING SUMMARY")
print("=" * 60)
print(f"\n  Total patients ranked   : {len(ranking_df)}")
print(f"  Benefit (ITE < 0)       : {(ranking_df['ite'] < 0).sum()} "
      f"({(ranking_df['ite'] < 0).mean()*100:.1f}%)")
print(f"  No benefit (ITE ≥ 0)    : {(ranking_df['ite'] >= 0).sum()} "
      f"({(ranking_df['ite'] >= 0).mean()*100:.1f}%)")
print(f"\n  Group breakdown:")
for g in groups:
    subset = ranking_df[ranking_df['benefit_group'] == g]
    print(f"    {g:<20} : {len(subset):6d} patients  "
          f"mean ITE={subset['ite'].mean():.4f}  "
          f"mortality={subset['actual_mortality'].mean()*100:.1f}%")

print("\n── Complete. Output files ──────────────────────────────")
print("  patient_ranking.csv")
print("  benefit_groups_summary.csv")
print("  ranking_ite_distribution.png")
print("  ranking_age_distribution.png")
print("  ranking_clinical_profile.png")
print("  ranking_mean_ite_by_group.png")