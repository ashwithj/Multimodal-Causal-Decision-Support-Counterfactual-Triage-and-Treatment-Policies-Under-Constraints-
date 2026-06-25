"""
xgb_sensitivity.py
==================
Sensitivity analysis for XGBoost T-Learner hyperparameters.
Varies n_estimators, max_depth, learning_rate and reports
PEHE and ATE stability across 27 combinations.

Outputs:
  sensitivity_pehe_lr_depth.png     — heatmap: learning_rate x max_depth (PEHE)
  sensitivity_pehe_nest_depth.png   — heatmap: n_estimators x max_depth (PEHE)
  sensitivity_ate_lr_depth.png      — heatmap: learning_rate x max_depth (ATE)
  sensitivity_results.csv           — full 27-row results table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Hyperparameter grid ────────────────────────────────────────────────────────
N_ESTIMATORS  = [50, 100, 200]
MAX_DEPTHS    = [3, 5, 7]
LEARNING_RATES= [0.05, 0.1, 0.2]

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA  (same as t_learner_v3.py)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("XGBoost T-Learner Sensitivity Analysis")
print("=" * 60)

print("\n[1] Loading data...")
spine = pd.read_csv("0_final_dataset.csv")
spine = spine.iloc[:, :-53]

Y = spine['hospital_expire_flag'].astype(float)
w = spine['icu_admitted'].astype(float)

COLS_TO_DROP = [
    'subject_id', 'hadm_id', 'admittime', 'note_text', 'diagnostic_category',
    'dischtime', 'deathtime', 'admission_type', 'admit_provider_id',
    'admission_location', 'discharge_location', 'language',
    'edregtime', 'edouttime', 'anchor_year_group', 'dod',
    'hospital_expire_flag', 'icu_admitted', 'diagnostic_category'
]
cols_present = [c for c in COLS_TO_DROP if c in spine.columns]
X = spine.drop(columns=cols_present, errors='ignore')

CATEGORICAL_COLS = ['gender', 'race', 'insurance', 'marital_status']
cat_cols_present = [c for c in CATEGORICAL_COLS if c in X.columns]
X = pd.get_dummies(X, columns=cat_cols_present, drop_first=False)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X = X.drop(columns=X.select_dtypes(include='object').columns, errors='ignore')

# ── Overlap trimming (matches t_learner_v4.py) ────────────────────────────────
import joblib
prop_model  = joblib.load("best_model.pkl")
train_cols  = prop_model.get_booster().feature_names
X_aligned   = X.reindex(columns=train_cols, fill_value=0)
propensity  = prop_model.predict_proba(X_aligned)[:, 1]
OVERLAP_LO, OVERLAP_HI = 0.1, 0.9
overlap_mask = (propensity >= OVERLAP_LO) & (propensity <= OVERLAP_HI)
X = X[overlap_mask].reset_index(drop=True)
Y = Y[overlap_mask].reset_index(drop=True)
w = w[overlap_mask].reset_index(drop=True)
print(f"  Patients after overlap trimming: {overlap_mask.sum()} ({(~overlap_mask).sum()} removed)")
print(f"  Features : {X.shape[1]}")

# ── Train/test split (matches t_learner_v3.py — 60/40) ───────────────────────
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, Y, w,
    test_size=0.40, random_state=RANDOM_STATE, stratify=Y
)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
w_train = w_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)
w_test  = w_test.reset_index(drop=True)

X_train_np = X_train.to_numpy(dtype=np.float32)
y_train_np = y_train.to_numpy(dtype=np.float32)
w_train_np = w_train.to_numpy(dtype=np.float32)
X_test_np  = X_test.to_numpy(dtype=np.float32)
y_test_np  = y_test.to_numpy(dtype=np.float32)
w_test_np  = w_test.to_numpy(dtype=np.float32)

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. NN COUNTERFACTUALS  (computed once, reused across all 27 combinations)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Computing NN counterfactuals (done once)...")

def build_nn_counterfactuals(X_all, w_all, y_all, k=5):
    n = len(y_all)
    ite_nn = np.zeros(n)
    treated_idx = np.where(w_all == 1)[0]
    control_idx = np.where(w_all == 0)[0]
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return np.full(n, np.nan)
    X_treated, y_treated = X_all[treated_idx], y_all[treated_idx]
    X_control, y_control = X_all[control_idx], y_all[control_idx]
    nn_treated = NearestNeighbors(n_neighbors=min(k, len(X_treated))).fit(X_treated)
    nn_control = NearestNeighbors(n_neighbors=min(k, len(X_control))).fit(X_control)
    for i in range(n):
        x_i = X_all[i].reshape(1, -1)
        if w_all[i] == 1:
            _, idx = nn_control.kneighbors(x_i)
            ite_nn[i] = y_all[i] - np.mean(y_control[idx[0]])
        else:
            _, idx = nn_treated.kneighbors(x_i)
            ite_nn[i] = np.mean(y_treated[idx[0]]) - y_all[i]
    return ite_nn

X_all_np = np.vstack([X_train_np, X_test_np])
w_all_np = np.concatenate([w_train_np, w_test_np])
y_all_np = np.concatenate([y_train_np, y_test_np])

t_nn = time.time()
ite_nn_all  = build_nn_counterfactuals(X_all_np, w_all_np, y_all_np, k=5)
ite_nn_test = ite_nn_all[len(X_train_np):]
print(f"  Done in {time.time()-t_nn:.1f}s  |  NN ATE: {ite_nn_test.mean()*100:+.2f} pp")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SENSITIVITY GRID
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[3] Running {len(N_ESTIMATORS)*len(MAX_DEPTHS)*len(LEARNING_RATES)} combinations...")

def compute_class_weights(y):
    y = np.asarray(y)
    neg, pos = np.sum(y == 0), np.sum(y == 1)
    total = len(y)
    return (total / (2 * neg) if neg > 0 else 1.0,
            total / (2 * pos) if pos > 0 else 1.0)

def run_xgb_t_learner(X_tr, w_tr, y_tr, X_te, w_te, n_est, depth, lr):
    """Train T-learner with given hyperparameters, return PEHE and ATE."""
    # Split by treatment arm
    ctrl_mask  = w_tr == 0
    treat_mask = w_tr == 1

    X1 = X_tr[treat_mask]; y1 = y_tr[treat_mask]
    X0 = X_tr[ctrl_mask];  y0 = y_tr[ctrl_mask]

    _, pw1 = compute_class_weights(y1)
    _, pw0 = compute_class_weights(y0)

    mu1 = xgb.XGBClassifier(
        n_estimators=n_est, max_depth=depth, learning_rate=lr,
        objective='binary:logistic', eval_metric='auc',
        scale_pos_weight=pw1, random_state=RANDOM_STATE, n_jobs=-1
    )
    mu0 = xgb.XGBClassifier(
        n_estimators=n_est, max_depth=depth, learning_rate=lr,
        objective='binary:logistic', eval_metric='auc',
        scale_pos_weight=pw0, random_state=RANDOM_STATE, n_jobs=-1
    )

    mu1.fit(X1, y1)
    mu0.fit(X0, y0)

    y_cf1 = mu1.predict_proba(X_te)[:, 1]
    y_cf0 = mu0.predict_proba(X_te)[:, 1]
    ite   = y_cf1 - y_cf0

    ate  = float(ite.mean())
    pehe = float(np.sqrt(np.mean((ite - ite_nn_test) ** 2)))

    return ate, pehe

results = []
total = len(N_ESTIMATORS) * len(MAX_DEPTHS) * len(LEARNING_RATES)
count = 0

for n_est in N_ESTIMATORS:
    for depth in MAX_DEPTHS:
        for lr in LEARNING_RATES:
            count += 1
            t0 = time.time()
            ate, pehe = run_xgb_t_learner(
                X_train_np, w_train_np, y_train_np,
                X_test_np,  w_test_np,
                n_est, depth, lr
            )
            elapsed = time.time() - t0
            print(f"  [{count:2d}/{total}] n_est={n_est:3d} depth={depth} lr={lr:.2f} "
                  f"→ PEHE={pehe:.4f}  ATE={ate*100:+.2f}pp  ({elapsed:.1f}s)")
            results.append({
                'n_estimators' : n_est,
                'max_depth'    : depth,
                'learning_rate': lr,
                'pehe'         : pehe,
                'ate_pp'       : ate * 100,
            })

results_df = pd.DataFrame(results)
results_df.to_csv("sensitivity_trimmed_results.csv", index=False)
print("\n  Saved: sensitivity_trimmed_results.csv")
print(results_df.sort_values('pehe').round(4).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 4. HEATMAP PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Generating heatmaps...")

def make_heatmap(pivot_df, title, xlabel, ylabel, fmt, cmap, save_path, annot_fmt='.4f'):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot_df, annot=True, fmt=annot_fmt, cmap=cmap,
        linewidths=0.5, linecolor='white', ax=ax,
        cbar_kws={'label': title.split('\n')[0]}
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

# ── Heatmap 1: learning_rate × max_depth → PEHE (averaged over n_estimators) ─
pivot1 = results_df.groupby(['learning_rate', 'max_depth'])['pehe'].mean().unstack()
make_heatmap(
    pivot1,
    title='PEHE: learning_rate × max_depth\n(averaged over n_estimators)',
    xlabel='max_depth', ylabel='learning_rate',
    fmt='.4f', cmap='YlOrRd_r',
    save_path='sensitivity_trimmed_pehe_lr_depth.png'
)

# ── Heatmap 2: n_estimators × max_depth → PEHE (averaged over learning_rate) ─
pivot2 = results_df.groupby(['n_estimators', 'max_depth'])['pehe'].mean().unstack()
make_heatmap(
    pivot2,
    title='PEHE: n_estimators × max_depth\n(averaged over learning_rate)',
    xlabel='max_depth', ylabel='n_estimators',
    fmt='.4f', cmap='YlOrRd_r',
    save_path='sensitivity_trimmed_pehe_nest_depth.png'
)

# ── Heatmap 3: learning_rate × max_depth → ATE ────────────────────────────────
pivot3 = results_df.groupby(['learning_rate', 'max_depth'])['ate_pp'].mean().unstack()
make_heatmap(
    pivot3,
    title='ATE (pp): learning_rate × max_depth\n(averaged over n_estimators)',
    xlabel='max_depth', ylabel='learning_rate',
    fmt='.2f', cmap='RdBu',
    save_path='sensitivity_trimmed_ate_lr_depth.png',
    annot_fmt='.2f'
)

# ── Heatmap 4: n_estimators × learning_rate → PEHE ───────────────────────────
pivot4 = results_df.groupby(['n_estimators', 'learning_rate'])['pehe'].mean().unstack()
make_heatmap(
    pivot4,
    title='PEHE: n_estimators × learning_rate\n(averaged over max_depth)',
    xlabel='learning_rate', ylabel='n_estimators',
    fmt='.4f', cmap='YlOrRd_r',
    save_path='sensitivity_trimmed_pehe_nest_lr.png'
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
best = results_df.loc[results_df['pehe'].idxmin()]
worst = results_df.loc[results_df['pehe'].idxmax()]
pehe_range = results_df['pehe'].max() - results_df['pehe'].min()
ate_range  = results_df['ate_pp'].max() - results_df['ate_pp'].min()

print("\n" + "=" * 60)
print("SENSITIVITY SUMMARY")
print("=" * 60)
print(f"\n  Best config  (lowest PEHE):")
print(f"    n_estimators={int(best.n_estimators)}  max_depth={int(best.max_depth)}  "
      f"learning_rate={best.learning_rate}")
print(f"    PEHE={best.pehe:.4f}  ATE={best.ate_pp:+.2f}pp")
print(f"\n  Worst config (highest PEHE):")
print(f"    n_estimators={int(worst.n_estimators)}  max_depth={int(worst.max_depth)}  "
      f"learning_rate={worst.learning_rate}")
print(f"    PEHE={worst.pehe:.4f}  ATE={worst.ate_pp:+.2f}pp")
print(f"\n  PEHE range across all configs : {pehe_range:.4f}")
print(f"  ATE  range across all configs : {ate_range:.2f} pp")
print(f"\n  → If PEHE range is small (<0.01), results are ROBUST")
print(f"  → If PEHE range is large (>0.05), results are SENSITIVE to tuning")

print("\n── Complete. Output files ──────────────────────────────")
print("  sensitivity_trimmed_results.csv")
print("  sensitivity_pehe_lr_depth.png")
print("  sensitivity_pehe_nest_depth.png")
print("  sensitivity_pehe_nest_lr.png")
print("  sensitivity_ate_lr_depth.png")