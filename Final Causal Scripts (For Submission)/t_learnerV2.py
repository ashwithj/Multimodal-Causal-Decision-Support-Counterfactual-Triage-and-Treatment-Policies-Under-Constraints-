# ── CHANGE 1: Data loading (replaces professor's merge logic) ─────────────────
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load your dataset ──────────────────────────────────────────────────────────
print("Loading dataset...")
spine = pd.read_csv("0_final_dataset.csv")
spine = spine.iloc[:, :-53]

# ── CHANGE 2: Column names ─────────────────────────────────────────────────────
STRATIFICATION_COL = "icu_admitted"          # was "IONM_changes"
outcome_cols       = ['hospital_expire_flag'] # was his list of surgical outcomes

# ── CHANGE 3: Propensity model (replaces his TF .keras loader) ────────────────
propensity_model = joblib.load("best_model.pkl")

# ── CHANGE 4: Save paths (replaces /lakehouse/... paths) ─────────────────────
base_save_dir = "causal_output"
ite_save_dir  = "causal_output/ITE"
os.makedirs(base_save_dir, exist_ok=True)
os.makedirs(ite_save_dir,  exist_ok=True)

# ── Prepare features ───────────────────────────────────────────────────────────
COLS_TO_DROP = [
    'subject_id', 'hadm_id', 'admittime', 'note_text', 'diagnostic_category',
    'dischtime', 'deathtime', 'admission_type', 'admit_provider_id',
    'admission_location', 'discharge_location', 'language',
    'edregtime', 'edouttime', 'anchor_year_group', 'dod',
    'hospital_expire_flag', 'icu_admitted', 'diagnostic_category'
]
cols_present = [c for c in COLS_TO_DROP if c in spine.columns]
features_df  = spine.drop(columns=cols_present, errors='ignore')

CATEGORICAL_COLS = ['gender', 'race', 'insurance', 'marital_status']
cat_cols_present = [c for c in CATEGORICAL_COLS if c in features_df.columns]
features_df = pd.get_dummies(features_df, columns=cat_cols_present, drop_first=False)
features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
features_df = features_df.drop(
    columns=features_df.select_dtypes(include='object').columns, errors='ignore'
)

X = features_df
w = spine[STRATIFICATION_COL].astype(int)

print(f"Loaded: {len(X)} patients, {X.shape[1]} features")
print(f"Treatment ({STRATIFICATION_COL}): {w.mean()*100:.1f}% positive")


# ══════════════════════════════════════════════════════════════════════════════
# PROFESSOR'S CODE BELOW — UNCHANGED EXCEPT display() → print() (Change 5)
# ══════════════════════════════════════════════════════════════════════════════

# ── build_nn_counterfactuals (Cell 9) ─────────────────────────────────────────
def build_nn_counterfactuals(X_all, w_all, y_all, k=5):
    """
    Estimate individual treatment effects using k-nearest neighbor counterfactual matching.
    """
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
            matched_control_outcome = np.mean(y_control[idx[0]])
            ite_nn[i] = y_all[i] - matched_control_outcome
        else:
            _, idx = nn_treated.kneighbors(x_i)
            matched_treated_outcome = np.mean(y_treated[idx[0]])
            ite_nn[i] = matched_treated_outcome - y_all[i]

    return ite_nn


# ── Utility functions (Cell 21) ───────────────────────────────────────────────
def compute_class_weights(y):
    y = np.asarray(y)
    neg = np.sum(y == 0)
    pos = np.sum(y == 1)
    total = len(y)
    weight_for_0 = total / (2 * neg) if neg > 0 else 1.0
    weight_for_1 = total / (2 * pos) if pos > 0 else 1.0
    return {0: weight_for_0, 1: weight_for_1}, weight_for_1 / weight_for_0


def prepare_t_learner_data(X_train, w_train, y_train, oversample=False):
    """Splits and prepares data for the two T-Learner models (mu_1 and mu_0)."""
    train_df = pd.DataFrame(X_train)
    train_df['w'] = w_train
    train_df['y'] = y_train

    df1 = train_df[train_df['w'] == 1].drop(columns=['w'])
    df0 = train_df[train_df['w'] == 0].drop(columns=['w'])

    X1_train = df1.drop(columns=['y']).to_numpy(dtype=np.float32)
    y1_train = df1['y'].to_numpy(dtype=np.float32)
    X0_train = df0.drop(columns=['y']).to_numpy(dtype=np.float32)
    y0_train = df0['y'].to_numpy(dtype=np.float32)

    if np.any(np.isnan(X1_train)):
        col_means1 = np.nanmean(X1_train, axis=0)
        X1_train[np.where(np.isnan(X1_train))] = np.take(col_means1, np.where(np.isnan(X1_train))[1])
    if np.any(np.isnan(X0_train)):
        col_means0 = np.nanmean(X0_train, axis=0)
        X0_train[np.where(np.isnan(X0_train))] = np.take(col_means0, np.where(np.isnan(X0_train))[1])

    return X1_train, y1_train, X0_train, y0_train


def calculate_t_learner_metrics(model1, model0, X_test, w_test, y_test):
    """Calculates ITE and performance metrics using two trained models."""

    y_cf1_prob = model1.predict_proba(X_test)[:, 1]
    y_cf0_prob = model0.predict_proba(X_test)[:, 1]

    ite = y_cf1_prob - y_cf0_prob

    y_prob_observed = np.where(w_test == 1, y_cf1_prob, y_cf0_prob)
    y_pred_observed = (y_prob_observed >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_observed, average="binary", zero_division=0
    )

    if len(np.unique(y_test)) < 2:
        auroc = np.nan
    else:
        auroc = roc_auc_score(y_test, y_prob_observed)

    treated_mask = (w_test == 1)
    control_mask = (w_test == 0)

    if np.sum(treated_mask) > 1 and len(np.unique(y_test[treated_mask])) > 1:
        treated_auroc = roc_auc_score(y_test[treated_mask], y_cf1_prob[treated_mask])
    else:
        treated_auroc = np.nan

    if np.sum(control_mask) > 1 and len(np.unique(y_test[control_mask])) > 1:
        control_auroc = roc_auc_score(y_test[control_mask], y_cf0_prob[control_mask])
    else:
        control_auroc = np.nan

    ate_true_nn = np.mean(ite_nn_test)
    ate_pred    = np.mean(ite)
    ate_error   = np.abs(ate_true_nn - ate_pred)
    pehe        = np.sqrt(np.mean((ite - ite_nn_test) ** 2))

    y1_pred = (y_cf1_prob[treated_mask] >= 0.5).astype(int)
    y0_pred = (y_cf0_prob[control_mask] >= 0.5).astype(int)

    metrics = {
        "Accuracy":         float(accuracy_score(y_test, y_pred_observed)),
        "Precision":        float(precision),
        "Recall":           float(recall),
        "F1 Score":         float(f1),
        "AUROC":            float(auroc),
        "Treated_AUROC":    float(treated_auroc),
        "Control_AUROC":    float(control_auroc),
        "ATE Error":        float(ate_error),
        "PEHE":             pehe,
        "ITE":              ite,
        "Treated_Accuracy": float(accuracy_score(y_test[treated_mask], y1_pred))
                            if np.sum(treated_mask) > 0 else np.nan,
        "Control_Accuracy": float(accuracy_score(y_test[control_mask], y0_pred))
                            if np.sum(control_mask) > 0 else np.nan,
    }

    return (model1, model0), metrics


# ── T-Learner model functions (Cells 22-25) ───────────────────────────────────
def run_t_learner_lr(X_train, w_train, y_train, X_test, w_test, y_test,
                     C=1.0, solver='liblinear', oversample=True):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test  = np.asarray(X_test,  dtype=np.float32)
    w_test  = np.asarray(w_test).reshape(-1)
    y_test  = np.asarray(y_test).reshape(-1)
    X1_train, y1_train, X0_train, y0_train = prepare_t_learner_data(X_train, w_train, y_train, oversample=oversample)

    model1 = Pipeline([('scaler', StandardScaler()),
                       ('logreg', LogisticRegression(C=C, solver=solver, random_state=42,
                                                     class_weight='balanced', max_iter=1000))])
    model1.fit(X1_train, y1_train)

    model0 = Pipeline([('scaler', StandardScaler()),
                       ('logreg', LogisticRegression(C=C, solver=solver, random_state=42,
                                                     class_weight='balanced', max_iter=1000))])
    model0.fit(X0_train, y0_train)

    return calculate_t_learner_metrics(model1, model0, X_test, w_test, y_test)


def run_t_learner_dt(X_train, w_train, y_train, X_test, w_test, y_test, max_depth=5, oversample=True):
    X_train, X_test, w_test, y_test = map(np.asarray, [X_train, X_test, w_test, y_test])
    X1_train, y1_train, X0_train, y0_train = prepare_t_learner_data(X_train, w_train, y_train, oversample=oversample)
    model1 = DecisionTreeClassifier(max_depth=max_depth, random_state=42, class_weight='balanced')
    model1.fit(X1_train, y1_train)
    model0 = DecisionTreeClassifier(max_depth=max_depth, random_state=42, class_weight='balanced')
    model0.fit(X0_train, y0_train)
    return calculate_t_learner_metrics(model1, model0, X_test, w_test, y_test)


def run_t_learner_rf(X_train, w_train, y_train, X_test, w_test, y_test, n_estimators=100, oversample=True):
    X_train, X_test, w_test, y_test = map(np.asarray, [X_train, X_test, w_test, y_test])
    X1_train, y1_train, X0_train, y0_train = prepare_t_learner_data(X_train, w_train, y_train, oversample=oversample)
    model1 = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced', n_jobs=-1)
    model1.fit(X1_train, y1_train)
    model0 = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced', n_jobs=-1)
    model0.fit(X0_train, y0_train)
    return calculate_t_learner_metrics(model1, model0, X_test, w_test, y_test)


def run_t_learner_gb(X_train, w_train, y_train, X_test, w_test, y_test, n_estimators=100, learning_rate=0.1, oversample=False):
    X_train, X_test, w_test, y_test = map(np.asarray, [X_train, X_test, w_test, y_test])
    X1_train, y1_train, X0_train, y0_train = prepare_t_learner_data(X_train, w_train, y_train, oversample=oversample)
    class_weights1, _ = compute_class_weights(y1_train)
    sample_weights1 = np.array([class_weights1[0] if y == 0 else class_weights1[1] for y in y1_train])
    class_weights0, _ = compute_class_weights(y0_train)
    sample_weights0 = np.array([class_weights0[0] if y == 0 else class_weights0[1] for y in y0_train])
    model1 = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    model1.fit(X1_train, y1_train, sample_weight=sample_weights1)
    model0 = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    model0.fit(X0_train, y0_train, sample_weight=sample_weights0)
    return calculate_t_learner_metrics(model1, model0, X_test, w_test, y_test)


def run_t_learner_svc(X_train, w_train, y_train, X_test, w_test, y_test, C=1.0, oversample=True):
    X_train, X_test, w_test, y_test = map(np.asarray, [X_train, X_test, w_test, y_test])
    X1_train, y1_train, X0_train, y0_train = prepare_t_learner_data(X_train, w_train, y_train, oversample=oversample)
    model1 = Pipeline([('scaler', StandardScaler()), ('svc', SVC(C=C, probability=True, random_state=42, class_weight='balanced'))])
    model1.fit(X1_train, y1_train)
    model0 = Pipeline([('scaler', StandardScaler()), ('svc', SVC(C=C, probability=True, random_state=42, class_weight='balanced'))])
    model0.fit(X0_train, y0_train)
    return calculate_t_learner_metrics(model1, model0, X_test, w_test, y_test)


def run_t_learner_knn(X_train, w_train, y_train, X_test, w_test, y_test, n_neighbors=5, oversample=True):
    X_train, X_test, w_test, y_test = map(np.asarray, [X_train, X_test, w_test, y_test])
    X1_train, y1_train, X0_train, y0_train = prepare_t_learner_data(X_train, w_train, y_train, oversample=oversample)
    model1 = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1))])
    model1.fit(X1_train, y1_train)
    model0 = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1))])
    model0.fit(X0_train, y0_train)
    return calculate_t_learner_metrics(model1, model0, X_test, w_test, y_test)


def run_t_learner_xgb(X_train, w_train, y_train, X_test, w_test, y_test, oversample=False):
    X_train, X_test, w_test, y_test = map(np.asarray, [X_train, X_test, w_test, y_test])
    X1_train, y1_train, X0_train, y0_train = prepare_t_learner_data(X_train, w_train, y_train, oversample=oversample)
    _, pos_weight1 = compute_class_weights(y1_train)
    _, pos_weight0 = compute_class_weights(y0_train)
    model1 = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc',
                                scale_pos_weight=pos_weight1, random_state=42, n_jobs=-1)
    model1.fit(X1_train, y1_train)
    model0 = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc',
                                scale_pos_weight=pos_weight0, random_state=42, n_jobs=-1)
    model0.fit(X0_train, y0_train)
    return calculate_t_learner_metrics(model1, model0, X_test, w_test, y_test)


# ── Main loop (Cell 27) — only 5 lines changed, marked with # CHANGED ─────────
all_outcome_summaries = {}

for outcome_var in outcome_cols:
    print("-" * 75)
    print(f"✨ Processing Outcome Variable: {outcome_var}")

    temp_X = X.copy()
    temp_w = w.copy()
    temp_y = spine[outcome_var].astype(float)  # CHANGED: direct column access

    temp_df_combined = pd.concat([temp_X, temp_y.rename('y_outcome'), temp_w.rename('w_strat')], axis=1)
    temp_df_combined = temp_df_combined.dropna(subset=['y_outcome', 'w_strat'])

    current_X = temp_df_combined.drop(columns=['y_outcome', 'w_strat'])
    current_y = temp_df_combined['y_outcome']
    current_w = temp_df_combined['w_strat']

    N_samples = len(current_y)
    if N_samples < 20 or current_y.nunique() < 2 or current_w.nunique() < 2:
        print(f"⚠️ Skipping {outcome_var}: Insufficient samples or variance")
        continue

    outcome_save_dir = os.path.join(base_save_dir, outcome_var)  # CHANGED: path
    os.makedirs(outcome_save_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        current_X, current_y, current_w,
        test_size=0.4, random_state=42, stratify=current_w
    )

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    w_train = w_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    w_test  = w_test.reset_index(drop=True)

    missing_threshold = 0.5
    cols_to_drop = X_train.columns[X_train.isna().mean() > missing_threshold]
    if len(cols_to_drop) > 0:
        print(f"Dropping columns with >{missing_threshold*100}% missing: {len(cols_to_drop)} columns")
    X_train = X_train.drop(columns=cols_to_drop)
    X_test  = X_test.drop(columns=cols_to_drop)

    col_means_train = X_train.mean()
    X_train = X_train.fillna(col_means_train)
    X_test  = X_test.fillna(col_means_train)

    X_train_np = X_train.to_numpy(dtype=np.float32)
    w_train_np = w_train.to_numpy(dtype=np.float32)
    y_train_np = y_train.to_numpy(dtype=np.float32)
    X_test_np  = X_test.to_numpy(dtype=np.float32)
    w_test_np  = w_test.to_numpy(dtype=np.float32)
    y_test_np  = y_test.to_numpy(dtype=np.float32)

    # --- NN-PEHE precomputation (model-agnostic) ---
    X_all_np = np.vstack([X_train_np, X_test_np])
    w_all_np = np.concatenate([w_train_np, w_test_np])
    y_all_np = np.concatenate([y_train_np, y_test_np])

    print("  Computing NN counterfactuals (this may take a while)...")
    ite_nn_all  = build_nn_counterfactuals(X_all_np, w_all_np, y_all_np, 5)
    ite_nn_test = ite_nn_all[len(X_train_np):]

    models_to_run = {
        "T-Learner (LogReg)":       lambda: run_t_learner_lr(X_train_np, w_train_np, y_train_np, X_test_np, w_test_np, y_test_np),
        "T-Learner (DecisionTree)": lambda: run_t_learner_dt(X_train_np, w_train_np, y_train_np, X_test_np, w_test_np, y_test_np),
        "T-Learner (RandomForest)": lambda: run_t_learner_rf(X_train_np, w_train_np, y_train_np, X_test_np, w_test_np, y_test_np),
        "T-Learner (GradientBoost)":lambda: run_t_learner_gb(X_train_np, w_train_np, y_train_np, X_test_np, w_test_np, y_test_np),
        "T-Learner (SVC)":          lambda: run_t_learner_svc(X_train_np, w_train_np, y_train_np, X_test_np, w_test_np, y_test_np),
        "T-Learner (K-NearestNeigh)":lambda: run_t_learner_knn(X_train_np, w_train_np, y_train_np, X_test_np, w_test_np, y_test_np),
        "T-Learner (XGBoost)":      lambda: run_t_learner_xgb(X_train_np, w_train_np, y_train_np, X_test_np, w_test_np, y_test_np),
    }

    outcome_summary_results = []
    BASE_METRICS      = ["Accuracy", "Precision", "Recall", "F1 Score", "AUROC", "ATE Error", "PEHE"]
    T_LEARNER_METRICS = ["Treated_Accuracy", "Control_Accuracy", "Treated_AUROC", "Control_AUROC"]
    ALL_COLUMNS       = ["Model"] + BASE_METRICS + T_LEARNER_METRICS

    best_model_name  = None
    best_nn_pehe     = np.inf
    best_true_plugin = None
    best_ite         = None

    for model_name, run_func in models_to_run.items():
        print(f"\n-> Running {model_name}...")
        row = {"Model": model_name}
        for col in BASE_METRICS + T_LEARNER_METRICS:
            row[col] = np.nan
        try:
            models, results = run_func()
            metrics = {k: v for k, v in results.items() if k != "ITE"}
            row.update(metrics)
            outcome_summary_results.append(row)
            np.save(os.path.join(outcome_save_dir, f"{model_name}_ITE.npy"), results['ITE'])
            nn_pehe_val = results.get("PEHE", np.nan)
            if np.isfinite(nn_pehe_val) and nn_pehe_val < best_nn_pehe:
                best_nn_pehe     = nn_pehe_val
                best_model_name  = model_name
                best_true_plugin = results.get("ATE Error", np.nan)
                best_ite         = results["ITE"]
        except Exception as e:
            print(f"🛑 Error running {model_name}: {e}")
            outcome_summary_results.append(row)

    print("\n" + "*" * 60)
    print(f"🏆 Best Model for Outcome: {outcome_var}")
    print(f"Model: {best_model_name}")
    print(f"Lowest NN-PEHE: {best_nn_pehe:.4f}")
    print(f"ATE Error: {best_true_plugin:.4f}")
    print("*" * 60)

    os.makedirs(ite_save_dir, exist_ok=True)  # CHANGED: path
    best_ite_path = os.path.join(ite_save_dir, f"ite_t-learner_{outcome_var}.npy")
    np.save(best_ite_path, best_ite)

    outcome_summary_df = pd.DataFrame(outcome_summary_results)
    for col in ALL_COLUMNS:
        if col not in outcome_summary_df.columns:
            outcome_summary_df[col] = np.nan
    outcome_summary_df = outcome_summary_df[ALL_COLUMNS]
    all_outcome_summaries[outcome_var] = outcome_summary_df

    print("\n" + "=" * 75)
    print(f"📊 SUMMARY TABLE FOR T-LEARNERS ON OUTCOME: {outcome_var}")
    print(f"Test Set Size: {len(y_test)}")
    test_dist = pd.Series(y_test).value_counts().to_dict()
    w_dist_0  = pd.Series(y_test[w_test == 0]).value_counts().to_dict()
    w_dist_1  = pd.Series(y_test[w_test == 1]).value_counts().to_dict()
    print(f"Test Y Distribution: {test_dist}")
    print(f"Test Y | W=0 (Control): {w_dist_0}")
    print(f"Test Y | W=1 (Treated): {w_dist_1}")
    print("=" * 75)
    print(outcome_summary_df.round(4).to_string())  # CHANGED 5: display() → print()
    print("\n" + "-" * 75)

    # Save summary CSV
    outcome_summary_df.to_csv(
        os.path.join(outcome_save_dir, f"t_learner_summary_{outcome_var}.csv"), index=False
    )

print("-" * 75)
print("✅ T-Learner Processing Complete.")
print("📝 All T-Learner outcome summary tables stored in 'all_outcome_summaries'.")


# ── Plotting functions (Cell 28) — UNCHANGED ──────────────────────────────────
T_LEARNER_MODEL_NAMES = [
    'T-Learner (LogReg)', 'T-Learner (DecisionTree)', 'T-Learner (RandomForest)',
    'T-Learner (GradientBoost)', 'T-Learner (SVC)', 'T-Learner (K-NearestNeigh)',
    'T-Learner (XGBoost)',
]

CLASSIFICATION_METRICS = ["Accuracy", "Precision", "Recall", "F1 Score", "AUROC",
                           "Treated_Accuracy", "Control_Accuracy"]
ATE_METRIC  = "ATE Error"
PEHE_METRIC = "PEHE"

MODEL_COLORS = {
    'T-Learner (LogReg)':        'mediumslateblue',
    'T-Learner (DecisionTree)':  'darkorchid',
    'T-Learner (RandomForest)':  'lightcoral',
    'T-Learner (GradientBoost)': 'palegreen',
    'T-Learner (SVC)':           'khaki',
    'T-Learner (K-NearestNeigh)':'midnightblue',
    'T-Learner (XGBoost)':       'gray',
}


def plot_t_learner_classification_metric(df_summary, metric_name, outcome_var, save_dir, colors):
    plt.style.use('seaborn-v0_8-whitegrid')
    df_plot = df_summary[df_summary['Model'].isin(T_LEARNER_MODEL_NAMES)].copy()
    df_plot = df_plot[['Model', metric_name]].dropna(subset=[metric_name]).set_index('Model')
    if df_plot.empty: return
    df_plot = df_plot.reindex([m for m in T_LEARNER_MODEL_NAMES if m in df_plot.index])
    models      = df_plot.index.tolist()
    values      = df_plot[metric_name].values
    plot_colors = [colors[m] for m in models]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(models)), values, color=plot_colors, width=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(range(1, len(models) + 1), fontsize=10)
    ax.set_xlabel("Model Index", fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.set_ylabel(metric_name, fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}',
                ha='center', va='bottom', fontsize=9)
    legend_labels = [f"({i+1}) {model}" for i, model in enumerate(models)]
    ax.legend(bars, legend_labels, title="Model Key", bbox_to_anchor=(1.01, 1),
              loc='upper left', fontsize=9)
    plt.title(f'T-Learner {metric_name} Comparison: {outcome_var}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'T-Learner_{outcome_var}_{metric_name.replace(" ", "_")}_comparison.png'))
    plt.close()


def plot_t_learner_ate_comparison(df_summary, metric_name, outcome_var, save_dir, colors):
    plt.style.use('seaborn-v0_8-whitegrid')
    df_plot = df_summary[df_summary['Model'].isin(T_LEARNER_MODEL_NAMES)].copy()
    df_plot = df_plot[['Model', metric_name]].dropna(subset=[metric_name]).set_index('Model')
    if df_plot.empty: return
    df_plot = df_plot.reindex([m for m in T_LEARNER_MODEL_NAMES if m in df_plot.index])
    models      = df_plot.index.tolist()
    values      = df_plot[metric_name].values
    plot_colors = [colors[m] for m in models]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(models)), values, color=plot_colors, width=0.8)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    max_val = np.max(np.abs(values)) if len(values) > 0 else 1
    ax.set_ylim(-max_val * 1.2, max_val * 1.2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(range(1, len(models) + 1), fontsize=10)
    ax.set_xlabel("Model Index", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}',
                ha='center', va='center', fontsize=9)
    legend_labels = [f"({i+1}) {model}" for i, model in enumerate(models)]
    ax.legend(bars, legend_labels, title="Model Key", bbox_to_anchor=(1.01, 1),
              loc='upper left', fontsize=9)
    plt.title(f'T-Learner {metric_name} Comparison: {outcome_var}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'T-Learner_{outcome_var}_{metric_name.replace(" ", "_")}_comparison.png'))
    plt.close()


def plot_t_learner_ite_distribution(model_name, outcome_var, ite_array, save_dir):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(ite_array, bins=15, kde=True, ax=ax, color='skyblue',
                 edgecolor='black', line_kws={'linewidth': 2, 'alpha': 0.8})
    ax.axvline(0, color='r', linestyle='--', linewidth=1.5, label='No Effect')
    ax.set_title(f'ITE Distribution for {model_name}: {outcome_var}', fontsize=14)
    ax.set_xlabel('Estimated ITE', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_ITE_Distribution.png'))
    plt.close()


def plot_t_learner_pehe_comparison(df_summary, outcome_var, save_dir, colors):
    plt.style.use('seaborn-v0_8-whitegrid')
    df_plot = df_summary[df_summary['Model'].isin(T_LEARNER_MODEL_NAMES)].copy()
    df_plot = df_plot[['Model', 'PEHE']].dropna(subset=['PEHE'])
    if df_plot.empty: return
    df_plot     = df_plot.sort_values('PEHE', ascending=True).set_index('Model')
    models      = df_plot.index.tolist()
    values      = df_plot['PEHE'].values
    plot_colors = [colors[m] for m in models]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(models)), values, color=plot_colors, width=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(range(1, len(models) + 1), fontsize=10)
    ax.set_xlabel("Model Index (Sorted by PEHE)", fontsize=12)
    ax.set_ylabel('PEHE (Lower is Better)', fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.05 + 0.01),
                f'{yval:.4f}', ha='center', va='bottom', fontsize=9)
    legend_labels = [f"({i+1}) {model}" for i, model in enumerate(models)]
    ax.legend(bars, legend_labels, title="Model Key", bbox_to_anchor=(1.01, 1),
              loc='upper left', fontsize=9)
    plt.title(f'T-Learner PEHE Comparison: {outcome_var}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'T-Learner_{outcome_var}_PEHE_comparison.png'))
    plt.close()


def generate_t_learner_plots(all_outcome_summaries, base_save_dir):
    for outcome_var, df_summary in all_outcome_summaries.items():
        print("-" * 50)
        print(f"Generating T-Learner plots for Outcome: {outcome_var}")
        outcome_save_dir = os.path.join(base_save_dir, outcome_var)
        os.makedirs(outcome_save_dir, exist_ok=True)

        for metric in CLASSIFICATION_METRICS:
            try:
                plot_t_learner_classification_metric(df_summary, metric, outcome_var, outcome_save_dir, MODEL_COLORS)
                print(f"  Saved {metric} plot.")
            except Exception as e:
                print(f"🛑 Error plotting {metric}: {e}")

        try:
            plot_t_learner_ate_comparison(df_summary, ATE_METRIC, outcome_var, outcome_save_dir, MODEL_COLORS)
            print(f"  Saved {ATE_METRIC} plot.")
        except Exception as e:
            print(f"🛑 Error plotting {ATE_METRIC}: {e}")

        try:
            plot_t_learner_pehe_comparison(df_summary, outcome_var, outcome_save_dir, MODEL_COLORS)
            print("  Saved PEHE plot.")
        except Exception as e:
            print(f"🛑 Error plotting PEHE: {e}")

        for model_name in df_summary[df_summary['Model'].isin(T_LEARNER_MODEL_NAMES)]['Model'].unique():
            try:
                ite_file = os.path.join(outcome_save_dir, f"{model_name}_ITE.npy")
                if os.path.exists(ite_file):
                    ite_data = np.load(ite_file)
                    plot_t_learner_ite_distribution(model_name, outcome_var, ite_data, outcome_save_dir)
                    print(f"  Saved ITE distribution plot for {model_name}.")
            except Exception as e:
                print(f"🛑 Error plotting ITE for {model_name}: {e}")


generate_t_learner_plots(all_outcome_summaries, base_save_dir)