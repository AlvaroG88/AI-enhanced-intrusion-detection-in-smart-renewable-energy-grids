# temporal_feature_selection.py
# ------------------------------------------------------------
# Step 3 – Temporal Feature Selection (RQ2 support)
# Ranks temporal features by multiple importance metrics and
# saves both per-method rankings and a consolidated averaged ranking.
# ------------------------------------------------------------

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

# -----------------------------
# Locate processed CSV
# -----------------------------
candidates = []
candidates += glob.glob("processed_smart_grid_attacks*.csv")
candidates += glob.glob("Dataset/processed_smart_grid_attacks*.csv")
if not candidates:
    raise FileNotFoundError(
        "Processed CSV not found. Place it as 'processed_smart_grid_attacks.csv' "
        "in the repo root or under 'Dataset/'."
    )
INPUT = candidates[0]
OUTDIR = Path("feature_selection")  # <- all Step-3 outputs here
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"Using input: {INPUT}")

# -----------------------------
# Load data & target
# -----------------------------
df = pd.read_csv(INPUT)

if "is_attack" not in df.columns:
    df["is_attack"] = (df["attack_type"].fillna("benign").str.lower() != "benign").astype(int)

# -----------------------------
# Temporal features
# -----------------------------
num_features = ["hour", "day", "weekday"]
bin_features = [f for f in ["is_weekend", "is_peak_hour"] if f in df.columns]
cat_features = ["time_segment"]  # morning/afternoon/night

X = df[num_features + bin_features + cat_features].copy()
y = df["is_attack"].astype(int)

mask = X.notna().all(axis=1)
X, y = X[mask], y[mask]

# -----------------------------
# Train/validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------
# Preprocessing for RF/LR/Permutation
# - Keep ALL time_segment dummies (drop=None) so everything is visible
# - Fix categories to ensure all 3 are present even if missing in train
# -----------------------------
ohe_all = OneHotEncoder(
    drop=None,
    handle_unknown="ignore",
    categories=[["morning", "afternoon", "night"]],
)

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("bin", "passthrough", bin_features),
        ("cat", ohe_all, cat_features),
    ],
    remainder="drop",
)

pre.fit(X_train)

cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(cat_features))
final_feature_names = num_features + bin_features + cat_names
print("Final features used by RF/LR/Permutation:", final_feature_names)

X_train_t = pre.transform(X_train)
X_val_t = pre.transform(X_val)

# -----------------------------
# 1) MUTUAL INFORMATION (DISCRETE-AWARE)
#    Compute MI on a discrete-friendly matrix (no scaling),
#    marking all columns as discrete so hour/weekday are not washed out.
# -----------------------------
# Build MI design:
# - hour, day, weekday as ints
# - binary features as ints
# - one-hot time_segment (all categories kept)
ohe_mi = OneHotEncoder(
    drop=None, handle_unknown="ignore",
    categories=[["morning", "afternoon", "night"]]
)
Z_cat = ohe_mi.fit_transform(X_train[["time_segment"]]).toarray()
cat_names_mi = list(ohe_mi.get_feature_names_out(["time_segment"]))

Z_num = X_train[["hour", "day", "weekday"]].astype(int).to_numpy()
bin_cols_present = [c for c in ["is_weekend", "is_peak_hour"] if c in X_train.columns]
Z_bin = X_train[bin_cols_present].astype(int).to_numpy() if bin_cols_present else np.empty((len(X_train), 0))

Z = np.hstack([Z_num, Z_bin, Z_cat])
mi_feature_names = ["hour", "day", "weekday"] + bin_cols_present + cat_names_mi

discrete_mask = np.ones(Z.shape[1], dtype=bool)
mi_scores = mutual_info_classif(Z, y_train, discrete_features=discrete_mask, random_state=42)
mi = pd.Series(mi_scores, index=mi_feature_names, name="mutual_info")

# -----------------------------
# 2) Random Forest
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train_t, y_train)
rf_imp = pd.Series(rf.feature_importances_, index=final_feature_names, name="rf_importance")

# -----------------------------
# 3) Logistic Regression (abs standardized coeffs)
# -----------------------------
lr = LogisticRegression(
    solver="liblinear",
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)
lr.fit(X_train_t, y_train)
lr_imp = pd.Series(np.abs(lr.coef_).ravel(), index=final_feature_names, name="logreg_abs_coef")

y_pred = lr.predict(X_val_t)
print("\nLogistic Regression (validation) — quick check:")
print(classification_report(y_val, y_pred, digits=3))

# -----------------------------
# 4) Permutation Importance (validation)
# -----------------------------
perm = permutation_importance(
    rf, X_val_t, y_val, n_repeats=15, random_state=42, n_jobs=-1
)
perm_imp = pd.Series(perm.importances_mean, index=final_feature_names, name="permutation_mean")

# -----------------------------
# Combine (for average rank later)
# -----------------------------
imp_df = pd.concat([mi, rf_imp, lr_imp, perm_imp], axis=1).fillna(0.0)

# ------------------------------------------------------------
# Per-method rankings (no averaging)
# ------------------------------------------------------------
def save_ranking(series: pd.Series, method_name: str, outdir: Path, top_n: int = None):
    rank_df = series.sort_values(ascending=False).reset_index()
    rank_df.columns = ["feature", method_name]

    # Save CSV (full list)
    csv_path = outdir / f"feature_ranking_{method_name}.csv"
    rank_df.to_csv(csv_path, index=False)

    # Chart (top_n or all)
    chart_df = rank_df if top_n is None else rank_df.head(top_n)
    plt.figure(figsize=(10, max(4, 0.35 * len(chart_df))))
    plt.barh(chart_df["feature"], chart_df[method_name])
    plt.title(f"Temporal Feature Ranking — {method_name} (higher is better)")
    plt.xlabel(method_name)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig_path = outdir / f"feature_ranking_{method_name}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\n[{method_name}] Top 10 features:")
    print(rank_df.head(10))
    print("Saved:", csv_path.resolve(), "|", fig_path.resolve())

# Save per-method rankings (set top_n=None to plot ALL features)
save_ranking(mi,        "mutual_info",        OUTDIR, top_n=None)
save_ranking(rf_imp,    "rf_importance",      OUTDIR, top_n=None)
save_ranking(lr_imp,    "logreg_abs_coef",    OUTDIR, top_n=None)
save_ranking(perm_imp,  "permutation_mean",   OUTDIR, top_n=None)

# -----------------------------
# Average-rank summary
# -----------------------------
imp_norm = (imp_df - imp_df.min()) / (imp_df.max() - imp_df.min() + 1e-12)
ranks = imp_norm.rank(ascending=False, method="average")
imp_df["avg_rank"] = ranks.mean(axis=1)

summary = imp_df.sort_values("avg_rank").reset_index().rename(columns={"index": "feature"})

csv_path = OUTDIR / "feature_importance_summary.csv"
summary.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path.resolve()}")

plt.figure(figsize=(10, 6))
plt.barh(summary["feature"], -summary["avg_rank"])
plt.title("Temporal Feature Importance — Average Rank (lower is better)")
plt.xlabel("Negative Average Rank")
plt.gca().invert_yaxis()
plt.tight_layout()
fig_path = OUTDIR / "temporal_feature_importance.png"
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Saved: {fig_path.resolve()}")

print("\nTop features (average rank):")
print(summary.head(10))
print("\nDone.")
