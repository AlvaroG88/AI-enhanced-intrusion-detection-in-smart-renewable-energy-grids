# temporal_modeling_xgb.py
# ------------------------------------------------------------
# Step 4 – Temporal Prediction Modeling (XGBoost)
# - Binary mode: predict attack occurrence (is_attack)
# - Multiclass mode (optional): predict attack_type (non-benign classes)
# - Uses time-aware split to avoid look-ahead bias
# - Evaluates with Accuracy, Precision, Recall, F1, Confusion Matrix
# - Saves trained model and metrics
# ------------------------------------------------------------

from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, ConfusionMatrixDisplay
)

from xgboost import XGBClassifier
import joblib

# =========================
# ====== CONFIG ===========
# =========================
# Mode: "binary" (attack likelihood) or "multiclass" (attack_type)
MODE = "binary"          # change to "multiclass" if you want types

# Where to save artifacts
OUTDIR = Path("xgb_models")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Which features to use (temporal only)
USE_FEATURES = {
    "hour": True,            # <- best temporal feature per Step 3
    "weekday": True,
    "is_weekend": True,
    "time_segment": True,    # will be one-hot encoded (morning/afternoon/night)
    "is_peak_hour": True,    # use True only if exists in your CSV
}

# XGBoost base hyperparameters (good defaults; adjust if needed)
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="binary:logistic",   # overwritten in multiclass mode
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42
)

# =========================
# === 1) Load dataset  ====
# =========================
candidates = []
candidates += glob.glob("processed_smart_grid_attacks*.csv")
candidates += glob.glob("Dataset/processed_smart_grid_attacks*.csv")
if not candidates:
    raise FileNotFoundError("Cannot find processed CSV. Put it in repo root or Dataset/.")

INPUT = candidates[0]
print(f"Using input: {INPUT}")
df = pd.read_csv(INPUT)

# Target construction
if "is_attack" not in df.columns:
    df["is_attack"] = (df["attack_type"].fillna("benign").str.lower() != "benign").astype(int)

# =========================
# === 2) Select features ==
# =========================
feature_cols = []
num_bin_cols = []  # numeric/binary passthrough
cat_cols = []      # categorical (one-hot)

if USE_FEATURES.get("hour", False) and "hour" in df.columns: feature_cols.append("hour"); num_bin_cols.append("hour")
if USE_FEATURES.get("weekday", False) and "weekday" in df.columns: feature_cols.append("weekday"); num_bin_cols.append("weekday")
if USE_FEATURES.get("is_weekend", False) and "is_weekend" in df.columns: feature_cols.append("is_weekend"); num_bin_cols.append("is_weekend")
if USE_FEATURES.get("is_peak_hour", False) and "is_peak_hour" in df.columns: feature_cols.append("is_peak_hour"); num_bin_cols.append("is_peak_hour")
if USE_FEATURES.get("time_segment", False) and "time_segment" in df.columns: feature_cols.append("time_segment"); cat_cols.append("time_segment")

# Drop rows with missing features
X = df[feature_cols].copy()
mask = X.notna().all(axis=1)
df = df[mask].copy()
X = X[mask]

# Targets by mode
if MODE == "binary":
    y = df["is_attack"].astype(int)
    objective = "binary:logistic"
    num_classes = None
    model_name = "xgb_attack_likelihood"
else:
    # Multiclass over non-benign types (drop benign)
    nonb = df[df["is_attack"] == 1].copy()
    if nonb["attack_type"].nunique() < 2:
        raise ValueError("Not enough non-benign classes for multiclass training.")
    X = nonb[feature_cols].copy()
    y = nonb["attack_type"].astype(str).values
    objective = "multi:softprob"
    classes = sorted(pd.unique(y))
    num_classes = len(classes)
    model_name = "xgb_attack_type"
    print("Classes:", classes)

# =========================
# === 3) Time-aware split =
# =========================
# Sort by original timestamp to avoid look-ahead bias
if "timestamp" in df.columns:
    # If string format: sort lexicographically works for ISO; otherwise parse
    try:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
    except Exception:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).tz_localize(None)
else:
    # Fallback: build a pseudo-time index
    ts = pd.Series(np.arange(len(df)), index=df.index)

# Align X and y to df used for y above
if MODE == "binary":
    aligned_index = df.index
else:
    aligned_index = nonb.index

ts = ts.loc[aligned_index]
X = X.loc[aligned_index]
if MODE == "binary":
    y = y.loc[aligned_index]

# Split: first 80% time for train, last 20% for test
order = ts.sort_values().index
cut = int(0.8 * len(order))
train_idx, test_idx = order[:cut], order[cut:]

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = (y.loc[train_idx], y.loc[test_idx]) if MODE == "binary" else (
    y[np.isin(aligned_index, train_idx)],
    y[np.isin(aligned_index, test_idx)]
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# =========================
# === 4) Preprocess + XGB =
# =========================
pre = ColumnTransformer(
    transformers=[
        ("num_bin", "passthrough", num_bin_cols),
        ("cat", OneHotEncoder(drop=None, handle_unknown="ignore",
                              categories=[["morning","afternoon","night"]] if "time_segment" in cat_cols else "auto"),
         cat_cols),
    ],
    remainder="drop",
)

params = XGB_PARAMS.copy()
params["objective"] = objective
if MODE == "multiclass":
    params["num_class"] = num_classes

# Class imbalance handling (binary)
if MODE == "binary":
    pos_rate = y_train.mean()
    # scale_pos_weight ≈ (neg/pos)
    params["scale_pos_weight"] = float((1 - pos_rate) / max(pos_rate, 1e-6))

clf = XGBClassifier(**params)

pipe = Pipeline(steps=[
    ("pre", pre),
    ("xgb", clf),
])

pipe.fit(X_train, y_train)

# =========================
# === 5) Evaluation =======
# =========================
y_pred = pipe.predict(X_test)

if MODE == "binary":
    y_proba = pipe.predict_proba(X_test)[:, 1]
else:
    y_proba = None

print("\nClassification report (test):")
print(classification_report(y_test, y_pred, digits=3))

# Metrics summary
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
metrics = dict(accuracy=acc, precision=prec, recall=rec, f1=f1)
print("Metrics:", metrics)

# Confusion matrix
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize=None)
plt.title(f"Confusion Matrix — {model_name}")
plt.tight_layout()
cm_path = OUTDIR / f"{model_name}_confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print("Saved:", cm_path.resolve())

# =========================
# === 6) Save artifacts ===
# =========================
model_path = OUTDIR / f"{model_name}.joblib"
joblib.dump(pipe, model_path)
with open(OUTDIR / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print("Saved model:", model_path.resolve())
print("Saved metrics:", (OUTDIR / f"{model_name}_metrics.json").resolve())

print("\nDone.")
