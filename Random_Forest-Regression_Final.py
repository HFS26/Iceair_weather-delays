# ===============================================================
# Random Forest Regression — with month/weekday/aircraft (ordinal-encoded)
# Color-coded importances: numeric (red) vs categorical (dark red)
# Works for target = "departure_delay_min" OR "arrival_delay_min"
# ===============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, accuracy_score
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

# === configure which columns are categorical (string features you encoded) ===
CAT_COLS = ["schedule_month", "schedule_weekday", "aircraft_type"]

# === colors (feel free to adjust) ===
COLOR_NUMERIC = "#1f77b4"      # blue #1f77b4 red #d62728
COLOR_CATEGORICAL = "#0d3b66"  # dark blue #0d3b66 dark red #8C1C13

# ---------------------------------------------------------------
# 1) Load & Prepare Data (simple)
# ---------------------------------------------------------------
def load_and_prepare_data(file_path, target_column="arrival_delay_min"):
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    if target_column not in df.columns:
        sys.exit(f"Error: Target column '{target_column}' not found in dataset.")

    # Drop text/time/identifier columns (keep schedule_* + aircraft_type)
    drop_cols = [
        "Origin-Airport", "Destination-Airport",
        "origin", "destination", "flight_number",
        "scheduled_departure_time", "acutal_departure_time",
        "scheduled_arrival_time", "actual_arrival_time"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Ensure target present and split out
    df = df.dropna(subset=[target_column])
    y = df[target_column].astype(float)
    X = df.drop(columns=[target_column])

    # ---- Leakage guard (drop both binary + minute versions for current and opposite task) ----
    if target_column.startswith("departure"):
        leak_cols = ["departure_delay", "departure_delay_min", "arrival_delay", "arrival_delay_min"]
    else:
        leak_cols = ["arrival_delay", "arrival_delay_min", "departure_delay", "departure_delay_min"]
    leak_cols = [c for c in leak_cols if c in X.columns]
    if leak_cols:
        print(f"🛡️ Dropping leakage columns: {leak_cols}")
        X = X.drop(columns=leak_cols)

    # ---- Encode 3 string categoricals into ONE integer column each (no one-hot) ----
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = (
                X[c].astype("string").fillna("Unknown").astype("category").cat.codes
            ).astype("int32")
        else:
            print(f"Missing categorical column: {c} (skipping)")

    # Keep numeric columns only (encoded cats included), fill NaNs
    X = X.select_dtypes(include=[np.number]).fillna(X.mean(numeric_only=True))

    print(f"After cleaning: {X.shape[1]} features, {len(X)} samples.")
    return X, y

# ---------------------------------------------------------------
# 1b) Actual vs Predicted outcome bars
# ---------------------------------------------------------------
def plot_actual_vs_predicted_outcomes(y_test, preds, threshold_minutes=15):
    """
    Show counts for On-time vs Delay from the regression output.
    - 'Delay' if minutes > threshold_minutes
    - 'On-time' otherwise
    """
    y_test = np.asarray(y_test)
    preds = np.asarray(preds)

    # Actual and predicted binary labels
    actual_cls = np.where(y_test > threshold_minutes, "Delay", "On-time")
    pred_cls   = np.where(preds > threshold_minutes, "Delay", "On-time")

    classes = ["On-time", "Delay"]

    def counts(labels):
        return [
            np.sum(labels == "On-time"),
            np.sum(labels == "Delay"),
        ]

    actual_counts = counts(actual_cls)
    pred_counts   = counts(pred_cls)

    # Accuracy
    acc = (actual_cls == pred_cls).mean()

    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width/2, actual_counts, width, color="#2ca02c", label="Actual")
    plt.bar(x + width/2, pred_counts,  width, color="#1f77b4",  label="Predicted")
    plt.xticks(x, classes)
    plt.ylabel("Count")
    plt.title(f"Actual vs Predicted Outcomes (Test set)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------
# 2) Train & Evaluate
# ---------------------------------------------------------------
def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\nTraining with {X_train.shape[1]} features...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )

    print("\nTraining Random Forest Regressor...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # Version-safe RMSE (sklearn >=1.4)
    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(y_test, preds)
    except ImportError:
        rmse = mean_squared_error(y_test, preds, squared=False)

    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    print(f"\nHoldout RMSE: {rmse:.4f}")
    print(f"Holdout  MAE: {mae:.4f}")
    print(f"Holdout   R²: {r2:.4f}")

    feature_importance_plot(model, X_train, cat_cols=CAT_COLS)

    # Optional: classification-style view for a chosen threshold
    THRESHOLD_MINUTES = 15
    y_true_cls = (y_test.values > THRESHOLD_MINUTES).astype(int)
    y_pred_cls = (preds > THRESHOLD_MINUTES).astype(int)
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
    acc = accuracy_score(y_true_cls, y_pred_cls)

    print(f"\n=== Classification-style summary (>{THRESHOLD_MINUTES} min delay) ===")
    print("Confusion matrix [rows=true, cols=pred]:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")

    # Outcome bars (On-time / Delay)
    plot_actual_vs_predicted_outcomes(y_test, preds, threshold_minutes=THRESHOLD_MINUTES)

    return model

# ---------------------------------------------------------------
# 3) Feature Importances (color-coded)
# ---------------------------------------------------------------
def feature_importance_plot(model, X_train, top_n=15, cat_cols=None):
    if cat_cols is None:
        cat_cols = []

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    feats = X_train.columns[idx]
    vals = importances[idx]

    # Identify which of the top features are categorical vs numeric
    is_cat = np.array([f in cat_cols for f in feats])

    colors = [COLOR_CATEGORICAL if is_cat[i] else COLOR_NUMERIC for i in range(len(feats))]

    plt.figure(figsize=(8, 6))
    plt.barh(feats[::-1], vals[::-1], color=colors[::-1])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Important Features")

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLOR_NUMERIC, label="Numeric features"),
        mpatches.Patch(color=COLOR_CATEGORICAL, label="Categorical (encoded)"),
    ]
    plt.legend(handles=legend_patches, loc="lower right")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------
# 4) Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    data_file = "flights_with_schedule_fields.csv"
    target = "departure_delay_min"   # or "arrival_delay_min"
    # target = "arrival_delay_min"

    X, y = load_and_prepare_data(data_file, target_column=target)
    _ = train_and_evaluate(X, y)









