# ===============================================================
# Probabilistic Flight Outcomes (Cancel / Delay / On-time)
# - Cancellation inferred from empty actual times
# - Two Random Forest classifiers with probabilities
# - Ordinal encoding for: schedule_month, schedule_weekday, aircraft_type
# - Metrics + feature importances for both models
# - Color-coded importance plots (by model AND by feature type)
# - Actual vs Predicted Outcomes bar chart (green=actual, blue=predicted)
# ===============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

CAT_COLS = ["schedule_month", "schedule_weekday", "aircraft_type"]

# ---------------------------------------------------------------
# 1) Load & Prepare Data
# ---------------------------------------------------------------
def load_and_prepare_data(file_path, delay_target="departure_delay"):
    """
    delay_target: 'departure_delay' or 'arrival_delay' (binary 0/1).
    Cancellation = BOTH actual_departure_time AND actual_arrival_time missing/empty.
    """
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    if delay_target not in df.columns:
        sys.exit(f"Error: Delay target '{delay_target}' not found in dataset.")

    # --- Build cancellation label from actual times BEFORE dropping them ---
    for col in ["actual_departure_time", "actual_arrival_time"]:
        if col in df.columns:
            # Treat empty strings/whitespace as NaN
            df[col] = df[col].astype("string").replace(r"^\s*$", np.nan, regex=True)
        else:
            print(f"Column '{col}' not found; cancellation inference may be incomplete.")

    if {"actual_departure_time", "actual_arrival_time"}.issubset(df.columns):
        y_cancel = (df["actual_departure_time"].isna() & df["actual_arrival_time"].isna()).astype(int)
    else:
        print("Could not infer cancellations (missing actual time columns). Assuming no cancellations.")
        y_cancel = pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    # --- Drop identifiers / raw timestamp columns from FEATURES only ---
    drop_cols = [
        "Origin-Airport", "Destination-Airport",
        "origin", "destination", "flight_number",
        "scheduled_departure_time", "acutal_departure_time",
        "scheduled_arrival_time", "actual_arrival_time", "actual_departure_time"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # --- Delay target (binary). Keep NA rows for cancel model alignment ---
    y_delay = df[delay_target].astype("float").round().astype("Int64")

    # --- Build X then drop leakage ---
    X = df.drop(columns=[delay_target]) if delay_target in df.columns else df.copy()

    # Never use minute-level delay features or the opposite binary label (leakage)
    always_leak = ["departure_delay_min", "arrival_delay_min"]
    other_label = "arrival_delay" if delay_target == "departure_delay" else "departure_delay"
    leakage_cols = [c for c in always_leak + [other_label] if c in X.columns]
    if leakage_cols:
        print(f"🛡️ Dropping leakage columns: {leakage_cols}")
        X = X.drop(columns=leakage_cols)

    # --- Ordinal-encode the 3 categoricals into single integer columns ---
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = (
                X[c].astype("string").fillna("Unknown").astype("category").cat.codes
            ).astype("int32")
        else:
            print(f"Missing categorical column: {c} (skipping)")

    # Keep numerics & impute simple means
    X = X.select_dtypes(include=[np.number]).fillna(X.mean(numeric_only=True))

    print(f"After cleaning: {X.shape[1]} features, {len(X)} samples.")
    return X, y_delay, y_cancel

# ---------------------------------------------------------------
# 2) Train two models (Cancel + Delay), output probabilities + importances
# ---------------------------------------------------------------
def train_and_evaluate_prob(X, y_delay, y_cancel, test_size=0.2, random_state=42):
    # Split once; stratify on cancellations if both classes exist
    strat = y_cancel if y_cancel.nunique(dropna=True) > 1 else None
    X_train, X_test, yd_train, yd_test, yc_train, yc_test = train_test_split(
        X, y_delay, y_cancel, test_size=test_size, random_state=random_state, stratify=strat
    )

    # --- Cancellation model ---
    cancel_model = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1
    )
    print("\nTraining Cancellation Model (binary)...")
    cancel_model.fit(X_train, yc_train.astype(int))
    p_cancel_test = cancel_model.predict_proba(X_test)[:, 1]

    # Metrics for cancellation
    try:
        auc_cancel = roc_auc_score(yc_test, p_cancel_test)
        print(f"Cancellation ROC-AUC: {auc_cancel:.3f}")
    except Exception:
        pass

    TH_CANCEL = 0.5
    y_cancel_pred = (p_cancel_test >= TH_CANCEL).astype(int)
    acc_c = accuracy_score(yc_test, y_cancel_pred)
    cm_c  = confusion_matrix(yc_test, y_cancel_pred, labels=[0, 1])
    print(f"Cancellation Accuracy @ {TH_CANCEL:.2f}: {acc_c:.4f}")
    print("Cancellation Confusion Matrix [rows=true, cols=pred, 0/1=no/yes]:")
    print(cm_c)
    print("\nCancellation Classification Report:")
    print(classification_report(yc_test, y_cancel_pred, digits=3))

    # Feature importances for cancellation model
    print_top_importances(cancel_model, X_train.columns, title="Cancellation Model — Top Features")
    feature_importance_plot(
        cancel_model, X_train.columns,
        title="Cancellation Model — Top Features",
        model_tag="cancel", cat_cols=CAT_COLS
    )

    # --- Delay model trained on NON-CANCELLED rows with known delay label ---
    mask_nc_train = (yc_train == 0) & (yd_train.notna())
    delay_model = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1
    )
    print("\nTraining Delay Model (binary, among non-cancelled flights)...")
    delay_model.fit(X_train[mask_nc_train], yd_train[mask_nc_train].astype(int))

    # Predict P(delay | not cancel) on full test
    p_delay_given_not_cancel = delay_model.predict_proba(X_test)[:, 1]

    # Combine into overall probabilities
    p_delay_test = (1.0 - p_cancel_test) * p_delay_given_not_cancel

    # Simple labeling logic (tweak thresholds if needed)
    TH_DELAY = 0.5
    pred_label = np.where(
        p_cancel_test >= TH_CANCEL, "Cancel",
        np.where(p_delay_test >= TH_DELAY, "Delay", "On-time")
    )

    # Preview
    print("\n=== Probabilistic Outputs (head) ===")
    preview = pd.DataFrame({
        "p_cancel": p_cancel_test,
        "p_delay|not_cancel": p_delay_given_not_cancel,
        "p_delay": p_delay_test,
        "predicted_label": pred_label
    }).head(10)
    print(preview.to_string(index=False))

    # Evaluate delay only on non-cancelled flights with known delay label
    mask_eval = (yc_test == 0) & (yd_test.notna())
    if mask_eval.sum() > 0:
        yd_eval = yd_test[mask_eval].astype(int)
        delay_pred_binary = (p_delay_test[mask_eval] >= TH_DELAY).astype(int)
        acc = accuracy_score(yd_eval, delay_pred_binary)
        cm = confusion_matrix(yd_eval, delay_pred_binary, labels=[0, 1])
        print(f"\nDelay Accuracy @ {TH_DELAY:.2f} (non-cancelled only): {acc:.4f}")
        print("Delay Confusion Matrix [rows=true, cols=pred, 0/1=on-time/delay]:")
        print(cm)
        print("\nDelay Classification Report (non-cancelled only):")
        print(classification_report(yd_eval, delay_pred_binary, digits=3))
    else:
        print("\nNo non-cancelled eval rows with known delay label; skipping delay evaluation.")

    # Feature importances for delay model
    print_top_importances(delay_model, X_train.columns, title="Delay Model — Top Features")
    feature_importance_plot(
        delay_model, X_train.columns,
        title="Delay Model — Top Features",
        model_tag="delay", cat_cols=CAT_COLS
    )

    # --- NEW: Actual vs Predicted Outcomes bar chart (green=actual, blue=predicted) ---
    y_true_overall = np.where(
        yc_test == 1, "Cancel",
        np.where(yd_test.fillna(0).astype(int) == 1, "Delay", "On-time")
    )
    plot_actual_vs_predicted_counts(y_true_overall, pred_label)

    return cancel_model, delay_model

# ---------------------------------------------------------------
# 3) Helpers: importance table + color-coded plot
# ---------------------------------------------------------------
def print_top_importances(model, colnames, mask=None, top_n=15, title="Top Features"):
    """
    Prints a neat table of top-N importances.
    mask is ignored unless it matches the columns length (safety).
    """
    cols = np.array(colnames)
    if (
        mask is not None and hasattr(mask, "__len__")
        and len(mask) == len(cols) and getattr(mask, "dtype", None) == bool
    ):
        cols = cols[mask]

    imps = model.feature_importances_
    order = np.argsort(imps)[::-1][:top_n]
    table = pd.DataFrame({
        "feature": cols[order],
        "importance": np.round(imps[order], 6)
    })
    print(f"\n{title} (table):")
    print(table.to_string(index=False))


def feature_importance_plot(model, colnames, top_n=15, title="Top Features", model_tag="delay", cat_cols=None, mask=None):
    """
    Bar plot of top-N feature importances with colors:
    - Model tag: 'cancel' vs 'delay' (different base palettes)
    - Feature type: categorical (CAT_COLS) vs numeric (others) → different shades
    """
    if cat_cols is None:
        cat_cols = []

    cols = np.array(colnames)
    if (
        mask is not None and hasattr(mask, "__len__")
        and len(mask) == len(cols) and getattr(mask, "dtype", None) == bool
    ):
        cols = cols[mask]

    imps = model.feature_importances_
    order = np.argsort(imps)[::-1][:top_n]
    feats = cols[order]
    vals = imps[order]

    # Determine categorical vs numeric by name membership in cat_cols
    is_cat = np.array([f in cat_cols for f in feats])

    # Color scheme: different pair per model
    if model_tag == "cancel":
        color_num = "#1f77b4"  # blue #1f77b4 red #d62728
        color_cat = "#0d3b66"  # dark blue #0d3b66 dark red #8C1C13
    else:  # "delay"
        color_num = "#2ca02c"  # green #2ca02c orange #a74e00
        color_cat = "#006400"  # darker green #006400 darker orange #A74E00

    bar_colors = [color_cat if is_cat[i] else color_num for i in range(len(feats))]

    plt.figure(figsize=(8, 6))
    plt.barh(feats[::-1], vals[::-1], color=bar_colors[::-1])
    plt.xlabel("Feature Importance")
    plt.title(title)
    plt.tight_layout()

    # Legend patches
    legend_patches = [
        mpatches.Patch(color=color_num, label="Numeric features"),
        mpatches.Patch(color=color_cat, label="Categorical features"),
    ]
    plt.legend(handles=legend_patches, loc="lower right")
    plt.show()

# ---------------------------------------------------------------
# 4) Actual vs Predicted Outcomes bar chart
# ---------------------------------------------------------------
def plot_actual_vs_predicted_counts(y_true_labels, y_pred_labels):
    """
    Grouped bar chart comparing Actual vs Predicted counts for:
    On-time, Delay, Cancel. Uses green for Actual and blue for Predicted.
    """
    class_names = ["On-time", "Delay", "Cancel"]
    actual_counts = pd.Series(y_true_labels).value_counts().reindex(class_names, fill_value=0)
    pred_counts   = pd.Series(y_pred_labels).value_counts().reindex(class_names, fill_value=0)

    x = np.arange(len(class_names))
    width = 0.38

    COLOR_ACTUAL = "#2ca02c"  # green
    COLOR_PRED   = "#1f77b4"  # blue

    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width/2, actual_counts.values, width, color=COLOR_ACTUAL, label="Actual")
    plt.bar(x + width/2, pred_counts.values, width, color=COLOR_PRED,   label="Predicted")
    plt.xticks(x, class_names)
    plt.ylabel("Count")
    plt.title("Actual vs Predicted Outcomes (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------
# 5) Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    data_file = "flights_with_schedule_fields.csv"
    delay_target = "departure_delay"   # or "arrival_delay"
    #delay_target = "arrival_delay"

    X, y_delay, y_cancel = load_and_prepare_data(data_file, delay_target)
    _ = train_and_evaluate_prob(X, y_delay, y_cancel)





