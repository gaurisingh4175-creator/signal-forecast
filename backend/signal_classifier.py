
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)


# =====================================================
# 1. FILE PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_PATH = DATA_DIR / "engineered_signal_data.csv"

MODEL_PATH = MODEL_DIR / "signal_classifier.pkl"


# Create output folders
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. DEFINE FEATURES AND TARGET
# =====================================================

FEATURES = [

    # Time features
    "Hour",
    "Minute",
    "DayOfWeek",
    "IsWeekend",

    # RSRP lag features
    "RSRP_Lag_1",
    "RSRP_Lag_2",
    "RSRP_Lag_3",
    "RSRP_Lag_4",

    # RSRQ lag features
    "RSRQ_Lag_1",
    "RSRQ_Lag_2",
    "RSRQ_Lag_3",
    "RSRQ_Lag_4",

    # MOS lag features
    "MOS_Lag_1",
    "MOS_Lag_2",
    "MOS_Lag_3",
    "MOS_Lag_4",

    # Rolling statistics
    "RSRP_Rolling_Mean",
    "RSRP_Rolling_Std",

    "RSRQ_Rolling_Mean",
    "RSRQ_Rolling_Std",

    "MOS_Rolling_Mean",
    "MOS_Rolling_Std",

    # Signal changes
    "RSRP_Change",
    "RSRQ_Change",
    "MOS_Change"
]

TARGET = "Future_Quality"

CLASS_LABELS = [
    "Excellent",
    "Good",
    "Average",
    "Poor"
]


# =====================================================
# 3. LOAD ENGINEERED DATASET
# =====================================================

def load_data():

    print("\nLoading engineered dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run feature_engineering.py first."
        )

    df = pd.read_csv(DATA_PATH)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    return df


# =====================================================
# 4. PREPARE DATA
# =====================================================

def prepare_data(df):

    print("\nPreparing classification data...")

    required_columns = FEATURES + [
        TARGET,
        "Timestamp"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Keep only valid target labels
    df = df.dropna(
        subset=required_columns
    ).copy()

    df = df[
        df[TARGET].isin(CLASS_LABELS)
    ].copy()

    # Convert features to numeric
    X = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_mask = X.notna().all(axis=1)

    df = df.loc[valid_mask].copy()
    X = X.loc[valid_mask].copy()

    y = df[TARGET].copy()

    print("Prepared records:", len(df))
    print("Number of features:", len(FEATURES))

    return df, X, y


# =====================================================
# 5. CHRONOLOGICAL TRAIN-TEST SPLIT
# =====================================================

def split_data(df, X, y):

    print("\nSplitting data chronologically...")

    # Split by unique timestamps so observations
    # from the same time across sites stay together.

    unique_times = np.sort(
        df["Timestamp"].unique()
    )

    if len(unique_times) < 2:
        raise ValueError(
            "Not enough unique timestamps to split data."
        )

    split_index = int(
        len(unique_times) * 0.80
    )

    split_index = max(
        1,
        min(split_index, len(unique_times) - 1)
    )

    cutoff_time = unique_times[split_index]

    train_mask = (
        df["Timestamp"] < cutoff_time
    )

    test_mask = (
        df["Timestamp"] >= cutoff_time
    )

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Training or testing set is empty."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "Training set contains fewer than two classes."
        )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    print("Training ends before:", cutoff_time)
    print("Testing starts from:", cutoff_time)

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


# =====================================================
# 6. TRAIN RANDOM FOREST CLASSIFIER
# =====================================================

def train_model(X_train, y_train):

    print("\nTraining Random Forest Classifier...")

    model = RandomForestClassifier(

        n_estimators=200,

        max_depth=12,

        min_samples_split=5,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed!")

    return model


# =====================================================
# 7. MODEL EVALUATION
# =====================================================

def evaluate_model(model, X_test, y_test):

    print("\nEvaluating model...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0
    )

    print("\n========================================")
    print("RANDOM FOREST CLASSIFICATION RESULTS")
    print("========================================")

    print(f"\nAccuracy: {accuracy:.4f}")

    print(
        f"Balanced Accuracy: {balanced_accuracy:.4f}"
    )

    print(f"Macro F1 Score: {macro_f1:.4f}")

    print("\nClassification Report:\n")

    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_LABELS,
        zero_division=0
    )

    print(report)

    # Save classification report
    report_dict = classification_report(
        y_test,
        predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_LABELS,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()

    report_df.to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    return predictions


# =====================================================
# 8. CONFUSION MATRIX
# =====================================================

def create_confusion_matrix(y_test, predictions):

    print("\nCreating confusion matrix...")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_LABELS
    )

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS
    )

    plt.title(
        "Random Forest - Signal Quality Confusion Matrix"
    )

    plt.xlabel("Predicted Quality")
    plt.ylabel("Actual Quality")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "classification_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("Confusion matrix saved!")


# =====================================================
# 9. FEATURE IMPORTANCE
# =====================================================

def create_feature_importance(model):

    print("\nCalculating feature importance...")

    importance_df = pd.DataFrame({

        "Feature": FEATURES,

        "Importance": model.feature_importances_

    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_df.to_csv(
        OUTPUT_DIR / "classification_feature_importance.csv",
        index=False
    )

    print("\nTOP 15 IMPORTANT FEATURES")

    print(
        importance_df.head(15).to_string(index=False)
    )

    top_features = importance_df.head(15)

    plt.figure(figsize=(10, 7))

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature"
    )

    plt.title(
        "Top 15 Features - Signal Quality Classification"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "classification_feature_importance.png",
        dpi=300
    )

    plt.close()

    print("Feature importance chart saved!")


# =====================================================
# 10. SAVE TRAINED MODEL
# =====================================================

def save_model(model):

    print("\nSaving trained model...")

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "classes": CLASS_LABELS
        },
        MODEL_PATH
    )

    print("Model saved successfully!")
    print("Model location:", MODEL_PATH)


# =====================================================
# 11. MAIN FUNCTION
# =====================================================

def main():

    print("=" * 60)
    print("MOBILE NETWORK SIGNAL QUALITY CLASSIFICATION")
    print("=" * 60)

    df = load_data()

    df, X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(
        df,
        X,
        y
    )

    model = train_model(
        X_train,
        y_train
    )

    predictions = evaluate_model(
        model,
        X_test,
        y_test
    )

    create_confusion_matrix(
        y_test,
        predictions
    )

    create_feature_importance(
        model
    )

    save_model(model)

    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
