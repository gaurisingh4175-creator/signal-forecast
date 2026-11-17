
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

from pathlib import Path


# =====================================================
# 1. FILE PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_PATH = DATA_DIR / "engineered_signal_data.csv"
MODEL_PATH = MODEL_DIR / "signal_classifier.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD TRAINED CLASSIFICATION MODEL
# =====================================================

def load_model():

    print("\nLoading trained Random Forest model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run signal_classifier.py first."
        )

    saved_data = joblib.load(MODEL_PATH)

    model = saved_data["model"]
    features = saved_data["features"]

    print("Model loaded successfully!")
    print("Model classes:", model.classes_)
    print("Number of features:", len(features))

    return model, features


# =====================================================
# 3. LOAD ENGINEERED DATA
# =====================================================

def load_data():

    print("\nLoading engineered dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    print("Dataset loaded:", df.shape)

    return df


# =====================================================
# 4. PREPARE FEATURES
# =====================================================

def prepare_features(df, features):

    print("\nPreparing features for SHAP...")

    required = features + ["Timestamp"]

    df = df.dropna(
        subset=required
    ).copy()

    df = df.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    X = df[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_mask = X.notna().all(axis=1)

    df = df.loc[valid_mask].reset_index(drop=True)
    X = X.loc[valid_mask].reset_index(drop=True)

    print("Prepared records:", len(X))

    return df, X


# =====================================================
# 5. CHRONOLOGICAL TRAIN-TEST SPLIT
# =====================================================

def chronological_split(df, X):

    print("\nCreating chronological split...")

    unique_times = np.sort(
        df["Timestamp"].unique()
    )

    split_index = int(
        len(unique_times) * 0.80
    )

    split_index = max(
        1,
        min(split_index, len(unique_times) - 1)
    )

    cutoff = unique_times[split_index]

    train_mask = (
        df["Timestamp"] < cutoff
    )

    test_mask = (
        df["Timestamp"] >= cutoff
    )

    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Chronological split produced an empty dataset."
        )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    return X_train, X_test


# =====================================================
# 6. CREATE SHAP EXPLAINER
# =====================================================

def calculate_shap_values(model, X_train, X_test):

    print("\nInitializing SHAP TreeExplainer...")

    # Use a small training sample as background data
    background = X_train.sample(
        n=min(100, len(X_train)),
        random_state=42
    )

    # Explain a sample of test observations
    X_explain = X_test.head(
        min(100, len(X_test))
    ).copy()

    explainer = shap.TreeExplainer(
        model,
        data=background
    )

    print("Calculating SHAP values...")

    shap_values = explainer.shap_values(
        X_explain,
        check_additivity=False
    )

    print("SHAP values calculated successfully!")

    return explainer, shap_values, X_explain


# =====================================================
# 7. NORMALIZE SHAP OUTPUT FORMAT
# =====================================================

def get_class_shap_values(shap_values, class_index, n_classes):

    # Newer SHAP versions may return:
    # (samples, features, classes)

    if isinstance(shap_values, np.ndarray):

        if shap_values.ndim == 3:
            return shap_values[:, :, class_index]

        if shap_values.ndim == 2:
            return shap_values

    # Older SHAP versions may return a list:
    # one (samples, features) array per class

    if isinstance(shap_values, list):

        return np.asarray(
            shap_values[class_index]
        )

    raise ValueError(
        "Unexpected SHAP output format."
    )


def get_all_class_values(shap_values, n_classes):

    if isinstance(shap_values, np.ndarray):

        if shap_values.ndim == 3:
            return shap_values

        if shap_values.ndim == 2:
            return shap_values[:, :, np.newaxis]

    if isinstance(shap_values, list):

        return np.stack(
            shap_values,
            axis=2
        )

    raise ValueError(
        "Cannot interpret SHAP output."
    )


# =====================================================
# 8. GLOBAL FEATURE IMPORTANCE
# =====================================================

def create_global_importance(
    shap_values,
    features,
    n_classes
):

    print("\nCalculating global SHAP importance...")

    all_values = get_all_class_values(
        shap_values,
        n_classes
    )

    # Average absolute SHAP values over observations
    # and classes.

    mean_importance = np.mean(
        np.abs(all_values),
        axis=(0, 2)
    )

    importance_df = pd.DataFrame({

        "Feature": features,

        "Mean_Absolute_SHAP": mean_importance

    })

    importance_df = importance_df.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    importance_df.to_csv(
        OUTPUT_DIR / "shap_global_importance.csv",
        index=False
    )

    print("\nTOP 15 SHAP IMPORTANT FEATURES")

    print(
        importance_df.head(15).to_string(index=False)
    )

    plt.figure(figsize=(10, 7))

    top_features = importance_df.head(15).sort_values(
        "Mean_Absolute_SHAP"
    )

    plt.barh(
        top_features["Feature"],
        top_features["Mean_Absolute_SHAP"]
    )

    plt.xlabel("Mean Absolute SHAP Value")

    plt.title(
        "Global SHAP Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_global_importance.png",
        dpi=300
    )

    plt.close()

    print("Global SHAP importance saved!")

    return importance_df


# =====================================================
# 9. SHAP SUMMARY / BEESWARM PLOT
# =====================================================

def create_summary_plot(
    shap_values,
    X_explain,
    model
):

    print("\nCreating SHAP summary plot...")

    classes = list(model.classes_)

    # Explain the Good class if it exists.
    # Otherwise, explain the first learned class.

    if "Good" in classes:
        selected_class = "Good"
    else:
        selected_class = classes[0]

    class_index = classes.index(
        selected_class
    )

    class_values = get_class_shap_values(
        shap_values,
        class_index,
        len(classes)
    )

    plt.figure(figsize=(11, 8))

    shap.summary_plot(
        class_values,
        X_explain,
        max_display=15,
        show=False
    )

    plt.title(
        f"SHAP Summary Plot - {selected_class} Class"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_summary_beeswarm.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"SHAP summary plot saved for {selected_class}!"
    )

    return selected_class, class_index


# =====================================================
# 10. SHAP WATERFALL PLOT
# =====================================================

def create_waterfall_plot(
    explainer,
    shap_values,
    X_explain,
    model,
    class_index,
    selected_class,
    features
):

    print("\nCreating individual SHAP waterfall plot...")

    class_values = get_class_shap_values(
        shap_values,
        class_index,
        len(model.classes_)
    )

    # Explain the first test observation
    row_index = 0

    # Select the expected value for the chosen class
    expected = np.asarray(
        explainer.expected_value
    )

    if expected.ndim == 0:
        base_value = float(expected)
    else:
        base_value = float(
            expected.reshape(-1)[class_index]
        )

    explanation = shap.Explanation(

        values=class_values[row_index],

        base_values=base_value,

        data=X_explain.iloc[row_index].values,

        feature_names=features

    )

    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False
    )

    plt.title(
        f"Individual SHAP Explanation - {selected_class}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_individual_waterfall.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Individual SHAP waterfall saved!")


# =====================================================
# 11. CLASS-WISE SHAP IMPORTANCE
# =====================================================

def create_classwise_importance(
    shap_values,
    features,
    model
):

    print("\nCalculating class-wise SHAP importance...")

    classes = list(model.classes_)

    for class_index, class_name in enumerate(classes):

        class_values = get_class_shap_values(
            shap_values,
            class_index,
            len(classes)
        )

        importance = np.mean(
            np.abs(class_values),
            axis=0
        )

        class_df = pd.DataFrame({

            "Feature": features,

            "Mean_Absolute_SHAP": importance

        })

        class_df = class_df.sort_values(
            "Mean_Absolute_SHAP",
            ascending=False
        )

        safe_name = str(class_name).lower()

        class_df.to_csv(
            OUTPUT_DIR / f"shap_{safe_name}_importance.csv",
            index=False
        )

    print("Class-wise SHAP importance saved!")


# =====================================================
# 12. MAIN FUNCTION
# =====================================================

def main():

    print("=" * 60)
    print("SHAP EXPLAINABLE AI - NETWORK QUALITY CLASSIFICATION")
    print("=" * 60)

    model, features = load_model()

    df = load_data()

    df, X = prepare_features(
        df,
        features
    )

    X_train, X_test = chronological_split(
        df,
        X
    )

    explainer, shap_values, X_explain = calculate_shap_values(
        model,
        X_train,
        X_test
    )

    create_global_importance(
        shap_values,
        features,
        len(model.classes_)
    )

    selected_class, class_index = create_summary_plot(
        shap_values,
        X_explain,
        model
    )

    create_waterfall_plot(
        explainer,
        shap_values,
        X_explain,
        model,
        class_index,
        selected_class,
        features
    )

    create_classwise_importance(
        shap_values,
        features,
        model
    )

    print("\n" + "=" * 60)
    print("SHAP EXPLANATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nSelected explanation class:", selected_class)


if __name__ == "__main__":
    main()
