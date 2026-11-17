
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from feature_engineering import (
    load_data,
    clean_data,
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_change_features
)


# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "signal_classifier.pkl"


# ==========================================
# LOAD MODEL ONCE
# ==========================================

_model_bundle = None
_shap_explainer = None


def load_classifier():

    global _model_bundle

    if _model_bundle is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Classifier model not found: {MODEL_PATH}"
            )

        _model_bundle = joblib.load(MODEL_PATH)

        if "model" not in _model_bundle or "features" not in _model_bundle:
            raise ValueError("Invalid classifier model file.")

    return _model_bundle


# ==========================================
# PREPARE LATEST SITE-WISE FEATURES
# ==========================================

def prepare_latest_features():

    df = load_data()

    df = clean_data(df)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_change_features(df)

    df = df.sort_values(
        ["Site_ID", "Timestamp"]
    )

    latest_rows = (
        df.groupby("Site_ID", sort=False)
        .tail(1)
        .copy()
    )

    bundle = load_classifier()
    features = bundle["features"]

    missing = [
        feature for feature in features
        if feature not in latest_rows.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    X = latest_rows[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_mask = X.notna().all(axis=1)

    latest_rows = latest_rows.loc[valid_mask].copy()
    X = X.loc[valid_mask].copy()

    if X.empty:
        raise ValueError(
            "Not enough historical data to generate predictions."
        )

    return latest_rows, X


# ==========================================
# CREATE SHAP EXPLAINER
# ==========================================

def get_shap_explainer(model):

    global _shap_explainer

    if _shap_explainer is None:

        _shap_explainer = shap.TreeExplainer(model)

    return _shap_explainer


# ==========================================
# GENERATE LOCAL SHAP EXPLANATIONS
# ==========================================

def generate_shap_explanations(model, X, predictions):

    explainer = get_shap_explainer(model)

    shap_values = explainer.shap_values(X)

    # Support both older list output and newer ndarray output.
    if isinstance(shap_values, list):

        shap_values = np.stack(
            shap_values,
            axis=-1
        )

    else:

        shap_values = np.asarray(shap_values)

    classes = np.asarray(model.classes_)

    # Expected format:
    # (number of sites, number of features, number of classes)

    if shap_values.ndim != 3:
        raise ValueError(
            f"Unexpected SHAP dimensions: {shap_values.shape}"
        )

    if shap_values.shape[0] != len(X):
        raise ValueError("SHAP site count does not match input data.")

    if shap_values.shape[1] != X.shape[1]:
        raise ValueError("SHAP feature count does not match model features.")

    if shap_values.shape[2] != len(classes):
        raise ValueError("SHAP class count does not match model classes.")

    all_explanations = []

    for index in range(len(X)):

        predicted_class = predictions[index]

        class_matches = np.where(
            classes == predicted_class
        )[0]

        if len(class_matches) == 0:
            raise ValueError(
                f"Predicted class not found: {predicted_class}"
            )

        class_index = int(class_matches[0])

        # Select SHAP contributions for the predicted class.
        site_shap_values = shap_values[
            index, :, class_index
        ]

        # Select the five features with the largest
        # absolute contribution values.
        top_indices = np.argsort(
            np.abs(site_shap_values)
        )[::-1][:5]

        site_explanations = []

        for feature_index in top_indices:

            feature_name = X.columns[feature_index]

            feature_value = float(
                X.iloc[index, feature_index]
            )

            contribution = float(
                site_shap_values[feature_index]
            )

            if contribution > 0:
                direction = "supports_prediction"

            elif contribution < 0:
                direction = "opposes_prediction"

            else:
                direction = "neutral"

            site_explanations.append({

                "feature": str(feature_name),

                "feature_value": round(
                    feature_value, 4
                ),

                "shap_value": round(
                    contribution, 6
                ),

                "direction": direction

            })

        all_explanations.append(site_explanations)

    return all_explanations


# ==========================================
# GENERATE CLASSIFICATION PREDICTIONS
# ==========================================

def get_latest_predictions():

    bundle = load_classifier()

    model = bundle["model"]

    latest_rows, X = prepare_latest_features()

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    # Calculate SHAP explanations.
    # If explanation generation fails, predictions
    # will still be returned.

    explanation_error = None

    try:

        shap_explanations = generate_shap_explanations(
            model,
            X,
            predictions
        )

    except Exception as error:

        print("SHAP explanation error:", error)

        shap_explanations = [
            [] for _ in range(len(predictions))
        ]

        explanation_error = str(error)

    results = []

    for index, (_, row) in enumerate(latest_rows.iterrows()):

        probability_map = {

            str(class_name): round(
                float(probability), 4
            )

            for class_name, probability in zip(
                model.classes_,
                probabilities[index]
            )

        }

        results.append({

            "site_id": str(row["Site_ID"]),

            "timestamp": pd.Timestamp(
                row["Timestamp"]
            ).isoformat(),

            "predicted_quality": str(
                predictions[index]
            ),

            "probabilities": probability_map,

            "shap_explanations": shap_explanations[index]

        })

    response = {

        "success": True,

        "total_sites": len(results),

        "predictions": results,

        "explanations_available": explanation_error is None

    }

    if explanation_error is not None:

        response["explanation_error"] = explanation_error

    return response
