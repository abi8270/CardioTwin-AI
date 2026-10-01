from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# BUILD DATASET
# ---------------------------------------------------------
def build_dataset():
    ehr = pd.read_csv(DATA_DIR / "ehr_static.csv")
    wearable = pd.read_csv(DATA_DIR / "wearable_timeseries.csv")
    labels = pd.read_csv(DATA_DIR / "risk_labels.csv")

    # Calculate wearable trends for each patient
    features = []

    for patient_id, group in wearable.groupby("patient_id"):
        group = group.sort_values("day")

        early = group.head(3)
        recent = group.tail(3)

        early_hrv = early["hrv_ms"].mean()
        recent_hrv = recent["hrv_ms"].mean()

        early_rhr = early["resting_hr_bpm"].mean()
        recent_rhr = recent["resting_hr_bpm"].mean()

        early_sleep = early["sleep_hours"].mean()
        recent_sleep = recent["sleep_hours"].mean()

        early_steps = early["steps"].mean()
        recent_steps = recent["steps"].mean()

        hrv_drop_pct = (
            (early_hrv - recent_hrv) / max(early_hrv, 1)
        )

        rhr_rise = recent_rhr - early_rhr

        sleep_drop = early_sleep - recent_sleep

        steps_drop_pct = (
            (early_steps - recent_steps) / max(early_steps, 1)
        )

        features.append(
            {
                "patient_id": patient_id,
                "hrv_drop_pct": hrv_drop_pct,
                "rhr_rise_bpm": rhr_rise,
                "sleep_drop_hrs": sleep_drop,
                "steps_drop_pct": steps_drop_pct,
            }
        )

    wearable_features = pd.DataFrame(features)

    # Merge EHR + wearable features + labels
    df = ehr.merge(
        wearable_features,
        on="patient_id",
        how="inner"
    )

    df = df.merge(
        labels[["patient_id", "risk_level"]],
        on="patient_id",
        how="inner"
    )

    return df


# ---------------------------------------------------------
# MAIN TRAINING
# ---------------------------------------------------------
def main():

    print("=" * 60)
    print("CardioTwin AI - Model Training")
    print("=" * 60)

    print("\nLoading dataset...")

    df = build_dataset()

    print(f"Dataset shape: {df.shape}")
    print("\nRisk distribution:")
    print(df["risk_level"].value_counts().sort_index())

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------

    feature_columns = [
        "age",
        "bmi",
        "systolic_bp",
        "cholesterol_mgdl",
        "diabetes_history",
        "smoker",
        "prior_cardiac_event",
        "family_history_cvd",
        "hrv_drop_pct",
        "rhr_rise_bpm",
        "sleep_drop_hrs",
        "steps_drop_pct",
    ]

    X = df[feature_columns]
    y = df["risk_level"]

    categorical_features = [
        "diabetes_history",
        "smoker",
        "prior_cardiac_event",
        "family_history_cvd",
    ]

    numeric_features = [
        "age",
        "bmi",
        "systolic_bp",
        "cholesterol_mgdl",
        "hrv_drop_pct",
        "rhr_rise_bpm",
        "sleep_drop_hrs",
        "steps_drop_pct",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # -----------------------------------------------------
    # Train/Test Split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\nTraining model...")

    pipeline.fit(X_train, y_train)

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    y_pred = pipeline.predict(X_test)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Low Risk",
                "Moderate Risk",
                "High Risk",
            ],
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # -----------------------------------------------------
    # Save Model
    # -----------------------------------------------------

    model_path = MODEL_DIR / "cardiotwin_model.joblib"

    joblib.dump(
        pipeline,
        model_path
    )

    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(f"\nModel file:")
    print(model_path)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()