"""
CardioTwin AI - What-If Simulation Engine
============================================
Core idea: don't retrain anything. Take a patient's CURRENT engineered
feature vector, apply a hypothetical perturbation (e.g. "+1hr sleep/night",
"+2000 steps/day"), and re-run it through the already-trained model to get
a SIMULATED risk trajectory. This is what makes the dashboard a "digital
twin" rather than a static risk calculator -- the same model, a different
hypothetical input.
"""

import numpy as np
import pandas as pd
import joblib

DATA_DIR = "/home/claude/cardiotwin/data"
MODEL_DIR = "/home/claude/cardiotwin/models"

model = joblib.load(f"{MODEL_DIR}/cardiotwin_model.joblib")
scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.joblib")
model_name = joblib.load(f"{MODEL_DIR}/model_name.joblib")

RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High / Early Warning"}


def get_patient_features(patient_id: int) -> pd.DataFrame:
    """Rebuild one patient's fused feature row exactly as train_model.py does."""
    from train_model import engineer_timeseries_features  # reuse same logic

    ehr = pd.read_csv(f"{DATA_DIR}/ehr_static.csv")
    ts = pd.read_csv(f"{DATA_DIR}/wearable_timeseries.csv")

    ehr = ehr.copy()
    ehr["sex"] = (ehr["sex"] == "M").astype(int)
    for col in ["diabetes_history", "smoker", "prior_cardiac_event", "family_history_cvd"]:
        ehr[col] = ehr[col].astype(int)

    ts_feats = engineer_timeseries_features(ts[ts.patient_id == patient_id])
    row = ehr[ehr.patient_id == patient_id].merge(ts_feats, on="patient_id")
    return row[feature_cols]


def predict(features_row: pd.DataFrame) -> dict:
    X = features_row.values
    Xs = scaler.transform(X) if model_name == "logistic_regression" else X
    probs = model.predict_proba(Xs)[0]
    pred = int(np.argmax(probs))
    return dict(
        risk_level=pred,
        risk_label=RISK_LABELS[pred],
        prob_low=round(float(probs[0]), 3),
        prob_moderate=round(float(probs[1]), 3),
        prob_high=round(float(probs[2]), 3),
    )


def simulate_whatif(patient_id: int, sleep_delta_hrs=0.0, steps_delta=0,
                     rhr_delta_bpm=0.0, hrv_delta_ms=0.0) -> dict:
    """
    Apply a hypothetical shift to the patient's RECENT wearable trend
    features (the ones a lifestyle change would plausibly move), then
    re-score through the trained model.

    Note: we perturb the *_recent and *_slope features directly. This is a
    transparent, documented approximation -- shifting "recent" reflects a
    sustained behavior change, and softening the slope reflects that trend
    reversing rather than continuing to decline.
    """
    current = get_patient_features(patient_id)
    simulated = current.copy()

    if "sleep_recent" in simulated.columns:
        simulated["sleep_recent"] += sleep_delta_hrs
        simulated["sleep_slope"] = simulated["sleep_slope"].clip(lower=0) if sleep_delta_hrs > 0 else simulated["sleep_slope"]
    if "steps_recent" in simulated.columns:
        simulated["steps_recent"] += steps_delta
        simulated["steps_slope"] = simulated["steps_slope"].clip(lower=0) if steps_delta > 0 else simulated["steps_slope"]
    if "rhr_recent" in simulated.columns:
        simulated["rhr_recent"] += rhr_delta_bpm
    if "hrv_recent" in simulated.columns:
        simulated["hrv_recent"] += hrv_delta_ms

    current_pred = predict(current)
    simulated_pred = predict(simulated)

    return dict(
        patient_id=patient_id,
        current=current_pred,
        simulated=simulated_pred,
        intervention=dict(
            sleep_delta_hrs=sleep_delta_hrs,
            steps_delta=steps_delta,
            rhr_delta_bpm=rhr_delta_bpm,
            hrv_delta_ms=hrv_delta_ms,
        ),
    )


if __name__ == "__main__":
    # Demo: find a moderate/high risk patient and show a what-if scenario
    labels = pd.read_csv(f"{DATA_DIR}/risk_labels.csv")
    demo_patient = labels[labels.risk_level == 2].iloc[0].patient_id

    print(f"Demo patient: {int(demo_patient)}\n")
    result = simulate_whatif(
        int(demo_patient),
        sleep_delta_hrs=1.0,
        steps_delta=2000,
        rhr_delta_bpm=-3.0,
    )
    print("Current state:  ", result["current"])
    print("Simulated state:", result["simulated"])
    print("\nIntervention applied:", result["intervention"])
