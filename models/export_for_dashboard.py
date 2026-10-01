"""
Export everything the dashboard needs as static JSON:
  - logistic regression coefficients + intercepts (so the dashboard can
    run the REAL trained model client-side, not a mock)
  - scaler mean/scale (StandardScaler params)
  - feature column order
  - a handful of demo patients: EHR, 14-day time-series, engineered
    features, and current prediction
  - feature importances for the "contributing factors" panel
"""
import json
import numpy as np
import pandas as pd
import joblib
from train_model import engineer_timeseries_features

DATA_DIR = "/home/claude/cardiotwin/data"
MODEL_DIR = "/home/claude/cardiotwin/models"
OUT_PATH = "/home/claude/cardiotwin/dashboard/model_export.json"

model = joblib.load(f"{MODEL_DIR}/cardiotwin_model.joblib")
scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.joblib")
model_name = joblib.load(f"{MODEL_DIR}/model_name.joblib")
importances = pd.read_csv(f"{MODEL_DIR}/feature_importances.csv", index_col=0).iloc[:, 0]

ehr = pd.read_csv(f"{DATA_DIR}/ehr_static.csv")
ts = pd.read_csv(f"{DATA_DIR}/wearable_timeseries.csv")
labels = pd.read_csv(f"{DATA_DIR}/risk_labels.csv")

assert model_name == "logistic_regression", "Export script assumes logistic regression was selected"

# Pick 6 demo patients: 2 low, 2 moderate, 2 high risk
demo_ids = []
for level in [0, 1, 2]:
    demo_ids += labels[labels.risk_level == level].patient_id.head(2).tolist()

ehr_enc = ehr.copy()
ehr_enc["sex"] = (ehr_enc["sex"] == "M").astype(int)
for col in ["diabetes_history", "smoker", "prior_cardiac_event", "family_history_cvd"]:
    ehr_enc[col] = ehr_enc[col].astype(int)

patients = []
for pid in demo_ids:
    pid = int(pid)
    ts_feats = engineer_timeseries_features(ts[ts.patient_id == pid])
    row = ehr_enc[ehr_enc.patient_id == pid].merge(ts_feats, on="patient_id")
    feat_vec = row[feature_cols].iloc[0].to_dict()

    ehr_raw = ehr[ehr.patient_id == pid].iloc[0].to_dict()
    ts_raw = ts[ts.patient_id == pid][["day", "hrv_ms", "resting_hr_bpm", "sleep_hours", "steps"]].to_dict(orient="records")

    patients.append(dict(
        patient_id=pid,
        ehr=ehr_raw,
        timeseries=ts_raw,
        features=feat_vec,
        true_risk_level=int(labels[labels.patient_id == pid].risk_level.iloc[0]),
    ))

export = dict(
    model=dict(
        coef=model.coef_.tolist(),         # shape (3, n_features)
        intercept=model.intercept_.tolist(),
        classes=model.classes_.tolist(),
    ),
    scaler=dict(
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
    ),
    feature_cols=feature_cols,
    feature_importances={k: round(float(v), 4) for k, v in importances.head(10).items()},
    patients=patients,
)

with open(OUT_PATH, "w") as f:
    json.dump(export, f, indent=1, default=str)

print(f"Exported {len(patients)} demo patients + model to {OUT_PATH}")
print(f"File size: {len(json.dumps(export)) / 1024:.1f} KB")
