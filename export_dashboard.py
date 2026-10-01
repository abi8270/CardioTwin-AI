from pathlib import Path
import json
import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

MODEL_PATH = MODEL_DIR / "cardiotwin_model.joblib"
OUTPUT_PATH = DASHBOARD_DIR / "dashboard_data.json"


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURE_COLS = [
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


# ============================================================
# LOAD DATA + MODEL
# ============================================================

print("Loading CardioTwin AI data...")

ehr = pd.read_csv(
    DATA_DIR / "ehr_static.csv"
)

wearable = pd.read_csv(
    DATA_DIR / "wearable_timeseries.csv"
)

labels = pd.read_csv(
    DATA_DIR / "risk_labels.csv"
)

model = joblib.load(
    MODEL_PATH
)

print("Data loaded successfully.")


# ============================================================
# BUILD WEARABLE TREND FEATURES
# ============================================================

def build_trend_features(group):

    group = group.sort_values("day")

    first = group.head(3)
    last = group.tail(3)

    # HRV
    hrv_first = first["hrv_ms"].mean()
    hrv_last = last["hrv_ms"].mean()

    hrv_drop_pct = max(
        0,
        (hrv_first - hrv_last)
        / max(hrv_first, 1)
        * 100
    )

    # Resting HR
    rhr_first = first["resting_hr_bpm"].mean()
    rhr_last = last["resting_hr_bpm"].mean()

    rhr_rise_bpm = (
        rhr_last - rhr_first
    )

    # Sleep
    sleep_first = first["sleep_hours"].mean()
    sleep_last = last["sleep_hours"].mean()

    sleep_drop_hrs = (
        sleep_first - sleep_last
    )

    # Steps
    steps_first = first["steps"].mean()
    steps_last = last["steps"].mean()

    steps_drop_pct = max(
        0,
        (steps_first - steps_last)
        / max(steps_first, 1)
        * 100
    )

    return {
        "hrv_drop_pct": float(hrv_drop_pct),
        "rhr_rise_bpm": float(rhr_rise_bpm),
        "sleep_drop_hrs": float(sleep_drop_hrs),
        "steps_drop_pct": float(steps_drop_pct),
    }


# ============================================================
# BUILD TREND DATA
# ============================================================

trend_rows = []

for patient_id, group in wearable.groupby(
    "patient_id"
):

    row = {
        "patient_id": patient_id
    }

    row.update(
        build_trend_features(group)
    )

    trend_rows.append(row)


trend_df = pd.DataFrame(
    trend_rows
)


# ============================================================
# MERGE EHR + WEARABLE + LABELS
# ============================================================

patients = (
    ehr
    .merge(
        trend_df,
        on="patient_id",
        how="left"
    )
    .merge(
        labels[["patient_id", "risk_level"]],
        on="patient_id",
        how="left"
    )
)


# ============================================================
# ONLY EXPORT FIRST 20 PATIENTS
# ============================================================

patients = patients.head(20).copy()


print(
    f"Dashboard patients selected: "
    f"{len(patients)}"
)


# ============================================================
# CURRENT MODEL PREDICTIONS
# ============================================================

X = patients[
    FEATURE_COLS
]

predictions = model.predict(
    X
)

probabilities = model.predict_proba(
    X
)

classes = list(
    model.classes_
)


# ============================================================
# PROBABILITY FORMATTER
# ============================================================

def get_probabilities(prob):

    result = {
        "0": 0.0,
        "1": 0.0,
        "2": 0.0,
    }

    for cls, value in zip(
        classes,
        prob
    ):

        result[
            str(int(cls))
        ] = float(value)

    return result


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

rf_model = model.named_steps["model"]

feature_importances = (
    rf_model.feature_importances_
)


try:

    transformed_features = (
        model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    clean_feature_names = [
        name
        .replace("num__", "")
        .replace("cat__", "")
        for name in transformed_features
    ]

except Exception:

    clean_feature_names = FEATURE_COLS


feature_importance_records = []

for name, value in zip(
    clean_feature_names,
    feature_importances
):

    feature_importance_records.append(
        {
            "feature": name,
            "importance": float(value),
        }
    )


feature_importance_records.sort(
    key=lambda x: x["importance"],
    reverse=True
)


# ============================================================
# PATIENT WEARABLE SERIES
# ============================================================

def patient_series(patient_id):

    group = wearable[
        wearable["patient_id"]
        == patient_id
    ].sort_values("day")

    series = []

    for _, row in group.iterrows():

        series.append(
            {
                "day": int(row["day"]),

                "hrv": float(
                    row["hrv_ms"]
                ),

                "rhr": float(
                    row["resting_hr_bpm"]
                ),

                "sleep": float(
                    row["sleep_hours"]
                ),

                "steps": int(
                    row["steps"]
                ),
            }
        )

    return series


# ============================================================
# BUILD PATIENT
# ============================================================

def build_patient(
    row,
    prediction,
    probability
):

    patient_id = int(
        row["patient_id"]
    )

    return {

        "patient_id":
            patient_id,

        "age":
            int(row["age"]),

        "sex":
            str(row["sex"]),

        "bmi":
            float(row["bmi"]),

        "systolic_bp":
            float(row["systolic_bp"]),

        "cholesterol_mgdl":
            float(
                row["cholesterol_mgdl"]
            ),

        "diabetes_history":
            str(
                row["diabetes_history"]
            ),

        "smoker":
            str(
                row["smoker"]
            ),

        "prior_cardiac_event":
            str(
                row["prior_cardiac_event"]
            ),

        "family_history_cvd":
            str(
                row["family_history_cvd"]
            ),

        "hrv_drop_pct":
            float(
                row["hrv_drop_pct"]
            ),

        "rhr_rise_bpm":
            float(
                row["rhr_rise_bpm"]
            ),

        "sleep_drop_hrs":
            float(
                row["sleep_drop_hrs"]
            ),

        "steps_drop_pct":
            float(
                row["steps_drop_pct"]
            ),

        "risk_level":
            int(prediction),

        "risk_probabilities":
            get_probabilities(
                probability
            ),

        "series":
            patient_series(
                patient_id
            ),
    }


# ============================================================
# ACTUAL RANDOM FOREST WHAT-IF
# ============================================================

def build_what_if(patient_row):

    base = patient_row.copy()

    # 5 sleep values
    sleep_changes = [
        -2,
        -1,
        0,
        1,
        2
    ]

    # 5 steps values
    steps_changes = [
        -3000,
        -1500,
        0,
        1500,
        3000
    ]

    # 5 RHR values
    rhr_changes = [
        -10,
        -5,
        0,
        5,
        10
    ]

    scenarios = []

    # ========================================================
    # 5 × 5 × 5 = 125 scenarios
    # ========================================================

    for sleep_change in sleep_changes:

        for steps_change in steps_changes:

            for rhr_change in rhr_changes:

                scenario = base.copy()

                # ------------------------------------------------
                # Sleep
                # ------------------------------------------------

                scenario[
                    "sleep_drop_hrs"
                ] = (
                    float(
                        base[
                            "sleep_drop_hrs"
                        ]
                    )
                    - sleep_change
                )

                # ------------------------------------------------
                # Steps
                # ------------------------------------------------

                scenario[
                    "steps_drop_pct"
                ] = max(
                    0,
                    float(
                        base[
                            "steps_drop_pct"
                        ]
                    )
                    - (
                        steps_change
                        / 100
                    )
                )

                # ------------------------------------------------
                # Resting HR
                # ------------------------------------------------

                scenario[
                    "rhr_rise_bpm"
                ] = (
                    float(
                        base[
                            "rhr_rise_bpm"
                        ]
                    )
                    + rhr_change
                )

                # ------------------------------------------------
                # Model input
                # ------------------------------------------------

                X_scenario = pd.DataFrame(
                    [{
                        feature:
                            scenario[feature]
                        for feature
                        in FEATURE_COLS
                    }]
                )

                # ------------------------------------------------
                # ACTUAL MODEL
                # ------------------------------------------------

                prediction = int(
                    model.predict(
                        X_scenario
                    )[0]
                )

                probability = (
                    model.predict_proba(
                        X_scenario
                    )[0]
                )

                scenarios.append(
                    {
                        "sleep_change":
                            sleep_change,

                        "steps_change":
                            steps_change,

                        "rhr_change":
                            rhr_change,

                        "prediction":
                            prediction,

                        "probabilities":
                            get_probabilities(
                                probability
                            ),
                    }
                )

    return scenarios


# ============================================================
# BUILD PATIENT RECORDS
# ============================================================

patient_records = []

print(
    "Generating actual Random Forest "
    "What-If scenarios..."
)

for index, (
    (_, row),
    prediction,
    probability
) in enumerate(
    zip(
        patients.iterrows(),
        predictions,
        probabilities
    ),
    start=1
):

    print(
        f"Processing patient "
        f"{index}/{len(patients)}..."
    )

    patient = build_patient(
        row,
        prediction,
        probability
    )

    # Actual Random Forest scenarios
    patient[
        "what_if_scenarios"
    ] = build_what_if(
        row
    )

    patient_records.append(
        patient
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

output = {

    "model": {

        "type":
            "RandomForestClassifier",

        "classes":
            [
                int(x)
                for x in classes
            ],

        "n_estimators":
            300,

        "feature_cols":
            FEATURE_COLS,
    },

    "feature_importances":
        feature_importance_records,

    "patients":
        patient_records,

    "meta": {

        "project":
            "CardioTwin AI",

        "description":
            (
                "Synthetic research PoC "
                "combining static EHR-like "
                "data with 14-day wearable "
                "trajectories."
            ),

        "prediction_target":
            (
                "Cardiovascular risk state "
                "and early-warning trajectory"
            ),

        "clinical_status":
            (
                "Research prototype. "
                "Not a medical diagnosis."
            ),

        "what_if_engine":
            "Actual Random Forest",

        "what_if_scenarios_per_patient":
            125,

        "dashboard_patients":
            len(patient_records),
    },
}


# ============================================================
# SAVE JSON
# ============================================================

DASHBOARD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )


# ============================================================
# DONE
# ============================================================

print()
print(
    "=========================================="
)

print(
    "CardioTwin AI Dashboard Export Complete!"
)

print(
    "=========================================="
)

print(
    f"Patients exported: "
    f"{len(patient_records)}"
)

print(
    f"Feature count: "
    f"{len(FEATURE_COLS)}"
)

print(
    "Model: RandomForestClassifier"
)

print(
    "What-If engine: Actual Random Forest"
)

print(
    "Scenarios per patient: 125"
)

print(
    f"Output: {OUTPUT_PATH}"
)

print(
    "=========================================="
)