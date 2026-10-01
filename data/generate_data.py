"""
CardioTwin AI - Synthetic Data Generator
========================================

Generates:
  1. Static EHR features per patient
     - demographics
     - medical history
     - laboratory-style features

  2. 14-day dynamic wearable time-series
     - resting heart rate
     - HRV
     - sleep
     - steps

  3. Rule-based ground-truth risk labels derived from the patient's
     own trajectory rather than fixed population thresholds.

All data is synthetic.
No real patient data is used.
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# generate_data.py is inside:
# CardioTwin-AI/repo/data/generate_data.py
#
# Therefore:
# parents[0] = data
# parents[1] = repo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Create data folder automatically if it does not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RNG = np.random.default_rng(42)

N_PATIENTS = 1200
N_DAYS = 14


# ============================================================
# RISK PROFILE
# ============================================================

@dataclass
class RiskProfile:
    name: str
    weight: float

    # HRV
    hrv_baseline_range: tuple
    hrv_daily_drift: tuple

    # Resting heart rate
    rhr_baseline_range: tuple
    rhr_daily_drift: tuple

    # Sleep
    sleep_baseline_range: tuple
    sleep_daily_drift: tuple

    # Steps
    steps_baseline_range: tuple
    steps_daily_drift: tuple


# ============================================================
# SYNTHETIC POPULATION PROFILES
# ============================================================

PROFILES = [

    RiskProfile(
        name="stable",
        weight=0.55,

        hrv_baseline_range=(45, 70),
        hrv_daily_drift=(0.05, 0.4),

        rhr_baseline_range=(58, 72),
        rhr_daily_drift=(-0.02, 0.3),

        sleep_baseline_range=(6.5, 8.0),
        sleep_daily_drift=(0.0, 0.15),

        steps_baseline_range=(6000, 11000),
        steps_daily_drift=(0, 150),
    ),

    RiskProfile(
        name="emerging_risk",
        weight=0.30,

        hrv_baseline_range=(35, 55),
        hrv_daily_drift=(-0.6, 0.4),

        rhr_baseline_range=(68, 82),
        rhr_daily_drift=(0.4, 0.3),

        sleep_baseline_range=(5.5, 7.0),
        sleep_daily_drift=(-0.08, 0.15),

        steps_baseline_range=(3500, 7000),
        steps_daily_drift=(-80, 150),
    ),

    RiskProfile(
        name="high_risk",
        weight=0.15,

        hrv_baseline_range=(20, 38),
        hrv_daily_drift=(-1.4, 0.5),

        rhr_baseline_range=(78, 95),
        rhr_daily_drift=(0.9, 0.4),

        sleep_baseline_range=(4.0, 6.0),
        sleep_daily_drift=(-0.15, 0.2),

        steps_baseline_range=(1500, 4500),
        steps_daily_drift=(-150, 150),
    ),
]


# ============================================================
# SYNTHETIC EHR GENERATION
# ============================================================

def sample_ehr(profile_name: str) -> dict:
    """
    Generate static/historical EHR-style information.

    The profile influences the distributions so that the EHR stream
    contains meaningful but imperfect signal.
    """

    # Age
    age = int(RNG.normal(52, 13))
    age = max(22, min(85, age))

    # Sex
    sex = RNG.choice(["M", "F"])

    # Profile-dependent risk adjustment
    risk_bump = {
        "stable": 0,
        "emerging_risk": 1,
        "high_risk": 2,
    }[profile_name]

    # BMI
    bmi = RNG.normal(
        24 + risk_bump * 2.5,
        3.2
    )

    # Systolic blood pressure
    systolic_bp = RNG.normal(
        118 + risk_bump * 12,
        9
    )

    # Total cholesterol
    cholesterol = RNG.normal(
        180 + risk_bump * 25,
        20
    )

    # Medical history
    diabetes = RNG.random() < (
        0.08 + 0.18 * risk_bump
    )

    smoker = RNG.random() < (
        0.10 + 0.15 * risk_bump
    )

    prior_cardiac_event = RNG.random() < (
        0.02 + 0.12 * risk_bump
    )

    family_history = RNG.random() < (
        0.15 + 0.10 * risk_bump
    )

    return {
        "age": age,
        "sex": sex,
        "bmi": round(bmi, 1),
        "systolic_bp": round(systolic_bp, 1),
        "cholesterol_mgdl": round(cholesterol, 1),
        "diabetes_history": diabetes,
        "smoker": smoker,
        "prior_cardiac_event": prior_cardiac_event,
        "family_history_cvd": family_history,
    }


# ============================================================
# DYNAMIC WEARABLE DATA GENERATION
# ============================================================

def sample_timeseries(profile: RiskProfile) -> pd.DataFrame:
    """
    Generate 14 days of wearable-style time-series data.

    Each patient gets:
      - individual baseline
      - daily trend/drift
      - random noise

    This allows the model to learn trajectory-related information.
    """

    # Patient-specific starting values
    hrv0 = RNG.uniform(
        *profile.hrv_baseline_range
    )

    rhr0 = RNG.uniform(
        *profile.rhr_baseline_range
    )

    sleep0 = RNG.uniform(
        *profile.sleep_baseline_range
    )

    steps0 = RNG.uniform(
        *profile.steps_baseline_range
    )

    rows = []

    for day in range(N_DAYS):

        # HRV
        hrv = (
            hrv0
            + profile.hrv_daily_drift[0] * day
            + RNG.normal(
                0,
                profile.hrv_daily_drift[1]
            )
        )

        # Resting heart rate
        rhr = (
            rhr0
            + profile.rhr_daily_drift[0] * day
            + RNG.normal(
                0,
                profile.rhr_daily_drift[1]
            )
        )

        # Sleep
        sleep = (
            sleep0
            + profile.sleep_daily_drift[0] * day
            + RNG.normal(
                0,
                profile.sleep_daily_drift[1]
            )
        )

        # Steps
        steps = (
            steps0
            + profile.steps_daily_drift[0] * day
            + RNG.normal(
                0,
                profile.steps_daily_drift[1]
            )
        )

        rows.append(
            {
                "day": day,
                "hrv_ms": max(
                    8,
                    round(hrv, 1)
                ),
                "resting_hr_bpm": max(
                    45,
                    round(rhr, 1)
                ),
                "sleep_hours": max(
                    2.0,
                    round(sleep, 2)
                ),
                "steps": max(
                    200,
                    int(steps)
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# TRAJECTORY-BASED RISK LABEL
# ============================================================

def compute_trajectory_label(
    ts: pd.DataFrame
) -> dict:
    """
    Calculate deterioration based on the patient's own trajectory.

    First 3 days:
        Personal baseline

    Last 3 days:
        Recent state

    Risk levels:
        0 = Low
        1 = Moderate
        2 = High
    """

    # Personal baseline
    base = ts.iloc[:3].mean()

    # Recent state
    recent = ts.iloc[-3:].mean()

    # --------------------------------------------------------
    # HRV deterioration
    # --------------------------------------------------------

    hrv_drop_pct = (
        (base.hrv_ms - recent.hrv_ms)
        / max(base.hrv_ms, 1)
    )

    # --------------------------------------------------------
    # Resting HR deterioration
    # --------------------------------------------------------

    rhr_rise = (
        recent.resting_hr_bpm
        - base.resting_hr_bpm
    )

    # --------------------------------------------------------
    # Sleep deterioration
    # --------------------------------------------------------

    sleep_drop = (
        base.sleep_hours
        - recent.sleep_hours
    )

    # --------------------------------------------------------
    # Activity deterioration
    # --------------------------------------------------------

    steps_drop_pct = (
        (base.steps - recent.steps)
        / max(base.steps, 1)
    )

    # --------------------------------------------------------
    # Composite deterioration score
    # --------------------------------------------------------

    score = (
        2.2 * hrv_drop_pct
        + 0.05 * rhr_rise
        + 0.35 * sleep_drop
        + 0.8 * steps_drop_pct
    )

    # Add realistic noise
    noisy_score = (
        score
        + RNG.normal(0, 0.45)
    )

    # --------------------------------------------------------
    # Risk classification
    # --------------------------------------------------------

    if noisy_score >= 1.8:
        level = 2

    elif noisy_score >= 0.4:
        level = 1

    else:
        level = 0

    return {
        "deterioration_score": round(
            float(score),
            3
        ),

        "risk_level": level,

        "hrv_drop_pct": round(
            float(hrv_drop_pct),
            3
        ),

        "rhr_rise_bpm": round(
            float(rhr_rise),
            2
        ),

        "sleep_drop_hrs": round(
            float(sleep_drop),
            2
        ),

        "steps_drop_pct": round(
            float(steps_drop_pct),
            3
        ),
    }


# ============================================================
# COMPLETE DATASET GENERATION
# ============================================================

def generate_dataset(
    n_patients: int = N_PATIENTS
):
    """
    Generate the complete synthetic CardioTwin dataset.
    """

    profile_names = [
        profile.name
        for profile in PROFILES
    ]

    weights = [
        profile.weight
        for profile in PROFILES
    ]

    ehr_rows = []
    ts_rows = []
    label_rows = []

    # --------------------------------------------------------
    # Generate each patient
    # --------------------------------------------------------

    for pid in range(
        1,
        n_patients + 1
    ):

        # Select population profile
        profile_name = RNG.choice(
            profile_names,
            p=weights
        )

        profile = next(
            p
            for p in PROFILES
            if p.name == profile_name
        )

        # ----------------------------------------------------
        # EHR
        # ----------------------------------------------------

        ehr = sample_ehr(
            profile_name
        )

        ehr["patient_id"] = pid

        ehr_rows.append(ehr)

        # ----------------------------------------------------
        # Wearable time-series
        # ----------------------------------------------------

        ts = sample_timeseries(
            profile
        )

        ts["patient_id"] = pid

        ts_rows.append(ts)

        # ----------------------------------------------------
        # Risk label
        # ----------------------------------------------------

        label = compute_trajectory_label(
            ts
        )

        label["patient_id"] = pid

        # Used only for debugging/evaluation.
        # The ML model must NOT use this feature.
        label["true_profile"] = profile_name

        label_rows.append(label)

    # --------------------------------------------------------
    # Convert to DataFrames
    # --------------------------------------------------------

    ehr_df = pd.DataFrame(
        ehr_rows
    )

    ts_df = pd.concat(
        ts_rows,
        ignore_index=True
    )

    label_df = pd.DataFrame(
        label_rows
    )

    return (
        ehr_df,
        ts_df,
        label_df
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CardioTwin AI - Synthetic Dataset Generator")
    print("=" * 60)

    print(
        f"\nGenerating data for {N_PATIENTS} patients..."
    )

    # Generate datasets
    ehr_df, ts_df, label_df = generate_dataset()

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    ehr_path = (
        DATA_DIR
        / "ehr_static.csv"
    )

    wearable_path = (
        DATA_DIR
        / "wearable_timeseries.csv"
    )

    label_path = (
        DATA_DIR
        / "risk_labels.csv"
    )

    ehr_df.to_csv(
        ehr_path,
        index=False
    )

    ts_df.to_csv(
        wearable_path,
        index=False
    )

    label_df.to_csv(
        label_path,
        index=False
    )

    # --------------------------------------------------------
    # Output summary
    # --------------------------------------------------------

    print("\nDATA GENERATION COMPLETE")
    print("-" * 60)

    print(
        f"EHR records      : {len(ehr_df)}"
    )

    print(
        f"Wearable records  : {len(ts_df)}"
    )

    print(
        f"Risk labels       : {len(label_df)}"
    )

    print("\nFiles created:")

    print(
        f"1. {ehr_path}"
    )

    print(
        f"2. {wearable_path}"
    )

    print(
        f"3. {label_path}"
    )

    # --------------------------------------------------------
    # Risk distribution
    # --------------------------------------------------------

    print("\nRisk level distribution:")

    print(
        label_df[
            "risk_level"
        ]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Sample EHR
    # --------------------------------------------------------

    print("\nSample EHR row:")

    print(
        ehr_df.iloc[0]
    )

    # --------------------------------------------------------
    # Sample risk label
    # --------------------------------------------------------

    print("\nSample risk label:")

    print(
        label_df.iloc[0]
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)