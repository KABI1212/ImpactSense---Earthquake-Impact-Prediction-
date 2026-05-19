from __future__ import annotations

import math
from typing import Mapping

CORE_FEATURE_COLUMNS = [
    "Magnitude",
    "Depth",
    "Latitude",
    "Longitude",
    "Fault_Proximity",
]

ENGINEERED_FEATURE_COLUMNS = [
    "Magnitude",
    "Depth",
    "Latitude",
    "Longitude",
    "Fault_Proximity",
    "Depth_Category_Encoded",
    "Magnitude_Category_Encoded",
    "Mag_Depth_Ratio",
    "Mag_Depth_Product",
    "Log_Depth",
    "Mag_Squared",
    "Geo_Cluster",
    "Location_Risk_Score",
    "Fault_Category_Encoded",
]

FIELD_LABELS = {
    "magnitude": "Magnitude",
    "depth": "Depth",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "fault_proximity": "Fault Proximity",
}

PHYSICAL_LIMITS = {
    "magnitude": (0.0, 10.0),
    "depth": (0.0, 800.0),
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
    "fault_proximity": (0.0, 120.0),
}

DEFAULT_FORM_VALUES = {
    "magnitude": 6.1,
    "depth": 42.0,
    "latitude": 18.52,
    "longitude": 73.86,
    "fault_proximity": 24.0,
}

SCENARIO_PRESETS = {
    "Custom": DEFAULT_FORM_VALUES,
    "Urban Fault Line": {
        "magnitude": 7.4,
        "depth": 18.0,
        "latitude": 34.05,
        "longitude": -118.24,
        "fault_proximity": 8.0,
    },
    "Offshore Deep Event": {
        "magnitude": 6.2,
        "depth": 315.0,
        "latitude": -12.4,
        "longitude": 130.85,
        "fault_proximity": 42.0,
    },
    "Mountain Interior": {
        "magnitude": 5.8,
        "depth": 92.0,
        "latitude": 27.72,
        "longitude": 85.32,
        "fault_proximity": 17.0,
    },
    "Severe Coastal Shock": {
        "magnitude": 8.3,
        "depth": 24.0,
        "latitude": 35.68,
        "longitude": 140.03,
        "fault_proximity": 5.0,
    },
}

ACTION_GUIDANCE = {
    "Low": "Monitor official updates and keep standard response teams on standby.",
    "Elevated": "Activate regional coordination, verify infrastructure status, and prepare public alerts.",
    "High": "Stand up emergency operations, validate transport corridors, and pre-position medical support.",
    "Severe": "Escalate immediately, trigger emergency protocols, and prioritize dense population zones.",
}

DEPTH_ENCODINGS = {
    "Deep": 0,
    "Intermediate": 1,
    "Shallow": 2,
}

MAGNITUDE_ENCODINGS = {
    "Light": 0,
    "Major": 1,
    "Minor": 2,
    "Moderate": 3,
    "Strong": 4,
}

FAULT_ENCODINGS = {
    "Far": 0,
    "Medium": 1,
    "Near": 2,
}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def categorize_depth(depth: float) -> str:
    if depth <= 70:
        return "Shallow"
    if depth <= 300:
        return "Intermediate"
    return "Deep"


def categorize_magnitude(magnitude: float) -> str:
    if magnitude <= 3.9:
        return "Minor"
    if magnitude <= 4.9:
        return "Light"
    if magnitude <= 5.9:
        return "Moderate"
    if magnitude <= 6.9:
        return "Strong"
    return "Major"


def categorize_fault_proximity(fault_proximity: float) -> str:
    if fault_proximity <= 20:
        return "Near"
    if fault_proximity <= 50:
        return "Medium"
    return "Far"


def encode_depth_category(label: str) -> int:
    return DEPTH_ENCODINGS[label]


def encode_magnitude_category(label: str) -> int:
    return MAGNITUDE_ENCODINGS[label]


def encode_fault_category(label: str) -> int:
    return FAULT_ENCODINGS[label]


def validate_inputs(
    payload: Mapping[str, float],
    observed_ranges: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for field_name, label in FIELD_LABELS.items():
        if field_name not in payload:
            errors.append(f"{label} is required.")
            continue

        value = payload[field_name]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"{label} must be a valid number.")
            continue

        minimum, maximum = PHYSICAL_LIMITS[field_name]
        if value < minimum or value > maximum:
            errors.append(f"{label} must stay within {minimum:g} to {maximum:g}.")

    if errors:
        return {"errors": errors, "warnings": warnings}

    if payload["magnitude"] < 2.5:
        warnings.append("Magnitude is below the model's observed training window and may be less reliable.")
    if payload["depth"] > 700:
        warnings.append("Depth is above the model's observed training window and will be treated as a deep event.")
    if payload["fault_proximity"] == 0:
        warnings.append("Zero fault proximity is an extreme case. Treat the prediction as a stress test.")

    if observed_ranges:
        for field_name, bounds in observed_ranges.items():
            if field_name not in payload:
                continue
            observed_min, observed_max = bounds
            value = float(payload[field_name])
            if value < observed_min or value > observed_max:
                label = FIELD_LABELS[field_name]
                warnings.append(
                    f"{label} is outside the observed training range ({observed_min:.2f} to {observed_max:.2f})."
                )

    return {"errors": errors, "warnings": warnings}


def classify_risk_level(probability: float, risk_score: float) -> str:
    probability_percent = clamp(probability * 100.0, 0.0, 100.0)
    score = clamp(risk_score, 0.0, 100.0)
    combined = max(probability_percent, score)

    if combined >= 75 or (probability_percent >= 65 and score >= 55):
        return "Severe"
    if combined >= 55 or (probability_percent >= 50 and score >= 40):
        return "High"
    if combined >= 35 or (probability_percent >= 30 and score >= 25):
        return "Elevated"
    return "Low"


def build_driver_summary(
    payload: Mapping[str, float],
    probability: float,
    risk_score: float,
) -> list[str]:
    drivers: list[str] = []

    if payload["magnitude"] >= 7.0:
        drivers.append("Major magnitude band is pushing the model toward a stronger impact signal.")
    elif payload["magnitude"] >= 6.0:
        drivers.append("Strong magnitude band is materially increasing the impact score.")

    if payload["depth"] <= 70:
        drivers.append("Shallow depth raises the surface-impact profile.")
    elif payload["depth"] >= 300:
        drivers.append("Deep focus reduces some surface effects but still matters at higher magnitudes.")

    if payload["fault_proximity"] <= 20:
        drivers.append("Near-fault location is a high-sensitivity factor in the trained feature set.")
    elif payload["fault_proximity"] >= 50:
        drivers.append("Greater distance from major faults is reducing the final risk level.")

    if risk_score >= 70:
        drivers.append("The regression score sits in the upper band of historical scenarios in this dataset.")
    elif probability >= 0.6:
        drivers.append("Classification confidence is above 60%, which supports escalation.")

    if not drivers:
        drivers.append("Inputs are sitting near the middle of the training distribution, so the model is less alarmed.")

    return drivers


def build_prediction_headline(level: str, probability: float, risk_score: float) -> str:
    probability_percent = clamp(probability * 100.0, 0.0, 100.0)
    score = clamp(risk_score, 0.0, 100.0)
    return (
        f"{level} risk outlook with a {probability_percent:.1f}% high-impact probability "
        f"and an impact score of {score:.1f}/100."
    )


def normalize_user_identity(identity: str) -> str:
    """Turn a raw email / username string into a display name."""
    cleaned = " ".join(identity.strip().split())
    if not cleaned:
        return "Guest User"

    if "@" in cleaned:
        cleaned = cleaned.split("@", 1)[0]

    cleaned = cleaned.replace(".", " ").replace("_", " ").replace("-", " ")
    normalized = " ".join(part for part in cleaned.split() if part)
    return normalized.title() if normalized else "Guest User"


def validate_public_login(identity: str, password: str) -> tuple[bool, str]:
    """Accept ANY non-empty username + non-empty password."""
    username = identity.strip()
    secret = password.strip()

    if not username:
        return False, "Please enter your name or email address."

    if not secret:
        return False, "Please enter a password."

    # Open access — every user is admitted.
    return True, ""
