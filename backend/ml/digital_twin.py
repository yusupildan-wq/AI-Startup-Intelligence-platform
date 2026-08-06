"""Inference interface for the persisted digital-twin artifact."""

from functools import lru_cache

import joblib
import numpy as np

from ml.feature_store import build_temporal_features
from ml.train_digital_twin import ARTIFACT_PATH


@lru_cache(maxsize=1)
def load_digital_twin():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError("Digital-twin artifact is missing. Run: python -m ml.train_digital_twin")
    return joblib.load(ARTIFACT_PATH)


def predict_digital_twin(history):
    artifact = load_digital_twin()
    features = build_temporal_features(history)
    X = np.asarray([[features.get(name, np.nan) for name in artifact["feature_names"]]], dtype=np.float32)
    X = artifact["imputer"].transform(X)
    prediction = artifact["model"].predict(X)[0] * artifact["target_scale"] + artifact["target_mean"]
    values = dict(zip(artifact["targets"], prediction.tolist()))
    values["future_customer_count"] = max(0, round(values["future_customer_count"]))
    values["cash_exhaustion_probability"] = round(float(np.clip(values.pop("cash_exhausted"), 0, 1)), 3)
    for key in ("future_revenue", "future_cash_on_hand", "revenue_growth"):
        values[key] = round(float(values[key]), 2)
    return {
        "forecast_horizon_months": artifact["metrics"]["forecast_horizon_months"],
        "predictions": values,
        "model": {
            "algorithm": "Extra Trees multi-output regression",
            "feature_count": artifact["metrics"]["feature_count"],
            "training_rows": artifact["metrics"]["training_rows"],
            "data_source": artifact["metrics"]["data_source"],
            "metrics": {key: value for key, value in artifact["metrics"].items() if isinstance(value, dict)},
        },
    }
