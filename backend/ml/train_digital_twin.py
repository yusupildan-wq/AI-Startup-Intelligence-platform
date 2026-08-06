"""Train and persist the first multi-output startup digital twin."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

from ml.feature_store import build_supervised_rows
from ml.synthetic_startups import generate_population

TARGETS = ("future_revenue", "future_customer_count", "future_cash_on_hand", "revenue_growth", "cash_exhausted")
ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "models" / "digital_twin_v1.joblib"
METADATA_PATH = Path(__file__).resolve().parents[1] / "models" / "digital_twin_v1_metrics.json"


def _matrix(histories, feature_names=None):
    rows = [row for history in histories for row in build_supervised_rows(history, horizon_months=3, minimum_history=6)]
    if feature_names is None:
        feature_names = tuple(sorted(rows[0]["features"]))
    X = np.asarray([[row["features"].get(name, np.nan) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.asarray([[row["labels"][target] for target in TARGETS] for row in rows], dtype=np.float64)
    return X, y, feature_names


def train_and_save(companies=160, months=24, seed=2026):
    histories = generate_population(companies=companies, months=months, seed=seed)
    split = round(companies * 0.8)
    X_train, y_train, feature_names = _matrix(histories[:split])
    X_test, y_test, _ = _matrix(histories[split:], feature_names)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    target_mean = y_train.mean(axis=0)
    target_scale = np.where(y_train.std(axis=0) < 1e-9, 1, y_train.std(axis=0))
    model = ExtraTreesRegressor(
        n_estimators=72, max_depth=18, min_samples_leaf=2, max_features=0.65,
        n_jobs=-1, random_state=seed,
    )
    model.fit(X_train, (y_train - target_mean) / target_scale)
    predictions = model.predict(X_test) * target_scale + target_mean

    metrics = {
        target: {
            "mae": round(float(mean_absolute_error(y_test[:, index], predictions[:, index])), 4),
            "r2": round(float(r2_score(y_test[:, index], predictions[:, index])), 4),
        }
        for index, target in enumerate(TARGETS)
    }
    metrics.update({
        "training_companies": split, "held_out_companies": companies - split,
        "training_rows": len(X_train), "test_rows": len(X_test),
        "feature_count": len(feature_names), "forecast_horizon_months": 3,
        "data_source": "synthetic_startup_population_v1",
    })
    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": model, "imputer": imputer, "feature_names": feature_names,
        "targets": TARGETS, "target_mean": target_mean, "target_scale": target_scale,
        "metrics": metrics,
    }, ARTIFACT_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))
