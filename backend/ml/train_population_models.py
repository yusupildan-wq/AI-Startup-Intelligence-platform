"""Train behavioral models used by customer and employee populations."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
ARTIFACT_PATH = MODEL_DIR / "population_models_v1.joblib"
METADATA_PATH = MODEL_DIR / "population_models_v1_metrics.json"


def _customer_data(n, rng):
    budget = rng.lognormal(np.log(130), 0.8, n)
    price_sensitivity = rng.beta(2.2, 1.8, n)
    quality_preference = rng.beta(2.0, 1.5, n)
    switching_cost = rng.beta(2.0, 2.5, n)
    price = rng.uniform(20, 350, n)
    quality = rng.beta(3, 2, n)
    reputation = rng.beta(2.5, 2, n)
    competitor_utility = rng.normal(0.25, 0.7, n)
    demand = rng.uniform(0.55, 1.45, n)
    utility = (
        2.5 * quality * quality_preference + 1.2 * reputation + 0.6 * switching_cost
        - 2.2 * price_sensitivity * np.maximum(price / np.maximum(budget, 1) - 0.35, 0)
        + 0.5 * demand + rng.normal(0, 0.45, n)
    )
    label = (utility > np.maximum(competitor_utility, 0.2)).astype(int)
    return np.column_stack([budget, price_sensitivity, quality_preference, switching_cost,
                            price, quality, reputation, competitor_utility, demand]), label


def _employee_data(n, rng):
    salary_ratio = rng.uniform(0.55, 1.5, n)
    morale = rng.beta(2.4, 1.8, n)
    burnout = rng.beta(1.8, 2.2, n)
    tenure = rng.uniform(0, 72, n)
    company_growth = rng.uniform(-0.35, 0.8, n)
    runway = rng.uniform(0, 36, n)
    manager_quality = rng.beta(2.5, 1.7, n)
    market_jobs = rng.uniform(0.2, 1.0, n)
    score = (-2.2 + 2.6 * burnout - 1.8 * morale - 1.0 * manager_quality
             - 0.9 * salary_ratio - 0.035 * tenure - 0.7 * company_growth
             + 1.4 * market_jobs + 0.9 * (runway < 6) + rng.normal(0, 0.45, n))
    probability = 1 / (1 + np.exp(-score))
    return np.column_stack([salary_ratio, morale, burnout, tenure, company_growth,
                            runway, manager_quality, market_jobs]), rng.binomial(1, probability)


def _adoption_data(n, rng):
    segment_need = rng.beta(2, 2, n)
    feature_fit = rng.beta(2.5, 1.8, n)
    usability = rng.beta(3, 1.8, n)
    awareness = rng.beta(1.8, 2.4, n)
    switching_cost = rng.beta(2, 2, n)
    price_change = rng.uniform(-0.3, 0.5, n)
    peer_adoption = rng.beta(1.6, 3, n)
    score = (-2.3 + 2.8 * segment_need * feature_fit + 1.6 * usability + 1.1 * awareness
             + 1.5 * peer_adoption - 1.0 * switching_cost - 1.7 * np.maximum(price_change, 0)
             + rng.normal(0, 0.4, n))
    probability = 1 / (1 + np.exp(-score))
    return np.column_stack([segment_need, feature_fit, usability, awareness, switching_cost,
                            price_change, peer_adoption]), rng.binomial(1, probability)


def _fit_evaluate(X, y, seed, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=seed, stratify=y)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    return model, {
        "training_rows": len(X_train), "test_rows": len(X_test),
        "accuracy": round(float(accuracy_score(y_test, prediction)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, prediction)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        "majority_baseline_accuracy": round(float(accuracy_score(y_test, dummy.predict(X_test))), 4),
        "positive_rate": round(float(np.mean(y)), 4),
    }


def train_and_save(n=40_000, seed=610):
    rng = np.random.default_rng(seed)
    customer, customer_metrics = _fit_evaluate(*_customer_data(n, rng), seed,
        HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=25, learning_rate=0.08, random_state=seed))
    employee, employee_metrics = _fit_evaluate(*_employee_data(n, rng), seed + 1,
        RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=8,
                               class_weight="balanced_subsample", n_jobs=-1, random_state=seed))
    adoption, adoption_metrics = _fit_evaluate(*_adoption_data(n, rng), seed + 2,
        HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=23, learning_rate=0.08, random_state=seed))
    metrics = {
        "data_source": "synthetic_behavioral_population_v1", "generated_rows_per_model": n,
        "customer_choice": customer_metrics, "employee_attrition": employee_metrics,
        "product_adoption": adoption_metrics,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"customer_choice": customer, "employee_attrition": employee,
                 "product_adoption": adoption, "metrics": metrics}, ARTIFACT_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))
