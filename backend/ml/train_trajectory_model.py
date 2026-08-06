"""Train an action-conditioned generative transition model for civilization futures."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.mixture import GaussianMixture

from ml.world_generator import generate_learned_world
from world.engine import WorldEngine
from world.events import ACTION_TYPES

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
ARTIFACT_PATH = MODEL_DIR / "trajectory_model_v1.joblib"
METADATA_PATH = MODEL_DIR / "trajectory_model_v1_metrics.json"
ACTIONS = tuple(sorted(ACTION_TYPES))
COMPANY_METRICS = ("cash", "customers", "price", "marketing", "engineers", "salespeople",
                   "support", "product_quality", "technical_debt", "reputation", "revenue")
MACRO_METRICS = ("demand_multiplier", "interest_rate", "unemployment_rate", "venture_sentiment")


def encode_state(state):
    values = []
    for company_id in ("player", "competitor_alpha", "competitor_beta"):
        company = state.companies[company_id]
        values.extend(float(getattr(company, metric)) for metric in COMPANY_METRICS)
    values.extend(float(getattr(state.macro, metric)) for metric in MACRO_METRICS)
    values.extend([float(state.investors.available_capital), float(state.investors.risk_appetite),
                   float(state.investors.valuation_multiple), state.month / 36])
    return np.asarray(values, dtype=np.float64)


def encode_input(state_vector, action):
    one_hot = np.zeros(len(ACTIONS), dtype=np.float64)
    one_hot[ACTIONS.index(action)] = 1
    return np.concatenate([state_vector, one_hot])


def collect_transitions(world_start, world_end, months, seed):
    rng = np.random.default_rng(seed); X, y = [], []
    for world_index in range(world_start, world_end):
        world = generate_learned_world(f"Training world {world_index}", seed + world_index * 101)
        engine = WorldEngine(world)
        for _ in range(months):
            before = encode_state(engine.state)
            action = str(rng.choice(ACTIONS))
            engine.advance(action)
            after = encode_state(engine.state)
            X.append(encode_input(before, action)); y.append(after - before)
            if not engine.state.companies["player"].alive: break
    return np.asarray(X), np.asarray(y)


def train_and_save(seed=515):
    X_train, y_train = collect_transitions(0, 100, 12, seed)
    X_test, y_test = collect_transitions(100, 130, 12, seed + 9000)
    model = ExtraTreesRegressor(n_estimators=96, max_depth=20, min_samples_leaf=2,
                                max_features=.8, n_jobs=-1, random_state=seed).fit(X_train, y_train)
    prediction = model.predict(X_test)
    residuals = y_train - model.predict(X_train)
    residual_scale = np.where(residuals.std(axis=0) < 1e-6, 1, residuals.std(axis=0))
    residual_model = GaussianMixture(n_components=8, covariance_type="diag", max_iter=180,
                                     random_state=seed).fit(residuals / residual_scale)
    target_names = tuple(
        [f"{company}_{metric}" for company in ("player", "competitor_alpha", "competitor_beta") for metric in COMPANY_METRICS]
        + list(MACRO_METRICS) + ["investor_capital", "investor_risk", "valuation_multiple", "month"]
    )
    key_indices = [target_names.index(name) for name in ("player_cash", "player_customers", "player_revenue",
                                                           "player_product_quality", "demand_multiplier")]
    key_metrics = {}
    for index in key_indices:
        key_metrics[target_names[index]] = {
            "mae": round(float(mean_absolute_error(y_test[:, index], prediction[:, index])), 4),
            "r2": round(float(r2_score(y_test[:, index], prediction[:, index])), 4),
        }
    rng = np.random.default_rng(seed + 2)
    sample_count = 80
    selected = np.arange(min(800, len(X_test)))
    samples = []
    for _ in range(sample_count):
        component = rng.choice(len(residual_model.weights_), size=len(selected), p=residual_model.weights_)
        noise = np.vstack([rng.normal(residual_model.means_[item], np.sqrt(residual_model.covariances_[item]))
                           for item in component]) * residual_scale
        samples.append(prediction[selected] + noise)
    samples = np.asarray(samples)
    lower, upper = np.percentile(samples, [10, 90], axis=0)
    coverage = float(np.mean((y_test[selected] >= lower) & (y_test[selected] <= upper)))
    metrics = {
        "algorithm": "Extra Trees transition mean + 8-component residual mixture",
        "data_source": "generated_civilization_trajectories_v1",
        "training_worlds": 100, "held_out_worlds": 30,
        "training_transitions": len(X_train), "test_transitions": len(X_test),
        "state_dimensions": len(encode_state(generate_learned_world("shape", 1))),
        "action_count": len(ACTIONS), "interval_80_coverage": round(coverage, 4),
        "targets": key_metrics,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "residual_model": residual_model, "residual_scale": residual_scale,
                 "actions": ACTIONS, "target_names": target_names, "metrics": metrics}, ARTIFACT_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))
