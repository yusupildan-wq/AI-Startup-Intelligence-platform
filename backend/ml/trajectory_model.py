from functools import lru_cache

import joblib
import numpy as np

from ml.train_trajectory_model import ARTIFACT_PATH, encode_input, encode_state


@lru_cache(maxsize=1)
def load_trajectory_model():
    return joblib.load(ARTIFACT_PATH)


def _sample_residual(artifact, rng):
    mixture = artifact["residual_model"]
    component = int(rng.choice(len(mixture.weights_), p=mixture.weights_))
    return (rng.normal(mixture.means_[component], np.sqrt(mixture.covariances_[component]))
            * artifact["residual_scale"])


def generate_trajectories(state, action, horizon=12, paths=150, seed=2028):
    artifact = load_trajectory_model()
    if action not in artifact["actions"]: raise ValueError("Unknown trajectory action")
    rng = np.random.default_rng(seed); initial = encode_state(state)
    names = artifact["target_names"]
    cash_index, customers_index, revenue_index = (names.index("player_cash"), names.index("player_customers"),
                                                   names.index("player_revenue"))
    vectors = np.repeat(initial.reshape(1, -1), paths, axis=0)
    trajectories = np.empty((paths, horizon, 3), dtype=float)
    action_matrix = np.vstack([encode_input(np.zeros_like(initial), action)[len(initial):]] * paths)
    for month in range(horizon):
        model_input = np.column_stack([vectors, action_matrix])
        deltas = artifact["model"].predict(model_input)
        noise = np.vstack([_sample_residual(artifact, rng) for _ in range(paths)])
        vectors = vectors + deltas + noise
        vectors[:, cash_index] = np.maximum(-10_000_000, vectors[:, cash_index])
        vectors[:, customers_index] = np.maximum(0, vectors[:, customers_index])
        vectors[:, revenue_index] = np.maximum(0, vectors[:, revenue_index])
        trajectories[:, month, :] = vectors[:, [cash_index, customers_index, revenue_index]]
    timeline = []
    for month in range(horizon):
        timeline.append({
            "month": state.month + month + 1,
            "cash_p10": round(float(np.percentile(trajectories[:, month, 0], 10)), 2),
            "cash_median": round(float(np.percentile(trajectories[:, month, 0], 50)), 2),
            "cash_p90": round(float(np.percentile(trajectories[:, month, 0], 90)), 2),
            "customers_median": round(float(np.percentile(trajectories[:, month, 1], 50))),
            "revenue_median": round(float(np.percentile(trajectories[:, month, 2], 50)), 2),
            "survival_probability": round(float(np.mean(trajectories[:, month, 0] > 0)), 3),
        })
    return {"action": action, "horizon": horizon, "paths": paths, "timeline": timeline,
            "model": artifact["metrics"],
            "limitations": "Generated from synthetic civilization trajectories; uncertainty is experimental."}
