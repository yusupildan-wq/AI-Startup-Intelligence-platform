"""Train a synthetic paired-counterfactual action-effect model."""

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from ml.train_trajectory_model import ACTIONS, encode_input, encode_state
from ml.world_generator import generate_learned_world
from world.engine import WorldEngine

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
ARTIFACT_PATH = MODEL_DIR / "causal_actions_v1.joblib"
METADATA_PATH = MODEL_DIR / "causal_actions_v1_metrics.json"
OUTCOMES = ("cash", "customers", "revenue", "product_quality")


def outcome_vector(state):
    company = state.companies["player"]
    return np.asarray([float(getattr(company, name)) for name in OUTCOMES])


def collect_paired_examples(world_start, world_end, states_per_world, seed):
    rng = np.random.default_rng(seed); X, y, groups = [], [], []
    for world_index in range(world_start, world_end):
        engine = WorldEngine(generate_learned_world(f"Causal world {world_index}", seed + world_index * 173))
        for state_index in range(states_per_world):
            state = deepcopy(engine.state); encoded = encode_state(state); before = outcome_vector(state)
            for action in ACTIONS:
                counterfactual = WorldEngine(deepcopy(state))
                counterfactual.advance(action)
                X.append(encode_input(encoded, action))
                y.append(outcome_vector(counterfactual.state) - before)
                groups.append(world_index * states_per_world + state_index)
            engine.advance(str(rng.choice(ACTIONS)))
            if not engine.state.companies["player"].alive:
                break
    return np.asarray(X), np.asarray(y), np.asarray(groups)


def train_and_save(seed=884):
    X_train, y_train, _ = collect_paired_examples(0, 70, 8, seed)
    X_test, y_test, groups = collect_paired_examples(70, 90, 8, seed + 10_000)
    model = ExtraTreesRegressor(n_estimators=128, max_depth=22, min_samples_leaf=2,
                                max_features=.85, n_jobs=-1, random_state=seed).fit(X_train, y_train)
    prediction = model.predict(X_test)
    state_size = X_test.shape[1] - len(ACTIONS)
    hold_index = ACTIONS.index("hold")
    effect_errors = []
    for group in np.unique(groups):
        mask = groups == group
        actual_group, predicted_group = y_test[mask], prediction[mask]
        effect_errors.append(np.abs(
            (actual_group - actual_group[hold_index]) -
            (predicted_group - predicted_group[hold_index])
        ))
    effect_errors = np.stack(effect_errors)
    metrics = {
        "algorithm": "paired-counterfactual S-learner with Extra Trees",
        "data_source": "synthetic_paired_civilization_interventions_v1",
        "identification": "Every action is applied to an identical copied pre-action world state with the same deterministic event seed.",
        "training_states": int(len(X_train) / len(ACTIONS)),
        "held_out_states": int(len(X_test) / len(ACTIONS)),
        "training_counterfactuals": len(X_train),
        "state_dimensions": state_size,
        "actions": list(ACTIONS),
        "outcomes": {
            name: {"outcome_mae": round(float(mean_absolute_error(y_test[:, index], prediction[:, index])), 4),
                   "outcome_r2": round(float(r2_score(y_test[:, index], prediction[:, index])), 4),
                   "treatment_effect_mae_vs_hold": round(float(effect_errors[:, :, index].mean()), 4)}
            for index, name in enumerate(OUTCOMES)
        },
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "actions": ACTIONS, "outcomes": OUTCOMES, "metrics": metrics},
                ARTIFACT_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))
