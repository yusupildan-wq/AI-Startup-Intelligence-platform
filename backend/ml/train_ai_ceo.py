"""Offline fitted-Q training for the AI CEO policy."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from ml.ai_ceo_environment import ACTIONS, StartupEnvironment

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "models" / "ai_ceo_v1.joblib"
METADATA_PATH = Path(__file__).resolve().parents[1] / "models" / "ai_ceo_v1_metrics.json"


def encode(states, actions):
    one_hot = np.eye(len(ACTIONS), dtype=np.float32)[actions]
    return np.column_stack([states, one_hot])


def collect_transitions(episodes=900, seed=73):
    rng = np.random.default_rng(seed)
    records = []
    for episode in range(episodes):
        env = StartupEnvironment(seed + episode)
        state = env.reset()
        done = False
        while not done:
            action = int(rng.integers(len(ACTIONS)))
            next_state, reward, done, _ = env.step(action)
            records.append((state, action, reward, next_state, done))
            state = next_state
    states, actions, rewards, next_states, dones = zip(*records)
    return (np.asarray(states), np.asarray(actions), np.asarray(rewards),
            np.asarray(next_states), np.asarray(dones, dtype=bool))


def _greedy_action(model, state):
    repeated = np.repeat(state.reshape(1, -1), len(ACTIONS), axis=0)
    return int(np.argmax(model.predict(encode(repeated, np.arange(len(ACTIONS))))))


def evaluate(model, episodes=160, seed=9000, random_policy=False):
    rng = np.random.default_rng(seed)
    returns, survived, values = [], 0, []
    for episode in range(episodes):
        env = StartupEnvironment(seed + episode)
        state, total, done, info = env.reset(), 0.0, False, {}
        while not done:
            action = int(rng.integers(len(ACTIONS))) if random_policy else _greedy_action(model, state)
            state, reward, done, info = env.step(action)
            total += reward
        returns.append(total); survived += int(info["cash"] > 0); values.append(info["company_value"])
    return {
        "average_return": round(float(np.mean(returns)), 4),
        "survival_rate": round(survived / episodes, 4),
        "median_company_value": round(float(np.median(values)), 2),
    }


def train_and_save(iterations=9, gamma=0.97):
    states, actions, rewards, next_states, dones = collect_transitions()
    X = encode(states, actions)
    targets = rewards.copy()
    model = None
    for iteration in range(iterations):
        model = ExtraTreesRegressor(
            n_estimators=48, max_depth=20, min_samples_leaf=3,
            max_features=0.8, n_jobs=-1, random_state=100 + iteration,
        )
        model.fit(X, targets)
        next_q = []
        for action in range(len(ACTIONS)):
            next_q.append(model.predict(encode(next_states, np.full(len(next_states), action))))
        targets = rewards + gamma * (~dones) * np.max(np.column_stack(next_q), axis=1)

    policy_metrics = evaluate(model)
    random_metrics = evaluate(model, random_policy=True)
    metrics = {
        "algorithm": "offline fitted Q iteration with Extra Trees",
        "training_episodes": 900, "training_transitions": len(states),
        "q_iterations": iterations, "discount_factor": gamma,
        "actions": list(ACTIONS), "policy": policy_metrics, "random_baseline": random_metrics,
        "data_source": "synthetic_startup_environment_v1",
    }
    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics, "actions": ACTIONS}, ARTIFACT_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))
