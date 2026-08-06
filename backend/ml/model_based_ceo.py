"""Beam-search planning over the learned generative civilization transition model."""

import numpy as np

from ml.trajectory_model import _sample_residual, load_trajectory_model
from ml.train_trajectory_model import encode_input, encode_state


def _indices(artifact):
    names = artifact["target_names"]
    return {name: names.index(name) for name in (
        "player_cash", "player_customers", "player_revenue", "player_product_quality"
    )}


def _constrain(vectors, indices):
    vectors[:, indices["player_cash"]] = np.maximum(-10_000_000, vectors[:, indices["player_cash"]])
    vectors[:, indices["player_customers"]] = np.maximum(0, vectors[:, indices["player_customers"]])
    vectors[:, indices["player_revenue"]] = np.maximum(0, vectors[:, indices["player_revenue"]])
    vectors[:, indices["player_product_quality"]] = np.clip(vectors[:, indices["player_product_quality"]], 0, 1)
    return vectors


def _value(vectors, indices):
    cash = vectors[:, indices["player_cash"]]
    revenue = vectors[:, indices["player_revenue"]]
    customers = vectors[:, indices["player_customers"]]
    quality = vectors[:, indices["player_product_quality"]]
    return cash + revenue * 10 + customers * 350 + quality * 100_000 - (cash <= 0) * 5_000_000


def _best_sequence_for_first_action(initial, first_action, artifact, horizon, beam_width):
    actions = artifact["actions"]; indices = _indices(artifact)
    beam = [(initial, (first_action,))]
    first_input = encode_input(initial, first_action).reshape(1, -1)
    beam[0] = (_constrain((initial + artifact["model"].predict(first_input)).reshape(1, -1), indices)[0],
               (first_action,))
    for _ in range(1, horizon):
        vectors = np.vstack([item[0] for item in beam])
        inputs = np.vstack([encode_input(vector, action) for vector in vectors for action in actions])
        parents = np.repeat(vectors, len(actions), axis=0)
        next_vectors = _constrain(parents + artifact["model"].predict(inputs), indices)
        sequences = [sequence + (action,) for _, sequence in beam for action in actions]
        best = np.argsort(_value(next_vectors, indices))[::-1][:beam_width]
        beam = [(next_vectors[index], sequences[index]) for index in best]
    return beam[0][1]


def _evaluate_sequence(initial, sequence, artifact, paths, rng):
    indices = _indices(artifact); vectors = np.repeat(initial.reshape(1, -1), paths, axis=0)
    for action in sequence:
        inputs = np.vstack([encode_input(vector, action) for vector in vectors])
        mean_delta = artifact["model"].predict(inputs)
        noise = np.vstack([_sample_residual(artifact, rng) for _ in range(paths)])
        vectors = _constrain(vectors + mean_delta + noise, indices)
    cash = vectors[:, indices["player_cash"]]; revenue = vectors[:, indices["player_revenue"]]
    customers = vectors[:, indices["player_customers"]]
    return {
        "survival_probability": float(np.mean(cash > 0)),
        "cash_p10": float(np.percentile(cash, 10)),
        "cash_median": float(np.median(cash)),
        "revenue_median": float(np.median(revenue)),
        "customers_median": float(np.median(customers)),
    }


def plan_actions(state, horizon=12, beam_width=10, paths=60, risk_aversion=.65, seed=932):
    artifact = load_trajectory_model(); initial = encode_state(state); rng = np.random.default_rng(seed)
    current_cash = max(abs(initial[_indices(artifact)["player_cash"]]), 100_000)
    current_revenue = max(initial[_indices(artifact)["player_revenue"]], 1)
    current_customers = max(initial[_indices(artifact)["player_customers"]], 1)
    candidates = []
    for action in artifact["actions"]:
        sequence = _best_sequence_for_first_action(initial, action, artifact, horizon, beam_width)
        outcome = _evaluate_sequence(initial, sequence, artifact, paths, rng)
        downside_cash = outcome["cash_p10"] / current_cash
        median_cash = outcome["cash_median"] / current_cash
        revenue_ratio = outcome["revenue_median"] / current_revenue
        customer_ratio = outcome["customers_median"] / current_customers
        score = (55 * outcome["survival_probability"] + 12 * median_cash +
                 10 * revenue_ratio + 5 * customer_ratio + 18 * risk_aversion * downside_cash)
        candidates.append({"first_action": action, "planned_sequence": list(sequence),
                           "risk_adjusted_score": round(float(score), 3),
                           **{key: round(value, 3 if "probability" in key else 2)
                              for key, value in outcome.items()}})
    candidates.sort(key=lambda item: item["risk_adjusted_score"], reverse=True)
    for rank, candidate in enumerate(candidates, 1): candidate["rank"] = rank
    return {"recommendation": candidates[0], "action_comparison": candidates,
            "search": {"horizon": horizon, "actions": len(artifact["actions"]),
                       "beam_width_per_first_action": beam_width, "stochastic_paths_per_plan": paths,
                       "risk_aversion": risk_aversion},
            "model": artifact["metrics"],
            "limitations": "Plans optimize a learned synthetic world model; model error compounds across the horizon."}
