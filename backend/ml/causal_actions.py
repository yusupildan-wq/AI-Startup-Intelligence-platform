from functools import lru_cache

import joblib
import numpy as np

from ml.train_causal_actions import ARTIFACT_PATH
from ml.train_trajectory_model import encode_input, encode_state


@lru_cache(maxsize=1)
def load_causal_actions():
    return joblib.load(ARTIFACT_PATH)


def estimate_action_effects(state):
    artifact = load_causal_actions(); encoded = encode_state(state)
    matrix = np.vstack([encode_input(encoded, action) for action in artifact["actions"]])
    outcomes = artifact["model"].predict(matrix)
    hold = outcomes[artifact["actions"].index("hold")]
    effects = []
    for index, action in enumerate(artifact["actions"]):
        delta = outcomes[index] - hold
        effects.append({
            "action": action,
            "effects_vs_hold": {name: round(float(delta[item]), 4)
                                for item, name in enumerate(artifact["outcomes"])},
        })
    return {"baseline": "hold", "effects": effects, "model": artifact["metrics"],
            "limitations": "Causal effects are identified inside paired synthetic worlds, not from real company interventions."}
