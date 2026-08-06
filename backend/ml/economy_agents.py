from functools import lru_cache

import joblib
import numpy as np

from ml.train_economy_agents import ARTIFACT_PATH, COMPETITOR_ACTIONS, MACRO_REGIMES


@lru_cache(maxsize=1)
def load_economy_agents():
    artifact = joblib.load(ARTIFACT_PATH)
    for name in ("valuation", "competitor"):
        if hasattr(artifact[name], "n_jobs"):
            artifact[name].n_jobs = 1
    return artifact


def investor_offer(features):
    artifact = load_economy_agents(); X = np.asarray([features], dtype=float)
    probability = float(artifact["investor"].predict_proba(X)[0, 1])
    amount = max(0, float(artifact["valuation"].predict(X)[0]))
    return probability, amount


def competitor_action(features):
    index = int(load_economy_agents()["competitor"].predict([features])[0])
    return COMPETITOR_ACTIONS[index]


def macro_regime(features):
    index = int(load_economy_agents()["macro"].predict([features])[0])
    return MACRO_REGIMES[index]
