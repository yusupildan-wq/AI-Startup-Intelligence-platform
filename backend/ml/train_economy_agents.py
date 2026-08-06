"""Train investor, competitor-policy, and macro-regime models."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
ARTIFACT_PATH = MODEL_DIR / "economy_agents_v1.joblib"
METADATA_PATH = MODEL_DIR / "economy_agents_v1_metrics.json"
COMPETITOR_ACTIONS = ("hold", "raise_price", "lower_price", "increase_marketing",
                      "decrease_marketing", "hire_engineer", "hire_sales",
                      "fundraise", "invest_in_product", "enter_new_market")
MACRO_REGIMES = ("recession", "stable", "expansion", "funding_boom")


def _investor_data(n, rng):
    growth = rng.uniform(-.5, 1.5, n); revenue = rng.lognormal(10.2, 1.2, n)
    cash = rng.lognormal(12.5, 1.1, n); runway = rng.uniform(0, 36, n)
    quality = rng.beta(2.5, 1.8, n); reputation = rng.beta(2.2, 2, n)
    sentiment = rng.beta(2, 2, n); rate = rng.uniform(.01, .12, n); ownership = rng.uniform(.35, 1, n)
    score = (-3 + 1.8 * growth + .000008 * revenue + .045 * runway + 1.4 * quality
             + .8 * reputation + 1.4 * sentiment - 9 * rate + .5 * ownership + rng.normal(0, .6, n))
    probability = 1 / (1 + np.exp(-score)); funded = rng.binomial(1, probability)
    amount = np.maximum(50_000, revenue * 12 * rng.uniform(1.5, 6, n) * (.5 + sentiment))
    amount *= funded
    X = np.column_stack([growth, revenue, cash, runway, quality, reputation, sentiment, rate, ownership])
    return X, funded, amount


def _competitor_data(n, rng):
    cash_ratio = rng.uniform(-.2, 3, n); growth = rng.uniform(-.5, 1, n)
    relative_price = rng.uniform(.5, 1.8, n); relative_quality = rng.uniform(-.7, .7, n)
    marketing_intensity = rng.uniform(0, .5, n); technical_debt = rng.beta(2, 2, n)
    market = rng.uniform(.55, 1.45, n); sentiment = rng.uniform(0, 1, n)
    headcount_pressure = rng.uniform(0, 1, n); competition = rng.uniform(0, 1, n)
    X = np.column_stack([cash_ratio, growth, relative_price, relative_quality, marketing_intensity,
                         technical_debt, market, sentiment, headcount_pressure, competition])
    utility = np.column_stack([
        np.full(n, .2), relative_quality + relative_price - .6, 1.1 - relative_price,
        market + cash_ratio * .25 - marketing_intensity, .8 - cash_ratio - market,
        technical_debt + cash_ratio * .15, growth + market + competition * .3,
        1.2 - cash_ratio + sentiment, technical_debt + .4 * relative_quality,
        market + sentiment - competition,
    ]) + rng.normal(0, .35, (n, len(COMPETITOR_ACTIONS)))
    return X, np.argmax(utility, axis=1)


def _macro_data(n, rng):
    demand = rng.uniform(.55, 1.45, n); rate = rng.uniform(.01, .12, n)
    unemployment = rng.uniform(.025, .12, n); sentiment = rng.uniform(0, 1, n)
    inflation = rng.uniform(0, .12, n); funding_growth = rng.uniform(-.8, 1.2, n)
    formation_growth = rng.uniform(-.4, .7, n); public_returns = rng.uniform(-.5, .7, n)
    scores = np.column_stack([
        1.7 * unemployment + 8 * rate + 4 * inflation - demand - sentiment,
        1.2 - np.abs(demand - 1) - np.abs(sentiment - .5),
        demand + sentiment + formation_growth + public_returns,
        1.5 * sentiment + funding_growth + public_returns - 5 * rate,
    ]) + rng.normal(0, .3, (n, len(MACRO_REGIMES)))
    return np.column_stack([demand, rate, unemployment, sentiment, inflation,
                            funding_growth, formation_growth, public_returns]), np.argmax(scores, axis=1)


def train_and_save(n=45_000, seed=818):
    rng = np.random.default_rng(seed)
    investor_X, funded, amounts = _investor_data(n, rng)
    X_train, X_test, y_train, y_test, amount_train, amount_test = train_test_split(
        investor_X, funded, amounts, test_size=.22, random_state=seed, stratify=funded)
    investor = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=25, random_state=seed).fit(X_train, y_train)
    investor_probability = investor.predict_proba(X_test)[:, 1]
    positive = amount_train > 0
    valuation = ExtraTreesRegressor(n_estimators=100, max_depth=16, min_samples_leaf=3,
                                    n_jobs=-1, random_state=seed).fit(X_train[positive], amount_train[positive])
    positive_test = amount_test > 0
    investor_metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, investor_probability)), 4),
        "accuracy": round(float(accuracy_score(y_test, investor.predict(X_test))), 4),
        "funding_rate": round(float(np.mean(funded)), 4),
        "amount_mae": round(float(mean_absolute_error(amount_test[positive_test], valuation.predict(X_test[positive_test]))), 2),
        "amount_r2": round(float(r2_score(amount_test[positive_test], valuation.predict(X_test[positive_test]))), 4),
    }

    competitor_X, competitor_y = _competitor_data(n, rng)
    X_train, X_test, y_train, y_test = train_test_split(competitor_X, competitor_y, test_size=.22, random_state=seed, stratify=competitor_y)
    competitor = RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=4,
                                        class_weight="balanced_subsample", n_jobs=-1, random_state=seed).fit(X_train, y_train)
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    competitor_metrics = {
        "accuracy": round(float(accuracy_score(y_test, competitor.predict(X_test))), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, competitor.predict(X_test))), 4),
        "majority_baseline": round(float(accuracy_score(y_test, dummy.predict(X_test))), 4),
    }

    macro_X, macro_y = _macro_data(n, rng)
    X_train, X_test, y_train, y_test = train_test_split(macro_X, macro_y, test_size=.22, random_state=seed, stratify=macro_y)
    macro = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=28,
                                           class_weight="balanced", random_state=seed).fit(X_train, y_train)
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    macro_metrics = {
        "accuracy": round(float(accuracy_score(y_test, macro.predict(X_test))), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, macro.predict(X_test))), 4),
        "majority_baseline": round(float(accuracy_score(y_test, dummy.predict(X_test))), 4),
    }
    metrics = {"data_source": "synthetic_economy_agents_v1", "rows_per_system": n,
               "investor": investor_metrics, "competitor_policy": competitor_metrics, "macro_regime": macro_metrics}
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"investor": investor, "valuation": valuation, "competitor": competitor,
                 "macro": macro, "metrics": metrics}, ARTIFACT_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))
