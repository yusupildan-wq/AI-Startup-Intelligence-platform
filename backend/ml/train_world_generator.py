"""Train a probabilistic generator over complete startup civilizations."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
ARTIFACT_PATH = MODEL_DIR / "world_generator_v1.joblib"
METADATA_PATH = MODEL_DIR / "world_generator_v1_metrics.json"

COMPANY_FIELDS = ("cash", "customers", "price", "marketing", "engineers", "salespeople",
                  "support", "product_quality", "technical_debt", "reputation")
FEATURE_NAMES = tuple(
    [f"company_{index}_{field}" for index in range(3) for field in COMPANY_FIELDS]
    + ["demand_multiplier", "interest_rate", "unemployment_rate", "venture_sentiment",
       "investor_capital", "investor_risk", "valuation_multiple",
       "smb_population", "midmarket_population", "enterprise_population",
       "smb_budget", "midmarket_budget", "enterprise_budget"]
)


def generate_training_worlds(n=35_000, seed=1337):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        archetype = int(rng.integers(4))
        regime = int(rng.choice(4, p=[.18, .48, .24, .10]))
        stage_scale = rng.lognormal([-.2, .4, .8, .2][archetype], .5)
        quality_center = [.52, .63, .74, .68][archetype]
        price_center = [45, 95, 260, 140][archetype]
        companies = []
        for company in range(3):
            strength = stage_scale * rng.lognormal(0, .35)
            headcount = max(1, round(strength * rng.uniform(3, 12)))
            companies.extend([
                rng.lognormal(np.log(180_000 + strength * 350_000), .55),
                rng.lognormal(np.log(40 + strength * 180), .5),
                rng.lognormal(np.log(price_center), .25),
                rng.lognormal(np.log(2_000 + strength * 7_000), .45),
                max(1, round(headcount * rng.uniform(.35, .65))),
                max(0, round(headcount * rng.uniform(.1, .35))),
                max(0, round(headcount * rng.uniform(.05, .2))),
                np.clip(rng.normal(quality_center, .12), .08, .98),
                np.clip(rng.beta(2, 3) + (.12 if archetype == 3 else 0), .02, .95),
                np.clip(rng.normal(.45 + strength * .06, .14), .05, .98),
            ])
        macro_centers = ((.7, .08, .085, .2), (1.0, .05, .05, .5),
                         (1.2, .035, .038, .7), (1.3, .025, .035, .92))[regime]
        demand, rate, unemployment, sentiment = [rng.normal(value, abs(value) * .08 + .01) for value in macro_centers]
        rows.append(companies + [
            demand, rate, unemployment, sentiment,
            rng.lognormal(np.log(30_000_000 + sentiment * 120_000_000), .45),
            np.clip(rng.normal(sentiment, .15), .05, .95), rng.uniform(3, 12) * (.7 + sentiment),
            rng.lognormal(np.log(8_000), .35), rng.lognormal(np.log(2_500), .35),
            rng.lognormal(np.log(600), .3), rng.lognormal(np.log(75), .2),
            rng.lognormal(np.log(240), .22), rng.lognormal(np.log(900), .25),
        ])
    return np.asarray(rows, dtype=np.float64)


def _sample_mixture(model, scaler, n, seed):
    rng = np.random.default_rng(seed)
    components = rng.choice(len(model.weights_), size=n, p=model.weights_)
    scaled = np.vstack([
        rng.multivariate_normal(model.means_[component], model.covariances_[component])
        if model.covariance_type == "full"
        else rng.normal(model.means_[component], np.sqrt(model.covariances_[component]))
        for component in components
    ])
    return scaler.inverse_transform(scaled)


def _validity(rows):
    valid = np.ones(len(rows), dtype=bool)
    for company in range(3):
        offset = company * len(COMPANY_FIELDS)
        valid &= rows[:, offset] > 0
        valid &= rows[:, offset + 1] >= 0
        valid &= rows[:, offset + 2] > 0
        valid &= (rows[:, offset + 7] >= 0) & (rows[:, offset + 7] <= 1)
        valid &= (rows[:, offset + 8] >= 0) & (rows[:, offset + 8] <= 1)
    return float(np.mean(valid))


def train_and_save(seed=1337):
    rows = generate_training_worlds(seed=seed)
    train, held_out = train_test_split(rows, test_size=.2, random_state=seed)
    scaler = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                 subsample=100_000, random_state=seed).fit(train)
    model = GaussianMixture(n_components=12, covariance_type="full", max_iter=250,
                            reg_covar=1e-4, random_state=seed).fit(scaler.transform(train))
    generated = _sample_mixture(model, scaler, len(held_out), seed + 1)
    discriminator_X = np.vstack([held_out, generated])
    discriminator_y = np.concatenate([np.ones(len(held_out)), np.zeros(len(generated))])
    X_train, X_test, y_train, y_test = train_test_split(
        discriminator_X, discriminator_y, test_size=.3, random_state=seed, stratify=discriminator_y)
    discriminator = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5,
                                           n_jobs=-1, random_state=seed).fit(X_train, y_train)
    auc = roc_auc_score(y_test, discriminator.predict_proba(X_test)[:, 1])
    diversity = float(np.mean(np.std(scaler.transform(generated), axis=0)))
    metrics = {
        "algorithm": "12-component full-covariance Gaussian mixture",
        "data_source": "synthetic_civilization_population_v1",
        "training_worlds": len(train), "held_out_worlds": len(held_out),
        "feature_count": len(FEATURE_NAMES),
        "held_out_log_likelihood": round(float(model.score(scaler.transform(held_out))), 4),
        "generated_validity_rate_before_constraints": round(_validity(generated), 4),
        "real_vs_generated_auc": round(float(auc), 4),
        "standardized_diversity": round(diversity, 4),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "feature_names": FEATURE_NAMES,
                 "company_fields": COMPANY_FIELDS, "metrics": metrics}, ARTIFACT_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))
