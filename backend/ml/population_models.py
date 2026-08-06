from functools import lru_cache

import joblib
import numpy as np

from ml.train_population_models import ARTIFACT_PATH


@lru_cache(maxsize=1)
def load_population_models():
    return joblib.load(ARTIFACT_PATH)


def customer_purchase_probability(segment, company, competitor_utility, demand_multiplier):
    X = [[segment.budget, segment.price_sensitivity, segment.quality_preference,
          segment.switching_cost, company.price, company.product_quality,
          company.reputation, competitor_utility, demand_multiplier]]
    return float(load_population_models()["customer_choice"].predict_proba(X)[0, 1])


def employee_departure_probability(salary_ratio, morale, burnout, tenure_months,
                                   company_growth, runway, manager_quality, market_jobs):
    X = [[salary_ratio, morale, burnout, tenure_months, company_growth, runway,
          manager_quality, market_jobs]]
    return float(load_population_models()["employee_attrition"].predict_proba(X)[0, 1])


def product_adoption_probability(segment_need, feature_fit, usability, awareness,
                                 switching_cost, price_change, peer_adoption):
    X = [[segment_need, feature_fit, usability, awareness, switching_cost,
          price_change, peer_adoption]]
    return float(load_population_models()["product_adoption"].predict_proba(X)[0, 1])
