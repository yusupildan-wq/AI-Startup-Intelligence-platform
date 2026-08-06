from functools import lru_cache
from uuid import uuid4

import joblib
import numpy as np

from ml.train_world_generator import ARTIFACT_PATH, COMPANY_FIELDS
from world.models import Company, CustomerSegment, InvestorMarket, MacroEconomy, WorldState


@lru_cache(maxsize=1)
def load_world_generator():
    return joblib.load(ARTIFACT_PATH)


def _sample(seed):
    artifact = load_world_generator(); model = artifact["model"]
    rng = np.random.default_rng(seed)
    component = int(rng.choice(len(model.weights_), p=model.weights_))
    scaled = (rng.multivariate_normal(model.means_[component], model.covariances_[component])
              if model.covariance_type == "full"
              else rng.normal(model.means_[component], np.sqrt(model.covariances_[component])))
    return artifact["scaler"].inverse_transform([scaled])[0]


def generate_learned_world(name, seed, scenario="balanced"):
    row = _sample(seed); companies = {}
    names = (("player", "Player Startup"), ("competitor_alpha", "Generated Rival A"),
             ("competitor_beta", "Generated Rival B"))
    for index, (company_id, company_name) in enumerate(names):
        values = dict(zip(COMPANY_FIELDS, row[index * len(COMPANY_FIELDS):(index + 1) * len(COMPANY_FIELDS)]))
        companies[company_id] = Company(
            company_id, company_name, max(25_000, float(values["cash"])), max(1, round(values["customers"])),
            max(5, float(values["price"])), max(250, float(values["marketing"])),
            max(1, round(values["engineers"])), max(0, round(values["salespeople"])),
            max(0, round(values["support"])), np.clip(values["product_quality"], .05, .98),
            np.clip(values["technical_debt"], .02, .95), np.clip(values["reputation"], .05, .98),
        )
    offset = 3 * len(COMPANY_FIELDS)
    demand, rate, unemployment, sentiment, capital, risk, multiple, smb, mid, enterprise, smb_budget, mid_budget, enterprise_budget = row[offset:]
    if scenario == "recession": demand, rate, unemployment, sentiment = .68, .08, .09, .18
    elif scenario == "funding_boom": demand, rate, unemployment, sentiment = 1.28, .025, .035, .92
    elif scenario == "technology_shift":
        demand = 1.12
        for company in companies.values(): company.technical_debt = min(.95, company.technical_debt + .2)
    regime = "recession" if demand < .82 else "funding_boom" if sentiment > .82 else "expansion" if demand > 1.12 else "stable"
    return WorldState(
        id=str(uuid4()), name=name, seed=seed, companies=companies,
        segments={
            "smb": CustomerSegment("smb", "Small businesses", max(1000, round(smb)), max(20, smb_budget), .8, .45, .18, .018),
            "midmarket": CustomerSegment("midmarket", "Mid-market", max(300, round(mid)), max(80, mid_budget), .45, .7, .42, .012),
            "enterprise": CustomerSegment("enterprise", "Enterprise", max(100, round(enterprise)), max(250, enterprise_budget), .2, .9, .72, .007),
        },
        investors=InvestorMarket(max(1_000_000, float(capital)), np.clip(risk, .05, .95), max(1, float(multiple))),
        macro=MacroEconomy(regime, np.clip(demand, .55, 1.45), np.clip(rate, .005, .15),
                           np.clip(unemployment, .02, .15), np.clip(sentiment, .02, .98)),
    )
