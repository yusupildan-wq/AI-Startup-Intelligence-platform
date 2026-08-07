"""Matched-policy benchmark on unseen generated civilizations."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from ml.ai_ceo import recommend_action
from ml.ai_ceo_environment import StartupState
from ml.model_based_ceo import plan_actions
from ml.world_generator import generate_learned_world
from world.engine import WorldEngine
from world.events import ACTION_TYPES

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
METADATA_PATH = MODEL_DIR / "central_benchmark_v1_metrics.json"
CONTROLLERS = ("model_based_ceo", "transferred_rl_ceo", "runway_heuristic", "hold", "random")


def rl_state(world):
    company = world.companies["player"]
    return StartupState(
        cash=company.cash, customers=company.customers, price=company.price, marketing=company.marketing,
        engineers=company.engineers, salespeople=company.salespeople, support=company.support,
        product_quality=company.product_quality, market_fit=company.reputation,
        tech_debt=company.technical_debt, ownership=company.founder_ownership,
        market=world.macro.demand_multiplier, month=world.month,
    )


def heuristic_action(world):
    company = world.companies["player"]
    headcount = company.engineers + company.salespeople + company.support
    monthly_cost = headcount * 7_000 + company.marketing + 1_500
    burn = monthly_cost - company.revenue
    runway = company.cash / max(burn, 1) if burn > 0 else 36
    if runway < 4: return "reduce_headcount"
    if runway < 10 and world.macro.venture_sentiment > .65: return "fundraise"
    if company.technical_debt > .55 and company.cash > 100_000: return "invest_in_product"
    if company.customers_churned > company.customers_acquired and company.support < 4: return "hire_support"
    if company.product_quality < .55 and company.cash > monthly_cost * 8: return "hire_engineer"
    if runway > 15 and world.macro.demand_multiplier > .95: return "increase_marketing"
    return "hold"


def outcome(engine):
    company = engine.state.companies["player"]
    value = company.cash + company.revenue * 12 + company.customers * 500 + company.product_quality * 100_000
    return {"survived": float(company.alive), "cash": company.cash, "revenue": company.revenue,
            "customers": company.customers, "enterprise_value_proxy": value,
            "founder_ownership": company.founder_ownership,
            "founder_value_proxy": value * company.founder_ownership}


def confidence_interval(values, statistic="mean", seed=99):
    values = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(2_000, len(values)), replace=True)
    estimates = np.median(samples, axis=1) if statistic == "median" else np.mean(samples, axis=1)
    return [round(float(value), 4) for value in np.percentile(estimates, [2.5, 97.5])]


def evaluate(worlds=24, horizon=12, seed=81_000):
    rng = np.random.default_rng(seed); rows = []
    for index in range(worlds):
        initial = generate_learned_world(f"Held-out benchmark {index}", seed + index * 997)
        plan = plan_actions(initial, horizon=horizon, beam_width=3, paths=20,
                            risk_aversion=.65, seed=seed + index * 31)["recommendation"]["planned_sequence"]
        engines = {name: WorldEngine(deepcopy(initial)) for name in CONTROLLERS}
        action_logs = {name: [] for name in CONTROLLERS}
        for month in range(horizon):
            actions = {
                "model_based_ceo": plan[month],
                "transferred_rl_ceo": recommend_action(rl_state(engines["transferred_rl_ceo"].state), rollout_months=0,
                                                         seed=seed + index * 101 + month)["recommendation"]["action"],
                "runway_heuristic": heuristic_action(engines["runway_heuristic"].state),
                "hold": "hold",
                "random": str(rng.choice(sorted(ACTION_TYPES))),
            }
            for name, engine in engines.items():
                if engine.state.companies["player"].alive:
                    engine.advance(actions[name]); action_logs[name].append(actions[name])
        for name, engine in engines.items():
            rows.append({"world": index, "controller": name, "actions": action_logs[name], **outcome(engine)})

    summaries = {}
    for name in CONTROLLERS:
        selected = [row for row in rows if row["controller"] == name]
        survival = [row["survived"] for row in selected]
        values = [row["founder_value_proxy"] for row in selected]
        enterprise_values = [row["enterprise_value_proxy"] for row in selected]
        cash = [row["cash"] for row in selected]
        revenue = [row["revenue"] for row in selected]
        ownership = [row["founder_ownership"] for row in selected]
        fundraises = [row["actions"].count("fundraise") for row in selected]
        summaries[name] = {
            "survival_rate": round(float(np.mean(survival)), 4),
            "survival_95_ci": confidence_interval(survival, seed=seed + 1),
            "median_founder_value_proxy": round(float(np.median(values)), 2),
            "median_founder_value_95_ci": confidence_interval(values, "median", seed + 2),
            "median_enterprise_value_proxy": round(float(np.median(enterprise_values)), 2),
            "median_founder_ownership": round(float(np.median(ownership)), 4),
            "median_fundraise_actions": round(float(np.median(fundraises)), 2),
            "cash_p10": round(float(np.percentile(cash, 10)), 2),
            "median_ending_cash": round(float(np.median(cash)), 2),
            "median_ending_revenue": round(float(np.median(revenue)), 2),
        }
    model_rows = {row["world"]: row for row in rows if row["controller"] == "model_based_ceo"}
    for name in CONTROLLERS[1:]:
        opponent = {row["world"]: row for row in rows if row["controller"] == name}
        summaries["model_based_ceo"][f"value_win_rate_vs_{name}"] = round(float(np.mean([
            model_rows[index]["founder_value_proxy"] > opponent[index]["founder_value_proxy"]
            for index in model_rows])), 4)
    result = {
        "headline": "Matched controller evaluation on unseen synthetic civilizations",
        "worlds": worlds, "horizon_months": horizon, "world_seed_start": seed,
        "controllers": summaries,
        "methodology": "Every controller starts from an identical copy of each unseen generated world. The model-based CEO plans once, then all policies execute for 12 months in the event engine.",
        "primary_metric": "survival_rate",
        "secondary_metric": "median_founder_value_proxy",
        "enterprise_value_proxy": "cash + 12x monthly revenue + $500/customer + $100k x product quality",
        "founder_value_proxy": "enterprise value proxy x retained founder ownership",
        "limitations": "Evaluation is entirely inside the synthetic civilization. It measures policy performance in this simulator, not real-world startup performance. Fundraising frequency is reported because cash-heavy policies can exploit synthetic capital availability.",
    }
    MODEL_DIR.mkdir(exist_ok=True)
    METADATA_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
