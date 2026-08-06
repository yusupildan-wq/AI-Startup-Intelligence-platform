"""Inference and rollout utilities for the trained AI CEO."""

from functools import lru_cache

import joblib
import numpy as np

from ml.ai_ceo_environment import ACTIONS, StartupEnvironment, StartupState
from ml.train_ai_ceo import ARTIFACT_PATH, encode

EXPLANATIONS = {
    "hold": "Preserve the current operating plan while the company compounds its existing advantages.",
    "raise_price": "Increase revenue per customer; the policy expects pricing power to outweigh additional churn.",
    "lower_price": "Reduce adoption friction and favor customer growth over immediate revenue per account.",
    "increase_marketing": "Use available capital to accelerate acquisition while market conditions are attractive.",
    "decrease_marketing": "Protect runway because current acquisition spending has weak risk-adjusted value.",
    "hire_engineer": "Invest in product quality and reduce the long-term cost of technical debt.",
    "hire_sales": "Add sales capacity to convert more of the available market demand.",
    "hire_support": "Improve retention by increasing customer-support capacity.",
    "reduce_headcount": "Lower fixed burn to improve survival probability and strategic flexibility.",
    "fundraise": "Seek additional capital now, accepting potential dilution to reduce cash risk.",
    "invest_in_product": "Spend cash on an immediate product-quality and technical-debt intervention.",
    "enter_new_market": "Pay the expansion cost to search for additional market fit and future demand.",
}


@lru_cache(maxsize=1)
def load_ai_ceo():
    return joblib.load(ARTIFACT_PATH)


def state_from_startup(startup, snapshot=None):
    source = snapshot or {}
    employees = max(1, int(source.get("employee_count", startup["founder_count"])))
    return StartupState(
        cash=float(source.get("cash_on_hand", startup["initial_funding"])),
        customers=float(source.get("customer_count", startup["initial_customer_count"])),
        price=float(source.get("price_per_customer", startup["initial_price"])),
        marketing=float(source.get("marketing_spend", 1000)),
        engineers=max(1, round(employees * 0.5)), salespeople=max(0, round(employees * 0.2)),
        support=max(0, round(employees * 0.15)), product_quality=0.58, market_fit=0.52,
        tech_debt=0.35, ownership=0.85, market=1.0,
        month=int(source.get("month_number", 0)),
    )


def recommend_action(state, rollout_months=12, seed=404):
    artifact = load_ai_ceo()
    model = artifact["model"]
    vector = state.vector()
    repeated = np.repeat(vector.reshape(1, -1), len(ACTIONS), axis=0)
    q_values = model.predict(encode(repeated, np.arange(len(ACTIONS))))
    ranking = np.argsort(q_values)[::-1]
    top_actions = [{
        "rank": rank + 1, "action": ACTIONS[index], "action_label": ACTIONS[index].replace("_", " ").title(),
        "long_term_value": round(float(q_values[index]), 3), "explanation": EXPLANATIONS[ACTIONS[index]],
    } for rank, index in enumerate(ranking[:5])]

    env = StartupEnvironment(seed)
    current = env.set_state(state)
    trajectory = []
    for month in range(rollout_months):
        states = np.repeat(current.reshape(1, -1), len(ACTIONS), axis=0)
        action = int(np.argmax(model.predict(encode(states, np.arange(len(ACTIONS))))))
        current, reward, done, info = env.step(action)
        trajectory.append({"month": month + 1, "reward": round(reward, 3), **info})
        if done: break
    return {
        "recommendation": top_actions[0], "alternatives": top_actions[1:], "projected_trajectory": trajectory,
        "policy": artifact["metrics"],
        "limitations": "Policy trained in synthetic startup environment v1; decisions are experimental, not financial advice.",
    }
