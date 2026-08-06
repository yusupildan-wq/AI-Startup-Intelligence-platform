"""Monte Carlo strategy search powered by the trained startup models.

This module is intentionally read-only: exploring a strategy never writes a monthly
snapshot, so founders can compare decisions without changing their live simulation.
"""

import numpy as np

from calculation_engine import AVERAGE_SALARY, TOOLING_ESTIMATE
from prediction_engine import predict_churn_probability, predict_new_customers


def _percentile(values, percentile):
    return round(float(np.percentile(values, percentile)), 2)


def analyze_strategies(
    starting_state,
    churn_model,
    growth_model,
    horizon_months=12,
    simulations=250,
    seed=2026,
):
    """Rank a grid of pricing, marketing, and hiring strategies under uncertainty."""
    base_price = float(starting_state["price_per_customer"])
    base_marketing = max(0.0, float(starting_state.get("marketing_spend", 0)))
    base_employees = max(1, int(starting_state.get("employee_count", 1)))

    prices = sorted({round(max(1, base_price * factor), 2) for factor in (0.8, 1.0, 1.2)})
    marketing_levels = sorted({round(base_marketing * factor, 2) for factor in (0.75, 1.0, 1.5)} | {1000.0})
    employee_levels = sorted({max(1, base_employees - 1), base_employees, base_employees + 1})
    rng = np.random.default_rng(seed)
    results = []

    for price in prices:
        for marketing in marketing_levels:
            for employees in employee_levels:
                ending_cash, ending_revenue, ending_customers = [], [], []
                survived = 0

                for _ in range(simulations):
                    cash = float(starting_state["cash_on_hand"])
                    customers = max(0, int(starting_state["customer_count"]))

                    for _month in range(horizon_months):
                        churn_probability = predict_churn_probability(
                            churn_model, days_since_login=15, support_tickets=1, price=price
                        )
                        churned = rng.binomial(customers, min(max(churn_probability, 0), 1))
                        expected_acquired = predict_new_customers(
                            growth_model, marketing, price, customers
                        )
                        market_multiplier = float(np.clip(rng.normal(1.0, 0.15), 0.55, 1.45))
                        acquired = rng.poisson(max(0, expected_acquired * market_multiplier))
                        customers = max(0, customers - churned + acquired)
                        revenue = customers * price
                        costs = employees * AVERAGE_SALARY + marketing + TOOLING_ESTIMATE
                        cash += revenue - costs
                        if cash <= 0:
                            break

                    if cash > 0:
                        survived += 1
                    ending_cash.append(cash)
                    ending_revenue.append(customers * price)
                    ending_customers.append(customers)

                survival_probability = survived / simulations
                median_cash = _percentile(ending_cash, 50)
                # Reward upside, but strongly penalize plans with a high failure rate.
                score = median_cash * (0.35 + 0.65 * survival_probability)
                results.append({
                    "price": price,
                    "monthly_marketing": marketing,
                    "employee_count": employees,
                    "survival_probability": round(survival_probability, 3),
                    "ending_cash_p10": _percentile(ending_cash, 10),
                    "ending_cash_median": median_cash,
                    "ending_cash_p90": _percentile(ending_cash, 90),
                    "ending_revenue_median": _percentile(ending_revenue, 50),
                    "ending_customers_median": round(float(np.median(ending_customers))),
                    "score": round(score, 2),
                })

    ranked = sorted(results, key=lambda item: item["score"], reverse=True)
    for rank, result in enumerate(ranked, start=1):
        result["rank"] = rank

    best = ranked[0]
    return {
        "horizon_months": horizon_months,
        "simulations_per_strategy": simulations,
        "strategies_evaluated": len(ranked),
        "recommendation": (
            f"Set price to ${best['price']:,.0f}, budget ${best['monthly_marketing']:,.0f}/month "
            f"for marketing, and operate with {best['employee_count']} employee(s)."
        ),
        "best_strategy": best,
        "top_strategies": ranked[:5],
    }
