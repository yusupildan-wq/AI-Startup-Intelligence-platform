import numpy as np

from prediction_engine import (
    train_churn_model,
    predict_churn_probability,
    train_growth_model,
    predict_new_customers,
    train_fundraising_model,
    predict_fundraising_success,
)
from calculation_engine import compute_monthly_snapshot
from state_store import get_latest_snapshot, insert_monthly_snapshot, get_startup
from narration_layer import generate_narration
from market import get_market_multiplier, get_market_label

_churn_model, _ = train_churn_model()
_growth_model, _ = train_growth_model()
_fundraising_model, _ = train_fundraising_model()
_rng = np.random.default_rng()


def run_month(
    startup_id,
    marketing_spend,
    employee_count,
    avg_days_since_login=15,
    avg_support_tickets=1,
    attempt_fundraising=False,
):
    previous = get_latest_snapshot(startup_id)

    if previous is None:
        startup = get_startup(startup_id)
        previous = {
            "month_number": 0,
            "cash_on_hand": startup["initial_funding"],
            "customer_count": startup["initial_customer_count"],
            "investor_count": 0,
            "funding_raised_to_date": startup["initial_funding"],
            "price_per_customer": startup["initial_price"],
        }

    churn_probability = predict_churn_probability(
        _churn_model, avg_days_since_login, avg_support_tickets, float(previous["price_per_customer"])
    )
    customers_churned = round(previous["customer_count"] * churn_probability)

    market_multiplier = get_market_multiplier()
    market_label = get_market_label(market_multiplier)

    base_new_customers = predict_new_customers(
        _growth_model, marketing_spend, float(previous["price_per_customer"]), previous["customer_count"]
    )
    customers_acquired = round(base_new_customers * market_multiplier)

    new_customer_count = previous["customer_count"] - customers_churned + customers_acquired

    computed = compute_monthly_snapshot(
        previous_snapshot=previous,
        customer_count=new_customer_count,
        employee_count=employee_count,
        marketing_spend=marketing_spend,
    )

    fundraising_result = None
    investor_count = previous["investor_count"]
    funding_raised_to_date = float(previous["funding_raised_to_date"])

    if attempt_fundraising:
        # A profitable company has runway_months = inf, which breaks the model's math.
        # Cap it: 60 months (5 years) of runway is effectively "as good as infinite" for this prediction.
        capped_runway = min(computed["runway_months"], 60)

        success_probability = predict_fundraising_success(
            _fundraising_model, computed["growth_rate"], capped_runway, computed["revenue"]
        )
        raised = bool(_rng.binomial(1, success_probability))
        amount_raised = round(computed["revenue"] * 12 * 3, 2) if raised else 0.0

        fundraising_result = {
            "attempted": True,
            "success_probability": round(success_probability, 3),
            "raised": raised,
            "amount_raised": amount_raised,
        }

        if raised:
            investor_count += 1
            funding_raised_to_date += amount_raised
            computed["cash_on_hand"] += amount_raised

    next_month_number = previous["month_number"] + 1

    insert_monthly_snapshot(
        startup_id=startup_id,
        month_number=next_month_number,
        cash_on_hand=computed["cash_on_hand"],
        customer_count=computed["customer_count"],
        customers_churned=customers_churned,
        customers_acquired=customers_acquired,
        revenue=computed["revenue"],
        employee_count=computed["employee_count"],
        investor_count=investor_count,
        funding_raised_to_date=funding_raised_to_date,
        price_per_customer=previous["price_per_customer"],
        marketing_spend=computed["marketing_spend"],
    )

    narration_input = {
        "month_number": next_month_number,
        "revenue": computed["revenue"],
        "customer_count": computed["customer_count"],
        "customers_churned": customers_churned,
        "customers_acquired": customers_acquired,
        "cash_on_hand": computed["cash_on_hand"],
        "marketing_spend": computed["marketing_spend"],
        "employee_count": computed["employee_count"],
        "market_condition": market_label,
        "fundraising_result": fundraising_result,
    }
    narration = generate_narration(narration_input)

    if computed["runway_months"] == float("inf"):
        computed["runway_months"] = None

    return {
        **computed,
        "customers_churned": customers_churned,
        "customers_acquired": customers_acquired,
        "market_condition": market_label,
        "market_multiplier": market_multiplier,
        "investor_count": investor_count,
        "funding_raised_to_date": funding_raised_to_date,
        "fundraising_result": fundraising_result,
        "narration": narration,
    }
