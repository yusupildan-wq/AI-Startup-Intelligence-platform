from prediction_engine import train_churn_model, predict_churn_probability, train_growth_model, predict_new_customers
from calculation_engine import compute_monthly_snapshot
from state_store import get_latest_snapshot, insert_monthly_snapshot, get_startup
from narration_layer import generate_narration
from market import get_market_multiplier, get_market_label

_churn_model, _ = train_churn_model()
_growth_model, _ = train_growth_model()


def run_month(startup_id, marketing_spend, employee_count, avg_days_since_login=15, avg_support_tickets=1):
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
        investor_count=previous["investor_count"],
        funding_raised_to_date=previous["funding_raised_to_date"],
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
    }
    narration = generate_narration(narration_input)

    return {
        **computed,
        "customers_churned": customers_churned,
        "customers_acquired": customers_acquired,
        "market_condition": market_label,
        "market_multiplier": market_multiplier,
        "narration": narration,
    }
