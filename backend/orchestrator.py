from prediction_engine import train_churn_model, predict_churn_probability
from calculation_engine import compute_monthly_snapshot
from state_store import get_latest_snapshot, insert_monthly_snapshot

_model, _ = train_churn_model()


def run_month(startup_id, marketing_spend, employee_count, avg_days_since_login=15, avg_support_tickets=1):
    previous = get_latest_snapshot(startup_id)

    churn_probability = predict_churn_probability(
        _model, avg_days_since_login, avg_support_tickets, float(previous["price_per_customer"])
    )
    customers_churned = round(previous["customer_count"] * churn_probability)
    new_customer_count = previous["customer_count"] - customers_churned

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
        revenue=computed["revenue"],
        employee_count=computed["employee_count"],
        investor_count=previous["investor_count"],
        funding_raised_to_date=previous["funding_raised_to_date"],
        price_per_customer=previous["price_per_customer"],
        marketing_spend=computed["marketing_spend"],
    )

    return computed
