from calculation_engine import compute_monthly_snapshot


def test_revenue_is_customer_count_times_price():
    previous_snapshot = {"price_per_customer": 30.0, "cash_on_hand": 10000.0, "customer_count": 100}
    result = compute_monthly_snapshot(
        previous_snapshot=previous_snapshot,
        customer_count=100,
        employee_count=2,
        marketing_spend=500.0,
    )
    assert result["revenue"] == 3000.0


def test_burn_rate_when_costs_exceed_revenue():
    previous_snapshot = {"price_per_customer": 30.0, "cash_on_hand": 10000.0, "customer_count": 10}
    result = compute_monthly_snapshot(
        previous_snapshot=previous_snapshot,
        customer_count=10,
        employee_count=1,
        marketing_spend=0.0,
    )
    assert result["burn_rate"] == 6000.0


def test_runway_is_infinite_when_profitable():
    previous_snapshot = {"price_per_customer": 30.0, "cash_on_hand": 10000.0, "customer_count": 1000}
    result = compute_monthly_snapshot(
        previous_snapshot=previous_snapshot,
        customer_count=1000,
        employee_count=1,
        marketing_spend=0.0,
    )
    assert result["burn_rate"] < 0
    assert result["runway_months"] == float("inf")
